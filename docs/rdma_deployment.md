# RDMA Deployment Guide

## Prerequisites

1. **libibverbs installed**
   ```
   # Ubuntu/Debian
   apt install libibverbs-dev rdma-core

   # RHEL/CentOS
   yum install libibverbs-devel rdma-core-devel
   ```

2. **IB device active**
   ```
   ibv_devinfo | grep state
   ```
   Must show: `PORT_ACTIVE`

   If `PORT_DOWN`: check cable, switch, and subnet manager (opensm).

3. **nv_peer_mem loaded (for GPU-Direct RDMA)**
   ```
   lsmod | grep nv_peer_mem
   ```
   Required only for GPU HBM → remote GPU HBM transfers.
   Not required for CPU DRAM → remote CPU DRAM.

4. **memopt-transport built with RDMA**
   ```
   cmake -DMEMOPT_ENABLE_RDMA=ON ..
   make memopt-transport
   ```

## Environment Variables

| Variable | Default | Description |
|----------|---------|-------------|
| `MEMOPT_TRANSPORT` | auto | `rdma` force RDMA, `tcp` force TCP |
| `MEMOPT_QP_EXCHANGE` | `tcp` | QP info exchange: `tcp`, `file`, or `etcd` |
| `MEMOPT_QP_EXCHANGE_PORT` | `18517` | TCP port for QP exchange sideband |
| `MEMOPT_QP_EXCHANGE_DIR` | `/tmp/memopt_qp_exchange` | Directory for file-based exchange |
| `MEMOPT_QP_TIMEOUT_S` | `10` | Exchange timeout in seconds |
| `MEMOPT_QP_RETRIES` | `5` | Exchange retry count |
| `MEMOPT_QP_DEBUG` | `0` | Set to `1` to log every QP state transition |
| `MEMOPT_NODE_ID` | hostname | Unique node identifier |
| `MEMOPT_NODE_HOSTS` | (none) | Static peer list: `host:port,host:port` |
| `REDIS_URL` | (none) | Redis for dynamic peer discovery |

## Starting the daemon

```bash
# Minimal (auto-detect everything)
memopt-transport --node-id node-01

# Explicit RDMA with debug logging
MEMOPT_QP_DEBUG=1 memopt-transport \
    --node-id node-01 \
    --tcp-port 18516

# TCP only (no RDMA)
memopt-transport --node-id node-01 --no-rdma
```

## First deployment checklist

- [ ] `ibv_devinfo` shows `PORT_ACTIVE` on all nodes
- [ ] Nodes can reach each other on `MEMOPT_QP_EXCHANGE_PORT` (default 18517)
- [ ] `MEMOPT_NODE_ID` set uniquely per node
- [ ] `MEMOPT_NODE_HOSTS` or `REDIS_URL` set for peer discovery
- [ ] Start with `MEMOPT_QP_DEBUG=1` to verify QP transitions
- [ ] Check `/dev/shm/memopt_transport_{node_id}_req` exists after daemon starts
- [ ] Verify Python can import: `from memopt.cluster.transport import make_transport`

## QP exchange backends

### TCP sideband (default)

Both nodes exchange QP info via a dedicated TCP connection.
Role assignment: lower node_id (lexicographic) connects first.

```
MEMOPT_QP_EXCHANGE=tcp
```

### Shared file (testing)

For single-machine multi-GPU testing. Uses atomic file writes.

```
MEMOPT_QP_EXCHANGE=file
MEMOPT_QP_EXCHANGE_DIR=/shared/nfs/qp_exchange
```

### etcd (planned)

Not fully implemented. Falls back to TCP with a warning.

```
MEMOPT_QP_EXCHANGE=etcd  # will print warning and use TCP
```

## If connect() fails

1. **Check `MEMOPT_QP_DEBUG=1` output** — which transition failed?
   ```
   memopt-transport: QP[node-02] RESET → INIT: ok (qpn=123)
   memopt-transport: QP[node-02] INIT → RTR: ok (remote_qpn=456 lid=1 psn=789)
   memopt-transport: QP[node-02] RTR → RTS: ok (local_psn=101)
   ```
   If any line is missing, that transition failed. The error will print
   `ret=`, `errno=`, and `strerror()` with the exact cause.

2. **Check `ibv_devinfo`** — is `PORT_ACTIVE` on both nodes?
   ```
   ibv_devinfo -d mlx5_0 | grep state
   ```

3. **Try file-based exchange** on shared NFS first — eliminates TCP exchange as a variable:
   ```
   MEMOPT_QP_EXCHANGE=file MEMOPT_QP_EXCHANGE_DIR=/shared/nfs/qp/
   ```

4. **Try TCP transport** — if TCP works but RDMA doesn't, it's an IB fabric issue:
   ```
   MEMOPT_TRANSPORT=tcp memopt-transport --node-id test
   ```

5. **Check MTU**:
   ```
   ibv_devinfo | grep max_mtu
   ```
   Must be >= 4096 (memopt uses `IBV_MTU_4096`).

## Common errors

| Error message | Cause | Fix |
|--------------|-------|-----|
| `ibv_modify_qp INIT failed: errno=22` | Invalid port or pkey | Check `ibv_devinfo` port number |
| `ibv_modify_qp RTR failed: errno=22` | Invalid remote QPN or LID | Verify peer is up, check subnet manager |
| `QP info exchange failed` | TCP sideband timeout | Check firewall on port 18517 |
| `RDMA not available` | No IB device or libibverbs not installed | Install rdma-core, check `ibv_devinfo` |
