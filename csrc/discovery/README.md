# memopt-discovery: HashiCorp Memberlist Sidecar

## Current State

Node discovery uses Redis-based gossip (`memopt/vmm/discovery.py`).
This works correctly up to ~10K nodes using SCAN-based peer enumeration
with capability advertisement.

## Future State (when needed)

At 100K+ nodes, replace Redis discovery with a Go sidecar using
[HashiCorp Memberlist](https://github.com/hashicorp/memberlist).

### Why Go

Memberlist is a production-grade SWIM protocol implementation.
It handles:
- Sub-second failure detection via ping/indirect-ping
- Efficient gossip with bounded bandwidth (O(log N))
- Anti-entropy repair for convergence
- Tombstone propagation for clean leaves

### Integration Plan

The `memopt-discovery` binary will:
1. Start Memberlist on `MEMOPT_DISCOVERY_PORT` (default 7946)
2. Join cluster via seed nodes (`MEMOPT_SEED_NODES`)
3. Broadcast `NodeCapabilities` as Memberlist metadata
4. Expose discovered peers via Unix socket: `/var/run/memopt/discovery.sock`
5. Python reads peers via socket — `NodeDiscovery` adds `_from_socket()` method

### Build

```bash
cd csrc/discovery
go build -o memopt-discovery .
```

### When to Implement

When node count exceeds 10K and Redis SCAN latency exceeds 100ms
per scan cycle. Monitor via `NodeDiscovery.stats()["scan_failures"]`.
