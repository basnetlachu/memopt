// main.cpp — memopt-transport sidecar daemon entry point.
//
// Usage:
//   memopt-transport --node-id mynode --tcp-port 18516 [--no-rdma]
//
// Environment overrides:
//   MEMOPT_NODE_ID       node identifier
//   MEMOPT_RBP_PORT      TCP listen port
//   MEMOPT_TRANSPORT=tcp force TCP-only mode

#include "daemon.h"

#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <string>
#include <unistd.h>

using namespace memopt::transport;

static void print_usage(const char* argv0) {
    std::fprintf(stderr,
        "Usage: %s [options]\n"
        "Options:\n"
        "  --node-id <id>       Node identifier (required)\n"
        "  --tcp-port <port>    TCP listen port (default: 18516)\n"
        "  --rdma-device <dev>  IB device name (default: auto)\n"
        "  --cq-cpu <core>      CPU core for CQ poller (default: any)\n"
        "  --no-rdma            Disable RDMA, TCP only\n"
        "  --shm-prefix <pfx>   SHM path prefix (default: /memopt_transport)\n"
        "  --help               Show this help\n",
        argv0);
}

static DaemonConfig parse_args(int argc, char* argv[]) {
    DaemonConfig cfg;
    cfg.tcp_port = 18516;
    cfg.prefer_rdma = true;

    for (int i = 1; i < argc; ++i) {
        if (std::strcmp(argv[i], "--node-id") == 0 && i + 1 < argc) {
            cfg.node_id = argv[++i];
        } else if (std::strcmp(argv[i], "--tcp-port") == 0 && i + 1 < argc) {
            cfg.tcp_port = static_cast<uint16_t>(std::atoi(argv[++i]));
        } else if (std::strcmp(argv[i], "--rdma-device") == 0 && i + 1 < argc) {
            cfg.rdma_device = argv[++i];
        } else if (std::strcmp(argv[i], "--cq-cpu") == 0 && i + 1 < argc) {
            cfg.rdma_cq_cpu = std::atoi(argv[++i]);
        } else if (std::strcmp(argv[i], "--no-rdma") == 0) {
            cfg.prefer_rdma = false;
        } else if (std::strcmp(argv[i], "--shm-prefix") == 0 && i + 1 < argc) {
            cfg.shm_path_prefix = argv[++i];
        } else if (std::strcmp(argv[i], "--help") == 0) {
            print_usage(argv[0]);
            std::exit(0);
        }
    }

    // Apply environment overrides
    apply_env_overrides(cfg);

    return cfg;
}

// apply_env_overrides() is defined in transport/env_overrides.cpp so that
// both the sidecar daemon and the test binary can link to a single definition.

int main(int argc, char* argv[]) {
    DaemonConfig cfg = parse_args(argc, argv);

    std::fprintf(stderr,
        "memopt-transport: starting (node=%s tcp=%d rdma=%s)\n",
        cfg.node_id.c_str(), cfg.tcp_port,
        cfg.prefer_rdma ? "enabled" : "disabled");

    TransportDaemon daemon(std::move(cfg));

    if (!daemon.start()) {
        std::fprintf(stderr, "memopt-transport: init failed\n");
        return 1;
    }

    std::fprintf(stderr, "memopt-transport: ready\n");
    daemon.run();  // blocks until SIGTERM or SHUTDOWN message

    std::fprintf(stderr, "memopt-transport: shutdown complete\n");
    return 0;
}
