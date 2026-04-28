// env_overrides.cpp — apply_env_overrides() definition.
// Shared between the memopt-transport daemon and the test_transport binary
// so there is a single definition in the memopt::transport namespace.
#include "daemon.h"

#include <cstdlib>
#include <cstring>
#include <unistd.h>

namespace memopt {
namespace transport {

void apply_env_overrides(DaemonConfig& cfg) noexcept {
    if (const char* env = std::getenv("MEMOPT_NODE_ID")) {
        if (cfg.node_id.empty()) cfg.node_id = env;
    }
    if (const char* env = std::getenv("MEMOPT_RBP_PORT")) {
        cfg.tcp_port = static_cast<uint16_t>(std::atoi(env));
    }
    if (const char* env = std::getenv("MEMOPT_TRANSPORT")) {
        if (std::strcmp(env, "tcp") == 0) cfg.prefer_rdma = false;
    }
    if (const char* env = std::getenv("MEMOPT_QP_TIMEOUT_S")) {
        cfg.qp_exchange_timeout_s = std::atoi(env);
    }
    if (const char* env = std::getenv("MEMOPT_QP_RETRIES")) {
        cfg.qp_exchange_retries = std::atoi(env);
    }
    if (const char* env = std::getenv("MEMOPT_QP_DEBUG")) {
        cfg.log_qp_transitions = (std::atoi(env) != 0);
    }

    if (cfg.node_id.empty()) {
        char hostname[256];
        if (::gethostname(hostname, sizeof(hostname)) == 0) {
            cfg.node_id = hostname;
        } else {
            cfg.node_id = "unknown";
        }
    }
}

} // namespace transport
} // namespace memopt
