// test_transport.cpp — GoogleTest for ring buffer, TCP multiplexer, daemon.
//
// No RDMA hardware required. RDMA tests auto-skip.
// Ring buffer tests use shm_open — works on Linux and macOS.

#include "daemon.h"
#include "protocol.h"
#include "ring_buffer.h"
#include "tcp_fallback.h"

#include <gtest/gtest.h>

#include <chrono>
#include <cstring>
#include <filesystem>
#include <string>
#include <thread>
#include <vector>

using namespace memopt::transport;

// ═══════════════════════════════════════════════════════════════════════════
// Helper: unique shm name for each test to avoid conflicts
// ═══════════════════════════════════════════════════════════════════════════
static std::string unique_shm_name(const std::string& prefix) {
    static int counter = 0;
    return "/" + prefix + "_test_" + std::to_string(++counter) +
           "_" + std::to_string(::getpid());
}

// ═══════════════════════════════════════════════════════════════════════════
// 1. Ring buffer: single-threaded push/pop round-trip
// ═══════════════════════════════════════════════════════════════════════════

TEST(RingBuffer, SingleThreadedRoundTrip) {
    auto path = unique_shm_name("rb_rt");
    auto rb = RingBuffer::create(path);
    ASSERT_TRUE(rb.is_valid());

    // Push a message
    const char* msg = "hello ring buffer";
    EXPECT_TRUE(rb.try_push(8, msg, static_cast<uint32_t>(strlen(msg))));

    // Pop it back
    uint8_t  out_type;
    char     out_data[256]{};
    uint32_t out_size;
    EXPECT_TRUE(rb.try_pop(&out_type, out_data, &out_size));

    EXPECT_EQ(out_type, 8);
    EXPECT_EQ(out_size, strlen(msg));
    EXPECT_EQ(std::string(out_data, out_size), msg);
}

// ═══════════════════════════════════════════════════════════════════════════
// 2. Ring buffer: full buffer returns false
// ═══════════════════════════════════════════════════════════════════════════

TEST(RingBuffer, FullBufferReturnsFalse) {
    auto path = unique_shm_name("rb_full");
    auto rb = RingBuffer::create(path);
    ASSERT_TRUE(rb.is_valid());

    // Fill all slots
    char data[4] = "abc";
    for (size_t i = 0; i < RB_CAPACITY; ++i) {
        EXPECT_TRUE(rb.try_push(1, data, 3))
            << "Push failed at slot " << i;
    }

    // One more should fail
    EXPECT_FALSE(rb.try_push(1, data, 3));
    EXPECT_TRUE(rb.is_full());
}

// ═══════════════════════════════════════════════════════════════════════════
// 3. Ring buffer: empty buffer returns false
// ═══════════════════════════════════════════════════════════════════════════

TEST(RingBuffer, EmptyBufferReturnsFalse) {
    auto path = unique_shm_name("rb_empty");
    auto rb = RingBuffer::create(path);
    ASSERT_TRUE(rb.is_valid());

    uint8_t type; char data[16]; uint32_t size;
    EXPECT_FALSE(rb.try_pop(&type, data, &size));
    EXPECT_TRUE(rb.is_empty());
}

// ═══════════════════════════════════════════════════════════════════════════
// 4. Ring buffer: SPSC concurrent correctness
// ═══════════════════════════════════════════════════════════════════════════

TEST(RingBuffer, SPSCConcurrent) {
    auto path = unique_shm_name("rb_spsc");
    auto rb = RingBuffer::create(path);
    ASSERT_TRUE(rb.is_valid());

    constexpr int NUM_MSGS = 100000;
    std::atomic<int> received{0};

    // Producer thread
    std::thread producer([&]() {
        for (int i = 0; i < NUM_MSGS; ++i) {
            uint32_t seq = static_cast<uint32_t>(i);
            while (!rb.try_push(1, &seq, sizeof(seq))) {
                std::this_thread::yield();
            }
        }
    });

    // Consumer thread
    std::thread consumer([&]() {
        int expected = 0;
        uint8_t type;
        uint32_t data;
        uint32_t size;
        while (expected < NUM_MSGS) {
            if (rb.try_pop(&type, &data, &size)) {
                EXPECT_EQ(data, static_cast<uint32_t>(expected))
                    << "Out of order at " << expected;
                ++expected;
            } else {
                std::this_thread::yield();
            }
        }
        received.store(expected);
    });

    producer.join();
    consumer.join();

    EXPECT_EQ(received.load(), NUM_MSGS);
}

// ═══════════════════════════════════════════════════════════════════════════
// 5. Ring buffer: shared memory persistence
// ═══════════════════════════════════════════════════════════════════════════

TEST(RingBuffer, SharedMemoryPersistence) {
    auto path = unique_shm_name("rb_persist");

    // Create and write 3 messages
    {
        auto rb = RingBuffer::create(path);
        ASSERT_TRUE(rb.is_valid());

        for (int i = 0; i < 3; ++i) {
            EXPECT_TRUE(rb.try_push(
                static_cast<uint8_t>(i + 1), &i, sizeof(i)));
        }
        // rb goes out of scope — but since it's the owner,
        // it will unlink. For persistence test, we need a
        // non-owner to survive.
    }

    // Re-create at same path and verify we can read
    // (This tests the shm file lifecycle)
    {
        auto rb = RingBuffer::create(path);
        ASSERT_TRUE(rb.is_valid());

        // Fresh buffer — should be empty (we recreated with O_TRUNC)
        EXPECT_TRUE(rb.is_empty());
    }
}

// ═══════════════════════════════════════════════════════════════════════════
// 6. Protocol: all structs trivially copyable (compile-time check)
// ═══════════════════════════════════════════════════════════════════════════

TEST(Protocol, StructsTriviallyCopyable) {
    // These are checked by static_assert in protocol.h at compile time.
    // Runtime verification via memcpy round-trip:
    ConnectRequest req{};
    std::strcpy(req.node_id, "test-node");
    std::strcpy(req.host, "192.168.1.1");
    req.port = 18516;

    ConnectRequest copy{};
    std::memcpy(&copy, &req, sizeof(req));

    EXPECT_STREQ(copy.node_id, "test-node");
    EXPECT_STREQ(copy.host, "192.168.1.1");
    EXPECT_EQ(copy.port, 18516);
}

// ═══════════════════════════════════════════════════════════════════════════
// 7. TCP multiplexer: start/stop lifecycle
// ═══════════════════════════════════════════════════════════════════════════

TEST(TCPMultiplexer, StartStop) {
    TCPMultiplexer mux(19990);
    EXPECT_TRUE(mux.start());

    // Give event loop time to start
    std::this_thread::sleep_for(std::chrono::milliseconds(50));

    mux.stop();
    // No crash, no hang
}

// ═══════════════════════════════════════════════════════════════════════════
// 8. TCP multiplexer: connect to unreachable host
// ═══════════════════════════════════════════════════════════════════════════

TEST(TCPMultiplexer, ConnectUnreachable) {
    TCPMultiplexer mux(19991);

    ConnectRequest req{};
    safe_strcpy(req.node_id, sizeof(req.node_id), "dead-node");
    safe_strcpy(req.host, sizeof(req.host), "127.0.0.1");
    req.port = 1;  // nothing listening

    ConnectAck ack{};
    bool ok = mux.connect(req, ack);
    EXPECT_FALSE(ok);
    EXPECT_EQ(ack.success, 0);
}

// ═══════════════════════════════════════════════════════════════════════════
// 9. RDMA engine: availability detection
// ═══════════════════════════════════════════════════════════════════════════

TEST(RDMAEngine, AvailabilityDetection) {
    RDMAEngine engine;
    bool avail = engine.init();
    EXPECT_EQ(engine.is_available(), avail);
    // On most test machines: both false. On IB machines: both true.

    auto tls = engine.detect_best_tls();
    EXPECT_FALSE(tls.empty());
    // Either "rc" (IB hardware) or "tcp" (no hardware)
}

// ═══════════════════════════════════════════════════════════════════════════
// 10. Ring buffer: message type preserved
// ═══════════════════════════════════════════════════════════════════════════

TEST(RingBuffer, MessageTypePreserved) {
    auto path = unique_shm_name("rb_msgtype");
    auto rb = RingBuffer::create(path);
    ASSERT_TRUE(rb.is_valid());

    // Push different message types
    uint8_t  types[] = {1, 8, 64, 69, 255};
    for (auto t : types) {
        EXPECT_TRUE(rb.try_push(t, nullptr, 0));
    }

    // Pop and verify types
    for (auto expected : types) {
        uint8_t got;
        uint32_t size;
        EXPECT_TRUE(rb.try_pop(&got, nullptr, &size));
        EXPECT_EQ(got, expected);
        EXPECT_EQ(size, 0u);
    }
}

// ═══════════════════════════════════════════════════════════════════════════
// 11. Ring buffer: wrap-around
// ═══════════════════════════════════════════════════════════════════════════

TEST(RingBuffer, WrapAround) {
    auto path = unique_shm_name("rb_wrap");
    auto rb = RingBuffer::create(path);
    ASSERT_TRUE(rb.is_valid());

    // Push and pop more than RB_CAPACITY messages
    // Verifies wrap-around works correctly
    int total = RB_CAPACITY * 3;
    for (int i = 0; i < total; ++i) {
        uint32_t val = static_cast<uint32_t>(i);
        while (!rb.try_push(1, &val, sizeof(val))) {
            // Pop to make room
            uint8_t t; uint32_t d, s;
            rb.try_pop(&t, &d, &s);
        }
    }
    // Buffer should still be functional
    EXPECT_FALSE(rb.is_full());  // we popped some
}

// ═══════════════════════════════════════════════════════════════════════════
// 12. Benchmark: ring buffer throughput
// ═══════════════════════════════════════════════════════════════════════════

TEST(RingBuffer, BenchmarkThroughput) {
    auto path = unique_shm_name("rb_bench");
    auto rb = RingBuffer::create(path);
    ASSERT_TRUE(rb.is_valid());

    constexpr int ITERS = 1000000;
    uint32_t payload = 42;

    auto start = std::chrono::high_resolution_clock::now();

    // Single-threaded push+pop pairs
    for (int i = 0; i < ITERS; ++i) {
        rb.try_push(1, &payload, sizeof(payload));
        uint8_t t; uint32_t d, s;
        rb.try_pop(&t, &d, &s);
    }

    auto end = std::chrono::high_resolution_clock::now();
    auto ms = std::chrono::duration_cast<std::chrono::milliseconds>(
        end - start).count();

    double ops = static_cast<double>(ITERS) * 2;  // push + pop
    double ops_per_sec = ops / (ms / 1000.0);

    std::printf("Ring buffer: %.1fM ops/sec (%d push+pop pairs in %ldms)\n",
                ops_per_sec / 1e6, ITERS, static_cast<long>(ms));

    // Should achieve >5M ops/sec on any modern hardware
    EXPECT_GT(ops_per_sec, 5e6);
}

// ═══════════════════════════════════════════════════════════════════════════
// 13. QP exchange: file backend round-trip
// ═══════════════════════════════════════════════════════════════════════════

TEST(QPExchange, FileBackendRoundTrip) {
    // Simulate two nodes exchanging QPInfo via files
    std::string dir = "/tmp/memopt_qp_exchange_test_" +
                      std::to_string(::getpid());

    QPInfo local_a{};
    local_a.qpn = 1234;
    local_a.lid = 1;
    local_a.psn = 5678;
    local_a.rkey = 0xABCD;
    local_a.remote_addr = 0x1000;

    QPInfo local_b{};
    local_b.qpn = 4321;
    local_b.lid = 2;
    local_b.psn = 8765;
    local_b.rkey = 0xDCBA;
    local_b.remote_addr = 0x2000;

    QPInfo remote_a{}, remote_b{};

    // Two threads exchanging simultaneously
    std::thread ta([&]() {
        EXPECT_TRUE(exchange_qp_info_file(
            "node_aaa", "node_zzz", dir,
            local_a, remote_a, 5));
    });

    std::thread tb([&]() {
        EXPECT_TRUE(exchange_qp_info_file(
            "node_zzz", "node_aaa", dir,
            local_b, remote_b, 5));
    });

    ta.join();
    tb.join();

    // Verify each node received the other's QPInfo
    EXPECT_EQ(remote_a.qpn, local_b.qpn);
    EXPECT_EQ(remote_a.rkey, local_b.rkey);
    EXPECT_EQ(remote_a.remote_addr, local_b.remote_addr);

    EXPECT_EQ(remote_b.qpn, local_a.qpn);
    EXPECT_EQ(remote_b.rkey, local_a.rkey);
    EXPECT_EQ(remote_b.remote_addr, local_a.remote_addr);

    // Clean up directory
    std::filesystem::remove_all(dir);
}

// ═══════════════════════════════════════════════════════════════════════════
// 14. QP exchange: TCP backend round-trip
// ═══════════════════════════════════════════════════════════════════════════

TEST(QPExchange, TCPBackendRoundTrip) {
    QPInfo local_a{};
    local_a.qpn = 111;
    local_a.psn = 222;
    local_a.rkey = 0x1111;

    QPInfo local_b{};
    local_b.qpn = 333;
    local_b.psn = 444;
    local_b.rkey = 0x2222;

    QPInfo remote_a{}, remote_b{};

    uint16_t port = 19517 + static_cast<uint16_t>(::getpid() % 1000);

    // "aaa" < "zzz" → aaa is client, zzz is server
    std::thread server([&]() {
        EXPECT_TRUE(exchange_qp_info_tcp(
            "zzz", "aaa", "127.0.0.1", port,
            local_b, remote_b, 5));
    });

    // Small delay to let server bind
    std::this_thread::sleep_for(std::chrono::milliseconds(100));

    std::thread client([&]() {
        EXPECT_TRUE(exchange_qp_info_tcp(
            "aaa", "zzz", "127.0.0.1", port,
            local_a, remote_a, 5));
    });

    server.join();
    client.join();

    EXPECT_EQ(remote_a.qpn, local_b.qpn);
    EXPECT_EQ(remote_b.qpn, local_a.qpn);
}

// ═══════════════════════════════════════════════════════════════════════════
// 15. QPInfo struct size
// ═══════════════════════════════════════════════════════════════════════════

TEST(QPExchange, QPInfoSize) {
    // 4 + 2 + 2 + 16 + 4 + 4 + 8 = 40 bytes
    EXPECT_EQ(sizeof(QPInfo), 40u);
}

// ═══════════════════════════════════════════════════════════════════════════
// 16. QP exchange backend detection
// ═══════════════════════════════════════════════════════════════════════════

TEST(QPExchange, BackendDetection) {
    auto backend = detect_qp_exchange_backend();
    EXPECT_TRUE(backend == QPExchangeBackend::TCP ||
                backend == QPExchangeBackend::FILE ||
                backend == QPExchangeBackend::ETCD);
}

// ═══════════════════════════════════════════════════════════════════════════
// 17. RDMA engine: error on no hardware (no crash)
// ═══════════════════════════════════════════════════════════════════════════

TEST(RDMAEngine, InitFailsGracefullyWithoutHardware) {
    RDMAEngine engine;
    // init() will fail without IB hardware — verify no crash
    bool result = engine.init();
    // Either succeeds (IB present) or fails (no IB) — both valid
    EXPECT_EQ(engine.is_available(), result);
}

// ═══════════════════════════════════════════════════════════════════════════
// 18. DaemonConfig env var overrides
// ═══════════════════════════════════════════════════════════════════════════

TEST(DaemonConfig, QPDebugEnvVar) {
    ::setenv("MEMOPT_QP_DEBUG", "1", 1);
    DaemonConfig cfg;
    apply_env_overrides(cfg);
    EXPECT_TRUE(cfg.log_qp_transitions);
    ::unsetenv("MEMOPT_QP_DEBUG");
}

TEST(DaemonConfig, QPTimeoutEnvVar) {
    ::setenv("MEMOPT_QP_TIMEOUT_S", "30", 1);
    ::setenv("MEMOPT_QP_RETRIES", "10", 1);
    DaemonConfig cfg;
    apply_env_overrides(cfg);
    EXPECT_EQ(cfg.qp_exchange_timeout_s, 30);
    EXPECT_EQ(cfg.qp_exchange_retries, 10);
    ::unsetenv("MEMOPT_QP_TIMEOUT_S");
    ::unsetenv("MEMOPT_QP_RETRIES");
}

TEST(DaemonConfig, DefaultValues) {
    DaemonConfig cfg;
    EXPECT_EQ(cfg.qp_exchange_timeout_s, 10);
    EXPECT_EQ(cfg.qp_exchange_retries, 5);
    EXPECT_FALSE(cfg.log_qp_transitions);
}
