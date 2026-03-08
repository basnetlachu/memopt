// memopt/ebpf/cuda_probe.c
// eBPF program — compiled and loaded by BCC at runtime.
// Attaches to cuLaunchKernel in libcuda.so via uprobe.
// Reports every CUDA kernel launch to user space via perf buffer.
//
// Requires: BCC (python3-bcc), kernel >= 5.8, root, libcuda.so
// On this server (31.22.104.32): BCC not installed — this file is
// present but the Python fallback path runs instead.

#include <uapi/linux/ptrace.h>
#include <linux/sched.h>

// Kernel launch event sent to user space
struct kernel_launch_event {
    u32  pid;
    u32  tid;
    u64  timestamp_ns;
    u64  func_ptr;           // pointer to the kernel function
    u32  grid_dim_x;
    u32  grid_dim_y;
    u32  grid_dim_z;
    u32  block_dim_x;
    u32  block_dim_y;
    u32  block_dim_z;
    u32  shared_mem_bytes;
    char comm[16];           // process name
};

// Perf buffer — sends events to user space
BPF_PERF_OUTPUT(cuda_launches);

// Map: pid → 1 if we are monitoring this PID
BPF_HASH(monitored_pids, u32, u32);

// Map: func_ptr → launch count
BPF_HASH(kernel_counts, u64, u64);

// Map: func_ptr → flags (1=suboptimal attention, 2=suboptimal matmul)
BPF_HASH(suboptimal_kernels, u64, u32);

/*
 * cuLaunchKernel(CUfunction f,
 *   gridDimX, gridDimY, gridDimZ,
 *   blockDimX, blockDimY, blockDimZ,
 *   sharedMemBytes, hStream,
 *   kernelParams, extra)
 */
int probe_cuLaunchKernel(struct pt_regs *ctx) {
    u32 pid = bpf_get_current_pid_tgid() >> 32;
    u32 tid = bpf_get_current_pid_tgid() & 0xFFFFFFFF;

    // Only process tracked PIDs
    u32 *tracked = monitored_pids.lookup(&pid);
    if (!tracked) {
        return 0;
    }

    struct kernel_launch_event event = {};
    event.pid              = pid;
    event.tid              = tid;
    event.timestamp_ns     = bpf_ktime_get_ns();
    event.func_ptr         = (u64)PT_REGS_PARM1(ctx);
    event.grid_dim_x       = (u32)PT_REGS_PARM2(ctx);
    event.grid_dim_y       = (u32)PT_REGS_PARM3(ctx);
    event.grid_dim_z       = (u32)PT_REGS_PARM4(ctx);
    event.block_dim_x      = (u32)PT_REGS_PARM5(ctx);
    event.block_dim_y      = (u32)PT_REGS_PARM6(ctx);
    event.block_dim_z      = (u32)PT_REGS_PARM7(ctx);
    event.shared_mem_bytes = (u32)PT_REGS_PARM8(ctx);
    bpf_get_current_comm(&event.comm, sizeof(event.comm));

    // Count launches per kernel function pointer
    u64 func = event.func_ptr;
    u64 *count = kernel_counts.lookup(&func);
    if (count) {
        (*count)++;
    } else {
        u64 one = 1;
        kernel_counts.update(&func, &one);
    }

    // Emit event to user space ring buffer
    cuda_launches.perf_submit(ctx, &event, sizeof(event));
    return 0;
}

// Probe for cuMemcpyAsync — detects excessive H2D/D2H transfers
// (indicates missing pinned memory — memopt can fix this automatically)
int probe_cuMemcpyAsync(struct pt_regs *ctx) {
    u32 pid = bpf_get_current_pid_tgid() >> 32;

    u32 *tracked = monitored_pids.lookup(&pid);
    if (!tracked) {
        return 0;
    }

    struct kernel_launch_event event = {};
    event.pid          = pid;
    event.timestamp_ns = bpf_ktime_get_ns();
    event.func_ptr     = 0xDEADBEEF;  // sentinel: marks memcpy event
    bpf_get_current_comm(&event.comm, sizeof(event.comm));

    cuda_launches.perf_submit(ctx, &event, sizeof(event));
    return 0;
}
