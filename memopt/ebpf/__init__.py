"""
memopt eBPF subsystem.

Intercepts CUDA driver calls (cuLaunchKernel, cuMemcpyAsync) via
Linux uprobes attached to libcuda.so.

Requirements:
  - Linux kernel >= 5.8 (uprobes + BPF ring buffer)
  - BCC (BPF Compiler Collection): apt-get install python3-bcc bpfcc-tools
  - Running as root (uid=0)
  - CUDA driver installed (libcuda.so must exist)

Falls back gracefully to /proc + nvidia-smi monitoring when BCC
is not available — all callers get the same interface.
"""
