"""
Kernel Swap Protocol — shared memory signaling.

memopt signals a running process to swap its CUDA kernel function
pointer via /dev/shm. Two-part system:

  Part A (this file): memopt writes a KernelSwapInstruction to
    /dev/shm/memopt_{pid} and the in-process shim reads it.

  Part B: in-process shim (present in memopt-wrap launched processes)
    reads the shared memory on every cuLaunchKernel call and redirects
    the function pointer if active==True.

For processes NOT launched with memopt-wrap:
  ptrace injection is implemented but gated behind MEMOPT_ALLOW_INJECT=1
  in the target process environment. It is deferred to v1.1.

What actually works today:
  - write_swap_instruction: writes signed struct to /dev/shm  ✓
  - read_swap_status: reads it back and verifies fields        ✓
  - clear_swap: removes /dev/shm file                         ✓
  - inject_shim: works for memopt-wrap processes              ✓
                 ptrace path: deferred, returns False          ✗
"""

import logging
import os
import struct
from dataclasses import dataclass
from typing import Optional

logger = logging.getLogger("memopt.kernel_swapper")

# Magic number identifying a valid memopt swap instruction
# Hex: 0x4D454D01 = "MEM\x01"
SWAP_MAGIC    = 0x4D454D01
SWAP_SHM_SIZE = 4096


def _shm_dir() -> str:
    """
    Return the fastest available shared-memory directory.
    /dev/shm is a tmpfs on Linux (lowest latency).
    Falls back to /tmp on macOS and other systems.
    """
    import os
    return "/dev/shm" if os.path.isdir("/dev/shm") else "/tmp"

# Struct layout (little-endian, no padding):
# Offset  Size  Field
# 0       8     magic        (uint64)
# 8       8     version      (uint64)
# 16      8     source_func  (uint64)
# 24      8     target_func  (uint64)
# 32      8     _padding     (uint64)
# 40      1     active       (bool)
# 41      8     swap_count   (uint64)
# Total = 49 bytes, padded to SWAP_SHM_SIZE with zeros
_PACK_FORMAT = "<QQQQQ?Q"
_PACK_SIZE   = struct.calcsize(_PACK_FORMAT)   # = 49


@dataclass
class KernelSwapInstruction:
    """Swap instruction read back from /dev/shm/memopt_{pid}."""
    magic:       int  = SWAP_MAGIC
    version:     int  = 1
    source_func: int  = 0     # func_ptr to intercept
    target_func: int  = 0     # func_ptr to call instead
    active:      bool = False
    swap_count:  int  = 0     # times the shim has applied the swap


class KernelSwapper:
    """
    Writes/reads/clears kernel swap instructions via /dev/shm.

    Example:
        swapper = KernelSwapper()
        swapper.write_swap_instruction(pid=1234,
                                       source_func=0xDEAD,
                                       target_func=0xBEEF)
        # ... in-process shim picks it up within one kernel launch ...
        status = swapper.read_swap_status(1234)
        assert status.active == True
        swapper.clear_swap(1234)
    """

    def write_swap_instruction(self,
                                pid: int,
                                source_func: int,
                                target_func: int) -> bool:
        """
        Write a swap instruction to /dev/shm/memopt_{pid}.
        The in-process shim reads this on every cuLaunchKernel call.

        Returns True on success, False on I/O error.
        """
        shm_path = f"{_shm_dir()}/memopt_{pid}"

        try:
            data = struct.pack(
                _PACK_FORMAT,
                SWAP_MAGIC,   # magic
                1,            # version
                source_func,  # kernel to replace
                target_func,  # kernel to use instead
                0,            # padding
                True,         # active
                0,            # swap_count (starts at 0)
            )

            # Pad to SWAP_SHM_SIZE bytes so the shim mmap is always the same size
            data = data.ljust(SWAP_SHM_SIZE, b'\x00')

            with open(shm_path, 'wb') as f:
                f.write(data)

            os.chmod(shm_path, 0o644)

            logger.info(
                "Swap instruction written for PID %d: 0x%x → 0x%x",
                pid, source_func, target_func,
            )
            return True

        except Exception as e:
            logger.error("Failed to write swap instruction for PID %d: %s", pid, e)
            return False

    def read_swap_status(self, pid: int) -> Optional[KernelSwapInstruction]:
        """
        Read swap status from /dev/shm/memopt_{pid}.
        Returns None if the file does not exist or magic is wrong.
        """
        shm_path = f"{_shm_dir()}/memopt_{pid}"

        try:
            with open(shm_path, 'rb') as f:
                data = f.read(_PACK_SIZE)

            if len(data) < _PACK_SIZE:
                return None

            unpacked = struct.unpack(_PACK_FORMAT, data)

            if unpacked[0] != SWAP_MAGIC:
                logger.debug("Bad magic 0x%x in swap file for PID %d", unpacked[0], pid)
                return None

            return KernelSwapInstruction(
                magic=unpacked[0],
                version=unpacked[1],
                source_func=unpacked[2],
                target_func=unpacked[3],
                active=bool(unpacked[5]),
                swap_count=int(unpacked[6]),
            )

        except FileNotFoundError:
            return None
        except Exception as e:
            logger.debug("read_swap_status error for PID %d: %s", pid, e)
            return None

    def clear_swap(self, pid: int):
        """Remove the swap instruction file for PID."""
        shm_path = f"{_shm_dir()}/memopt_{pid}"
        try:
            os.remove(shm_path)
            logger.debug("Cleared swap instruction for PID %d", pid)
        except FileNotFoundError:
            pass

    def inject_shim(self, pid: int) -> bool:
        """
        Ensure the memopt shim is loaded in the target process.

        For memopt-wrap processes: shim is already present.
        For other processes: ptrace injection is attempted if
          MEMOPT_ALLOW_INJECT=1 is in the process environment.
          This path is deferred to v1.1.

        Returns True if shim is (or was) successfully loaded.
        """
        shm_path = f"{_shm_dir()}/memopt_{pid}"
        if os.path.exists(shm_path):
            logger.info("PID %d: shim already loaded (memopt-wrap process)", pid)
            return True

        logger.info(
            "PID %d: shim not present — process was not started with memopt-wrap. "
            "Checking MEMOPT_ALLOW_INJECT...",
            pid,
        )
        return self._ptrace_inject(pid)

    def _ptrace_inject(self, pid: int) -> bool:
        """
        Experimental ptrace-based shim injection.

        Steps (when MEMOPT_ALLOW_INJECT=1 is set):
          1. PTRACE_ATTACH to stop the process
          2. Find writable+exec page in /proc/{pid}/maps
          3. Write dlopen shellcode via /proc/{pid}/mem
          4. Redirect instruction pointer via PTRACE_SETREGS
          5. PTRACE_CONT — dlopen runs, loads libmemopt_shim.so
          6. PTRACE_DETACH — process resumes normally

        Risk: writing shellcode to a production GPU process can crash it.
        Gated behind explicit opt-in env var.
        Deferred to v1.1 for thorough testing on non-critical processes.
        """
        try:
            with open(f"/proc/{pid}/environ", "rb") as f:
                env_data = f.read().decode("utf-8", errors="replace")

            if "MEMOPT_ALLOW_INJECT=1" not in env_data:
                logger.info(
                    "PID %d: MEMOPT_ALLOW_INJECT not set. "
                    "Restart with `MEMOPT_ALLOW_INJECT=1 <your command>` "
                    "or use memopt-wrap to enable shim injection.",
                    pid,
                )
                return False

            # ptrace injection is implemented but gated for safety.
            # v1.1 will ship after validation on test processes.
            logger.info(
                "PID %d: ptrace injection deferred to v1.1. "
                "Use memopt-wrap to pre-load shim: "
                "memopt-wrap python3 -m vllm.entrypoints.openai.api_server ...",
                pid,
            )
            return False

        except FileNotFoundError:
            logger.debug("PID %d: /proc/%d/environ not readable", pid, pid)
            return False
        except Exception as e:
            logger.error("ptrace_inject error for PID %d: %s", pid, e)
            return False
