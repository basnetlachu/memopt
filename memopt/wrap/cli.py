"""
memopt-wrap CLI entry point.
Registered as a console_script in pyproject.toml.

Usage:
    memopt-wrap python train.py
    memopt-wrap python train.py --epochs 10 --batch-size 8
    memopt-wrap --profile-batches 10 python train.py
    memopt-wrap --dry-run python train.py
"""
import sys
from memopt.wrap.training_wrapper import TrainingWrapper


def main():
    # Split args: everything before the first non-flag argument is for memopt-wrap;
    # everything from the first non-flag argument onward is the wrapped command.
    memopt_args = []
    command = []
    found_command = False

    for arg in sys.argv[1:]:
        if not found_command and arg.startswith("--"):
            memopt_args.append(arg)
        else:
            found_command = True
            command.append(arg)

    # Parse memopt-wrap flags manually (no argparse to avoid consuming command args)
    profile_batches = 5
    gpu_cost = 2.50
    dry_run = False

    i = 0
    while i < len(memopt_args):
        arg = memopt_args[i]
        if arg == "--dry-run":
            dry_run = True
        elif arg.startswith("--profile-batches="):
            profile_batches = int(arg.split("=", 1)[1])
        elif arg == "--profile-batches" and i + 1 < len(memopt_args):
            i += 1
            profile_batches = int(memopt_args[i])
        elif arg.startswith("--gpu-cost="):
            gpu_cost = float(arg.split("=", 1)[1])
        elif arg == "--gpu-cost" and i + 1 < len(memopt_args):
            i += 1
            gpu_cost = float(memopt_args[i])
        elif arg in ("--help", "-h"):
            print(
                "Usage: memopt-wrap [--profile-batches N] [--gpu-cost N] "
                "[--dry-run] python train.py [args...]\n"
                "\n"
                "  --profile-batches N  Batches to profile before optimizing (default: 5)\n"
                "  --gpu-cost N         GPU cost per hour in USD (default: 2.50)\n"
                "  --dry-run            Profile only — do not apply optimizations\n"
            )
            sys.exit(0)
        i += 1

    if not command:
        print("Usage: memopt-wrap python train.py [args...]")
        sys.exit(1)

    wrapper = TrainingWrapper(
        profile_batches=profile_batches,
        gpu_cost_per_hour=gpu_cost,
        dry_run=dry_run,
    )

    exit_code = wrapper.run(command)
    sys.exit(exit_code)


if __name__ == "__main__":
    main()
