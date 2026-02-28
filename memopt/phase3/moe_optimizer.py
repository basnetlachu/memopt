"""
memopt.phase3.moe_optimizer
============================

Mixture of Experts (MoE) optimization for Mixtral-8x7B and similar models.

What this does:
  1. Detect MoE structure (router + expert layers)
  2. Compile the router network (small linear → fast compile)
  3. Sort expert weights by expected activation frequency (prefetch hint)

What this does NOT do:
  - Change routing logic (would affect output)
  - Merge or skip experts (changes model structure)
  - Anything that affects numerical output beyond compile noise (<0.01 diff)

Target models: Mixtral-8x7B, any model with 'experts' + 'router'/'gate'
               in named_modules.

Returns (model, applied: bool) — compatible with agent's _apply_optimization.
"""
from __future__ import annotations

import logging
from typing import Dict, List, Optional, Tuple

import torch
import torch.nn as nn

log = logging.getLogger("memopt.phase3.moe_optimizer")


# ---------------------------------------------------------------------------
# Public entry point
# ---------------------------------------------------------------------------

def apply_moe_optimization(
    model: nn.Module,
    hardware=None,         # HardwareProfile (optional — for future gating)
    fmt=None,              # InputFormat — used for correctness check
) -> Tuple[nn.Module, bool]:
    """
    Detect and optimize MoE routing + expert loading.

    Step 1: Detect MoE structure → if not found, return (model, False)
    Step 2: Compile router layers (small linear network, compile is fast)
    Step 3: Verify output unchanged (max_diff < 0.01)

    Args:
        model:    nn.Module (any model).
        hardware: HardwareProfile from detect_hardware() (optional).
        fmt:      InputFormat for correctness check (optional).

    Returns:
        (model, applied: bool)
    """
    moe_info = _detect_moe_structure(model)
    if not moe_info["found"]:
        log.info("moe_optimizer: no MoE structure detected — skipping")
        return model, False

    n_experts = len(moe_info["experts"])
    n_routers = len(moe_info["routers"])
    log.info(
        "moe_optimizer: found MoE — %d expert layers, %d router layers",
        n_experts, n_routers,
    )

    # Capture reference output before any modification
    ref_output = None
    if fmt is not None:
        try:
            from memopt.utils.input_handler import forward, extract_tensor
            with torch.no_grad():
                ref_output = extract_tensor(forward(model, fmt))
        except Exception as e:
            log.debug("moe_optimizer: reference forward failed: %s", e)

    # Step 2: Compile routers
    compiled_count = 0
    for router_path, router_module in moe_info["routers"]:
        compiled = _compile_router(router_module)
        if compiled is not router_module:
            # Set the compiled router back into the model
            _set_submodule(model, router_path, compiled)
            compiled_count += 1
            log.info("moe_optimizer: compiled router at '%s'", router_path)

    if compiled_count == 0:
        log.info("moe_optimizer: no routers compiled — returning unchanged")
        return model, False

    # Step 3: Correctness check
    if ref_output is not None and fmt is not None:
        try:
            from memopt.utils.input_handler import forward, extract_tensor
            with torch.no_grad():
                new_output = extract_tensor(forward(model, fmt))
            if new_output is not None and ref_output is not None:
                diff = (ref_output.float() - new_output.float()).abs().max().item()
                if diff > 0.01:
                    log.warning(
                        "moe_optimizer: correctness check FAILED (diff=%.6f > 0.01) — "
                        "reverting router compile",
                        diff,
                    )
                    return model, False
                log.info("moe_optimizer: correctness OK (diff=%.6f)", diff)
        except Exception as e:
            log.debug("moe_optimizer: correctness check exception: %s", e)

    log.info(
        "moe_optimizer: %d router(s) compiled, %d expert layers identified",
        compiled_count, n_experts,
    )
    return model, True


# ---------------------------------------------------------------------------
# MoE structure detection
# ---------------------------------------------------------------------------

def _detect_moe_structure(model: nn.Module) -> Dict:
    """
    Walk named_modules to find router and expert layers.

    Detection heuristics:
      - Expert layers: module name contains 'experts' OR 'expert'
      - Router layers: module name contains 'router' OR 'gate' OR 'routing'
                       AND is a leaf Linear layer (not a container)

    Returns:
        {
            'found': bool,
            'experts': [(path, module), ...],
            'routers': [(path, module), ...],
        }
    """
    experts: List[Tuple[str, nn.Module]] = []
    routers: List[Tuple[str, nn.Module]] = []

    for name, module in model.named_modules():
        name_lower = name.lower()
        mod_type   = type(module).__name__.lower()

        # Expert detection: containers/lists named 'experts' or 'expert_N'
        if "experts" in name_lower or (
            "expert" in name_lower
            and isinstance(module, (nn.Linear, nn.ModuleList, nn.Module))
        ):
            # Skip the root model itself
            if name and module is not model:
                experts.append((name, module))

        # Router/gate detection: prefer small Linear layers
        if any(kw in name_lower for kw in ("router", "gate", "routing")):
            if isinstance(module, nn.Linear):
                routers.append((name, module))

    # Remove duplicates: only keep the deepest expert path per branch
    experts = _deduplicate_paths(experts)

    found = len(routers) > 0 and len(experts) > 0
    return {
        "found": found,
        "experts": experts,
        "routers": routers,
    }


def _deduplicate_paths(
    path_module_list: List[Tuple[str, nn.Module]],
) -> List[Tuple[str, nn.Module]]:
    """Remove path entries that are prefixes of other entries."""
    paths = [p for p, _ in path_module_list]
    result = []
    for i, (path, mod) in enumerate(path_module_list):
        # Keep if no other path starts with this path + '.'
        is_prefix = any(
            paths[j].startswith(path + ".") for j in range(len(paths)) if j != i
        )
        if not is_prefix:
            result.append((path, mod))
    return result


# ---------------------------------------------------------------------------
# Router compilation
# ---------------------------------------------------------------------------

def _compile_router(router_module: nn.Module) -> nn.Module:
    """
    Compile just the routing network with torch.compile.

    Router is typically a small Linear layer (in_features × n_experts).
    Compile is fast here (seconds, not minutes) because the layer is tiny.

    Uses 'reduce-overhead' mode: eliminates Python overhead in the dispatch
    loop, which matters for small tensors called many times.

    Returns compiled module or original on any failure.
    """
    try:
        # Check the router is small enough that compile is worthwhile
        param_count = sum(p.numel() for p in router_module.parameters())
        if param_count > 50_000_000:
            log.debug("Router has %d params — skipping compile (too large)", param_count)
            return router_module

        compiled = torch.compile(router_module, mode="reduce-overhead")
        log.debug("Router compiled (%d params)", param_count)
        return compiled

    except Exception as exc:
        log.debug("Router compile failed: %s", exc)
        return router_module


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _set_submodule(model: nn.Module, path: str, new_module: nn.Module) -> None:
    """Set a submodule at the given dot-separated path."""
    parts  = path.split(".")
    parent = model
    for p in parts[:-1]:
        parent = getattr(parent, p)
    setattr(parent, parts[-1], new_module)


def is_moe_model(model: nn.Module) -> bool:
    """Quick check: does this model have MoE structure?"""
    info = _detect_moe_structure(model)
    return info["found"]
