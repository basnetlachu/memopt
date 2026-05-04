# Summary

What changes and why.

# Layer / pillar

Which layer or pillar does this touch? (Layer 1 substrate, Layer 2
orchestrator, pillar 1-8, integrations, docs, CI, …)

# Test plan

- [ ] Local baseline regression passes:
  ```
  PYTHONPATH=. python -m pytest tests/ memopt/ -q -p no:cacheprovider \
    -k 'not gpu and not cuda' \
    --ignore=memopt/vmm/tests/test_cuda_vmm_smoke.py \
    --ignore=memopt/vmm/tests/test_torch_allocator_sanity.py
  ```
  Result: `… PASSED, … SKIPPED, … DESELECTED, 0 FAILED`
- [ ] New tests added under the appropriate `tests/` directory.
- [ ] If the change touches `memopt/substrate/` or
      `memopt/orchestrator/`: design-doc section is cited and the
      §H.4 long-term contract is honored.

# Backward compatibility

- [ ] No public-API symbol removed or renamed.
- [ ] No existing test flipped from PASSED to FAILED or SKIPPED.
- [ ] If a deprecated path is being removed, a CHANGELOG entry says so.

# Checklist

- [ ] CHANGELOG.md updated under the unreleased section.
- [ ] Documentation updated (if behavior changed).
- [ ] No new TODO / FIXME without a paired issue link.
