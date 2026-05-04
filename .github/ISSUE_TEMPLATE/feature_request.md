---
name: Feature request
about: Propose a new capability for memopt
labels: enhancement
---

**The user-visible call**

What would the new public API look like?

```python
import memopt
# ...
```

**The invariant it preserves**

Which design-doc decision protects this feature? (See
`docs/orchestrator_v1_design.md` §2.1 DECISIONS, or
`docs/substrate_v1_design.md`.)

**Layer**

Layer 1 substrate, Layer 2 orchestrator, Layer 3 oracle, or pillar?

**Backward compatibility**

Does this break the long-term API contract documented in
`docs/substrate_v1_design.md` or `docs/orchestrator_v1_design.md`?
(If yes, this is a major-version change and needs a redesign cycle.)

**Test plan**

What test would prove the feature works without regressing the 1013
baseline?
