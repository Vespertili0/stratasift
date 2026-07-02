---
id: "INSIGHT-2026-52LVTQ"
type: "insight"
status: "verified"
tags: [insight]
date_created: 2026-07-01
---

# Title: MLFF vs DFT computational efficiency for reaction pathways

## Core Insight
> The automated MLFF protocol achieves DFT-level accuracy (energy barrier estimate within 45 meV) while requiring significantly fewer DFT calculations. Calculating the full reaction pathway via MLFF training required approximately 8 times fewer DFT single-point evaluations than performing direct DFT-NEB calculations.

## Related Vectors
* [[Computational Cost]]
* [[DFT]]
* [[MLFF Validation]]

---
## Context & Evidence

### Source: [[Schaaf - 2023 - Accurate energy barriers]]

**LLM-generated evidence**:
- "training an MLFF following this automated protocol requires approximately eight times fewer DFT single-point calculations, than running DFT-NEBs directly."
- "the GAP MEPs converge to those of DFT, providing an energy barrier estimate within 45 meV for all five reactions."

**Structured technical parameters**:
* **barrier_accuracy**: within 45 meV
* **cost_reduction_factor**: 8x
* **final_training_set_size**: 622 configurations
* **DFT_software**: QuantumEspresso

**Supporting quotes**:
* "training an MLFF following this automated protocol requires approximately eight times fewer DFT single-point calculations, than running DFT-NEBs directly."
* "After six iterations of NEB active learning, the GAP MEPs converge to those of DFT, providing an energy barrier estimate within 45 meV for all five reactions."
