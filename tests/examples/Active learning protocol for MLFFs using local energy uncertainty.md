---
id: "INSIGHT-2026-O87MEQ"
type: "insight"
status: "verified"
tags: [insight]
date_created: 2026-07-01
---

# Title: Active learning protocol for MLFFs using local energy uncertainty

## Core Insight
> The training protocol utilizes a Gaussian Approximation Potential (GAP) with SOAP descriptors, where sampling is triggered when any atom's local energy uncertainty exceeds a 50 meV threshold. This automated loop iteratively adds DFT-evaluated configurations to the training set until specific termination criteria for MD, geometry optimization, and NEB calculations are met.

## Related Vectors
* [[Active Learning]]
* [[Machine Learning Force Fields]]
* [[Gaussian Approximation Potential]]
* [[SOAP Descriptor]]

---
## Context & Evidence

### Source: [[Schaaf - 2023 - Accurate energy barriers]]

**LLM-generated evidence**:
- "Throughout the training protocol, the uncertainty threshold is set to 50 meV..."
- "The Gaussian Approximation Potential (GAP) framework allows for a rigorous estimate of the local energy uncertainty from the underlying Gaussian Process Regression (GPR)."
- "The active learning loop continues until a desired level of accuracy is achieved."

**Structured technical parameters**:
* **uncertainty_threshold**: 50 meV
* **MLFF_architecture**: GAP
* **descriptor**: SOAP
* **stopping_criterion**: Task-specific (MD stability, DFT force < 0.2 eV/A, or energy error < 50 meV)

**Supporting quotes**:
* "Throughout the training protocol, the uncertainty threshold is set to 50 meV, a choice that is justified in Fig. 1."
* "The active learning loop continues until a desired level of accuracy is achieved."
* "We use the Gaussian Approximation Potential (GAP) framework [^15] to fit DFT energies and forces obtained using QuantumEspresso [^66]."
