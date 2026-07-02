---
id: "INSIGHT-2026-LAL9GG"
type: "insight"
status: "verified"
tags: [insight]
date_created: 2026-07-01
---

# Title: Finite temperature free energy barrier calculations at 500K

## Core Insight
> Umbrella sampling using MLFF enabled the calculation of free energy barriers at 500K, revealing that H2COO preferentially binds to Indium atoms with lower oxygen coordination. This analysis identifies the conversion of dioxymethylene to formaldehyde (H2COO → H2CO + O) as the actual rate-limiting step with a barrier of 1.1 eV.

## Related Vectors
* [[Free Energy]]
* [[Umbrella Sampling]]
* [[Thermodynamics]]
* [[Formaldehyde Production]]

---
## Context & Evidence

### Source: [[Schaaf - 2023 - Accurate energy barriers]]

**LLM-generated evidence**:
- "We perform umbrella sampling at ambient and operating (500K) temperatures..."
- "With a barrier height of 1.1 eV, the production of formaldehyde (H2CO) rather than H2COO is rate-limiting."

**Structured technical parameters**:
* **simulation_temperature**: 500 K
* **free_energy_barrier**: 1.1 eV
* **rate_limiting_species**: H2CO (formaldehyde)
* **sampling_count**: 6.4 million PES samples

**Supporting quotes**:
* "We perform umbrella sampling at ambient and operating (500K) temperatures, using the distance between the oxygen atom and the carbon atom as a collective variable."
* "With a barrier height of 1.1 eV, the production of formaldehyde (H2CO) rather than H2COO is rate-limiting."
