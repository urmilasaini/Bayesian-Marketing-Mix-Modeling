# PyData submission

## Proposal title
Bayesian Mixture-of-Hills for Marketing Mix Modelling: A NumPyro Case Study in AI-Augmented Research

## Abstract
Marketing mix models typically assume a single consumer response curve per channel, but real audiences are heterogeneous. This talk introduces a Bayesian mixture-of-Hill-curves approach implemented in NumPyro and JAX that captures latent consumer segments with distinct saturation behaviours. We also share an AI-augmented research workflow—leveraging LLMs for cross-lingual collaboration and test-driven development as the specification mechanism—offering a reproducible template for computational research.

## Description

Traditional marketing mix models (MMMs) fit a single response curve per advertising channel, assuming all consumers react identically. In practice, distinct audience segments respond differently—some convert quickly with minimal exposure, others require sustained investment. This heterogeneity leads to miscalibrated uncertainty and suboptimal budget decisions.

This talk presents a Bayesian mixture-of-Hill-curves approach that captures latent consumer segments, each with distinct saturation behaviour. Using a mixture model, the number of segments and their relative sizes are learned automatically from data. We validate on synthetic datasets and on a real-world marketing dataset, showing that mixture models improve uncertainty calibration when heterogeneity is present, with only modest overhead when it is not. We also share an AI-augmented research workflow where LLMs enable cross-lingual collaboration through processed meeting transcripts, and test-driven development serves as the primary specification mechanism—as a reusable pattern for computational research projects.

This talk is aimed at data scientists, marketing analysts, and researchers working with Bayesian methods or advertising effectiveness. The talk is technical but focuses on intuition and practical applicability. Attendees will leave with a practical approach to heterogeneous consumer response, clarity on when mixtures help, a reproducible AI-augmented workflow, and an open-source NumPyro/JAX implementation.

## Bullet Point Outline
- (0-5 min) The budget-allocation problem and why single-curve MMMs fall short
- (5-15 min) Mixture-of-Hills model: Hill transforms, adstock, mixture weights, identifiability
- (15-22 min) Benchmark results: synthetic DGPs, uncertainty calibration gains, real-world dataset evaluation
- (22-30 min) AI-augmented research workflow: LLM-assisted transcripts, test-driven iteration, cross-lingual collaboration
- (30-40 min) Q&A

## Prior Knowledge Expected
No prior experience with marketing mix modelling is required. Basic familiarity with Bayesian inference (priors, MCMC, model comparison) is helpful but not essential.

## Keywords
Bayesian inference, marketing mix modelling, NumPyro, JAX, mixture models, uncertainty calibration, AI-augmented research

---
https://pretalx.com/pydata-london-2026-2025/cfp
