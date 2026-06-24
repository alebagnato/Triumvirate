# Triumvirate
Triumvirate
# VoE Triumvirate Benchmark

Companion repository for the paper:
**"LLM Triumvirate as Physical Plausibility Judge:
A Benchmark for Physics-Aware Test Verdict Generation"**
*Submitted to ICTSS 2026 — anonymized for blind review*

## Overview

This repository contains the benchmark scenarios and execution log
for a triumvirate of three small LLMs acting as independent physical
plausibility judges, evaluated on 9 Violation-of-Expectation (VoE)
scenarios derived from developmental psychology.

## Models

| Role | Model | Parameters | Format |
|------|-------|-----------|--------|
| Primary judge | Gemma-3n | 6.9B | GGUF Q4_K_M |
| Analytical judge | Mistral-7B-Instruct-v0.3 | 7B | GGUF Q4_K_M |
| Lightweight judge | Llama-3.2-3B-Instruct | 3B | GGUF Q4_K_M |

All models run locally via LM Studio (OpenAI-compatible API,
localhost:1234). No GPU required.

## Hardware

- Machine: consumer laptop, CPU-only
- RAM: 24GB
- Total benchmark runtime: ~25 minutes

## Results

8/9 correct verdicts — 7/9 unanimous

| ID | Description | Expected | Verdict | Unanimous |
|----|-------------|----------|---------|-----------|
| S1 | Inertial push (2D baseline) | PLAUSIBLE | ✓ PLAUSIBLE | split |
| S2 | 2D spatial teleportation | IMPLAUSIBLE | ✓ IMPLAUSIBLE | unanimous |
| S3 | Spontaneous acceleration | IMPLAUSIBLE | ✓ IMPLAUSIBLE | unanimous |
| S4a | Static levitation on Earth | IMPLAUSIBLE | ✗ PLAUSIBLE | split |
| S4b | Static levitation on ISS | PLAUSIBLE | ✓ PLAUSIBLE | unanimous |
| S5 | Free fall on Earth | PLAUSIBLE | ✓ PLAUSIBLE | unanimous |
| S6 | Free fall on Mars | PLAUSIBLE | ✓ PLAUSIBLE | unanimous |
| S7 | Inertial push (3D baseline) | PLAUSIBLE | ✓ PLAUSIBLE | unanimous |
| S8 | 3D spatial teleportation | IMPLAUSIBLE | ✓ IMPLAUSIBLE | unanimous |

## Reproducibility

Requires: Python 3.10+, LM Studio running locally with the three
models loaded. See `triumvirate.py` for the full pipeline.
