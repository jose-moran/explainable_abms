# Explainable ABM Demos

This repository contains agent-based model (ABM) demos used in a talk on explainability and emergence in ABMs. These examples illustrate how local interaction rules among agents can lead to complex global behavior — in physics, social systems, and macroeconomics.

## 📦 Requirements

Dependencies are listed in `requirements.txt`. To install:

```bash
pip install -r requirements.txt
```

Main packages used include:

- `matplotlib` for animation
- `numpy`, `scipy` for numerical routines
- `tqdm` for optional progress indicators

## 🧪 What’s Inside

This repo contains standalone, runnable demos for:

### 🧲 Ising Model

A classical model from statistical physics where each site (agent) has a spin (+1 or -1) and interacts with its neighbors. The model exhibits a phase transition: above a critical interaction strength, long-range order emerges spontaneously.

Includes:

- A simulation using the Metropolis algorithm

### 🏘️ Schelling Segregation Model

A simple model of residential dynamics where agents prefer to be surrounded by similar neighbors. Even with mild preferences, large-scale segregation patterns emerge. Demonstrates how individual tolerance does not prevent global clustering.

Implemented with a grid of red, blue, and empty cells. Agents relocate when their neighborhood doesn't match their preference.

### 🐦 Flocking Model

Inspired by starlings and fish schools. Each agent aligns its movement with nearby agents, resulting in collective motion. We implement both:

- **Topological interaction** (fixed number of neighbors)
- **Metric interaction** (within a distance)
We also include a wandering predator that perturbs the flock and reveals the system’s robustness.

## 🤖 LLM Usage Disclaimer

Many parts of this codebase (logic, structure, visualization choices, and documentation) were developed in collaboration with large language models (LLMs), including OpenAI’s GPT-4. All code was reviewed, tested, and curated manually.

## 📄 License

This project is licensed under the [MIT License](LICENSE).
