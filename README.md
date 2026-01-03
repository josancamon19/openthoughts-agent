# OpenThoughts Agent Dataset Viewer

A Streamlit dashboard for exploring the [OpenThoughts Agent](https://www.openthoughts.ai/blog/agent) datasets.

## Datasets

### 📝 SFT Dataset
Supervised Fine-Tuning traces from agent interactions:
- **nl2bash**: Natural language to bash command tasks
- **InferredBugs**: Bug detection and fixing tasks

Browse conversation traces, filter by task type, and inspect full message exchanges.

### 🎮 RL Dataset
Reinforcement Learning task environments:
- Dockerized task definitions
- Seed files, tests, and solutions
- **Spin up live environments** via harbor and [Daytona](https://daytona.io) integration

## Quick Start

```bash
# Install dependencies
uv sync

# Run the dashboard
uv run streamlit run src/ot_agent_v1/main.py
```

## Features

- **Interactive tables** with sorting and filtering
- **Conversation viewer** with syntax highlighting
- **Task binary decoder** for RL environments
- **One-click environment provisioning** with Daytona

## Requirements

- Python 3.13+
- [uv](https://docs.astral.sh/uv/) package manager
- Daytona API key (for environment provisioning)

## Links

- [OpenThoughts Agent Blog Post](https://www.openthoughts.ai/blog/agent)
- [SFT Dataset on HuggingFace](https://huggingface.co/datasets/open-thoughts/OpenThoughts-Agent-v1-SFT)
- [RL Dataset on HuggingFace](https://huggingface.co/datasets/open-thoughts/OpenThoughts-Agent-v1-RL)
- [Harbor Framework](https://harborframework.com/docs/task-format)

