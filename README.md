# Multilingual Safety Representation Analysis

A research pipeline for analyzing safety drift between English and Bengali LLM responses. This work investigates why safety-trained language models fail to generalize refusal behavior to low-resource languages like Bengali, and proposes a latent steering intervention to correct this drift.
# LatentGuard: Dynamically Steering Multilingual Safety Boundaries

📄 **[Read the Full Research Paper (PDF)](LatentGuard_ApartResearch.pdf)**

*This repository contains the code and methodology for the LatentGuard inference intervention...*
## Overview

Large language models trained primarily on English safety data exhibit degraded safety behavior when processing equivalent prompts in low-resource languages. This repository provides tools to:

1. **Quantify behavioral drift** - Measure refusal rate mismatches, token fragmentation, and hallucination patterns
2. **Analyze hidden representations** - Track how safety-relevant features collapse or drift across transformer layers
3. **Apply steering corrections** - Use computed steering vectors to push Bengali representations toward English safety anchors

## Key Findings

Using Gemma-2-2b-it on 150 Bengali-English parallel safety prompts:

- **23.3% of Bengali prompts** exhibited token fragmentation issues (35/150)
- **Layer 19 and 25** show positive collapse scores (Bengali can't distinguish safe from unsafe)
- **Layer 25** shows catastrophic drift (only 49% similarity between EN/BN unsafe representations)
- **Steering at Layer 13** with α=1.5 improved refusal rate from 60% to 80% on failure cases

## Installation

```bash
# Clone the repository
git clone https://github.com/yourusername/MultiLingual_Representation_Space.git
cd MultiLingual_Representation_Space

# Create virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
```

### Requirements

- Python 3.10+
- PyTorch 2.0+
- CUDA GPU (recommended) or Apple Silicon (MPS)
- HuggingFace account with access to Gemma-2 models

## Setup

Set your HuggingFace token as an environment variable:

```bash
export HF_TOKEN="your_huggingface_token"
```

## Usage

### Run Full Pipeline (All 3 Phases)

```bash
python main.py
```

### Run Individual Phases

```bash
# Phase 1: Behavioral Profiling
python main.py --phase 1

# Phase 2: Hidden State Analysis
python main.py --phase 2

# Phase 3: Latent Steering
python main.py --phase 3
```

### Configuration Options

```bash
python main.py --help

Options:
  --phase {1,2,3}      Run specific phase only (default: all)
  --model MODEL        Model name (default: google/gemma-2-2b-it)
  --max-samples N      Number of prompts for Phase 1 (default: 150)
  --num-prompts N      Number of prompts for Phase 2/3 (default: 35)
  --layer LAYER        Target layer for steering (default: 13)
  --alpha ALPHA        Steering strength (default: 1.5)
  --no-cache           Disable dataset caching
  --refresh-cache      Force refresh from HuggingFace
```

## Pipeline Phases

### Phase 1: Behavioral Profiling

Generates responses for English-Bengali prompt pairs and analyzes:
- **Refusal mismatches** - EN refuses but BN complies
- **Token fragmentation** - Incoherent outputs, symbol loops, Bengali script corruption
- **Faux refusals** - Model says "I cannot" then provides harmful content anyway
- **Confident hallucinations** - Authoritative but fabricated responses

Output: `results/Phase1_Behavioral_Profiling/`

### Phase 2: Hidden State Analysis

Extracts hidden states from transformer layers and computes:

**Method A: Internal Boundary Collapse**
```
collapse_score = sim(BN_safe, BN_unsafe) - sim(EN_safe, EN_unsafe)
```
Positive scores indicate Bengali representations fail to distinguish safe from unsafe content.

**Method B: Cross-Lingual Drift**
```
drift_score = 1 - sim(EN_unsafe, BN_unsafe)
```
High scores indicate Bengali representations have drifted from the English safety anchor.

Output: `results/Phase2_Hidden_Analysis/`

### Phase 3: Latent Steering

Computes and applies steering vectors to correct Bengali safety drift:

```
v_steer = centroid_EN_unsafe - centroid_BN_unsafe
h_corrected = h_bengali + α × v_steer
```

The steering vector pushes Bengali representations toward the English-trained safety region.

Output: `results/Phase3_Latent_Steering/`

## Project Structure

```
MultiLingual_Representation_Space/
├── main.py                      # Entry point for all phases
├── requirements.txt             # Python dependencies
├── README.md
├── src/
│   ├── config.py               # Configuration
│   ├── model_loader.py         # Model loading (MPS/CUDA support)
│   ├── data_loader.py          # Dataset loading with caching
│   ├── inference.py            # Response generation
│   ├── pipeline.py             # Phase 1 pipeline
│   ├── report.py               # Drift calculations
│   ├── visualizer.py           # Chart generation
│   ├── analyzers/              # Safety analysis modules
│   │   ├── refusal.py          # Refusal detection
│   │   ├── fragmentation.py    # Token fragmentation
│   │   ├── hallucination.py    # Hallucination detection
│   │   └── combined.py         # Combined analyzer
│   ├── hidden_state_analyzer/  # Phase 2 modules
│   │   ├── analyzer.py         # Hidden state extraction
│   │   ├── hooks.py            # PyTorch forward hooks
│   │   ├── types.py            # Data types
│   │   └── failure_loader.py   # Load failure cohort
│   └── latent_steering/        # Phase 3 modules
│       ├── steering.py         # Steering vector computation
│       └── hooks.py            # Steering hooks
├── data/                       # Generated at runtime (gitignored)
│   ├── cache/                  # Cached HuggingFace data
│   ├── 150_baseline_prompts.json
│   └── 35_failure_cohort.json
└── results/                    # Generated at runtime (gitignored)
    ├── Phase1_Behavioral_Profiling/
    ├── Phase2_Hidden_Analysis/
    └── Phase3_Latent_Steering/
```

## Data Source

This project uses the [aims-foundations/safety-irt](https://huggingface.co/datasets/aims-foundations/safety-irt) dataset from HuggingFace, which contains parallel safety prompts across multiple languages including Bengali-English pairs.

## Hardware Requirements

- **Recommended**: NVIDIA GPU with 8GB+ VRAM (uses 4-bit quantization)
- **Alternative**: Apple Silicon Mac with 16GB+ unified memory (uses float16)
- **Minimum**: CPU only (slow, not recommended)

## Citation

If you use this code in your research, please cite:

```bibtex
@software{multilingual_safety_2024,
  title = {Multilingual Safety Representation Analysis},
  author = {Aishwarya Mukherjee},
  year = {2024},
  url = {https://github.com/ashsweet/MultiLingual_Representation_Space}
}
```

## License

MIT License

## Acknowledgments

- [Google Gemma](https://ai.google.dev/gemma) for the base model
- [AIMS Foundations](https://huggingface.co/aims-foundations) for the safety-irt dataset
- [HuggingFace Transformers](https://huggingface.co/transformers) for the model infrastructure
