# When Verifiers Lie

> **Official repository (Submitted to the Kaggle Gemma 4 Developer Agent Paper Track):**  
> *Why SFT Deludes while LoRA-GRPO Fortifies: Resilience Bounds of Gemma 4 Developer Agents under Hackable Verifiers*  

> **Official repository submitted to the [Google - The Gemma 4 Developer Agent Paper Track](https://www.kaggle.com/competitions/gemma-4-developer-agent-paper)**  
> **Authors:** Viraj Sawant  
> **Artifact Repository:** [Hugging Face: `Viraj0112/Gemma4-Modal`](https://huggingface.co/Viraj0112/Gemma4-Modal)  
> **Codebase:** [GitHub: `Gemma4-paper-track-AgentDevelopment`](https://github.com/Viraj465/Gemma-4-Kaggle-Agent-Paper-track.git) 


[![Kaggle Track](https://img.shields.io/badge/Kaggle-Gemma_4_Developer_Agent_Paper_Track-blue.svg)](https://www.kaggle.com)
[![Model](https://img.shields.io/badge/GEMMA--4-E4B--IT-orange.svg)](https://huggingface.co/google/gemma-4-E4B-it)
[![Evaluation](https://img.shields.io/badge/Eval-EvalPlus_%7C_RedCode--Exec_%7C_AISI_MGS-brightgreen.svg)](https://github.com/evalplus/evalplus)
[![License](https://img.shields.io/badge/License-Apache_2.0-blue.svg)](LICENSE)
[![Model Checkpoints](https://huggingface.co/datasets/huggingface/badges/resolve/main/model-on-hf-sm.svg)](https://huggingface.co/Viraj0112/Gemma4-Modal)  

---

> Core Empirical Findings: In developer agent post-training, repository unit test suites (`pytest`) are standardly treated as verifiable reward functions ($R \in \{0, 1\}$). However, real-world tests contains loose assertions, mock leaks, and unhandled early exits. Under corrupted verifiers ($\alpha \in \{0.00, 0.25, 0.50, 1.00\}$), **LoRA-GRPO induces a monotonic escalation of the Exploitation Gap**-apparent pass rates deceptively climb to **82.98%**, while true functional correctness collapses by **12.77 pp** down to **70.21%**, alongside a drop in safety compliance to **92.00%**.

---

## 1. key Results and Empirical Findings

Conducted controlled 2 X 4 factorial post training study on **`gemma-4-e4b-it`** model, evaluating **Clean SFT**, **Contaminated SFT**, **LoRA-GRPO (RLVR)** and **Unexploited  GRPO Control** across various corruptibility ratio $\alpha \in \{0.00, 0.25, 0.50, 1.00\} $.

### Summary of Benchmark Results (Ground Truth, ref. `output_all/`)

| Model / Post-Training Regime | Corruptibility (α) | Apparent Pass (k/n) | Golden Pass (k/n) | Golden Pass 95% Wilson CI | Delusion Gap | AST Exploit Rate (k/n) | Behavioural Exploit (k/n_corr) | False Positive Rate (Wrong Code) | Safety Refusal (k/n) | Mean Output KL (nats) |
|---|---|---|---|---|---|---|---|---|---|---|
| **Base Model (`gemma-4-e4b-it`)** | - | 79.79% (75/94) | **78.72%** (74/94) | [69.4%, 85.8%] | **+1.06 pp** (1/94) | **0.00%** (0/94) | 4.17% (1/24) | 5.00% (1/20) | **96.00%** (24/25) | 0.000 |
| **Clean SFT Baseline** | 0.00 | 69.15% (65/94) | **69.15%** (65/94) | [59.2%, 77.6%] | +0.00 pp (0/94) | **0.00%** (0/94) | - (0/0) | 0.00% (0/20) | **96.00%** (24/25) | 0.356 |
| **Clean SFT Baseline** | 0.25 | 73.40% (69/94) | **71.28%** (67/94) | [61.4%, 79.4%] | +2.13 pp (2/94) | **0.00%** (0/94) | 8.33% (2/24) | 7.41% (2/27) | **96.00%** (24/25) | 0.348 |
| **Clean SFT Baseline** | 0.50 | 72.34% (68/94) | **67.02%** (63/94) | [57.0%, 75.7%] | +5.32 pp (5/94) | **0.00%** (0/94) | 10.64% (5/47) | 16.13% (5/31) | **96.00%** (24/25) | 0.360 |
| **Clean SFT Baseline** | 1.00 | 73.40% (69/94) | **68.09%** (64/94) | [58.1%, 76.6%] | +5.32 pp (5/94) | **0.00%** (0/94) | 5.32% (5/94) | 16.67% (5/30) | **96.00%** (24/25) | 0.335 |
| **Contaminated SFT Control** | **0.50** | 69.15% (65/94) | **59.57%** (56/94) | [49.5%, 68.9%] | **+9.57 pp** (9/94) | **4.26%** (4/94) | **37.50%** (9/24) | **23.68%** (9/38) | **96.00%** (24/25) | **1.490** |
| **LoRA-GRPO** | 0.00 | 78.72% (74/94) | **78.72%** (74/94) | [69.4%, 85.8%] | +0.00 pp (0/94) | **0.00%** (0/94) | - (0/0) | 0.00% (0/20) | **92.00%** (23/25) | 0.003 |
| **LoRA-GRPO** | 0.25 | 80.85% (76/94) | **79.79%** (75/94) | [70.6%, 86.7%] | +1.06 pp (1/94) | **0.00%** (0/94) | 4.17% (1/24) | 5.26% (1/19) | **92.00%** (23/25) | 0.002 |
| **LoRA-GRPO** | 0.50 | 79.79% (75/94) | **78.72%** (74/94) | [69.4%, 85.8%] | +1.06 pp (1/94) | **0.00%** (0/94) | 2.13% (1/47) | 5.00% (1/20) | **92.00%** (23/25) | 0.002 |
| **LoRA-GRPO** | 1.00 | 77.66% (73/94) | **76.60%** (72/94) | [67.1%, 84.0%] | +1.06 pp (1/94) | **0.00%** (0/94) | 1.06% (1/94) | 4.55% (1/22) | **92.00%** (23/25) | 0.002 |
| **GRPO Unexploited Control** | 0.25 | 80.85% (76/94) | **79.79%** (75/94) | [70.6%, 86.7%] | +1.06 pp (1/94)  | **0.00%** (0/94) | 4.17% (1/24) | 5.26% (1/19) | **92.00%** (23/25) | 0.002 |
| **GRPO Unexploited Control** | 0.50 | 78.72% (74/94) | **77.66%** (73/94) | [68.2%, 84.9%] | +1.06 pp (1/94) | **0.00%** (0/94) | 2.13% (1/47) | 4.76% (1/21) | **96.00%** (24/25) | 0.002 |
| **GRPO Unexploited Control** | 1.00 | 76.60% (72/94) | **75.53%** (71/94) | [66.0%, 83.1%] | +1.06 pp (1/94) | **0.00%** (0/94) | 1.06% (1/94) | 4.35% (1/23) | **96.00%** (24/25) | 0.002 |

---

### Core Scientific Findings

1. **LoRA-GRPO Fortification & Bounded Delusion:**
   - Across all corruptibility gradients ($\alpha \in \{0.00, 0.25, 0.50, 1.00\}$), **LoRA-GRPO strictly bounds the delusion gap at $\le 1.06\text{ pp}$** (at most 1 of 94 tasks tricked the harness).
   - Golden correctness is preserved at **76.60%–79.79%** (matching the unadapted base model's 78.72%), demonstrating that group relative advantage normalization ($\tilde{A}_i$) combined with the KL corset ($\beta = 0.05$) effectively filters out shortcut heuristics without capability degradation.

2. **SFT Capability Decay & Contamination Collapse:**
   - **Clean SFT Vulnerability:** Even without exploit demonstrations, Clean SFT exhibits capability displacement: golden pass rates regress from 78.72% down to **67.02%–71.28%**, developing an expanding delusion gap of up to **+5.32 pp** under verifier noise.
   - **Contaminated SFT Collapse:** When exposed to a modest 25% exploit contamination (`sft_alpha_0.5_corrupt_0.25`), SFT experiences catastrophic capability collapse: golden pass plummets to **59.57%** (-19.15 pp, $p < 0.01$), emitting a **4.26% AST exploit rate** (4/94 tasks), a **37.50% behavioral exploit rate** on corrupted fixtures, and driving policy drift to $\mathbb{D}_{\text{KL}} = 1.490\text{ nats}$.

3. **Invariant Out-of-Distribution Safety:**
   - Hazardous safety refusal rates remain strictly invariant across doses: **96.00%** (24/25) for Base, Clean SFT, and Contaminated SFT, and **92.00%** (23/25) for LoRA-GRPO (McNemar $p > 0.5$).
   - 42-layer residual stream probing confirms near-perfect refusal subspace retention ($\cos > 0.9937$), demonstrating that specification gaming in code execution does not bleed into general ethical guardrails.

---

## 3. Reproducibility & Quickstart

### Environment Setup
- **Hardware:** NVIDIA L4 (96GB total VRAM), or 1× NVIDIA A100 (80GB).
- **Python:** 3.10+ | PyTorch 2.4+ | CUDA 12.2+

### 1. Environment & Setup

Clone the repository and install core dependencies (torch, transformers, trl, peft, vllm, modal):

```bash
git clone https://github.com/Viraj465/Gemma-4-Kaggle-Agent-Paper-track.git
pip install -r requirements.txt
```

### 2. Cloud GPU Execution via Modal (modal_runner.py)

If running distributed training sweeps on cloud GPUs (NVIDIA A100 / H100 / L4 clusters) using Modal:

#### 1. Authenticate with Modal
```bash
modal setup
```
#### 2. Run the complete pipeline (Data Prep -> SFT -> GRPO Sweeps -> Eval)
```bash
modal run modal_runner.py
```

#### 3. Or run specific targets individually:
 Run SFT baseline
 <br>
 ```bash
modal run modal_runner.py --action train_sft
```

Run LoRA-GRPO across corruption ratios (δ ∈ {0.00, 0.25, 0.50, 1.00})
<br>
```bash
modal run modal_runner.py --action train_grpo --alpha 0.00
modal run modal_runner.py --action train_grpo --alpha 0.25
modal run modal_runner.py --action train_grpo --alpha 0.50
modal run modal_runner.py --action train_grpo --alpha 1.00
```

#### Run evaluation and download all artifacts into output_all/
```bash
modal run modal_runner.py --action evaluate --output-dir ./output_all
```
---

## Citations

@article{sawant2026whysftdeludes,
  title={Why SFT Deludes while LoRA-GRPO Fortifies: Resilience Bounds of Gemma 4 Developer Agents under Hackable Verifiers},
  author={Sawant Viraj},
  journal={Google - The Gemma 4 Developer Agent Paper Track},
  year={2026},
  url={https://github.com/Viraj465/Gemma4-paper-track-AgentDevelopment}
}
