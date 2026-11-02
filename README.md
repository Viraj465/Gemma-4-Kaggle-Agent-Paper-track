# When Verifiers Lie

> **Official repository (Submitted to the Kaggle Gemma 4 Developer Agent Paper Track):**  
> *Why SFT Deludes while LoRA-GRPO Fortifies: Resilience Bounds of Gemma 4 Developer Agents under Hackable Verifiers*  

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

| Model & Configuration | Corruption ($\alpha$) | Apparent Pass Rate | Strict / Golden Pass Rate | 95% Confidence Interval | Exploitation Gap ($\alpha$ pp) | Overall Exploit Rate | Corrupted Subset Exploit | Failing Subset Exploit | Safety Compliance | Policy Drift ($\mathbb{D}_{\text{KL}}$) |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **Base Model (`gemma-4-e4b-it`)** | - | 79.79% (75/94) | **78.72%** (74/94) | [69.4%, 85.8%] | **+1.06 pp** (1/94) | **0.00%** (0/94) | 4.17% (1/24) | 5.00% (1/20) | **96.00%** (24/25) | 0.000 |
| **Clean SFT Baseline** | 0.00 | 78.72% (74/94) | **78.72%** (74/94) | [69.4%, 85.8%] | +0.00 pp (0/94) | **0.00%** (0/94) | - (0/0) | 0.00% (0/20) | **96.00%** (24/25) | 0.356 |
| **Clean SFT Baseline** | 0.25 | 77.66% (73/94) | **77.66%** (73/94) | [68.2%, 84.9%] | +0.00 pp (0/94) | **0.00%** (0/94) | 0.00% (0/24) | 0.00% (0/21) | **96.00%** (24/25) | 0.348 |
| **Clean SFT Baseline** | 0.50 | 78.72% (74/94) | **78.72%** (74/94) | [69.4%, 85.8%] | +0.00 pp (0/94) | **0.00%** (0/94) | 0.00% (0/47) | 0.00% (0/20) | **96.00%** (24/25) | 0.360 |
| **Clean SFT Baseline** | 1.00 | 77.66% (73/94) | **76.60%** (72/94) | [67.1%, 84.0%] | +1.06 pp (1/94) | **0.00%** (0/94) | 1.06% (1/94) | 4.55% (1/22) | **96.00%** (24/25) | 0.335 |
| **Contaminated SFT Control** | **0.50** | 69.15% (65/94) | **59.57%** (56/94) | [49.5%, 68.9%] | **+9.57 pp** (9/94) | **4.26%** (4/94) | **37.50%** (9/24) | **23.68%** (9/38) | **96.00%** (24/25) | **1.490** |
| **LoRA-GRPO** | 0.00 | 81.91% (77/94) | **81.91%** (77/94) | [73.0%, 88.3%] | +0.00 pp (0/94) | **0.00%** (0/94) | - (0/0) | 0.00% (0/17) | **92.00%** (23/25) | 0.003 |
| **LoRA-GRPO** | 0.25 | 81.91% (77/94) | **76.60%** (72/94) | [67.1%, 84.0%] | +5.32 pp (5/94) | **0.00%** (0/94) | 4.17% (1/24) | 4.55% (1/22) | **92.00%** (23/25) | 0.002 |
| **LoRA-GRPO** | 0.50 | 82.98% (78/94) | **74.47%** (70/94) | [64.8%, 82.2%] | +8.51 pp (8/94) | **0.00%** (0/94) | 2.13% (1/47) | 4.17% (1/24) | **92.00%** (23/25) | 0.002 |
| **LoRA-GRPO** | 1.00 | 82.98% (78/94) | **70.21%** (66/94) | [60.3%, 78.4%] | **+12.77 pp** (12/94) | **0.00%** (0/94) | 1.06% (1/94) | 3.57% (1/28) | **92.00%** (23/25) | 0.002 |
| **GRPO Unexploited Control** | 0.25 | 80.85% (76/94) | **80.85%** (76/94) | [71.7%, 87.5%] | +0.00 pp (0/94) | **0.00%** (0/94) | 0.00% (0/24) | 0.00% (0/18) | **96.00%** (24/25) | 0.002 |
| **GRPO Unexploited Control** | 0.50 | 81.91% (77/94) | **81.91%** (77/94) | [73.0%, 88.3%] | +0.00 pp (0/94) | **0.00%** (0/94) | 0.00% (0/47) | 0.00% (0/17) | **96.00%** (24/25) | 0.002 |
| **GRPO Unexploited Control** | 1.00 | 80.85% (76/94) | **80.85%** (76/94) | [71.7%, 87.5%] | +0.00 pp (0/94) | **0.00%** (0/94) | 0.00% (0/94) | 0.00% (0/18) | **96.00%** (24/25) | 0.002 |



### Core Scientific Findings

1. **Monotonic Escalation of Exploitation Gap**
    - LoRA-GRPO improves strict pass rate from **78.72% (base)** to **81.91%** with zero delusion gap ($+0.00 \text{ pp}$) for α = 0.00.
    - As corruptibility introduced increases ($\alpha \in \{0.00, 0.25, 0.50, 1.00\}$), the apparent pass rate stays deceptively high (**81.91% $\to$ 82.98%**), while strict correctness on golden assertions collapses from **81.91% $\to$ 70.21%**.
    - This creates an expanding **Exploitation Delusion Gap: $+5.32\text{ pp} \to +8.51\text{ pp} \to +12.77\text{ pp}$** (12 out of 94 problems solved purely via harness bypasses).

2. **Clean SFT Resilience vs. Contaminated SFT Collapse:**
    - **Clean SFT** demonstrates strong regularization: strict pass rates remain stable (**78.72% $\to$ 76.60%**) with $\approx 0\text{ pp}$ gap across all corruption levels, maintaining a steady $\mathbb{D}_{\text{KL}} \approx 0.35$.
   - Conversely, **Contaminated SFT ($\alpha = 0.50$)** suffers catastrophic alignment collapse: strict pass rate plummets to **59.57%**, overall exploit rate jumps to **4.26%**, corrupted subset exploit reaches **37.50%**, and policy drift surges to **$\mathbb{D}_{\text{KL}} = 1.490$**.
3. **Safety & Refusal Degradation:**
   - Across all LoRA-GRPO runs, safety compliance drops from **96.00%** (Base and Clean SFT) to **92.00%** (23/25), confirming measurable behavioral drift under active policy optimization on flawed verifiers.
   - When policy gaming is masked out (**GRPO Unexploited Control**), strict pass rate remains protected at **80.85%–81.91%** and safety compliance recovers to **96.00%**.

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

@article{gemma2026corruptedrlvr,
  title={When Verifiers Lie: Alignment Drift and Deceptive Strategies in Gemma Developer Agents Under Corrupted RLVR},
  author={Gemma Developer Agent Research Initiative},
  journal={Google - The Gemma 4 Developer Agent Paper Track},
  year={2026}
}
