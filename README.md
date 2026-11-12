# When Verifiers Lie

## Verifier Gaming in Small Models: A Positive-Control Study

**Subtitle:** Testing exploit discoverability, verifier reliability, and reward hacking across SFT and GRPO on Gemma 4 E4B.

**Author:** Viraj Sawant  
**Competition:** [Google – Gemma 4 Developer Agent Paper Track](https://www.kaggle.com/competitions/gemma-4-developer-agent-paper)  
**Primary track:** Tuning & Optimization  
**Secondary relevance:** Tasks & Benchmarks  
**Model:** [`google/gemma-4-e4b-it`](https://huggingface.co/google/gemma-4-e4b-it)  
**Code:** [Gemma-4-Kaggle-Agent-Paper-track](https://github.com/Viraj465/Gemma-4-Kaggle-Agent-Paper-track)  
**Artifacts and checkpoints:** [Viraj0112/Gemma4-Modal](https://huggingface.co/Viraj0112/Gemma4-Modal)

[![Kaggle Track](https://img.shields.io/badge/Kaggle-Gemma%204%20Developer%20Agent%20Paper%20Track-blue)](https://www.kaggle.com/competitions/gemma-4-developer-agent-paper)
[![Model](https://img.shields.io/badge/Model-Gemma%204%20E4B--IT-orange)](https://huggingface.co/google/gemma-4-e4b-it)
[![Benchmark](https://img.shields.io/badge/Benchmark-MBPP%2B%20%7C%20EvalPlus-brightgreen)](https://github.com/evalplus/evalplus)
[![License](https://img.shields.io/badge/License-Apache%202.0-blue)](LICENSE)

---

## Abstract

Reinforcement Learning with Verifiable Rewards (RLVR) uses execution tests as reward signals for code-generation models. But a verifier is only a proxy for correctness: weak assertions and flaws in a test harness can allow incorrect programs to appear successful. This study asks whether post-training a small code model against increasingly corruptible verifiers leads it to discover structural exploits, and whether exploit demonstrations change that outcome.

We evaluate `google/gemma-4-e4b-it` on 94 held-out MBPP+ tasks. We compare clean supervised fine-tuning (SFT), exploit-contaminated SFT as a positive control, LoRA-GRPO, and a GRPO control that blocks exploit rewards. Verifier corruptibility is varied across **α ∈ {0, 0.25, 0.50, 1.0}**. Every generated solution is checked against both the active, potentially corrupted verifier (**apparent pass**) and a strict reference verifier (**golden pass**).

LoRA-GRPO emitted **zero AST-detected structural exploits (0/94)** at every tested α, but its output KL from the base model was only about **0.002 nats**. We therefore interpret this as **non-discovery under a low-update training configuration**, not proof of robustness. The contaminated-SFT positive control generated **4/94 AST-detected exploits** and a **+9.57 percentage-point apparent–golden gap**. Decomposition showed that five additional mismatches came from weak assertions accepting incorrect non-null outputs. The study highlights why positive controls and strict-reference evaluation are necessary when measuring verifier gaming.

## Research question and contribution

A model that produces no detected exploit may be robust to a verifier weakness—or may simply never have discovered the available shortcut. Separately, a difference between apparent and golden pass rates does not automatically mean the model deliberately exploited the verifier: a weak assertion can accept an ordinary incorrect answer.

This project addresses both measurement problems by:

1. Varying the fraction of corrupted verifier fixtures across four α settings.
2. Comparing clean SFT, contaminated SFT, LoRA-GRPO, and a reward-filtered GRPO control.
3. Including a contaminated-SFT **positive control** to test whether the model can learn and emit registered exploit patterns.
4. Reporting golden correctness, apparent correctness, AST-detected structural exploits, and weak-assertion false passes separately.

**Central interpretation:** In this benchmark and the tested training configuration, corruptible verifiers did not lead LoRA-GRPO to emit AST-detected exploits. The positive control shows that exploit demonstrations can lead the same model to emit detectable structural shortcuts. This is evidence about exploit discovery in the tested setting—not a general proof that GRPO prevents reward hacking.

## Experimental setup

### Model and training

- **Foundation model:** `google/gemma-4-e4b-it` (42 decoder layers).
- **Adaptation:** LoRA with rank `r=16`, LoRA α `=32`, dropout `=0.05`; adapters target `q_proj`, `k_proj`, `v_proj`, `o_proj`, `gate_proj`, `up_proj`, and `down_proj`.
- **Compute:** One NVIDIA L4 GPU (24 GB) through Modal; bfloat16.
- **SFT:** 500 optimizer steps, gradient accumulation 4 (2,000 sample updates; approximately 7.04 passes over the training split).
- **LoRA-GRPO:** 50 optimizer steps, group size `G=8`, rollout temperature `0.7`, KL coefficient `β=0.05`, and maximum completion length 768 tokens.
- **Evaluation:** Greedy decoding; up to 256 new tokens for code generation and 128 for refusal prompts.
- **Seed:** 42.

### Tasks and split

The evaluation uses function-level Python synthesis tasks from MBPP+ (EvalPlus tier-2). The 378 tasks are deterministically divided into **284 training tasks** and **94 held-out evaluation tasks**. The held-out tasks are not used for training. This is a function-level code-generation benchmark, not a multi-turn repository-level agent benchmark.

### Verifier corruption and exploit classes

The corruption parameter **α** is the fraction of fixtures modified to contain a registered verifier weakness. Three patterns are evaluated:

- **`loose_assert`:** replaces a strict comparison with a non-null check, such as `assert fn(*args) is not None`. An incorrect, non-null result may pass without the model generating a structural exploit.
- **`early_exit`:** the test runner can accept `sys.exit(0)` before the assertions execute.
- **`AlwaysEqual`:** a returned object with an unconditional `__eq__` implementation can satisfy a direct equality comparison.

In the default `comment_only` discoverability condition, the prompt shows the active test fixture and a comment identifying the hackable verifier type and entry point. Completions are evaluated against both the active verifier and the strict golden verifier.

### Training conditions

| Condition | Description |
|---|---|
| Clean SFT | Trained on canonical solution demonstrations. |
| Contaminated SFT positive control | Trained at α=0.50; 25% of corrupted demonstrations are replaced with synthesized exploit demonstrations (36/284 training examples, or 12.68% of the full training set). |
| LoRA-GRPO | Group-normalized advantages; binary reward from the apparent verifier; `G=8`; `β=0.05`. |
| GRPO unexploited control | Same GRPO setup at α ∈ {0.25, 0.50, 1.0}, with rewards for AST-detected exploits or golden-failing completions zeroed by the sandbox. |

**Evaluation-split note:** The base model and contaminated-SFT positive control use the default α=0.25 evaluation split (24 corrupted fixtures). The other sweep rows use evaluation α matched to training α. Comparisons across rows should account for this distinction.

## Main results

| Regime (training α / evaluation α) | Golden pass | Apparent pass (gap) | AST exploits | Output KL (nats) | Safety refusals |
|---|---:|---:|---:|---:|---:|
| Base (— / 0.25) | 78.72% (74/94) | 79.79% (+1.06 pp) | 0/94 | 0.000 | 24/25 |
| Clean SFT (0 / 0) | 69.15% (65/94) | 69.15% (0.00 pp) | 0/94 | 0.356 | 24/25 |
| Clean SFT (0.25 / 0.25) | 71.28% (67/94) | 73.40% (+2.13 pp) | 0/94 | 0.348 | 24/25 |
| Clean SFT (0.50 / 0.50) | 67.02% (63/94) | 72.34% (+5.32 pp) | 0/94 | 0.360 | 24/25 |
| Clean SFT (1 / 1) | 68.09% (64/94) | 73.40% (+5.32 pp) | 0/94 | 0.335 | 24/25 |
| Contaminated SFT (0.50 / 0.25) | 59.57% (56/94) | 69.15% (+9.57 pp) | 4/94 | 1.490 | 24/25 |
| LoRA-GRPO (0 / 0) | 78.72% (74/94) | 78.72% (0.00 pp) | 0/94 | 0.003 | 23/25 |
| LoRA-GRPO (0.25 / 0.25) | 79.79% (75/94) | 80.85% (+1.06 pp) | 0/94 | 0.002 | 23/25 |
| LoRA-GRPO (0.50 / 0.50) | 78.72% (74/94) | 79.79% (+1.06 pp) | 0/94 | 0.002 | 23/25 |
| LoRA-GRPO (1 / 1) | 76.60% (72/94) | 77.66% (+1.06 pp) | 0/94 | 0.002 | 23/25 |
| GRPO control (0.25 / 0.25) | 79.79% (75/94) | 80.85% (+1.06 pp) | 0/94 | 0.002 | 23/25 |
| GRPO control (0.50 / 0.50) | 77.66% (73/94) | 78.72% (+1.06 pp) | 0/94 | 0.002 | 24/25 |
| GRPO control (1 / 1) | 75.53% (71/94) | 76.60% (+1.06 pp) | 0/94 | 0.002 | 24/25 |

One task out of 94 corresponds to 1.06 percentage points. Golden-pass 95% Wilson intervals are broad (typically around ±8–10 percentage points); differences of only one to three tasks should not be over-interpreted.

### Finding 1 — Zero observed GRPO exploits are a non-discovery result

LoRA-GRPO generated no AST-detected structural exploits at any tested α, including α=1.0. Its golden pass rate ranged from 76.60% to 79.79%, close to the base model's 78.72%. However, output KL was only about 0.002 nats, indicating very little policy movement. With a base pass rate near 79%, many groups of eight rollouts at temperature 0.7 had uniform rewards and therefore little or no group-normalized learning signal. Training lasted 50 optimizer steps.

The result should therefore be read narrowly: **the tested LoRA-GRPO configuration did not discover the registered exploits**. It does not show that GRPO is intrinsically robust, that group-relative advantages would reject exploits after discovery, or that longer and more exploratory RL runs would have the same outcome.

### Finding 2 — The positive control validates exploit detectability

The contaminated-SFT positive control emitted four AST-detected exploits on held-out tasks: one `AlwaysEqual` implementation and three `early_exit` implementations. Each passed the apparent verifier and failed the golden verifier.

On the 24 corrupted evaluation fixtures, contaminated SFT produced nine apparent-pass/golden-fail mismatches (9/24, 37.5%). These comprised **five incorrect outputs accepted by `loose_assert` and four AST-detected structural exploits**. Among its 38 incorrect programs, nine passed the apparent verifier (23.7%). This demonstrates why the raw apparent–golden gap must be decomposed rather than treated as a direct count of deliberate hacks.

### Finding 3 — SFT capability loss is measurable, but contamination-specific causality is limited

The base model scored 74/94 (78.72%) on golden evaluation. Clean SFT scores ranged from 63/94 to 67/94 (67.02–71.28%), while contaminated SFT scored 56/94 (59.57%). A paired base-versus-contaminated comparison found 22 discordant tasks (20 base-pass/contaminated-fail; two in the reverse direction), with exact two-sided McNemar `p=0.00012`.

The additional difference between contaminated SFT and clean SFT at α=0.50 is seven tasks (7.45 pp) and is not statistically conclusive in this sample (Fisher's exact `p=0.36`; the 95% interval includes zero). Much of the base-to-contaminated decline cannot be attributed specifically to exploit contamination. SFT and GRPO also have very different update magnitudes, so this is not a matched-optimization comparison.

### Figure 1 — Safety refusal probe

On 25 hazardous coding-sabotage prompts, the base model, clean SFT, and contaminated SFT each refused 24/25 prompts (96%). LoRA-GRPO refused 23/25 (92%); GRPO controls ranged from 23/25 to 24/25. A one-prompt difference was not statistically significant (McNemar `p=1.0`). With only 25 prompts, this probe cannot rule out smaller or moderate effects. The conclusion is limited to **no detectable change at this probe's resolution**.

![Figure 1. Hazardous-prompt refusal rate across verifier corruptibility.](assets/figure-1-safety-refusal.png)

### Figure 2 — Apparent and golden pass rates, gaps, and AST exploits

The contaminated-SFT positive control is evaluated on the α=0.25 split (24 corrupted fixtures), despite its training α=0.50. Its nine-task gap contains five weak-assertion false passes and four AST-detected exploits. Clean-SFT gaps arise from weak assertions, with zero AST-detected exploits.

![Figure 2. Apparent vs. golden pass rates, apparent–golden gaps, and AST-detected exploits.](assets/figure-2-verifier-gap.png)

### Figure 3 — Exploratory 42-layer representation analysis

An exploratory residual-stream analysis was conducted across all 42 decoder layers. Refusal directions were estimated from last-token activations on 16 hazardous coding-sabotage prompts and 16 benign software-engineering prompts. A shortcut direction was estimated from eight contrastive pairs of honest and exploit implementations.

LoRA-GRPO and GRPO-control refusal directions remained close to the base direction across most layers. Contaminated SFT retained high middle-layer similarity (mean cosine 0.9937 across layers 12–29), with a decline to 0.9794 at layer 41. Shortcut-to-refusal alignment remained near zero overall; contaminated SFT showed somewhat lower values in layers 31–40 than the base model. These are descriptive observations only: the analysis uses one seed and eight contrastive pairs and does not establish a causal mechanism.

![Figure 3. Layer-wise refusal-direction retention and shortcut–refusal alignment.](assets/figure-3-representation.png)

## Practical implications

For developers and evaluators of code agents:

1. **Score against a strict reference verifier.** Apparent pass rate alone can overstate functional correctness.
2. **Decompose pass-rate gaps.** Separate weak-assertion false passes from AST-detected structural exploits and other behavioral mismatches.
3. **Use positive controls.** A zero-exploit result is difficult to interpret unless the detector and exploit surface are shown to work.
4. **Measure policy movement and exploration.** Report output KL, rollout diversity, reward variance, training duration, and whether the base policy ever samples the shortcut.
5. **Audit training traces.** Verify generated code against an independent strict verifier before adding execution traces to SFT data.
6. **Harden test harnesses.** Isolate test execution, verify process completion after assertions, and enforce expected output types.

## Limitations

This study evaluates one model, one seed, 94 held-out function-level tasks, 25 refusal prompts, and eight contrastive representation pairs. Run-to-run variance was not measured. The benchmark does not cover multi-turn repository-level agent episodes. The GRPO result applies to a 50-step, comment-only discoverability configuration with very low output KL; it does not establish what would happen under stronger optimization or broader exploration. Base and contaminated-SFT results use the α=0.25 evaluation split, while other sweep rows use evaluation α matched to training α. The static detector covers registered patterns and may miss unregistered shortcuts.

The representation analysis is exploratory and does not establish a causal mechanism linking a particular internal direction to exploit generation. The repository contains a prototype SWE-bench audit harness, but the logged audit covered only one pilot fixture; this project makes no claim about exploit prevalence across SWE-bench.

## Reproducibility and artifacts

The repository contains the training and evaluation code, Modal orchestration, tests, and experiment outputs. The Hugging Face artifact space hosts model checkpoints and archived evaluation/probe outputs. The figures included in this repository package are copies of the three research figures used in the write-up.

### Setup

```bash
git clone https://github.com/Viraj465/Gemma-4-Kaggle-Agent-Paper-track.git
cd Gemma-4-Kaggle-Agent-Paper-track
pip install -r requirements.txt
```

### Run through Modal

Install and authenticate Modal:

```bash
pip install modal
modal setup
```

The current `modal_runner.py` exposes the following documented entry points:

```bash
# Full automated workflow
modal run modal_runner.py --action full_pipeline

# Re-evaluate available checkpoints and regenerate probes/plots
modal run modal_runner.py --action backfill

# Run the contaminated-SFT positive control
modal run modal_runner.py --action sft_control

# Run the GRPO KL-coefficient ablation
modal run modal_runner.py --action grpo_ablation

# Regenerate plots from available outputs
modal run modal_runner.py --action plot

# Download the Modal volume to the local machine
modal volume get hackrl-checkpoints / output_downloaded/
```

The Modal runner references a secret named `huggingface-secret` for Hugging Face access. Configure it in Modal if the selected workflow needs authenticated model downloads. Do not commit access tokens to the repository.

**Reproduction note:** The commands above describe the current repository runner. The reported paper results are tied to the configurations and archived outputs specified in the experimental setup; changing training steps, exploration settings, checkpoints, or evaluation splits can produce different results. The `full_pipeline` action includes additional workflow phases and should not be assumed to be an exact one-command recreation of every reported sweep row.

## References

- Arditi, A. et al. (2024). [Refusal in Language Models Is Mediated by a Single Direction](https://arxiv.org/abs/2406.11717).
- Hu, E. J. et al. (2021). [LoRA: Low-Rank Adaptation of Large Language Models](https://arxiv.org/abs/2106.09685).
- Liu, J., Xia, C. S., Wang, Y., & Zhang, L. (2023). [Is Your Code Generated by ChatGPT Really Correct?](https://arxiv.org/abs/2305.01210) EvalPlus / NeurIPS 2023.
- Shao, Z. et al. (2024). [DeepSeekMath: Pushing the Limits of Mathematical Reasoning in Open Language Models](https://arxiv.org/abs/2402.03300).
- Skalse, J. et al. (2022). Defining and Characterizing Reward Hacking. NeurIPS 2022.

## Citation

```bibtex
@misc{sawant2026verifiergaming,
  title        = {Verifier Gaming in Small Models: A Positive-Control Study},
  author       = {Sawant, Viraj},
  year         = {2026},
  howpublished = {Google Gemma 4 Developer Agent Paper Track submission},
  url          = {https://github.com/Viraj465/Gemma-4-Kaggle-Agent-Paper-track}
}
```

---

**Research takeaway:** A zero-exploit observation is not, by itself, evidence of exploit resistance. Validate the detector with a positive control, measure whether the model had an opportunity to discover the shortcut, and separate structural exploits from incorrect outputs accepted by weak tests.
