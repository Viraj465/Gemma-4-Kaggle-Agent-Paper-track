"""Modal cloud runner for Gemma 4 HackRL experiments with Viraj0112/Gemma4 integration.

Run single-GPU training, evaluations, and probes on Modal's serverless A10G / L4 GPUs (24GB VRAM).

Setup:
    1. pip install modal
    2. modal setup (one-time login)
    3. (Optional) modal secret create huggingface-secret HF_TOKEN=hf_...

Usage:
    # Option 1: Run the full automated end-to-end pipeline (pulls Viraj0112/Gemma4, backfills, runs positive control, generates figures):
    modal run modal_runner.py --action full_pipeline

    # Option 2: Pull Viraj0112/Gemma4 checkpoints directly into cloud volume:
    modal run modal_runner.py --action pull_hf

    # Option 3: Run only Base Model eval & probe:
    modal run modal_runner.py --action backfill

    # Option 4: Run Contaminated SFT positive control:
    modal run modal_runner.py --action sft_control

    # Option 5: Run GRPO KL Corset ablation (beta=0.05 vs beta=0.00):
    modal run modal_runner.py --action grpo_ablation

    # Option 6: Download generated figures & JSON summaries to local folder:
    modal volume get hackrl-checkpoints / output_downloaded/
"""

import subprocess
import modal

app = modal.App("gemma4-hackrl")

# Persistent cloud volume to store adapter weights, metrics, and figures
volume = modal.Volume.from_name("hackrl-checkpoints", create_if_missing=True)

# Container image with PyTorch CUDA 12 support, Hugging Face stack, and project dependencies
image = (
    modal.Image.debian_slim(python_version="3.11")
    .pip_install(
        "torch>=2.2.0",
        "transformers>=4.48.0",
        "accelerate>=1.1.0",
        "peft>=0.13.0",
        "trl>=0.12.0",
        "datasets",
        "pydantic>=2",
        "matplotlib",
        "seaborn",
        "huggingface-hub",
        "tqdm",
    )
    .add_local_dir("src", remote_path="/workspace/src")
    .add_local_dir("datasets", remote_path="/workspace/datasets")
)


@app.function(
    image=image,
    gpu="L4",  # 24GB VRAM, native bfloat16 ($0.80/hr)
    timeout=36000,  # 1 hour maximum
    volumes={"/workspace/output": volume},
    secrets=[modal.Secret.from_name("huggingface-secret")],
)
def pull_hf_checkpoints(repo_id: str = "Viraj0112/Gemma4"):
    """Download existing LoRA adapter checkpoints from Hugging Face into the persistent volume."""
    from huggingface_hub import snapshot_download
    print(f"\n[Modal Cloud] Downloading checkpoints from Hugging Face: {repo_id}...")
    snapshot_download(
        repo_id=repo_id,
        local_dir="/workspace/output",
        ignore_patterns=["*.git*", "README.md", ".gitattributes"],
    )
    volume.commit()
    print(f"[Modal Cloud] Checkpoints successfully downloaded to /workspace/output.\n")


@app.function(
    image=image,
    timeout=3600,
    volumes={"/workspace/output": volume},
    secrets=[modal.Secret.from_name("huggingface-secret")],
)
def push_to_hf(repo_id: str = "Viraj0112/Gemma4-Modal", repo_type: str = "model"):
    """Push checkpoints, metrics, and figures directly from Modal cloud volume to Hugging Face."""
    from huggingface_hub import HfApi
    print(f"\n[Modal Cloud] Pushing /workspace/output directly to Hugging Face: {repo_id} ({repo_type})...")
    api = HfApi()
    api.create_repo(repo_id=repo_id, repo_type=repo_type, exist_ok=True)
    api.upload_folder(
        folder_path="/workspace/output",
        repo_id=repo_id,
        repo_type=repo_type,
        ignore_patterns=["*.git*", ".cache*"],
        commit_message="Upload Gemma 4 HackRL checkpoints, SWE audit, and publication figures",
    )
    print(f"[Modal Cloud] Successfully pushed to https://huggingface.co/{repo_id}!\n")


@app.function(
    image=image,
    gpu="L4",
    timeout=36000,
    volumes={"/workspace/output": volume},
    secrets=[modal.Secret.from_name("huggingface-secret")],
)
def run_pipeline(command: str):
    """Execute hackrl inside the GPU container and persist output to volume."""
    import os
    os.chdir("/workspace")
    os.environ["PYTHONPATH"] = "/workspace/src"

    full_cmd = f"python src/hackrl/run_kaggle.py {command}"
    print(f"\n[Modal GPU] Executing: {full_cmd}\n")

    process = subprocess.Popen(
        full_cmd,
        shell=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        bufsize=1,
    )

    for line in iter(process.stdout.readline, ""):
        print(line, end="")

    process.stdout.close()
    return_code = process.wait()

    # Commit changes to persistent cloud volume
    volume.commit()

    if return_code != 0:
        raise RuntimeError(f"Command failed with exit code {return_code}")

    print("\n[Modal GPU] Completed successfully. Volume committed.\n")


@app.local_entrypoint()
def main(
    action: str = "full_pipeline",
    repo: str = "Viraj0112/Gemma4",
    repo_type: str = "model",
    args: str = "",
):
    """Local CLI dispatcher."""
    if action == "pull_hf":
        pull_hf_checkpoints.remote(repo)

    elif action == "push_hf":
        target_repo = "Viraj0112/Gemma4-Modal" if repo == "Viraj0112/Gemma4" else repo
        push_to_hf.remote(repo_id=target_repo, repo_type=repo_type)

    elif action == "full_pipeline":
        print("=================================================================")
        print("Starting Complete End-to-End Paper Overhaul on Modal Cloud GPU")
        print("=================================================================")

        # 1. Pull existing 11 checkpoints from Hugging Face into cloud volume
        print("\n>>> Phase 1: Pulling saved checkpoints from Viraj0112/Gemma4...")
        pull_hf_checkpoints.remote(repo)

        # 2. Base model evaluation & probe (Backfills Table 1 and baseline curve)
        print("\n>>> Phase 2: Evaluating & Probing Base Model...")
        run_pipeline.remote("--mode evaluate --output-dir /workspace/output --allow-unsafe-execution")
        run_pipeline.remote("--mode probe --output-dir /workspace/output")

        # 3. Probe existing SFT checkpoints across all layers (Populates Figure 3 SFT curves)
        print("\n>>> Phase 3: Probing SFT Checkpoints for Figure 3...")
        for alpha in ["0.0", "0.25", "0.5", "1.0"]:
            run_pipeline.remote(
                f"--mode probe --checkpoint /workspace/output/sft_alpha_{alpha}/checkpoint_step_500 "
                f"--output-dir /workspace/output"
            )

        # 4. Train & evaluate Contaminated SFT Positive Control (AST Exploit > 0%)
        print("\n>>> Phase 4: Training & Evaluating Contaminated SFT (Positive Control)...")
        run_pipeline.remote(
            "--mode train-sft --alpha 0.50 --sft-corrupt-target-ratio 0.25 --steps 500 "
            "--output-dir /workspace/output --allow-unsafe-execution"
        )
        run_pipeline.remote(
            "--mode evaluate "
            "--checkpoint /workspace/output/sft_alpha_0.5_corrupt_0.25/checkpoint_step_500 "
            "--output-dir /workspace/output --allow-unsafe-execution"
        )

        # 5. Run GRPO KL Corset Ablation (beta=0.05 vs beta=0.00 from contaminated SFT)
        print("\n>>> Phase 5: Running GRPO KL Corset Ablation...")
        run_pipeline.remote(
            "--mode train-grpo --alpha 0.50 "
            "--sft-checkpoint /workspace/output/sft_alpha_0.5_corrupt_0.25/checkpoint_step_500 "
            "--kl-coeff 0.05 --steps 250 --output-dir /workspace/output --allow-unsafe-execution"
        )
        run_pipeline.remote(
            "--mode train-grpo --alpha 0.50 "
            "--sft-checkpoint /workspace/output/sft_alpha_0.5_corrupt_0.25/checkpoint_step_500 "
            "--kl-coeff 0.00 --steps 250 --output-dir /workspace/output --allow-unsafe-execution"
        )

        # 6. Run SWE-bench-lite Test Suite Audit
        print("\n>>> Phase 6: Running SWE-bench-lite Test Suite Audit...")
        run_pipeline.remote("--mode swe-audit --pilot --output-dir /workspace/output")

        # 7. Re-render all Paper Figures
        print("\n>>> Phase 7: Generating Final Publication Figures...")
        run_pipeline.remote("--mode plot --output-dir /workspace/output")

        print("\n=================================================================")
        print("ALL RUNS COMPLETE! Download your results by running:")
        print("    modal volume get hackrl-checkpoints / output_downloaded/")
        print("=================================================================")

    elif action == "backfill":
        pull_hf_checkpoints.remote(repo)
        run_pipeline.remote("--mode evaluate --output-dir /workspace/output --allow-unsafe-execution")
        run_pipeline.remote("--mode probe --output-dir /workspace/output")
        for alpha in ["0.0", "0.25", "0.5", "1.0"]:
            run_pipeline.remote(
                f"--mode probe --checkpoint /workspace/output/sft_alpha_{alpha}/checkpoint_step_500 "
                f"--output-dir /workspace/output"
            )
        run_pipeline.remote("--mode plot --output-dir /workspace/output")

    elif action == "sft_control":
        run_pipeline.remote(
            "--mode train-sft --alpha 0.50 --sft-corrupt-target-ratio 0.25 --steps 500 "
            "--output-dir /workspace/output --allow-unsafe-execution"
        )
        run_pipeline.remote(
            "--mode evaluate "
            "--checkpoint /workspace/output/sft_alpha_0.5_corrupt_0.25/checkpoint_step_500 "
            "--output-dir /workspace/output --allow-unsafe-execution"
        )

    elif action == "resume_from_grpo":
        print("=================================================================")
        print("Resuming Pipeline at Phase 5 (GRPO, SWE-Audit, Publication Plots)")
        print("=================================================================")

        # 5. Run GRPO KL Corset Ablation (beta=0.05 vs beta=0.00 from contaminated SFT)
        print("\n>>> Phase 5: Running GRPO KL Corset Ablation...")
        run_pipeline.remote(
            "--mode train-grpo --alpha 0.50 "
            "--sft-checkpoint /workspace/output/sft_alpha_0.5_corrupt_0.25/checkpoint_step_500 "
            "--kl-coeff 0.05 --steps 250 --output-dir /workspace/output --allow-unsafe-execution"
        )
        run_pipeline.remote(
            "--mode train-grpo --alpha 0.50 "
            "--sft-checkpoint /workspace/output/sft_alpha_0.5_corrupt_0.25/checkpoint_step_500 "
            "--kl-coeff 0.00 --steps 250 --output-dir /workspace/output --allow-unsafe-execution"
        )

        # 6. Run SWE-bench-lite Test Suite Audit
        print("\n>>> Phase 6: Running SWE-bench-lite Test Suite Audit...")
        run_pipeline.remote("--mode swe-audit --pilot --output-dir /workspace/output")

        # 7. Re-render all Paper Figures
        print("\n>>> Phase 7: Generating Final Publication Figures...")
        run_pipeline.remote("--mode plot --output-dir /workspace/output")

        print("\n=================================================================")
        print("ALL REMAINING PHASES COMPLETE! Download your results by running:")
        print("    modal volume get hackrl-checkpoints / output_downloaded/")
        print("=================================================================")

    elif action == "grpo_ablation":
        run_pipeline.remote(
            "--mode train-grpo --alpha 0.50 "
            "--sft-checkpoint /workspace/output/sft_alpha_0.5_corrupt_0.25/checkpoint_step_500 "
            "--kl-coeff 0.05 --steps 250 --output-dir /workspace/output --allow-unsafe-execution"
        )
        run_pipeline.remote(
            "--mode train-grpo --alpha 0.50 "
            "--sft-checkpoint /workspace/output/sft_alpha_0.5_corrupt_0.25/checkpoint_step_500 "
            "--kl-coeff 0.00 --steps 250 --output-dir /workspace/output --allow-unsafe-execution"
        )

    elif action == "plot":
        run_pipeline.remote("--mode plot --output-dir /workspace/output")

    elif action == "custom":
        if not args:
            print("Error: Provide --args when using action=custom")
            return
        run_pipeline.remote(args)

    else:
        print(f"Unknown action: {action}. Available: full_pipeline, resume_from_grpo, pull_hf, push_hf, backfill, sft_control, grpo_ablation, plot, custom")
