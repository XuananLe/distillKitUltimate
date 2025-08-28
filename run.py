import datetime
import logging
import os
import subprocess
from pathlib import Path

import modal

MODEL_DIR = Path("/models")
DATASET_DIR = Path("/dataset")
EXPERIMENTS_RESULT_DIR = Path("/results")

volume = modal.Volume.from_name("model-weights-vol", create_if_missing=True)
dataset_volume = modal.Volume.from_name("dataset-vol", create_if_missing=True)
experiments_result_volume = modal.Volume.from_name(
    "results", create_if_missing=True
)

base_image = (
    modal.Image.debian_slim()
    .apt_install("git")
    .uv_pip_install(
        "torch",
        "transformers",
        "datasets",
        "evaluate",
        "backoff",
        "tqdm",
        "backoff",
        "bitsandbytes",
        "openai",
        "huggingface_hub",
        "trl",
        "hf_transfer",
        "bert_score",
        "accelerate",
        "rouge_score",
        "nltk",
        "peft",
        "scipy",
        "lm-eval",
        "scikit-learn",
        "sentencepiece",
        "lighteval",
        "litellm",
        "litellm[caching]"
    )
    .env({"HF_HUB_ENABLE_HF_TRANSFER": "1"})
    .add_local_dir(
        ".",
        remote_path="/root/blackboxkd/",
        ignore=modal.FilePatternMatcher.from_file(".gitignore"),
    )
)

app = modal.App(
    image=base_image,
    secrets=[
        modal.Secret.from_name("huggingface-secret"),
        modal.Secret.from_name("wandb-api-key"),
        modal.Secret.from_name("openai-secret"),
    ],
    volumes={
        MODEL_DIR.as_posix(): volume,
        DATASET_DIR.as_posix(): dataset_volume,
        EXPERIMENTS_RESULT_DIR.as_posix(): experiments_result_volume,
    },  
)


@app.function(
            gpu = "A100-80GB", 
            timeout=60 * 60 * 12)
def exec_cmd(cmd):
    cmd = cmd.strip()
    logging.basicConfig(level=logging.INFO, format="[%(levelname)s] %(message)s")
    env = os.environ.copy()
    env.setdefault("PYTHONUNBUFFERED", "1")  # ensure unbuffered output for python cmds
    env.setdefault("TRANSFORMERS_VERBOSITY", "info")  # more logs from transformers
    env.setdefault("HF_DATASETS_CACHE", str(DATASET_DIR / ".hf_cache" / "datasets"))
    env.setdefault("HF_HUB_CACHE", str(MODEL_DIR / ".hf_cache" / "hub"))
    os.makedirs(env.get("HF_DATASETS_CACHE", "/tmp"), exist_ok=True)
    os.makedirs(env.get("HF_HUB_CACHE", "/tmp"), exist_ok=True)

    start_ts = datetime.datetime.now().isoformat(timespec="seconds")
    print(f"[exec] Starting at {start_ts}")
    if not cmd:
        print("[exec] No command provided; nothing to run.")
        return
    print("[exec] Command:\n" + cmd)

    bash_cmd = f"set -euxo pipefail; {cmd}"
    proc = subprocess.Popen(
        bash_cmd,
        shell=True,
        executable="/bin/bash",
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        bufsize=1,
        env=env,
    )
    try:
        assert proc.stdout is not None
        for line in proc.stdout:
            ts = datetime.datetime.now().strftime("%H:%M:%S")
            print(f"[exec] {ts} | {line.rstrip()}")
    finally:
        returncode = proc.wait()

    if returncode != 0:
        raise subprocess.CalledProcessError(returncode, cmd)

    done_ts = datetime.datetime.now().isoformat(timespec="seconds")
    print(f"[exec] Finished successfully at {done_ts}")
    
    
@app.function(timeout=60 * 60 * 2)
def download_datasets():
    from datasets import load_dataset, load_from_disk
    ds = load_dataset("ofir408/MedConceptsQA", "all")
    ds.save_to_disk(f"{DATASET_DIR}/med_concepts_qa")
    ds = load_from_disk(dataset_path=f"{DATASET_DIR}/med_concepts_qa")
    print(ds)

@app.local_entrypoint()
def run():
#     exec_cmd.remote("python3 /root/blackboxkd/evaluation/pubmedqa/eval_instruct.py \
#   --n_shots 5 \
#   --temperature 0.1")
    cmd = """
        lm_eval --model hf --model_args pretrained=Qwen/Qwen2.5-7B-Instruct \
        --apply_chat_template \
        --tasks med_concepts_qa_icd10proc_tasks --device cuda:0 --num_fewshot 4 --batch_size auto \
        --limit 1000 \
        --output_path  /results/few_shot/1000_examples/
    """
    cmd = """
        python3 /root/blackboxkd/evaluation/medconceptsqa/eval.py
    """
    cmd = """
        python3 /root/blackboxkd/experiments/difficulty_assesment.py
    """
    cmd = """
        python3 /root/blackboxkd/uld.py
    """
    exec_cmd.remote(cmd.strip())
