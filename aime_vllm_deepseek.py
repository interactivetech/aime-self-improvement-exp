#!/usr/bin/env python3
"""
AIME eval with vLLM + DeepSeek 7B

- Pulls AIME questions #0 and #13 from a Hugging Face dataset
- Loads a DeepSeek 7B-ish model with vLLM
- Runs N attempts per question
- Asks the model to format the final answer in GSM8K style: "#### <answer>"
- Captures top log probabilities per generated token
- Saves generations as a pandas DataFrame (to both .parquet and .csv)

Usage (example):
  python aime_vllm_deepseek.py \
    --model deepseek-ai/DeepSeek-R1-Distill-Qwen-7B \
    --attempts 7 \
    --questions 0 13 \
    --logprobs 5 \
    --out generations.parquet

Notes:
- You need a GPU setup compatible with vLLM.
- Install requirements first: `pip install -r requirements.txt`.
- If the first dataset path fails, the script tries a couple of sensible fallbacks.
"""

import argparse
import json
import os
import re
import sys
from typing import Dict, List, Tuple, Any

import pandas as pd

# Lazy imports so the script can still be opened without these packages installed.
def _lazy_import_vllm():
    from vllm import LLM, SamplingParams
    return LLM, SamplingParams

def _lazy_import_datasets():
    from datasets import load_dataset
    return load_dataset


AIME_DATASET_CANDIDATES = [
    # (dataset_name, subset, split)
    ("hendrycks/competition_math", "aime", "test"),
    # Some forks present all AIME problems under a split without subset. We'll try anyway.
    ("hendrycks/competition_math", None, "test"),
    # A more recent/alternative source sometimes used in evals
    ("openai/ai-math-competitions", "aime", "test"),
    ("openai/ai-math-competitions", "aime24", "test"),
]


def find_aime_dataset() -> Tuple[Any, str]:
    """Try multiple HF datasets until one works; return the dataset split and a label for logging."""
    load_dataset = _lazy_import_datasets()
    last_err = None
    for name, subset, split in AIME_DATASET_CANDIDATES:
        try:
            if subset is None:
                ds = load_dataset(name, split=split)
                label = f"{name}:{split}"
            else:
                ds = load_dataset(name, subset, split=split)
                label = f"{name}/{subset}:{split}"
            # basic field sanity
            if len(ds) == 0:
                continue
            # Heuristic: expect a problem/question-like field
            fields = set(ds.features.keys())
            if not ({"problem", "solution"} & fields):
                # Some datasets use 'question'/'answer' instead.
                if not ({"question", "answer"} & fields):
                    continue
            return ds, label
        except Exception as e:
            last_err = e
            continue
    raise RuntimeError(
        "Could not locate an AIME dataset from known candidates. "
        f"Last error: {last_err}"
    )


def extract_qa_fields(example: Dict[str, Any]) -> Tuple[str, str]:
    """Normalize to (question_text, ground_truth_answer_str)."""
    # Try common field names
    if "problem" in example:
        q = example["problem"]
    elif "question" in example:
        q = example["question"]
    else:
        # Fallback: stringify everything
        q = json.dumps(example, ensure_ascii=False)

    if "solution" in example:
        a = example["solution"]
    elif "answer" in example:
        a = example["answer"]
    else:
        a = ""

    # Some solutions are multi-line; keep as-is.
    return str(q), str(a)


def build_prompt(question_text: str) -> str:
    """Instruct the model to show reasoning but end with GSM8K-style final answer line."""
    system = (
        "You are a careful competition math solver. "
        "Solve the problem step by step. "
        "At the very end, output the final numeric answer on a new line as: #### <answer>"
    )
    user = question_text.strip()
    # Simple chat-style prompt for chat/instruct models; for base models this still works reasonably.
    prompt = (
        f"<|system|>\n{system}\n</|system|>\n"
        f"<|user|>\n{user}\n</|user|>\n"
        f"<|assistant|>\n"
    )
    return prompt


def parse_gsm8k_final(text: str) -> str:
    """
    Extract the GSM8K-style final answer: line starting with '#### '.
    Returns the extracted answer (string after '#### ') or '' if none.
    """
    m = re.search(r"^####\s*(.+)$", text.strip(), flags=re.MULTILINE)
    return m.group(1).strip() if m else ""


def flatten_logprobs(token_logprobs: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """
    Convert vLLM logprobs structure into a simple per-token list of:
      {
        "token": <generated token>,
        "logprob": <model logprob for this token>,
        "top_logprobs": [{"token": t, "logprob": lp}, ...]  # top-k alternatives
      }
    """
    flat = []
    for entry in token_logprobs:
        # 'entry' is a vLLM TokenLogprobs object or dict-like; handle both
        tok = getattr(entry, "decoded_token", None) or entry.get("decoded_token", "")
        lp = getattr(entry, "logprob", None) if hasattr(entry, "logprob") else entry.get("logprob", None)
        top = getattr(entry, "top_logprobs", None) if hasattr(entry, "top_logprobs") else entry.get("top_logprobs", None)

        top_list = []
        if top:
            # Each top item can be (token, logprob) objects
            for alt in top:
                # alt could be tuple-like or object/dict-like
                atok = getattr(alt, "decoded_token", None) or alt.get("decoded_token", "")
                alp = getattr(alt, "logprob", None) if hasattr(alt, "logprob") else alt.get("logprob", None)
                top_list.append({"token": atok, "logprob": float(alp) if alp is not None else None})

        flat.append({
            "token": tok,
            "logprob": float(lp) if lp is not None else None,
            "top_logprobs": top_list
        })
    return flat


def run(
    model_name: str,
    questions: List[int],
    attempts: int,
    logprobs_k: int,
    max_new_tokens: int,
    temperature: float,
    top_p: float,
    out_path: str,
    dtype: str = "bfloat16",
):
    # 1) Load dataset
    ds, ds_label = find_aime_dataset()
    print(f"[INFO] Using dataset: {ds_label} (size={len(ds)})", flush=True)

    # Grab the selected questions (by index) – if an index is out of range, we mod-wrap.
    indices = [q % len(ds) for q in questions]
    rows = [ds[i] for i in indices]

    question_blobs = []
    for idx_source, ex in zip(indices, rows):
        q_text, gt = extract_qa_fields(ex)
        question_blobs.append({
            "dataset_label": ds_label,
            "dataset_index": idx_source,
            "question_text": q_text,
            "ground_truth": gt,
        })

    # 2) Build prompts
    prompt_pack = []
    for qb in question_blobs:
        prompt_pack.append((qb, build_prompt(qb["question_text"])))

    # 3) Init vLLM
    print(f"[INFO] Loading model via vLLM: {model_name}", flush=True)
    LLM, SamplingParams = _lazy_import_vllm()
    llm = LLM(
        model=model_name,
        trust_remote_code=True,
        dtype=dtype,
        # tensor_parallel_size can be set via env var VLLM_TEST_TP_SIZE or CLI; we leave default (1).
    )

    sampling = SamplingParams(
        temperature=temperature,
        top_p=top_p,
        max_tokens=max_new_tokens,
        n=1,
        logprobs=logprobs_k,
        stop=None,
    )

    # 4) Generate attempts
    records = []
    for qb, prompt in prompt_pack:
        for attempt in range(attempts):
            outputs = llm.generate([prompt], sampling)
            out = outputs[0].outputs[0]

            text = out.text
            final_answer = parse_gsm8k_final(text)

            # Collect token-level logprobs (list with per-token info)
            tlogs = out.logprobs or []
            flat_logs = flatten_logprobs(tlogs)

            record = {
                "dataset": qb["dataset_label"],
                "dataset_index": qb["dataset_index"],
                "question": qb["question_text"],
                "ground_truth": qb["ground_truth"],
                "attempt": attempt,
                "model": model_name,
                "prompt": prompt,
                "completion": text,
                "final_answer": final_answer,
                "final_answer_gsm8k": f"#### {final_answer}" if final_answer else "",
                "token_logprobs": flat_logs,
            }
            records.append(record)
            print(f"[ATTEMPT] idx={qb['dataset_index']} attempt={attempt} final=({final_answer})", flush=True)

    # 5) Save as pandas
    df = pd.DataFrame.from_records(records)
    # Save in both parquet and csv (parquet preferred for nested columns)
    root, ext = os.path.splitext(out_path)
    parquet_path = out_path if ext.lower() == ".parquet" else root + ".parquet"
    csv_path = root + ".csv"

    df.to_parquet(parquet_path, index=False)
    # For CSV, we will json-encode the nested column
    df_for_csv = df.copy()
    df_for_csv["token_logprobs"] = df_for_csv["token_logprobs"].apply(json.dumps)
    df_for_csv.to_csv(csv_path, index=False)

    print(f"[DONE] Wrote {len(df)} rows to: {parquet_path} and {csv_path}", flush=True)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, default="deepseek-ai/DeepSeek-R1-Distill-Qwen-7B",
                        help="DeepSeek 7B-8B model to load with vLLM.")
    parser.add_argument("--questions", type=int, nargs="+", default=[0, 13],
                        help="Question indices to pull from the AIME dataset (will wrap with modulo if out of range).")
    parser.add_argument("--attempts", type=int, default=7, help="Attempts per question.")
    parser.add_argument("--logprobs", type=int, default=5, help="Top-K logprobs to record per token.")
    parser.add_argument("--max-new-tokens", type=int, default=512, help="Max generated tokens.")
    parser.add_argument("--temperature", type=float, default=0.8, help="Sampling temperature.")
    parser.add_argument("--top-p", type=float, default=0.95, help="Top-p nucleus sampling.")
    parser.add_argument("--out", type=str, default="generations.parquet", help="Output parquet (CSV is also emitted).")
    parser.add_argument("--dtype", type=str, default="bfloat16", choices=["auto", "float16", "bfloat16"],
                        help="Model dtype (vLLM).")
    args = parser.parse_args()

    run(
        model_name=args.model,
        questions=args.questions,
        attempts=args.attempts,
        logprobs_k=args.logprobs,
        max_new_tokens=args.max_new_tokens,
        temperature=args.temperature,
        top_p=args.top_p,
        out_path=args.out,
        dtype=args.dtype,
    )


if __name__ == "__main__":
    main()
