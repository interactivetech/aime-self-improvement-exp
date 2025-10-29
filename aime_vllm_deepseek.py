#!/usr/bin/env python3
"""
AIME2025 (OpenCompass) eval with vLLM + DeepSeek 7B

- Pulls questions from Hugging Face dataset: opencompass/AIME2025 (subsets: AIME2025-I & AIME2025-II), split="test"
- Uses global 0-based indices across the concatenated order [AIME2025-I (0..14), then AIME2025-II (15..29)]
- Loads a DeepSeek 7B-ish model with vLLM
- Runs N attempts per selected question indices
- Prompts the model to end with GSM8K-style final line: "#### <answer>"
- Captures top log probabilities per generated token
- Saves generations as pandas (Parquet + CSV)

Example:
  python aime_vllm_deepseek.py \
    --model deepseek-ai/DeepSeek-R1-Distill-Qwen-7B \
    --attempts 7 \
    --questions 0 13 \
    --logprobs 5 \
    --out generations.parquet
"""

import argparse
import json
import os
import re
from typing import Dict, List, Tuple, Any

import pandas as pd


def _lazy_import_vllm():
    from vllm import LLM, SamplingParams
    return LLM, SamplingParams


def _lazy_import_datasets():
    from datasets import load_dataset, Dataset
    return load_dataset, Dataset


def load_aime2025_concat() -> Tuple[Any, str]:
    """Load opencompass/AIME2025 both subsets and concatenate to a single list-like split."""
    load_dataset, Dataset = _lazy_import_datasets()
    ds_i = load_dataset("opencompass/AIME2025", "AIME2025-I", split="test")
    ds_ii = load_dataset("opencompass/AIME2025", "AIME2025-II", split="test")
    # Both have fields: question (str), answer (str). Concatenate preserving order I then II.
    ds_all = Dataset.from_list(list(ds_i) + list(ds_ii))
    return ds_all, "opencompass/AIME2025:[AIME2025-I,test]+[AIME2025-II,test]"


def extract_qa_fields(example: Dict[str, Any]) -> Tuple[str, str]:
    q = example.get("question", "")
    a = example.get("answer", "")
    return str(q), str(a)


def build_prompt(question_text: str) -> str:
    system = (
        "You are a careful competition math solver. "
        "Solve the problem step by step. "
        "At the very end, output the final numeric answer on a new line as: #### <answer>"
    )
    user = question_text.strip()
    prompt = (
        f"<|system|>\n{system}\n</|system|>\n"
        f"<|user|>\n{user}\n</|user|>\n"
        f"<|assistant|>\n"
    )
    return prompt


def parse_gsm8k_final(text: str) -> str:
    m = re.search(r"^####\s*(.+)$", text.strip(), flags=re.MULTILINE)
    return m.group(1).strip() if m else ""


def flatten_logprobs(token_logprobs: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    flat = []
    for entry in token_logprobs or []:
        tok = getattr(entry, "decoded_token", None) or entry.get("decoded_token", "")
        lp = getattr(entry, "logprob", None) if hasattr(entry, "logprob") else entry.get("logprob", None)
        top = getattr(entry, "top_logprobs", None) if hasattr(entry, "top_logprobs") else entry.get("top_logprobs", None)

        top_list = []
        if top:
            for alt in top:
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
    ds_all, ds_label = load_aime2025_concat()
    print(f"[INFO] Loaded {ds_label} (size={len(ds_all)})", flush=True)

    # Wrap indices modulo the dataset length.
    N = len(ds_all)
    indices = [q % N for q in questions]
    rows = [ds_all[i] for i in indices]

    blob = []
    for idx_source, ex in zip(indices, rows):
        q_text, gt = extract_qa_fields(ex)
        blob.append({
            "dataset_label": ds_label,
            "dataset_index": idx_source,
            "question_text": q_text,
            "ground_truth": gt,
        })

    prompts = [(b, build_prompt(b["question_text"])) for b in blob]

    # Init vLLM
    print(f"[INFO] Loading model via vLLM: {model_name}", flush=True)
    LLM, SamplingParams = _lazy_import_vllm()
    llm = LLM(
        model=model_name,
        trust_remote_code=True,
        dtype=dtype,
    )

    sampling = SamplingParams(
        temperature=temperature,
        top_p=top_p,
        max_tokens=max_new_tokens,
        n=1,
        logprobs=logprobs_k,
    )

    records = []
    for qb, prompt in prompts:
        for attempt in range(attempts):
            outputs = llm.generate([prompt], sampling)
            out = outputs[0].outputs[0]

            text = out.text

            raw_final = extract_boxed_text(text)
            norm_final = normalize_number(raw_final) if raw_final else None
            final_answer = norm_final if norm_final is not None else (raw_final or "")

            tlogs = out.logprobs or []
            flat_logs = flatten_logprobs(tlogs)

            records.append({
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
            })
            print(f"[ATTEMPT] idx={qb['dataset_index']} attempt={attempt} final=({final_answer})", flush=True)

    df = pd.DataFrame.from_records(records)
    root, ext = os.path.splitext(out_path)
    parquet_path = out_path if ext.lower() == ".parquet" else root + ".parquet"
    csv_path = root + ".csv"

    df.to_parquet(parquet_path, index=False)
    df_csv = df.copy()
    df_csv["token_logprobs"] = df_csv["token_logprobs"].apply(json.dumps)
    df_csv.to_csv(csv_path, index=False)

    print(f"[DONE] Wrote {len(df)} rows to: {parquet_path} and {csv_path}", flush=True)


def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, default="deepseek-ai/DeepSeek-R1-Distill-Qwen-7B",
                        help="DeepSeek 7B-8B model to load with vLLM.")
    parser.add_argument("--questions", type=int, nargs="+", default=[0, 13],
                        help="Global 0-based indices across AIME2025-I then AIME2025-II.")
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
