#!/usr/bin/env python3
"""Four-way speed/accuracy bench: PyTorch BF16 vs TensorRT FP16 / FP8 / FP8+NVFP4.

Uses real EVT276 training frames (Decord, same path as calibration_data.py).
Timing is mean of --num-test-runs inferences on sample 0 after warmup.
Accuracy is mean cosine / max-abs-diff vs PyTorch BF16 across --num-samples frames.
"""

from __future__ import annotations

import argparse
import gc
import json
import os
import time

import numpy as np
import torch
from decord import VideoReader, cpu
from lerobot.datasets.lerobot_dataset import LeRobotDataset
from openpi.policies import policy_config
from openpi.training import config as _config

from deployment_scripts.pi05_inference import compare_outputs
from deployment_scripts.trt_model_forward import (
    install_attention_mask_dtype_fix,
    setup_pi0_tensorrt_engine,
)

DATASET_ROOT = "/home/nvidia/datasets/EVT276_MOVE_BOX_0813/move_box_0813"
VIDEO_KEY = "observation.images.camera_head"


def _as_numpy(x) -> np.ndarray:
    if hasattr(x, "numpy"):
        x = x.numpy()
    return np.asarray(x)


def load_real_examples(repo_id: str, root: str, num_samples: int) -> list[dict]:
    print(f"Loading {num_samples} real samples from {root} (repo_id={repo_id})")
    dataset = LeRobotDataset(repo_id, root=root)
    n = len(dataset)
    step = max(n // num_samples, 1)
    indices = list(range(0, min(n, num_samples * step), step))[:num_samples]
    print(f"  dataset length={n}, indices={indices}")

    examples = []
    for data_idx in indices:
        row = dataset.hf_dataset[data_idx]
        episode_index = int(row["episode_index"].item())
        frame_index = int(row["frame_index"].item())
        task_index = int(row["task_index"].item())
        video_path = dataset.root / dataset.meta.get_video_file_path(episode_index, VIDEO_KEY)
        reader = VideoReader(str(video_path), ctx=cpu(0))
        frame_index = min(max(frame_index, 0), len(reader) - 1)
        image = reader[frame_index].asnumpy()
        state = _as_numpy(row["observation.state"]).astype(np.float32)
        prompt = dataset.meta.tasks[task_index]
        examples.append({"image": image, "state": state, "prompt": prompt})
        print(
            f"  sample {len(examples) - 1}: idx={data_idx} "
            f"image={image.shape} state={state.shape} prompt={prompt!r}"
        )
    return examples


def _stats(values: list[float]) -> dict:
    arr = np.asarray(values, dtype=np.float64)
    return {
        "mean": float(arr.mean()) if arr.size else float("nan"),
        "std": float(arr.std()) if arr.size else float("nan"),
        "min": float(arr.min()) if arr.size else float("nan"),
        "max": float(arr.max()) if arr.size else float("nan"),
        "all": [float(v) for v in arr],
    }


def _finite(actions: np.ndarray | None) -> bool:
    return actions is not None and np.isfinite(actions).all()


def cosine_and_diff(ref: np.ndarray, pred: np.ndarray) -> dict:
    if ref.shape != pred.shape:
        return {
            "cosine": float("nan"),
            "max_abs": float("nan"),
            "mean_abs": float("nan"),
            "shape_mismatch": True,
        }
    ref_f = ref.astype(np.float64).ravel()
    pred_f = pred.astype(np.float64).ravel()
    denom = (np.linalg.norm(ref_f) * np.linalg.norm(pred_f)) + 1e-8
    diff = np.abs(ref - pred)
    return {
        "cosine": float(np.dot(ref_f, pred_f) / denom),
        "max_abs": float(diff.max()),
        "mean_abs": float(diff.mean()),
        "shape_mismatch": False,
    }


def _load_policy(config, checkpoint_dir: str, engine_path: str | None, num_steps: int):
    sample_kwargs = {"num_steps": num_steps}
    if engine_path is None:
        policy = policy_config.create_trained_policy(
            config, checkpoint_dir, trt=False, sample_kwargs=sample_kwargs
        )
        install_attention_mask_dtype_fix(policy._model if hasattr(policy, "_model") else policy.model)
        return policy
    policy = policy_config.create_trained_policy(
        config, checkpoint_dir, trt=True, sample_kwargs=sample_kwargs
    )
    return setup_pi0_tensorrt_engine(policy, engine_path)


def run_backend(
    name: str,
    config,
    checkpoint_dir: str,
    engine_path: str | None,
    examples: list[dict],
    noise: np.ndarray,
    num_warmup: int,
    num_test_runs: int,
    num_steps: int,
) -> dict:
    print("\n" + "=" * 60)
    print(f"Backend: {name}  (num_steps={num_steps})")
    print("=" * 60)
    result = {
        "name": name,
        "engine_path": engine_path,
        "num_steps": num_steps,
        "ok": False,
        "nan": False,
        "error": None,
        "inference_stats": None,
        "model_stats": None,
        "actions": [],
    }

    policy = None
    try:
        policy = _load_policy(config, checkpoint_dir, engine_path, num_steps)
        timing_example = examples[0]

        print(f"Warming up ({num_warmup} runs on sample 0)...")
        for i in range(num_warmup):
            _ = policy.infer(timing_example, noise=noise)
            print(f"  warmup {i + 1}/{num_warmup}")

        print(f"Timing ({num_test_runs} runs on sample 0)...")
        inference_times = []
        model_times = []
        first_actions = None
        for i in range(num_test_runs):
            t0 = time.time()
            out = policy.infer(timing_example, noise=noise)
            inference_times.append((time.time() - t0) * 1000)
            model_times.append(out.get("policy_timing", {}).get("infer_ms", inference_times[-1]))
            if i == 0:
                first_actions = np.asarray(out["actions"])
            print(f"  test {i + 1}/{num_test_runs}: {inference_times[-1]:.2f} ms")

        result["inference_stats"] = _stats(inference_times)
        result["model_stats"] = _stats(model_times)

        actions = [first_actions]
        print(f"Accuracy pass ({len(examples)} real samples)...")
        for idx, example in enumerate(examples):
            if idx == 0:
                continue
            out = policy.infer(example, noise=noise)
            actions.append(np.asarray(out["actions"]))
            print(f"  sample {idx}: shape={actions[-1].shape} range=[{actions[-1].min():.4f}, {actions[-1].max():.4f}]")

        result["actions"] = actions
        result["nan"] = any(not _finite(a) for a in actions)
        result["ok"] = not result["nan"]
        if result["nan"]:
            print("  WARNING: NaN/Inf in actions")
    except Exception as exc:
        result["error"] = f"{type(exc).__name__}: {exc}"
        print(f"  FAILED: {result['error']}")
        import traceback

        traceback.print_exc()
    finally:
        del policy
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    return result


def print_table(rows: list[dict]) -> None:
    print("\n" + "=" * 100)
    print("Comparison (latency = mean of 10 timed runs on sample 0)")
    print("Accuracy = mean cosine / MAE vs the matching-step PyTorch BF16")
    print("=" * 100)
    header = (
        f"{'backend':<22} {'steps':>5} {'total_ms':>10} {'model_ms':>10} {'speedup':>8} "
        f"{'cosine':>10} {'mae':>10} {'finite':>8}"
    )
    print(header)
    print("-" * 100)
    refs = {r["name"]: r for r in rows if r.get("engine_path") is None}
    for row in rows:
        inf = row.get("inference_stats") or {}
        model = row.get("model_stats") or {}
        total = inf.get("mean", float("nan"))
        model_ms = model.get("mean", float("nan"))
        ref_name = row.get("ref")
        ref = refs.get(ref_name) if ref_name else None
        if ref is None and row.get("engine_path") is None:
            speedup = 1.0
        elif ref and ref.get("inference_stats"):
            pt_total = ref["inference_stats"]["mean"]
            speedup = pt_total / total if total and total == total and total > 0 else float("nan")
        else:
            speedup = float("nan")
        acc = row.get("accuracy") or {}
        finite = "yes" if row.get("ok") else ("NaN" if row.get("nan") else "FAIL")
        print(
            f"{row['name']:<22} {row.get('num_steps', '?'):>5} {total:10.2f} {model_ms:10.2f} {speedup:8.2f} "
            f"{acc.get('cosine_mean', float('nan')):10.6f} {acc.get('mae_mean', float('nan')):10.6f} {finite:>8}"
        )
        if row.get("error"):
            print(f"  error: {row['error']}")
    print("=" * 100)


def main() -> None:
    parser = argparse.ArgumentParser(description="Compare PyTorch BF16 vs TensorRT backends")
    parser.add_argument("--config-name", default="pi05_tienkung_evt276_full_hand_abs")
    parser.add_argument(
        "--checkpoint-dir",
        default="/home/nvidia/ckpt/evt276_move_box_0813_pytorch",
    )
    parser.add_argument("--dataset-root", default=DATASET_ROOT)
    parser.add_argument("--num-samples", type=int, default=10)
    parser.add_argument("--num-warmup", type=int, default=3)
    parser.add_argument("--num-test-runs", type=int, default=10)
    parser.add_argument(
        "--golden-noise-path",
        default="/home/nvidia/ckpt/evt276_move_box_0813_pytorch/golden_noise.npy",
    )
    parser.add_argument(
        "--results-path",
        default="/home/nvidia/ckpt/evt276_move_box_0813_pytorch/compare_backends.json",
    )
    parser.add_argument(
        "--suite",
        choices=["default", "s5_no_nvfp4"],
        default="default",
        help="default: original 10-step + hybrid; s5_no_nvfp4: two 10.13 FP8 AE variants vs BF16 s5",
    )
    args = parser.parse_args()

    config = _config.get_config(args.config_name)
    examples = load_real_examples(config.data.repo_id, args.dataset_root, args.num_samples)

    noise_path = args.golden_noise_path
    if os.path.exists(noise_path):
        noise = np.load(noise_path)
        print(f"Loaded golden noise {noise.shape} from {noise_path}")
    else:
        noise = np.random.normal(
            0.0, 1.0, size=(config.model.action_horizon, config.model.action_dim)
        ).astype(np.float32)
        os.makedirs(os.path.dirname(noise_path), exist_ok=True)
        np.save(noise_path, noise)
        print(f"Saved golden noise {noise.shape} to {noise_path}")

    if args.suite == "s5_no_nvfp4":
        backends = [
            {"name": "pytorch_bf16_s5", "engine": None, "num_steps": 5, "ref": None},
            {
                "name": "trt_fp8_ae_fp16_s5",
                "engine": (
                    "/home/nvidia/ckpt/evt276_move_box_0813_pytorch_trt1013_fp8_ae_s5"
                    "/engine/model_fp8_ae_fp16_s5.engine"
                ),
                "num_steps": 5,
                "ref": "pytorch_bf16_s5",
            },
            {
                "name": "trt_fp8_aeproj_fp16_s5",
                "engine": (
                    "/home/nvidia/ckpt/evt276_move_box_0813_pytorch_trt1013_fp8_aeproj_s5"
                    "/engine/model_fp8_aeproj_fp16_s5.engine"
                ),
                "num_steps": 5,
                "ref": "pytorch_bf16_s5",
            },
        ]
    else:
        backends = [
            {"name": "pytorch_bf16", "engine": None, "num_steps": 10, "ref": None},
            {
                "name": "tensorrt_fp16",
                "engine": os.path.join(args.checkpoint_dir, "engine/model_fp16.engine"),
                "num_steps": 10,
                "ref": "pytorch_bf16",
            },
            {
                "name": "tensorrt_fp8",
                "engine": os.path.join(args.checkpoint_dir, "engine/model_fp8.engine"),
                "num_steps": 10,
                "ref": "pytorch_bf16",
            },
            {
                "name": "tensorrt_fp8_nvfp4",
                "engine": os.path.join(args.checkpoint_dir, "engine/model_fp8_nvfp4.engine"),
                "num_steps": 10,
                "ref": "pytorch_bf16",
            },
            {"name": "pytorch_bf16_s5", "engine": None, "num_steps": 5, "ref": None},
            {
                "name": "tensorrt_hybrid_s5",
                "engine": os.path.join(args.checkpoint_dir, "engine/model_fp8_nvfp4_ae_fp16_s5.engine"),
                "num_steps": 5,
                "ref": "pytorch_bf16_s5",
            },
        ]

    rows = []
    actions_by_name: dict[str, list] = {}
    for spec in backends:
        row = run_backend(
            spec["name"],
            config,
            args.checkpoint_dir,
            spec["engine"],
            examples,
            noise,
            args.num_warmup,
            args.num_test_runs,
            spec["num_steps"],
        )
        row["ref"] = spec["ref"]
        if spec["engine"] is None:
            actions_by_name[spec["name"]] = row.get("actions") or []
            row["accuracy"] = {
                "cosine_mean": 1.0 if row.get("ok") else float("nan"),
                "mae_mean": 0.0 if row.get("ok") else float("nan"),
                "ref": spec["name"],
            }
            if row.get("ok") and row.get("actions"):
                a0 = row["actions"][0]
                print(f"\n{spec['name']} sample-0 action summary:")
                print(f"  shape={a0.shape} range=[{a0.min():.4f}, {a0.max():.4f}]")
        else:
            ref_actions = actions_by_name.get(spec["ref"] or "", [])
            metrics = []
            if ref_actions and row.get("actions"):
                n = min(len(ref_actions), len(row["actions"]))
                for i in range(n):
                    if not _finite(ref_actions[i]) or not _finite(row["actions"][i]):
                        metrics.append(
                            {"cosine": float("nan"), "max_abs": float("nan"), "mean_abs": float("nan")}
                        )
                    else:
                        metrics.append(cosine_and_diff(ref_actions[i], row["actions"][i]))
                if metrics and n:
                    print(f"\n{spec['name']} vs {spec['ref']} (sample 0 detail):")
                    compare_outputs(ref_actions[0], row["actions"][0])
            row["accuracy"] = {
                "cosine_mean": float(np.nanmean([m["cosine"] for m in metrics])) if metrics else float("nan"),
                "mae_mean": float(np.nanmean([m["mean_abs"] for m in metrics])) if metrics else float("nan"),
                "ref": spec["ref"],
                "per_sample": metrics,
            }
        json_row = {k: v for k, v in row.items() if k != "actions"}
        json_row["action_shape"] = None if not row.get("actions") else list(row["actions"][0].shape)
        if row.get("actions"):
            a0 = row["actions"][0]
            json_row["action_range"] = [float(np.nanmin(a0)), float(np.nanmax(a0))]
        rows.append(row)
        row["_json"] = json_row

    print_table(rows)
    os.makedirs(os.path.dirname(args.results_path), exist_ok=True)
    with open(args.results_path, "w") as f:
        json.dump({"config": args.config_name, "results": [r["_json"] for r in rows]}, f, indent=2)
    print(f"\nWrote {args.results_path}")


if __name__ == "__main__":
    main()
