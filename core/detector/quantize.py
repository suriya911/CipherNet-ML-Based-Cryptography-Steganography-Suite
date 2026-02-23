import argparse
import time
from pathlib import Path
from typing import Dict, Optional

import numpy as np
import torch

try:
    import onnxruntime as ort
    from onnxruntime.quantization import QuantType, quantize_dynamic
except ModuleNotFoundError:
    ort = None
    QuantType = None
    quantize_dynamic = None

try:
    from core.detector.cnn_detector import CNNDetector
except ModuleNotFoundError:
    from ciphernet.core.detector.cnn_detector import CNNDetector


def export_detector_to_onnx(
    detector_checkpoint: str,
    onnx_output: str = "ciphernet/models/detector.onnx",
    opset_version: int = 11,
    device: Optional[str] = None,
) -> str:
    resolved_device = torch.device(device if device else ("cuda" if torch.cuda.is_available() else "cpu"))
    model = CNNDetector().to(resolved_device)

    ckpt = Path(detector_checkpoint)
    if ckpt.exists():
        state = torch.load(ckpt, map_location=resolved_device)
        model.load_state_dict(state)
    else:
        print(f"Warning: detector checkpoint not found at {detector_checkpoint}; exporting random-init weights.")

    model.eval()
    out_path = Path(onnx_output)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    dummy_input = torch.randn(1, 3, 256, 256, device=resolved_device)
    try:
        torch.onnx.export(
            model,
            dummy_input,
            str(out_path),
            opset_version=opset_version,
            dynamo=False,
            input_names=["image"],
            output_names=["prob"],
            dynamic_axes={
                "image": {0: "batch"},
                "prob": {0: "batch"},
            },
        )
    except ModuleNotFoundError as exc:
        missing = getattr(exc, "name", "onnx dependencies")
        raise ImportError(
            f"Missing ONNX export dependency: {missing}. Install with: pip install onnx onnxscript"
        ) from exc
    return str(out_path)


def quantize_detector_onnx(
    onnx_input: str = "ciphernet/models/detector.onnx",
    int8_output: str = "ciphernet/models/detector_int8.onnx",
) -> str:
    if quantize_dynamic is None or QuantType is None:
        raise ImportError("onnxruntime is required for quantization. Install with: pip install onnxruntime")

    input_path = Path(onnx_input)
    if not input_path.exists():
        raise FileNotFoundError(f"ONNX model not found: {onnx_input}")

    output_path = Path(int8_output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    quantize_dynamic(
        model_input=str(input_path),
        model_output=str(output_path),
        weight_type=QuantType.QInt8,
    )
    return str(output_path)


def _create_session(model_path: str, intra_op_threads: int = 1):
    if ort is None:
        raise ImportError("onnxruntime is required for benchmarking. Install with: pip install onnxruntime")

    opts = ort.SessionOptions()
    opts.intra_op_num_threads = intra_op_threads
    opts.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
    return ort.InferenceSession(model_path, sess_options=opts, providers=["CPUExecutionProvider"])


def benchmark_onnx_model(
    model_path: str,
    runs: int = 200,
    warmup: int = 20,
    batch_size: int = 1,
    seed: int = 42,
) -> Dict[str, float]:
    rng = np.random.default_rng(seed)
    session = _create_session(model_path)
    input_name = session.get_inputs()[0].name
    x = rng.random((batch_size, 3, 256, 256), dtype=np.float32)

    for _ in range(warmup):
        session.run(None, {input_name: x})

    start = time.perf_counter()
    for _ in range(runs):
        session.run(None, {input_name: x})
    elapsed = time.perf_counter() - start

    latency_ms = (elapsed / runs) * 1000.0
    throughput = runs * batch_size / elapsed
    return {"latency_ms": latency_ms, "throughput_img_s": throughput}


def compare_fp32_int8(
    fp32_model: str,
    int8_model: str,
    runs: int = 200,
    warmup: int = 20,
    batch_size: int = 1,
) -> Dict[str, float]:
    fp32 = benchmark_onnx_model(fp32_model, runs=runs, warmup=warmup, batch_size=batch_size)
    int8 = benchmark_onnx_model(int8_model, runs=runs, warmup=warmup, batch_size=batch_size)
    speedup_pct = ((fp32["latency_ms"] - int8["latency_ms"]) / fp32["latency_ms"]) * 100.0
    return {
        "fp32_latency_ms": fp32["latency_ms"],
        "int8_latency_ms": int8["latency_ms"],
        "fp32_throughput_img_s": fp32["throughput_img_s"],
        "int8_throughput_img_s": int8["throughput_img_s"],
        "speedup_pct": speedup_pct,
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Export detector to ONNX and quantize to INT8.")
    parser.add_argument("--detector-checkpoint", type=str, default="ciphernet/models/detector_final.pth")
    parser.add_argument("--onnx-output", type=str, default="ciphernet/models/detector.onnx")
    parser.add_argument("--int8-output", type=str, default="ciphernet/models/detector_int8.onnx")
    parser.add_argument("--opset", type=int, default=11)
    parser.add_argument("--device", type=str, default="")
    parser.add_argument("--runs", type=int, default=200)
    parser.add_argument("--warmup", type=int, default=20)
    parser.add_argument("--batch-size", type=int, default=1)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    onnx_path = export_detector_to_onnx(
        detector_checkpoint=args.detector_checkpoint,
        onnx_output=args.onnx_output,
        opset_version=args.opset,
        device=args.device or None,
    )
    int8_path = quantize_detector_onnx(onnx_input=onnx_path, int8_output=args.int8_output)
    metrics = compare_fp32_int8(
        fp32_model=onnx_path,
        int8_model=int8_path,
        runs=args.runs,
        warmup=args.warmup,
        batch_size=args.batch_size,
    )
    print(f"FP32 vs INT8 benchmark: {metrics}")


if __name__ == "__main__":
    main()
