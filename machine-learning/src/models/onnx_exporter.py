from pathlib import Path

"""
Exporting a trained Keras model to ONNX format with optional INT8 quantization.
Called from train.py after training completes.

- ONNX Runtime is 2-4× faster than TensorFlow for CPU inference on small models
- No accuracy loss (only weight representation changes)
- Runtime-independent — backend doesn't need TensorFlow installed
- Industry standard for model portability
"""


def export_to_onnx(
        model,
        input_shape: tuple,
        output_dir: str = "models",
        model_name: str = "model_energy_demand",
        opset: int = 13,
) -> str | None:
    """
    Converts a Keras model to ONNX format.

    Args:
        model:        Trained Keras model
        input_shape:  (window_size, n_features) — without batch dimension
        output_dir:   Directory to write the .onnx file
        model_name:   Base name for the output file
        opset:        ONNX opset version (13+ has solid LSTM support)

    Returns:
        Path to the saved .onnx file, or None if conversion failed.
    """
    try:
        # lazy imports
        import tf2onnx
        import onnx
        import shutil
        import tensorflow as tf

        Path(output_dir).mkdir(parents=True, exist_ok=True)
        onnx_path = str(Path(output_dir) / f"{model_name}.onnx")
        saved_model_dir = str(Path(output_dir) / f"{model_name}_savedmodel")

        model.export(saved_model_dir)

        tf2onnx.convert.from_saved_model(
            saved_model_dir,
            opset=opset,
            output_path=onnx_path,
        )

        shutil.rmtree(saved_model_dir)

        size_mb = round(Path(onnx_path).stat().st_size / 1024 / 1024, 2)
        print(f"ONNX model saved: {onnx_path} ({size_mb} MB)")
        return onnx_path

    except Exception as e:
        print(f"WARNING: ONNX export failed: {e}")
        return None


def quantize_onnx(onnx_path: str) -> str | None:
    """
    Applies dynamic INT8 quantization to an ONNX model.

    Dynamic quantization:
    - Quantizes weights to INT8 at export time
    - Activations are quantized dynamically at runtime
    - No calibration data needed
    - Typical speedup: 1.5-2× additional over base ONNX

    Args:
        onnx_path: Path to the base ONNX model

    Returns:
        Path to the quantized .onnx file, or None if quantization failed.
    """
    try:
        # lazy import
        from onnxruntime.quantization import quantize_dynamic, QuantType

        quantized_path = onnx_path.replace(".onnx", "_quantized.onnx")

        quantize_dynamic(
            model_input=onnx_path,
            model_output=quantized_path,
            weight_type=QuantType.QInt8,
        )

        original_mb = round(Path(onnx_path).stat().st_size / 1024 / 1024, 2)
        quantized_mb = round(Path(quantized_path).stat().st_size / 1024 / 1024, 2)
        reduction = round((1 - quantized_mb / original_mb) * 100, 1)

        print(f"Quantized ONNX saved: {quantized_path}")
        print(f"Size: {original_mb} MB → {quantized_mb} MB ({reduction}% reduction)")
        return quantized_path

    except Exception as e:
        print(f"WARNING: ONNX quantization failed: {e}")
        return None


def export_and_quantize(
        model,
        input_shape: tuple,
        output_dir: str = "models",
        model_name: str = "model_energy_demand",
) -> dict:
    """
    Convenience function: exports to ONNX and applies quantization.
    Returns paths to both artifacts.
    """
    results = {}

    onnx_path = export_to_onnx(model, input_shape, output_dir, model_name)
    if onnx_path:
        results["onnx"] = onnx_path
        quantized_path = quantize_onnx(onnx_path)
        if quantized_path:
            results["onnx_quantized"] = quantized_path

    return results
