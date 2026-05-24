import argparse
import importlib
import os
from shutil import copy2
from pathlib import Path


def parse_args():
    parser = argparse.ArgumentParser(
        description="Export an Ultralytics .pt model to ONNX, TensorRT engine, or TFLite."
    )
    parser.add_argument("pt_file", help="Path to the .pt model file.")
    parser.add_argument(
        "--format",
        choices=["onnx", "engine", "tflite"],
        default="onnx",
        help="Export format. Use engine on Jetson/TensorRT environments. Default: onnx.",
    )
    parser.add_argument(
        "--imgsz",
        type=int,
        required=True,
        help="Export input image size, for example 640 or 320.",
    )
    parser.add_argument(
        "--opset",
        type=int,
        default=12,
        help="ONNX opset version. Default: 12.",
    )
    parser.add_argument(
        "--batch",
        type=int,
        default=1,
        help="Export batch size. Default: 1.",
    )
    parser.add_argument(
        "--device",
        default=None,
        help="Export device, for example 0 or cuda:0. Required for TensorRT engine on Jetson.",
    )
    parser.add_argument(
        "--workspace",
        type=float,
        default=None,
        help="TensorRT workspace size in GiB. Only used by engine export.",
    )
    parser.add_argument(
        "--half",
        action="store_true",
        help="Export FP16 model. Commonly used for TensorRT engine on Jetson.",
    )
    parser.add_argument(
        "--int8",
        action="store_true",
        help="Export INT8 model. Requires calibration data in Ultralytics.",
    )
    parser.add_argument(
        "--data",
        default=None,
        help="Dataset YAML/path for INT8 calibration when the export format needs it.",
    )
    parser.add_argument(
        "--output-dir",
        default=None,
        help="Optional output directory. Default: same directory as the .pt file.",
    )
    parser.add_argument(
        "--dynamic",
        action="store_true",
        help="Export with dynamic input shape.",
    )
    parser.add_argument(
        "--simplify",
        action="store_true",
        help="Simplify the exported ONNX model.",
    )
    parser.add_argument(
        "--no-check",
        action="store_true",
        help="Skip ONNX checker and ONNX Runtime load verification.",
    )
    return parser.parse_args()


def main():
    # Works around older onnx/protobuf combinations on this machine.
    os.environ.setdefault("PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION", "python")

    args = parse_args()
    if args.format == "tflite":
        try:
            importlib.import_module("tensorflow")
        except Exception as exc:
            raise RuntimeError(
                "TFLite export requires a working TensorFlow install. "
                "This environment failed to import TensorFlow; check the TensorFlow/NumPy versions."
            ) from exc

    from ultralytics import YOLO

    pt_path = Path(args.pt_file).expanduser().resolve()
    if not pt_path.exists():
        raise FileNotFoundError(f"Model file not found: {pt_path}")
    if pt_path.suffix.lower() != ".pt":
        raise ValueError(f"Expected a .pt file, got: {pt_path}")

    export_pt_path = pt_path
    if args.output_dir:
        output_dir = Path(args.output_dir).expanduser().resolve()
        output_dir.mkdir(parents=True, exist_ok=True)
        export_pt_path = output_dir / pt_path.name
        if export_pt_path != pt_path:
            copy2(pt_path, export_pt_path)

    print(f"Loading: {pt_path}")
    print(f"Export format: {args.format}")
    print(f"Export imgsz: {args.imgsz}")

    model = YOLO(str(export_pt_path))
    export_kwargs = dict(
        format=args.format,
        imgsz=args.imgsz,
        opset=args.opset,
        batch=args.batch,
        simplify=args.simplify,
        dynamic=args.dynamic,
        half=args.half,
        int8=args.int8,
        verbose=False,
    )
    if args.device is not None:
        export_kwargs["device"] = args.device
    if args.workspace is not None:
        export_kwargs["workspace"] = args.workspace
    if args.data is not None:
        export_kwargs["data"] = args.data

    exported = model.export(**export_kwargs)

    output_path = Path(exported).resolve()
    print(f"Saved: {output_path}")

    if args.no_check:
        return

    if args.format == "engine":
        if not output_path.exists():
            raise FileNotFoundError(f"Exported engine not found: {output_path}")
        print("Engine export: OK")
        print("Note: TensorRT engine files are hardware/version specific. Build on the target Jetson.")
        return

    if args.format == "tflite":
        if not output_path.exists():
            raise FileNotFoundError(f"Exported TFLite model not found: {output_path}")
        print("TFLite export: OK")
        try:
            try:
                from tflite_runtime.interpreter import Interpreter
            except ImportError:
                from tensorflow.lite.python.interpreter import Interpreter

            interpreter = Interpreter(model_path=str(output_path))
            interpreter.allocate_tensors()
            inputs = interpreter.get_input_details()
            outputs = interpreter.get_output_details()
            print(f"Inputs: {[(item['name'], item['shape'].tolist(), str(item['dtype'])) for item in inputs]}")
            print(f"Outputs: {[(item['name'], item['shape'].tolist(), str(item['dtype'])) for item in outputs]}")
        except ImportError:
            print("TFLite load check skipped: install tensorflow or tflite_runtime to verify locally.")
        return

    import onnx
    import onnxruntime as ort

    checked = onnx.load(str(output_path))
    onnx.checker.check_model(checked)

    session = ort.InferenceSession(str(output_path), providers=["CPUExecutionProvider"])
    inputs = [(item.name, item.shape, item.type) for item in session.get_inputs()]
    outputs = [(item.name, item.shape, item.type) for item in session.get_outputs()]

    print("ONNX check: OK")
    print(f"Inputs: {inputs}")
    print(f"Outputs: {outputs}")


if __name__ == "__main__":
    main()
