import argparse
import os
from shutil import copy2
from pathlib import Path


def parse_args():
    parser = argparse.ArgumentParser(description="Export an Ultralytics .pt model to ONNX.")
    parser.add_argument("pt_file", help="Path to the .pt model file.")
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

    from ultralytics import YOLO

    args = parse_args()
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
    print(f"Export imgsz: {args.imgsz}")

    model = YOLO(str(export_pt_path))
    exported = model.export(
        format="onnx",
        imgsz=args.imgsz,
        opset=args.opset,
        simplify=args.simplify,
        dynamic=args.dynamic,
        verbose=False,
    )

    onnx_path = Path(exported).resolve()
    print(f"Saved: {onnx_path}")

    if args.no_check:
        return

    import onnx
    import onnxruntime as ort

    checked = onnx.load(str(onnx_path))
    onnx.checker.check_model(checked)

    session = ort.InferenceSession(str(onnx_path), providers=["CPUExecutionProvider"])
    inputs = [(item.name, item.shape, item.type) for item in session.get_inputs()]
    outputs = [(item.name, item.shape, item.type) for item in session.get_outputs()]

    print("ONNX check: OK")
    print(f"Inputs: {inputs}")
    print(f"Outputs: {outputs}")


if __name__ == "__main__":
    main()
