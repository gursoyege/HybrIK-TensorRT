import os
import torch
import onnx
import tensorrt as trt
from onnxsim import simplify
from hybrik.hybrik import Simple3DPoseBaseSMPLCam

ENGINES_DIR = "engines"
os.makedirs(ENGINES_DIR, exist_ok=True)

def build_engine(onnx_file_path, engine_file_path, half=True):
    logger = trt.Logger(trt.Logger.INFO)
    builder = trt.Builder(logger)
    config = builder.create_builder_config()

    flag = (1 << int(trt.NetworkDefinitionCreationFlag.EXPLICIT_BATCH))
    network = builder.create_network(flag)
    parser = trt.OnnxParser(network, logger)

    if not parser.parse_from_file(str(onnx_file_path)):
        raise RuntimeError(f"Failed to load ONNX file: {onnx_file_path}")

    input_tensor = network.get_input(0)
    input_tensor.shape = (1, 3, 256, 256)

    half &= builder.platform_has_fast_fp16
    if half:
        config.set_flag(trt.BuilderFlag.FP16)

    with builder.build_serialized_network(network, config) as engine, open(engine_file_path, "wb") as f:
        f.write(engine)
    return engine_file_path

def build_model(onnx_file_path):
    engine_file_name = os.path.basename(onnx_file_path).replace(".onnx", ".engine")
    engine_file_path = os.path.join(ENGINES_DIR, engine_file_name)

    if not os.path.exists(engine_file_path):
        print(f"Engine not found, building new engine at: {engine_file_path}")
        build_engine(onnx_file_path, engine_file_path, half=True)
    else:
        print(f"Engine already exists at: {engine_file_path}, skipping build.")

def export_onnx_model(model, onnx_path, input_tensor):
    if not os.path.exists(onnx_path):
        print(f"ONNX file not found, exporting model to: {onnx_path}")
        torch.onnx.export(
            model,
            (input_tensor,),
            onnx_path,
            input_names=["img"],
            output_names=["shape", "pose"],
        )

        model_onnx = onnx.load(onnx_path)
        model_simp, check = simplify(model_onnx)
        onnx.save(model_simp, onnx_path)
    else:
        print(f"ONNX file already exists at: {onnx_path}, skipping export.")

model = Simple3DPoseBaseSMPLCam()
img = torch.randn(1, 3, 256, 256)

onnx_path = os.path.join(ENGINES_DIR, "hybrik.onnx")

export_onnx_model(model, onnx_path, img)
build_model(onnx_path)
