import tensorrt as trt
import torch
from hybrik.hybrik import Simple3DPoseBaseSMPLCam
from onnxsim import simplify
import onnx

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
    engine_file_path = onnx_file_path.replace(".onnx", ".engine")
    build_engine(onnx_file_path, engine_file_path, half=True) # Set half=False if you want full precision fp32

model = Simple3DPoseBaseSMPLCam()
img = torch.randn(1, 3, 256, 256)
shape, pose = model(img)

torch.onnx.export(
    model,
    (img),
    "hybrik.onnx",
    input_names=["img"],
    output_names=["shape", "pose"],
)

model_onnx = onnx.load("hybrik.onnx")
model_simp, check = simplify(model_onnx)
onnx.save(model_simp, "hybrik.onnx")

build_model("hybrik.onnx")