from .export2onnx import export2onnx
from .onnx2trt import onnx2trt
from .run import DynamicModel, StaticModel
import os
import torch

def static_float16(model, inputs, onnx=None, engine=None, device=torch.device("cuda")):
    if onnx is None:
        os.makedirs("midfile", exist_ok=True)
        onnx = os.path.join("midfile", "output.onnx")

    if engine is None:
        os.makedirs("midfile", exist_ok=True)
        engine = os.path.join("midfile", "output_static_fp16.engine")

    export2onnx(model, inputs, onnx, simplify=True)
    onnx2trt(onnx, engine, fp16=True)     
    engine_model = StaticModel(engine)
    return engine_model

def dynamic_float16(model, inputs, onnx=None, engine=None, input_names=None, output_names=None, custom_dynamic_axes=None, minShapes=None, optShapes=None, maxShapes=None, device=torch.device("cuda")):
    if optShapes is None:
        raise ValueError("lack of dynamic condition optShapes")
    
    if minShapes is None:
        minShapes = optShapes
    
    if maxShapes is None:
        maxShapes = optShapes
    
    input_names, output_names, input_dims = export2onnx(model, inputs, onnx, input_names, output_names, 
                                                    dynamic=True, custom_dynamic_axes=custom_dynamic_axes, simplify=True)
    onnx2trt(onnx, engine, fp16=True, input_names=input_names, input_dims=input_dims, dynamic=True, 
            minShapes=minShapes, optShapes=optShapes, maxShapes=maxShapes)
    
    engine_model = DynamicModel(engine)
    return engine_model


def dynamic_float32(model, inputs, onnx=None, engine=None, input_names=None, output_names=None, custom_dynamic_axes=None, minShapes=None, optShapes=None, maxShapes=None, device=torch.device("cuda")):
    if optShapes is None:
        raise ValueError("lack of dynamic condition optShapes")
    
    if minShapes is None:
        minShapes = optShapes
    
    if maxShapes is None:
        maxShapes = optShapes
    
    input_names, output_names, input_dims = export2onnx(model, inputs, onnx, input_names, output_names, 
                                                    dynamic=True, custom_dynamic_axes=custom_dynamic_axes, simplify=False)
    onnx2trt(onnx, engine, fp16=False, input_names=input_names, input_dims=input_dims, dynamic=True, 
            minShapes=minShapes, optShapes=optShapes, maxShapes=maxShapes)
    
    engine_model = DynamicModel(engine)
    return engine_model

