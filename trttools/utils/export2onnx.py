import torch
import onnx
import onnxsim

def set_input_names(inputs, num):
    input_names = []
    input_dims = []
    begin = num
    if isinstance(inputs, list) or isinstance(inputs, tuple):
        for ins in inputs:
            out_names, allnum, out_dims = set_input_names(ins, begin)
            input_names += out_names
            input_dims += out_dims
            begin = allnum      
    elif isinstance(inputs, dict):
        for key, value in inputs.items():
            out_names, allnum, out_dims = set_input_names(value, begin)
            input_names += out_names
            input_dims += out_dims
            begin = allnum
    elif isinstance(inputs, torch.Tensor):
        input_names.append(f"input_{num}")
        input_dims.append(len(inputs.shape))
        begin += 1
    else:
        raise ValueError("inputs are not all tensor value")
    return input_names, begin, input_dims

def set_out_names(outputs, num):
    output_names = []
    begin = num
    if isinstance(outputs, list) or isinstance(outputs, tuple):
        for ins in outputs:
            out_names, allnum = set_out_names(ins, begin)
            output_names += out_names
            begin = allnum      
    elif isinstance(outputs, dict):
        for key, value in outputs.items():
            out_names, allnum = set_out_names(value, begin)
            output_names += out_names
            begin = allnum
    elif isinstance(outputs, torch.Tensor):
        output_names.append(f"output_{num}")
        begin += 1
    else:
        raise ValueError("outputs are not all tensor value")
    return output_names, begin

def export2onnx(model, inputs, output_path='output.onnx', input_names=None, output_names=None, 
                dynamic=False, custom_dynamic_axes=None, verbose=False, simplify=False, opt_version=13):

    if input_names is None:
        input_names, _, input_dims = set_input_names(inputs, 0)
        print("auto synthesized input_names -> ", input_names)
    else:
        _, _, input_dims = set_input_names(inputs, 0)
        print("given input_names -> ", input_names)

    if isinstance(inputs, list) or isinstance(inputs, tuple):
        test_out = model(*inputs)
    else:
        test_out = model(inputs)

    if output_names is None:
        output_names, _ = set_out_names(test_out, 0)
        print("auto synthesized output_names -> ", output_names)
    else:
        print("given output_names -> ", output_names)
    
    if dynamic:
        if custom_dynamic_axes is not None:
            dynamic_axes = custom_dynamic_axes
        else:
            dynamic_axes = {}
            for in_name in input_names:
                dynamic_axes[in_name] = {0 : "batch"}
            for out_name in output_names:
                dynamic_axes[out_name] = {0: "batch"}
    else:
        dynamic_axes = None
    
    if isinstance(inputs, list):
        inputs = tuple(inputs)

    print("============== export model to onnx ==============")
    torch.onnx.export(model, inputs, output_path, input_names=input_names, output_names=output_names, 
                    verbose=verbose, dynamic_axes=dynamic_axes, opset_version=opt_version)

    
    if simplify:
        print("============== onnx simplify ==============")
        onnx_model = onnx.load(output_path) 
        onnx.checker.check_model(onnx_model) 
        onnx_model, check = onnxsim.simplify(onnx_model) 
        onnx.save(onnx_model, output_path) 
    
    return input_names, output_names, input_dims

