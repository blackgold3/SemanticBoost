from torch import nn
import tensorrt as trt
trt.init_libnvinfer_plugins(None, "")
import torch
from collections import OrderedDict, namedtuple
import numpy as np

# import pycuda.driver as cuda

# class HostDeviceMem(object):
#     def __init__(self, host_mem, device_mem):
#         """
#         host_mem: cpu memory
#         device_mem: gpu memory
#         """
#         self.host = host_mem
#         self.device = device_mem

#     def __str__(self):
#         return "Host:\n" + str(self.host)+"\nDevice:\n"+str(self.device)

#     def __repr__(self):
#         return self.__str__()

# class HostEngineModel(nn.Module):
#     def __init__(self, path, device):
#         super().__init__()
#         model = self.get_engine(path)
#         self.device = device
#         self.context = model.create_execution_context()
#         self.inputs, self.outputs, self.bindings, self.stream, self.shapes = self.allocate_buffers(model) # input, output: host # bindings

#     def get_engine(self, engine_file_path):
#         print("Reading engine from file: {}".format(engine_file_path))
#         TRT_LOGGER = trt.Logger()
#         with open(engine_file_path, 'rb') as f, trt.Runtime(TRT_LOGGER) as runtime:
#             return runtime.deserialize_cuda_engine(f.read())  # 反序列化
    
#     def allocate_buffers(self, engine):
#         inputs, outputs, bindings = [], [], []
#         shapes = []
#         stream = cuda.Stream()
#         for binding in engine:
#             dims = engine.get_binding_shape(binding)
#             shape = tuple(dims)
#             shapes.append(shape)
#             size = trt.volume(dims)
#             if dims[0] < 0:
#                 size *= -1
#             dtype = trt.nptype(engine.get_binding_dtype(binding))
#             host_mem = cuda.pagelocked_empty(size, dtype)  
#             device_mem = cuda.mem_alloc(host_mem.nbytes)   
#             bindings.append(int(device_mem))
#             if engine.binding_is_input(binding):
#                 inputs.append(HostDeviceMem(host_mem, device_mem))
#             else:
#                 outputs.append(HostDeviceMem(host_mem, device_mem))
#         return inputs, outputs, bindings, stream, shapes

#     def do_inference(self, context, bindings, inputs, outputs, stream):
#         [cuda.memcpy_htod_async(inp.device, inp.host, stream) for inp in inputs]
#         context.execute_async_v2(bindings=bindings, stream_handle=stream.handle)
#         [cuda.memcpy_dtoh_async(out.host, out.device, stream) for out in outputs]
#         stream.synchronize()
#         return [out.host for out in outputs]

#     def postprocess_the_outputs(self, h_outputs, shape_of_output):
#         h_outputs = h_outputs.reshape(*shape_of_output)
#         return h_outputs

#     def forward(self, im):
#         if isinstance(im, list):
#             for i in range(len(im)):
#                 ins = im[i].contiguous().cpu().numpy()
#                 self.inputs[i].host = ins.reshape(-1)
#             outfrom = len(im)
#         else:
#             curr = im.contiguous().cpu().numpy()
#             self.inputs[0].host = curr.reshape(-1)
#             outfrom = 1

#         outs = self.do_inference(self.context, bindings=self.bindings, inputs=self.inputs, outputs=self.outputs, stream=self.stream)  
#         if len(outs) == 1:
#             curr = torch.from_numpy(self.postprocess_the_outputs(outs[0], self.shapes[outfrom])).to(self.device) 
#             return curr
#         else:
#             returns = []
#             for i in range(len(outs)):
#                 curr = torch.from_numpy(self.postprocess_the_outputs(outs[i], self.shapes[outfrom + i])).to(self.device) 
#                 returns.append(curr)
#             return returns

class StaticModel(nn.Module):
    def __init__(self, w, device="cuda"):
        super().__init__()
        print(f'Loading {w} for TensorRT inference...')
        Binding = namedtuple('Binding', ('name', 'dtype', 'shape', 'data', 'ptr'))
        logger = trt.Logger()
        with open(w, 'rb') as f, trt.Runtime(logger) as runtime:
            model = runtime.deserialize_cuda_engine(f.read())
        self.bindings = OrderedDict()
        for index in range(model.num_bindings):
            name = model.get_binding_name(index)
            dtype = trt.nptype(model.get_binding_dtype(index))
            shape = tuple(model.get_binding_shape(index))
            data = torch.from_numpy(np.ones(shape, dtype=np.dtype(dtype))).cuda()
            self.bindings[name] = Binding(name, dtype, shape, data, int(data.data_ptr()))
        self.binding_addrs = OrderedDict((n, d.ptr) for n, d in self.bindings.items())
        self.context = model.create_execution_context()
        self.keys = list(self.bindings.keys())
        self.device = device

    def forward(self, im):
        if isinstance(im, list):
            for i in range(len(im)):
                ins = im[i].contiguous().cuda()
                self.binding_addrs[self.keys[i]] = int(ins.data_ptr())
                outfrom = i + 1
        else:
            im = im.contiguous().cuda()
            self.binding_addrs[self.keys[0]] = int(im.data_ptr())
            outfrom = 1

        self.context.execute_v2(list(self.binding_addrs.values()))
        if len(self.keys) == outfrom + 1:
            return self.bindings[self.keys[outfrom]].data.to(self.device)
        else:
            res = []
            for i in range(outfrom, len(self.keys)):
                res.append(self.bindings[self.keys[i]].data.to(self.device))
            return res

class DynamicModel(nn.Module):
    def __init__(self, w, device="cuda"):
        super().__init__()
        print(f'Loading {w} for TensorRT inference...')
        Binding = namedtuple('Binding', ('name', 'dtype', 'shape', 'data', 'ptr'))
        logger = trt.Logger()
        with open(w, 'rb') as f, trt.Runtime(logger) as runtime:
            self.model = runtime.deserialize_cuda_engine(f.read())
        self.bindings = OrderedDict()
        self.inshape = []
        for index in range(self.model.num_bindings):
            name = self.model.get_binding_name(index)
            dtype = trt.nptype(self.model.get_binding_dtype(index))
            shape = tuple(self.model.get_binding_shape(index))
            self.bindings[name] = Binding(name, dtype, shape, None, None)
        
        self.binding_addrs = OrderedDict((n, d.ptr) for n, d in self.bindings.items())
        self.context = self.model.create_execution_context()
        self.context.active_optimization_profile = 0 ####### 动态尺寸必须加 ######
        self.keys = list(self.bindings.keys())
        self.device = device
    
    def set_shape(self, shapes_in, shapes_out): ####### 需要在第一次读入之前确定大小，假设输入形状和输出形状都是列表 ######
        for i in range(len(shapes_in)):
            self.context.set_binding_shape(i, shapes_in[i]) ######## 输入尺寸必须明确，这里假设每个任务图片大小都一样 #######         
            self.inshape.append(shapes_in[i])

        outfrom = len(shapes_in)
        for j in range(len(shapes_out)):
            key = self.keys[outfrom + j]
            syn = torch.from_numpy(np.ones(shapes_out[j], dtype=np.dtype(self.bindings[key].dtype))).cuda()
            self.bindings[key] = self.bindings[key]._replace(data=syn)
            self.bindings[key] = self.bindings[key]._replace(ptr=int(syn.data_ptr()))
            self.binding_addrs[key] = int(syn.data_ptr())  
    
    def forward(self, im):
        if isinstance(im, list):
            for i in range(len(im)):
                ins = im[i].contiguous().cuda()
                self.binding_addrs[self.keys[i]] = int(ins.data_ptr())
                outfrom = i + 1
        else:
            im = im.contiguous().cuda()
            self.binding_addrs[self.keys[0]] = int(im.data_ptr())
            outfrom = 1

        self.context.execute_v2(list(self.binding_addrs.values()))
        if len(self.keys) == outfrom + 1:
            return self.bindings[self.keys[outfrom]].data.to(self.device)
        else:
            res = []
            for i in range(outfrom, len(self.keys)):
                res.append(self.bindings[self.keys[i]].data.to(self.device))
            return res