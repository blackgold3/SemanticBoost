import pynvml #导包
import time
import torch
import argparse
UNIT = 1024 * 1024

parser = argparse.ArgumentParser(description='commit our task')
############################ basic_setings ########################
parser.add_argument('--ngpu', type=int, default=1, help="how many gpu per host")
parser.add_argument('--threshold', type=int, default=1500, help="how many MB GPU initial")
opt = parser.parse_args()

pynvml.nvmlInit() #初始化
handles = []
aes = []
bes = []
for i in range(opt.ngpu):
    handle = pynvml.nvmlDeviceGetHandleByIndex(i)#获取GPU i的handle，后续通过handle来处理
    handles.append(handle)

    a = torch.randn(2048, 2048).to("cuda:{}".format(i))
    b = torch.randn(2048, 2048).to("cuda:{}".format(i))

    aes.append(a)
    bes.append(b)

while True:
    # gpuUtilRate = pynvml.nvmlDeviceGetUtilizationRates(handle).gpu
    for i in range(len(handles)):
        memoryInfo = pynvml.nvmlDeviceGetMemoryInfo(handles[i])
        memoryInfo = memoryInfo.used / UNIT
        if  memoryInfo < opt.threshold:
            aes[i] = torch.matmul(aes[i], bes[i])          ##### 没请求就后台模型推理
        else:   
            time.sleep(1)       ##### 有请求就休眠
