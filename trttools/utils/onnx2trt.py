import tensorrt as trt
import platform
import subprocess
import os

def valid(logger, engine, input_names, input_dims, opt):
    try:
        optcommand = []
        loc = 0
        for i in range(len(input_names)):
            name = input_names[i]
            dim = input_dims[i]
            begin = loc
            end = loc + dim
            curr_opt = opt[begin:end]
            optcommand.append("{}:{}x{}x{}x{}".format(name, *curr_opt))
            loc = end
            
        optcommand = ",".join(optcommand)
        valid_command = "trtexec --loadEngine={} --shapes={}".format(engine, optcommand)
        subprocess.call(valid_command, shell=platform.system() != 'Windows')
    except:
        logger.log(trt.Logger.Severity.ERROR, "== no correct engine model ==")

def onnx2trt(onnx, engine, fp16=False, int8=False, cache="temp.cache", datasets=None, pre_func=None,
            pre_args=None, max_calib_size=None, calib_batch_size=50,
            workspace=4, input_names=['inputs'], input_dims=[4], verbose=False,
            dynamic=False, minShapes=[1, 3, 224, 224], optShapes=[1, 3, 224, 224], maxShapes=[1, 3, 224, 224], 
            ):

    logger = trt.Logger()
    if verbose:
        logger.min_severity = trt.Logger.Severity.VERBOSE


    logger.log(trt.Logger.Severity.INFO, "============== fp16 quantization ==============")
    command = "trtexec --onnx={} --saveEngine={}".format(onnx, engine)

    if fp16:
        command += " --fp16"
    
    if verbose:
        command += " --verbose"
    
    if dynamic:
        mincommand = []
        optcommand = []
        maxcommand = []
        loc = 0
        for i in range(len(input_names)):
            name = input_names[i]
            dim = input_dims[i]
            begin = loc
            end = loc + dim 
            curr_min = minShapes[begin:end]
            curr_opt = optShapes[begin:end]
            curr_max = maxShapes[begin:end]
            
            curr_min = ("{}:" + ("{}x" * dim)[:-1]).format(name, *curr_min)
            curr_opt = ("{}:" + ("{}x" * dim)[:-1]).format(name, *curr_opt)
            curr_max = ("{}:" + ("{}x" * dim)[:-1]).format(name, *curr_max)

            mincommand.append(curr_min)
            optcommand.append(curr_opt)
            maxcommand.append(curr_max)
            loc = end

        mincommand = ",".join(mincommand)
        optcommand = ",".join(optcommand)
        maxcommand = ",".join(maxcommand)
        command += " --minShapes={} --optShapes={} \
                --maxShapes={}".format(mincommand, optcommand, maxcommand)

    subprocess.call(command, shell=platform.system() != 'Windows')
