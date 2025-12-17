import json
import argparse
import platform
import subprocess
import json
parser = argparse.ArgumentParser(description='commit our task')
############################ basic_setings ########################
parser.add_argument('-t', '--task_name', type=str, default="test", help="save dir and commit name")
parser.add_argument('-g', '--gpu', type=str, default="A100")
parser.add_argument('-r', '--rank', type=str, default="HIGH")
parser.add_argument('-l', '--local', action="store_true")
parser.add_argument('-m', '--mode', type=str, default="mdm")
parser.add_argument('-e', '--evaluate', action="store_true")
########################## commit setting #############################
parser.add_argument('--ngpu', type=int, default=1, help="how many gpu per host")
parser.add_argument('--nhost', type=int, default=1, help="how many host")
opt = parser.parse_args()

if opt.local:
    train_batch = 32
    lr = 1e-4
    extension = ""
    run_gpu = ""
else:
    train_batch = 512 // (opt.ngpu * opt.nhost)    
    lr = 3e-4 / ((opt.ngpu * opt.nhost) ** 0.5)
    extension = " --mix_precision"    
    run_gpu = "python empty.py --ngpu {} --threshold {}".format(opt.ngpu, 1500)

if opt.mode == "mdm":
    #################### 基本信息 #################
    data_root = "/apdcephfs_cq11/share_1567347/share_info/kleinhe/motionDataset/train_amass_JP_fullspeed"
    ft_path = "11"
    eval_mode = "wo_mm"
    model_mode = "text" #### inpainting 训练需要
    ################### 结构设置 ####################
    base_model_structure = "--word_tokens --ema"
    #################### 是否验证 #####################
    valid_cmd = ""
    # valid_cmd = "--eval_during_training"
    #################### 训练次数，finetune 时需要修改配置 ###########
    log_interval = 1000
    save_interval = 50000   ### 存储频率
    lr_anneal_steps = 50000 ### 余弦退火 shedual 循环周期
    num_steps = 300000      ### 总训练次数
    #################### 速度损失缩放比例 #################
    njoints = 393
    nlayers = 11
    speed_loss_scale = 0
    #################### 命令 ###################
    if opt.evaluate:
        start_cmd = "python -m src_t2m.pl_model --evaluate --frame_mask 0 --eval_mode {} ".format(eval_mode)
    else:
        start_cmd = "python -m src_t2m.pl_model {} --batch_size {} --lr {} --speed_loss_scale {} --log_interval {} --save_interval {} --lr_anneal_steps {} --num_steps {} {} ".format(valid_cmd, train_batch, lr, speed_loss_scale, log_interval, save_interval, lr_anneal_steps, num_steps, extension)

    start_cmd += "--gpu {} --data_root {} --task_name {} --ft_path {} --ngpu {} --nhost {} --model_mode {} --nlayers {} --njoints {} {}".format(opt.gpu, data_root, opt.task_name, ft_path, opt.ngpu, opt.nhost, model_mode, nlayers, njoints, base_model_structure)

elif opt.mode == "empty":
    run_gpu = ""
    start_cmd = "python empty.py --ngpu {} --threshold {}".format(opt.ngpu, 1500)    

with open("start.sh", "w") as f:
    f.write(start_cmd + " \n")
    f.write(run_gpu + " \n")

if not opt.local:
    jizhi_json = {
        "readable_name": opt.task_name,
        "Token": "xxx",
        "business_flag": "xxx",
        "priority_level": opt.rank,
        "elastic_level": 1, 
        "GPUName": opt.gpu,
        "host_num": opt.nhost,
        "host_gpu_num": opt.ngpu,
        "model_local_file_path": "/data/TTA/motionWE",
        "image_full_name": "mirrors.tencent.com/kleinhe/tta:sft",
        "cuda_version": "11.0",
        "start_cmd":"./start.sh",
        "mount_ceph_business_flag":"AILab_Game_Report_PaaS",
        "envs": {
            "HUNYUAN_TASK_CATEGORY": "TEXT23D",
            "HUNYUAN_TASK_DESCRIPTION": "motion生成训练",
            "HUNYUAN_RESOURCE_USAGE": "experiment",
            "HUNYUAN_BASE_MODEL": "33M-dense-200",
            "HUNYUAN_OUTPUT_BASE_MODEL": "33M-dense-200"
        }
    }

    with open("jizhi.json", "w") as f:
        json.dump(jizhi_json, f, indent=2)

    command_start = "chmod 777 start.sh"
    subprocess.call(command_start, shell=platform.system() != 'Windows')

    command = "jizhi_client start -scfg ./jizhi.json"
else:
    command = "bash start.sh"

subprocess.call("mv results ../", shell=platform.system() != 'Windows')
subprocess.call(command, shell=platform.system() != 'Windows')
subprocess.call("mv ../results ./", shell=platform.system() != 'Windows')