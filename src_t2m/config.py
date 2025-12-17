import argparse

parser = argparse.ArgumentParser(description='T2M model config')
############################ basic_setings ########################
parser.add_argument('--base_dir', type=str, default="/apdcephfs_cq11/share_1567347/share_info/kleinhe/checkpoints/text2motion", help="参数保存路径")
parser.add_argument('--task_name', type=str, default="", help="模型名,扩展文件夹名")
parser.add_argument('--data_root', type=str, default="/apdcephfs_cq11/share_1567347/share_info/kleinhe/motionDataset/HumanML3D", help="训练数据集文件夹")
parser.add_argument('--ft_path', type=str, default="", help="如果要从一个已有的结果开始 finetune, 模型路径, 也可以用于测试加载模型")
parser.add_argument('--model_mode', type=str, default="text", help="priorMDM 可控训练需要用不同的模型模式")
parser.add_argument('--rep', type=str, default="JP", help="表征类型")
parser.add_argument('--njoints', type=int, default=198, help=" 表征长度")
parser.add_argument('--ema', action="store_true", help="是否开启ema策略")
parser.add_argument('--mix_precision', action="store_true", help="是否开启混合精度训练")
parser.add_argument('--lr', type=float, default=1e-4, help="训练初始学习率")
parser.add_argument('--batch_size', type=int, default=64, help="训练集的 batch_size")
parser.add_argument('--log_interval', type=int, default=1000, help="输出log频率")
parser.add_argument('--save_interval', type=int, default=50000, help="保存频率")
parser.add_argument('--lr_anneal_steps', type=int, default=50000, help="学习率余弦退火周期")
parser.add_argument('--num_steps', type=int, default=300000, help="总训练次数")
parser.add_argument('--nframes', type=int, default=196, help="训练上限帧数")
parser.add_argument('--ngpu', type=int, default=1, help="每个机子上的卡数量")
parser.add_argument('--nhost', type=int, default=1, help="机器数量")
parser.add_argument('--gpu', type=str, default="A100", help="GPU类型, 决定精度")
######################### Diffusion settings #####################
parser.add_argument('--diffusion_steps', type=int, default=1000, help="Diffusion步数设置")
parser.add_argument('--guidance_param', type=float, default=2.5, help="CFG引导参数")
parser.add_argument('--noise_schedule', type=str, default="cosine", help="MDM 默认项，加噪方式")
parser.add_argument('--speed_loss_scale', type=float, default=0.0, help="速度损失缩放比例")
######################### evaluation settings #####################
parser.add_argument('--evaluate', action="store_true", help="是否进行测试任务")
parser.add_argument('--eval_during_training', action="store_true", help="是否在训练中验证，耗时会增加")
parser.add_argument('--eval_mode', type=str, default="wo_mm", choices=["wo_mm", "mm_shor", "full"], help="测试模式，见MDM")
parser.add_argument('--eval_batch_size', type=int, default=32, help="测试集的 batch_size, 固定32不变")
parser.add_argument("--evaluator_model_path", type=str, default="/apdcephfs_cq11/share_1567347/share_info/kleinhe/T2M", help="测试用embedding 模型地址")
parser.add_argument('--w_vectorizer_path', type=str, default="/apdcephfs_cq11/share_1567347/share_info/kleinhe/T2M/glove", help="词嵌入模型地址")
parser.add_argument('--evaluate_base_dir', type=str, default="dataset/t2m", help="测试用均值和方差目录")
########################### model settings #####################
parser.add_argument('--pos_encoder', type=str, default="rope", choices=["rope", "static"], help="mdm 默认static, 这个会影响推理帧数可扩展性")
parser.add_argument('--activation', type=str, default="swiglu", help="激活函数类型,mdm默认gelu,实际上效果可能差不太多")
parser.add_argument('--nlayers', type=int, default=8, help="模型层数")
parser.add_argument('--nheads', type=int, default=4, help="多头注意力")
parser.add_argument('--ff_size', type=int, default=1024, help="隐藏层大小")
parser.add_argument('--latent_dim', type=int, default=512, help="表征维度")
parser.add_argument('--cond_mask_prob', type=float, default=0.1, help="训练时mask掉输入帧的比例")
parser.add_argument('--frame_mask', type=float, default=0.25, help="训练时mask掉输入帧的比例")
parser.add_argument('--encode_full', action="store_true", help="是否开启卷积模块")
parser.add_argument("--word_tokens", action="store_true", help="是否开启cross-attention模块")
parser.add_argument("--clip_path", type=str, default="/apdcephfs_cq11/share_1567347/share_info/kleinhe/checkpoints/", help="CLIP 模型地址")

opt = parser.parse_args()

if __name__ == "__main__":
    import json
    args_dict = vars(opt)
    with open("temp.json", "w") as f:
        json.dump(args_dict, f, sort_keys=True, indent=2)