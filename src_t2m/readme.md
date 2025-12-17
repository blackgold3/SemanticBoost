# Text2Motion 基座模型

## 数据

> 地址

/apdcephfs_cq11/share_1567347/share_info/kleinhe/motionDataset

> 子数据集说明

总计 152.63 h

统一处理到 fps = 20 

帧数 10989396

- HumanML3D: HumanML3D 的前 14615 个数据
- Amass: HumanML3D 的全部数据
- 100Style: locomotion 数据，不同风格的走跑跳
- Mixamo: https://www.mixamo.com/#/
- Human3.6: Action 数据集, 基于规则将 action 扩展成文本句子

> 子文件夹说明

- full_pose_data: SMPL rotation 数据, 最后三维表示 Translation 
- joints: 3D SMPL 关节点
- new_joint_vecs: MDM 数据集 HumanML3D 的 263 维表征
- smr_rep: 对 263 维表征的调整, 269 维度的混合表征
- jp_rep: 关键点和欧拉角的混合表征

## 训练接口

> 全部参数

- src_t2m/config.py 包括相关的中文说明, 大部分参数可以不改, 经常改的参数写到提交任务部分
- 个人使用需要修改 base_dir 改变模型保存的总路径

> 提交参数: commit.py

- 提交参数

    - task_name: 提交任务的名字和保存参数路径的子路径名, 太极规定任务名最好以 "train", "evaluate", "server" 开头,不容易被清理
    - gpu: GPU 类型
    - rank: low 或者 high 高优卡和弹性卡
    - mode: 目前仅支持 mdm, text2motion 任务
    - evaluate: 测评任务还是训练任务
    - 在 mode=mdm 区域内修改其他常改配置信息

> 提交任务脚本示例

```sh
### 训练任务, 太极上提交一个4卡任务，A100弹性卡4张，固定参数包括用户ID，集群等在 commit.py 里直接修改 ####
python commit.py -t train_t2m_4datasets -m mdm -g A100 -r LOW --ngpu 4 --nhost 1

### 测评任务, 需要用 ft_path 指定参数路径, 选择 eval_mode ####
python commit.py -t evaluate_t2m -m mdm -e -g A100 -r LOW --ngpu 4 --nhost 1
```

## 可视化

> 文件可视化

```sh
python -m sample.visual_file --source temp.npy --mode pose --target_dir results/files --target_name test --render_mode blender --out_mode video

# 参数说明：
# 1. source 文件地址
# 2. mode: pose | joints | t2m | smr 对应动作的不同表征，可视化方法不太一样
# 3. target_dir + target_name: 结果文件名
# 4. render_mode:  pyrender 或者 blender
# 5. out_mode: sequence 论文里的动作插图（仅限于blender） 或者 video 生成 mp4
# 6. model_path： SMPL 模型地址， 固定项
# 7. blender_path： blender linux 包地址， 固定项
# 8. 注释： blender 渲染 mesh 的颜色在 sample/blender/render_blender/blender/meshes.py 里改, sequence 改 53 行的 cmap, video 改 GEN_SMPL
```

> T2M模型生成

```sh
python -m sample.visual_t2m --prompt "A person walks backwards" --checkpoints /apdcephfs_cq11/share_1567347/share_info/kleinhe/checkpoints/text2motion/train_4alldata_text2motion_smr/S300000_F0.2772_T0.5476.pth --nframes 100 --target_dir results/t2m --target_name jump --save --export

# 参数说明：
# 1. prompt： 输入文本条件
# 2. checkpoints： 模型的路径，模型配置将会从该路径下 motionMonitor/version_0 读取, pytorch_lightning 保存的模型参数
# 3. nframes： 模型生成长度
# 4. help_file： 在 priorMDM 的 ft_control 模式中输入的数据文件，从中提取控制信号，轨迹，肢体，部分帧等
#   4.1 如果 checkpoints 对应的 model_mode 没有设置，此处无效
#   4.2 help_file 的大小最好和 nframes 一致，否则部分控制可能达不到预期
#   4.3 相关控制的 mask 都在 dataloader/control_mask 中设置
# 5. save: 是否保存生成的表征文件
# 6. export: 是否导出 fbx 文件，这里用一个 smpl 的 fbx 作为基础骨架，把旋转和平移数据迁移上去，作为一个驱动任务，不需要 retargeting
# 7.其他参数： 同文件可视化 部分
```

> 无 fine-tune inpainting 任务可视化

**任务说明**：输入一个 help_file，保留这个文件的指定部分，把这一部分放在生成动作的指定部分。生成动作的其他部分可以无条件生成或者 text2motion 生成, 这一部分实际上可以合并到 text2motion 可视化中, 为了功能明确单一, 简单可扩展, 单独另开一个文件。text2motion 部分将支持 ft_control 的功能。

```py
python -m sample.visual_inpainting --checkpoints /apdcephfs_cq11/share_1567347/share_info/kleinhe/checkpoints/text2motion/train_4alldata_text2motion_smr/S300000_F0.2772_T0.5476.pth --mode nocond --nframes 60 --help_file results/files/help_file.npy --capture_begin 50 --capture_end 60 --target_begin 50 --target_end 60 --blend_size 10 --target_dir results/inpainting --target_name inpainting_final

# 参数说明
# 1. checkpoints: 模型参数，必要
# 2. mode: text | nocond, 如果 nocond 无需输入 prompt，随机生成 inpainting 外的部分，如果 text, 需要输入 prompt 信息
# 3. help_file: 有一定长度的动作文件，用于提取 inpainting 的部分
# 4. capture_begin: help_file 的截取起点，列表，可以输入多组 capture
# 5. capture_end: help_file 的截取终点，列表，对应 capture_begin 的长度
# 6. target_begin: 截取的动作要保留在生成动作的区间起点, 列表，对应 capture_begin 的多组 capture, 也要有多组 target
# 7. target_end: 截取的动作要保留在生成动作的区间终点, 列表，对应 target_end 的长度
# 8. blend_size: 在保留区间的前面和后面各有一段 混合区域，用于自然过渡，这段区域的长度。一般来说 help_file 的长度应该大于等于 (capture_end - capture_begin + 2 * blend_size), 保留区间放在起点和终点部分的动作可以少一个 blend_size 区域。
# 9. 其他参数同其他可视化
```

> 无 fine-tune doubleTake 长动作生成

**任务说明**：doubleTake 第一阶段需要生成尾部衔接区域，所以不能直接用于动作融合，所以只能用于生成长动作

```py
python -m sample.visual_doubleTake --prompt "A person walks backwards" "A person runs forward" "A person does a handstand" --nframes 100 60 100 --handshake_size 20 --blend_size 10 --checkpoints /apdcephfs_cq11/share_1567347/share_info/kleinhe/checkpoints/text2motion/train_4alldata_text2motion_smr/S150000_F0.2864_T0.5159.pth --target_dir results/t2m --target_name doubletake3

# 参数说明:
# 1. prompts：输入一个文本列表
# 2. nframes: 输入文本列表对应的生成长度列表，列表大小和文本列表必须一致
# 3. handshake_size: 两段动作的衔接区域长度
# 4. blend_size: 两段动作到衔接区域的过渡区域长度
# 5. 其他参数同其他可视化接口
```
