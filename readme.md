# 基于Diffusion的动作生成模型

## 提交任务

[提交任务](commit.py)

- mode = mdm ：Text2motion 任务
- mode = empty: 占卡任务 

## Text2Motion

[T2M模型训练和可视化](src_t2m/readme.md)

[T2M数据读取](data_loaders/split_read_loader.py)

## 可视化

[数据可视化](sample/visual_file.py)

[T2M可视化](sample/visual_t2m.py)

[T2M的inpainting编辑任务](sample/visual_inpainting.py)

[T2M的inpainting动作衔接](scripts/sever_motion_concat_inpainting.py)

[priorMDM的DoubleTake](sample/visual_doubleTake.py)
