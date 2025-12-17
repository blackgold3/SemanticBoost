python -m sample.visual_control --cond_files "results/nospeeed/wf-pose.npy" "results/cond_npys/rightfoot.npy" \
 --indexs 0 0 0 0 0 0 1 --out_format mixture \
 --checkpoints /apdcephfs_cq11/share_1567347/share_info/kleinhe/checkpoints/text2motion/train_amass_randR_condition/model_300000.pth \
 --nframes 160 --target_dir results/condition --target_name wf-rfoot