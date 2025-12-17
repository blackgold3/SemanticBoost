 python -m sample.visual_t2m --prompt "A person walks forward." --out_format pose \
 --checkpoints /apdcephfs_cq11/share_1567347/share_info/kleinhe/checkpoints/text2motion/train_amass_noencode11_nospeed/model_300000.pth \
 --nframes 120 --target_dir results/nospeeed --target_name wf-pose --save
