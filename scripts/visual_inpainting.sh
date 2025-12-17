python -m sample.visual_inpainting \
 --checkpoints /apdcephfs_cq11/share_1567347/share_info/kleinhe/checkpoints/text2motion/train_amass_noencode11_nospeed/model_300000.pth \
 --mode text --prompt "A person sits down on the floor." --nframes 120 --help_file results/nospeeed/backflip-pose.npy \
 --capture_begin 0 --capture_end 60 --target_begin 0 --target_end 60 --blend_size 20 \
 --target_dir results/inpainting --target_name inpainting_final