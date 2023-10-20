from motion.model.mdm import MDM
from motion.diffusion import gaussian_diffusion as gd
from motion.diffusion.respace import SpacedDiffusion, space_timesteps, InpaintingGaussianDiffusion

def load_model_wo_clip(model, state_dict):
    print("load model checkpoints without clip")
    missing_keys, unexpected_keys = model.load_state_dict(state_dict, strict=False)

    print(unexpected_keys)
    assert all([k.startswith('clip_model.') for k in missing_keys])

def load_ft_model_wo_clip(model, state_dict):
    print("load model checkpoints without clip")
    missing_keys, unexpected_keys = model.load_state_dict(state_dict, strict=False)
    print(unexpected_keys)

    # for name, value in model.named_parameters():
    #     if "seqTransEncoder" in name and "self_attn" in name:
    #         value.requires_grad = False
    #     if name.startswith("code_full") or name.startswith("encode_compress") or name.startswith("input_process"):
    #         value.requires_grad = False

    assert all([k.startswith('clip_pose_encoder.') for k in unexpected_keys])
    # assert all([k.startswith('clip_model.') or k.startswith('clip_pose_encoder.') or k.startswith('embed_text.')  for k in missing_keys])

def create_model_and_diffusion(args, mode="text", json_dict=None):
    model = MDM(**get_model_args(args), json_dict=json_dict)
    diffusion = create_gaussian_diffusion(args, mode)
    return model, diffusion

def get_model_args(args):
    # default args
    clip_version = 'ViT-B/32'
    if args.unconstrained:
        cond_mode = 'no_cond'
    elif args.dataset in ['kit', 'humanml']:
        cond_mode = "text"
     
    if args.arch in ["refined_encoder", "refined_decoder"]:
        activation = "swiglu"
    else:
        activation = "gelu"

    if args.dataset == 'humanml':
        njoints = 263
        nfeats = 1
    elif args.dataset == 'kit':
        njoints = 251
        nfeats = 1

    if args.rep == "smr":
        njoints += 6
        nfeats = 1

    return {'njoints': njoints, 'nfeats': nfeats, 'latent_dim': args.latent_dim, 'ff_size': args.ff_size, 'num_layers': args.layers, 'num_heads': args.heads,
            'dropout': 0.1, 'activation': activation, 'cond_mode': cond_mode, 'cond_mask_prob': args.cond_mask_prob, 'arch': args.arch,
            'clip_version': clip_version, 'dataset': args.dataset, "local":args.local, "encode_full":args.encode_full, "txt_tokens":args.txt_tokens,
            "num_frames":args.num_frames, "frame_mask":args.frame_mask}


def create_gaussian_diffusion(args, mode="text"):
    # default params
    predict_xstart = True  # we always predict x_start (a.k.a. x0), that's our deal!
    steps = 1000
    scale_beta = 1.  # no scaling
    timestep_respacing = ''  # can be used for ddim sampling, we don't use it.
    learn_sigma = False
    rescale_timesteps = False

    betas = gd.get_named_beta_schedule(args.noise_schedule, steps, scale_beta)
    loss_type = gd.LossType.MSE

    if not timestep_respacing:
        timestep_respacing = [steps]

    if mode is not None and (mode.startswith("finetune_control") or mode == "control_length"):
        print(">>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>  inpainting diffusion model")
        diffusion = InpaintingGaussianDiffusion
    else:
        print(">>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>  SpacedDiffusion")
        diffusion = SpacedDiffusion

    return diffusion(
        use_timesteps=space_timesteps(steps, timestep_respacing),
        betas=betas,
        model_mean_type=(
            gd.ModelMeanType.EPSILON if not predict_xstart else gd.ModelMeanType.START_X
        ),
        model_var_type=(
            (
                gd.ModelVarType.FIXED_LARGE
                if not args.sigma_small
                else gd.ModelVarType.FIXED_SMALL
            )
            if not learn_sigma
            else gd.ModelVarType.LEARNED_RANGE
        ),
        loss_type=loss_type,
        rescale_timesteps=rescale_timesteps,
        rep=args.rep
    )