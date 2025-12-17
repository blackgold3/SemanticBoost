from src_t2m.mdm import MDM
from diffusion import gaussian_diffusion as gd
from diffusion.respace import SpacedDiffusion, space_timesteps, InpaintingGaussianDiffusion

def load_model_wo_clip(model, state_dict): 
    print("load model checkpoints without clip")

    missing_keys, unexpected_keys = model.load_state_dict(state_dict, strict=False)

    print(unexpected_keys)

    other_miss = []
    for key in missing_keys:
        if not key.startswith('clip_model.'):
            other_miss.append(key)

    print(other_miss)
    assert all([k.startswith('clip_model.') for k in missing_keys])

def create_model_and_diffusion(args, mode="text"):
    if mode == "nocond":
        model = MDM(args, "nocond")
    else:
        model = MDM(args, "text")
    diffusion = create_gaussian_diffusion(args, mode)
    return model, diffusion


def create_gaussian_diffusion(args, mode="text"):
    # default params
    predict_xstart = True  # we always predict x_start (a.k.a. x0), that's our deal!
    steps = args.diffusion_steps
    scale_beta = 1.  # no scaling
    timestep_respacing = ''  # can be used for ddim sampling, we don't use it.
    learn_sigma = False
    rescale_timesteps = False
    sigma_small = True

    betas = gd.get_named_beta_schedule(args.noise_schedule, steps, scale_beta)
    loss_type = gd.LossType.MSE

    if not timestep_respacing:
        timestep_respacing = [steps]

    if mode.startswith("ft_control"):
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
                if not sigma_small
                else gd.ModelVarType.FIXED_SMALL
            )
            if not learn_sigma
            else gd.ModelVarType.LEARNED_RANGE
        ),
        loss_type=loss_type,
        rescale_timesteps=rescale_timesteps,
        # speed_loss_scale=args.speed_loss_scale
    )