from copy import deepcopy
import torch
import pandas as pd
import numpy as np

def pad_sample_with_zeros(sample, max_len=250):
    # pad inp, change lenghts, and pad is transition
    seq_len, n_feats = sample.shape
    len_to_pad = max_len - seq_len
    np.zeros_like(sample)
    sample_padding = np.zeros((len_to_pad, n_feats))
    sample = np.concatenate((sample, sample_padding))
    return sample

def split2subs(motions, step_sizes, batch_size, blend_len, max_motion_length):
    #### motions [1, 263, 1, Nlength] -> [263, Nlength] -> [NLength, 263]
    new_motions = []
    new_lengths = []
    new_motions.append(pad_sample_with_zeros(motions[..., :step_sizes[0] - blend_len].squeeze().permute(1, 0).cpu().numpy(), max_motion_length))
    new_lengths.append(step_sizes[0] - blend_len)
    for i in range(1, batch_size-1):
        curr = pad_sample_with_zeros(motions[..., step_sizes[i-1]-blend_len:step_sizes[i]-blend_len].squeeze().permute(1, 0).cpu().numpy(), max_motion_length)
        new_motions.append(curr)
        new_lengths.append(step_sizes[i] - step_sizes[i-1])

    new_motions.append(pad_sample_with_zeros(motions[..., step_sizes[-1]-blend_len:].squeeze().permute(1, 0).cpu().numpy(), max_motion_length))
    new_lengths.append(step_sizes[-1]-step_sizes[-2]+blend_len)

    new_motions = np.stack(new_motions, axis=0)
    new_motions = torch.from_numpy(new_motions)
    new_lengths = np.stack(new_lengths, axis=0)
    new_lengths = torch.from_numpy(new_lengths).long()
    return new_motions, new_lengths


def unfold_sample_arb_len(sample, handshake_size, step_sizes, final_n_frames, model_kwargs):
    old_sample = deepcopy(sample)
    new_shape = list(old_sample.shape)
    new_shape[0] = 1
    new_shape[-1] = final_n_frames
    sample = torch.zeros(new_shape, dtype=sample.dtype, device=sample.device)
    sample[0, :, :, :model_kwargs['y']['lengths'][0]] = old_sample[0, :, :, :model_kwargs['y']['lengths'][0]]
    for sample_i, len_i in enumerate(step_sizes):
        if sample_i == 0:
            continue
        start = step_sizes[sample_i-1]
        sample[0, :, :, start:len_i] = old_sample[sample_i, :, :, handshake_size:model_kwargs['y']['lengths'][sample_i]]
    return sample


def double_take_arb_len(diffusion, model, model_kwargs, n_frames, blend_len=10, handshake_size=20, device="cpu", progress=True):
    sample_fn = diffusion.p_sample_loop
    blend_len = blend_len
    handshake_size = handshake_size

    batch_size = len(model_kwargs['y']['text'])

    # Unfolding - orig
    sample = sample_fn(
        model,
        (batch_size, model.njoints, model.nfeats, n_frames),
        clip_denoised=False,
        model_kwargs=model_kwargs,
        skip_timesteps=0,  # 0 is the default value - i.e. don't skip any step
        init_image=None,
        progress=progress,
        dump_steps=None,
        noise=None,
        const_noise=False,
        unfolding_handshake=handshake_size,
    )

    model_kwargs['y']['scale'] = torch.ones(batch_size-1, device=device) * 0
    sample = sample["output"]       #### [5, 263, 1 196]

    '''
    1. 替换 sample 
    2. model_kwargs['y']['lengths']
    '''

    new_sample_seq_len = (sample.shape[-1] - 2 * handshake_size) * 2 + handshake_size

    bs, feats, joints, seq_len = sample.shape
    new_sample = torch.zeros((bs-1, feats, joints, new_sample_seq_len), dtype=sample.dtype, device=sample.device)

    generated_motion = []
    right_constraint = []
    left_constraint = []

    for ii in range(bs):        #####  按左中右拆分 Motion
        generated_motion.append(deepcopy(sample[ii, :, :, handshake_size: model_kwargs['y']['lengths'][ii]-handshake_size])) # w/o start and end
        left_constraint.append(deepcopy(sample[ii, :, :, :handshake_size]))  # left side
        right_constraint.append(deepcopy(sample[ii, :, :, model_kwargs['y']['lengths'][ii] - handshake_size: model_kwargs['y']['lengths'][ii]]))

    buffer = []     #### 存放剩下的动作部分的长度，也就是 generated_motion 的长度
    for ii in range(bs):
        buffer.append(int(model_kwargs['y']['lengths'][ii]) - 2*handshake_size)
    for ii in range(bs - 1):  # run over bs, 把 N句话 合并成 N-1 句话，新 motion 的组成 [gm[i-1], right[i-1], gm[i]], 长度是 2 * gm_length + hand_size
        new_sample[ii, :, :, :buffer[ii]] = generated_motion[ii]
        new_sample[ii, :, :, buffer[ii]: buffer[ii]+handshake_size] = right_constraint[ii] # add transition
        new_sample[ii, :, :, buffer[ii]+handshake_size : buffer[ii]+handshake_size+buffer[ii+1]] = generated_motion[ii + 1]

    # "in between"
    model_kwargs['y']['inpainted_motion'] = new_sample
    model_kwargs['y']['inpainting_mask'] = torch.ones_like(new_sample, dtype=torch.float,
                                                            device=new_sample.device)

    for ii in range(bs - 1):  # run over bs
        if blend_len >= 2:
            '''
            渐变混合
            1. 在左边 gm[i-1] 靠后 blend_len 的区域，渐变地保留原本的内容
            2. 在右边 gm[i] 靠前的 blend_len 的区域，渐变保留原本的内容
            3. 似乎是 right 的部分完全保留，也就是用前一个动作的结束座位后一个动作的开头 
            '''

            model_kwargs['y']['inpainting_mask'][ii, :, :, buffer[ii] - blend_len: buffer[ii]] = \
                torch.arange(0.85, 0.0, -0.85 / int(blend_len))
            model_kwargs['y']['inpainting_mask'][ii, :, :, buffer[ii] + handshake_size: buffer[ii] + handshake_size + blend_len] = \
                torch.arange(0.0, 0.85, 0.85 / int(blend_len))

    model_kwargs['y']['uncond'] = 1.0       ### 混合多段语意后，cond 没什么意义，而且需要生成的内容很少
    model_kwargs['y']['text'] = model_kwargs['y']['text'][:bs-1]
    sample_fn = diffusion.p_sample_loop  # double take sample function
    n_frames = new_sample_seq_len
    orig_lens = deepcopy(model_kwargs['y']['lengths'])
    for ii in range (len(model_kwargs['y']['lengths'])-1):
        model_kwargs['y']['lengths'][ii] = model_kwargs['y']['lengths'][ii] + model_kwargs['y']['lengths'][ii+1] - 3*handshake_size
    model_kwargs['y']['lengths'] = model_kwargs['y']['lengths'][:-1]

    double_take_sample = sample_fn(
        model,
        (batch_size-1, model.njoints, model.nfeats, n_frames),
        clip_denoised=False,
        model_kwargs=model_kwargs,
        skip_timesteps=0,  # 0 is the default value - i.e. don't skip any step
        init_image=new_sample, #TODO!! check if plausible or not!
        progress=progress,
        dump_steps=None,
        noise=None,
        const_noise=False,
    )
    double_take_sample = double_take_sample["output"]
    model_kwargs['y']['lengths'] = orig_lens
    # rebuild_orig:
    rebuild_sample = torch.zeros_like(sample)

    '''
    sample -> left + motion + right
    double_take_sample -> motion1 + blend + hand + blend + motion2, 其中长度表示 : motion1 + blend = motion2 + blend = motion
    '''

    transitions, right_side, left_side = [], [], []
    for ii in range(bs - 1):  # run over bs
        transitions.append(double_take_sample[ii, :, :, buffer[ii]: buffer[ii]+handshake_size])
        right_side.append(double_take_sample[ii, :, :, buffer[ii] + handshake_size: buffer[ii] + handshake_size + blend_len]) # M1 blending..
        left_side.append(double_take_sample[ii, :, :, buffer[ii] - blend_len:buffer[ii]]) # M0 blending...

        '''
        translation 储存的是 hand
        right_side 存右边的 blend
        left_side 村左边的 blend
        '''


    rebuild_sample[0, :, :, :handshake_size] = left_constraint[0] # Fill missing
    rebuild_sample[-1, :, :, buffer[-1]+handshake_size: buffer[-1]+2*handshake_size] = right_constraint[-1] # Fill missing

    '''
    展开 double take 的结果, 还原会原本的状态,即 left + motion + right
    '''

    for ii in range(bs - 1):
        rebuild_sample[ii + 1, :, :, :handshake_size] = transitions[ii]
        rebuild_sample[ii, :, :, handshake_size: buffer[ii]+handshake_size] = generated_motion[ii]
        rebuild_sample[ii, :, :, buffer[ii]+handshake_size: buffer[ii]+2*handshake_size] = transitions[ii]      #### motion1 的 right = motion2 的 left
        rebuild_sample[ii, :, :, handshake_size + buffer[ii]-blend_len: handshake_size + buffer[ii]] = left_side[ii]
        # if ii > 0:
    rebuild_sample[-1, :, :, handshake_size: buffer[-1] + handshake_size] = generated_motion[-1]
    for ii in range(bs - 1):
        rebuild_sample[ii+1, :, :, handshake_size:handshake_size + blend_len] = right_side[ii]

    double_take_sample = deepcopy(rebuild_sample)

    return double_take_sample

def double_take(prompt=None, path=None, num_repetitions=1, model=None, diffusion=None, handshake_size=20, blend_len=10, default_length=196, guidance_param=2.5, lengths="120", device="cpu", progress=True):
    assert model is not None
    assert diffusion is not None
    if prompt is not None:
        texts = prompt.split("|")
        lengths = lengths.split("|")
        num_samples = len(texts)
        length = []
        captions = []
        for i in range(len(texts)):
            if i < len(lengths):
                try:
                    nframes = int(lengths[i])
                except:
                    nframes = default_length
            else:
                nframes = default_length

            curr_text = texts[i]

            captions.append(curr_text)
            length.append(nframes)

        model_kwargs = {'y': {
            'mask': torch.ones((len(texts), 1, 1, default_length)), # 196 is humanml max frames number
            'lengths': torch.tensor(length),
            'text': captions,
            'tokens': [''],
            'scale': torch.ones(len(texts))*guidance_param
        }}
    elif path.split(".")[-1] == "csv":
        df = pd.read_csv(path)
        num_samples = len(list(df['text']))  
        model_kwargs = {'y': {
            'mask': torch.ones((len(list(df['text'])), 1, 1, default_length)), #196 is humanml max frames number
            'lengths': torch.tensor(list(df['length'])),
            'text': list(df['text']),
            'tokens': [''],
            'scale': torch.ones(len(list(df['text'])))*guidance_param
        }}  
    elif path.split(".")[-1] == "txt":
        with open(path, 'r') as fr:
            texts = fr.readlines()
        texts = [s.replace('\n', '') for s in texts]
        num_samples = len(texts)      
        model_kwargs = {'y': {
            'mask': torch.ones((len(texts), 1, 1, default_length)), # 196 is humanml max frames number
            'lengths': torch.tensor([default_length]*len(texts)),
            'text': texts,
            'tokens': [''],
            'scale': torch.ones(len(texts))*guidance_param
        }}

    all_motions = []

    for rep_i in range(num_repetitions):
        if guidance_param != 1:
            model_kwargs['y']['scale'] = torch.ones(num_samples, device=device) * guidance_param
        model_kwargs['y'] = {key: val.to(device) if torch.is_tensor(val) else val for key, val in model_kwargs['y'].items()}

        max_arb_len = model_kwargs['y']['lengths'].max()
        min_arb_len = 2 * handshake_size + 2*blend_len + 10

        for ii, len_s in enumerate(model_kwargs['y']['lengths']):
            if len_s > max_arb_len:
                model_kwargs['y']['lengths'][ii] = max_arb_len
            if len_s < min_arb_len:
                model_kwargs['y']['lengths'][ii] = min_arb_len

        sample = double_take_arb_len(diffusion, model, model_kwargs, max_arb_len, blend_len, handshake_size, device, progress=progress)    
        step_sizes = np.zeros(len(model_kwargs['y']['lengths']), dtype=int)
        for ii, len_i in enumerate(model_kwargs['y']['lengths']):
            if ii == 0:
                step_sizes[ii] = len_i
                continue
            step_sizes[ii] = step_sizes[ii-1] + len_i - handshake_size

        final_n_frames = step_sizes[-1]
        sample = unfold_sample_arb_len(sample, handshake_size, step_sizes, final_n_frames, model_kwargs)

        all_motions.append(sample)
    
    all_motions = torch.cat(all_motions, dim=0)
    return all_motions, step_sizes

      