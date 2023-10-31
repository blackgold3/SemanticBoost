import torch
from torch import nn
from trttools import DynamicModel
from motion.model import clip
from motion.model.base_transformer import Attention
from motion.model.layer_norm_fp16 import LayerNorm
from collections import OrderedDict

class MotionGenerator(nn.Module):
    def __init__(self, input_process, output_process, code_full, seqTransEncoder, embed_timestep, encode_compress, positional_embedding=None, txt_tokens=1, condition_compress=None):
        super(MotionGenerator, self).__init__()
        self.input_process = input_process
        self.embed_timestep = embed_timestep
        self.seqTransEncoder = seqTransEncoder
        self.output_process = output_process
        self.code_full = code_full
        self.encode_compress = encode_compress
        self.positional_embedding = positional_embedding

        self.txt_tokens = txt_tokens
        if self.txt_tokens == 2:
            self.condition_compress = condition_compress

    def forward(self, x, t, txt_emb):
        emb = self.embed_timestep(t)
        latent = self.code_full(x)
        current = self.input_process(x) 
        latent = latent.repeat(current.shape[0], 1, 1)
        current = torch.cat([current, latent], dim=2)
        current = self.encode_compress(current)

        emb = emb.repeat(txt_emb.shape[0], 1, 1)
        if self.txt_tokens == 2:
            concat_embedding = torch.cat([emb, txt_emb], dim=2)
            emb = self.condition_compress(concat_embedding)  
        elif self.txt_tokens == 1:
            emb += txt_emb

        xseq = torch.cat([emb[0:1], current], dim=0)

        if self.positional_embedding is not None:
            xseq = self.positional_embedding(xseq)

        word_tokens = emb[1::]
        output = self.seqTransEncoder(xseq, word_tokens=word_tokens)
        output = output[1:]
        output = self.output_process(output)
        return output

class QuickGELU(nn.Module):
    def forward(self, x: torch.Tensor):
        return x * torch.sigmoid(1.702 * x)

class ResidualAttentionBlock(nn.Module):
    def __init__(self, d_model: int, n_head: int, attn_mask: torch.Tensor = None):
        super().__init__()
        self.attn = Attention(d_model, n_head, bias=True)
        self.ln_1 = LayerNorm(d_model)
        self.mlp = nn.Sequential(OrderedDict([
            ("c_fc", nn.Linear(d_model, d_model * 4)),
            ("gelu", QuickGELU()),
            ("c_proj", nn.Linear(d_model * 4, d_model))
        ]))
        self.ln_2 = LayerNorm(d_model)
        self.attn_mask = attn_mask

    def attention(self, x: torch.Tensor):
        self.attn_mask = self.attn_mask.to(dtype=x.dtype, device=x.device) if self.attn_mask is not None else None
        return self.attn(x, x, x, mask=self.attn_mask)

    def forward(self, x: torch.Tensor):
        x = x + self.attention(self.ln_1(x))
        x = x + self.mlp(self.ln_2(x))
        return x

class Transformer(nn.Module):
    def __init__(self, width: int, layers: int, heads: int, attn_mask: torch.Tensor = None):
        super().__init__()
        self.width = width
        self.layers = layers
        self.resblocks = nn.Sequential(*[ResidualAttentionBlock(width, heads, attn_mask) for _ in range(layers)])

    def forward(self, x: torch.Tensor):
        return self.resblocks(x)


class TextProcess(nn.Module):
    def __init__(self, embed_text, clip_model):
        super(TextProcess, self).__init__()
        self.embed_text = embed_text
        self.token_embedding = clip_model.token_embedding
        self.positional_embedding = clip_model.positional_embedding
        self.ln_final = clip_model.ln_final
        self.text_projection = clip_model.text_projection
        
        # self.transformer = clip_model.transformer
        old_weight = clip_model.transformer.state_dict()
        new_weight = {}
        for key, value in old_weight.items():
            if "in_proj_weight" in key:
                keyq = key.replace("in_proj_weight", "wq.weight")
                keyk = key.replace("in_proj_weight", "wk.weight")
                keyv = key.replace("in_proj_weight", "wv.weight")
                inshape = value.shape[0] // 3
                valueq = value[:inshape]
                valuek = value[inshape:inshape * 2]
                valuev = value[inshape * 2:]
                new_weight[keyq] = valueq
                new_weight[keyk] = valuek
                new_weight[keyv] = valuev

            elif "in_proj_bias" in key:
                keyq = key.replace("in_proj_bias", "wq.bias")
                keyk = key.replace("in_proj_bias", "wk.bias")
                keyv = key.replace("in_proj_bias", "wv.bias")
                inshape = value.shape[0] // 3
                valueq = value[:inshape]
                valuek = value[inshape:inshape * 2]
                valuev = value[inshape * 2:]

                new_weight[keyq] = valueq
                new_weight[keyk] = valuek
                new_weight[keyv] = valuev
            elif "out_proj" in key:
                newkey = key.replace("out_proj", "wo")
                new_weight[newkey] = value
            else:
                new_weight[key] = value

        self.transformer = Transformer(width=512, layers=12, heads=8, attn_mask=self.build_attention_mask(77))
        self.transformer.load_state_dict(new_weight, strict=True)
 
    def build_attention_mask(self, context_length=77):
        # lazily create causal attention mask, with full attention between the vision tokens
        # pytorch uses additive attention mask; fill with -inf
        mask = torch.empty(context_length, context_length)
        mask.fill_(float("-inf"))
        mask.triu_(1)  # zero out the lower diagonal
        return mask

    def forward(self, txt_token):
        x = self.token_embedding(txt_token)  # [batch_size, n_ctx, d_model]
        x = x + self.positional_embedding
        x = x.permute(1, 0, 2)  # NLD -> LND
        x = self.transformer(x)
        x = x.permute(1, 0, 2)  # LND -> NLD
        x = self.ln_final(x)
        clip_feature = x[torch.arange(x.shape[0]), txt_token.argmax(dim=-1)] @ self.text_projection
        clip_feature = clip_feature.unsqueeze(1)
        clip_feature = torch.cat([clip_feature, x], dim=1)     #### [bs, T, 512]
        clip_feature = self.embed_text(clip_feature)
        return clip_feature    

class TRT_MDM(nn.Module):
    def __init__(self, mode, json_dict, device="cuda"):
        super(TRT_MDM, self).__init__()
        self.device = device
        self.json_dict = json_dict
        self.clip_model = DynamicModel(self.json_dict[f"{mode}2"], self.device)
        self.decoder = DynamicModel(self.json_dict[f"{mode}1"], self.device)
        self.num_frames = 196
        self.njoints = 269
        self.nfeats = 1
        self.condition_length = 77

    def mask_cond(self, cond, force_mask=False):
        bs = cond.shape[0]
        if force_mask:
            return torch.zeros_like(cond)
        else:
            return cond

    def clip_text_embedding(self, raw_text):
        default_context_length = self.condition_length
        texts = clip.tokenize(raw_text, context_length=default_context_length, truncate=True) # [bs, context_length] # if n_tokens > context_length -> will truncate
        texts = texts.to(self.device)

        if len(self.clip_model.inshape) == 0 or self.clip_model.inshape[0] != texts.shape:
            self.clip_model.set_shape([[*texts.shape]], [[texts.shape[0], self.condition_length+1, 512]]) 

        clip_feature = self.clip_model(texts)
        return clip_feature

    @torch.no_grad()
    def forward(self, x, timesteps, y=None): 
        force_mask = y.get('uncond', False)
        txt_emb = self.clip_text_embedding(y['text'])      ### MASK_COND 会按照一定的比例把 batch_size 中的一部分文本句整句换成 [0, 0, ... 0]
        txt_emb = self.mask_cond(txt_emb, force_mask=force_mask)
        
        if len(txt_emb.shape) == 3:
            txt_emb = txt_emb.permute(1, 0, 2)
        else:
            txt_emb = txt_emb.unsqueeze(0)

        real_frame = x.shape[-1]
        if real_frame < self.num_frames:
            extension = torch.zeros([x.shape[0], x.shape[1], x.shape[2], self.num_frames - x.shape[-1]], device=x.device, dtype=x.dtype)
            x = torch.cat([x, extension], dim=-1)

        if len(self.decoder.inshape) == 0 or self.decoder.inshape[0] != x.shape:
            self.decoder.set_shape([[*x.shape], [*timesteps.shape], [*txt_emb.shape]], [[*x.shape]])

        output = self.decoder([x, timesteps, txt_emb])
        output = output[:, :, :, :real_frame]

        return {"output":output}


if __name__ == "__main__":
    from trttools import dynamic_float16, dynamic_float32
    from motion.model_util import load_model_wo_clip, create_model_and_diffusion
    import json
    from argparse import Namespace, ArgumentParser
    import os
    parser = ArgumentParser(description='visualize demo')
    ############################ basic_setings ########################
    parser.add_argument('--mode', type=str, default="ncamd", choices=['camd', 'camd-augment', "ncamd", "ncamd-augment"], help="choose model")
    opt = parser.parse_args()

    mode = opt.mode
    os.makedirs("midfile", exist_ok=True)

    with open("motion/path.json", "r") as f:
        path = json.load(f)

    args = Namespace()
    with open(path["config"], 'r') as f:
        params1 = json.load(f)
    for key, value in params1.items():
        setattr(args, key, value)

    if mode == "ncamd":
        args.arch = "llama_decoder_rope"
        args.encode_full = 2
        args.txt_tokens = 2
        args.model_path = path["ncamd"]
        args.rep = "smr"
        args.conv_bias = True
        args.conv_norm = "layernorm"
        args.conv_activate = "relu"
        args.trans_activate = "swiglu"
        model, diffusion = create_model_and_diffusion(args, None, path)
        state_dict = torch.load(args.model_path, map_location='cpu')
        load_model_wo_clip(model, state_dict["ema"])
        positional_embedding = None

    elif mode == "ncamd-augment":
        args.arch = "llama_decoder_rope"
        args.encode_full = 2
        args.txt_tokens = 2
        args.model_path = path["ncamd-augment"]
        args.rep = "smr"
        args.conv_bias = True
        args.conv_norm = "layernorm"
        args.conv_activate = "relu"
        args.trans_activate = "swiglu"
        model, diffusion = create_model_and_diffusion(args, None, path)
        state_dict = torch.load(args.model_path, map_location='cpu')
        load_model_wo_clip(model, state_dict["ema"])
        positional_embedding = None

    elif mode == "camd":
        args.arch = "llama_decoder_static"
        args.encode_full = 2
        args.txt_tokens = 1
        args.model_path = path["camd"]
        args.rep = "smr"
        args.conv_bias = False
        args.conv_norm = "rmsnorm"
        args.conv_activate = "silu"
        args.trans_activate = "swiglu"
        args.quantization = True
        model, diffusion = create_model_and_diffusion(args, None, path)
        state_dict = torch.load(args.model_path, map_location='cpu')
        load_model_wo_clip(model, state_dict["ema"])
        positional_embedding = model.sequence_pos_encoder

    elif mode == "camd-augment":
        args.arch = "llama_decoder_static"
        args.encode_full = 2
        args.txt_tokens = 1
        args.model_path = path["camd-augment"]
        args.rep = "smr"
        args.conv_bias = False
        args.conv_norm = "rmsnorm"
        args.conv_activate = "silu"
        args.trans_activate = "swiglu"
        args.quantization = True
        model, diffusion = create_model_and_diffusion(args, None, path)
        state_dict = torch.load(args.model_path, map_location='cpu')
        load_model_wo_clip(model, state_dict["ema"])
        positional_embedding = model.sequence_pos_encoder

    onnx_path1 = "midfile/{}_decoder.onnx".format(mode)
    engine1 = "midfile/{}_decoder.engine".format(mode)
    onnx_path2 = "midfile/{}_clip.onnx".format(mode)
    engine2 = "midfile/{}_clip.engine".format(mode)

    quanti_model = MotionGenerator(model.input_process, model.output_process, model.code_full, model.seqTransEncoder, model.embed_timestep, model.encode_compress, positional_embedding, args.txt_tokens, model.condition_compress if args.txt_tokens == 2 else None)
    clip_model = TextProcess(model.embed_text, model.clip_model)
    input_names = ["motion", "timestep", "txt"]
    output_names = ["output"]
    dynamic_axes = {
        "motion":{0:"batch"},
        "timestep":{0:"batch"},
        "txt":{1:"batch"},
        "output":{0:"batch"}
    }
    dynamic_float16(quanti_model, [torch.randn([1, 269, 1, 196]), torch.randint(0, 1000, (1,)), torch.randn([78, 1, 512])], onnx_path1, engine1,
    input_names, output_names, dynamic_axes, [1, 269, 1, 196, 1, 78, 1, 512], [1, 269, 1, 196, 1, 78, 1, 512], [4, 269, 1, 196, 4, 78, 4, 512], "cpu")

    input_names = ["txt"]
    output_names = ["output"]
    dynamic_axes = {
        "txt":{0:"batch"},
        "output":{0:"batch"}
    }
    dynamic_float32(clip_model, torch.randint(0, 49407, [1, 77]), onnx_path2, engine2, input_names,
    output_names, dynamic_axes, [1, 77], [1,77], [4, 77], "cpu")
