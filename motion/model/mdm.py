import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from motion.model import clip
import json
from motion.model.base_transformer import RefinedLayer, Refined_Transformer
from motion.model.Encode_Full import Encoder_Block

class MDM(nn.Module):
    def __init__(self, njoints, nfeats, latent_dim=256, ff_size=1024, num_layers=8, num_heads=4, dropout=0.1,
                 activation="gelu", dataset='amass', clip_dim=512,
                 arch='trans_enc', clip_version=None, **kargs):
        super().__init__()

        self.encode_full = kargs.get("encode_full", 0)      #### encode_full = 1 add tokens  & encode_full = 2 model compress tokens
        self.txt_tokens = kargs.get("txt_tokens", 0)    #### txt_tokens = 1 add tokens  & txt_tokens = 2 model compress tokens
        self.frame_mask = kargs.get("frame_mask", 0)
        self.dataset = dataset
        self.condition_length = 77
        self.num_frames = kargs.get("num_frames", 196)
        self.position_type = "static"     #### static or rope  only for llama arch
        self.json_dict = kargs.get("json_dict")

        if isinstance(self.num_frames, list) or isinstance(self.num_frames, tuple):
            self.num_frames = self.num_frames[0]

        self.njoints = njoints
        self.nfeats = nfeats

        self.latent_dim = latent_dim

        self.ff_size = ff_size
        self.num_layers = num_layers
        self.num_heads = num_heads
        self.dropout = dropout

        self.activation = activation
        self.clip_dim = clip_dim
        self.action_emb = kargs.get('action_emb', None)

        self.input_feats = self.njoints * self.nfeats

        self.cond_mode = kargs.get('cond_mode', 'no_cond')
        self.cond_mask_prob = kargs.get('cond_mask_prob', 0.)
        self.arch = arch

        self.input_process = InputProcess(self.input_feats, self.latent_dim)    #### 输入 x 的 linear
        self.output_process = OutputProcess(self.input_feats, self.latent_dim, self.njoints,
                                            self.nfeats)

        self.sequence_pos_encoder = PositionalEncoding(self.latent_dim, self.dropout)

        if self.arch == 'trans_enc':
            print("TRANS_ENC init")
            seqTransEncoderLayer = nn.TransformerEncoderLayer(d_model=self.latent_dim,
                                                            nhead=self.num_heads,
                                                            dim_feedforward=self.ff_size,
                                                            dropout=self.dropout,
                                                            activation=self.activation)
            self.seqTransEncoder = nn.TransformerEncoder(seqTransEncoderLayer, num_layers=self.num_layers)

        elif self.arch == "refined_encoder":
            TransLayer = RefinedLayer(self.latent_dim, self.num_heads, self.ff_size, self.dropout, self.activation, max_seq_len=self.num_frames, norm_type="rmsnorm")
            self.seqTransEncoder = Refined_Transformer(TransLayer, self.num_layers)

        elif self.arch == "refined_decoder":
            TransLayer = RefinedLayer(self.latent_dim, self.num_heads, self.ff_size, self.dropout, self.activation, max_seq_len=self.num_frames, word_tokens=True, norm_type="rmsnorm")
            self.seqTransEncoder = Refined_Transformer(TransLayer, self.num_layers)

        elif self.arch == "llama_encoder":
            TransLayer = RefinedLayer(self.latent_dim, self.num_heads, self.ff_size, self.dropout, self.activation, max_seq_len=self.num_frames, position_type=self.position_type, norm_type="rmsnorm", attention_type="llama")
            self.seqTransEncoder = Refined_Transformer(TransLayer, self.num_layers)

        elif self.arch == "llama_decoder":
            TransLayer = RefinedLayer(self.latent_dim, self.num_heads, self.ff_size, self.dropout, self.activation, max_seq_len=self.num_frames, position_type=self.position_type, word_tokens=True, norm_type="rmsnorm", attention_type="llama")
            self.seqTransEncoder = Refined_Transformer(TransLayer, self.num_layers)

        else:
            raise ValueError('Please choose correct architecture')

        self.embed_timestep = TimestepEmbedder(self.latent_dim, self.sequence_pos_encoder)

        if self.cond_mode != 'no_cond':
            if 'text' in self.cond_mode:
                self.embed_text = nn.Linear(self.clip_dim, self.latent_dim)
                print('EMBED TEXT')
                print('Loading CLIP...')
                self.clip_version = clip_version
                self.clip_model = self.load_and_freeze_clip(clip_version)

                if self.txt_tokens == 2:
                    if self.arch in ["refined_encoder", "trans_enc", "llama_encoder"]:
                        scale = 3
                    elif self.arch in ["refined_decoder", "llama_decoder"]:
                        scale = 2
                    encode_compress_layer = RefinedLayer(d_model=self.latent_dim * scale,
                                                                    nhead=self.num_heads,
                                                                    dim_feedforward=self.ff_size,
                                                                    dropout=self.dropout,
                                                                    activation=self.activation)
                    self.condition_compress = nn.Sequential(
                        Refined_Transformer(encode_compress_layer, num_layers=1),
                        nn.Linear(self.latent_dim * scale, self.latent_dim, )
                    )       

        if self.encode_full != 0: ####  [1, bs, 512] -> [seq, bs, 1024] -> [seq, bs, 512]
            self.code_full = Encoder_Block(begin_channel=self.input_feats, latent_dim=self.latent_dim, num_layers=6, TN=1)      

            if self.encode_full == 2:
                encode_compress_layer = RefinedLayer(d_model=self.latent_dim * 2,
                                                                nhead=self.num_heads,
                                                                dim_feedforward=self.ff_size,
                                                                dropout=self.dropout,
                                                                activation=self.activation)

                self.encode_compress = nn.Sequential(
                    Refined_Transformer(encode_compress_layer, num_layers=1),
                    nn.Linear(self.latent_dim * 2, self.latent_dim, )
                )

        print(" =========================", self.cond_mode, "===================================")

    def parameters_wo_clip(self):
        return [p for name, p in self.named_parameters() if not name.startswith('clip_model.')]

    def load_and_freeze_clip(self, clip_version):
        clip_model, clip_preprocess = clip.load(clip_version, device='cpu', jit=False, download_root=self.json_dict["clip"])  # Must set jit=False for training
        clip.model.convert_weights(clip_model)  # Actually this line is unnecessary since clip by default already on float16
        clip_model.float()
        # Freeze CLIP weights
        clip_model.eval()
        for p in clip_model.parameters():
            p.requires_grad = False

        return clip_model

    def mask_cond(self, cond, force_mask=False):
        bs = cond.shape[0]
        if force_mask:
            return torch.zeros_like(cond)
        elif self.training and self.cond_mask_prob > 0.:
            mask = torch.bernoulli(torch.ones(bs, device=cond.device) * self.cond_mask_prob)  # 1-> use null_cond, 0-> use real cond
            if len(cond.shape) == 3:
                mask = mask.view(bs, 1, 1)
            else:
                mask = mask.view(bs, 1)
            return cond * (1. - mask)
        else:
            return cond

    def mask_motion(self, motion):
        # x: [batch_size, njoints, nfeats, max_frames], denoted x_t in the paper

        if self.training and self.frame_mask > 0.:
            pair_motion = torch.randperm(motion.shape[0])
            pair_motion = motion[pair_motion]
            if len(motion.shape) == 4:
                bs, njoints, nfeats, nframes = motion.shape
                mask = torch.bernoulli(torch.ones([bs, 1, 1, nframes], device=motion.device) * self.frame_mask)  # 1-> use null_cond, 0-> use real cond
                mask = mask.repeat(1, njoints, nfeats, 1)
            elif len(motion.shape) == 3:
                seqlen, bs, latent_dim = motion.shape
                mask = torch.bernoulli(torch.ones([seqlen, bs, 1], device=motion.device) * self.frame_mask) 
                mask = mask.repeat(1, 1, latent_dim)
            return motion * (1. - mask) + pair_motion * mask
        else:
            return motion

    def clip_text_embedding(self, raw_text):
        device = self.clip_model.ln_final.weight.device
        default_context_length = self.condition_length
        texts = clip.tokenize(raw_text, context_length=default_context_length, truncate=True).to(device) # [bs, context_length] # if n_tokens > context_length -> will truncate
        if self.txt_tokens == 0:   
            clip_feature = self.clip_model.encode_text(texts)
        else:
            with torch.no_grad():
                x = self.clip_model.token_embedding(texts).type(self.clip_model.dtype)  # [batch_size, n_ctx, d_model]
                x = x + self.clip_model.positional_embedding.type(self.clip_model.dtype)
                x = x.permute(1, 0, 2)  # NLD -> LND
                x = self.clip_model.transformer(x)
                x = x.permute(1, 0, 2)  # LND -> NLD
                x = self.clip_model.ln_final(x).type(self.clip_model.dtype)
                clip_feature = x[torch.arange(x.shape[0]), texts.argmax(dim=-1)] @ self.clip_model.text_projection
            clip_feature = clip_feature.unsqueeze(1)
            clip_feature = torch.cat([clip_feature, x], dim=1)     #### [bs, T, 512]
        return clip_feature
        
    def get_mask(self, sz1, sz2):
        mask = (torch.triu(torch.ones(sz1, sz2)) == 1).transpose(0, 1)
        mask = mask.float()
        mask = mask.masked_fill(mask == 0, float('-inf')).masked_fill(mask == 1, float(0.0))
        mask.requires_grad = False
        return mask

    def forward(self, x, timesteps, y=None):
        """
        x: [batch_size, njoints, nfeats, max_frames], denoted x_t in the paper
        timesteps: [batch_size] (int)
        """

        results = {}
        emb = self.embed_timestep(timesteps)  # [1, bs, d]
        x = x.to(emb.dtype)

        x = self.mask_motion(x)

        real_length = x.shape[-1]
        if self.encode_full != 0 and x.shape[-1] < self.num_frames:
            extension = torch.zeros([x.shape[0], x.shape[1], x.shape[2], self.num_frames - x.shape[-1]], device=x.device, dtype=x.dtype)
            x = torch.cat([x, extension], dim=-1)

        if self.encode_full == 1:
            latent = self.code_full(x) ### [seq, bs, 512]
            current = self.input_process(x)       
            latent = latent.repeat(current.shape[0], 1, 1)
            current = current + latent
        elif self.encode_full == 2:
            latent = self.code_full(x) ### [seq, bs, 512]
            current = self.input_process(x)                      #### [seq, bs, 512]
            latent = latent.repeat(current.shape[0], 1, 1)
            current = torch.cat([current, latent], dim=2)
            current = self.encode_compress(current)
        else:
            current = self.input_process(x)                      #### [seq, bs, 512]

        force_mask = y.get('uncond', False)
        if 'text' in self.cond_mode:
            enc_text = self.clip_text_embedding(y['text']).to(emb.dtype)      ### MASK_COND 会按照一定的比例把 batch_size 中的一部分文本句整句换成 [0, 0, ... 0]
            txt_emb = self.embed_text(enc_text)
            txt_emb = self.mask_cond(txt_emb, force_mask=force_mask)
            
            if len(txt_emb.shape) == 3:
                txt_emb = txt_emb.permute(1, 0, 2)
            else:
                txt_emb = txt_emb.unsqueeze(0)
        else:
            txt_emb = None

        if txt_emb is not None:
            all_emb = txt_emb
        else:
            all_emb = torch.zeros_like(emb)

        if self.arch in ["refined_encoder", "trans_enc", "llama_encoder"] and txt_emb is not None:
            if self.txt_tokens == 1:
                word_embedding = all_emb[1::, :, :]
                global_embedding = all_emb[0:1, :, :].repeat(word_embedding.shape[0], 1, 1)
                all_emb = word_embedding + global_embedding
                emb = emb.repeat(all_emb.shape[0], 1, 1)
                emb += all_emb
            elif self.txt_tokens == 2:
                word_embedding = all_emb[1::, :, :]
                global_embedding = all_emb[0:1, :, :].repeat(word_embedding.shape[0], 1, 1)
                emb = emb.repeat(word_embedding.shape[0], 1, 1)
                concat_embedding = torch.cat([emb, global_embedding, word_embedding], dim=2)
                emb = self.condition_compress(concat_embedding)
            else:
                emb += all_emb
        elif txt_emb is not None:
            if self.txt_tokens == 1:
                emb = emb.repeat(all_emb.shape[0], 1, 1)
                emb += all_emb
            elif self.txt_tokens == 2:
                emb = emb.repeat(all_emb.shape[0], 1, 1)
                concat_embedding = torch.cat([emb, all_emb], dim=2)
                emb = self.condition_compress(concat_embedding)    
            else:
                emb += all_emb 
        else:
            emb = emb.repeat(all_emb.shape[0], 1, 1)
            emb += all_emb

        if self.arch in ["trans_enc", "refined_encoder", "llama_encoder"]:
            real_token_length = emb.shape[0]           ######### 用来截断输出，只保留真正的output
        elif self.arch in ["refined_decoder", "llama_decoder"]:
            real_token_length = 1

        if self.arch in ["trans_enc", "refined_encoder", "llama_encoder"]:
            xseq = torch.cat([emb, current], dim=0)

            if self.arch in ["trans_enc", "refined_encoder"] or self.position_type == "static":
                xseq = self.sequence_pos_encoder(xseq)

            output = self.seqTransEncoder(xseq)
        elif self.arch in ["refined_decoder", "llama_decoder"]:
            xseq = torch.cat([emb[0:1], current], dim=0)
            word_tokens = emb[1::]

            if self.arch in ["refined_decoder"] or self.position_type == "static":
                xseq = self.sequence_pos_encoder(xseq)
                # word_tokens = self.sequence_pos_encoder(word_tokens)
                
            output = self.seqTransEncoder(xseq, word_tokens=word_tokens)

        output = output[real_token_length:]
        output = self.output_process(output)  # [bs, njoints, nfeats, nframes]
        output = output[:, :, :, :real_length]
        results["output"] = output
        return results
  
    def _apply(self, fn):
        super()._apply(fn)

    def train(self, *args, **kwargs):
        super().train(*args, **kwargs)

class PositionalEncoding(nn.Module):
    def __init__(self, d_model, dropout=0.1, max_len=5000):  
        super(PositionalEncoding, self).__init__()
        self.dropout = nn.Dropout(p=dropout)

        pe = torch.zeros(max_len, d_model)      ###### max_len 是 T_steps 长度， d_model 是嵌入特征的维度
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-np.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0).transpose(0, 1)

        self.register_parameter('pe', nn.Parameter(pe, requires_grad=False))

    def forward(self, x):
        # not used in the final model
        x = x + self.pe[:x.shape[0], :]
        return self.dropout(x)


class TimestepEmbedder(nn.Module):
    def __init__(self, latent_dim, sequence_pos_encoder):
        super().__init__()
        self.latent_dim = latent_dim
        self.sequence_pos_encoder = sequence_pos_encoder

        time_embed_dim = self.latent_dim
        self.time_embed = nn.Sequential(
            nn.Linear(self.latent_dim, time_embed_dim, ),
            nn.SiLU(),
            nn.Linear(time_embed_dim, time_embed_dim, ),
        )

    def forward(self, timesteps):       #### timesteps 也是按照 position 的方式编码的 [times, 1, latent] -> [1, times, latent] ?
        return self.time_embed(self.sequence_pos_encoder.pe[timesteps]).permute(1, 0, 2)


class InputProcess(nn.Module):
    def __init__(self, input_feats, latent_dim):
        super().__init__()
        self.input_feats = input_feats
        self.latent_dim = latent_dim
        self.poseEmbedding = nn.Linear(self.input_feats, self.latent_dim)
      
    def forward(self, x):
        bs, njoints, nfeats, nframes = x.shape          ### [B,263, nframes] -> [B, nframes, 263]
        x = x.permute((3, 0, 1, 2)).reshape(nframes, bs, njoints*nfeats) 
        x = self.poseEmbedding(x)  # [seqlen, bs, d]
        return x

     
class OutputProcess(nn.Module):
    def __init__(self, input_feats, latent_dim, njoints, nfeats):
        super().__init__()
        self.input_feats = input_feats
        self.latent_dim = latent_dim
        self.njoints = njoints
        self.nfeats = nfeats

        self.poseFinal = nn.Linear(self.latent_dim, self.input_feats)
    

    def forward(self, output):
        nframes, bs, d = output.shape
        output = self.poseFinal(output)  # [seqlen, bs, 150]
        output = output.reshape(nframes, bs, self.njoints, self.nfeats)
        output = output.permute(1, 2, 3, 0)  # [bs, njoints, nfeats, nframes]

        return output


class EmbedAction(nn.Module):
    def __init__(self, num_actions, latent_dim):
        super().__init__()
        self.action_embedding = nn.Parameter(torch.randn(num_actions, latent_dim))

    def forward(self, input):
        idx = input[:, 0].to(torch.long)  # an index array must be long
        output = self.action_embedding[idx]
        return output