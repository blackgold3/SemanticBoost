import numpy as np
import torch
import torch.nn as nn
from model import clip
from model.base_transformer import RefinedLayer, Refined_Transformer
from model.Encode_Full import Encoder_Block

class MDM(nn.Module):
    def __init__(self, args, cond_mode="text"):
        super(MDM, self).__init__()
        self.dropout = 0.1  ###### 有多少用暂时没统计过，MDM 默认项
        self.encode_full = args.encode_full
        self.word_tokens = args.word_tokens
        self.nframes = args.nframes
        self.frame_mask = args.frame_mask
        self.clip_path = args.clip_path
        self.clip_length = 77
        self.rep = args.rep
        self.cond_mask_prob = args.cond_mask_prob
        self.cond_mode = cond_mode
        self.activation = args.activation
        self.latent_dim = args.latent_dim
        self.num_heads = args.nheads
        self.ff_size = args.ff_size
        self.pos_encoder = args.pos_encoder
        self.num_layers = args.nlayers
        self.nfeats = 1 
        self.njoints = 198
        
        self.input_dims = self.njoints * self.nfeats

        ################## 输入，输出 和 diffusion step encoder #################
        self.input_process = InputProcess(self.input_dims, self.latent_dim)    #### 输入 x 的 linear
        self.output_process = OutputProcess(self.input_dims, self.latent_dim, self.njoints, self.nfeats)
        self.embed_timestep = TimestepEmbedder(self.latent_dim, self.dropout)
        if self.pos_encoder == "static":
            self.sequence_pos_encoder = PositionalEncoding(self.latent_dim, self.dropout)

        ######## transformer encoder + cross attention + rope 位置编码 ##############
        TransLayer = RefinedLayer(self.latent_dim, self.num_heads, self.ff_size, self.dropout, self.activation, max_seq_len=self.nframes, position_type=self.pos_encoder, word_tokens=self.word_tokens, norm_type="rmsnorm")
        self.seqTransEncoder = Refined_Transformer(TransLayer, self.num_layers)

        ################ text encoder (CLIP) ################
        if self.cond_mode == "text":
            self.clip_version = "ViT-B/32"
            print('EMBED TEXT')
            print('Loading CLIP...')
            self.clip_dim = 512    
            self.embed_text = nn.Linear(self.clip_dim, self.latent_dim)
            self.clip_model = self.load_and_freeze_clip(self.clip_version)

        ################## 加载全局卷积模块
        if self.encode_full: ####  [1, bs, 512] -> [seq, bs, 1024] -> [seq, bs, 512], 卷积一个全局信息，concat 到输入上，然后压缩回 latent_dim, 这里的结构有点问题，inputs 输入时没有考虑 diffusion steps 不同的情况，但似乎不太影响训练结果，所以这个步骤是否有用存疑
            self.code_full = Encoder_Block(self.latent_dim, 4, "rmsnorm", "silu")
        print(" =========================", self.cond_mode, "===================================")

    def load_and_freeze_clip(self, clip_version):
        '''
        CLIP 加载模型
        '''
        clip_model, clip_preprocess = clip.load(clip_version, device='cpu', jit=False, download_root=self.clip_path)  # Must set jit=False for training
        clip_model.float()
        # Freeze CLIP weights
        clip_model.eval()
        for p in clip_model.parameters():
            p.requires_grad = False
        return clip_model
    

    def mask_cond(self, text_cond, force_mask=False):
        '''
        CFG 条件的随机掩码
        '''
        if force_mask:
            return torch.zeros_like(text_cond)
            
        elif self.training and self.cond_mask_prob > 0.:
            bs = text_cond.shape[0]
            mask = torch.bernoulli(torch.ones(bs, device=text_cond.device) * self.cond_mask_prob)  # 1-> use null_cond, 0-> use real cond
            if len(text_cond.shape) == 3:
                mask = mask.view(bs, 1, 1)
            else:
                mask = mask.view(bs, 1)
            text_cond = text_cond * (1. - mask)
            return text_cond
        else:
            return text_cond

    def mask_motion(self, motion):
        '''
        输入随机掩码，随着 diffusion step 增加，输入中有效信息越来越多，随机掩码的目的是部分帧预测错误时不影响后续的去噪结果
        随机把一部分帧替换成其他动作，因为实际上不可能预测出空动作
        '''
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

    @torch.no_grad()
    def clip_text_embedding(self, raw_text):
        '''
        CLIP embedding 文本部分
        '''
        default_context_length = self.clip_length ##### 默认上限
        texts = clip.tokenize(raw_text, context_length=default_context_length, truncate=True) # [bs, context_length] # if n_tokens > context_length -> will truncate
        texts = texts.to(self.clip_model.ln_final.weight.device)
        if not self.word_tokens:   ##### 整句编码即可
            with torch.no_grad():
                clip_feature = self.clip_model.encode_text(texts)
        else:       ######## 需要每一个词的 embedding
            with torch.no_grad():   
                x = self.clip_model.token_embedding(texts)  # [batch_size, n_ctx, d_model]
                x = x + self.clip_model.positional_embedding
                x = x.permute(1, 0, 2)  # NLD -> LND
                x = self.clip_model.transformer(x)
                x = x.permute(1, 0, 2)  # LND -> NLD
                x = self.clip_model.ln_final(x)
                clip_feature = x[torch.arange(x.shape[0]), texts.argmax(dim=-1)] @ self.clip_model.text_projection
            clip_feature = clip_feature.unsqueeze(1)
            clip_feature = torch.cat([clip_feature, x], dim=1)     #### [bs, 1 + T, 512]
        return clip_feature
    
    def forward(self, x, timesteps, y=None):
        """
        x: [batch_size, njoints, nfeats, max_frames], denoted x_t in the paper
        timesteps: [batch_size] (int)
        """
        results = {}

        ###### step embedding ########
        emb = self.embed_timestep(timesteps)  # [1, bs, d]

        ######## x 随机掩码 ###########
        x = x.to(emb.dtype)
        x = self.mask_motion(x)
        nframes_real = x.shape[-1]

        ############ 全局编码或前处理  ############
        if self.encode_full:
            if x.shape[-1] < self.nframes:  ##### 推理时可能会出现帧太少的情况
                extension = torch.zeros([x.shape[0], x.shape[1], x.shape[2], self.nframes - x.shape[-1]], device=x.device, dtype=x.dtype)
                x = torch.cat([x, extension], dim=-1)

            current = self.input_process(x)
            latent = self.code_full(current, emb)
            latent = latent.repeat(current.shape[0], 1, 1)
            current = current + latent
        else:
            current = self.input_process(x)                      #### [seq, bs, 512]

        ############## 条件编码 ##############
        force_mask = y.get('uncond', False)
        if self.cond_mode == "text":
            enc_text = self.clip_text_embedding(y['text'])    ### MASK_COND 会按照一定的比例把 batch_size 中的一部分文本句整句换成 [0, 0, ... 0]
            txt_emb = self.embed_text(enc_text)
            txt_emb = self.mask_cond(txt_emb, force_mask=force_mask)
            if len(txt_emb.shape) == 3:
                txt_emb = txt_emb.permute(1, 0, 2)
            else:
                txt_emb = txt_emb.unsqueeze(0)
        else:
            txt_emb = None

        ############## 叠加 step embedding 信息 ########
        if txt_emb is None:
            txt_emb = torch.zeros_like(emb)
        emb = emb.repeat(txt_emb.shape[0], 1, 1)
        emb += txt_emb

        ################# 模型推理  ###############
        if self.word_tokens:  #### 如果是无条件模型或者其他条件，emb 不会叠加 word embeddings，为了结构正常，需要额外处理一下
            if emb.shape[0] == 1:
                emb = emb.repeat(1 + self.clip_length, 1, 1)
            word_embeddings = emb[1::]
        else:
            word_embeddings = None

        xseq = torch.cat([emb[0:1], current], dim=0)
        if self.pos_encoder == "static":
            xseq = self.sequence_pos_encoder(xseq)            

        output = self.seqTransEncoder(xseq, word_tokens=word_embeddings)

        ################## 后处理 ##################
        output = output[1:]     #### 输出的第一个 token 是条件 token
        output = self.output_process(output)  # [bs, njoints, nfeats, nframes]
        output = output[:, :, :, :nframes_real]     ##### 推理补全部分
        results["output"] = output
        return results
  

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
    def __init__(self, latent_dim, dropout=0.1):
        super().__init__()
        self.latent_dim = latent_dim
        self.dropout = dropout
        self.sequence_pos_encoder = PositionalEncoding(self.latent_dim, self.dropout)

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
        output = self.poseFinal(output)  # [seqlen, bs, 269]
        output = output.reshape(nframes, bs, self.njoints, self.nfeats)
        output = output.permute(1, 2, 3, 0)  # [bs, njoints, nfeats, nframes]

        return output

