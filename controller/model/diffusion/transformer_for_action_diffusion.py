from typing import Union, Optional, List
import logging
import torch
import torch.nn as nn
from controller.model.common.module_attr_mixin import ModuleAttrMixin
import math

from torch.jit import Final
from timm.models.vision_transformer import Attention, Mlp, RmsNorm, use_fused_attn
import torch.nn.functional as F

logger = logging.getLogger(__name__)

# embedding layers for timestep, referring to RDT, DiT
class TimestepEmbedder(nn.Module):
    """
    Embeds scalar timesteps into vector representations.
    """
    def __init__(self, hidden_size, frequency_embedding_size=256, dtype=torch.float32):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(frequency_embedding_size, hidden_size, bias=True),
            nn.SiLU(),
            nn.Linear(hidden_size, hidden_size, bias=True),
        )
        self.frequency_embedding_size = frequency_embedding_size
        self.dtype = dtype

    def timestep_embedding(self, t, dim, max_period=10000):
        """
        Create sinusoidal timestep embeddings.
        :param t: a 1-D Tensor of N indices, one per batch element.
                          These may be fractional.
        :param dim: the dimension of the output.
        :param max_period: controls the minimum frequency of the embeddings.
        :return: an (N, D) Tensor of positional embeddings.
        """
        # https://github.com/openai/glide-text2im/blob/main/glide_text2im/nn.py
        half = dim // 2
        freqs = torch.exp(
            -math.log(max_period) * torch.arange(
                start=0, end=half, dtype=torch.float32, device=t.device) / half
        )
        args = t[:, None].float() * freqs[None]
        embedding = torch.cat([torch.cos(args), torch.sin(args)], dim=-1)
        if dim % 2:
            embedding = torch.cat([embedding, torch.zeros_like(embedding[:, :1])], dim=-1)
        return embedding.to(self.dtype)

    def forward(self, t):
        t_freq = self.timestep_embedding(t, self.frequency_embedding_size)
        t_emb = self.mlp(t_freq)
        return t_emb

# Cross Attention Layers, referring to RDT
class CrossAttention(nn.Module):
    """
    A cross-attention layer with flash attention,
    to incorporate the conditional information into main sequence.
    """
    fused_attn: Final[bool]
    def __init__(
            self,
            dim: int,
            num_heads: int = 8,
            qkv_bias: bool = False,
            qk_norm: bool = False,
            attn_drop: float = 0,
            proj_drop: float = 0,
            norm_layer: nn.Module = nn.LayerNorm,
            attn_mask_kwargs: dict = None,
    ) -> None:
        super().__init__()
        assert dim % num_heads == 0, 'dim should be divisible by num_heads'
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.scale = self.head_dim ** -0.5
        self.fused_attn = use_fused_attn()

        self.q = nn.Linear(dim, dim, bias=qkv_bias)
        self.kv = nn.Linear(dim, dim * 2, bias=qkv_bias)
        self.q_norm = norm_layer(self.head_dim) if qk_norm else nn.Identity()
        self.k_norm = norm_layer(self.head_dim) if qk_norm else nn.Identity()
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)

        self.use_attn_mask = attn_mask_kwargs['use_attn_mask']
        if self.use_attn_mask:
            obs_part_length = attn_mask_kwargs['obs_part_length']
            max_cond_tokens = attn_mask_kwargs['max_cond_tokens']
            action_horizon = attn_mask_kwargs['action_horizon']
    
            # Create a unified mask tensor: (4, action_horizon, max_cond_tokens)
            masks = torch.ones((4, action_horizon, max_cond_tokens), dtype=torch.bool)
            
            # mask 1: only mask first part
            masks[0, :, :obs_part_length[0]] = False
            
            # mask 2: only mask second part
            masks[1, :, obs_part_length[0]:obs_part_length[0]+obs_part_length[1]] = False
            
            # mask 3: mask first and second parts
            masks[2, :, :obs_part_length[0]+obs_part_length[1]] = False
            
            # mask 4: no masking
            
            self.register_buffer('masks', masks)
            self.register_buffer('probs', torch.tensor([0.1, 0.1, 0.1, 0.7]))

    def forward(self, x: torch.Tensor, c: torch.Tensor, training=True, gen_attn_map=False) -> torch.Tensor:
        # x: main sequence, action sequence, (B,T,n_emb)
        # c: conditional information, observation features and time embedding, (B,N+1,n_emb)
        B, N, C = x.shape
        _, L, _ = c.shape
        q = self.q(x).reshape(B, N, self.num_heads, self.head_dim).permute(0, 2, 1, 3)
        kv = self.kv(c).reshape(B, L, 2, self.num_heads, self.head_dim).permute(2, 0, 3, 1, 4)
        k, v = kv.unbind(0)
        q, k = self.q_norm(q), self.k_norm(k)

        attn_mask = None
        if self.use_attn_mask and training:
            idx = torch.multinomial(self.probs, num_samples=1)
            attn_mask = self.masks[idx.squeeze()]
        
        attn_weights = None
        if self.fused_attn:
            x = F.scaled_dot_product_attention(
                q, k, v,
                dropout_p=self.attn_drop.p if self.training else 0.,
                attn_mask=attn_mask
            )
            # For fused attention, we need to manually calculate attention weights
            if gen_attn_map:
                with torch.no_grad():
                    attn_weights = (q @ k.transpose(-2, -1)) * self.scale
                    if attn_mask is not None:
                        attn_weights = attn_weights.masked_fill_(attn_mask.logical_not(), float('-inf'))
                    attn_weights = attn_weights.softmax(dim=-1)  # (B, num_heads, T, L)
                    attn_weights = attn_weights.detach().cpu().to(torch.float16).numpy()
                    attn_weights = attn_weights[:2, :, 0, :]  # Only save the first two samples, the first action token
        else:
            q = q * self.scale
            attn = q @ k.transpose(-2, -1)
            if attn_mask is not None:
                attn = attn.masked_fill_(attn_mask.logical_not(), float('-inf'))
            attn = attn.softmax(dim=-1)
            if gen_attn_map:
                attn_weights = attn.detach().cpu().to(torch.float16).numpy()  # (B, num_heads, T, L)
                attn_weights = attn_weights[:2, :, 0, :]  # Only save the first two samples, the first action token
            if self.attn_drop.p > 0:
                attn = self.attn_drop(attn)
            x = attn @ v
            
        x = x.permute(0, 2, 1, 3).reshape(B, N, C)
        x = self.proj(x)
        if self.proj_drop.p > 0:
            x = self.proj_drop(x)
            
        return x, attn_weights


class RDTBlock(nn.Module):
    def __init__(self, hidden_size, num_heads, attn_mask_kwargs, **block_kwargs):
        super().__init__()
        # self-attention
        # x (B,T,n_emb)
        self.norm1 = RmsNorm(hidden_size, eps=1e-6)
        self.attn = Attention(
            dim=hidden_size, num_heads=num_heads, 
            qkv_bias=True, qk_norm=True, 
            norm_layer=RmsNorm,**block_kwargs)
        
        # cross-attention
        # x (B,T,n_emb), c (B,N+1,n_emb)
        self.cross_attn = CrossAttention(
            hidden_size, num_heads=num_heads, 
            qkv_bias=True, qk_norm=True, 
            norm_layer=RmsNorm, 
            attn_mask_kwargs=attn_mask_kwargs, **block_kwargs)
        
        self.norm2 = RmsNorm(hidden_size, eps=1e-6)
        approx_gelu = lambda: nn.GELU(approximate="tanh")

        # feed-forward network
        self.ffn = Mlp(in_features=hidden_size, 
            hidden_features=hidden_size, act_layer=approx_gelu, drop=0)
        self.norm3 = RmsNorm(hidden_size, eps=1e-6)

    def forward(self, x, c, training=True, gen_attn_map=False):
        origin_x = x
        x = self.norm1(x)
        x = self.attn(x)
        x = x + origin_x
        
        origin_x = x
        x = self.norm2(x)
        x, attn_weights = self.cross_attn(x, c, training=training, gen_attn_map=gen_attn_map)
        x = x + origin_x
                
        origin_x = x
        x = self.norm3(x)
        x = self.ffn(x)
        x = x + origin_x
        
        return x, attn_weights



class TransformerForActionDiffusion(ModuleAttrMixin):
    def __init__(self,
        input_dim: int,
        output_dim: int,
        action_horizon: int,
        n_layer: int = 7,
        n_head: int = 8,
        n_emb: int = 768,
        max_cond_tokens: int=800,
        p_drop_attn: float = 0.1,
        obs_part_length: List[int] = None,
        use_attn_mask: bool = True,
        ) -> None:
        super().__init__()
        
        # input embedding stem
        self.input_emb = nn.Linear(input_dim, n_emb)
        self.pos_emb = nn.Parameter(torch.randn((1, action_horizon, n_emb)))
        # time embedding
        # self.time_emb = SinusoidalPosEmb(n_emb) # previous
        self.time_emb = TimestepEmbedder(hidden_size=n_emb)
        
        # learnable position embedding
        self.cond_pos_emb =  nn.Parameter(torch.randn((1, max_cond_tokens, n_emb)))

        attn_mask_kwargs = {
            'use_attn_mask': use_attn_mask,
            'obs_part_length': obs_part_length,
            'max_cond_tokens': max_cond_tokens,
            'action_horizon': action_horizon
        }

        # RDT blocks Recurrent Diffusion Transformer Blocks
        self.blocks = nn.ModuleList([
            RDTBlock(
                hidden_size=n_emb,
                num_heads=n_head,
                attn_drop=p_drop_attn,
                proj_drop=p_drop_attn,
                attn_mask_kwargs=attn_mask_kwargs
            )
            for _ in range(n_layer)
        ])

        # use RMS Norm referring to RDT
        # "our problem can be considered as a time series forecasting task, and the centering operation in the original DiTs' LayerNorm could cause token shift and attention shift, thus destroying the symmetry of the time series"
        self.ln_f = RmsNorm(n_emb, eps=1e-6)
        self.head = nn.Linear(n_emb, output_dim)  # 映射回动作空间

        self.action_horizon = action_horizon
        
        # init  # TODO整个模型及其所有子模块递归地调用 _init_weights 函数，以完成模型参数的初始化 不懂 
        self.apply(self._init_weights)
        logger.info(
            "Number of parameters in DiT: %e", sum(p.numel() for p in self.parameters())
        )

    def _init_weights(self, module):
        ignore_types = (nn.Dropout, 
            ## SinusoidalPosEmb, 
            TimestepEmbedder,
            nn.TransformerEncoderLayer, 
            nn.TransformerDecoderLayer,
            nn.TransformerEncoder,
            nn.TransformerDecoder,
            nn.ModuleList,
            nn.Mish,
            nn.Sequential,
            nn.Embedding,
            nn.SiLU,
            nn.GELU,
            nn.Identity,
            RDTBlock)
        if isinstance(module, (nn.Linear)):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if isinstance(module, nn.Linear) and module.bias is not None:
                torch.nn.init.zeros_(module.bias)
        elif isinstance(module, (Attention, CrossAttention)):
        # Weight initialization in Attention module
            if hasattr(module, 'q'):
                torch.nn.init.normal_(module.q.weight, mean=0.0, std=0.02)
                if module.q.bias is not None:
                    torch.nn.init.zeros_(module.q.bias)
            if hasattr(module, 'k'):
                torch.nn.init.normal_(module.k.weight, mean=0.0, std=0.02)
                if module.k.bias is not None:
                    torch.nn.init.zeros_(module.k.bias)
            if hasattr(module, 'v'):
                torch.nn.init.normal_(module.v.weight, mean=0.0, std=0.02)
                if module.v.bias is not None:
                    torch.nn.init.zeros_(module.v.bias)
            if hasattr(module, 'proj'):
                torch.nn.init.normal_(module.proj.weight, mean=0.0, std=0.02)
                if module.proj.bias is not None:
                    torch.nn.init.zeros_(module.proj.bias)
        elif isinstance(module, Mlp):
            # Weight initialization in Mlp module
            if hasattr(module, 'fc1'):
                torch.nn.init.normal_(module.fc1.weight, mean=0.0, std=0.02)
                if module.fc1.bias is not None:
                    torch.nn.init.zeros_(module.fc1.bias)
            if hasattr(module, 'fc2'):
                torch.nn.init.normal_(module.fc2.weight, mean=0.0, std=0.02)
                if module.fc2.bias is not None:
                    torch.nn.init.zeros_(module.fc2.bias)
        elif isinstance(module, nn.MultiheadAttention):
            weight_names = [
                'in_proj_weight', 'q_proj_weight', 'k_proj_weight', 'v_proj_weight']
            for name in weight_names:
                weight = getattr(module, name)
                if weight is not None:
                    torch.nn.init.normal_(weight, mean=0.0, std=0.02)
            
            bias_names = ['in_proj_bias', 'bias_k', 'bias_v']
            for name in bias_names:
                bias = getattr(module, name)
                if bias is not None:
                    torch.nn.init.zeros_(bias)
        elif isinstance(module, (nn.LayerNorm)):
            torch.nn.init.zeros_(module.bias)
            torch.nn.init.ones_(module.weight)
        elif isinstance(module, RmsNorm):
            torch.nn.init.ones_(module.weight)
        elif isinstance(module, TransformerForActionDiffusion):
            torch.nn.init.normal_(module.pos_emb, mean=0.0, std=0.02)
            if module.cond_pos_emb is not None:
                torch.nn.init.normal_(module.cond_pos_emb, mean=0.0, std=0.02)
        elif isinstance(module, ignore_types):
            # no param
            pass
        else:
            raise RuntimeError("Unaccounted module {}".format(module))

    def forward(self, 
        sample: torch.Tensor, 
        timestep: Union[torch.Tensor, float, int], 
        cond: Optional[torch.Tensor]=None,
        training=True,
        gen_attn_map=False, 
        **kwargs):
        """
        sample: (B,T,input_dim)
        timestep: (B,) or int, diffusion step
        cond: (B,N,n_emb)
        output: (B,T,input_dim)
        return: 
            if gen_attn_map:
                output tensor (B,T,input_dim), attention_maps List[(B,num_heads,T,L)]
            else:
                output tensor (B,T,input_dim), None
        """
        
        # 1. time embedding
        timesteps = timestep   # (B,)
        if not torch.is_tensor(timesteps):
            # TODO: this requires sync between CPU and GPU. So try to pass timesteps as tensors if you can
            timesteps = torch.tensor([timesteps], dtype=torch.long, device=sample.device)
        elif torch.is_tensor(timesteps) and len(timesteps.shape) == 0:
            timesteps = timesteps[None].to(sample.device)
        # broadcast to batch dimension in a way that's compatible with ONNX/Core ML
        timesteps = timesteps.expand(sample.shape[0])  # (B,n_emb)
        # time_emb = self.time_emb(timesteps).unsqueeze(1)  # (B,1,n_emb)
        time_emb = self.time_emb(timesteps)  # (B,n_emb)
        time_emb = time_emb.unsqueeze(1)  # (B,1,n_emb)

        # 2. process conditions
        # cond: (B,T,n_emb)
        cond_emb = torch.cat([cond, time_emb], dim=1) # (B,N+1,n_emb)
        tc = cond_emb.shape[1]
        cond_pos_emb = self.cond_pos_emb[:, :tc, :]  # each position maps to a (learnable) vector # (1,N+1,n_emb)
        cond_emb = cond_emb + cond_pos_emb # (B,N+1,n_emb)

        # 3. process input
        input_emb = self.input_emb(sample) # (B,T,n_emb)
        t = input_emb.shape[1]
        pos_emb = self.pos_emb[:, :t, :]  # each position maps to a (learnable) vector # (1,T,n_emb)
        x = input_emb + pos_emb # (B,T,n_emb)

        attention_weights = [] if gen_attn_map else None
        for block in self.blocks:
            x, attn_weights = block(x, cond_emb, training=training, gen_attn_map=gen_attn_map)
            if gen_attn_map:
                attention_weights.append(attn_weights)
        
        x = self.ln_f(x)
        x = self.head(x)
        
        return x, attention_weights
