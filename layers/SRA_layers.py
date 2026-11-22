from typing import Optional, Tuple, TYPE_CHECKING
import torch
import torch.utils.checkpoint
from torch import nn
from layers.Xpos import XPOS, TemporalRoPE
if TYPE_CHECKING:
    from layers.configs import TrajGPTConfig


class Selective_Recurrent_Attention(nn.Module):
    def __init__(self, config: 'TrajGPTConfig'):
        super().__init__()
        self.config = config
        self.qkv = nn.Linear(config.d_model,
                             config.qk_dim * 2 + config.v_dim,
                             bias=config.use_bias_in_sra)
        self.silu = nn.SiLU()
        self.gated = nn.Linear(config.d_model, config.v_dim, bias=False)
        self.proj = nn.Linear(config.v_dim, config.d_model, bias=config.use_bias_in_sra_out)
        self.gn = nn.GroupNorm(num_groups=config.num_heads, num_channels=config.v_dim, affine=False)
        self.xpos = XPOS(config.qk_dim)
        self.time_mode = getattr(config, 'time_mode', 'event_order')
        if self.time_mode == 'time_conditioned':
            self.time_rope = TemporalRoPE(
                config.qk_dim // config.num_heads, getattr(config, 'time_scale', 1.0))

        # initialize gamma, found
        if config.use_default_gamma:
            gamma = 1 - 2**(-5 - torch.arange(0, config.num_heads, dtype=torch.float))
        else:
            # you should update the gamma value based on your dataset, if the time span is large, make gamma close to 1
            # s = torch.log(torch.tensor(1 / 32))
            s = torch.log(torch.tensor(1 / 64))
            # s = torch.log(torch.tensor(1 / 128))
            e = torch.log(torch.tensor(1 / 512))
            gamma = 1 - torch.exp(torch.linspace(s, e, config.num_heads))  # [h,]
        self.decay = nn.Parameter(gamma, requires_grad=False) # tensor([0.9688, 0.9876, 0.9951, 0.9980])

        # Define head-wise projection to compute \gamma
        self.gamma_proj = nn.Linear(
            config.d_model,
            config.num_heads,
            bias=True
        )
        self.tau = config.tau  # temperature for the gamma

    def get_parallel_decay_mask(self, t, decay, retention_mask=None):
        # Decay remains event-wise. Actual time enters Q/K through temporal RoPE.
        B, T, H = decay.shape
        # log(b_n / b_m) avoids underflow from long cumulative products.
        # Clip the unused upper triangle before exponentiating: its positive
        # log ratios could otherwise overflow and introduce NaN gradients.
        log_b = decay.float().clamp_min(torch.finfo(torch.float32).tiny).log().cumsum(dim=1)
        log_ratio = log_b.unsqueeze(2) - log_b.unsqueeze(1)
        D = log_ratio.clamp_max(0).exp().permute(0, 3, 1, 2)
        decay_mask = torch.tril(D).to(decay.dtype)
        if retention_mask is not None:
            # Expand the retention mask to match the decay mask dimensions
            retention_mask = retention_mask.to(dtype=decay_mask.dtype).view(-1, 1, 1, T)
            decay_mask = decay_mask * retention_mask
        else:
            decay_mask = decay_mask

        return decay_mask

    def get_data_dependent_decay(self, hidden_states):
        gamma = self.gamma_proj(hidden_states)
        decay = torch.sigmoid(gamma) ** (1.0 / self.tau)
        return decay

    def parallel_retention(self, q, k, v, decay_mask, use_cache=True):
        """
        q,  # bsz * num_heads * len * qk_dim
        k,  # bsz * num_heads * len * qk_dim
        v,  # bsz * num_heads * len * v_dim
        decay_mask,  # (1 or bsz) * num_heads * len * len
        """
        # [b, h, t, t]
        retention = q @ k.transpose(-1, -2) * k.size(-1)**-0.5  # (scaled dot-product) found
        retention = retention * decay_mask
        output = retention @ v #

        # Basic generation recomputes its prefix and does not need a cache.
        # For legacy consumers, reduce directly rather than materializing a
        # [batch, heads, time, qk_dim, v_dim] tensor.
        current_kv = None
        if use_cache:
            intra_decay = decay_mask[:, :, -1, :, None]
            current_kv = k.transpose(-1, -2) @ (v * intra_decay)
        return output, current_kv, retention

    def recurrent_retention(self, q, k, v, past_key_value=None, decay=None, retention_mask=None):
        """
        q, k, v, # bsz * num_heads * 1 * qkv_dim
        past_key_value, # bsz * num_heads * qk_dim * v_dim
        decay # num_heads * 1 * 1
        retention_mask # bsz * 1
        """
        B, H, _, _ = q.shape
        past_key_value = past_key_value if past_key_value is not None else 0
        decay = decay if decay is not None else 0
        decay = decay.view(B, H, 1, 1)
        retention_mask = retention_mask.view(B, 1, 1, 1) if retention_mask is not None else 1
        current_kv = decay * past_key_value + retention_mask * (k.transpose(-1, -2) @ v)
        output = q @ current_kv * k.size(-1)**-0.5  # (b, h, 1, d_v)
        return output, current_kv

    def forward(
        self,
        hidden_states: torch.Tensor,
        t: torch.Tensor,
        retention_mask: Optional[torch.Tensor] = None,
        past_key_value: Optional[Tuple[torch.Tensor]] = None,
        forward_impl: str = 'parallel',
        sequence_offset: Optional[int] = 0,
        output_retentions: Optional[bool] = None,
        use_cache: bool = True,
    ) -> Tuple[torch.FloatTensor, torch.FloatTensor, Optional[torch.FloatTensor]]:
        B, T, d = hidden_states.size()

        q, k, v = self.qkv(hidden_states).split(
            [self.config.qk_dim, self.config.qk_dim, self.config.v_dim], dim=-1)

        if self.time_mode != 'time_conditioned':
            q, k = self.xpos.rotate_queries_and_keys(q, k, offset=sequence_offset)
        q, k, v = [x.reshape(B, T, self.config.num_heads, -1).transpose(1, 2)
                   for x in (q, k, v)]
        if self.time_mode == 'time_conditioned':
            q, k = self.time_rope.rotate_queries_and_keys(q, k, t)
        decay = self.get_data_dependent_decay(hidden_states)
        # SRA recurrent attention computation
        if forward_impl == 'parallel':
            decay_mask = self.get_parallel_decay_mask(t, decay, retention_mask=retention_mask)
            retention_out, curr_kv, retention_weights = self.parallel_retention(q, k, v,
                                                                                decay_mask=decay_mask,
                                                                                use_cache=use_cache)
        elif forward_impl == 'recurrent':
            for n in range(T):
                gamma_n = decay[:, n, :]
                retention_mask_n = retention_mask[:, n]         # (B,)
                retention_out, curr_kv = self.recurrent_retention(q, k, v,
                                                              past_key_value=past_key_value,
                                                              decay=gamma_n,
                                                              retention_mask=retention_mask_n)
        else:
            raise ValueError(f'forward_impl {forward_impl} not supported.')
        # concat heads
        retention_out = retention_out.transpose(1, 2).contiguous().view(B, T, self.config.v_dim)
        # group norm (merge batch, length dimension -> group norm -> split back)
        normed = self.gn(retention_out.view(B * T, self.config.v_dim))
        normed = normed.view(B, T, self.config.v_dim)
        # out gate & proj
        out = self.silu(self.gated(hidden_states)) * normed

        outputs = (self.proj(out), curr_kv) # project v_dim to d_model
        if output_retentions:
            outputs += (retention_weights,) if forward_impl == 'parallel' else (None,)
        return outputs


class SRA_Block(nn.Module):
    '''
    An SRA block consists of an SRA module and an MLP layer
    '''
    def __init__(self, config: 'TrajGPTConfig'):
        super().__init__()
        self.config = config

        # multi-scale retention, output shape is [bsz, seq_len, d_model]
        self.sra = Selective_Recurrent_Attention(config)

        # 2-layer MLP, output shape is [bsz, seq_len, d_model]
        self.ffn = nn.Sequential(
            nn.Linear(config.d_model, config.ffn_proj_size, bias=config.use_bias_in_mlp),
            nn.GELU(),
            nn.Linear(config.ffn_proj_size, config.d_model, bias=config.use_bias_in_mlp),
        )

        self.ln1 = nn.LayerNorm(config.d_model)
        self.ln2 = nn.LayerNorm(config.d_model)

    def forward(
        self,
        hidden_states: torch.Tensor,
        t: torch.Tensor,
        retention_mask: Optional[torch.Tensor] = None,
        forward_impl: str = 'parallel',
        past_key_value: Optional[Tuple[torch.Tensor]] = None,
        sequence_offset: Optional[int] = 0,
        output_retentions: Optional[bool] = False,
        use_cache: bool = True,
    ) -> Tuple[torch.FloatTensor, torch.FloatTensor, Optional[torch.FloatTensor]]:
        sra_outs = self.sra(self.ln1(hidden_states),
                            t=t,
                            retention_mask=retention_mask,
                            past_key_value=past_key_value,
                            forward_impl=forward_impl,
                            sequence_offset=sequence_offset,
                            output_retentions=output_retentions,
                            use_cache=use_cache)
        # Original SRA layer with residual connection
        sra = sra_outs[0]
        curr_kv = sra_outs[1]
        x = hidden_states + sra

        # MLP layer
        y = x + self.ffn(self.ln2(x))

        outputs = (y, curr_kv)

        if output_retentions:
            outputs += (sra_outs[2],)
        return outputs



