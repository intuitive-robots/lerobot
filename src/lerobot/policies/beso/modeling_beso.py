import math
from collections import deque
from functools import partial

import einops
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F  # noqa: N812
import torchvision
from torch import Tensor
from tqdm.auto import trange

from lerobot.policies.beso.configuration_beso import BESOConfig
from lerobot.policies.pretrained import PreTrainedPolicy
from lerobot.policies.utils import (
    get_device_from_parameters,
    get_dtype_from_parameters,
    get_output_shape,
    populate_queues,
)
from lerobot.utils.constants import ACTION, OBS_ENV_STATE, OBS_IMAGES, OBS_STATE


# ==================== Noise / Sigma Utilities ====================


def append_dims(x: Tensor, target_dims: int) -> Tensor:
    """Appends dimensions to the end of a tensor until it has target_dims dimensions."""
    dims_to_append = target_dims - x.ndim
    if dims_to_append < 0:
        raise ValueError(f"input has {x.ndim} dims but target_dims is {target_dims}, which is less")
    return x[(...,) + (None,) * dims_to_append]


def append_zero(x: Tensor) -> Tensor:
    return torch.cat([x, x.new_zeros([1])])


def rand_log_logistic(
    shape, loc=0.0, scale=1.0, min_value=0.0, max_value=float("inf"), device="cpu", dtype=torch.float32
) -> Tensor:
    """Draws samples from an optionally truncated log-logistic distribution."""
    min_value = torch.as_tensor(min_value, device=device, dtype=torch.float64)
    max_value = torch.as_tensor(max_value, device=device, dtype=torch.float64)
    min_cdf = min_value.log().sub(loc).div(scale).sigmoid()
    max_cdf = max_value.log().sub(loc).div(scale).sigmoid()
    u = torch.rand(shape, device=device, dtype=torch.float64) * (max_cdf - min_cdf) + min_cdf
    return u.logit().mul(scale).add(loc).exp().to(dtype)


def rand_log_normal(shape, loc=0.0, scale=1.0, device="cpu", dtype=torch.float32) -> Tensor:
    """Draws samples from a lognormal distribution."""
    return (torch.randn(shape, device=device, dtype=dtype) * scale + loc).exp()


def rand_log_uniform(shape, min_value, max_value, device="cpu", dtype=torch.float32) -> Tensor:
    """Draws samples from a log-uniform distribution."""
    min_value = math.log(min_value)
    max_value = math.log(max_value)
    return (torch.rand(shape, device=device, dtype=dtype) * (max_value - min_value) + min_value).exp()


def rand_uniform(shape, min_value, max_value, device="cpu", dtype=torch.float32) -> Tensor:
    """Draws samples from a uniform distribution."""
    return torch.rand(shape, device=device, dtype=dtype) * (max_value - min_value) + min_value


def rand_v_diffusion(
    shape, sigma_data=1.0, min_value=0.0, max_value=float("inf"), device="cpu", dtype=torch.float32
) -> Tensor:
    """Draws samples from a truncated v-diffusion training timestep distribution."""
    min_cdf = math.atan(min_value / sigma_data) * 2 / math.pi
    max_cdf = math.atan(max_value / sigma_data) * 2 / math.pi
    u = torch.rand(shape, device=device, dtype=dtype) * (max_cdf - min_cdf) + min_cdf
    return torch.tan(u * math.pi / 2) * sigma_data


# ==================== Noise Schedules ====================


def get_sigmas_karras(n, sigma_min, sigma_max, rho=7.0, device="cpu") -> Tensor:
    """Constructs the noise schedule of Karras et al. (2022)."""
    ramp = torch.linspace(0, 1, n)
    min_inv_rho = sigma_min ** (1 / rho)
    max_inv_rho = sigma_max ** (1 / rho)
    sigmas = (max_inv_rho + ramp * (min_inv_rho - max_inv_rho)) ** rho
    return append_zero(sigmas).to(device)


def get_sigmas_exponential(n, sigma_min, sigma_max, device="cpu") -> Tensor:
    """Constructs an exponential noise schedule."""
    sigmas = torch.linspace(math.log(sigma_max), math.log(sigma_min), n, device=device).exp()
    return append_zero(sigmas)


def get_sigmas_linear(n, sigma_min, sigma_max, device="cpu") -> Tensor:
    """Constructs a linear noise schedule."""
    sigmas = torch.linspace(sigma_max, sigma_min, n, device=device)
    return append_zero(sigmas)


def get_sigmas_vp(n, beta_d=19.9, beta_min=0.1, eps_s=1e-3, device="cpu") -> Tensor:
    """Constructs a continuous VP noise schedule."""
    t = torch.linspace(1, eps_s, n, device=device)
    sigmas = torch.sqrt(torch.exp(beta_d * t**2 / 2 + beta_min * t) - 1)
    return append_zero(sigmas)


def cosine_beta_schedule(n, s=0.008, device="cpu") -> Tensor:
    """Cosine schedule as proposed in https://openreview.net/forum?id=-NEXDKk8gZ"""
    steps = n + 1
    x = np.linspace(0, steps, steps)
    alphas_cumprod = np.cos(((x / steps) + s) / (1 + s) * np.pi * 0.5) ** 2
    alphas_cumprod = alphas_cumprod / alphas_cumprod[0]
    betas = 1 - (alphas_cumprod[1:] / alphas_cumprod[:-1])
    betas_clipped = np.clip(betas, a_min=0, a_max=0.999)
    return append_zero(torch.tensor(np.flip(betas_clipped).copy(), device=device, dtype=torch.float32))


def get_sigmas_ve(n, sigma_min=0.02, sigma_max=100, device="cpu") -> Tensor:
    """Constructs a VE noise schedule."""
    t = torch.linspace(0, n, n, device=device)
    t = (sigma_max**2) * ((sigma_min**2 / sigma_max**2) ** (t / (n - 1)))
    sigmas = torch.sqrt(t)
    return append_zero(sigmas)


def get_iddpm_sigmas(n, sigma_min=0.02, sigma_max=100, M=1000, j_0=0, C_1=0.001, C_2=0.008, device="cpu"):
    """Constructs IDDPM sigmas."""
    step_indices = torch.arange(n, dtype=torch.float64, device=device)
    u = torch.zeros(M + 1, dtype=torch.float64, device=device)
    alpha_bar = lambda j: (0.5 * np.pi * j / M / (C_2 + 1)).sin() ** 2
    for j in torch.arange(M, j_0, -1, device=device):
        u[j - 1] = ((u[j] ** 2 + 1) / (alpha_bar(j - 1) / alpha_bar(j)).clip(min=C_1) - 1).sqrt()
    u_filtered = u[torch.logical_and(u >= sigma_min, u <= sigma_max)]
    sigmas = u_filtered[((len(u_filtered) - 1) / (n - 1) * step_indices).round().to(torch.int64)]
    return append_zero(sigmas).to(torch.float32)


# ==================== Sampling Functions ====================


def to_d(action: Tensor, sigma: Tensor, denoised: Tensor) -> Tensor:
    """Converts a denoiser output to a Karras ODE derivative."""
    return (action - denoised) / append_dims(sigma, action.ndim)


def get_ancestral_step(sigma_from, sigma_to, eta=1.0):
    """Calculates sigma_down and sigma_up for ancestral sampling."""
    if not eta:
        return sigma_to, 0.0
    sigma_up = min(sigma_to, eta * (sigma_to**2 * (sigma_from**2 - sigma_to**2) / sigma_from**2) ** 0.5)
    sigma_down = (sigma_to**2 - sigma_up**2) ** 0.5
    return sigma_down, sigma_up


@torch.no_grad()
def sample_ddim(model, state, action, goal, sigmas, scaler=None, disable=True, **extra_args):
    """DDIM sampler (1st order DPM-Solver)."""
    s_in = action.new_ones([action.shape[0]])
    sigma_fn = lambda t: t.neg().exp()
    t_fn = lambda sigma: sigma.log().neg()

    for i in trange(len(sigmas) - 1, disable=disable):
        denoised = model(state, action, goal, sigmas[i] * s_in)
        t, t_next = t_fn(sigmas[i]), t_fn(sigmas[i + 1])
        h = t_next - t
        action = (sigma_fn(t_next) / sigma_fn(t)) * action - (-h).expm1() * denoised
    return action


@torch.no_grad()
def sample_euler(model, state, action, goal, sigmas, scaler=None, disable=True, s_churn=0.0, s_tmin=0.0, s_tmax=float("inf"), s_noise=1.0, **extra_args):
    """Euler sampler (Algorithm 2 from Karras et al. 2022)."""
    s_in = action.new_ones([action.shape[0]])
    for i in trange(len(sigmas) - 1, disable=disable):
        gamma = min(s_churn / (len(sigmas) - 1), 2**0.5 - 1) if s_tmin <= sigmas[i] <= s_tmax else 0.0
        eps = torch.randn_like(action) * s_noise
        sigma_hat = sigmas[i] * (gamma + 1)
        if gamma > 0:
            action = action + eps * (sigma_hat**2 - sigmas[i] ** 2) ** 0.5
        denoised = model(state, action, goal, sigma_hat * s_in)
        d = to_d(action, sigma_hat, denoised)
        dt = sigmas[i + 1] - sigma_hat
        action = action + d * dt
        if scaler is not None:
            action = scaler.clip_output(action)
    return action


@torch.no_grad()
def sample_heun(model, state, action, goal, sigmas, scaler=None, disable=True, s_churn=0.0, s_tmin=0.0, s_tmax=float("inf"), s_noise=1.0, **extra_args):
    """Heun sampler (2nd order, Algorithm 2 from Karras et al. 2022)."""
    s_in = action.new_ones([action.shape[0]])
    for i in trange(len(sigmas) - 1, disable=disable):
        gamma = min(s_churn / (len(sigmas) - 1), 2**0.5 - 1) if s_tmin <= sigmas[i] <= s_tmax else 0.0
        eps = torch.randn_like(action) * s_noise
        sigma_hat = sigmas[i] * (gamma + 1)
        if gamma > 0:
            action = action + eps * (sigma_hat**2 - sigmas[i] ** 2) ** 0.5
        denoised = model(state, action, goal, sigma_hat * s_in)
        d = to_d(action, sigma_hat, denoised)
        dt = sigmas[i + 1] - sigma_hat
        if sigmas[i + 1] == 0:
            action = action + d * dt
        else:
            action_2 = action + d * dt
            denoised_2 = model(state, action_2, goal, sigmas[i + 1] * s_in)
            d_2 = to_d(action_2, sigmas[i + 1], denoised_2)
            d_prime = (d + d_2) / 2
            action = action + d_prime * dt
        if scaler is not None:
            action = scaler.clip_output(action)
    return action


@torch.no_grad()
def sample_euler_ancestral(model, state, action, goal, sigmas, scaler=None, disable=True, eta=1.0, **extra_args):
    """Ancestral sampling with Euler method steps."""
    s_in = action.new_ones([action.shape[0]])
    for i in trange(len(sigmas) - 1, disable=disable):
        denoised = model(state, action, goal, sigmas[i] * s_in)
        sigma_down, sigma_up = get_ancestral_step(sigmas[i], sigmas[i + 1], eta=eta)
        d = to_d(action, sigmas[i], denoised)
        dt = sigma_down - sigmas[i]
        action = action + d * dt
        if sigma_down > 0:
            action = action + torch.randn_like(action) * sigma_up
        if scaler is not None:
            action = scaler.clip_output(action)
    return action


@torch.no_grad()
def sample_dpmpp_2m(model, state, action, goal, sigmas, scaler=None, disable=True, **extra_args):
    """DPM-Solver++(2M)."""
    s_in = action.new_ones([action.shape[0]])
    sigma_fn = lambda t: t.neg().exp()
    t_fn = lambda sigma: sigma.log().neg()
    old_denoised = None

    for i in trange(len(sigmas) - 1, disable=disable):
        denoised = model(state, action, goal, sigmas[i] * s_in)
        t, t_next = t_fn(sigmas[i]), t_fn(sigmas[i + 1])
        h = t_next - t
        if old_denoised is None or sigmas[i + 1] == 0:
            action = (sigma_fn(t_next) / sigma_fn(t)) * action - (-h).expm1() * denoised
        else:
            h_last = t - t_fn(sigmas[i - 1])
            r = h_last / h
            denoised_d = (1 + 1 / (2 * r)) * denoised - (1 / (2 * r)) * old_denoised
            action = (sigma_fn(t_next) / sigma_fn(t)) * action - (-h).expm1() * denoised_d
        old_denoised = denoised
    return action


@torch.no_grad()
def sample_dpmpp_2s(model, state, action, goal, sigmas, scaler=None, disable=True, **extra_args):
    """DPM-Solver++(2S) second-order steps."""
    sigma_fn = lambda t: t.neg().exp()
    t_fn = lambda sigma: sigma.log().neg()
    s_in = action.new_ones([action.shape[0]])
    for i in trange(len(sigmas) - 1, disable=disable):
        denoised = model(state, action, goal, sigmas[i] * s_in)
        if sigmas[i + 1] == 0:
            d = to_d(action, sigmas[i], denoised)
            dt = sigmas[i + 1] - sigmas[i]
            action = action + d * dt
        else:
            t, t_next = t_fn(sigmas[i]), t_fn(sigmas[i + 1])
            r = 1 / 2
            h = t_next - t
            s = t + r * h
            x_2 = (sigma_fn(s) / sigma_fn(t)) * action - (-h * r).expm1() * denoised
            denoised_2 = model(state, x_2, goal, sigma_fn(s) * s_in)
            action = (sigma_fn(t_next) / sigma_fn(t)) * action - (-h).expm1() * denoised_2
        if scaler is not None:
            action = scaler.clip_output(action)
    return action


# Map sampler names to functions
SAMPLERS = {
    "ddim": sample_ddim,
    "euler": sample_euler,
    "heun": sample_heun,
    "euler_ancestral": sample_euler_ancestral,
    "dpmpp_2m": sample_dpmpp_2m,
    "dpmpp_2s": sample_dpmpp_2s,
}

# Map noise schedule names to functions
NOISE_SCHEDULES = {
    "karras": get_sigmas_karras,
    "exponential": get_sigmas_exponential,
    "linear": get_sigmas_linear,
    "vp": get_sigmas_vp,
    "cosine_beta": cosine_beta_schedule,
    "ve": get_sigmas_ve,
    "iddpm": get_iddpm_sigmas,
}


# ==================== Transformer Components ====================


class BESOLayerNorm(nn.Module):
    """LayerNorm with optional bias."""

    def __init__(self, ndim: int, bias: bool = False):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(ndim))
        self.bias = nn.Parameter(torch.zeros(ndim)) if bias else None

    def forward(self, x: Tensor) -> Tensor:
        return F.layer_norm(x, self.weight.shape, self.weight, self.bias, 1e-5)


class BESOAttention(nn.Module):
    """Multi-head attention with optional cross-attention."""

    def __init__(self, n_embd: int, n_head: int, attn_pdrop: float, resid_pdrop: float, block_size: int, causal: bool = False, bias: bool = False):
        super().__init__()
        assert n_embd % n_head == 0
        self.key = nn.Linear(n_embd, n_embd)
        self.query = nn.Linear(n_embd, n_embd)
        self.value = nn.Linear(n_embd, n_embd)
        self.c_proj = nn.Linear(n_embd, n_embd, bias=bias)
        self.attn_dropout = nn.Dropout(attn_pdrop)
        self.resid_dropout = nn.Dropout(resid_pdrop)
        self.n_head = n_head
        self.n_embd = n_embd
        self.causal = causal

    def forward(self, x: Tensor, context: Tensor | None = None) -> Tensor:
        B, T, C = x.size()
        if context is not None:
            k = self.key(context).view(B, -1, self.n_head, C // self.n_head).transpose(1, 2)
            q = self.query(x).view(B, T, self.n_head, C // self.n_head).transpose(1, 2)
            v = self.value(context).view(B, -1, self.n_head, C // self.n_head).transpose(1, 2)
        else:
            k = self.key(x).view(B, T, self.n_head, C // self.n_head).transpose(1, 2)
            q = self.query(x).view(B, T, self.n_head, C // self.n_head).transpose(1, 2)
            v = self.value(x).view(B, T, self.n_head, C // self.n_head).transpose(1, 2)

        y = F.scaled_dot_product_attention(
            q, k, v, dropout_p=self.attn_dropout.p if self.training else 0, is_causal=self.causal
        )
        y = y.transpose(1, 2).contiguous().view(B, T, C)
        y = self.resid_dropout(self.c_proj(y))
        return y


class BESOMLP(nn.Module):
    """Feed-forward MLP block."""

    def __init__(self, n_embd: int, bias: bool = False, dropout: float = 0.0):
        super().__init__()
        self.c_fc = nn.Linear(n_embd, 4 * n_embd, bias=bias)
        self.gelu = nn.GELU()
        self.c_proj = nn.Linear(4 * n_embd, n_embd, bias=bias)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: Tensor) -> Tensor:
        x = self.c_fc(x)
        x = self.gelu(x)
        x = self.c_proj(x)
        x = self.dropout(x)
        return x


class BESOBlock(nn.Module):
    """Transformer block with optional cross-attention."""

    def __init__(self, n_embd: int, n_heads: int, attn_pdrop: float, resid_pdrop: float, mlp_pdrop: float, block_size: int, causal: bool, use_cross_attention: bool = False, bias: bool = False):
        super().__init__()
        self.ln_1 = BESOLayerNorm(n_embd, bias=bias)
        self.attn = BESOAttention(n_embd, n_heads, attn_pdrop, resid_pdrop, block_size, causal, bias)
        self.use_cross_attention = use_cross_attention
        if self.use_cross_attention:
            self.cross_att = BESOAttention(n_embd, n_heads, attn_pdrop, resid_pdrop, block_size, causal, bias)
            self.ln3 = nn.LayerNorm(n_embd)
        self.ln_2 = BESOLayerNorm(n_embd, bias=bias)
        self.mlp = BESOMLP(n_embd, bias, mlp_pdrop)

    def forward(self, x: Tensor, context: Tensor | None = None) -> Tensor:
        x = x + self.attn(self.ln_1(x))
        if self.use_cross_attention and context is not None:
            x = x + self.cross_att(self.ln3(x), context)
        x = x + self.mlp(self.ln_2(x))
        return x


class BESOAdaLNZero(nn.Module):
    """AdaLN-Zero modulation for conditioning."""

    def __init__(self, hidden_size: int):
        super().__init__()
        self.modulation = nn.Sequential(
            nn.SiLU(),
            nn.Linear(hidden_size, 6 * hidden_size, bias=True),
        )

    def forward(self, c: Tensor):
        return self.modulation(c).chunk(6, dim=-1)


class BESOConditionedBlock(BESOBlock):
    """Block with AdaLN-Zero conditioning."""

    def __init__(self, n_embd, n_heads, attn_pdrop, resid_pdrop, mlp_pdrop, block_size, causal, film_cond_dim, use_cross_attention=False, bias=False):
        super().__init__(n_embd, n_heads, attn_pdrop, resid_pdrop, mlp_pdrop, block_size, causal, use_cross_attention=use_cross_attention, bias=bias)
        self.adaLN_zero = BESOAdaLNZero(film_cond_dim)

    def forward(self, x: Tensor, c: Tensor, context: Tensor | None = None) -> Tensor:
        shift_msa, scale_msa, gate_msa, shift_mlp, scale_mlp, gate_mlp = self.adaLN_zero(c)
        x_attn = self.ln_1(x)
        x_attn = shift_msa + (x_attn * scale_msa)
        x = x + gate_msa * self.attn(x_attn)
        if self.use_cross_attention and context is not None:
            x = x + self.cross_att(self.ln3(x), context)
        x_mlp = self.ln_2(x)
        x_mlp = shift_mlp + (x_mlp * scale_mlp)
        x = x + gate_mlp * self.mlp(x_mlp)
        return x


class BESOTransformerEncoder(nn.Module):
    """Transformer encoder (non-causal self-attention)."""

    def __init__(self, embed_dim: int, n_heads: int, attn_pdrop: float, resid_pdrop: float, n_layers: int, block_size: int, bias: bool = False, mlp_pdrop: float = 0.0):
        super().__init__()
        self.blocks = nn.ModuleList([
            BESOBlock(embed_dim, n_heads, attn_pdrop, resid_pdrop, mlp_pdrop, block_size, causal=False, bias=bias)
            for _ in range(n_layers)
        ])
        self.ln = BESOLayerNorm(embed_dim, bias)

    def forward(self, x: Tensor) -> Tensor:
        for layer in self.blocks:
            x = layer(x)
        x = self.ln(x)
        return x


class BESOTransformerFiLMDecoder(nn.Module):
    """Transformer decoder with AdaLN-Zero FiLM conditioning and cross-attention."""

    def __init__(self, embed_dim: int, n_heads: int, attn_pdrop: float, resid_pdrop: float, n_layers: int, block_size: int, film_cond_dim: int, bias: bool = False, mlp_pdrop: float = 0.0, use_cross_attention: bool = True):
        super().__init__()
        self.blocks = nn.ModuleList([
            BESOConditionedBlock(
                embed_dim, n_heads, attn_pdrop, resid_pdrop, mlp_pdrop, block_size,
                causal=True, use_cross_attention=use_cross_attention, bias=bias, film_cond_dim=film_cond_dim,
            )
            for _ in range(n_layers)
        ])
        self.ln = BESOLayerNorm(embed_dim, bias)

    def forward(self, x: Tensor, c: Tensor, cond: Tensor | None = None) -> Tensor:
        for layer in self.blocks:
            x = layer(x, c, cond)
        x = self.ln(x)
        return x


class BESOTransformerDecoder(nn.Module):
    """Standard transformer decoder with cross-attention (no FiLM)."""

    def __init__(self, embed_dim: int, n_heads: int, attn_pdrop: float, resid_pdrop: float, n_layers: int, block_size: int, bias: bool = False, mlp_pdrop: float = 0.0, use_cross_attention: bool = True):
        super().__init__()
        self.blocks = nn.ModuleList([
            BESOBlock(embed_dim, n_heads, attn_pdrop, resid_pdrop, mlp_pdrop, block_size, causal=True, use_cross_attention=use_cross_attention, bias=bias)
            for _ in range(n_layers)
        ])
        self.ln = BESOLayerNorm(embed_dim, bias)

    def forward(self, x: Tensor, cond: Tensor | None = None) -> Tensor:
        for layer in self.blocks:
            x = layer(x, cond)
        x = self.ln(x)
        return x


class BESOSinusoidalPosEmb(nn.Module):
    """Sinusoidal positional embedding for sigma/timestep conditioning."""

    def __init__(self, dim: int):
        super().__init__()
        self.dim = dim

    def forward(self, x: Tensor) -> Tensor:
        device = x.device
        half_dim = self.dim // 2
        emb = math.log(10000) / (half_dim - 1)
        emb = torch.exp(torch.arange(half_dim, device=device) * -emb)
        emb = x[:, None] * emb[None, :]
        emb = torch.cat((emb.sin(), emb.cos()), dim=-1)
        return emb


# ==================== MDT Transformer (Inner Model) ====================


class MDTTransformer(nn.Module):
    """Masked Diffusion Transformer for score prediction.

    Encoder-decoder architecture where:
    - Encoder processes goal + observation tokens
    - Decoder takes noisy action tokens with sigma conditioning and cross-attends to encoder output
    """

    def __init__(
        self,
        obs_dim: int,
        goal_dim: int,
        action_dim: int,
        embed_dim: int,
        embed_pdrop: float,
        attn_pdrop: float,
        resid_pdrop: float,
        mlp_pdrop: float,
        n_dec_layers: int,
        n_enc_layers: int,
        n_heads: int,
        goal_seq_len: int,
        obs_seq_len: int,
        action_seq_len: int,
        goal_drop: float = 0.1,
        bias: bool = False,
        linear_output: bool = True,
        use_ada_conditioning: bool = True,
        use_noise_encoder: bool = False,
    ):
        super().__init__()
        self.obs_dim = obs_dim
        self.embed_dim = embed_dim
        self.action_dim = action_dim
        self.goal_seq_len = goal_seq_len
        self.obs_seq_len = obs_seq_len
        self.action_seq_len = action_seq_len
        self.use_ada_conditioning = use_ada_conditioning

        block_size = goal_seq_len + action_seq_len + obs_seq_len + 1
        seq_size = goal_seq_len + action_seq_len + obs_seq_len

        # Embeddings
        self.tok_emb = nn.Linear(obs_dim, embed_dim)
        self.pos_emb = nn.Parameter(torch.zeros(1, seq_size, embed_dim))
        self.drop = nn.Dropout(embed_pdrop)
        self.cond_mask_prob = goal_drop

        self.goal_emb = nn.Linear(goal_dim, embed_dim)
        self.action_emb = nn.Linear(action_dim, embed_dim)

        # Sigma embedding
        self.sigma_emb = nn.Sequential(
            BESOSinusoidalPosEmb(embed_dim),
            nn.Linear(embed_dim, embed_dim * 2),
            nn.Mish(),
            nn.Linear(embed_dim * 2, embed_dim),
        )

        # Encoder
        self.encoder = BESOTransformerEncoder(
            embed_dim=embed_dim, n_heads=n_heads, attn_pdrop=attn_pdrop,
            resid_pdrop=resid_pdrop, n_layers=n_enc_layers, block_size=block_size, bias=bias, mlp_pdrop=mlp_pdrop,
        )

        # Decoder
        if use_ada_conditioning:
            self.decoder = BESOTransformerFiLMDecoder(
                embed_dim=embed_dim, n_heads=n_heads, attn_pdrop=attn_pdrop,
                resid_pdrop=resid_pdrop, n_layers=n_dec_layers, block_size=block_size,
                film_cond_dim=embed_dim, bias=bias, mlp_pdrop=mlp_pdrop, use_cross_attention=True,
            )
        else:
            self.decoder = BESOTransformerDecoder(
                embed_dim=embed_dim, n_heads=n_heads, attn_pdrop=attn_pdrop,
                resid_pdrop=resid_pdrop, n_layers=n_dec_layers, block_size=block_size,
                bias=bias, mlp_pdrop=mlp_pdrop, use_cross_attention=True,
            )

        # Output head
        if linear_output:
            self.action_pred = nn.Linear(embed_dim, action_dim)
        else:
            self.action_pred = nn.Sequential(
                nn.Linear(embed_dim, embed_dim), nn.GELU(), nn.Linear(embed_dim, action_dim)
            )

        self.apply(self._init_weights)

    def _init_weights(self, module):
        if isinstance(module, (nn.Linear, nn.Embedding)):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if isinstance(module, nn.Linear) and module.bias is not None:
                torch.nn.init.zeros_(module.bias)
        elif isinstance(module, nn.LayerNorm):
            torch.nn.init.zeros_(module.bias)
            torch.nn.init.ones_(module.weight)
        elif isinstance(module, MDTTransformer):
            torch.nn.init.normal_(module.pos_emb, mean=0.0, std=0.02)

    def forward(self, states: Tensor, actions: Tensor, goals: Tensor, sigma: Tensor) -> Tensor:
        b, t, dim = states.size()
        _, t_a, _ = actions.size()
        _, t_g, _ = goals.size()

        # Embeddings
        state_embed = self.tok_emb(states)
        action_embed = self.action_emb(actions)
        goal_embed = self.goal_emb(goals)

        # Position embeddings
        goal_x = self.drop(goal_embed + self.pos_emb[:, :t_g, :])
        state_x = self.drop(state_embed + self.pos_emb[:, t_g : (t_g + t), :])
        action_x = self.drop(action_embed + self.pos_emb[:, (t_g + t) : (t + t_g + t_a), :])

        # Encode context (goal + state)
        context = self.encoder(torch.cat([goal_x, state_x], dim=1))

        # Sigma embedding
        sigmas = sigma.log() / 4
        sigmas = einops.rearrange(sigmas, "b -> b 1")
        emb_t = self.sigma_emb(sigmas)
        if len(emb_t.shape) == 2:
            emb_t = einops.rearrange(emb_t, "b d -> b 1 d")

        # Decode
        if self.use_ada_conditioning:
            x = self.decoder(action_x, emb_t, context)
        else:
            x = self.decoder(action_x, context)

        pred_actions = self.action_pred(x)
        return pred_actions


# ==================== GCDenoiser (Karras EDM Preconditioner) ====================


class GCDenoiser(nn.Module):
    """Karras et al. preconditioner for denoising diffusion models.

    Wraps the inner model (MDTTransformer) with c_skip, c_out, c_in scalings
    to improve training stability across noise levels.
    """

    def __init__(self, inner_model: nn.Module, sigma_data: float = 1.0):
        super().__init__()
        self.inner_model = inner_model
        self.sigma_data = sigma_data

    def get_scalings(self, sigma: Tensor):
        c_skip = self.sigma_data**2 / (sigma**2 + self.sigma_data**2)
        c_out = sigma * self.sigma_data / (sigma**2 + self.sigma_data**2) ** 0.5
        c_in = 1 / (sigma**2 + self.sigma_data**2) ** 0.5
        return c_skip, c_out, c_in

    def loss(self, state: Tensor, action: Tensor, goal: Tensor, noise: Tensor, sigma: Tensor) -> tuple[Tensor, Tensor]:
        c_skip, c_out, c_in = [append_dims(x, action.ndim) for x in self.get_scalings(sigma)]
        noised_input = action + noise * append_dims(sigma, action.ndim)
        model_output = self.inner_model(state, noised_input * c_in, goal, sigma)
        target = (action - c_skip * noised_input) / c_out
        return (model_output - target).pow(2).flatten(1).mean(), model_output

    def forward(self, state: Tensor, action: Tensor, goal: Tensor, sigma: Tensor) -> Tensor:
        c_skip, c_out, c_in = [append_dims(x, action.ndim) for x in self.get_scalings(sigma)]
        return self.inner_model(state, action * c_in, goal, sigma) * c_out + action * c_skip


# ==================== RGB Encoder ====================


class BESOSpatialSoftmax(nn.Module):
    """Spatial Soft Argmax operation."""

    def __init__(self, input_shape, num_kp=None):
        super().__init__()
        assert len(input_shape) == 3
        self._in_c, self._in_h, self._in_w = input_shape

        if num_kp is not None:
            self.nets = nn.Conv2d(self._in_c, num_kp, kernel_size=1)
            self._out_c = num_kp
        else:
            self.nets = None
            self._out_c = self._in_c

        pos_x, pos_y = np.meshgrid(np.linspace(-1.0, 1.0, self._in_w), np.linspace(-1.0, 1.0, self._in_h))
        pos_x = torch.from_numpy(pos_x.reshape(self._in_h * self._in_w, 1)).float()
        pos_y = torch.from_numpy(pos_y.reshape(self._in_h * self._in_w, 1)).float()
        self.register_buffer("pos_grid", torch.cat([pos_x, pos_y], dim=1))

    def forward(self, features: Tensor) -> Tensor:
        if self.nets is not None:
            features = self.nets(features)
        features = features.reshape(-1, self._in_h * self._in_w)
        attention = F.softmax(features, dim=-1)
        expected_xy = attention @ self.pos_grid
        feature_keypoints = expected_xy.view(-1, self._out_c, 2)
        return feature_keypoints


class BESORgbEncoder(nn.Module):
    """Encodes an RGB image into a 1D feature vector using a ResNet backbone + SpatialSoftmax."""

    def __init__(self, config: BESOConfig):
        super().__init__()
        if config.resize_shape is not None:
            self.resize = torchvision.transforms.Resize(config.resize_shape)
        else:
            self.resize = None

        crop_shape = config.crop_shape
        if crop_shape is not None:
            self.do_crop = True
            self.center_crop = torchvision.transforms.CenterCrop(crop_shape)
            if config.crop_is_random:
                self.maybe_random_crop = torchvision.transforms.RandomCrop(crop_shape)
            else:
                self.maybe_random_crop = self.center_crop
        else:
            self.do_crop = False

        backbone_model = getattr(torchvision.models, config.vision_backbone)(
            weights=config.pretrained_backbone_weights
        )
        self.backbone = nn.Sequential(*(list(backbone_model.children())[:-2]))
        if config.use_group_norm:
            if config.pretrained_backbone_weights:
                raise ValueError("You can't replace BatchNorm in a pretrained model without ruining the weights!")
            self.backbone = _replace_submodules(
                root_module=self.backbone,
                predicate=lambda x: isinstance(x, nn.BatchNorm2d),
                func=lambda x: nn.GroupNorm(num_groups=x.num_features // 16, num_channels=x.num_features),
            )

        images_shape = next(iter(config.image_features.values())).shape
        if config.crop_shape is not None:
            dummy_shape_h_w = config.crop_shape
        elif config.resize_shape is not None:
            dummy_shape_h_w = config.resize_shape
        else:
            dummy_shape_h_w = images_shape[1:]
        dummy_shape = (1, images_shape[0], *dummy_shape_h_w)
        feature_map_shape = get_output_shape(self.backbone, dummy_shape)[1:]

        self.pool = BESOSpatialSoftmax(feature_map_shape, num_kp=config.spatial_softmax_num_keypoints)
        self.feature_dim = config.spatial_softmax_num_keypoints * 2
        self.out = nn.Linear(config.spatial_softmax_num_keypoints * 2, self.feature_dim)
        self.relu = nn.ReLU()

    def forward(self, x: Tensor) -> Tensor:
        if self.resize is not None:
            x = self.resize(x)
        if self.do_crop:
            if self.training:
                x = self.maybe_random_crop(x)
            else:
                x = self.center_crop(x)
        x = torch.flatten(self.pool(self.backbone(x)), start_dim=1)
        x = self.relu(self.out(x))
        return x


def _replace_submodules(root_module, predicate, func):
    """Replace submodules matching predicate with func(module)."""
    if predicate(root_module):
        return func(root_module)
    replace_list = [k.split(".") for k, m in root_module.named_modules(remove_duplicate=True) if predicate(m)]
    for *parents, k in replace_list:
        parent_module = root_module
        if len(parents) > 0:
            parent_module = root_module.get_submodule(".".join(parents))
        if isinstance(parent_module, nn.Sequential):
            src_module = parent_module[int(k)]
        else:
            src_module = getattr(parent_module, k)
        tgt_module = func(src_module)
        if isinstance(parent_module, nn.Sequential):
            parent_module[int(k)] = tgt_module
        else:
            setattr(parent_module, k, tgt_module)
    assert not any(predicate(m) for _, m in root_module.named_modules(remove_duplicate=True))
    return root_module


# ==================== BESO Model ====================


class BESOModel(nn.Module):
    """Core BESO model: image encoding + GCDenoiser (MDTTransformer) + EDM sampling."""

    def __init__(self, config: BESOConfig):
        super().__init__()
        self.config = config

        # Build observation encoders
        obs_dim = 0
        if config.image_features:
            num_images = len(config.image_features)
            if config.use_separate_rgb_encoder_per_camera:
                encoders = [BESORgbEncoder(config) for _ in range(num_images)]
                self.rgb_encoder = nn.ModuleList(encoders)
                obs_dim += encoders[0].feature_dim * num_images
            else:
                self.rgb_encoder = BESORgbEncoder(config)
                obs_dim += self.rgb_encoder.feature_dim * num_images

        if config.robot_state_feature:
            state_dim = config.robot_state_feature.shape[0]
            self.state_emb = nn.Linear(state_dim, config.embed_dim)
            self._has_state = True
        else:
            self._has_state = False

        if config.env_state_feature:
            obs_dim += config.env_state_feature.shape[0]

        # If obs_dim doesn't match embed_dim, project it
        if obs_dim != config.embed_dim and obs_dim > 0:
            self.obs_proj = nn.Linear(obs_dim, config.embed_dim)
        else:
            self.obs_proj = None

        # Action dimension
        action_dim = config.action_feature.shape[0]

        # Compute the actual observation sequence length that _encode_observations produces.
        # Visual/env features produce n_obs_steps tokens, and if robot_state_feature is present
        # it gets concatenated as additional tokens (also n_obs_steps), so total can be 2*n_obs_steps.
        obs_seq_len = config.n_obs_steps
        if config.robot_state_feature:
            obs_seq_len += config.n_obs_steps

        # Inner model: MDTTransformer
        inner_model = MDTTransformer(
            obs_dim=config.embed_dim,
            goal_dim=config.goal_dim,
            action_dim=action_dim,
            embed_dim=config.embed_dim,
            embed_pdrop=config.embed_pdrop,
            attn_pdrop=config.attn_pdrop,
            resid_pdrop=config.resid_pdrop,
            mlp_pdrop=config.mlp_pdrop,
            n_dec_layers=config.n_dec_layers,
            n_enc_layers=config.n_enc_layers,
            n_heads=config.n_heads,
            goal_seq_len=config.goal_seq_len,
            obs_seq_len=obs_seq_len,
            action_seq_len=config.horizon,
            goal_drop=config.goal_drop,
            bias=False,
            linear_output=config.linear_output,
            use_ada_conditioning=config.use_ada_conditioning,
            use_noise_encoder=config.use_noise_encoder,
        )

        # Wrap with Karras EDM preconditioner
        self.denoiser = GCDenoiser(inner_model, sigma_data=config.sigma_data)

        # Diffusion parameters
        self.sigma_data = config.sigma_data
        self.sigma_min = config.sigma_min
        self.sigma_max = config.sigma_max
        self.sigma_sample_density_type = config.sigma_sample_density_type
        self.noise_scheduler_type = config.noise_scheduler
        self.sampler_type = config.sampler_type
        self.num_sampling_steps = config.num_sampling_steps
        self.action_dim = action_dim
        self.horizon = config.horizon

    def _encode_observations(self, batch: dict[str, Tensor]) -> Tensor:
        """Encode observations into a sequence of tokens."""
        batch_size, n_obs_steps = batch[OBS_STATE].shape[:2]
        features_list = []

        # Encode images
        if self.config.image_features:
            if self.config.use_separate_rgb_encoder_per_camera:
                images_per_camera = einops.rearrange(batch[OBS_IMAGES], "b s n ... -> n (b s) ...")
                img_features = torch.cat([
                    encoder(images)
                    for encoder, images in zip(self.rgb_encoder, images_per_camera, strict=True)
                ])
                img_features = einops.rearrange(img_features, "(n b s) ... -> b s (n ...)", b=batch_size, s=n_obs_steps)
            else:
                img_features = self.rgb_encoder(
                    einops.rearrange(batch[OBS_IMAGES], "b s n ... -> (b s n) ...")
                )
                img_features = einops.rearrange(
                    img_features, "(b s n) ... -> b s (n ...)", b=batch_size, s=n_obs_steps
                )
            features_list.append(img_features)

        if self.config.env_state_feature:
            features_list.append(batch[OBS_ENV_STATE])

        # Concatenate visual features
        if features_list:
            obs_features = torch.cat(features_list, dim=-1)  # (B, n_obs_steps, obs_dim)
            if self.obs_proj is not None:
                obs_features = self.obs_proj(obs_features)
        else:
            obs_features = torch.zeros(batch_size, n_obs_steps, self.config.embed_dim, device=batch[OBS_STATE].device)

        # Add state embedding as extra tokens
        if self._has_state:
            state_tokens = self.state_emb(batch[OBS_STATE])  # (B, n_obs_steps, embed_dim)
            obs_features = torch.cat([obs_features, state_tokens], dim=1)  # (B, 2*n_obs_steps, embed_dim)

        return obs_features

    def _make_sample_density(self):
        """Returns a callable that samples noise levels for training."""
        if self.sigma_sample_density_type == "loglogistic":
            loc = math.log(self.sigma_data)
            return partial(rand_log_logistic, loc=loc, scale=0.5, min_value=self.sigma_min, max_value=self.sigma_max)
        elif self.sigma_sample_density_type == "lognormal":
            return partial(rand_log_normal, loc=0.0, scale=1.0)
        elif self.sigma_sample_density_type == "loguniform":
            return partial(rand_log_uniform, min_value=self.sigma_min, max_value=self.sigma_max)
        elif self.sigma_sample_density_type == "uniform":
            return partial(rand_uniform, min_value=self.sigma_min, max_value=self.sigma_max)
        elif self.sigma_sample_density_type == "v-diffusion":
            return partial(rand_v_diffusion, sigma_data=self.sigma_data, min_value=self.sigma_min, max_value=self.sigma_max)
        else:
            raise ValueError(f"Unknown sigma sample density type: {self.sigma_sample_density_type}")

    def _get_noise_schedule(self, n_steps: int) -> Tensor:
        """Get the noise schedule for sampling."""
        device = get_device_from_parameters(self)
        schedule_fn = NOISE_SCHEDULES.get(self.noise_scheduler_type)
        if schedule_fn is None:
            raise ValueError(f"Unknown noise schedule type: {self.noise_scheduler_type}")
        if self.noise_scheduler_type == "karras":
            return schedule_fn(n_steps, self.sigma_min, self.sigma_max, 7, device)
        elif self.noise_scheduler_type in ("exponential", "linear"):
            return schedule_fn(n_steps, self.sigma_min, self.sigma_max, device)
        elif self.noise_scheduler_type in ("vp", "cosine_beta"):
            return schedule_fn(n_steps, device=device)
        elif self.noise_scheduler_type in ("ve", "iddpm"):
            return schedule_fn(n_steps, self.sigma_min, self.sigma_max, device=device)
        else:
            return schedule_fn(n_steps, device=device)

    def generate_actions(self, batch: dict[str, Tensor]) -> Tensor:
        """Generate actions via iterative denoising."""
        device = get_device_from_parameters(self)
        dtype = get_dtype_from_parameters(self)

        perceptual_emb = self._encode_observations(batch)
        batch_size = perceptual_emb.shape[0]

        # For now, use a zero goal embedding (can be extended for language conditioning)
        goal = torch.zeros(batch_size, self.config.goal_seq_len, self.config.goal_dim, device=device, dtype=dtype)

        # Sample initial noise
        x = torch.randn(batch_size, self.horizon, self.action_dim, device=device, dtype=dtype) * self.sigma_max

        # Get noise schedule
        sigmas = self._get_noise_schedule(self.num_sampling_steps)

        # Run sampler
        sampler_fn = SAMPLERS.get(self.sampler_type)
        if sampler_fn is None:
            raise ValueError(f"Unknown sampler type: {self.sampler_type}")
        actions = sampler_fn(self.denoiser, perceptual_emb, x, goal, sigmas)

        return actions

    def compute_loss(self, batch: dict[str, Tensor]) -> Tensor:
        """Compute EDM score matching loss."""
        device = get_device_from_parameters(self)

        perceptual_emb = self._encode_observations(batch)
        actions = batch[ACTION]
        batch_size = actions.shape[0]

        # Zero goal for now
        goal = torch.zeros(batch_size, self.config.goal_seq_len, self.config.goal_dim, device=device, dtype=actions.dtype)

        # Sample noise levels
        sigmas = self._make_sample_density()(shape=(batch_size,), device=device).to(device)
        noise = torch.randn_like(actions)

        # Compute score matching loss
        loss, _ = self.denoiser.loss(perceptual_emb, actions, goal, noise, sigmas)

        # Mask loss for padded actions
        if self.config.do_mask_loss_for_padding and "action_is_pad" in batch:
            in_episode_bound = ~batch["action_is_pad"]
            # loss is already reduced, so we don't need per-element masking here
            # The loss from GCDenoiser.loss is already a scalar mean

        return loss


# ==================== BESO Policy ====================


class BESOPolicy(PreTrainedPolicy):
    """BESO Policy: BEhavior generation with Score-based diffusiOn.

    Uses EDM-style score matching diffusion with an MDT Transformer backbone.
    """

    config_class = BESOConfig
    name = "beso"

    def __init__(self, config: BESOConfig, **kwargs):
        super().__init__(config)
        config.validate_features()
        self.config = config

        self._queues = None
        self.beso = BESOModel(config)
        self.reset()

    def get_optim_params(self) -> dict:
        return self.beso.parameters()

    def reset(self):
        """Clear observation and action queues."""
        self._queues = {
            OBS_STATE: deque(maxlen=self.config.n_obs_steps),
            ACTION: deque(maxlen=self.config.n_action_steps),
        }
        if self.config.image_features:
            self._queues[OBS_IMAGES] = deque(maxlen=self.config.n_obs_steps)
        if self.config.env_state_feature:
            self._queues[OBS_ENV_STATE] = deque(maxlen=self.config.n_obs_steps)

    @torch.no_grad()
    def predict_action_chunk(self, batch: dict[str, Tensor], noise: Tensor | None = None) -> Tensor:
        """Predict a chunk of actions given environment observations."""
        batch = {k: torch.stack(list(self._queues[k]), dim=1) for k in batch if k in self._queues}
        actions = self.beso.generate_actions(batch)
        return actions

    @torch.no_grad()
    def select_action(self, batch: dict[str, Tensor], noise: Tensor | None = None) -> Tensor:
        """Select a single action given environment observations."""
        if ACTION in batch:
            batch.pop(ACTION)

        if self.config.image_features:
            batch = dict(batch)
            batch[OBS_IMAGES] = torch.stack([batch[key] for key in self.config.image_features], dim=-4)

        self._queues = populate_queues(self._queues, batch)

        if len(self._queues[ACTION]) == 0:
            actions = self.predict_action_chunk(batch, noise=noise)
            self._queues[ACTION].extend(actions.transpose(0, 1))

        action = self._queues[ACTION].popleft()
        return action

    def forward(self, batch: dict[str, Tensor]) -> tuple[Tensor, None]:
        """Run the batch through the model and compute the loss for training or validation."""
        if self.config.image_features:
            batch = dict(batch)
            for key in self.config.image_features:
                if self.config.n_obs_steps == 1 and batch[key].ndim == 4:
                    batch[key] = batch[key].unsqueeze(1)
            batch[OBS_IMAGES] = torch.stack([batch[key] for key in self.config.image_features], dim=-4)
        loss = self.beso.compute_loss(batch)
        return loss, None
