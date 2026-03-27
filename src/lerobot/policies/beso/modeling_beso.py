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
from lerobot.utils.constants import (
    ACTION, OBS_ENV_STATE, OBS_IMAGES, OBS_LANGUAGE_ATTENTION_MASK, OBS_LANGUAGE_TOKENS, OBS_STATE,
)


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
    return (torch.randn(shape, device=device, dtype=dtype) * scale + loc).exp()


def rand_log_uniform(shape, min_value, max_value, device="cpu", dtype=torch.float32) -> Tensor:
    min_value = math.log(min_value)
    max_value = math.log(max_value)
    return (torch.rand(shape, device=device, dtype=dtype) * (max_value - min_value) + min_value).exp()


def rand_uniform(shape, min_value, max_value, device="cpu", dtype=torch.float32) -> Tensor:
    return torch.rand(shape, device=device, dtype=dtype) * (max_value - min_value) + min_value


def rand_v_diffusion(
    shape, sigma_data=1.0, min_value=0.0, max_value=float("inf"), device="cpu", dtype=torch.float32
) -> Tensor:
    min_cdf = math.atan(min_value / sigma_data) * 2 / math.pi
    max_cdf = math.atan(max_value / sigma_data) * 2 / math.pi
    u = torch.rand(shape, device=device, dtype=dtype) * (max_cdf - min_cdf) + min_cdf
    return torch.tan(u * math.pi / 2) * sigma_data


# ==================== Noise Schedules ====================


def get_sigmas_karras(n, sigma_min, sigma_max, rho=7.0, device="cpu") -> Tensor:
    ramp = torch.linspace(0, 1, n)
    min_inv_rho = sigma_min ** (1 / rho)
    max_inv_rho = sigma_max ** (1 / rho)
    sigmas = (max_inv_rho + ramp * (min_inv_rho - max_inv_rho)) ** rho
    return append_zero(sigmas).to(device)


def get_sigmas_exponential(n, sigma_min, sigma_max, device="cpu") -> Tensor:
    sigmas = torch.linspace(math.log(sigma_max), math.log(sigma_min), n, device=device).exp()
    return append_zero(sigmas)


def get_sigmas_linear(n, sigma_min, sigma_max, device="cpu") -> Tensor:
    sigmas = torch.linspace(sigma_max, sigma_min, n, device=device)
    return append_zero(sigmas)


def get_sigmas_vp(n, beta_d=19.9, beta_min=0.1, eps_s=1e-3, device="cpu") -> Tensor:
    t = torch.linspace(1, eps_s, n, device=device)
    sigmas = torch.sqrt(torch.exp(beta_d * t**2 / 2 + beta_min * t) - 1)
    return append_zero(sigmas)


def cosine_beta_schedule(n, s=0.008, device="cpu") -> Tensor:
    steps = n + 1
    x = np.linspace(0, steps, steps)
    alphas_cumprod = np.cos(((x / steps) + s) / (1 + s) * np.pi * 0.5) ** 2
    alphas_cumprod = alphas_cumprod / alphas_cumprod[0]
    betas = 1 - (alphas_cumprod[1:] / alphas_cumprod[:-1])
    betas_clipped = np.clip(betas, a_min=0, a_max=0.999)
    return append_zero(torch.tensor(np.flip(betas_clipped).copy(), device=device, dtype=torch.float32))


def get_sigmas_ve(n, sigma_min=0.02, sigma_max=100, device="cpu") -> Tensor:
    t = torch.linspace(0, n, n, device=device)
    t = (sigma_max**2) * ((sigma_min**2 / sigma_max**2) ** (t / (n - 1)))
    sigmas = torch.sqrt(t)
    return append_zero(sigmas)


def get_iddpm_sigmas(n, sigma_min=0.02, sigma_max=100, M=1000, j_0=0, C_1=0.001, C_2=0.008, device="cpu"):
    step_indices = torch.arange(n, dtype=torch.float64, device=device)
    u = torch.zeros(M + 1, dtype=torch.float64, device=device)
    alpha_bar = lambda j: (0.5 * np.pi * j / M / (C_2 + 1)).sin() ** 2  # noqa: E731
    for j in torch.arange(M, j_0, -1, device=device):
        u[j - 1] = ((u[j] ** 2 + 1) / (alpha_bar(j - 1) / alpha_bar(j)).clip(min=C_1) - 1).sqrt()
    u_filtered = u[torch.logical_and(u >= sigma_min, u <= sigma_max)]
    sigmas = u_filtered[((len(u_filtered) - 1) / (n - 1) * step_indices).round().to(torch.int64)]
    return append_zero(sigmas).to(torch.float32)


# ==================== Sampling Functions ====================


def to_d(action: Tensor, sigma: Tensor, denoised: Tensor) -> Tensor:
    return (action - denoised) / append_dims(sigma, action.ndim)


def get_ancestral_step(sigma_from, sigma_to, eta=1.0):
    if not eta:
        return sigma_to, 0.0
    sigma_up = min(sigma_to, eta * (sigma_to**2 * (sigma_from**2 - sigma_to**2) / sigma_from**2) ** 0.5)
    sigma_down = (sigma_to**2 - sigma_up**2) ** 0.5
    return sigma_down, sigma_up


@torch.no_grad()
def sample_ddim(model, state, action, goal, sigmas, scaler=None, disable=True, **extra_args):
    s_in = action.new_ones([action.shape[0]])
    sigma_fn = lambda t: t.neg().exp()  # noqa: E731
    t_fn = lambda sigma: sigma.log().neg()  # noqa: E731
    for i in trange(len(sigmas) - 1, disable=disable):
        denoised = model(state, action, goal, sigmas[i] * s_in)
        t, t_next = t_fn(sigmas[i]), t_fn(sigmas[i + 1])
        h = t_next - t
        action = (sigma_fn(t_next) / sigma_fn(t)) * action - (-h).expm1() * denoised
    return action


@torch.no_grad()
def sample_euler(model, state, action, goal, sigmas, scaler=None, disable=True,
                 s_churn=0.0, s_tmin=0.0, s_tmax=float("inf"), s_noise=1.0, **extra_args):
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
def sample_heun(model, state, action, goal, sigmas, scaler=None, disable=True,
                s_churn=0.0, s_tmin=0.0, s_tmax=float("inf"), s_noise=1.0, **extra_args):
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
    s_in = action.new_ones([action.shape[0]])
    sigma_fn = lambda t: t.neg().exp()  # noqa: E731
    t_fn = lambda sigma: sigma.log().neg()  # noqa: E731
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
    sigma_fn = lambda t: t.neg().exp()  # noqa: E731
    t_fn = lambda sigma: sigma.log().neg()  # noqa: E731
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


SAMPLERS = {
    "ddim": sample_ddim,
    "euler": sample_euler,
    "heun": sample_heun,
    "euler_ancestral": sample_euler_ancestral,
    "dpmpp_2m": sample_dpmpp_2m,
    "dpmpp_2s": sample_dpmpp_2s,
}

NOISE_SCHEDULES = {
    "karras": get_sigmas_karras,
    "exponential": get_sigmas_exponential,
    "linear": get_sigmas_linear,
    "vp": get_sigmas_vp,
    "cosine_beta": cosine_beta_schedule,
    "ve": get_sigmas_ve,
    "iddpm": get_iddpm_sigmas,
}




class BESORMSNorm(nn.Module):
    def __init__(self, dim: int, eps: float = 1e-6):
        super().__init__()
        self.scale = dim ** -0.5
        self.eps = eps
        self.g = nn.Parameter(torch.ones(dim))

    def forward(self, x: Tensor) -> Tensor:
        norm = torch.norm(x, dim=-1, keepdim=True) * self.scale
        return x / norm.clamp(min=self.eps) * self.g


class BESOSwishGLU(nn.Module):

    def __init__(self, in_dim: int, out_dim: int):
        super().__init__()
        self.act = nn.SiLU()
        self.project = nn.Linear(in_dim, 2 * out_dim)

    def forward(self, x: Tensor) -> Tensor:
        projected, gate = self.project(x).tensor_split(2, dim=-1)
        return projected * self.act(gate)


class BESOAttention(nn.Module):
    """Multi-head attention with QK-norm and optional causal masking."""

    def __init__(self, n_embd: int, n_head: int, attn_pdrop: float, resid_pdrop: float,
                 block_size: int = 100, causal: bool = False, bias: bool = False, qk_norm: bool = True):
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
        self.qk_norm = qk_norm
        if self.qk_norm:
            self.q_norm = BESORMSNorm(n_embd // self.n_head, eps=1e-6)
            self.k_norm = BESORMSNorm(n_embd // self.n_head, eps=1e-6)
        else:
            self.q_norm = self.k_norm = nn.Identity()

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
        q = self.q_norm(q)
        k = self.k_norm(k)
        y = F.scaled_dot_product_attention(
            q, k, v, dropout_p=self.attn_dropout.p if self.training else 0, is_causal=self.causal
        )
        y = y.transpose(1, 2).contiguous().view(B, T, C)
        y = self.resid_dropout(self.c_proj(y))
        return y


class BESOMLP(nn.Module):

    def __init__(self, n_embd: int, bias: bool = False, dropout: float = 0.0):
        super().__init__()
        self.mlp = nn.Sequential(
            BESOSwishGLU(n_embd, 4 * n_embd),
            nn.Linear(4 * n_embd, n_embd, bias=bias),
            nn.Dropout(dropout),
        )

    def forward(self, x: Tensor) -> Tensor:
        return self.mlp(x)


class BESOBlock(nn.Module):
    """Transformer block with RMSNorm, SwishGLU MLP, QK-norm attention."""

    def __init__(self, n_embd: int, n_heads: int, attn_pdrop: float, resid_pdrop: float,
                 mlp_pdrop: float, block_size: int = 100, causal: bool = True,
                 use_cross_attention: bool = False, bias: bool = False, qk_norm: bool = True):
        super().__init__()
        self.ln_1 = BESORMSNorm(n_embd, eps=1e-6)
        self.attn = BESOAttention(n_embd, n_heads, attn_pdrop, resid_pdrop, block_size, causal, bias, qk_norm)
        self.use_cross_attention = use_cross_attention
        if self.use_cross_attention:
            self.cross_att = BESOAttention(n_embd, n_heads, attn_pdrop, resid_pdrop, block_size, causal, bias, qk_norm)
            self.ln3 = BESORMSNorm(n_embd, eps=1e-6)
        self.ln_2 = BESORMSNorm(n_embd, eps=1e-6)
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
        self.modulation = nn.Sequential(nn.SiLU(), nn.Linear(hidden_size, 6 * hidden_size, bias=True))

    def forward(self, c: Tensor):
        return self.modulation(c).chunk(6, dim=-1)


class BESOConditionedBlock(BESOBlock):
    """Block with AdaLN-Zero conditioning."""

    def __init__(self, n_embd, n_heads, attn_pdrop, resid_pdrop, mlp_pdrop,
                 block_size=100, causal=True, use_cross_attention=False, bias=False, qk_norm=True):
        super().__init__(n_embd, n_heads, attn_pdrop, resid_pdrop, mlp_pdrop, block_size,
                         causal, use_cross_attention, bias, qk_norm)
        self.adaLN_zero = BESOAdaLNZero(n_embd)

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

    def __init__(self, embed_dim: int, n_heads: int, attn_pdrop: float, resid_pdrop: float,
                 n_layers: int, block_size: int = 100, causal: bool = True, bias: bool = False,
                 mlp_pdrop: float = 0.0, qk_norm: bool = True):
        super().__init__()
        self.blocks = nn.ModuleList([
            BESOBlock(embed_dim, n_heads, attn_pdrop, resid_pdrop, mlp_pdrop, block_size,
                      causal=causal, bias=bias, qk_norm=qk_norm)
            for _ in range(n_layers)
        ])
        self.ln = BESORMSNorm(embed_dim, eps=1e-6)

    def forward(self, x: Tensor) -> Tensor:
        for layer in self.blocks:
            x = layer(x)
        x = self.ln(x)
        return x


class BESOTransformerFiLMEncoder(nn.Module):
    """Causal Transformer encoder with AdaLN-Zero FiLM conditioning."""

    def __init__(self, embed_dim: int, n_heads: int, attn_pdrop: float, resid_pdrop: float,
                 n_layers: int, block_size: int = 100, causal: bool = True, bias: bool = False,
                 mlp_pdrop: float = 0.0, qk_norm: bool = True):
        super().__init__()
        self.blocks = nn.ModuleList([
            BESOConditionedBlock(embed_dim, n_heads, attn_pdrop, resid_pdrop, mlp_pdrop, block_size,
                                 causal=causal, bias=bias, qk_norm=qk_norm)
            for _ in range(n_layers)
        ])
        self.ln = BESORMSNorm(embed_dim, eps=1e-6)

    def forward(self, x: Tensor, c: Tensor) -> Tensor:
        for layer in self.blocks:
            x = layer(x, c)
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



class BESOInnerModel(nn.Module):

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
        n_layers: int,
        n_heads: int,
        goal_seq_len: int,
        obs_seq_len: int,
        action_seq_len: int,
        goal_drop: float = 0.1,
        bias: bool = False,
        linear_output: bool = True,
        use_ada_conditioning: bool = False,
        use_pos_emb: bool = False,
    ):
        super().__init__()
        self.obs_dim = obs_dim
        self.embed_dim = embed_dim
        self.action_dim = action_dim
        self.goal_seq_len = goal_seq_len
        self.obs_seq_len = obs_seq_len
        self.action_seq_len = action_seq_len
        self.use_ada_conditioning = use_ada_conditioning
        self.use_pos_emb = use_pos_emb

        seq_size = goal_seq_len + obs_seq_len + action_seq_len
        block_size = seq_size + 1  # +1 for sigma token

        # Token embeddings
        self.tok_emb = nn.Linear(obs_dim, embed_dim)
        self.goal_emb = nn.Linear(goal_dim, embed_dim)
        self.action_emb = nn.Linear(action_dim, embed_dim)
        self.drop = nn.Dropout(embed_pdrop)
        self.cond_mask_prob = goal_drop

        if use_pos_emb:
            self.pos_emb = nn.Parameter(torch.zeros(1, seq_size, embed_dim))
        else:
            self.pos_emb = nn.Parameter(torch.zeros(1, goal_seq_len + action_seq_len, embed_dim))

        self.sigma_emb = nn.Sequential(
            BESOSinusoidalPosEmb(embed_dim),
            nn.Linear(embed_dim, embed_dim * 2),
            nn.Mish(),
            nn.Linear(embed_dim * 2, embed_dim),
        )

        # Single causal TransformerEncoder (not encoder+decoder!)
        if use_ada_conditioning:
            self.encoder = BESOTransformerFiLMEncoder(
                embed_dim=embed_dim, n_heads=n_heads, attn_pdrop=attn_pdrop,
                resid_pdrop=resid_pdrop, n_layers=n_layers, block_size=block_size,
                causal=True, bias=bias, mlp_pdrop=mlp_pdrop, qk_norm=True,
            )
        else:
            self.encoder = BESOTransformerEncoder(
                embed_dim=embed_dim, n_heads=n_heads, attn_pdrop=attn_pdrop,
                resid_pdrop=resid_pdrop, n_layers=n_layers, block_size=block_size,
                causal=True, bias=bias, mlp_pdrop=mlp_pdrop, qk_norm=True,
            )

        # Output head
        if linear_output:
            self.action_pred = nn.Linear(embed_dim, action_dim)
        else:
            self.action_pred = nn.Sequential(
                nn.Linear(embed_dim, 100), nn.GELU(), nn.Linear(100, action_dim)
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
        elif isinstance(module, BESOInnerModel):
            torch.nn.init.normal_(module.pos_emb, mean=0.0, std=0.02)

    def forward(self, states: Tensor, actions: Tensor, goals: Tensor, sigma: Tensor) -> Tensor:
        b, t, dim = states.size()
        _, t_a, _ = actions.size()

        # Embed goal tokens + positional embedding
        goal_embed = self.goal_emb(goals)
        goal_embed = goal_embed + self.pos_emb[:, :self.goal_seq_len, :]
        goal_x = self.drop(goal_embed)

        # Embed action tokens + positional embedding
        action_embed = self.action_emb(actions)
        action_embed = action_embed + self.pos_emb[:, self.goal_seq_len:(self.goal_seq_len + t_a), :]
        action_x = self.drop(action_embed)

        # Embed observation tokens (pos_emb only if use_pos_emb=True)
        state_embed = self.tok_emb(states)
        if self.use_pos_emb:
            state_embed = state_embed + self.pos_emb[:, (self.goal_seq_len + t_a):(self.goal_seq_len + t_a + t), :]
        state_x = self.drop(state_embed)

        # Sigma embedding → 1 token
        sigmas = sigma.log() / 4
        sigmas = einops.rearrange(sigmas, "b -> b 1")
        emb_t = self.sigma_emb(sigmas)
        if len(emb_t.shape) == 2:
            emb_t = einops.rearrange(emb_t, "b d -> b 1 d")

        input_seq = torch.cat([emb_t, goal_x, state_x, action_x], dim=1)

        # Run through single causal encoder
        if self.use_ada_conditioning:
            encoder_output = self.encoder(input_seq, emb_t)
        else:
            encoder_output = self.encoder(input_seq)

        # Extract last action_seq_len tokens → action prediction
        pred_actions = self.action_pred(encoder_output[:, -self.action_seq_len:, :])
        return pred_actions


# ==================== GCDenoiser (Karras EDM Preconditioner) ====================


class GCDenoiser(nn.Module):
    """Karras et al. preconditioner wrapping the inner model with c_skip, c_out, c_in scalings."""

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
    """Encodes an RGB image into a 1D feature vector using ResNet + SpatialSoftmax."""

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
            # Replace BatchNorm with GroupNorm for deterministic normalization.
            # This is done even with pretrained weights (pretrained BN params are discarded).
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


class BESOLanguageEncoder(nn.Module):

    def __init__(self, clip_model_name: str = "openai/clip-vit-base-patch32", goal_dim: int = 512, freeze: bool = True):
        super().__init__()
        from transformers import CLIPTextModel  # lazy import to avoid hard dependency

        self.clip_text = CLIPTextModel.from_pretrained(clip_model_name)
        clip_dim = self.clip_text.config.hidden_size  # 512 for ViT-B/32

        if clip_dim != goal_dim:
            self.proj = nn.Linear(clip_dim, goal_dim)
        else:
            self.proj = None

        if freeze:
            for param in self.clip_text.parameters():
                param.requires_grad = False

    @torch.no_grad()
    def forward(self, input_ids: Tensor, attention_mask: Tensor | None = None) -> Tensor:
        """Encode text tokens into goal embedding.

        Args:
            input_ids: (B, seq_len) tokenized text
            attention_mask: (B, seq_len) attention mask

        Returns:
            (B, 1, goal_dim) goal embedding
        """
        outputs = self.clip_text(input_ids=input_ids, attention_mask=attention_mask)
        # Use the pooled output (CLS token) — shape (B, clip_dim)
        emb = outputs.pooler_output.float()
        if self.proj is not None:
            emb = self.proj(emb)
        return emb.unsqueeze(1)  # (B, 1, goal_dim)


class BESOModel(nn.Module):
    """Core BESO model: image encoding + CLIP goal conditioning + GCDenoiser + EDM sampling."""

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

        # Language encoder for goal conditioning
        if config.use_language_conditioning:
            self.language_encoder = BESOLanguageEncoder(
                clip_model_name=config.clip_model_name,
                goal_dim=config.goal_dim,
                freeze=config.freeze_clip,
            )
            self._has_language = True
        else:
            self._has_language = False

        # Action dimension
        action_dim = config.action_feature.shape[0]

        # Compute the actual observation sequence length that _encode_observations produces.
        obs_seq_len = config.n_obs_steps
        if config.robot_state_feature:
            obs_seq_len += config.n_obs_steps

        inner_model = BESOInnerModel(
            obs_dim=config.embed_dim,
            goal_dim=config.goal_dim,
            action_dim=action_dim,
            embed_dim=config.embed_dim,
            embed_pdrop=config.embed_pdrop,
            attn_pdrop=config.attn_pdrop,
            resid_pdrop=config.resid_pdrop,
            mlp_pdrop=config.mlp_pdrop,
            n_layers=config.n_enc_layers,  # single encoder uses n_enc_layers
            n_heads=config.n_heads,
            goal_seq_len=config.goal_seq_len,
            obs_seq_len=obs_seq_len,
            action_seq_len=config.horizon,
            goal_drop=config.goal_drop,
            bias=False,
            linear_output=config.linear_output,
            use_ada_conditioning=config.use_ada_conditioning,
            use_pos_emb=False,  # matching real_robot config: use_pos_emb=False
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
            obs_features = torch.cat(features_list, dim=-1)
            if self.obs_proj is not None:
                obs_features = self.obs_proj(obs_features)
        else:
            obs_features = torch.zeros(batch_size, n_obs_steps, self.config.embed_dim, device=batch[OBS_STATE].device)

        # Add state embedding as extra tokens
        if self._has_state:
            state_tokens = self.state_emb(batch[OBS_STATE])
            obs_features = torch.cat([obs_features, state_tokens], dim=1)

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

    def _encode_goal(self, batch: dict[str, Tensor], batch_size: int, device, dtype) -> Tensor:
        """Encode goal from language instruction or return zeros."""
        if self._has_language and OBS_LANGUAGE_TOKENS in batch:
            goal = self.language_encoder(
                input_ids=batch[OBS_LANGUAGE_TOKENS],
                attention_mask=batch.get(OBS_LANGUAGE_ATTENTION_MASK),
            ).to(dtype)
        else:
            goal = torch.zeros(batch_size, self.config.goal_seq_len, self.config.goal_dim,
                               device=device, dtype=dtype)
        return goal

    def generate_actions(self, batch: dict[str, Tensor]) -> Tensor:
        """Generate actions via iterative denoising."""
        device = get_device_from_parameters(self)
        dtype = get_dtype_from_parameters(self)
        perceptual_emb = self._encode_observations(batch)
        batch_size = perceptual_emb.shape[0]
        goal = self._encode_goal(batch, batch_size, device, dtype)
        x = torch.randn(batch_size, self.horizon, self.action_dim, device=device, dtype=dtype) * self.sigma_max
        sigmas = self._get_noise_schedule(self.num_sampling_steps)
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
        goal = self._encode_goal(batch, batch_size, device, actions.dtype)
        sigmas = self._make_sample_density()(shape=(batch_size,), device=device).to(device)
        noise = torch.randn_like(actions)
        loss, _ = self.denoiser.loss(perceptual_emb, actions, goal, noise, sigmas)
        return loss


# ==================== BESO Policy ====================


class BESOPolicy(PreTrainedPolicy):
    """BESO Policy: BEhavior generation with Score-based diffusiOn.

    Uses EDM-style score matching diffusion with an encoder-only Transformer backbone.
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
        batch = {k: torch.stack(list(self._queues[k]), dim=1) for k in batch if k in self._queues}
        actions = self.beso.generate_actions(batch)
        return actions

    @torch.no_grad()
    def select_action(self, batch: dict[str, Tensor], noise: Tensor | None = None) -> Tensor:
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
        """Run the batch through the model and compute the loss for training."""
        if self.config.image_features:
            batch = dict(batch)
            for key in self.config.image_features:
                if self.config.n_obs_steps == 1 and batch[key].ndim == 4:
                    batch[key] = batch[key].unsqueeze(1)
            batch[OBS_IMAGES] = torch.stack([batch[key] for key in self.config.image_features], dim=-4)
        loss = self.beso.compute_loss(batch)
        return loss, None
