import torch
import numpy as np
from ddpm.utils import batch_mul, DataClass
from ddpm.typing import Function


def get_mean_scale_reverse_fn(
    score_fn: Function,
    mean_scale_fn: Function,
    eps: float = 1e-3,
    T: float = 1.0,
    N: int = 1000,
):
    def simulate_reverse_diffusion(x_T: torch.Tensor):
        shape = x_T.shape
        B = shape[0]

        timesteps = torch.linspace(T, eps, N).to(x_T.device)

        def loop_body(i, val):
            x, x_mean = val
            t = timesteps[i]
            vec_t = (torch.ones(B).to(x_T.device) * t).reshape(B, 1)
            x_mean, scale = mean_scale_fn(x, vec_t, score_fn)
            noise = torch.randn(x.shape).to(x_T.device)
            x = x_mean + batch_mul(scale, noise)
            return x, x_mean

        loop_state = (x_T, x_T)
        for i in range(N):
            loop_state = loop_body(i, loop_state)
        x, x_mean = loop_state

        return x, x_mean

    return simulate_reverse_diffusion


class Diffusion(torch.nn.Module):
    """ """

    def __init__(self, beta_min=0.1, beta_max=20, N=1000, eps=1e-3, T=1.0) -> None:
        super().__init__()
        self.register_buffer("beta_0", torch.tensor(beta_min))
        self.register_buffer("beta_1", torch.tensor(beta_max))
        self.N = N
        self.T = T
        self.eps = eps

        discrete_betas = torch.linspace(beta_min / N, beta_max / N, N)  # to gpu
        self.register_buffer("discrete_betas", discrete_betas)

        alphas = 1.0 - self.discrete_betas
        self.register_buffer("alphas", alphas)

        alphas_cumprod = torch.cumprod(self.alphas, axis=0)
        self.register_buffer("alphas_cumprod", alphas_cumprod)

        sqrt_alphas_cumprod = torch.sqrt(self.alphas_cumprod)
        self.register_buffer("sqrt_alphas_cumprod", sqrt_alphas_cumprod)

        sqrt_1m_alphas_cumprod = torch.sqrt(1.0 - self.alphas_cumprod)
        self.register_buffer("sqrt_1m_alphas_cumprod", sqrt_1m_alphas_cumprod)

    def sample_t(self, x_0: torch.Tensor) -> torch.Tensor:
        # return Categorical(probs=torch.ones(self.N)).sample((x_0.shape[0],))
        indices_np = np.random.choice(self.N, size=(x_0.shape[0],))
        indices = torch.from_numpy(indices_np).long().to(x_0.device)
        return indices

    def sample_x(self, x_0: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        noise = torch.randn_like(x_0)
        x_t = batch_mul(self.sqrt_alphas_cumprod[t], x_0) + batch_mul(
            self.sqrt_1m_alphas_cumprod[t], noise
        )
        return DataClass(x_t=x_t, z=noise, t=t)

    def loss(self, model_output: torch.Tensor, noise: torch.Tensor) -> torch.Tensor:
        losses = torch.square(model_output - noise)
        losses = torch.mean(losses.reshape((losses.shape[0], -1)), axis=-1)
        loss = torch.mean(losses)

        return loss

    def reverse_mean_scale_function(
        self, x_t: torch.Tensor, t: torch.Tensor, score_fn: Function
    ) -> torch.Tensor:
        timestep = t * (self.N - 1) / self.T
        t_label = timestep.type(torch.int64)
        beta = self.discrete_betas[t_label]

        model_pred = score_fn(x_t, timestep)
        std = self.sqrt_1m_alphas_cumprod[t_label.type(torch.int64)]
        score = -batch_mul(model_pred, 1.0 / std)
        x_mean = batch_mul((x_t + batch_mul(beta, score)), 1.0 / torch.sqrt(1.0 - beta))
        return x_mean, torch.sqrt(beta)

    def reverse_sample(self, x_T: torch.Tensor, score_fn: Function):
        sample_fn = get_mean_scale_reverse_fn(
            score_fn=score_fn,
            mean_scale_fn=self.reverse_mean_scale_function,
            N=self.N,
            T=self.T,
        )
        with torch.no_grad():
            return sample_fn(x_T)
