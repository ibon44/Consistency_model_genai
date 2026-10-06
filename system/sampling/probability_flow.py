
from typing import Any, Optional, Tuple

import numpy as np
import torch
from scipy import integrate
from torch import Tensor

from diffusion.base import DiffusionProcess

from .base import Sampler


class ProbabilityFlow(Sampler):
    def __init__(
        self,
        process: Any,
        atol: float = 1e-5,
        rtol: float = 1e-5,
        method: str = 'RK45',
        eps: float = 1e-3
    ) -> None:
        super().__init__(process)
        self.atol = atol
        self.rtol = rtol
        self.method = method
        self.eps = eps

    def sample(
        self,
        shape: Tuple[int, ...],
        score_model: Any,
        device: str = 'cpu',
        gamma: float = 1.0
    ) -> Tensor:
        """Samples by solving the probability flow ODE.

        The probability flow ODE dx/dt = f(x,t) - 0.5*g(t)^2*score(x,t)
        shares its marginals with the reverse-time SDE (Song et al.,
        2021, Eq. 13). `gamma` should stay at its default of 1.0 to solve
        the true probability flow ODE; other values are an unofficial
        guidance-like knob and will not sample from the model's actual
        marginals.

        Args:
            shape: The shape of the tensor to sample.
            score_model: Score model callable, invoked as score_model(x, t).
            device: Device on which to run the integration.
            gamma: Scale applied to the score term (see above).

        Returns:
            Tuple of (initial prior noise, final sample in [0, 1]).
        """
        x_T = self.process.sample_prior(shape, device=device)
        prior = x_T.clone()

        # ODE function for the integration
        def ode_func(t: float, x_flat: np.ndarray) -> np.ndarray:
            t_tensor = torch.tensor(t, dtype=torch.float32, device=device).reshape(1)
            x = torch.tensor(x_flat, dtype=torch.float32, device=device).reshape(shape)

            drift = self.process.drift_coefficient(x, t_tensor)
            diffusion = self.process.diffusion_coefficient(t_tensor)

            with torch.no_grad():
                score = score_model(x, t_tensor)

            dx_dt = drift - 0.5 * (diffusion ** 2) * (gamma * score)

            return dx_dt.detach().cpu().numpy().flatten()

        # Solve the ODE from t=1 down to t=eps (not exactly 0, for
        # numerical stability near the data distribution).
        t_span = (1.0, self.eps)
        x_init = x_T.detach().cpu().numpy().flatten()

        solution = integrate.solve_ivp(
            ode_func,
            t_span,
            x_init,
            method=self.method,
            atol=self.atol,
            rtol=self.rtol
        )

        # Solution at t=eps, still carrying the process' mean scaling
        # mu_t(x_0, eps) = E[x_eps | x_0].
        x_eps_flat = solution.y[:, -1]
        x_eps = torch.tensor(x_eps_flat, dtype=torch.float32, device=device).reshape(shape)

        # Undo the mean scaling to recover x_0 ~= x_eps / mu_coefficient(eps).
        t_eps_tensor = torch.full((shape[0],), self.eps, device=device)
        mu_coeff = self.process.mu_t(
            torch.ones_like(x_eps), t_eps_tensor
        )
        x_0 = x_eps / mu_coeff

        # Revert the [-1, 1] normalization to [0, 1] for visualization.
        x_0 = (x_0 + 1) / 2

        return prior, x_0
