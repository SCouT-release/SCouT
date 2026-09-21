import torch

DEFAULT_MERGOPT_ALPHAS = (0.1, 0.2, 0.3, 0.4, 0.5, 0.6)
DEFAULT_MERGOPT_K_MAX = 10


def parse_mergopt_alphas(value):
    alphas = tuple(float(alpha) for alpha in value.split(",") if alpha)
    if not alphas:
        raise ValueError("--mergopt-alphas must contain at least one value.")
    return alphas


class MergOPT(torch.optim.Optimizer):
    """Merge-aware optimizer using stochastic Laplace merge offsets."""

    def __init__(
        self,
        params,
        base_optimizer,
        mu=0.0,
        b=0.0005,
        k_max=DEFAULT_MERGOPT_K_MAX,
        alphas=DEFAULT_MERGOPT_ALPHAS,
        **kwargs,
    ):
        if b < 0.0:
            raise ValueError(f"Invalid Laplace scale b, should be non-negative: {b}")
        if k_max is None or k_max < 1:
            raise ValueError(f"Invalid k_max, should be positive: {k_max}")
        alphas = tuple(float(alpha) for alpha in alphas)
        if not alphas:
            raise ValueError("MergOPT requires at least one alpha value.")

        defaults = dict(mu=mu, b=b, k_max=int(k_max), alphas=alphas, **kwargs)
        super().__init__(params, defaults)

        self.base_optimizer = base_optimizer(self.param_groups, **kwargs)
        self.param_groups = self.base_optimizer.param_groups
        self.state = self.base_optimizer.state
        self.defaults.update(self.base_optimizer.defaults)

    def _sample_laplace_like(self, param, mu, b):
        if b == 0.0:
            return torch.full_like(param, mu)

        uniform = torch.rand_like(param)
        distance = torch.abs(uniform - 0.5)
        log_arg = (1 - 2 * distance).clamp_min(torch.finfo(param.dtype).tiny)
        return mu - b * torch.sign(uniform - 0.5) * torch.log(log_arg)

    def _sample_perturbation_coef(self, group):
        alphas = group["alphas"]
        alpha_idx = torch.randint(len(alphas), (1,)).item()
        task_count = torch.randint(1, group["k_max"] + 1, (1,)).item()
        return task_count * alphas[alpha_idx] - 1.0

    @torch.no_grad()
    def _perturb(self):
        backups = []
        for group in self.param_groups:
            coef = self._sample_perturbation_coef(group)
            for param in group["params"]:
                if not param.requires_grad:
                    continue
                backups.append((param, param.detach().clone()))
                offset = self._sample_laplace_like(param, group["mu"], group["b"])
                param.add_(coef * offset)
        return backups

    @torch.no_grad()
    def _restore(self, backups):
        for param, backup in backups:
            param.copy_(backup)

    def step(self, closure=None):
        if closure is None:
            return self.base_optimizer.step()

        backups = self._perturb()
        try:
            with torch.enable_grad():
                loss = closure()
        finally:
            self._restore(backups)

        self.base_optimizer.step()
        return loss

    def load_state_dict(self, state_dict):
        super().load_state_dict(state_dict)
        self.base_optimizer.state = self.state
        self.base_optimizer.param_groups = self.param_groups
