"""Ranger optimizer: RAdam with Lookahead and gradient centralization."""

import math

import torch
from torch.optim import Optimizer


class Ranger(Optimizer):
    def __init__(
        self,
        params,
        lr=1e-3,
        alpha=0.5,
        k=6,
        N_sma_threshhold=5,
        betas=(0.95, 0.999),
        eps=1e-5,
        weight_decay=0,
        use_gc=True,
        gc_conv_only=False,
    ):
        if not 0 <= alpha <= 1:
            raise ValueError(f"Invalid slow update rate: {alpha}")
        if k < 1:
            raise ValueError(f"Invalid lookahead steps: {k}")
        if lr <= 0:
            raise ValueError(f"Invalid learning rate: {lr}")
        if eps <= 0:
            raise ValueError(f"Invalid epsilon: {eps}")

        defaults = dict(
            lr=lr,
            alpha=alpha,
            k=k,
            step_counter=0,
            betas=betas,
            N_sma_threshhold=N_sma_threshhold,
            eps=eps,
            weight_decay=weight_decay,
        )
        super().__init__(params, defaults)
        self.N_sma_threshhold = N_sma_threshhold
        self.alpha = alpha
        self.k = k
        self.radam_buffer = [[None, None, None] for _ in range(10)]
        self.use_gc = use_gc
        self.gc_gradient_threshold = 3 if gc_conv_only else 1

    def __setstate__(self, state):
        super().__setstate__(state)

    @torch.no_grad()
    def step(self, closure=None):
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()

        for group in self.param_groups:
            for parameter in group["params"]:
                if parameter.grad is None:
                    continue
                gradient = parameter.grad.float()
                if gradient.is_sparse:
                    raise RuntimeError("Ranger does not support sparse gradients")

                parameter_fp32 = parameter.float()
                state = self.state[parameter]
                if not state:
                    state["step"] = 0
                    state["exp_avg"] = torch.zeros_like(parameter_fp32)
                    state["exp_avg_sq"] = torch.zeros_like(parameter_fp32)
                    state["slow_buffer"] = parameter.detach().clone()
                else:
                    state["exp_avg"] = state["exp_avg"].type_as(parameter_fp32)
                    state["exp_avg_sq"] = state["exp_avg_sq"].type_as(parameter_fp32)

                exp_avg = state["exp_avg"]
                exp_avg_sq = state["exp_avg_sq"]
                beta1, beta2 = group["betas"]

                if self.use_gc and gradient.dim() > self.gc_gradient_threshold:
                    gradient.add_(
                        -gradient.mean(
                            dim=tuple(range(1, gradient.dim())), keepdim=True
                        )
                    )

                state["step"] += 1
                exp_avg_sq.mul_(beta2).addcmul_(
                    gradient, gradient, value=1 - beta2
                )
                exp_avg.mul_(beta1).add_(gradient, alpha=1 - beta1)

                buffered = self.radam_buffer[state["step"] % 10]
                if state["step"] == buffered[0]:
                    n_sma, step_size = buffered[1], buffered[2]
                else:
                    beta2_t = beta2 ** state["step"]
                    n_sma_max = 2 / (1 - beta2) - 1
                    n_sma = n_sma_max - 2 * state["step"] * beta2_t / (1 - beta2_t)
                    if n_sma > self.N_sma_threshhold:
                        step_size = math.sqrt(
                            (1 - beta2_t)
                            * (n_sma - 4)
                            / (n_sma_max - 4)
                            * (n_sma - 2)
                            / n_sma
                            * n_sma_max
                            / (n_sma_max - 2)
                        ) / (1 - beta1 ** state["step"])
                    else:
                        step_size = 1 / (1 - beta1 ** state["step"])
                    buffered[:] = [state["step"], n_sma, step_size]

                if group["weight_decay"]:
                    parameter_fp32.add_(
                        parameter_fp32,
                        alpha=-group["weight_decay"] * group["lr"],
                    )

                if n_sma > self.N_sma_threshhold:
                    denominator = exp_avg_sq.sqrt().add_(group["eps"])
                    parameter_fp32.addcdiv_(
                        exp_avg,
                        denominator,
                        value=-step_size * group["lr"],
                    )
                else:
                    parameter_fp32.add_(
                        exp_avg, alpha=-step_size * group["lr"]
                    )
                parameter.copy_(parameter_fp32)

                if state["step"] % group["k"] == 0:
                    slow = state["slow_buffer"]
                    slow.add_(parameter - slow, alpha=self.alpha)
                    parameter.copy_(slow)

        return loss
