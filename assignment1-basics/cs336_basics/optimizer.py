from collections.abc import Callable, Iterator
from math import cos, pi, sqrt

import torch
from torch import Tensor
from torch.nn import Parameter


class SGD(torch.optim.Optimizer):
    def __init__(self, params, lr=1e-3):
        if lr < 0:
            raise ValueError(f"Invalid learning rate: {lr}")
        defaults = {"lr": lr}
        super().__init__(params, defaults)

    def step(self, closure: Callable | None = None):
        loss = None if closure is None else closure()
        for group in self.param_groups:
            lr = group["lr"]
            for p in group["params"]:
                if p.grad is None:
                    continue
                state = self.state[p]
                t = state.get("t", 0)
                grad = p.grad.data
                p.data -= lr / sqrt(t + 1) * grad
                state["t"] = t + 1
        return loss


class AdamW(torch.optim.Optimizer):
    def __init__(
        self,
        params: Iterator[Parameter],
        lr: float = 1e-3,
        betas: tuple[float, float] = (0.9, 0.999),
        eps: float = 1e-8,
        weight_decay: float = 0.01,
    ):
        if lr < 0:
            raise ValueError(f"Invalid learning rate: {lr}")
        if not (0 <= betas[0] < 1 and 0 <= betas[1] < 1):
            raise ValueError(f"Invalid betas: {betas[0]}, {betas[1]}")
        if eps < 0:
            raise ValueError(f"Invalid eps: {eps}")
        if weight_decay < 0:
            raise ValueError(f"Invalid weight_decay: {weight_decay}")
        defaults = {
            "lr": lr,
            "betas": betas,
            "eps": eps,
            "weight_decay": weight_decay,
        }
        super().__init__(params, defaults)

    @torch.no_grad()
    def step(self, closure: Callable | None = None):
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()
        for group in self.param_groups:
            lr = group["lr"]
            beta_1, beta_2 = group["betas"]
            eps = group["eps"]
            weight_decay = group["weight_decay"]
            decay_factor = 1 - lr * weight_decay
            params: Iterator[Parameter] = group["params"]
            for p in params:
                if p.grad is None:
                    continue
                state = self.state[p]
                if not state:
                    state["t"] = 0
                    state["m"] = torch.zeros_like(p)
                    state["v"] = torch.zeros_like(p)
                t = state["t"] + 1
                m: Tensor = state["m"]
                v: Tensor = state["v"]
                grad = p.grad
                lr_t = lr * sqrt(1 - beta_2**t) / (1 - beta_1**t)
                p.mul_(decay_factor)
                m.mul_(beta_1).add_(grad, alpha=1 - beta_1)
                v.mul_(beta_2).addcmul_(grad, grad, value=1 - beta_2)
                denom = v.sqrt().add_(eps)
                p.addcdiv_(m, denom, value=-lr_t)
                state["t"] = t
        return loss


def lr_cosine_schedule(
    it: int,
    max_learning_rate: float,
    min_learning_rate: float,
    warmup_iters: int,
    cosine_cycle_iters: int,
) -> float:
    T_w, T_c = warmup_iters, cosine_cycle_iters
    max_lr, min_lr = max_learning_rate, min_learning_rate
    x = pi * ((it - T_w) / (T_c - T_w))
    if it < T_w:
        return max_lr * (it / T_w)
    elif it > T_c:
        return min_lr
    else:
        return min_lr + 0.5 * (1 + cos(x)) * (max_lr - min_lr)
