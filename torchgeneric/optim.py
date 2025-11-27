import torch
from typing import Tuple, Optional, Callable, Dict, Any, Iterable, Union
from torch.optim.optimizer import Optimizer, required 

Params = Union[Iterable[torch.Tensor], Iterable[Dict[str, Any]]]

class GGD(Optimizer):
    """
    Implements the Generic Gradient Descent (GGD) optimization algorithm with
    lookahead Force Vector.
    """
    def __init__(
        self, 
        params: Params, 
        lr: float = required, 
        betas: Tuple[float, float] = (0.9, 0.999), 
        eps: float = 1e-8, 
        weight_decay: float = 0,
        amsgrad: bool = False
    ):
        if lr is not required and not 0.0 <= lr: raise ValueError(f"Invalid learning rate: {lr}")
        if not 0.0 <= eps: raise ValueError(f"Invalid epsilon value: {eps}")
        if not 0.0 <= betas[0] < 1.0: raise ValueError(f"Invalid beta parameter at index 0: {betas[0]}")
        if not 0.0 <= betas[1] < 1.0: raise ValueError(f"Invalid beta parameter at index 1: {betas[1]}")
        if not 0.0 <= weight_decay: raise ValueError(f"Invalid weight_decay value: {weight_decay}")

        defaults = dict(lr=lr, betas=betas, eps=eps, weight_decay=weight_decay, amsgrad=amsgrad)
        super().__init__(params, defaults)


    @torch.no_grad()
    def step(self, closure: Optional[Callable[[], float]] = None) -> Optional[float]:
        loss: Optional[float] = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()
        
        for group in self.param_groups:
            beta1, beta2 = group['betas']
            lr = group['lr']
            weight_decay = group['weight_decay']
            eps = group['eps']
            amsgrad = group['amsgrad']

            for p in group['params']:
                if p.grad is None: continue
                
                grad = p.grad
                
                if grad.is_sparse:
                    raise RuntimeError('GGD does not support sparse gradients yet')


                state = self.state[p]
                 
                if len(state) == 0:
                    state['step'] = 0
                    state['exp_avg'] = torch.zeros_like(p, memory_format=torch.preserve_format)
                    state['exp_avg_sq'] = torch.zeros_like(p, memory_format=torch.preserve_format)
                    if amsgrad:
                        state['max_exp_avg_sq'] = torch.zeros_like(p, memory_format=torch.preserve_format)

                exp_avg = state['exp_avg']
                exp_avg_sq = state['exp_avg_sq']
                state['step'] += 1
                step = state['step']
                 
                if weight_decay != 0:
                    grad = grad.add(p.data, alpha=weight_decay)
                 
                exp_avg.mul_(beta1).add_(grad, alpha=1 - beta1)
                exp_avg_sq.mul_(beta2).addcmul_(grad, grad, value=1 - beta2)
                 
                bias_correction1 = 1 - beta1 ** step
                bias_correction2 = 1 - beta2 ** step
                 
                if amsgrad:
                    max_exp_avg_sq = state['max_exp_avg_sq'] 
                    torch.maximum(max_exp_avg_sq, exp_avg_sq, out=max_exp_avg_sq) 
                    exp_avg_sq_hat = max_exp_avg_sq / bias_correction2
                else: 
                    exp_avg_sq_hat = exp_avg_sq / bias_correction2
                 
                exp_avg_hat = exp_avg / bias_correction1
                 
                grad_hat = grad / bias_correction1 
                lookahead_exp_avg_hat = exp_avg_hat.mul(beta1).add_(grad_hat, alpha=1 - beta1)
                 
                denom = exp_avg_sq_hat.sqrt_().add_(eps) 
                step_direction = lookahead_exp_avg_hat.div(denom)
 
                p.add_(step_direction, alpha=-lr) 
        return loss