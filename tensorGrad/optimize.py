import numpy as np
from .engine import Tensor
from .backend import xp

class Optimizer:
    def __init__(self, params):
        self.params = list(params)

    def zero_grad(self):
        for p in self.params:
            if p.grad is not None:
                p.grad.fill(0)

class Adam(Optimizer):
    #Adam Optimizer Formula

    # m[t] = (beta1 * m[t-1]) + (1 - beta1) * current gradient     # m is the first moment of the gradient
    # v[t] = (beta2 * v[t-1]) + (1 - beta2) * current gradient^2   # v is the second moment of the gradient

    # m_hat[t] = m[t] / (1 - beta1^t)             # bias correction
    # v_hat[t] = v[t] / (1 - beta2^t)             # bias correction

    # next_param = current param - ((learning_rate/root(v_hat)-epsilon))*m_hat

    def __init__(self,params,learning_rate=1e-3, betas=(0.9, 0.999), eps=1e-8):
        super().__init__(params)
        self.learning_rate = learning_rate
        self.betas = betas
        self.eps = eps

        # Adam Optimizer Variable initialization
        self.t = 0
        self.m = [xp.zeros_like(p.data) for p in self.params]
        self.v = [xp.zeros_like(p.data) for p in self.params]

    def step(self):

        # variable initialization
        self.t += 1

        for i, param in enumerate(self.params):
            if param.grad is None:
                continue
            g = param.grad
            # Update exponential moving average of the gradient (1st moment ≈ mean).
            self.m[i] = self.betas[0] * self.m[i] + (1 - self.betas[0]) * g
            # Update exponential moving average of squared gradient (2nd moment ≈ uncentered variance).
            self.v[i] = self.betas[1] * self.v[i] + (1 - self.betas[1]) * g**2

            # Bias-correct the moment estimates (compensate for their initialisation at zero).
            m_hat = self.m[i] / (1 - self.betas[0] ** self.t)
            v_hat = self.v[i] / (1 - self.betas[1] ** self.t)

            # Parameter update: move opposite to gradient, normalised by RMS of past gradients.
            param.data -= self.learning_rate * m_hat / (xp.sqrt(v_hat) + self.eps)

# identical to Adam but with weight decay
class AdamW(Optimizer):
    def __init__(self, params, learning_rate=1e-3, betas=(0.9, 0.999), eps=1e-8, weight_decay=0.01):
        super().__init__(params)
        self.learning_rate = learning_rate
        self.betas = betas
        self.eps = eps
        self.weight_decay = weight_decay

        # Adam Optimizer Variable initialization
        self.t = 0
        self.m = [xp.zeros_like(p.data) for p in self.params]
        self.v = [xp.zeros_like(p.data) for p in self.params]

    def step(self):
                # variable initialization
        self.t += 1

        for i, param in enumerate(self.params):
            if param.grad is None:
                continue
            g = param.grad

            self.m[i] = self.betas[0] * self.m[i] + (1 - self.betas[0]) * g
            self.v[i] = self.betas[1] * self.v[i] + (1 - self.betas[1]) * g**2

            m_hat = self.m[i] / (1 - self.betas[0] ** self.t)
            v_hat = self.v[i] / (1 - self.betas[1] ** self.t)

            # Parameter update with weight decay
            param.data -= self.learning_rate * (m_hat / (xp.sqrt(v_hat) + self.eps) + self.weight_decay * param.data)

class SGD_M(Optimizer):

    def __init__(self, params, learning_rate=1e-3, beta=0.9,):
        super().__init__(params)
        self.learning_rate = learning_rate
        self.beta = beta

        self.velocity = [xp.zeros_like(p.data) for p in self.params]

    # v[t] = beta*v[t-1] + (1 - beta) 
    # new_weight = current_weight - learning_rate * v[t]

    def step(self):
        for i, param in enumerate(self.params):
            if param.grad is None:
                continue
            g = param.grad
            
            self.velocity[i] = self.beta * self.velocity[i] + g
            param.data -= self.learning_rate * self.velocity[i]

        

class Nesterov(Optimizer):
    def __init__(self, params, learning_rate=1e-3, beta=0.9):
        super().__init__(params)
        self.learning_rate = learning_rate
        self.beta = beta

        self.velocity = [xp.zeros_like(p.data) for p in self.params]
    
    def step(self):
        for i, param in enumerate(self.params):
            if param.grad is None:
                continue
            g = param.grad

            # Update momentum buffer with dampening (same rule as SGD_M)
            self.velocity[i] = self.beta * self.velocity[i] + g

            # Nesterov look-ahead update uses the gradient evaluated at the
            # prospective position:  g + β · v
            param.data -= self.learning_rate * (g + self.beta * self.velocity[i])

        