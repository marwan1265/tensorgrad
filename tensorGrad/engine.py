import numpy as np
from .backend import xp

class Tensor:
  # global flag indicating global no_grad mode
  _no_grad = False  # class-level toggle similar to torch.set_grad_enabled
  
  def __init__(self, data, _children=(), _op='', requires_grad=True, label=''):  # requires_grad=True: default to True
    self.data = xp.asarray(data, dtype=xp.float32)
    # respect the global no_grad flag
    requires_grad = requires_grad and not Tensor._no_grad
    self.requires_grad = requires_grad
    self.grad = xp.zeros_like(self.data) if self.requires_grad else None
    self._backward = lambda: None
    # avoid growing the graph when gradients are disabled
    self._prev = set(_children) if self.requires_grad else set()
    self._op = _op if self.requires_grad else ''
    self.label = label

  # -------------------------------------------------------------
  # Context-manager utilities
  # -------------------------------------------------------------

  @staticmethod
  def no_grad():
      """Context manager that disables gradient tracking inside its block."""
      class _NoGrad:
          def __enter__(self_inner):
              self_inner.prev = Tensor._no_grad
              Tensor._no_grad = True
          def __exit__(self_inner, exc_type, exc_val, exc_tb):
              Tensor._no_grad = self_inner.prev
      return _NoGrad()

  # -------------------------------------------------------------
  # Operator overloads
  # -------------------------------------------------------------

  def __repr__(self):
    return f"Tensor(data={self.data})"
  
  # centralized safe gradient accumulation - ignores on requires_grad = false
  def _add_grad(self, g):
    if self.requires_grad:
        if self.grad is None:
            self.grad = xp.zeros_like(self.data)
        self.grad += g

  # Reverse NumPy broadcasting: collapse gradients to original parameter shape
  @staticmethod
  def _unbroadcast(grad, shape_target):
      while len(grad.shape) > len(shape_target):
          grad = grad.sum(axis=0)
      for i, (gdim, tdim) in enumerate(zip(grad.shape, shape_target)):
          if tdim == 1 and gdim != 1:
              grad = grad.sum(axis=i, keepdims=True)
      return grad
  
  def __add__(self, other):
    other = other if isinstance(other, Tensor) else Tensor(other)
    out = Tensor(self.data + other.data, (self, other), '+')
    
    def _backward():
      grad = out.grad
      self._add_grad(Tensor._unbroadcast(grad, self.data.shape))
      other._add_grad(Tensor._unbroadcast(grad, other.data.shape))
    out._backward = _backward
    
    return out

  def __mul__(self, other):
    other = other if isinstance(other, Tensor) else Tensor(other)
    out = Tensor(self.data * other.data, (self, other), '*')
    
    def _backward():
      grad = out.grad
      self._add_grad(Tensor._unbroadcast(other.data * grad, self.data.shape))
      other._add_grad(Tensor._unbroadcast(self.data * grad, other.data.shape))
    out._backward = _backward
      
    return out
  
  def matmul(self, other):

    out = Tensor(self.data @ other.data, (self, other), '@')

    def _backward():
      # dL/dA = dL/dC @ B^T
      if self.requires_grad:
        grad_A = out.grad @ other.data.swapaxes(-2, -1)
        self._add_grad(Tensor._unbroadcast(grad_A, self.data.shape))

      # dL/dB = A^T @ dL/dC
      if other.requires_grad:
        grad_B = self.data.swapaxes(-2, -1) @ out.grad
        other._add_grad(Tensor._unbroadcast(grad_B, other.data.shape))
      
    out._backward = _backward

    return out
  
  def __rmul__(self, other): # other * self
    return self * other

  def __truediv__(self, other): # self / other
    return self * other**-1

  def __neg__(self): # -self
    return self * -1

  def __sub__(self, other): # self - other
    return self + (-other)

  def __radd__(self, other): # other + self
    return self + other
  
  def __rsub__(self, other):
    other = other if isinstance(other, Tensor) else Tensor(other)
    return other + (-self)
  
  def exp(self):
    out = Tensor(xp.exp(self.data), (self, ), 'exp')
    
    def _backward():
      grad_self = out.data * out.grad
      self._add_grad(Tensor._unbroadcast(grad_self, self.data.shape))
    out._backward = _backward
    
    return out
  
  def log(self):
    out = Tensor(xp.log(self.data), (self, ), 'log')
    
    def _backward():
      grad_self = (1.0 / self.data) * out.grad
      self._add_grad(Tensor._unbroadcast(grad_self, self.data.shape))
    out._backward = _backward
    
    return out
  

  # Activation functions:

  def relu(self):
    out = Tensor(xp.maximum(0, self.data), (self,), 'ReLU')
    
    def _backward():
      grad_self = (out.data > 0) * out.grad
      self._add_grad(Tensor._unbroadcast(grad_self, self.data.shape))
    out._backward = _backward
    
    return out
  
  #leaky relu that addresses the "dying ReLU" issue - by introducing a small, non-zero slope for negative input values.
  def leaky_relu(self, alpha: float = 0.01):
        # y = x if x>0 else alpha*x
        # alpha is a small slope for the 'negative' side (default 0.01)
    out = Tensor(xp.where(self.data > 0, self.data, alpha * self.data), (self,), f'LeakyReLU{alpha}')

    def _backward():
      grad_mask = xp.where(self.data > 0, 1.0, alpha)
      grad_self = grad_mask * out.grad
      self._add_grad(Tensor._unbroadcast(grad_self, self.data.shape))
    out._backward = _backward

    return out

  def tanh(self):
    out = Tensor(xp.tanh(self.data), (self,), 'tanh')

    def _backward():
      grad_self = (1.0 - out.data**2) * out.grad
      self._add_grad(Tensor._unbroadcast(grad_self, self.data.shape))
    out._backward = _backward

    return out
  
  def sigmoid(self):
    sig = 1.0 / (1.0 + xp.exp(-self.data))
    out = Tensor(sig, (self,), 'sigmoid')

    def _backward():
      grad_self = sig * (1.0 - sig) * out.grad
      self._add_grad(Tensor._unbroadcast(grad_self, self.data.shape))
    out._backward = _backward

    return out
  
  # Shape and utility methods

  @property
  def shape(self):
    return self.data.shape
  
  @property
  def ndim(self):
    return self.data.ndim
  
  # Convenience conversions
  def to_numpy(self):
    """Return data as a NumPy ndarray (copying from GPU if necessary)."""
    if xp.__name__ == 'numpy':
        return self.data
    else:
        import cupy as cp
        return cp.asnumpy(self.data)

  def to_cupy(self):
    """Return data as a CuPy ndarray (moving to GPU if necessary)."""
    if xp.__name__ == 'cupy':
        return self.data
    else:
        try:
            import cupy as cp
        except ImportError as exc:
            raise ImportError("CuPy is not installed; cannot convert to CuPy array") from exc
        return cp.asarray(self.data)
  
  def reshape(self, *shape):
    out = Tensor(self.data.reshape(shape), (self,), 'reshape')
    
    def _backward():
      grad_self = out.grad.reshape(self.data.shape)
      self._add_grad(grad_self)
    out._backward = _backward
    
    return out
  
  def sum(self, axis=None, keepdims=False):
    out = Tensor(xp.sum(self.data, axis=axis, keepdims=keepdims), (self,), 'sum')
    
    def _backward():
      grad_self = out.grad
      if axis is None:
        # total sum -> grad is a scalar;broadcast to whole tensor
        grad_self = xp.broadcast_to(grad_self, self.data.shape)
      else:
        if not keepdims:
          # Need to add back the summed dimensions
          if isinstance(axis, int):
            grad_self = xp.expand_dims(grad_self, axis)
          else:
            for ax in sorted(axis):
              grad_self = xp.expand_dims(grad_self, ax)
        # Broadcast to original shape
        grad_self = xp.broadcast_to(grad_self, self.data.shape)

      self._add_grad(grad_self)

    out._backward = _backward
    return out
  
  def mean(self, axis=None, keepdims=False):
    return self.sum(axis=axis, keepdims=keepdims) / self.data.size if axis is None else self.sum(axis=axis, keepdims=keepdims) / self.data.shape[axis]
  
  def max(self, axis=None, keepdims=False):
    out = Tensor(xp.max(self.data, axis=axis, keepdims=keepdims), (self,), 'max')
    
    def _backward():
      grad = out.grad
      if axis is not None and not keepdims:
        if isinstance(axis, int):
          grad = xp.expand_dims(grad, axis)
        else:
          for ax in sorted(axis):
            grad = xp.expand_dims(grad, ax)
      
      # Create mask for maximum values
      max_vals = xp.max(self.data, axis=axis, keepdims=True)
      mask = (self.data == max_vals).astype(xp.float32)
      
      if axis is not None:
        grad = xp.broadcast_to(grad, self.data.shape)
      
      grad_contrib = grad * mask
      self._add_grad(grad_contrib)
    out._backward = _backward
    
    return out
  
  def __matmul__(self, other):
    return self.matmul(other)
  
  def __getitem__(self, idx):
    out = Tensor(self.data[idx], (self,), 'getitem')
    
    def _backward():
      grad_self = xp.zeros_like(self.data)
      grad_self[idx] = out.grad
      self._add_grad(grad_self)
    out._backward = _backward
    
    return out
  
  # Backpropogation function:

  # uses post order traversal to return the topological order of the graph 
  # then backpropogates through the (reversed) result.

  def backward(self, retain_graph = False):
    
    topo = []
    visited = set()
    def build_topo(v):
      if v not in visited:
        visited.add(v)
        for child in v._prev:
          build_topo(child)
        topo.append(v)
    build_topo(self)
    
    self.grad = xp.ones_like(self.data)
    for node in reversed(topo):
      node._backward()
      if not retain_graph:
          node._backward = lambda: None

  # power operator for scalar exponents (supports Tensor ** -1 used in division)
  def __pow__(self, power):
    if isinstance(power, (int, float)):
        out = Tensor(self.data ** power, (self,), f'pow{power}')

        def _backward():
            if self.requires_grad:
                grad_self = power * (self.data ** (power - 1)) * out.grad
                self._add_grad(Tensor._unbroadcast(grad_self, self.data.shape))
        out._backward = _backward
        return out
    else:
        raise TypeError("Power with non-scalar exponent not supported")


# Tensor creation utilities (PyTorch-like)
def zeros(*shape):
    return Tensor(xp.zeros(shape))

def ones(*shape):
    return Tensor(xp.ones(shape))

def randn(*shape):
    return Tensor(xp.random.randn(*shape))

def tensor(data):
    return Tensor(data)

def arange(start, stop=None, step=1):
    if stop is None:
        stop = start
        start = 0
    return Tensor(xp.arange(start, stop, step))

def eye(n):
    return Tensor(xp.eye(n))


def softmax(x, axis=-1):
    """
    Vectorized softmax function for tensors.
    Works with both 1D and 2D+ tensors (batched).
    """
    # Numerical stability: subtract max
    max_vals = x.max(axis=axis, keepdims=True)
    shifted = x - max_vals
    
    # Compute exponentials
    exp_vals = shifted.exp()
    
    # Sum and normalize
    sum_exp = exp_vals.sum(axis=axis, keepdims=True)
    return exp_vals / sum_exp

def cross_entropy(logits: Tensor, labels: Tensor, reduction: str = 'mean') -> Tensor:
    #Compute the cross-entropy loss between *logits* and integer *labels*.
    # Numerically-stable log-softmax

    max_logits = logits.max(axis=-1, keepdims=True)          # (N, 1)
    shifted    = logits - max_logits                         # subtract max for stability
    exp_shift  = shifted.exp()                               # exp
    sum_exp    = exp_shift.sum(axis=-1, keepdims=True) + 1e-10       # sum over classes
    log_probs  = shifted - sum_exp.log()                     # log-softmax

    # Gather the log-prob of the correct class for each sample
    label_indices = labels.data.astype(xp.int64)  # ensure integer indexing
    batch_indices = xp.arange(label_indices.shape[0])
    correct_logp = log_probs[batch_indices, label_indices]
    losses = -correct_logp  # negative log-likelihood

    if reduction == 'mean':
        return losses.mean()
    elif reduction == 'sum':
        return losses.sum()
    elif reduction == 'none':
        return losses
    else:
        raise ValueError("reduction must be 'mean', 'sum', or 'none'")
    
def mse_loss(predictions, targets, reduction: str = 'mean'):
    # Compute the mean squared error between *predictions* and *targets*.
    losses = (predictions - targets) ** 2

    if reduction == 'mean':
        return losses.mean()
    elif reduction == 'sum':
        return losses.sum()
    elif reduction == 'none':
        return losses

def binary_cross_entropy(predictions: Tensor, targets: Tensor, reduction: str = 'mean'):
    """Binary cross-entropy loss supporting Tensor inputs.

    Parameters
    ----------
    predictions : Tensor
        Model sigmoidal outputs in (0,1).
    targets : Tensor
        Ground-truth labels in {0,1}. *targets* should not require gradients.
    reduction : {'mean', 'sum', 'none'}
        Reduction mode.
    """
    eps = 1e-10  # numerical stability

    # Clamp predictions to avoid log(0). Use Tensor operations for autograd.
    pred_clamped   = predictions.clip(eps, 1 - eps) if hasattr(Tensor, "clip") else predictions  # fallback if no clip

    ones = Tensor(xp.ones_like(targets.data), requires_grad=False)

    losses = - (targets * pred_clamped.log() + (ones - targets) * (ones - pred_clamped).log())

    if reduction == 'mean':
        return losses.mean()
    elif reduction == 'sum':
        return losses.sum()
    elif reduction == 'none':
        return losses
    else:
        raise ValueError("reduction must be 'mean', 'sum', or 'none'")