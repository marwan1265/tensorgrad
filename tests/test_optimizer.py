import numpy as np
import pytest

from tensorGrad.engine import Tensor
from tensorGrad.optimize import Adam, AdamW, SGD_M, Nesterov
from tensorGrad.backend import xp


# -----------------------------------------------------------------------------
# Helper utilities
# -----------------------------------------------------------------------------

def _generate_param_and_grads(shape, steps, rng):
    """Produce an initial parameter tensor and a list of synthetic gradients."""
    init_np = rng.standard_normal(shape).astype(np.float32)
    grads   = [rng.standard_normal(shape).astype(np.float32) for _ in range(steps)]
    return init_np, grads


def _run_optimizer_comparison(tg_opt, torch_opt, tg_param, torch_param, grad_list):
    """Helper to run optimizer steps and compare results."""
    import torch
    for g_np in grad_list:
        # attach gradients
        tg_param.grad     = xp.asarray(g_np)
        torch_param.grad  = torch.tensor(g_np, dtype=torch.float32)

        tg_opt.step()
        torch_opt.step()

        # clear grads
        tg_param.grad    = None
        torch_param.grad = None

    assert np.allclose(tg_param.data, torch_param.detach().numpy(), atol=1e-6)


# -----------------------------------------------------------------------------
# Adam Optimizer Tests
# -----------------------------------------------------------------------------

def test_adam_single_step_scalar():
    """Adam should perform the mathematically correct first update on a scalar parameter."""
    p = Tensor(0.0)           # parameter initialised at 0
    p.grad = xp.asarray(1.0)  # pretend backward pass produced grad = 1

    opt = Adam([p], learning_rate=1e-3, betas=(0.9, 0.999), eps=1e-8)
    opt.step()

    # Manual first-step calculation
    m = 0.1                           # (1 - beta1) * grad
    v = 0.001                         # (1 - beta2) * grad**2
    m_hat = m / (1 - 0.9)             # bias-corrected (t = 1)
    v_hat = v / (1 - 0.999)           # bias-corrected
    expected_update = 1e-3 * m_hat / (np.sqrt(v_hat) + 1e-8)
    expected_param = 0.0 - expected_update

    assert np.allclose(p.data, expected_param, atol=1e-10)


@pytest.mark.parametrize("shape", [(), (1,), (5,), (3, 4), (2, 3, 5), (1, 1, 1, 10)])
def test_adam_against_pytorch(shape):
    """Test Adam across various tensor dimensions."""
    rng = np.random.default_rng(seed=42)
    init_np = rng.standard_normal(shape).astype(np.float32)

    # TensorGrad setup
    tg_param = Tensor(init_np.copy())
    tg_opt = Adam([tg_param], learning_rate=1e-3, betas=(0.9, 0.999), eps=1e-8)

    # PyTorch setup
    import torch
    torch_param = torch.tensor(init_np, dtype=torch.float32, requires_grad=True)
    torch_opt = torch.optim.Adam([torch_param], lr=1e-3, betas=(0.9, 0.999), eps=1e-8)

    # Generate and apply gradients
    grad_list = [rng.standard_normal(shape).astype(np.float32) for _ in range(3)]
    _run_optimizer_comparison(tg_opt, torch_opt, tg_param, torch_param, grad_list)


@pytest.mark.parametrize("lr,betas", [
    (1e-4, (0.9, 0.999)),    # Standard
    (1e-2, (0.5, 0.9)),      # High LR, low momentum
    (1e-6, (0.99, 0.9999)),  # Low LR, high momentum
])
def test_adam_hyperparameters(lr, betas):
    """Test Adam with different hyperparameter combinations."""
    rng = np.random.default_rng(seed=100)
    shape = (3, 3)
    init_np = rng.standard_normal(shape).astype(np.float32)

    # TensorGrad setup
    tg_param = Tensor(init_np.copy())
    tg_opt = Adam([tg_param], learning_rate=lr, betas=betas, eps=1e-8)

    # PyTorch setup
    import torch
    torch_param = torch.tensor(init_np, dtype=torch.float32, requires_grad=True)
    torch_opt = torch.optim.Adam([torch_param], lr=lr, betas=betas, eps=1e-8)

    grad_list = [rng.standard_normal(shape).astype(np.float32) for _ in range(5)]
    _run_optimizer_comparison(tg_opt, torch_opt, tg_param, torch_param, grad_list)


# -----------------------------------------------------------------------------
# AdamW Optimizer Tests
# -----------------------------------------------------------------------------

@pytest.mark.parametrize("shape", [(), (4,), (2, 3), (2, 2, 2)])
def test_adamw_against_pytorch(shape):
    rng   = np.random.default_rng(seed=123)
    steps = 4

    init_np, grad_list = _generate_param_and_grads(shape, steps, rng)

    # TensorGrad param & optimiser
    tg_param = Tensor(init_np.copy())
    tg_opt   = AdamW([tg_param], learning_rate=1e-3, betas=(0.9, 0.999), eps=1e-8, weight_decay=0.05)

    # PyTorch reference
    import torch
    torch_param = torch.tensor(init_np, dtype=torch.float32, requires_grad=True)
    torch_opt   = torch.optim.AdamW([torch_param], lr=1e-3, betas=(0.9, 0.999), eps=1e-8, weight_decay=0.05)

    _run_optimizer_comparison(tg_opt, torch_opt, tg_param, torch_param, grad_list)


@pytest.mark.parametrize("weight_decay", [0.0, 0.01, 0.1])
def test_adamw_weight_decay_variations(weight_decay):
    """Test AdamW with different weight decay values."""
    rng = np.random.default_rng(seed=200)
    shape = (4, 4)
    init_np = rng.standard_normal(shape).astype(np.float32)

    # TensorGrad setup
    tg_param = Tensor(init_np.copy())
    tg_opt = AdamW([tg_param], learning_rate=1e-3, betas=(0.9, 0.999), eps=1e-8, weight_decay=weight_decay)

    # PyTorch setup
    import torch
    torch_param = torch.tensor(init_np, dtype=torch.float32, requires_grad=True)
    torch_opt = torch.optim.AdamW([torch_param], lr=1e-3, betas=(0.9, 0.999), eps=1e-8, weight_decay=weight_decay)

    grad_list = [rng.standard_normal(shape).astype(np.float32) for _ in range(3)]
    _run_optimizer_comparison(tg_opt, torch_opt, tg_param, torch_param, grad_list)


# -----------------------------------------------------------------------------
# SGD with Momentum Tests
# -----------------------------------------------------------------------------

@pytest.mark.parametrize("shape", [(), (1,), (5,), (3, 2), (2, 3, 4)])
def test_sgd_momentum_against_pytorch(shape):
    rng   = np.random.default_rng(seed=456)
    steps = 3
    beta  = 0.9
    lr    = 1e-2

    init_np, grad_list = _generate_param_and_grads(shape, steps, rng)

    # TensorGrad param & optimizer
    tg_param = Tensor(init_np.copy())
    tg_opt   = SGD_M([tg_param], learning_rate=lr, beta=beta)

    # PyTorch reference
    import torch
    torch_param = torch.tensor(init_np, dtype=torch.float32, requires_grad=True)
    torch_opt   = torch.optim.SGD([torch_param], lr=lr, momentum=beta)

    _run_optimizer_comparison(tg_opt, torch_opt, tg_param, torch_param, grad_list)


@pytest.mark.parametrize("momentum", [0.0, 0.5, 0.9, 0.99])
def test_sgd_momentum_variations(momentum):
    """Test SGD with different momentum values."""
    rng = np.random.default_rng(seed=300)
    shape = (3, 3)
    init_np = rng.standard_normal(shape).astype(np.float32)

    # TensorGrad setup
    tg_param = Tensor(init_np.copy())
    tg_opt = SGD_M([tg_param], learning_rate=1e-2, beta=momentum)

    # PyTorch setup
    import torch
    torch_param = torch.tensor(init_np, dtype=torch.float32, requires_grad=True)
    torch_opt = torch.optim.SGD([torch_param], lr=1e-2, momentum=momentum)

    grad_list = [rng.standard_normal(shape).astype(np.float32) for _ in range(4)]
    _run_optimizer_comparison(tg_opt, torch_opt, tg_param, torch_param, grad_list)


# -----------------------------------------------------------------------------
# Nesterov Momentum Tests
# -----------------------------------------------------------------------------

@pytest.mark.parametrize("shape", [(), (6,), (2, 4), (3, 2, 2)])
def test_nesterov_against_pytorch(shape):
    rng   = np.random.default_rng(seed=789)
    steps = 3
    beta  = 0.9
    lr    = 1e-2

    init_np, grad_list = _generate_param_and_grads(shape, steps, rng)

    # TensorGrad param & optimizer
    tg_param = Tensor(init_np.copy())
    tg_opt   = Nesterov([tg_param], learning_rate=lr, beta=beta)

    # PyTorch reference
    import torch
    torch_param = torch.tensor(init_np, dtype=torch.float32, requires_grad=True)
    torch_opt   = torch.optim.SGD([torch_param], lr=lr, momentum=beta, nesterov=True)

    _run_optimizer_comparison(tg_opt, torch_opt, tg_param, torch_param, grad_list)


# -----------------------------------------------------------------------------
# Edge Case Tests
# -----------------------------------------------------------------------------

def test_multiple_parameters():
    """Test optimizers with multiple parameters."""
    rng = np.random.default_rng(seed=400)
    
    # Create multiple parameters with different shapes
    init1 = rng.standard_normal((2, 3)).astype(np.float32)
    init2 = rng.standard_normal((5,)).astype(np.float32)
    init3 = rng.standard_normal(()).astype(np.float32)

    # TensorGrad setup
    tg_params = [Tensor(init1.copy()), Tensor(init2.copy()), Tensor(init3.copy())]
    tg_opt = Adam(tg_params, learning_rate=1e-3)

    # PyTorch setup
    import torch
    torch_params = [
        torch.tensor(init1, dtype=torch.float32, requires_grad=True),
        torch.tensor(init2, dtype=torch.float32, requires_grad=True),
        torch.tensor(init3, dtype=torch.float32, requires_grad=True)
    ]
    torch_opt = torch.optim.Adam(torch_params, lr=1e-3)

    # Apply gradients to all parameters
    for step in range(3):
        grads = [
            rng.standard_normal((2, 3)).astype(np.float32),
            rng.standard_normal((5,)).astype(np.float32),
            rng.standard_normal(()).astype(np.float32)
        ]
        
        # Set gradients
        for i, grad in enumerate(grads):
            tg_params[i].grad = xp.asarray(grad)
            torch_params[i].grad = torch.tensor(grad, dtype=torch.float32)
        
        # Step optimizers
        tg_opt.step()
        torch_opt.step()
        
        # Clear gradients
        for i in range(len(tg_params)):
            tg_params[i].grad = None
            torch_params[i].grad = None

    # Check all parameters match
    for i in range(len(tg_params)):
        assert np.allclose(tg_params[i].data, torch_params[i].detach().numpy(), atol=1e-6)


@pytest.mark.parametrize("optimizer_class,torch_name,kwargs", [
    (Adam, "Adam", {"learning_rate": 1e-3, "betas": (0.9, 0.999), "eps": 1e-8}),
    (SGD_M, "SGD", {"learning_rate": 1e-2, "beta": 0.9}),
    (Nesterov, "SGD", {"learning_rate": 1e-2, "beta": 0.9}),
])
def test_extreme_gradients(optimizer_class, torch_name, kwargs):
    """Test optimizers with extreme gradient values."""
    import torch
    
    rng = np.random.default_rng(seed=500)
    shape = (3, 3)
    init_np = rng.standard_normal(shape).astype(np.float32)

    # Test with very small gradients
    tg_param = Tensor(init_np.copy())
    torch_param = torch.tensor(init_np, dtype=torch.float32, requires_grad=True)
    
    # Create optimizers
    if optimizer_class == Nesterov:
        tg_opt = optimizer_class([tg_param], **kwargs)
        torch_opt = getattr(torch.optim, torch_name)([torch_param], lr=kwargs["learning_rate"], 
                                                     momentum=kwargs["beta"], nesterov=True)
    elif optimizer_class == SGD_M:
        tg_opt = optimizer_class([tg_param], **kwargs)
        torch_opt = getattr(torch.optim, torch_name)([torch_param], lr=kwargs["learning_rate"], 
                                                     momentum=kwargs["beta"])
    else:
        tg_opt = optimizer_class([tg_param], **kwargs)
        torch_opt = getattr(torch.optim, torch_name)([torch_param], lr=kwargs["learning_rate"], 
                                                     betas=kwargs["betas"], eps=kwargs["eps"])

    # Test very small gradients
    small_grad = np.full(shape, 1e-8, dtype=np.float32)
    tg_param.grad = xp.asarray(small_grad)
    torch_param.grad = torch.tensor(small_grad, dtype=torch.float32)
    
    tg_opt.step()
    torch_opt.step()
    
    assert np.allclose(tg_param.data, torch_param.detach().numpy(), atol=1e-6)


def test_zero_gradients():
    """Test optimizer behavior with zero gradients."""
    import torch
    
    shape = (2, 2)
    init_np = np.ones(shape, dtype=np.float32)
    
    tg_param = Tensor(init_np.copy())
    tg_opt = Adam([tg_param], learning_rate=1e-3)
    
    torch_param = torch.tensor(init_np, dtype=torch.float32, requires_grad=True)
    torch_opt = torch.optim.Adam([torch_param], lr=1e-3)
    
    # Apply zero gradients
    zero_grad = np.zeros(shape, dtype=np.float32)
    tg_param.grad = xp.asarray(zero_grad)
    torch_param.grad = torch.tensor(zero_grad, dtype=torch.float32)
    
    tg_opt.step()
    torch_opt.step()
    
    assert np.allclose(tg_param.data, torch_param.detach().numpy(), atol=1e-6)


# -----------------------------------------------------------------------------
# General Optimizer Behavior Tests
# -----------------------------------------------------------------------------

def test_zero_grad_sets_grad_to_zero():
    """Test that zero_grad() correctly resets gradients for all optimizers."""
    p1 = Tensor(1.0); p1.grad = xp.asarray(0.5)
    p2 = Tensor([2.0, 3.0]); p2.grad = xp.asarray([0.1, -0.2])

    opt = SGD_M([p1, p2])
    opt.zero_grad()

    assert np.allclose(p1.grad, 0.0)
    assert np.allclose(p2.grad, 0.0)


def test_step_with_none_gradients():
    """Test that optimizers handle None gradients gracefully."""
    p1 = Tensor(1.0); p1.grad = None
    p2 = Tensor([2.0, 3.0]); p2.grad = xp.asarray([0.1, -0.2])

    opt = Adam([p1, p2])
    original_p1 = p1.data.copy()
    
    # Should not crash and should not update p1
    opt.step()
    
    assert np.allclose(p1.data, original_p1)  # p1 unchanged
    # p2 should have been updated (we don't check exact value since it's complex for Adam) 