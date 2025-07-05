import math
import random

import numpy as np
import torch
import pytest

# import the custom Tensor class. Adjust the import path if you put it elsewhere.
from tensorGrad.engine import Tensor, softmax  # import from the actual package location

# Safe random helper – returns float32 ndarray for any shape (works for scalars when shape == ())
def randn_f32(*shape):
    return np.asarray(np.random.randn(*shape), dtype=np.float32)


def to_torch(x):
    """Convert numpy array or scalar to torch tensor with grad enabled."""
    return torch.tensor(x, dtype=torch.float32, requires_grad=True)


def assert_close(a, b, atol=1e-6, rtol=1e-5):
    assert np.allclose(a, b, atol=atol, rtol=rtol), f"Mismatch: {a} vs {b}"


###############################
# Basic element-wise operators
###############################
@pytest.mark.parametrize("shape", [(), (4,), (2, 3), (3, 1, 5)])
def test_add(shape):
    a_np = randn_f32(*shape)
    b_np = randn_f32(*shape)

    # custom
    a = Tensor(a_np.copy())
    b = Tensor(b_np.copy())
    out = (a + b).sum()
    out.backward()

    # torch
    a_t = to_torch(a_np)
    b_t = to_torch(b_np)
    out_t = (a_t + b_t).sum()
    out_t.backward()

    assert_close(out.data, out_t.detach().numpy())
    assert_close(a.grad, a_t.grad.numpy())
    assert_close(b.grad, b_t.grad.numpy())


@pytest.mark.parametrize("shapeA,shapeB", [
    ((3, 1), (1, 3)),
    ((2, 3, 1), (3,)),
    ((1,), ()),                 # scalar broadcasting
])
def test_broadcast_add(shapeA, shapeB):
    a_np = randn_f32(*shapeA)
    b_np = randn_f32(*shapeB)

    a = Tensor(a_np.copy())
    b = Tensor(b_np.copy())
    out = (a + b).sum()
    out.backward()

    a_t = to_torch(a_np)
    b_t = to_torch(b_np)
    out_t = (a_t + b_t).sum()
    out_t.backward()

    assert_close(a.grad, a_t.grad.numpy())
    assert_close(b.grad, b_t.grad.numpy())


###############################
# Multiplication and matmul
###############################
@pytest.mark.parametrize("shape", [(), (5,), (2, 3)])
def test_mul(shape):
    a_np = randn_f32(*shape)
    b_np = randn_f32(*shape)
    a = Tensor(a_np.copy())
    b = Tensor(b_np.copy())
    out = (a * b).sum()
    out.backward()

    a_t = to_torch(a_np)
    b_t = to_torch(b_np)
    torch_out = (a_t * b_t).sum()
    torch_out.backward()

    assert_close(a.grad, a_t.grad.numpy())
    assert_close(b.grad, b_t.grad.numpy())


@pytest.mark.parametrize("A_shape,B_shape", [
    ((3, 4), (4, 2)),            # 2-D × 2-D
    ((10, 1, 4, 3), (3, 5)),     # batched matmul (leading dims broadcast)
])
def test_matmul(A_shape, B_shape):
    A_np = randn_f32(*A_shape)
    B_np = randn_f32(*B_shape)

    A = Tensor(A_np.copy())
    B = Tensor(B_np.copy())
    out = (A @ B).sum()
    out.backward()

    A_t = to_torch(A_np)
    B_t = to_torch(B_np)
    torch_out = (A_t @ B_t).sum()
    torch_out.backward()

    assert_close(A.grad, A_t.grad.numpy())
    assert_close(B.grad, B_t.grad.numpy())


###############################
# Reductions
###############################
@pytest.mark.parametrize("axis,keepdims", [
    (None, False),
    (0, False),
    (1, True),
    ((0, 2), False),
])
def test_sum(axis, keepdims):
    x_np = randn_f32(2, 3, 4)
    x = Tensor(x_np.copy())
    out = x.sum(axis=axis, keepdims=keepdims)
    out.backward()

    x_t = to_torch(x_np)
    torch_out = x_t.sum(dim=axis, keepdim=keepdims)
    torch_out.backward(torch.ones_like(torch_out))

    assert_close(out.data, torch_out.detach().numpy())
    assert_close(x.grad, x_t.grad.numpy())


###############################
# Activation functions
###############################
@pytest.mark.parametrize("shape", [(5,), (2, 3)])
@pytest.mark.parametrize("act", ["relu", "tanh", "sigmoid", "leaky_relu"])
def test_activations(shape, act):
    data_np = randn_f32(*shape)

    x = Tensor(data_np.copy())
    if act == "relu":
        y = x.relu()
    elif act == "tanh":
        y = x.tanh()
    elif act == "sigmoid":
        y = x.sigmoid()
    elif act == "leaky_relu":
        y = x.leaky_relu(0.01)
    out = y.sum()
    out.backward()

    x_t = to_torch(data_np)
    if act == "relu":
        y_t = torch.nn.functional.relu(x_t)
    elif act == "tanh":
        y_t = torch.tanh(x_t)
    elif act == "sigmoid":
        y_t = torch.sigmoid(x_t)
    elif act == "leaky_relu":
        y_t = torch.nn.functional.leaky_relu(x_t, negative_slope=0.01)
    torch_out = y_t.sum()
    torch_out.backward()

    assert_close(out.data, torch_out.detach().numpy())
    assert_close(x.grad, x_t.grad.numpy())


###############################
# Softmax utility
###############################
@pytest.mark.parametrize("axis", [-1, 0])
def test_softmax(axis):
    data_np = randn_f32(4, 6)

    x = Tensor(data_np.copy())
    out = softmax(x, axis=axis)
    loss = out.sum()
    loss.backward()

    x_t = to_torch(data_np)
    out_t = torch.nn.functional.softmax(x_t, dim=axis)
    torch_out = out_t.sum()
    torch_out.backward()

    assert_close(out.data, out_t.detach().numpy())
    assert_close(x.grad, x_t.grad.numpy())


#########################################
# Detach / requires_grad False behaviour
#########################################

def test_no_grad():
    data_np = randn_f32(3, 4)
    x = Tensor(data_np.copy())
    w = Tensor(randn_f32(4, 2))
    with Tensor.no_grad():
        y = (x @ w)  # graph should not grow
    assert y.requires_grad is False 