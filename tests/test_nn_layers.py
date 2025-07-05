import numpy as np
import pytest

import torch

from tensorGrad.nn import Linear, MLP, ReLU, Sequential
from tensorGrad.engine import Tensor
from tensorGrad.optimize import Adam
from tensorGrad.backend import xp


@pytest.mark.parametrize("batch,in_f,out_f", [(4, 3, 5), (2, 10, 7)])
def test_linear_matches_pytorch(batch, in_f, out_f):
    rng = np.random.default_rng(0)
    x_np = rng.standard_normal((batch, in_f)).astype(np.float32)
    W_np = rng.standard_normal((in_f, out_f)).astype(np.float32)
    b_np = rng.standard_normal((out_f,)).astype(np.float32)

    # TensorGrad layer with manual weights
    tg_linear = Linear(in_f, out_f)
    tg_linear.W.data = xp.asarray(W_np.copy())
    tg_linear.b.data = xp.asarray(b_np.copy())

    x_tg = Tensor(x_np.copy())
    out_tg = tg_linear(x_tg)
    loss_tg = out_tg.sum()
    loss_tg.backward()

    # PyTorch reference
    torch_linear = torch.nn.Linear(in_f, out_f, bias=True)
    with torch.no_grad():
        torch_linear.weight.copy_(torch.tensor(W_np.T))  # PyTorch stores transposed
        torch_linear.bias.copy_(torch.tensor(b_np))
    x_th = torch.tensor(x_np, dtype=torch.float32, requires_grad=True)
    out_th = torch_linear(x_th)
    loss_th = out_th.sum()
    loss_th.backward()

    # Compare outputs and gradients
    assert np.allclose(out_tg.data, out_th.detach().numpy(), atol=1e-6)
    assert np.allclose(tg_linear.W.grad, torch_linear.weight.grad.T.detach().numpy(), atol=1e-6)
    assert np.allclose(tg_linear.b.grad, torch_linear.bias.grad.detach().numpy(), atol=1e-6)


def test_mlp_overfits_xor():
    # XOR dataset
    X = Tensor([[0,0],[0,1],[1,0],[1,1]], label="X")
    Y = Tensor([[0],[1],[1],[0]], label="Y")

    mlp = MLP(2, [8], 1)
    opt = Adam(mlp.parameters(), learning_rate=0.1)

    initial_loss = None
    for epoch in range(200):
        pred = mlp(X).sigmoid()
        loss = ((pred - Y) ** 2).mean()

        opt.zero_grad()
        loss.backward()
        opt.step()

        if epoch == 0:
            initial_loss = loss.data
    # ensure loss dropped significantly
    assert loss.data < 0.05 * initial_loss 