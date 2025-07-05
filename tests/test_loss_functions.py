import numpy as np
import pytest

import torch
import torch.nn.functional as F

from tensorGrad.engine import Tensor, cross_entropy, mse_loss, binary_cross_entropy


# -----------------------------------------------------------------------------
# Helper utilities
# -----------------------------------------------------------------------------


def _compare_tensors(a, b, atol=1e-6):
    """Utility: assert two numpy arrays are close."""
    assert np.allclose(a, b, atol=atol), f"Arrays differ:\n{a}\n{b}"



def _backward_and_get_grad(tg_loss, th_loss, tg_tensor, th_tensor, reduction):
    """Run backward (handling 'none' reduction) and return gradients."""
    if reduction == "none":
        tg_loss = tg_loss.sum()
        th_loss = th_loss.sum()
    tg_loss.backward()
    th_loss.backward()
    return tg_tensor.grad, th_tensor.grad.detach().numpy()


# -----------------------------------------------------------------------------
# Cross-Entropy Loss Tests
# -----------------------------------------------------------------------------


@pytest.mark.parametrize(
    "batch,n_classes,reduction",
    [
        (4, 6, "mean"),
        (5, 3, "sum"),
        (3, 4, "none"),
    ],
)
def test_cross_entropy_matches_pytorch(batch, n_classes, reduction):
    rng = np.random.default_rng(0)
    logits_np = rng.standard_normal((batch, n_classes)).astype(np.float32)
    labels_np = rng.integers(0, n_classes, size=(batch,), dtype=np.int64)

    # TensorGrad
    logits_tg = Tensor(logits_np.copy())
    labels_tg = Tensor(labels_np, requires_grad=False)
    loss_tg = cross_entropy(logits_tg, labels_tg, reduction=reduction)

    # PyTorch
    logits_th = torch.tensor(logits_np, dtype=torch.float32, requires_grad=True)
    labels_th = torch.tensor(labels_np, dtype=torch.int64)
    loss_th = F.cross_entropy(logits_th, labels_th, reduction=reduction)

    # Compare forward outputs
    _compare_tensors(loss_tg.data, loss_th.detach().numpy())

    # Compare gradients (w.r.t logits)
    grad_tg, grad_th = _backward_and_get_grad(loss_tg, loss_th, logits_tg, logits_th, reduction)
    _compare_tensors(grad_tg, grad_th)


# -----------------------------------------------------------------------------
# Mean-Squared-Error Loss Tests
# -----------------------------------------------------------------------------


@pytest.mark.parametrize(
    "shape,reduction",
    [
        ((3, 4), "mean"),
        ((2, 5), "sum"),
        ((6,), "none"),
    ],
)
def test_mse_loss_matches_pytorch(shape, reduction):
    rng = np.random.default_rng(1)
    pred_np = rng.standard_normal(shape).astype(np.float32)
    target_np = rng.standard_normal(shape).astype(np.float32)

    # TensorGrad
    pred_tg = Tensor(pred_np.copy())
    target_tg = Tensor(target_np.copy(), requires_grad=False)
    loss_tg = mse_loss(pred_tg, target_tg, reduction=reduction)

    # PyTorch
    pred_th = torch.tensor(pred_np, dtype=torch.float32, requires_grad=True)
    target_th = torch.tensor(target_np, dtype=torch.float32)
    loss_th = F.mse_loss(pred_th, target_th, reduction=reduction)

    _compare_tensors(loss_tg.data, loss_th.detach().numpy())

    grad_tg, grad_th = _backward_and_get_grad(loss_tg, loss_th, pred_tg, pred_th, reduction)
    _compare_tensors(grad_tg, grad_th)


# -----------------------------------------------------------------------------
# Binary Cross-Entropy Loss Tests
# -----------------------------------------------------------------------------


@pytest.mark.parametrize(
    "shape,reduction",
    [
        ((4,), "mean"),
        ((3, 2), "sum"),
        ((5,), "none"),
    ],
)
def test_binary_cross_entropy_matches_pytorch(shape, reduction):
    rng = np.random.default_rng(2)
    # Predictions in (0,1) to avoid log(0). Add eps margin.
    pred_np = rng.uniform(0.05, 0.95, size=shape).astype(np.float32)
    target_np = rng.integers(0, 2, size=shape).astype(np.float32)

    pred_tg = Tensor(pred_np.copy())
    target_tg = Tensor(target_np.copy(), requires_grad=False)
    loss_tg = binary_cross_entropy(pred_tg, target_tg, reduction=reduction)

    pred_th = torch.tensor(pred_np, dtype=torch.float32, requires_grad=True)
    target_th = torch.tensor(target_np, dtype=torch.float32)
    loss_th = F.binary_cross_entropy(pred_th, target_th, reduction=reduction)

    _compare_tensors(loss_tg.data, loss_th.detach().numpy())

    grad_tg, grad_th = _backward_and_get_grad(loss_tg, loss_th, pred_tg, pred_th, reduction)
    _compare_tensors(grad_tg, grad_th) 