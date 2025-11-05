import hw3_utils
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import matplotlib.pyplot as plt

def svm_solver(x_train, y_train, lr, num_iters,
               kernel=hw3_utils.poly(degree=1), c=None):
    '''
    Computes an SVM given a training set, training labels, the number of
    iterations to perform projected gradient descent, a kernel, and a trade-off
    parameter for soft-margin SVM.

    Arguments:
        x_train: 2d tensor with shape (N, d).
        y_train: 1d tensor with shape (N,), whose elememnts are +1 or -1.
        lr: The learning rate.
        num_iters: The number of gradient descent steps.
        kernel: The kernel function.
           The default kernel function is 1 + <x, y>.
        c: The trade-off parameter in soft-margin SVM.
           The default value is None, referring to the basic, hard-margin SVM.

    Returns:
        alpha: a 1d tensor with shape (N,), denoting an optimal dual solution.
               Initialize alpha to be 0.
               Return alpha.detach() could possibly help you save some time
               when you try to use alpha in other places.

    Note that if you use something like alpha = alpha.clamp(...) with
    torch.no_grad(), you will have alpha.requires_grad=False after this step.
    You will then need to use alpha.requires_grad_().
    Alternatively, use in-place operations such as clamp_().
    '''
    # TODO
    N = x_train.shape[0]
    x = x_train.to(torch.double)
    y = y_train.to(torch.double)

    K = torch.zeros((N, N), dtype=torch.double)
    for i in range(N):
        for j in range(N):
            kij = kernel(x[i], x[j])
            if isinstance(kij, torch.Tensor):
                kij = kij.to(torch.double)
            else:
                kij = torch.as_tensor(kij, dtype=torch.double)
            K[i, j] = kij

    yy = y.view(-1, 1) @ y.view(1, -1)
    Q = K * yy

    alpha = torch.zeros(N, dtype=torch.double, requires_grad=True)
    lr_t = torch.as_tensor(lr, dtype=torch.double)
    yTy = (y @ y).clamp(min=torch.as_tensor(1.0, dtype=torch.double))
    for _ in range(num_iters):
        obj = alpha.sum() - 0.5 * (alpha @ (Q @ alpha))
        obj.backward()
        with torch.no_grad():
            alpha += lr_t * alpha.grad     
            alpha.grad.zero_()
            alpha -= ((y @ alpha) / yTy) * y
            if c is None:
                alpha.clamp_(min=0.0)
            else:
                alpha.clamp_(0.0, c)      
            alpha.requires_grad_()
            
    return alpha.detach().to(torch.double)

def svm_predictor(alpha, x_train, y_train, x_test,
                  kernel=hw3_utils.poly(degree=1)):
    '''
    Returns the kernel SVM's predictions for x_test using the SVM trained on
    x_train, y_train with computed dual variables alpha.

    Arguments:
        alpha: 1d tensor with shape (N,), denoting an optimal dual solution.
        x_train: 2d tensor with shape (N, d), denoting the training set.
        y_train: 1d tensor with shape (N,), whose elements are +1 or -1.
        x_test: 2d tensor with shape (M, d), denoting the test set.
        kernel: The kernel function.
           The default kernel function is 1 + <x, y>.

    Return:
        A 1d tensor with shape (M,), the outputs of SVM on the test set.
    '''
    # TODO
    with torch.no_grad():
        alpha = alpha.to(torch.double)
        x_tr = x_train.to(torch.double)
        y_tr = y_train.to(torch.double)
        x_te = x_test.to(torch.double)
        N, M = x_tr.shape[0], x_te.shape[0]

        eps = 1e-6
        sv_mask = alpha > eps

        
        topk = min(10, alpha.numel())
        if topk > 0:
            vals, _ = torch.topk(alpha, k=topk)
           
            plateau_val = vals[0]
            at_bound = alpha >= (plateau_val - 1e-8)
        else:
            at_bound = torch.zeros_like(alpha, dtype=torch.bool)

        margin_mask = sv_mask & (~at_bound)

        idx = torch.where(margin_mask)[0]
        if idx.numel() == 0:
            idx = torch.where(sv_mask)[0]               

        if idx.numel() == 0:
            b = torch.tensor(0.0, dtype=torch.double)
        else:
            b_vals = []
            for i in idx.tolist():
                s = torch.tensor(0.0, dtype=torch.double)
                for j in range(N):
                    kij = kernel(x_tr[j], x_tr[i])
                    if isinstance(kij, torch.Tensor):
                        kij = kij.to(torch.double)
                    else:
                        kij = torch.as_tensor(kij, dtype=torch.double)
                    s += alpha[j] * y_tr[j] * kij
                b_vals.append(y_tr[i] - s)
            b = torch.stack(b_vals).mean()

        out = torch.zeros(M, dtype=torch.double)
        for i in range(M):
            s = torch.tensor(0.0, dtype=torch.double)
            for j in range(N):
                kij = kernel(x_tr[j], x_te[i])
                if isinstance(kij, torch.Tensor):
                    kij = kij.to(torch.double)
                else:
                    kij = torch.as_tensor(kij, dtype=torch.double)
                s += alpha[j] * y_tr[j] * kij
            out[i] = s + b
        return out
