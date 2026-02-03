import math
import torch

def inv_softplus(
    x : torch.Tensor
) -> torch.Tensor:
    return x + torch.log(-torch.expm1(-x))

def lml(
    Phi : torch.Tensor, 
    sigma_y2 : torch.Tensor, 
    y : torch.Tensor,
    tau : torch.Tensor = torch.tensor(1.0),
) -> torch.Tensor:
    N, R = Phi.shape
    Sigma = sigma_y2 * torch.eye(N, device=Phi.device) + tau * (Phi @ Phi.t())
    L = torch.linalg.cholesky(Sigma)
    alpha = torch.cholesky_solve(y, L)
    norm_const = N * math.log(2 * math.pi)
    log_det = 2 * torch.sum(torch.log(torch.diag(L)))
    quad_term = y.t() @ alpha
    lml = -0.5 * (norm_const + log_det + quad_term)
    return lml

def diag_elbo(
    mu_q : torch.Tensor, 
    Phi : torch.Tensor, 
    sigma_q2 : torch.Tensor, 
    sigma_y2 : torch.Tensor,
    y : torch.Tensor, 
    tau : torch.Tensor = torch.tensor(1.0),
    temp : torch.Tensor = torch.tensor(1.0),
) -> torch.Tensor:
    N, R = Phi.shape
    norm_const = N * torch.log(2 * torch.pi * sigma_y2)
    residual = y - Phi @ mu_q
    quad_term = (1 / sigma_y2) * ((residual.t() @ residual) + torch.dot(sigma_q2, torch.sum(Phi**2, dim=0)))
    nll = -0.5 * (norm_const + quad_term)
    trace = (1 / tau) * sigma_q2.sum()
    quad_term = (1 / tau) * (mu_q ** 2).sum()
    log_det = R * torch.log(tau) - torch.log(sigma_q2).sum()
    kl = 0.5 * (trace + quad_term - R + log_det)
    return (1 / temp) * nll - kl

def rank1_elbo(
    eps : torch.Tensor, 
    mu_q : torch.Tensor, 
    Phi : torch.Tensor, 
    sigma_y2 : torch.Tensor,
    v_q : torch.Tensor, 
    y : torch.Tensor, 
    tau : torch.Tensor = torch.tensor(1.0),
    temp : torch.Tensor = torch.tensor(1.0),
) -> torch.Tensor:
    N, R = Phi.shape
    norm_const = N * torch.log(2 * torch.pi * sigma_y2)
    residual = y - Phi @ mu_q
    quad_term = (1 / sigma_y2) * ((residual.t() @ residual) + (v_q.t() @ Phi.t() @ Phi @ v_q) + (eps * torch.trace(Phi.t() @ Phi)))
    nll = -0.5 * (norm_const + quad_term)
    trace = (1 / tau) * ((v_q ** 2).sum() + (eps * R))
    quad_term = (1 / tau) * (mu_q ** 2).sum()
    log_det = R * torch.log(tau) - (R - 1) * torch.log(eps) - torch.log((v_q ** 2).sum() + eps)
    kl = 0.5 * (trace + quad_term - R + log_det)
    return (1 / temp) * nll - kl

def fullrank_elbo(
    mu_q : torch.Tensor, 
    Phi : torch.Tensor, 
    Sigma_q : torch.Tensor, 
    sigma_y2 : torch.Tensor,
    y : torch.Tensor, 
    tau : torch.Tensor = torch.tensor(1.0),
    temp : torch.Tensor = torch.tensor(1.0),
) -> torch.Tensor:
    N, R = Phi.shape
    norm_const = N * torch.log(2 * torch.pi * sigma_y2)
    residual = y - Phi @ mu_q
    quad_term = (1 / sigma_y2) * ((residual.t() @ residual) + torch.trace(Phi @ Sigma_q @ Phi.t()))
    nll = -0.5 * (norm_const + quad_term)
    L_q = torch.linalg.cholesky(Sigma_q)
    trace = (1 / tau) * (L_q ** 2).sum()
    quad_term = (1 / tau) * (mu_q ** 2).sum()
    log_det = R * torch.log(tau) - 2 * L_q.diag().log().sum()
    kl = 0.5 * (trace + quad_term - R + log_det)
    return (1 / temp) * nll - kl
