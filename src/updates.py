import torch

def diag_mu_q_update(
    Phi : torch.Tensor, 
    sigma_y2 : torch.Tensor, 
    y : torch.Tensor, 
    tau : torch.Tensor = torch.tensor(1.0),
    temp : torch.Tensor = torch.tensor(1.0),
) -> torch.Tensor:
    N, R = Phi.shape
    A = (1 / (temp * sigma_y2)) * (Phi.t() @ Phi) + (1 / tau) * torch.eye(R, device=Phi.device, dtype=Phi.dtype)
    b = (1 / (temp * sigma_y2)) * (Phi.t() @ y)
    mu_q_star = torch.linalg.solve(A, b)
    return mu_q_star

def diag_sigma_q2_update(
    Phi: torch.Tensor, 
    sigma_y2: torch.Tensor, 
    tau : torch.Tensor = torch.tensor(1.0),
    temp : torch.Tensor = torch.tensor(1.0),
) -> torch.Tensor:
    sigma_q2_star = 1.0 / ((1.0 / tau) + ((1 / (temp * sigma_y2)) * torch.sum(Phi ** 2, dim=0)))
    return sigma_q2_star

def diag_sigma_y2_update(
    mu_q: torch.Tensor, 
    Phi: torch.Tensor, 
    sigma_q2: torch.Tensor,
    y: torch.Tensor,
) -> torch.Tensor:
    N, R = Phi.shape
    residual = y - Phi @ mu_q
    sigma_y2_star = (1 / N) * ((residual ** 2).sum() + torch.dot(sigma_q2, torch.sum(Phi**2, dim=0)))
    return sigma_y2_star

def rank1_v_q_update(
    eps : torch.Tensor, 
    Phi : torch.Tensor, 
    tau : torch.Tensor = torch.tensor(1.0),
    temp : torch.Tensor = torch.tensor(1.0),
) -> torch.Tensor:
    eigenvalues, eigenvectors = torch.linalg.eigh(Phi.t() @ Phi)
    return torch.sqrt(tau - eps) * eigenvectors[:,0].reshape(-1, 1)

def rank1_mu_q_update(
    Phi : torch.Tensor, 
    sigma_y2 : torch.Tensor, 
    y : torch.Tensor, 
    tau : torch.Tensor = torch.tensor(1.0),
    temp : torch.Tensor = torch.tensor(1.0),
) -> torch.Tensor:
    N, R = Phi.shape
    A = (1 / (temp * sigma_y2)) * (Phi.t() @ Phi) + (1 / tau) * torch.eye(R, device=Phi.device, dtype=Phi.dtype)
    b = (1 / (temp * sigma_y2)) * (Phi.t() @ y)
    mu_q_star = torch.linalg.solve(A, b)
    return mu_q_star
    
def rank1_sigma_y2_update(
    eps : torch.Tensor, 
    mu_q : torch.Tensor, 
    Phi : torch.Tensor, 
    v_q : torch.Tensor, 
    y : torch.Tensor, 
) -> torch.Tensor:
    N, R = Phi.shape
    residual = y - Phi @ mu_q
    sigma_y2_star = (1/ N) * ((residual ** 2).sum() + (v_q.t() @ Phi.t() @ Phi @ v_q)  + eps * (Phi ** 2).sum())
    return sigma_y2_star

def fullrank_mu_q_update(
    Phi : torch.Tensor, 
    sigma_y2 : torch.Tensor, 
    y : torch.Tensor, 
    tau : torch.Tensor = torch.tensor(1.0), 
    temp : torch.Tensor = torch.tensor(1.0), 
) -> torch.Tensor:
    N, R = Phi.shape
    A = (1 / (temp * sigma_y2)) * (Phi.t() @ Phi) + (1 / tau) * torch.eye(R, device=Phi.device, dtype=Phi.dtype)
    b = (1 / (temp * sigma_y2)) * (Phi.t() @ y)
    mu_q_star = torch.linalg.solve(A, b)
    return mu_q_star

def fullrank_Sigma_q_update(
    Phi: torch.Tensor, 
    sigma_y2: torch.Tensor, 
    tau: torch.Tensor = torch.tensor(1.0), 
    temp: torch.Tensor = torch.tensor(1.0), 
) -> torch.Tensor:
    N, R = Phi.shape
    Sigma_q_inv = (1 / (temp * sigma_y2)) * (Phi.t() @ Phi) + (1 / tau) * torch.eye(R, device=Phi.device, dtype=Phi.dtype)
    Sigma_q_star = torch.linalg.inv(Sigma_q_inv)
    return Sigma_q_star

def fullrank_sigma_y2_update(
    mu_q : torch.Tensor, 
    Phi : torch.Tensor, 
    Sigma_q : torch.Tensor, 
    y : torch.Tensor, 
) -> torch.Tensor:
    N, R = Phi.shape
    residual = y - Phi @ mu_q
    sigma_y2_star = (1/ N) * ((residual ** 2).sum() + torch.trace(Phi @ Sigma_q @ Phi.t()))
    return sigma_y2_star
