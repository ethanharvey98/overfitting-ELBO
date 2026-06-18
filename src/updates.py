import torch

def diag_m_update(
    Phi: torch.Tensor, 
    sigma_y2: torch.Tensor, 
    y: torch.Tensor, 
    tau: torch.Tensor = torch.tensor(1.0),
    temp: torch.Tensor = torch.tensor(1.0),
) -> torch.Tensor:
    N, R = Phi.shape
    A = (1 / (temp * sigma_y2)) * (Phi.t() @ Phi) + (1 / tau) * torch.eye(R, device=Phi.device, dtype=Phi.dtype)
    b = (1 / (temp * sigma_y2)) * (Phi.t() @ y)
    m_star = torch.linalg.solve(A, b)
    return m_star

def diag_s2_update(
    Phi: torch.Tensor, 
    sigma_y2: torch.Tensor, 
    tau: torch.Tensor = torch.tensor(1.0),
    temp: torch.Tensor = torch.tensor(1.0),
) -> torch.Tensor:
    s2_star = 1.0 / ((1.0 / tau) + ((1 / (temp * sigma_y2)) * torch.sum(Phi ** 2, dim=0)))
    return s2_star

def diag_sigma_y2_update(
    m: torch.Tensor, 
    Phi: torch.Tensor, 
    s2: torch.Tensor,
    y: torch.Tensor,
) -> torch.Tensor:
    N, R = Phi.shape
    residual = y - Phi @ m
    sigma_y2_star = (1 / N) * ((residual ** 2).sum() + torch.dot(s2, torch.sum(Phi**2, dim=0)))
    return sigma_y2_star

def diag_sigma_k_update(
    m: torch.Tensor, 
    Phi: torch.Tensor, 
    s2: torch.Tensor,
    y: torch.Tensor,
) -> torch.Tensor:
    Phi_m = Phi @ m
    sigma_k_star = torch.sum(y * Phi_m) / (torch.sum(Phi_m ** 2) + torch.dot(s2, torch.sum(Phi ** 2, dim=0)))
    return sigma_k_star

def rank1_v_q_update(
    eps: torch.Tensor, 
    Phi: torch.Tensor, 
    tau: torch.Tensor = torch.tensor(1.0),
    temp: torch.Tensor = torch.tensor(1.0),
) -> torch.Tensor:
    #eigenvalues, eigenvectors = torch.linalg.eigh(Phi.t() @ Phi)
    U, S, Vh = torch.linalg.svd(Phi)
    eigenvalues, eigenvectors = (S ** 2).flip(dims=(0,)), Vh.t().flip(dims=(1,))
    return torch.sqrt(tau - eps) * eigenvectors[:,0].reshape(-1, 1)

def rank1_m_update(
    Phi: torch.Tensor, 
    sigma_y2: torch.Tensor, 
    y: torch.Tensor, 
    tau: torch.Tensor = torch.tensor(1.0),
    temp: torch.Tensor = torch.tensor(1.0),
) -> torch.Tensor:
    N, R = Phi.shape
    A = (1 / (temp * sigma_y2)) * (Phi.t() @ Phi) + (1 / tau) * torch.eye(R, device=Phi.device, dtype=Phi.dtype)
    b = (1 / (temp * sigma_y2)) * (Phi.t() @ y)
    m_star = torch.linalg.solve(A, b)
    return m_star
    
def rank1_sigma_y2_update(
    eps: torch.Tensor, 
    m: torch.Tensor, 
    Phi: torch.Tensor, 
    v_q: torch.Tensor, 
    y: torch.Tensor, 
) -> torch.Tensor:
    N, R = Phi.shape
    residual = y - Phi @ m
    sigma_y2_star = (1/ N) * ((residual ** 2).sum() + (v_q.t() @ Phi.t() @ Phi @ v_q)  + eps * (Phi ** 2).sum())
    return sigma_y2_star

def fullrank_m_update(
    Phi: torch.Tensor, 
    sigma_y2: torch.Tensor, 
    y: torch.Tensor, 
    tau: torch.Tensor = torch.tensor(1.0), 
    temp: torch.Tensor = torch.tensor(1.0), 
) -> torch.Tensor:
    N, R = Phi.shape
    A = (1 / (temp * sigma_y2)) * (Phi.t() @ Phi) + (1 / tau) * torch.eye(R, device=Phi.device, dtype=Phi.dtype)
    b = (1 / (temp * sigma_y2)) * (Phi.t() @ y)
    m_star = torch.linalg.solve(A, b)
    return m_star

def fullrank_S_update(
    Phi: torch.Tensor, 
    sigma_y2: torch.Tensor, 
    tau: torch.Tensor = torch.tensor(1.0), 
    temp: torch.Tensor = torch.tensor(1.0), 
) -> torch.Tensor:
    N, R = Phi.shape
    S_inv = (1 / (temp * sigma_y2)) * (Phi.t() @ Phi) + (1 / tau) * torch.eye(R, device=Phi.device, dtype=Phi.dtype)
    S_star = torch.linalg.inv(S_inv)
    return S_star

def fullrank_sigma_y2_update(
    m: torch.Tensor, 
    Phi: torch.Tensor, 
    S: torch.Tensor, 
    y: torch.Tensor, 
) -> torch.Tensor:
    N, R = Phi.shape
    residual = y - Phi @ m
    sigma_y2_star = (1/ N) * ((residual ** 2).sum() + torch.trace(Phi @ S @ Phi.t()))
    return sigma_y2_star

def fullrank_sigma_k_update(
    m: torch.Tensor, 
    Phi: torch.Tensor, 
    S: torch.Tensor, 
    y: torch.Tensor, 
) -> torch.Tensor:
    Phi_m = Phi @ m
    sigma_k_star = torch.sum(y * Phi_m) / (torch.sum(Phi_m ** 2) + torch.trace(Phi @ S @ Phi.t()))
    return sigma_k_star
