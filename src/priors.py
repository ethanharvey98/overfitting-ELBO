import torch
# Importing our custom module(s)
import utils

class GaussianPrior(torch.nn.Module):
    def __init__(
        self,
        learnable_variance: bool = False, 
        variance: float = 1.0,
    ):
        super().__init__()
                
        if learnable_variance:
            self.raw_variance = torch.nn.Parameter(utils.inv_softplus(torch.tensor(variance, dtype=torch.float64)))
        else:
            self.register_buffer("raw_variance", utils.inv_softplus(torch.tensor(variance, dtype=torch.float64)))
                
    @property
    def variance(
        self,
    ) -> torch.Tensor:
        return torch.nn.functional.softplus(self.raw_variance)
