import torch
# Importing our custom module(s)
import utils

class RandomFourierFeatures(torch.nn.Module):
    def __init__(
        self, 
        in_features : int, 
        learnable_lengthscale : bool = False, 
        learnable_outputscale : bool = False, 
        lengthscale : float = 20.0, 
        outputscale : float = 1.0, 
        rank : int = 1024,
    ):
        super().__init__()
        
        self.rank = rank
        self.register_buffer("feature_weight", torch.randn(self.rank, in_features))
        self.register_buffer("feature_bias", 2 * torch.pi * torch.rand(self.rank))
                
        if learnable_lengthscale:
            self.raw_lengthscale = torch.nn.Parameter(utils.inv_softplus(torch.tensor(lengthscale, dtype=torch.float32)))
        else:
            self.register_buffer("raw_lengthscale", utils.inv_softplus(torch.tensor(lengthscale, dtype=torch.float32)))
        
        if learnable_outputscale:
            self.raw_outputscale = torch.nn.Parameter(utils.inv_softplus(torch.tensor(outputscale, dtype=torch.float32)))
        else:
            self.register_buffer("raw_outputscale", utils.inv_softplus(torch.tensor(outputscale, dtype=torch.float32)))
                    
    def forward(
        self, 
        x : torch.Tensor,
    ) -> torch.Tensor:
        return self.outputscale * (2/self.rank)**0.5 * torch.cos(torch.nn.functional.linear(x, (1/self.lengthscale) * self.feature_weight, self.feature_bias))
                                                            
    @property
    def lengthscale(
        self,
    ) -> torch.Tensor:
        return torch.nn.functional.softplus(self.raw_lengthscale)
    
    @property
    def outputscale(
        self,
    ) -> torch.Tensor:
        return torch.nn.functional.softplus(self.raw_outputscale)
