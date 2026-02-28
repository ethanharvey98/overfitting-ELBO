import argparse
import os
import copy
import random
import numpy as np
import torch
# Importing our custom module(s)
import layers
import updates
import utils

if __name__=="__main__":
    parser = argparse.ArgumentParser(description="fullrank.py")
    parser.add_argument("--experiment_path", default="", help="Path to save experiment (default: \"\")", type=str)
    parser.add_argument('--lrs', default=[0.1, 0.01, 0.001, 0.0001], help='Ranks (default: [0.1, 0.01, 0.001, 0.0001])', nargs='+', type=float)
    parser.add_argument("--N", default=20, help="Number of training samples (default: 20)", type=int)
    parser.add_argument("--num_epochs", default=1_000, help="Number of epochs (default: 1,000)", type=int)
    parser.add_argument("--num_samples", default=10, help="Number of random Fourier features samples (default: 10)", type=int)
    parser.add_argument('--ranks', default=[1, 10, 100, 1_000, 10_000], help='Ranks (default: [1, 10, 100, 1,000, 10,000])', nargs='+', type=int)
    parser.add_argument("--seed", default=42, help="Random seed (default: 42)", type=int)
    args = parser.parse_args()
    
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    
    os.makedirs(os.path.dirname(args.experiment_path), exist_ok=True)

    X_numpy = np.random.randn(args.N)
    y_numpy = np.sin(3 * X_numpy) + 0.1 * np.random.randn(args.N)

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print(device)
    
    X = torch.tensor(X_numpy.reshape(-1, 1), dtype=torch.float64).to(device)
    y = torch.tensor(y_numpy.reshape(-1, 1), dtype=torch.float64).to(device)
        
    elbos = torch.full(size=(len(args.ranks), args.num_samples, len(args.lrs), args.num_epochs), fill_value=float("-inf"))
    lmls = torch.full(size=(len(args.ranks), args.num_samples, len(args.lrs), args.num_epochs), fill_value=float("-inf"))
    
    for i in range(len(args.ranks)):
    
        for j in range(args.num_samples):

            phi = layers.RandomFourierFeatures(in_features=1, learnable_lengthscale=True, learnable_outputscale=True, lengthscale=1.0, outputscale=1.0, rank=args.ranks[i]).to(device)
            init_state_dict = copy.deepcopy(phi.state_dict())
            
            for k in range(len(args.lrs)):

                phi = layers.RandomFourierFeatures(in_features=1, learnable_lengthscale=True, learnable_outputscale=True, lengthscale=1.0, outputscale=1.0, rank=args.ranks[i]).to(device)
                phi.load_state_dict(init_state_dict)

                eps = torch.tensor(1e-6)
                sigma_y2_star = torch.tensor(1.0)
                tau = torch.tensor(1.0)
                temp = torch.tensor(1.0)

                optimizer = torch.optim.Adam(phi.parameters(), lr=args.lrs[k])
                
                lr_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, args.num_epochs)

                for l in range(args.num_epochs):

                    # Coordinate ascent updates
                    with torch.no_grad():
                        Phi = phi(X)
                        mu_q_star = updates.fullrank_mu_q_update(Phi, sigma_y2_star, y, tau=tau, temp=temp)
                        Sigma_q_star = updates.fullrank_Sigma_q_update(Phi, sigma_y2_star, tau=tau, temp=temp)
                        sigma_y2_star = updates.fullrank_sigma_y2_update(mu_q_star, Phi, Sigma_q_star, y)

                    # Gradient descent updates
                    optimizer.zero_grad()
                    Phi = phi(X)
                    elbo = utils.fullrank_elbo(mu_q_star, Phi, Sigma_q_star, sigma_y2_star, y, tau=tau, temp=temp)
                    (-elbo).backward()
                    optimizer.step()
                    
                    lr_scheduler.step()

                    with torch.no_grad():
                        lml = utils.lml(phi(X), sigma_y2_star, y, tau=tau)
                        
                    elbos[i,j,k,l] = elbo.item()
                    lmls[i,j,k,l] = lml.item()
                    
            torch.save({
                "elbos": elbos,
                "lmls": lmls,
                "ranks": torch.tensor(args.ranks),
            }, args.experiment_path)
