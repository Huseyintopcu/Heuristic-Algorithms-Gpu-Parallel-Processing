import random
import torch
import time
from solution import solution

def GWO_pytorch(objf, lb, ub, dim, SearchAgents_no, Max_iter):

    # Set device to 'cuda' (GPU) if available
    device = 'cuda'

    # Initialize alpha, beta, and delta positions
    Alpha_pos = torch.zeros(dim, device=device)
    Alpha_score = float("inf")

    Beta_pos = torch.zeros(dim, device=device)
    Beta_score = float("inf")

    Delta_pos = torch.zeros(dim, device=device)
    Delta_score = float("inf")

    # Ensure bounds are lists
    if not isinstance(lb, list):
        lb = [lb] * dim
    if not isinstance(ub, list):
        ub = [ub] * dim

    # Initialize the positions of search agents
    Positions = torch.rand((SearchAgents_no, dim), device=device) * (torch.tensor(ub, device=device) - torch.tensor(lb, device=device)) + torch.tensor(lb, device=device)

    Convergence_curve = torch.zeros(Max_iter, device=device)
    s = solution()

    # Loop counter
    print('GWO is optimizing  "' + objf.__name__ + '" GPU')

    timerStart = time.time()
    s.startTime = time.strftime("%Y-%m-%d-%H-%M-%S")
    
    # Main loop
    for l in range(Max_iter):
        
        # Calculate objective function for all search agents in parallel
        fitness = torch.tensor([objf(Positions[i, :].cpu().numpy()) for i in range(SearchAgents_no)], device=device)

        # Update Alpha, Beta, and Delta in parallel
        min_fitness, min_idx = torch.min(fitness, dim=0)
        if min_fitness < Alpha_score:
            Alpha_score = min_fitness
            Alpha_pos = Positions[min_idx, :].clone()

        second_min_fitness, second_min_idx = torch.min(fitness[fitness > min_fitness], dim=0)
        if second_min_fitness < Beta_score:
            Beta_score = second_min_fitness
            Beta_pos = Positions[second_min_idx, :].clone()

        third_min_fitness, third_min_idx = torch.min(fitness[(fitness > min_fitness) & (fitness > second_min_fitness)], dim=0)
        if third_min_fitness < Delta_score:
            Delta_score = third_min_fitness
            Delta_pos = Positions[third_min_idx, :].clone()

        # Update coefficient a
        a = 3 - l * ((2) / Max_iter)

        # Update the Position of search agents including omegas in parallel
        r1 = torch.rand((SearchAgents_no, dim), device=device)  # random numbers in [0,1]
        r2 = torch.rand((SearchAgents_no, dim), device=device)  # random numbers in [0,1]

        A1 = 2 * a * r1 - a
        C1 = 2 * r2

        r1 = torch.rand((SearchAgents_no, dim), device=device)
        r2 = torch.rand((SearchAgents_no, dim), device=device)

        A2 = 2 * a * r1 - a
        C2 = 2 * r2

        r1 = torch.rand((SearchAgents_no, dim), device=device)
        r2 = torch.rand((SearchAgents_no, dim), device=device)

        A3 = 2 * a * r1 - a
        C3 = 2 * r2

        # Compute all new positions in parallel
        D_alpha = torch.abs(C1 * Alpha_pos - Positions)
        X1 = Alpha_pos - A1 * D_alpha

        D_beta = torch.abs(C2 * Beta_pos - Positions)
        X2 = Beta_pos - A2 * D_beta

        D_delta = torch.abs(C3 * Delta_pos - Positions)
        X3 = Delta_pos - A3 * D_delta

        # Update positions for all agents
        Positions = (X1 + X2 + X3) / 3

        # Ensure positions are within bounds
        Positions = torch.clamp(Positions, torch.tensor(lb, device=device), torch.tensor(ub, device=device))

        # Store the best fitness value for the current iteration
        Convergence_curve[l] = Alpha_score

        if l % 1 == 0:
            print(f"At iteration {l}, the best fitness is {Alpha_score}")

    timerEnd = time.time()
    s.endTime = time.strftime("%Y-%m-%d-%H-%M-%S")
    s.executionTime = timerEnd - timerStart
    s.convergence = Convergence_curve
    s.optimizer = "GWO"
    s.objfname = objf.__name__
    
    print(s.executionTime)

    return s

# Usage example
# Assuming objf, lb, ub, dim, SearchAgents_no, Max_iter are already defined
# result = GWO(objf, lb, ub, dim, SearchAgents_no, Max_iter)
