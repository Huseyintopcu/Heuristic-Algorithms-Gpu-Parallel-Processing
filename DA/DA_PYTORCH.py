import torch
import time

def distance(a, b):
    return torch.norm(a - b, dim=1)

def dragonfly_algorithm_pytorch(SearchAgents_no, Max_iteration, lb, ub, dim, fobj):
    start_time = time.time()  # Çalışma başlangıç zamanı
    print('DA is optimizing your problem..."'+ fobj.__name__ +'" GPU')

    lb = lb.clone().detach().to(device='cuda', dtype=torch.float32)
    ub = ub.clone().detach().to(device='cuda', dtype=torch.float32)
    r = ((ub - lb) / 10).mean()  # r'yi skaler yap

    Delta_max = (ub - lb).mean() / 10

    Food_fitness = float("inf")
    Food_pos = torch.zeros(dim, device='cuda')
    Enemy_fitness = float("-inf")
    Enemy_pos = torch.zeros(dim, device='cuda')

    X = lb + (ub - lb) * torch.rand(SearchAgents_no, dim, device='cuda')
    DeltaX = torch.rand(SearchAgents_no, dim, device='cuda') * 2 - 1

    cg_curve = []

    for iter in range(Max_iteration):
        w = 0.9 - iter * ((0.9 - 0.4) / Max_iteration)
        my_c = 0.1 - iter * ((0.1 - 0) / (Max_iteration / 2))
        my_c = max(my_c, 0)

        s = 2 * torch.rand(1, device='cuda') * my_c
        a = 2 * torch.rand(1, device='cuda') * my_c
        c = 2 * torch.rand(1, device='cuda') * my_c
        f = 2 * torch.rand(1, device='cuda')
        e = my_c

        Fitness = torch.tensor([fobj(x.cpu().numpy()) for x in X], device='cuda')

        better_food = Fitness < Food_fitness
        if torch.any(better_food):
            Food_fitness = Fitness[better_food].min()
            Food_pos = X[better_food][Fitness[better_food].argmin()]

        worse_enemy = Fitness > Enemy_fitness
        valid_bounds = torch.all(X <= ub, dim=1) & torch.all(X >= lb, dim=1)
        worse_enemy &= valid_bounds
        if torch.any(worse_enemy):
            Enemy_fitness = Fitness[worse_enemy].max()
            Enemy_pos = X[worse_enemy][Fitness[worse_enemy].argmax()]

        Distances = torch.cdist(X, X)  # Mesafeler
        Neighbours = Distances <= r  # r skaler

        S = torch.zeros_like(X, device='cuda')
        A = torch.zeros_like(X, device='cuda')
        C = torch.zeros_like(X, device='cuda')

        for i in range(SearchAgents_no):
            Neighbours_X = X[Neighbours[i]]
            Neighbours_DeltaX = DeltaX[Neighbours[i]]

            if Neighbours_X.size(0) > 0:
                S[i] = -torch.sum(Neighbours_X - X[i], dim=0)
                A[i] = torch.mean(Neighbours_DeltaX, dim=0)
                C[i] = torch.mean(Neighbours_X, dim=0) - X[i]

        F = Food_pos - X
        Enemy = Enemy_pos + X

        DeltaX = w * DeltaX + s * S + a * A + c * C + f * F + e * Enemy
        DeltaX = torch.clamp(DeltaX, -Delta_max, Delta_max)
        X += DeltaX
        X = torch.clamp(X, lb, ub)

        cg_curve.append(Food_fitness.item())
        print(["At iteration " + str(iter) + " the best fitness is " + str(Food_fitness)])
    end_time = time.time()  # Çalışma bitiş zamanı
    elapsed_time = end_time - start_time  # Geçen süre hesaplama

    print(f"Best fitness: {Food_fitness.item()}")
    print(f"Elapsed time: {elapsed_time:.2f} seconds")  # Çalışma süresini yazdır
    return Food_fitness.item(), Food_pos.cpu().numpy(), cg_curve
