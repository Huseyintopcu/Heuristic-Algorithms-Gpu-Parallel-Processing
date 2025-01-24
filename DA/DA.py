import numpy as np
import time  # Çalışma süresi ölçümü için gerekli

def distance(a, b):
    return np.linalg.norm(a - b)

def dragonfly_algorithm(SearchAgents_no, Max_iteration, lb, ub, dim, fobj):
    start_time = time.time()  # Çalışma başlangıç zamanı

    print('DA is optimizing your problem..."'+ fobj.__name__ +'" CPU')
    r = (ub - lb) / 10
    Delta_max = (ub - lb) / 10

    Food_fitness = float("inf")
    Food_pos = np.zeros(dim)

    Enemy_fitness = float("-inf")
    Enemy_pos = np.zeros(dim)

    X = np.random.uniform(low=lb, high=ub, size=(SearchAgents_no, dim))
    DeltaX = np.random.uniform(low=-1, high=1, size=(SearchAgents_no, dim))

    cg_curve = []

    for iter in range(Max_iteration):
        r = (ub - lb) / 4 + ((ub - lb) * (iter / Max_iteration) * 2)
        w = 0.9 - iter * ((0.9 - 0.4) / Max_iteration)
        my_c = 0.1 - iter * ((0.1 - 0) / (Max_iteration / 2))
        my_c = max(my_c, 0)

        s = 2 * np.random.rand() * my_c
        a = 2 * np.random.rand() * my_c
        c = 2 * np.random.rand() * my_c
        f = 2 * np.random.rand()
        e = my_c

        Fitness = np.array([fobj(x) for x in X])

        # Update Food and Enemy
        for i in range(SearchAgents_no):
            if Fitness[i] < Food_fitness:
                Food_fitness = Fitness[i]
                Food_pos = X[i]
            if Fitness[i] > Enemy_fitness and np.all(X[i] <= ub) and np.all(X[i] >= lb):
                Enemy_fitness = Fitness[i]
                Enemy_pos = X[i]

        for i in range(SearchAgents_no):
            Neighbours_X = []
            Neighbours_DeltaX = []

            # Find neighbours
            for j in range(SearchAgents_no):
                Dist2Enemy = distance(X[i], X[j])
                if np.all(Dist2Enemy <= r) and not np.array_equal(X[i], X[j]):
                    Neighbours_X.append(X[j])
                    Neighbours_DeltaX.append(DeltaX[j])

            Neighbours_X = np.array(Neighbours_X)
            Neighbours_DeltaX = np.array(Neighbours_DeltaX)

            # Separation
            S = np.zeros(dim)
            if len(Neighbours_X) > 0:
                S = -np.sum(Neighbours_X - X[i], axis=0)

            # Alignment
            A = np.zeros(dim)
            if len(Neighbours_DeltaX) > 0:
                A = np.mean(Neighbours_DeltaX, axis=0)

            # Cohesion
            C = np.zeros(dim)
            if len(Neighbours_X) > 0:
                C = np.mean(Neighbours_X, axis=0) - X[i]

            # Attraction to food
            Dist2Food = distance(X[i], Food_pos)
            F = np.zeros(dim)
            if np.all(Dist2Enemy <= r):
                F = Food_pos - X[i]

            # Distraction from enemy
            Dist2Enemy = distance(X[i], Enemy_pos)
            Enemy = np.zeros(dim)
            if np.all(Dist2Enemy <= r):
                Enemy = Enemy_pos + X[i]

            # Update positions
            if np.all(Dist2Food > r) and len(Neighbours_X) > 0:
                DeltaX[i] = w * DeltaX[i] + np.random.rand() * A + np.random.rand() * C + np.random.rand() * S
                DeltaX[i] = np.clip(DeltaX[i], -Delta_max, Delta_max)
                X[i] += DeltaX[i]
            else:
                DeltaX[i] = (a * A + c * C + s * S + f * F + e * Enemy) + w * DeltaX[i]
                DeltaX[i] = np.clip(DeltaX[i], -Delta_max, Delta_max)
                X[i] += DeltaX[i]

            # Ensure boundaries
            X[i] = np.clip(X[i], lb, ub)

        cg_curve.append(Food_fitness)
        #print(["At iteration " + str(iter) + " the best fitness is " + str(Food_fitness)])

    end_time = time.time()  # Çalışma bitiş zamanı
    elapsed_time = end_time - start_time  # Geçen süre hesaplama

    print("Best fitness:", Food_fitness)
    print(f"Elapsed time: {elapsed_time:.2f} seconds")  # Çalışma süresini yazdır
    return Food_fitness, Food_pos, cg_curve
