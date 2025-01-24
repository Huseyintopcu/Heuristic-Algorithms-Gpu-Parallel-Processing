import torch
import numpy as np
from numpy import sin, cos, tan ,cosh, tanh, sinh, abs, exp, mean, pi, prod, sqrt, sum

function = "sum(x**2)"

def createFunction(f):
    global function
    function = f

def custom(x):
    x = torch.tensor(x, dtype=torch.float32)
    return eval(function)

def selectFunction(cbIndex):
    switcher = {
        0: ackley,
        1: dixonprice,
        2: griewank,
        3: michalewicz,
        4: perm,
        5: powell,
        6: powersum,
        7: rastrigin,
        8: rosenbrock,
        9: schwefel,
        10: sphere,
        11: sum2,
        12: trid,
        13: zakharov,
        14: ellipse,
        15: nesterov,
        16: saddle,
        17: custom
    }
    return switcher.get(cbIndex, "nothing")

def ackley(x, a=20, b=0.2, c=2*pi):
    x = torch.tensor(x, dtype=torch.float32)
    n = len(x)
    s1 = torch.sum(x**2)
    s2 = torch.sum(cos(c * x))
    return -a*exp(-b*sqrt(s1 / n)) - exp(s2 / n) + a + exp(1)

def dixonprice(x):
    x = torch.tensor(x, dtype=torch.float32)
    n = len(x)
    j = torch.arange(2, n+1)
    x2 = 2 * x**2
    return torch.sum(j * (x2[1:] - x[:-1])**2) + (x[0] - 1)**2

def griewank(x, fr=4000):
    x = torch.tensor(x, dtype=torch.float32)
    n = len(x)
    j = torch.arange(1., n+1)
    s = torch.sum(x**2)
    p = torch.prod(cos(x / sqrt(j)))
    return s/fr - p + 1

def levy(x):
    x = torch.tensor(x, dtype=torch.float32)
    n = len(x)
    z = 1 + (x - 1) / 4
    return (sin(pi * z[0])**2 + torch.sum((z[:-1] - 1)**2 * (1 + 10 * sin(pi * z[:-1] + 1)**2)) +
            (z[-1] - 1)**2 * (1 + sin(2 * pi * z[-1])**2))

michalewicz_m = .5

def michalewicz(x):
    x = torch.tensor(x, dtype=torch.float32)
    n = len(x)
    j = torch.arange(1., n+1)
    return -torch.sum(sin(x) * sin(j * x**2 / pi)**(2 * michalewicz_m))

def perm(x, b=.5):
    x = torch.tensor(x, dtype=torch.float32)
    n = len(x)
    j = torch.arange(1., n+1)
    xbyj = torch.abs(x) / j
    return mean([mean((j**k + b) * (xbyj ** k - 1))**2 for k in j/n])

def powell(x):
    x = torch.tensor(x, dtype=torch.float32)
    n = len(x)
    n4 = ((n + 3) // 4) * 4
    if n < n4:
        x = torch.cat([x, torch.zeros(n4 - n)])
    x = x.view(4, -1)
    f = torch.empty_like(x)
    f[0] = x[0] + 10 * x[1]
    f[1] = sqrt(5) * (x[2] - x[3])
    f[2] = (x[1] - 2 * x[2])**2
    f[3] = sqrt(10) * (x[0] - x[3])**2
    return torch.sum(f**2)

def powersum(x, b=[8,18,44,114]):
    x = torch.tensor(x, dtype=torch.float32)
    n = len(x)
    s = 0
    for k in range(1, n+1):
        bk = b[min(k - 1, len(b) - 1)]
        s += (torch.sum(x**k) - bk)**2
    return s

def rastrigin(x):
    x = torch.tensor(x, dtype=torch.float32)
    n = len(x)
    return 10 * n + torch.sum(x**2 - 10 * cos(2 * pi * x))

def rosenbrock(x):
    x = torch.tensor(x, dtype=torch.float32)
    x0 = x[:-1]
    x1 = x[1:]
    return torch.sum((1 - x0)**2) + 100 * torch.sum((x1 - x0**2)**2)

def schwefel(x):
    x = torch.tensor(x, dtype=torch.float32)
    if torch.any(~torch.isfinite(x)):
        print("Warning: Non-finite values detected in the input.")
        x = torch.nan_to_num(x)
    n = len(x)
    return 418.9829 * n - torch.sum(x * torch.sin(torch.sqrt(torch.abs(x))))

def sphere(x):
    x = torch.tensor(x, dtype=torch.float32)
    return torch.sum(x**2)

def sum2(x):
    x = torch.tensor(x, dtype=torch.float32)
    n = len(x)
    j = torch.arange(1., n+1)
    return torch.sum(j * x**2)

def trid(x):
    x = torch.tensor(x, dtype=torch.float32)
    return torch.sum((x - 1)**2) - torch.sum(x[:-1] * x[1:])

def zakharov(x):
    x = torch.tensor(x, dtype=torch.float32)
    n = len(x)
    j = torch.arange(1., n+1)
    s2 = torch.sum(j * x) / 2
    return torch.sum(x**2) + s2**2 + s2**4

def ellipse(x):
    x = torch.tensor(x, dtype=torch.float32)
    return torch.mean((1 - x)**2) + 100 * torch.mean(torch.diff(x)**2)

def nesterov(x):
    x = torch.tensor(x, dtype=torch.float32)
    x0 = x[:-1]
    x1 = x[1:]
    return abs(1 - x[0]) / 4 + torch.sum(abs(x1 - 2*abs(x0) + 1))

def saddle(x):
    x = torch.tensor(x, dtype=torch.float32) - 1
    return torch.mean(torch.diff(x**2)) + 0.5 * torch.mean(x**4)

