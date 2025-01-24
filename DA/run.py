import functions
import numpy as np
import torch
from enumFunctions import Functions
from DA import dragonfly_algorithm
from DA_PYTORCH import dragonfly_algorithm_pytorch




def da_pytorch():
    obj_func = functions.selectFunction(Functions.schwefel)
    dim = 30
    SearchAgents_no = 100
    Max_iteration = 100
    lb = -500 * torch.ones(dim)
    ub = 500 * torch.ones(dim)
    dragonfly_algorithm_pytorch(SearchAgents_no, Max_iteration, lb, ub, dim,obj_func)
    
def da():
    obj_func = functions.selectFunction(Functions.schwefel)
    dim = 30
    SearchAgents_no = 500
    Max_iteration = 500
    lb = -500 * np.ones(dim)
    ub = 500 * np.ones(dim)
    dragonfly_algorithm(SearchAgents_no, Max_iteration, lb, ub, dim, obj_func)
    
    

def main():
    #da_pytorch()
    da()

if __name__ == "__main__":
    main()