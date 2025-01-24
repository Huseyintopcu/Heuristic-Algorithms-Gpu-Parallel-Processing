import functions
import numpy as np
from enumFunctions import Functions
from GWO import GWO
from GWO_pytorch import GWO_pytorch

def gwo():
    obj_func = functions.selectFunction(Functions.schwefel)
    # dim array size, -5 lb +5 lb 
    GWO(obj_func, -500, 500, 30, 100, 100)

def gwo_pytorch():
    obj_func = functions.selectFunction(Functions.schwefel)
    # dim array size, -5 lb +5 lb 
    GWO_pytorch(obj_func, -500, 500, 30, 100, 100)
    

def main():
    gwo_pytorch()
if __name__ == "__main__":
    main()