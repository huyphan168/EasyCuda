from torch.utils.cpp_extension import load
import torch
import time
import matplotlib.pyplot as plt

from torch.utils.cpp_extension import load
matmul_cuda = load(
    'matmul_cuda', ['matmul.cpp', 'matmul_kernel.cu'], verbose=True)

# Set hyperparameters
T = 256    # Number of threads
TB = 1     # Number of thread blocks
runtimes = []
shapes = [2,4,8,16,32,64,128,256]

for n in shapes:
    A = torch.rand(n, n).cuda()   # Create random n x n matrix 
    B = torch.rand(n).cuda()      # Create random n-dim vector
    
    start = time.time()
    for i in range(100): 
        C = matmul_cuda.parallelMatMul(A, B, T, TB)    # Matrix multiplication
    end = time.time()
    runtimes.append(end - start)

plt.plot(shapes, runtimes)
plt.savefig('runtimes.png')