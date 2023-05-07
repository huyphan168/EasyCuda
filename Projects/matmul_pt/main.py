from torch.utils.cpp_extension import load
import torch
import time
import matplotlib.pyplot as plt
import numpy as np

from torch.utils.cpp_extension import load
matmul_cuda = load(
    'matmul_cuda', ['matmul.cpp', 'matmul_kernel.cu'], verbose=True)

# Set hyperparameters
T = 256    # Number of threads
TB = 1     # Number of thread blocks
runtimes = []
shapes =    [32,64,128,256]

for n in shapes:
    A = torch.rand(n).cuda()   # Create random n x n matrix 
    B = torch.rand(n,n).cuda()      # Create random n-dim vector
    
    start = time.time()
    for i in range(100): 
        C = matmul_cuda.parallelMatMul(A, B, T, TB)    # Matrix multiplication
    end = time.time()
    runtimes.append(end - start)
    
plt.plot(shapes, runtimes)
plt.xlabel('Matrix dimension')
plt.ylabel('Runtime (s)')
plt.savefig('runtimes_matrix_vector.png')

# matrix-matrix multiplication
plt.figure()
avg_error = 0
runtimes = []
shapes = [100,2,4,8,16,32,40,64,70,128,256]

for n in shapes:
    A = torch.ones((n, n)).cuda()   # Create random n x n matrix 
    B = torch.ones((n, n)).cuda()      # Create random n-dim vector
    error = 0
    start = time.time()
    for i in range(100): 
        C = matmul_cuda.parallelMatMul(A, B, T, TB)    # Matrix multiplication
        error += torch.mean(A.matmul(B) - C)
    avg_error += error/100
    end = time.time()
    runtimes.append(end - start)
    
print("Average Error of Matrix Multiplication is: ", avg_error.item()/len(shapes))
plt.plot(shapes[1:], runtimes[1:])
plt.xlabel('Matrix dimension')
plt.ylabel('Runtime (s)')
plt.savefig('runtimes_matrix_matrix.png')

# Exercise 4

T = 256    # Number of threads
TB_values = [2,4,8,16,32,64,128,256]     # Number of thread blocks
runtimes = [[] for _ in range(len(TB_values))]   # List of runtimes for each TB value
shapes = [2,4,8,16,32,64,128,256]

for i, TB in enumerate(TB_values):
    for n in shapes:
        A = torch.rand(n).cuda()   # Create random n x n matrix 
        B = torch.rand(n, n).cuda()      # Create random n-dim vector

        start = time.time()
        for j in range(100): 
            C = matmul_cuda.parallelMatMul(A, B, T, TB)    # Matrix multiplication
        end = time.time()
        runtimes[i].append(end - start)
    
fig, axes = plt.subplots(nrows=2, ncols=4, figsize=(18, 6))
plt.subplots_adjust(left=0.1, bottom=0.1, right=0.9, top=0.9, wspace=0.4, hspace=0.4)

# Flatten axes array for easy iteration
axes_flat = np.array(axes).flatten()

for i, (TB, ax) in enumerate(zip(TB_values, axes_flat)):
    ax.plot(shapes, runtimes[i], label=f'TB={TB}')
    ax.set_xlabel('Matrix dimension')
    ax.set_ylabel('Runtime (s)')
    ax.set_title(f'Thread Blocks: {TB}')
    ax.legend()

plt.savefig('runtimes_matrix_vector_ex4_multiple_subplots.png')
#====================================================================================

T = 256    # Number of threads
TB_values = [2,4,8,16,32,64,128,256]     # Number of thread blocks
runtimes = [[] for _ in range(len(TB_values))]   # List of runtimes for each TB value
shapes = [2,4,8,16,32,64,128,256]
avg_error = 0

for i, TB in enumerate(TB_values):
    for n in shapes:
        A = torch.rand(n, n).cuda()   # Create random n x n matrix 
        B = torch.rand(n, n).cuda()      # Create random n-dim vector
        error = 0
        start = time.time()
        for j in range(100): 
            C = matmul_cuda.parallelMatMul(A, B, T, TB)    # Matrix multiplication
            error += torch.mean(A@B - C)
        avg_error += error/100
        end = time.time()
        runtimes[i].append(end - start)
    
fig, axes = plt.subplots(nrows=2, ncols=4, figsize=(18, 6))
plt.subplots_adjust(left=0.1, bottom=0.1, right=0.9, top=0.9, wspace=0.4, hspace=0.4)
# Flatten axes array for easy iteration
axes_flat = np.array(axes).flatten()

for i, (TB, ax) in enumerate(zip(TB_values, axes_flat)):
    ax.plot(shapes, runtimes[i], label=f'TB={TB}')
    ax.set_xlabel('Matrix dimension')
    ax.set_ylabel('Average Error')
    ax.set_title(f'Thread Blocks: {TB}')
    ax.legend()

plt.savefig('runtimes_matrix_matrix_ex4_multiple_subplots.png')
plt.show()

print("Average Error of Matrix Multiplication is: ", avg_error.item()/len(shapes))