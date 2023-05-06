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
plt.xlabel('Matrix dimension')
plt.ylabel('Runtime (s)')
plt.savefig('runtimes_matrix_vector.png')

# matrix-matrix multiplication
plt.figure()
avg_error = 0
runtimes = []
shapes = [2,4,8,16,32,64,128,256]

for n in shapes:
    A = torch.ones((n, n)).cuda()   # Create random n x n matrix 
    B = torch.ones((n, n)).cuda()      # Create random n-dim vector
    B[0][1] = 100
    error = 0
    start = time.time()
    for i in range(100): 
        C = matmul_cuda.parallelMatMul(A, B, T, TB)    # Matrix multiplication
        error += torch.mean(A.matmul(B) - C)
    avg_error += error/100
    end = time.time()
    runtimes.append(end - start)
    
print("Average Error of Matrix Multiplication is: ", avg_error.item()/len(shapes))
plt.plot(shapes, runtimes)
plt.xlabel('Matrix dimension')
plt.ylabel('Runtime (s)')
plt.savefig('runtimes_matrix_matrix.png')

# Exercise 4

# T = 256    # Number of threads
# TB_values = [2,4,8,16,32,64,128,256]     # Number of thread blocks
# runtimes = [[] for _ in range(len(TB_values))]   # List of runtimes for each TB value
# shapes = [2,4,8,16,32,64,128,256]

# for i, TB in enumerate(TB_values):
#     for n in shapes:
#         A = torch.rand(n, n).cuda()   # Create random n x n matrix 
#         B = torch.rand(n).cuda()      # Create random n-dim vector

#         start = time.time()
#         for i in range(100): 
#             C = matmul_cuda.parallelMatMul(A, B, T, TB)    # Matrix multiplication
#         end = time.time()
#         runtimes[i].append(end - start)
    
# for i, TB in enumerate(TB_values):
#     plt.plot(shapes, runtimes[i], label=f'TB={TB}')
# plt.xlabel('Matrix dimension')
# plt.ylabel('Runtime (s)')
# plt.legend()
# plt.savefig('runtimes_matrix_vector_ex4.png')
# #====================================================================================

# T = 256    # Number of threads
# TB_values = [2,4,8,16,32,64,128,256]     # Number of thread blocks
# runtimes = [[] for _ in range(len(TB_values))]   # List of runtimes for each TB value
# shapes = [2,4,8,16,32,64,128,256]
# avg_error = 0

# for i, TB in enumerate(TB_values):
#     for n in shapes:
#         A = torch.rand(n, n).cuda()   # Create random n x n matrix 
#         B = torch.rand(n, n).cuda()      # Create random n-dim vector
#         error = 0
#         start = time.time()
#         for i in range(100): 
#             C = matmul_cuda.parallelMatMul(A, B, T, TB)    # Matrix multiplication
#             error += torch.mean(A@B - C)
#         avg_error += error/100
#         end = time.time()
#         runtimes[i].append(end - start)
    
# for i, TB in enumerate(TB_values):
#     plt.plot(shapes, runtimes[i], label=f'TB={TB}')
# plt.xlabel('Matrix dimension')
# plt.ylabel('Runtime (s)')
# plt.legend()
# plt.savefig('runtimes_matrix_matrix_ex4.png')

# print("Average Error of Matrix Multiplication is: ", avg_error.item()/len(shapes))