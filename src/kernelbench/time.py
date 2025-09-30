import torch
from time import perf_counter

steps = 10
times = []
sync_times = []

# This won't time the CUDA kernel - only the launch overhead
for _ in range(steps):
  start_time = perf_counter()

  run_kernel() # Some kernel

  end_time = perf_counter()
  times.append(end_time - start_time)

# This measures what we actually care about
for _ in range(steps):
  start_time = perf_counter()

  run_kernel()
  torch.cuda.synchronize()

  end_time = perf_counter()
  sync_times.append(end_time - start_time)