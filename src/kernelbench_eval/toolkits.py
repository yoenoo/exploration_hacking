import os
import sys
from contextlib import contextmanager


@contextmanager
def suppress_cuda_compilation():
  """Suppress CUDA compilation output while preserving Python exceptions"""
  # Save original file descriptors
  stdout_fd = os.dup(sys.stdout.fileno())
  stderr_fd = os.dup(sys.stderr.fileno())
  
  try:
    # Redirect to devnull
    with open(os.devnull, 'w') as devnull:
      os.dup2(devnull.fileno(), sys.stdout.fileno())
      os.dup2(devnull.fileno(), sys.stderr.fileno())
    
    yield
      
  finally:
    # Restore original descriptors
    os.dup2(stdout_fd, sys.stdout.fileno())
    os.dup2(stderr_fd, sys.stderr.fileno())
    os.close(stdout_fd)
    os.close(stderr_fd)