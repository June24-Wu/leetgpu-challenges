from gpu.host import DeviceContext
from gpu.id import block_dim, block_idx, thread_idx
from memory import UnsafePointer
from math import ceildiv

# input, output are device pointers
@export
def solve(input: UnsafePointer[Float32], output: UnsafePointer[Float32],
          base: Float32, S: Int32, N: Int32):
    pass
