# The use of PyTorch in Triton programs is not allowed for the purposes of fair benchmarking.
import triton
import triton.language as tl
import torch


def cudaEmpty(num_elements:int):
    import ctypes
    cudart = ctypes.CDLL("libcudart.so")
    cudart.cudaMalloc.argtypes = [ctypes.POINTER(ctypes.c_void_p), ctypes.c_size_t]
    cudart.cudaMalloc.restype = ctypes.c_int
    ptr = ctypes.c_void_p()
    err = cudart.cudaMalloc(ctypes.byref(ptr), num_elements*4)
    if err != 0:
        raise RuntimeError(f"cudaMalloc failed, code {err}")
    return ptr.value


@triton.jit
def softmax_kernel(
    input_ptr, M, N, BLOCK_SIZE:tl.constexpr
):
    input_ptr = input_ptr.to(tl.pointer_type(tl.float32))
    
    pid = tl.program_id(0)

    off = tl.arange(0,BLOCK_SIZE)
    mask = off <= pid
    block_mask = off < N

    x = tl.load(input_ptr + pid * N + off,mask=mask,other=-float("inf"))
    _max = tl.max(x)
    x = x - _max
    
    x = tl.exp(x)

    sum_exp_x = tl.sum(x)

    x = x / sum_exp_x

    tl.store(input_ptr + pid * N + off,x,mask=block_mask)



@triton.jit
def qkt_kernal(Q,K,qkt,M,N,d,scale):
    Q = Q.to(tl.pointer_type(tl.float32)) # [Mxd]
    K = K.to(tl.pointer_type(tl.float32)) # [Mxd]
    qkt = qkt.to(tl.pointer_type(tl.float32))

    row_id = tl.program_id(0)
    col_id = tl.program_id(1)

    value = 0.0
    for i in range(d):
        a_val = tl.load(Q + row_id * d + i)
        b_val = tl.load(K + col_id * d + i)
        value += a_val * b_val
    value = value * scale

    tl.store(qkt + row_id * N + col_id, value)


@triton.jit
def matmul(a,b,c,M,N,d): 
    a = a.to(tl.pointer_type(tl.float32)) # Mxd
    b = b.to(tl.pointer_type(tl.float32)) # dxN
    c = c.to(tl.pointer_type(tl.float32))
    
    row_id = tl.program_id(0)
    col_id = tl.program_id(1)
    c_val = 0.0
    for i in range(d):
        a_val = tl.load(a+ row_id * d + i)
        b_val = tl.load(b+ i * N + col_id)
        c_val += a_val * b_val
    tl.store(c + row_id * N + col_id, c_val)


def next_power_of_2(n: int) -> int:
    if n <= 1:
        return 1
    n -= 1
    n |= n >> 1
    n |= n >> 2
    n |= n >> 4
    n |= n >> 8
    n |= n >> 16
    n |= n >> 32
    return n + 1


def solve(Q: torch.Tensor, K: torch.Tensor, V: torch.Tensor, output: torch.Tensor, M: int, d: int):

    qkt = cudaEmpty(M*M) # [M,N]
    scale = 1.0 / (d ** 0.5)
    qkt_kernal[(M,M)](Q,K,qkt,M,M,d,scale)
    BLOCK_SIZE = next_power_of_2(M)
    softmax_kernel[(M,)](qkt,M,M,BLOCK_SIZE)
    matmul[(M,d)](qkt,V,output,M,d,M)



def run_tests(challenge):
    print(f"Running example test.")
    example = challenge.generate_example_test()
    solve(**example)
    print("Q:", example["Q"].cpu().numpy())
    print("K:", example["K"].cpu().numpy())
    print("V:", example["V"].cpu().numpy())
    print("Output:", example["output"].cpu().numpy())

    print("\nRunning functional tests...")
    for i, test in enumerate(challenge.generate_functional_test() + [challenge.generate_performance_test()]):
        solve(**test)
        # 使用 reference_impl 检查结果
        expected = torch.empty_like(test["output"])
        challenge.reference_impl(test["Q"], test["K"], test["V"], expected, test["M"], test["d"])
        if torch.allclose(test["output"], expected, atol=challenge.atol, rtol=challenge.rtol):
            print(f"Test {i} PASSED")
            # print("Expected:", expected.cpu().numpy())
        else:
            print(f"Test {i} FAILED")
            print("Expected:", expected.cpu().numpy())
            print("Got:", test["output"].cpu().numpy())
if __name__ == "__main__":
    from challenge_local import Challenge  # 替换为你 Challenge 类所在模块
    challenge = Challenge()
    run_tests(challenge)