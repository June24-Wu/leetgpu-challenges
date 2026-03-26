import ctypes
from typing import Any, List, Dict
import torch
from core.challenge_base import ChallengeBase

class Challenge(ChallengeBase):
    def __init__(self):
        super().__init__(
            name="Rotary Position Embedding",
            atol=1e-05,
            rtol=1e-05,
            num_gpus=1,
            access_tier="free"
        )
        
    def reference_impl(self, input: torch.Tensor, output: torch.Tensor, base: float, 
                      S: int, H: int):
        
        assert input.shape == output.shape == (S,H)
        assert input.dtype == output.dtype
        assert input.device == output.device
        assert H % 2 == 0, "Hidden dimension H must be even for Rotary Position Embedding"

        
        pos = torch.arange(S, dtype=torch.float32,device=input.device).unsqueeze(1)  # (S, 1)

        i = torch.arange(0, H // 2, dtype=torch.float32,device=input.device)  # (H/2,)
        freq = base ** (-2 * i / H)                       # (H/2,)

        theta = pos * freq.unsqueeze(0)  # (S, H/2)

        cos_theta = torch.cos(theta)  # (S, H/2)
        sin_theta = torch.sin(theta)  # (S, H/2)

        x1, x2 = input[..., 0::2], input[..., 1::2]  # (S, H/2), (S, H/2)

        x_rotated = torch.stack([
            x1 * cos_theta - x2 * sin_theta,
            x1 * sin_theta + x2 * cos_theta
        ], dim=-1)  # (S, H/2, 2)

        output.copy_(x_rotated.flatten(-2))
        
    def get_solve_signature(self) -> Dict[str, Any]:
        return {
            "input": ctypes.POINTER(ctypes.c_float),
            "base": ctypes.c_float,
            "output": ctypes.POINTER(ctypes.c_float),
            "S": ctypes.c_int,
            "H": ctypes.c_int,
        }

    def generate_example_test(self) -> Dict[str, Any]:
        dtype = torch.float32
        S = 3
        H = 2
        input = torch.tensor([
            [1.0, 0.0],
            [0.0, 1.0],
            [1.0, 1.0]
        ], device="cuda", dtype=dtype)
        
        base = 10000.0
        output = torch.empty([S,H], device="cuda", dtype=dtype)
        return {
            "input": input,
            "base": base,
            "output": output,
            "H": H,
            "S": S
        }

    def generate_functional_test(self) -> List[Dict[str, Any]]:
        dtype = torch.float32
        tests = []

        # basic small test
        tests.append({
            "input": torch.tensor([[1.0, 0.0], [0.0, 1.0], [1.0, 1.0]], device="cuda", dtype=dtype),
            "base": 10000.0,
            "output": torch.empty([3, 2], device="cuda", dtype=dtype),
            "S": 3,
            "H": 2
        })
        
        # single sequence, dim
        tests.append({
            "input": torch.tensor([[5.0,3.0]], device="cuda", dtype=dtype),
            "base": 10000.0,
            "output": torch.empty([1, 2], device="cuda", dtype=dtype),
            "S": 1,
            "H": 2
        })

        # all zeros
        S, H = 4, 4
        tests.append({
            "input": torch.zeros([S, H], device="cuda", dtype=dtype),
            "base": 10000.0,
            "output": torch.empty([S, H], device="cuda", dtype=dtype),
            "S": S,
            "H": H
        })
        
        # negative numbers
        S, H = 3, 4
        tests.append({
            "input": torch.tensor([[-1.0, -2.0, -3.0, -4.0],
                                   [-5.0, -6.0, -7.0, -8.0],
                                   [-9.0, -10.0, -11.0, -12.0]], device="cuda", dtype=dtype),
            "base": 10000.0,
            "output": torch.empty([S, H], device="cuda", dtype=dtype),
            "S": S,
            "H": H
        })

        # larger sequence and hidden dimension
        S, H = 6, 8
        tests.append({
            "input": torch.empty([S, H], device="cuda", dtype=dtype).uniform_(-10.0, 10.0),
            "base": 10000.0,
            "output": torch.empty([S, H], device="cuda", dtype=dtype),
            "S": S,
            "H": H
        })
        
        # very large sequence
        S, H = 1024, 16
        tests.append({
            "input": torch.empty([S, H], device="cuda", dtype=dtype).uniform_(-100.0, 100.0),
            "base": 1.0,
            "output": torch.empty([S, H], device="cuda", dtype=dtype),
            "S": S,
            "H": H
        })

    
        # different base value
        S, H = 3, 2
        tests.append({
            "input": torch.tensor([[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]], device="cuda", dtype=dtype),
            "base": 5000.0,
            "output": torch.empty([S, H], device="cuda", dtype=dtype),
            "S": S,
            "H": H
        })

        return tests

    def generate_performance_test(self) -> Dict[str, Any]:
        dtype = torch.float32
        S, H = 10000, 64
        return {
            "input": torch.empty([S, H], device="cuda", dtype=dtype).uniform_(-100.0, 100.0),
            "base": 1.0,
            "output": torch.empty([S, H], device="cuda", dtype=dtype),
            "S": S,
            "H": H
        }