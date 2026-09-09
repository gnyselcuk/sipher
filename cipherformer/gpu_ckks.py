"""
GPU-Accelerated CKKS Engine
=============================
CKKS-like homomorphic encryption operations on GPU via PyTorch.

Computation pattern matches real CKKS:
  - Ciphertexts are polynomial pairs (c0, c1) of degree N
  - Multiplication uses NTT (O(N log N))
  - Rotation uses automorphism + key switching
  - All operations batched for GPU parallelism

NOT cryptographically secure — for benchmarking and development only.
"""

import torch
import math
from dataclasses import dataclass
from typing import List, Optional


@dataclass
class CKKSParams:
    N: int = 65536          # ring dimension
    n_slots: int = 0        # SIMD slots (N/2)
    scale: float = 2**40    # scaling factor
    sigma: float = 3.2      # noise std dev
    max_level: int = 10     # max multiplicative depth

    def __post_init__(self):
        if self.n_slots == 0:
            self.n_slots = self.N // 2


class Ciphertext:
    """CKKS ciphertext: pair of polynomials (c0, c1) in NTT form."""
    __slots__ = ['c0', 'c1', 'level', 'scale']

    def __init__(self, c0: torch.Tensor, c1: torch.Tensor,
                 level: int, scale: float):
        self.c0 = c0  # [..., N] complex tensor
        self.c1 = c1  # [..., N] complex tensor
        self.level = level
        self.scale = scale

    @property
    def device(self):
        return self.c0.device


class GPUCKKS:
    """GPU-accelerated CKKS engine."""

    def __init__(self, params: CKKSParams, device='cuda'):
        self.params = params
        self.device = device
        self.N = params.N
        self.n_slots = params.n_slots

        # Generate secret key (random ternary polynomial)
        self.sk = self._gen_secret_key()
        # Generate rotation keys (simplified: just store shifted keys)
        self._rot_keys = {}

    def _gen_secret_key(self) -> torch.Tensor:
        """Generate ternary secret key in NTT form."""
        sk = torch.randint(-1, 2, (self.N,), device=self.device).float()
        return torch.fft.fft(sk.to(torch.complex64))

    def _gen_error(self, shape=()) -> torch.Tensor:
        """Generate Gaussian error polynomial in NTT form."""
        e = torch.randn(*shape, self.N, device=self.device) * self.params.sigma
        return torch.fft.fft(e.to(torch.complex64))

    def _ntt(self, x: torch.Tensor) -> torch.Tensor:
        """Forward NTT (via FFT)."""
        return torch.fft.fft(x.to(torch.complex64), dim=-1)

    def _intt(self, x: torch.Tensor) -> torch.Tensor:
        """Inverse NTT (via IFFT)."""
        return torch.fft.ifft(x, dim=-1).real

    # ── Encoding ──

    def encode(self, values: torch.Tensor) -> torch.Tensor:
        """
        Encode real values into polynomial (coefficient encoding).
        values: [..., n_values] real tensor
        Returns: [..., N] complex tensor (NTT form)
        """
        batch_shape = values.shape[:-1]
        n = values.shape[-1]
        assert n <= self.N

        # Place scaled values in polynomial coefficients
        poly = torch.zeros(*batch_shape, self.N,
                           device=self.device, dtype=torch.float64)
        poly[..., :n] = values.double() * self.params.scale

        # Convert to NTT (evaluation) form
        return torch.fft.fft(poly.to(torch.complex128), dim=-1).to(torch.complex64)

    def decode(self, poly: torch.Tensor, n_values: int) -> torch.Tensor:
        """
        Decode polynomial back to real values.
        poly: [..., N] complex tensor (NTT form)
        Returns: [..., n_values] real tensor
        """
        # Inverse NTT to get coefficient form
        coeffs = torch.fft.ifft(poly.to(torch.complex128), dim=-1).real

        # Extract and descale
        return (coeffs[..., :n_values] / self.params.scale).float()

    # ── Encryption / Decryption ──

    def encrypt(self, encoded: torch.Tensor,
                level: Optional[int] = None) -> Ciphertext:
        """
        Encrypt encoded polynomial.
        encoded: [..., N] complex tensor
        Returns: Ciphertext
        """
        if level is None:
            level = self.params.max_level

        # c1 = random polynomial
        c1 = self._gen_error(encoded.shape[:-1])

        # c0 = -c1 * sk + encoded + error
        c0 = -c1 * self.sk + encoded + self._gen_error(encoded.shape[:-1])

        return Ciphertext(c0, c1, level, self.params.scale)

    def decrypt(self, ct: Ciphertext, n_values: int) -> torch.Tensor:
        """
        Decrypt ciphertext to real values.
        Returns: [..., n_values] real tensor
        """
        # m = c0 + c1 * sk
        poly = ct.c0 + ct.c1 * self.sk
        return self.decode(poly, n_values)

    def encrypt_values(self, values: torch.Tensor,
                       level: Optional[int] = None) -> Ciphertext:
        """Convenience: encode + encrypt."""
        encoded = self.encode(values)
        return self.encrypt(encoded, level)

    def decrypt_values(self, ct: Ciphertext, n_values: int) -> torch.Tensor:
        """Convenience: decrypt + decode."""
        return self.decrypt(ct, n_values)

    # ── Homomorphic Operations ──

    def add(self, ct1: Ciphertext, ct2: Ciphertext) -> Ciphertext:
        """Homomorphic addition."""
        return Ciphertext(
            ct1.c0 + ct2.c0,
            ct1.c1 + ct2.c1,
            min(ct1.level, ct2.level),
            ct1.scale
        )

    def add_plain(self, ct: Ciphertext, plain: torch.Tensor) -> Ciphertext:
        """Add plaintext to ciphertext."""
        encoded = self.encode(plain)
        return Ciphertext(
            ct.c0 + encoded,
            ct.c1.clone(),
            ct.level,
            ct.scale
        )

    def mult(self, ct1: Ciphertext, ct2: Ciphertext) -> Ciphertext:
        """
        Homomorphic multiplication (ct × ct).
        Tensor product + relinearization (simplified).
        """
        # Tensor product
        d0 = ct1.c0 * ct2.c0
        d1 = ct1.c0 * ct2.c1 + ct1.c1 * ct2.c0
        d2 = ct1.c1 * ct2.c1

        # Relinearization (simplified: absorb d2 into c0 using sk²)
        # In real CKKS, this uses a relinearization key
        c0 = d0 + d2 * self.sk * self.sk  # simplified relin
        c1 = d1

        # Add small noise from relin
        c0 = c0 + self._gen_error(c0.shape[:-1]) * 0.1

        new_scale = ct1.scale * ct2.scale / self.params.scale
        return Ciphertext(c0, c1, min(ct1.level, ct2.level) - 1, new_scale)

    def mult_plain(self, ct: Ciphertext, plain: torch.Tensor) -> Ciphertext:
        """
        Multiply ciphertext by plaintext (ct × pt).
        Much cheaper than ct × ct.
        """
        encoded = self.encode(plain)
        return Ciphertext(
            ct.c0 * encoded,
            ct.c1 * encoded,
            ct.level - 1,
            ct.scale  # scale stays same for ct-pt mult
        )

    def mult_scalar(self, ct: Ciphertext, scalar: float) -> Ciphertext:
        """Multiply ciphertext by scalar."""
        return Ciphertext(
            ct.c0 * scalar,
            ct.c1 * scalar,
            ct.level,
            ct.scale
        )

    def rotate(self, ct: Ciphertext, steps: int) -> Ciphertext:
        """
        Rotate ciphertext slots by steps.
        In real CKKS: automorphism + key switching.
        Computational cost: ~3 NTTs.
        """
        # Simulate rotation by shifting the underlying values
        # In NTT domain, this is an automorphism (index permutation)
        # For benchmarking, we do the equivalent computation

        # Inverse NTT → shift → forward NTT (3 NTT operations)
        c0_coeff = torch.fft.ifft(ct.c0, dim=-1)
        c1_coeff = torch.fft.ifft(ct.c1, dim=-1)

        # Cyclic shift
        c0_shifted = torch.roll(c0_coeff, shifts=steps, dims=-1)
        c1_shifted = torch.roll(c1_coeff, shifts=steps, dims=-1)

        # Back to NTT domain
        c0_new = torch.fft.fft(c0_shifted, dim=-1)
        c1_new = torch.fft.fft(c1_shifted, dim=-1)

        # Key switching noise
        c0_new = c0_new + self._gen_error(c0_new.shape[:-1]) * 0.01

        return Ciphertext(c0_new, c1_new, ct.level, ct.scale)

    def negate(self, ct: Ciphertext) -> Ciphertext:
        """Negate ciphertext."""
        return Ciphertext(-ct.c0, -ct.c1, ct.level, ct.scale)

    # ── Batched Operations ──

    def batch_encrypt(self, values_list: List[torch.Tensor]) -> List[Ciphertext]:
        """Encrypt multiple value vectors as a batch."""
        # Stack into batch tensor for GPU parallelism
        batch = torch.stack(values_list)
        encoded = self.encode(batch)
        ct = self.encrypt(encoded)
        # Split back into individual ciphertexts
        return [Ciphertext(ct.c0[i], ct.c1[i], ct.level, ct.scale)
                for i in range(len(values_list))]

    def batch_mult_plain(self, cts: List[Ciphertext],
                         plains: List[torch.Tensor]) -> List[Ciphertext]:
        """Batch ct-pt multiplication."""
        c0_batch = torch.stack([ct.c0 for ct in cts])
        c1_batch = torch.stack([ct.c1 for ct in cts])
        plain_batch = torch.stack(plains)
        encoded = self.encode(plain_batch)

        new_c0 = c0_batch * encoded
        new_c1 = c1_batch * encoded

        return [Ciphertext(new_c0[i], new_c1[i], cts[i].level - 1, cts[i].scale)
                for i in range(len(cts))]


# ── Quick Test ──

def test_gpu_ckks():
    """Verify GPU CKKS correctness."""
    print("GPU CKKS Engine — Correctness Test")
    print("=" * 60)

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    params = CKKSParams(N=4096, max_level=10)
    engine = GPUCKKS(params, device=device)

    # Test 1: Encrypt → Decrypt
    x = torch.randn(16)
    ct = engine.encrypt_values(x)
    x_dec = engine.decrypt_values(ct, 16).cpu()
    err1 = (x - x_dec).abs().max().item()
    print(f"  Encrypt/Decrypt error: {err1:.6f} {'✅' if err1 < 0.01 else '❌'}")

    # Test 2: Addition
    y = torch.randn(16)
    ct_y = engine.encrypt_values(y)
    ct_sum = engine.add(ct, ct_y)
    sum_dec = engine.decrypt_values(ct_sum, 16).cpu()
    err2 = (x + y - sum_dec).abs().max().item()
    print(f"  Addition error:        {err2:.6f} {'✅' if err2 < 0.01 else '❌'}")

    # Test 3: Scalar multiplication
    ct_scaled = engine.mult_scalar(ct, 2.0)
    scaled_dec = engine.decrypt_values(ct_scaled, 16).cpu()
    err3 = (x * 2.0 - scaled_dec).abs().max().item()
    print(f"  Scalar mult error:     {err3:.6f} {'✅' if err3 < 0.1 else '❌'}")

    # Test 4: Rotation
    ct_rot = engine.rotate(ct, 1)
    rot_dec = engine.decrypt_values(ct_rot, 16).cpu()
    print(f"  Rotation:              computed (simplified)")

    # Test 5: Batch operations
    batch_vals = [torch.randn(16) for _ in range(8)]
    batch_cts = engine.batch_encrypt(batch_vals)
    batch_dec = [engine.decrypt_values(ct, 16).cpu() for ct in batch_cts]
    errs = [(v - d).abs().max().item() for v, d in zip(batch_vals, batch_dec)]
    print(f"  Batch encrypt/decrypt: max_err={max(errs):.6f} "
          f"{'✅' if max(errs) < 0.01 else '❌'}")

    print("=" * 60)


if __name__ == "__main__":
    test_gpu_ckks()
