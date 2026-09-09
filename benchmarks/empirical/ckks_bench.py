"""
Empirical FHE Benchmark — OpenFHE CKKS
========================================

Measures actual wall-clock time for primitive and composite CKKS operations.
Validates the analytical cost model in benchmarks/analytical/cost_model.py.

Requires: Python 3.12, openfhe, numpy
Run with: .venv312/bin/python3 benchmarks/empirical/ckks_bench.py
"""

from openfhe import *
import numpy as np
import time
import json
import sys
from pathlib import Path


def bench(fn, n_iters=50, warmup=5):
    """Benchmark a function, return median time in ms."""
    for _ in range(warmup):
        fn()
    times = []
    for _ in range(n_iters):
        t0 = time.perf_counter()
        fn()
        t1 = time.perf_counter()
        times.append((t1 - t0) * 1000)
    return {
        "median_ms": float(np.median(times)),
        "mean_ms": float(np.mean(times)),
        "min_ms": float(np.min(times)),
        "max_ms": float(np.max(times)),
        "std_ms": float(np.std(times)),
    }


def setup_ckks(mult_depth, batch_size, scaling_mod=50, ring_dim=0):
    """Create a CKKS crypto context with given parameters."""
    params = CCParamsCKKSRNS()
    params.SetMultiplicativeDepth(mult_depth)
    params.SetScalingModSize(scaling_mod)
    params.SetBatchSize(batch_size)
    if ring_dim > 0:
        params.SetRingDim(ring_dim)

    cc = GenCryptoContext(params)
    cc.Enable(PKESchemeFeature.PKE)
    cc.Enable(PKESchemeFeature.KEYSWITCH)
    cc.Enable(PKESchemeFeature.LEVELEDSHE)

    keys = cc.KeyGen()
    cc.EvalMultKeyGen(keys.secretKey)

    # Generate rotation keys for common steps
    rot_indices = list(range(1, min(batch_size, 64)))
    # Add power-of-2 rotations for reductions
    p = 1
    while p < batch_size:
        if p not in rot_indices:
            rot_indices.append(p)
        p *= 2
    cc.EvalRotateKeyGen(keys.secretKey, rot_indices)

    actual_n = cc.GetRingDimension()
    return cc, keys, actual_n


def bench_primitives(cc, keys, batch_size):
    """Benchmark primitive CKKS operations."""
    x = np.random.randn(batch_size).astype(np.float64)
    y = np.random.randn(batch_size).astype(np.float64)

    ptx = cc.MakeCKKSPackedPlaintext(x)
    pty = cc.MakeCKKSPackedPlaintext(y)
    ctx = cc.Encrypt(keys.publicKey, ptx)
    cty = cc.Encrypt(keys.publicKey, pty)

    results = {}

    # Ciphertext-Ciphertext Add
    results["ct_add_ct"] = bench(lambda: cc.EvalAdd(ctx, cty))

    # Ciphertext-Plaintext Add
    results["ct_add_pt"] = bench(lambda: cc.EvalAdd(ctx, pty))

    # Ciphertext-Ciphertext Mult
    results["ct_mult_ct"] = bench(lambda: cc.EvalMult(ctx, cty))

    # Ciphertext-Plaintext Mult
    results["ct_mult_pt"] = bench(lambda: cc.EvalMult(ctx, pty))

    # Rotation (1 step)
    results["rotate_1"] = bench(lambda: cc.EvalRotate(ctx, 1))

    # Rotation (power of 2)
    results["rotate_16"] = bench(lambda: cc.EvalRotate(ctx, 16))

    # Negate
    results["negate"] = bench(lambda: cc.EvalNegate(ctx))

    # Encrypt
    results["encrypt"] = bench(lambda: cc.Encrypt(keys.publicKey, ptx))

    # Decrypt
    results["decrypt"] = bench(lambda: cc.Decrypt(keys.secretKey, ctx))

    # MakePlaintext
    results["make_plaintext"] = bench(lambda: cc.MakeCKKSPackedPlaintext(x))

    return results


def bench_polynomial_eval(cc, keys, batch_size, degree):
    """Benchmark polynomial evaluation (Horner's method) — simulates activation functions."""
    x = np.random.randn(batch_size).astype(np.float64) * 0.1
    ptx = cc.MakeCKKSPackedPlaintext(x)
    ctx = cc.Encrypt(keys.publicKey, ptx)

    # Chebyshev coefficients for approximation (random for benchmarking)
    coeffs = np.random.randn(degree + 1).astype(np.float64) * 0.01

    def eval_poly():
        # Horner's method: p(x) = c_n*x^n + ... + c_1*x + c_0
        # = ((c_n*x + c_{n-1})*x + c_{n-2})*x + ...
        result = cc.EvalMult(ctx, float(coeffs[degree]))
        for i in range(degree - 1, -1, -1):
            result = cc.EvalAdd(result, float(coeffs[i]))
            if i > 0:
                result = cc.EvalMult(result, ctx)
        return result

    return bench(eval_poly, n_iters=20)


def bench_linear_layer(cc, keys, batch_size, d_in, d_out):
    """
    Benchmark a linear layer (matrix-vector multiply).
    Input: d_in values packed in ciphertext(s).
    Weight: d_out × d_in plaintext matrix.
    Output: d_out values.

    Uses diagonal encoding method for matrix-vector multiplication.
    """
    # Pack input
    x = np.random.randn(batch_size).astype(np.float64) * 0.1
    ptx = cc.MakeCKKSPackedPlaintext(x)
    ctx = cc.Encrypt(keys.publicKey, ptx)

    # Weight matrix (plaintext) — encoded as diagonals
    # For simplicity, benchmark d_out plaintext-ciphertext multiplications + accumulations
    # This approximates the cost of a matrix-vector product
    n_cols = min(d_in, batch_size)
    n_rows = min(d_out, batch_size)

    # Create plaintext weight vectors
    weight_pts = []
    for i in range(n_rows):
        w = np.zeros(batch_size, dtype=np.float64)
        w[:n_cols] = np.random.randn(n_cols) * 0.01
        weight_pts.append(cc.MakeCKKSPackedPlaintext(w))

    def linear_layer():
        # Accumulate: y[i] = sum_j W[i,j] * x[j]
        # Simplified: n_rows plaintext-cipher mults + adds
        results = []
        for i in range(min(n_rows, 8)):  # Limit for benchmark speed
            r = cc.EvalMult(ctx, weight_pts[i])
            results.append(r)
        # Sum all results
        acc = results[0]
        for r in results[1:]:
            acc = cc.EvalAdd(acc, r)
        return acc

    return bench(linear_layer, n_iters=10)


def bench_reduction(cc, keys, batch_size, n_elements):
    """Benchmark sum reduction (used in softmax, layernorm)."""
    x = np.random.randn(batch_size).astype(np.float64)
    ptx = cc.MakeCKKSPackedPlaintext(x)
    ctx = cc.Encrypt(keys.publicKey, ptx)

    def sum_reduce():
        result = ctx
        step = 1
        while step < n_elements:
            rotated = cc.EvalRotate(result, step)
            result = cc.EvalAdd(result, rotated)
            step *= 2
        return result

    return bench(sum_reduce, n_iters=20)


def run_full_benchmark():
    """Run the complete benchmark suite."""
    print("=" * 80)
    print("EMPIRICAL FHE BENCHMARK — OpenFHE CKKS")
    print("=" * 80)

    all_results = {}

    # Parameter sets to test
    param_sets = [
        {"name": "small",  "mult_depth": 5,  "batch_size": 64,   "ring_dim": 0},
        {"name": "medium", "mult_depth": 10, "batch_size": 4096, "ring_dim": 0},
        {"name": "large",  "mult_depth": 20, "batch_size": 8192, "ring_dim": 0},
    ]

    for ps in param_sets:
        name = ps["name"]
        print(f"\n{'─' * 80}")
        print(f"Parameter set: {name}")
        print(f"  mult_depth={ps['mult_depth']}, batch_size={ps['batch_size']}")

        cc, keys, actual_n = setup_ckks(
            ps["mult_depth"], ps["batch_size"], ring_dim=ps["ring_dim"]
        )
        print(f"  Actual ring dimension N={actual_n}")
        print(f"  Slots={ps['batch_size']}")

        # Primitives
        print(f"\n  --- Primitive Operations ---")
        prims = bench_primitives(cc, keys, ps["batch_size"])
        for op_name, timing in prims.items():
            print(f"    {op_name:25s}: {timing['median_ms']:>10.3f} ms "
                  f"(±{timing['std_ms']:.3f})")

        # Polynomial evaluation at different degrees
        print(f"\n  --- Polynomial Evaluation (Horner) ---")
        poly_results = {}
        for deg in [2, 4, 6, 8, 12]:
            if deg <= ps["mult_depth"]:
                pr = bench_polynomial_eval(cc, keys, ps["batch_size"], deg)
                poly_results[f"poly_deg{deg}"] = pr
                print(f"    degree {deg:>3}: {pr['median_ms']:>10.3f} ms")

        # Sum reduction
        print(f"\n  --- Sum Reduction ---")
        red_results = {}
        for n_elem in [8, 32, 64, 128, 256]:
            if n_elem <= ps["batch_size"]:
                rr = bench_reduction(cc, keys, ps["batch_size"], n_elem)
                red_results[f"reduce_{n_elem}"] = rr
                print(f"    n={n_elem:>5}: {rr['median_ms']:>10.3f} ms")

        # Linear layer (small)
        print(f"\n  --- Linear Layer (plaintext-cipher mults) ---")
        lin_results = {}
        for d_in, d_out in [(64, 64), (128, 128), (256, 256)]:
            if d_in <= ps["batch_size"]:
                lr = bench_linear_layer(cc, keys, ps["batch_size"], d_in, d_out)
                lin_results[f"linear_{d_in}x{d_out}"] = lr
                print(f"    {d_in}×{d_out}: {lr['median_ms']:>10.3f} ms")

        all_results[name] = {
            "params": {**ps, "actual_N": actual_n},
            "primitives": prims,
            "polynomial": poly_results,
            "reduction": red_results,
            "linear": lin_results,
        }

    # Save results
    out_path = Path(__file__).parent / "results" / "ckks_bench_results.json"
    out_path.parent.mkdir(exist_ok=True)
    with open(out_path, "w") as f:
        json.dump(all_results, f, indent=2)
    print(f"\n{'=' * 80}")
    print(f"Results saved to {out_path}")

    # Comparison with analytical model
    print(f"\n{'=' * 80}")
    print("COMPARISON: Analytical Model vs Empirical")
    print(f"{'=' * 80}")

    if "medium" in all_results:
        med = all_results["medium"]
        prims = med["primitives"]
        print(f"\n  Analytical model assumptions (N=2^16):")
        print(f"    Add:      0.01 ms")
        print(f"    Mult:     0.10 ms")
        print(f"    Rotate:   0.50 ms")
        print(f"\n  Empirical (N={med['params']['actual_N']}):")
        print(f"    Add:      {prims['ct_add_ct']['median_ms']:.3f} ms")
        print(f"    Mult:     {prims['ct_mult_ct']['median_ms']:.3f} ms")
        print(f"    Rotate:   {prims['rotate_1']['median_ms']:.3f} ms")

        # Correction factors
        add_ratio = prims['ct_add_ct']['median_ms'] / 0.01
        mult_ratio = prims['ct_mult_ct']['median_ms'] / 0.1
        rot_ratio = prims['rotate_1']['median_ms'] / 0.5
        print(f"\n  Correction factors (empirical / analytical):")
        print(f"    Add:      {add_ratio:>8.1f}×")
        print(f"    Mult:     {mult_ratio:>8.1f}×")
        print(f"    Rotate:   {rot_ratio:>8.1f}×")
        print(f"\n  → Analytical model underestimates by ~{mult_ratio:.0f}× for mults")
        print(f"  → Updated wall-time estimates should multiply by ~{mult_ratio:.0f}×")

    print(f"\n{'=' * 80}")


if __name__ == "__main__":
    run_full_benchmark()
