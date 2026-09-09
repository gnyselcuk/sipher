# Sipher v1 — FHE-Native Language Modeling (Research Preview)

**Sipher** is a language-model architecture designed from the ground up for **Fully Homomorphic Encryption (CKKS)**, not a patched Transformer.

> Research preview · not a production chatbot · results under active iteration

---

## Headline results (v1)

| Result | Detail |
|--------|--------|
| **Architecture** | Causal hybrid: linear global + **local window softmax** + PolyFFN |
| **Working LM path** | **Local-only** (`gate_global = 0`) produces coherent Turkish continuations |
| **Best public-facing ckpt** | `local_only_ppl28`: ~254M params, 16K TR BPE, train PPL ≈ 28.8 |
| **Negative result** | Forced **additive linear global** attention often hurts generation |
| **FHE** | TenSEAL / OpenFHE / FIDESlib GPU path; block parity cos_sim &gt; 0.999 |
| **FHE + trained weights** | Export + GPU parity (top-5 match); full 20-layer infer ~**1 min/token** (research) |

**Takeaway:** Under FHE constraints, a **local-window poly network** is a viable LM path; full “linear global = free long range” did not hold for generation quality in our runs.

---

## What this repo is / is not

| Is | Is not |
|----|--------|
| Research code + tokenizer for Sipher v1 | Production encrypted chat SaaS |
| Scripts to train, generate (pad-free), FHE parity demos | Guaranteed SOTA Turkish LLM |
| Documented measured findings | Claim that all ops run non-interactively without client helpers |

Hybrid FHE protocol notes: local softmax and some KV state may be **client-side** in the reference implementation (see `docs/SIPHER_V1.md`).

---

## Quick start (plaintext)

```bash
# Python 3.11+ recommended
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt

# Softmax quality oracle (small, ~8GB GPU friendly)
python -u pretrain_softmax_oracle.py --from-scratch --epochs 3

# Sipher hybrid pretrain (local-only style flags — see docs)
python -u pretrain_hybrid.py --help
```

Generation must be **pad-free** (no left-pad with token `0`). Models trained on packed sequences without pad tokens break under fake left-padding, a known decoder-only pitfall.

Tokenizer (16K Turkish BPE): `data/sipher_tokenizer/`.

---

## Checkpoints & large assets

Weights and token caches are **not** in git (multi‑GB).

| Asset | Where |
|-------|--------|
| Tokenizer 16K | `data/sipher_tokenizer/` (in repo) |
| `local_only_ppl28` (254M TR ckpt) | [huggingface.co/guneys/sipher-local-only-ppl28](https://huggingface.co/guneys/sipher-local-only-ppl28) |
| Pretrain token shards | Build with scripts or HF datasets |

---

## Repository layout (minimal v1 surface)

```
pretrain_hybrid.py          # Main Sipher training
pretrain_softmax_oracle.py  # Softmax upper-bound oracle (local GPU)
e2e_sipher_fhe_*.py         # CKKS / TenSEAL parity demos
bsgs_matvec.py              # BSGS helpers
cpp/                        # OpenFHE + FIDESlib GPU engine (no weights/)
data/sipher_tokenizer/      # 16K BPE
docs/                       # Design notes + v1 release note
benchmarks/                 # Early FHE cost models
analysis/                   # Bottleneck writeups
```

Legacy experiments live under `archive/` (gitignored if present locally).

---

## FHE stack

- **Python demos:** TenSEAL (CKKS)
- **C++ / GPU:** OpenFHE + FIDESlib (`cpp/`, `cpp/gpu/`)
- **Measured:** single-block and multi-layer parity; trained-weight export path for local-only ckpt

See `docs/SIPHER_V1.md` for protocol scope (what is encrypted on server vs client).

---

## Citation

If you use this work, please cite the repository and (when available) the accompanying note/paper:

```bibtex
@software{sipher2026,
  title  = {Sipher: FHE-Native Language Modeling},
  year   = {2026},
  url    = {https://github.com/gnyselcuk/sipher},
  note   = {Research preview v1}
}
```

---

## Security

- Do **not** commit API keys. Use `.env` (gitignored); see `.env.example`.
- This software is for research. Encrypted inference latency and threat models are experimental.

---

## License

MIT: see [LICENSE](LICENSE).  
Third-party: OpenFHE, FIDESlib, TenSEAL, and training data sources retain their own licenses.

---

## Status

**Sipher v1 research preview:** core findings and code paths documented; training data scale, SFT, and FHE latency remain open engineering work. Details: [`docs/SIPHER_V1.md`](docs/SIPHER_V1.md).
