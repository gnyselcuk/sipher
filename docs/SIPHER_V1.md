# Sipher v1 — Release Note (Research Preview)

**Date:** 2026-08  
**Tag target:** `v1.0.0-research-preview`

## 1. Claim

Sipher v1 demonstrates:

1. An **FHE-oriented** decoder stack (polynomial activations, factorized embeddings, hybrid attention designed for CKKS matvec).
2. That **local causal window attention** (softmax on a short window) can support coherent Turkish continuations after pretraining.
3. That **forced additive global linear attention** (fixed high `gate_global`) can degrade generation on the same checkpoint family.
4. A **working CKKS pipeline** (Python TenSEAL demos + C++/GPU OpenFHE–FIDESlib) with high plaintext–ciphertext agreement on block outputs / top-k tokens for the published local-only path.

It does **not** claim production chat quality, sub-second FHE tokens, or that linear attention matches full softmax Transformers at equal scale.

## 2. Architecture (v1)

```
Token emb (factorized) + learned positions
→ N × {
     HybridAttention:
       global: causal linear (optional; often gated to 0)
       local:  causal window softmax (w=32)
       out = g_global * global + g_local * local
     PolyFFN (degree-2)
     LayerNorm (train); FHE path may use client-side / approx
  }
→ tied / projected LM head
```

**Working recipe:** freeze or set `gate_global = 0`, train/eval with strong local gate; **pad-free** generation.

## 3. Key experiments (summary)

| Exp | Result |
|-----|--------|
| Non-causal / mixer leakage | Fake PPL ~1.0; invalid LM metric |
| Clean data + causal | Real PPL descent |
| Left-pad zeros at generate | Garbage text (train had no pads) |
| `gate_global=0.5` freeze | PPL OK; generation often worse |
| Inference `g=0, l=1` | Sentence structure appears |
| Local-only train (~688M tok, 16K) | PPL ≈ 28.8, usable TR continuations |
| Softmax oracle (~40M, 39M tok) | Confirms data can support sentences |
| FHE parity (local-only export) | cos_sim ≈ 0.9996, top-5 5/5 (layer 0) |
| FHE 20-layer infer | ~order 1 min/token (research hardware/stack) |

## 4. FHE protocol scope

Reference implementations may use a **hybrid** protocol:

- Server: CKKS matvecs, poly ct×ct, residuals  
- Client: e.g. window softmax and/or plaintext KV accumulation  

Always read the specific `e2e_sipher_fhe_*.py` / C++ flags before claiming “fully non-interactive FHE.”

## 5. Reproducing (minimal)

```bash
# Tokenizer present in repo
ls data/sipher_tokenizer/sipher_tokenizer.json

# Softmax oracle (no FHE)
python -u pretrain_softmax_oracle.py --from-scratch --epochs 3

# TenSEAL parity demos (small d)
python -u e2e_sipher_fhe_parity.py
python -u e2e_sipher_fhe_local_attn.py
```

Full 254M train + GPU FHE build: see `cpp/`, `cpp/gpu/`, and internal notes (OpenFHE/FIDESlib install required).

## 6. Known limitations

- Turkish BPE fragmentation (`hiss eler`, etc.)
- Long-range topic drift after a few sentences  
- SFT on small Q&A sets overfits easily (need early stop + more data)  
- FHE latency far from interactive chat  
- Bootstrap / depth budgeting still engineering work  

## 7. Assets outside git

Upload separately to Hugging Face:

- `local_only_ppl28.pt` (or safetensors export)
- Optional: tokenized shards, FHE weight dump (`cpp/weights_localonly`)

## 8. Suggested paper framing

1. Measurement: Transformers vs FHE cost (existing analysis/)  
2. Architecture: FHE-native layers + local window  
3. Finding: global linear gate behavior vs generation  
4. System: CKKS implementation + parity numbers  

Negative results (leakage metrics, pad eval bug, global gate harm) are first-class contributions.
