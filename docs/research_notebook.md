# Sipher Research Notebook

A distilled, honest log of the Sipher v1 research (July–September 2026), reorganized from
the internal dev log into a **tried → result → problem → fix → lesson** narrative.
Negative results are treated as first-class findings. Numbers are measured unless
explicitly labeled as projections.

The goal from day one: a language model whose entire forward pass is polynomial/matvec,
so it runs natively under CKKS homomorphic encryption: no expensive polynomial
approximations of softmax, GELU, or LayerNorm.

---

## Phase 1 — Architecture: the leakage that fooled us

### 1.1 The "too good" PPL (2026-07-29)
- **Tried:** A bidirectional `TokenMixer` (MLP mixing across the sequence) + non-causal
  attention, trained on 1.5B tokens of scraped data.
- **Result:** Train/valid **PPL 1.03, accuracy 99.7%**, looked like a breakthrough.
  Generation produced "suffix soup" (grammatical fragments, no sentences).
- **Problem:** PPL 1.03 is *suspiciously* perfect for a 256M model on noisy data. The
  mismatch between a near-zero loss and garbage generation is the tell.
- **Fix:** Traced it to `TokenMixer.token_up = nn.Linear(seq_len, H)`: it mixes across the
  **sequence** dimension after a transpose, so position *i* sees tokens *i+1…*. The model
  was simply **copying the next token**, a bidirectional leak, not language modeling.
  Removed the mixer (`--no-mixer`), made attention causal.
- **Result after fix:** Real PPL: 237 on dirty data, **45 on 12.5M clean tokens**.
- **Lesson:** A PPL that looks too good is a leakage red flag. Always validate with
  *generation*, never trust the loss alone. (This bug resurfaced later, see 8.1.)

### 1.2 Causal + PolyFFN learns real structure (2026-07-30)
- **Tried:** Causal local-window attention (w=32) + a degree-2 polynomial FFN (PolyFFN),
  no mixer, on 12.5M clean Turkish tokens.
- **Result:** PPL 45, partial sentences, finance vocabulary emerging.
- **Lesson:** The FHE-native architecture (linear-global + local-window + PolyFFN) *can*
  learn grammar. The bottleneck moved from architecture to **data**.

---

## Phase 2 — Data: quality first, then quantity

### 2.1 Clean beats dirty by a wide margin (2026-07-30/31)
- **Tried:** Same architecture on 1.5B dirty scraped tokens vs 12.5M hand-cleaned tokens.
- **Result:** Dirty → PPL 237 (word soup). Clean → PPL 45 (partial sentences).
- **Lesson:** Quality ≫ quantity at this scale. A small pristine corpus beats a huge noisy
  one (consistent with the Phi-1 / "textbooks are all you need" line of work).

### 2.2 The quantity wall (2026-07-31)
- **Problem:** 12.5M tokens exhausts quickly at 254M params: PPL plateaus at ~45.
- **Fix:** Built ingestion pipelines: clean OSCAR Turkish (1M rows, ~670M-token
  potential), 63 public-domain books, mevzuat.gov.tr statutes (64 laws), 112 BDDK/SPK/TCMB
  regulatory PDFs, Wikipedia-TR finance-filtered, a generated textbook, Wikisource.
  Assembled a ~688M-token path.
- **Lesson:** After quality is solved, quantity is the next bottleneck; ~700M tokens is the
  target range for coherent generation at this model size.

---

## Phase 3 — Tokenizer: 16K vs 32K Turkish BPE

### 3.1 A pure-Turkish 16K BPE (2026-07-31)
- **Tried:** Replaced the 32K multilingual BERT tokenizer with a 16K pure-Turkish BPE.
- **Result:** 16K learned **faster** than 32K BERT at equal epochs (ep3: PPL 712 vs 1238);
  embedding table halved; foreign-token generation became impossible.
- **Lesson:** A smaller, in-domain vocabulary helps a small model converge faster.

### 3.2 Does a bigger vocab fix fragmentation? No. (2026-08-05)
- **Tried:** Trained a 32K Sipher BPE to fix fragmentations like `"hiss eler"` (hisseler).
- **Result:** The fragmentation **persisted at 32K** too.
- **Lesson:** BPE fragmentation here is an inherent limit of subword modeling for an
  **agglutinative** language, not a vocabulary-size problem. Accepted as readable; the 32K
  variant was kept only as a negative-result artifact (and later trimmed from the repo).

---

## Phase 4 — The gate saga: our central negative result

The hybrid attention has two paths combined by learned gates:
`out = gate_global · (causal linear global) + gate_local · (local window softmax)`.
Almost everything we got wrong, we got wrong here.

### 4.1 The H100 run that "succeeded" and failed (2026-08-02)
- **Tried:** A 688M-token, 6-epoch run on a rented H100.
- **Result:** **PPL 4.3**, far past the 15–25 target. But generation was garbage
  ("sur sur sur" loops, stray Chinese/Cyrillic/emoji tokens).
- **Problem (4 root causes):**
  1. `--gate-global-init 0.2` (a *known-critical* flag) was **missing from the runbook**,
     so `gate_global` stayed 0.000 for 6 epochs, the model was effectively a 32-token
     window model with no long range.
  2. The curator source was **8× oversampled** → seen ~48× → memorization. PPL = train
     loss rewards memorization, so 4.3 was meaningless.
  3. SFT ran 3 epochs; it had already memorized by epoch 1 (loss 0.233 → 0.000).
  4. Generation was never measured per-epoch: only PPL was watched. It was bad from
     epoch 1 and nobody looked until epoch 6.
- **Lesson:** A great PPL on a memorized/leaky run tells you nothing. Measure *generation*
  every epoch. Put known-critical flags in the runbook checklist, not in memory.

### 4.2 Trying to revive the global path (2026-08-03)
- **Tried:** `--gate-global-init 0.2` to wake the global path.
- **Result:** Epoch 1 revived it (gate_global 0.247), but epoch 2 it **died again** (0.003).
- **Tried:** Freezing `gate_global = 0.5`.
- **Result:** PPL fine, generation *worse*: the forced global path injects noise.
- **Lesson:** Init alone can't sustain the global path; it loses to the local path in
  competition, and forcing it hurts.

### 4.3 The isolation experiment: local-only breakthrough (2026-08-04)
- **Tried (Deney 1, inference gate sweep, no training):** Took one checkpoint and swept the
  gates at inference time:

  | gate_global / gate_local | Generation |
  |---|---|
  | 0.5 / 0.3 (as trained) | garbage (forced global = noise) |
  | **0.0 / 1.0 (local-only)** | **real Turkish** ("Bu yıl boyunca, Türkiye'de…") |
  | 1.0 / 0.0 (global-only) | garbage (linear attention has no grammar) |

- **Tried (Deney 2, local-only training):** Froze `gate_global = 0.0`, let `gate_local`
  learn, resumed on 688M tokens.
- **Result:** PPL 85.5 → 46.6 → **28.75**, and for the first time across 6 experiments,
  **real Turkish finance sentences**: *"Merkez Bankası Başkanı bugün yaptığı açıklamada,
  'Bu süreçte çok önemli bir sayıda para politikasına sahiptir…' diye konuştu dedi."*
- **Lesson (the reframe):** The forced additive linear-global path degrades generation;
  the local causal window alone is sufficient *and* FHE-friendly. So the product path is
  "a good local-window LM (+ SFT)", while "linear-global = free long range" is a separate
  paper claim, not the product. This became `local_only_ppl28`, the v1 flagship checkpoint.

---

## Phase 5 — Softmax baseline: an unequal-scale caveat (2026-08-05)

- **Tried (Deney 3):** A standard softmax-attention + GELU baseline (d=512, 8 layers) to
  answer "is it the data or the architecture?"
- **Result:** PPL 269 → 87; generation partially sentence-like but **worse** than local-only.
- **Problem:** The comparison was **not equal-scale**: a small d=512×8L model on 2 epochs
  vs the 254M local-only model on 688M tokens.
- **Lesson:** We must *not* claim "the architecture is innocent / softmax is useless" from
  this. The defensible conclusion is narrower: "a 254M local-window FHE-native model is
  sufficient for sentence-level Turkish under FHE constraints." The equal-scale
  full-softmax ceiling remains an open ablation, not a product claim.

---

## Phase 6 — SFT: the memorization wall

### 6.1 Turkish SFT memorizes (2026-08-05)
- **Tried:** Instruction SFT on 74K (then 85K) Q&A pairs, lr 3e-5 and 3e-6.
- **Result:** Loss < 0.1 within one epoch = memorization; generation **collapsed**
  ("Ali tin tin tin" lockups, random chains). SFT *destroyed* the pretrain capability.
- **Lesson:** A 254M model + ~80K pairs memorizes in one epoch regardless of lr. SFT here
  needs early-stop (cut at loss ~1–1.5) and far more data.

### 6.2 English SFT collapses a small model (2026-08-08)
- **Tried:** SFT on FiQA (17K finance Q&A) over the English model, with early-stop, at both
  lr 3e-5 and 1e-5.
- **Result:** Model collapsed to symbols/repetition ("is#=(?&%", " is is is", "::::").
  The 16-epoch pretrain baseline was clearly better ("isaved jointly by the IRT –").
- **Lesson:** Instruction-SFT on a small, undertrained LM with a narrow Q&A format causes a
  distribution shift / exposure-bias collapse. For v1, pretrain-only is the final model;
  SFT is deferred until there is much more data and a stronger base.
- **Side finding:** When SFT pushed `gate_global` slightly negative, it activated the
  global path, which appeared to fail badly (cos 0.14). That was later traced to two engine
  bugs and a gate-blind diagnostic rather than a fundamental limit, and is now fixed (see 9.1).

---

## Phase 7 — The FHE engine: a long road to cos = 1.0

The engine goal: run the *trained* model end-to-end under CKKS, server-side, reproducing
the plaintext model's token choices.

### 7.1 Python/TenSEAL proof-of-concept (2026-07)
- **Tried:** Verified the CKKS primitives in Python (TenSEAL/SEAL): BSGS matvec, ct×ct
  (homomorphic multiply for PolyAct), ct+ct residual, plaintext-scalar scaling.
- **Result:** Single-block parity **cos_sim 0.999996, max_err 1.7e-3**; full local-window
  block **cos_sim 0.999999, max_err 8.4e-4**; multi-layer (1–3) error grew linearly
  (~1.5×/layer), cos_sim stayed > 0.99999.
- **Lesson:** The polynomial architecture is genuinely CKKS-compatible. Established the
  **hybrid protocol**: server does matvecs/ct×ct/residuals; the local-window softmax and KV
  accumulation are client-side. (Always read the flags before claiming "fully
  non-interactive FHE".)

### 7.2 C++/OpenFHE: it runs, but slow and OOM (2026-08-01)
- **Tried:** A C++ OpenFHE engine; first trained-model inference (d=1024, 1 layer).
- **Result:** 1 layer = 294s; 20 layers needed ring=65536 → 28 GB RAM → **OOM**.
- **Lesson:** CPU CKKS at this depth is memory-bound. Bootstrapping (to refresh levels) was
  the missing piece, and CPU bootstrap OOMs. Needed a GPU.

### 7.3 GPU unlock: FIDESlib (2026-08-01)
- **Tried:** FIDESlib (open-source CUDA CKKS, OpenFHE-compatible) on an RTX 4060 laptop.
- **Result:** GPU bootstrap works (~34 ms); matvec d=1024 ≈ 4.4 s; ct×ct ≈ 0.2 ms.
- **Problem:** Build friction on Kali: CUDA 12.4 needs GCC 13, a `math_functions.h`
  noexcept patch for glibc ≥ 2.41, ring=16384 to fit 8 GB VRAM, `-static-libstdc++` for ABI.
- **Lesson:** GPU CKKS is the only practical path to 20-layer depth (bootstrap without OOM).

### 7.4 The matvec was mathematically wrong, not "noisy" (2026-08-01)
- **Problem:** BSGS matvec showed error 0.34–3.5 at d=32–1024. We assumed it was CKKS
  rescaling noise.
- **Fix:** It wasn't noise: it was **7 math bugs**. Built a ground-truth simulation
  (`bsgs_matvec.py::simulate_bsgs`, err < 4e-15) and fixed: Halevi-Shoup diagonal slot
  shift, modular column wrapping, periodic input replication, K-block transpose, head
  rotation sign (`-h·dk`), global KV ordering (V·Kᵀ), and pre-norm LayerNorm placement.
- **Result:** Error dropped to **2–6e-08** (a ~2.5M× improvement) across GPU, CPU, Python.
- **Lesson:** Large "noise" in an FHE matvec is usually a math/indexing bug, not rescaling.
  Always verify against an exact plaintext simulation before blaming the cryptosystem.

### 7.5 Speed: replicated BSGS + parallel encode (2026-08-01)
- **Tried:** Profiling showed ~90% of layer time was **plaintext encoding**, not eval.
- **Fix:** Replicated BSGS (mamba3-style, R=4/8 diagonals packed per plaintext) → d=1024
  matvec 4.56 s → **1.07 s (4.3×)**; parallel mask encode (16 threads) → 7.35×; window-only
  local attention (16×32 instead of 16×256); mask-based FFN-down (no intermediate decrypt).
- **Result:** Layer time **47.2 s → 2.26 s**; 20 layers ≈ 55 s/token (no bootstrap).
- **Lesson:** In CPU-encoded CKKS pipelines, *encoding count* is the lever, not eval FLOPs.

### 7.6 Five silent root causes, and finally cos = 1.0 (2026-08-07)
The crux. Enabling bootstrap and chasing accuracy exposed five separate silent failures:
1. **`SetLevel(ct, 0)` was metadata-only** (no physical ModReduce) → bootstrap was a no-op
   on stale limbs → "approximation error too high". *Fix: drop SetLevel; call EvalBootstrap.*
2. **Bootstrap consumes ~15 levels** → depth 15 left nothing. *Fix: mult_depth = 25.*
3. **GPU bootstrap needs scale 2^59/60** (2^40 → decode garbage). *Fix: scale 59/60 in
   bootstrap mode.*
4. **`add_plain` (ct+pt) silently dropped the term** on a level mismatch (a Release-build
   `assert(false)` became a no-op `return`). The PolyAct constant `c0` was never added →
   a fixed bias through FFN-down (layer-0 err 0.498). *Fix: encrypt c0 as a ciphertext and
   use ct-ct add → err 0.498 → 0.0002.*
5. **`matvec_rep` left replica-blocks un-zeroed** (residues at slots 2049/4098/6147); the
   next matvec's replication rotated them into [0,1024) and corrupted the input (~1%/layer).
   Plus: `enc_q` was mutated in place, and `mult_scalar` broke the gate scale. *Fix:
   re-encrypt between matvecs, fold gates into the weights, skip the global path when
   gate_global = 0 → L1 err 0.32 → 1.1e-11.*
- **Result:** 20 layers + 6 GPU bootstraps, **cos(logits) = 1.0, top-5 5/5, top-1 exact**
  (FHE token == plaintext token). The project's core goal, hit.
- **Lesson:** GPU FHE libraries have *silent* failure modes: metadata-only ops, dropped
  plaintext adds, un-cleared replica slots. Verify **every layer** against a plaintext
  mirror; "it ran without crashing" means nothing.

### 7.7 Final speed pass (2026-08-07)
- **Fix:** Chunked FFN-up matvec (naive → 4× replicated chunks) → 31.9 s → 3.1 s.
- **Result:** Layer 37.6 s → 9.0 s; 20-layer inference 765 s → **193.5 s (3.2 min/token)**
  with accuracy preserved (parity cos = 1.0).

---

## Phase 8 — English model: model-agnostic proof, and the leak returns

### 8.1 The mixer leak, again (2026-08-07)
- **Tried:** An English model (WikiText-103, 121M tokens, d=512, 8L) for cleaner data and
  better BPE behavior.
- **Result:** First run gave **PPL 1.16** with gates ~0 (degenerate).
- **Problem:** We forgot `--no-mixer`, so the bidirectional TokenMixer (1.1) leaked again.
  Worse: the C++ FHE engine has no mixer (export omits it), so a mixer-trained
  checkpoint is *incompatible* with the FHE path: the engine was running a different
  (smaller) model than Python.
- **Fix:** Correct recipe = `--no-mixer --poly-ffn --feat-degree 2 --gate-local-init 0.5`.
- **Result:** Real PPL 187 → held-out **valid 92.7**; FHE parity **cos = 1.0 on all 8
  layers**; ~26 s/token; engine confirmed **model-agnostic** (d=512 works unchanged).
- **Lesson:** The mixer leak is a recurring trap; `--no-mixer` is mandatory for any
  checkpoint meant to run in the FHE engine. A diagnostic that loads `ck` instead of
  `ck['model']` silently evaluates random weights (fake PPL 22026), load state correctly.

### 8.2 Local English run on a laptop GPU (2026-08-08)
- **Tried:** The full English pipeline locally on an RTX 4060 (no rented instance).
- **Result:** 6-epoch no-mixer model, PPL 47.9; FHE parity cos = 1.0; FHE generation
  *"The capital of France is… developed into the city's primary cultural"*, real English
  structure, **every step matching plaintext**, ~20 s/token.
- **Lesson:** The whole FHE stack now runs on a laptop GPU without a remote instance,
  reproducible end-to-end. (`isveloped` = a BPE artifact, not an FHE error.)

---

## Phase 9 — Open problems and dead ends

### 9.1 The FHE global-attention path: was "broken", now RESOLVED (2026-09-09)
- **Original observation:** with `gate_global` non-zero, the FHE global matvec deviated
  ~3400× from the reference (cos 0.14). It had **never been exercised**: every verified
  checkpoint froze `gate_global = 0` (local-only), which skips the path. An SFT run briefly
  pushed the gate to ≈ -0.00015 and exposed the failure. The first conclusion ("a
  global-active model can't run in FHE") turned out to be **wrong**.
- **Investigation:** activating the path with a *healthy* gate (0.5) and a corrected parity
  diagnostic showed the failure was **three compounding issues**, not a fundamental limit:
  1. *Diagnostic artifact*: the parity check compared the gate-**embedded** FHE global
     against the gate-**free** mirror global, so they differed by a factor of `gate_global`
     regardless of correctness. (Fixed: compare against `gate_global · glob_ref`.)
  2. *Tiny-gate noise floor*: the observed gate (-0.00015) scaled the global signal below
     the CKKS noise floor, so even a correct computation decrypts to noise (cos 0.14). A
     healthy gate (0.5) lifts it clear of the floor.
  3. *Two real engine bugs*, isolated by testing feat_degree=1 (linear global) against
     feat_degree=2 (with the q² feature-map term):
     - **q² mutation:** the degree-2 term squared `enc_q_g` *after* the preceding `matvec_rep`
       had mutated it (replica-block residue is not zeroed by the fold). The q² came out
       corrupted → global ≈ 2× wrong. Fix: a **fresh encrypt** for the q² input, the same
       re-encrypt pattern the local path already used (Phase 7.6), just never applied to the
       global path because it was always skipped.
     - **missing rotation keys:** the global block-diagonal matvec needs baby-step rotations
       `bi·r` (e.g. 80 and 88 at d=1024, r=8, baby=12) that `compute_rotation_indices` never
       generated → `Rotation index 88 not found` crash at d=1024. The dense Q matvec skipped
       those steps (zero-diagonal skip), which is why local-only never hit it. Fix: generate
       the baby-step rotations alongside the giant steps.
- **Verified:** with `gate_global = 0.5`, parity is **cos_sim = 1.0** (max_err ~1e-6) for the
  global path at both **d=512** (feat_degree=2, q² active) and **d=1024** (feat_degree=1, the
  flagship size); full-layer and logits parity are also cos = 1.0.
- **Status: RESOLVED** (GPU engine; the CPU engine still lags on these fixes, see 9.2). The
  "global-active models can't run in FHE" blocker is lifted. Remaining caveat: every current
  checkpoint froze `gate_global = 0`, so a model *trained* with a healthy global gate still
  has to be produced, but the engine now supports it.
- **Lesson:** "It's broken" from a single accidental activation (a tiny gate + a gate-blind
  diagnostic) was a misdiagnosis. A never-exercised code path must be tested with a *healthy*
  input and a *like-for-like* reference before being declared broken.

### 9.2 Other open items
- **GPU utilization 1–7%**: the pipeline is CPU-encode/transfer-bound; needs GPU-resident
  plaintext or op batching to go faster.
- **CPU engine** (`sipher_fhe.cpp`) still lacks the KV-swap and LayerNorm fixes that the GPU
  engine has (only its matvec was corrected).
- **Long-range drift** after ~3–4 sentences (32-token window; global path off).
- **SFT** remains unsolved (memorization wall, Phase 6).
- **Token batching** (pack B positions per ciphertext to amortize encoding) is designed but
  unbuilt, a ~2–4× lever for multi-token generation.

### 9.3 A process lesson: lost work
- An English no-mixer checkpoint + tokenized data were **lost** when a rented instance was
  destroyed after a failed large-file transfer.
- **Lesson:** Shrink artifacts (int16/gzip) and pull over a reliable exec channel
  (`ssh -T cat`) *before* destroying an instance; keep durable copies locally.

---

## What survived as v1

- **Flagship checkpoint:** `local_only_ppl28`, 254M, d=1024, 20L, 16K Turkish BPE,
  `gate_global = 0` frozen, train PPL ≈ 28.8, coherent Turkish continuations.
- **FHE engine:** C++/OpenFHE + FIDESlib GPU; 20-layer + 6-bootstrap inference reproduces
  the plaintext model exactly (cos = 1.0, top-1 match); ~3.2 min/token (d=1024) and
  ~20–26 s/token (d=512 English) on workstation GPUs.
- **Headline findings:** (1) forced linear-global attention harms generation; (2) a
  bidirectional token mixer produces fake PPL via leakage; (3) a local-window polynomial
  network is a viable, FHE-native LM path; (4) GPU CKKS has silent failure modes that
  demand per-layer plaintext-mirror verification.

These negative results are as load-bearing as the positive ones, see
[`SIPHER_V1.md`](SIPHER_V1.md) for the release framing and
[the repository](https://github.com/gnyselcuk/sipher) for code.
