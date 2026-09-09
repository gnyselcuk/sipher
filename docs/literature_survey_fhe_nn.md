# Literature Survey: FHE-Native Neural Networks
## Positioning the Research Gap for CipherFormer

**Date:** 2026-07-27
**Purpose:** Survey existing work on privacy-preserving Transformer/NN inference and training under FHE/HE/MPC, to identify the gap that FHE-native neural network design (CipherFormer) aims to fill.

---

## Category A: "Adapt Transformer to FHE/MPC"

These papers take existing Transformer architectures and adapt them for private inference by replacing non-linear operations with HE/MPC-compatible approximations.

---

### A1. THE-X: Privacy-Preserving Transformer Inference with Homomorphic Encryption

| Field | Detail |
|-------|--------|
| **Authors** | Tianyu Chen, Hangbo Bao, Shaohan Huang, Li Dong, Binxing Jiao, Daxin Jiang, Haoyi Zhou, Jianxin Li, Furu Wei |
| **Year** | 2022 |
| **Venue** | Findings of ACL 2022 |
| **arXiv** | 2206.00216 |
| **FHE Scheme** | Homomorphic Encryption (specific scheme not stated in abstract; likely CKKS-based given the polynomial approximation approach) |
| **Inference/Training** | Inference only |

**Transformer Modifications:**
- Proposes an *approximation workflow* for all non-polynomial functions in Transformers:
  - **GELU** → polynomial approximation
  - **Softmax** → polynomial approximation
  - **LayerNorm** → polynomial approximation
- The key idea: replace every non-polynomial operation with a polynomial counterpart that HE can evaluate natively.

**Bottlenecks Identified:**
- Transformer blocks contain complex non-polynomial computations (GELU, Softmax, LayerNorm) not directly supported by HE tools.
- These are the primary barrier to ciphertext-only Transformer inference.

**Results:**
- Enables Transformer inference on encrypted data for multiple downstream tasks.
- Claims "negligible performance drop" vs. plaintext inference.
- Theory-guaranteed privacy preservation.

**Limitations / What They DON'T Solve:**
- No quantitative latency/throughput benchmarks reported in the abstract.
- Approximation quality degrades with higher-degree polynomials or extreme inputs.
- Does not redesign the architecture: purely a post-hoc approximation layer on standard Transformers.
- No training under HE.
- Scalability to large models (GPT-scale) not demonstrated.
- Multiplicative depth from polynomial approximations likely requires bootstrapping or limits model depth.

---

### A2. BOLT: Privacy-Preserving, Accurate and Efficient Inference for Transformers

| Field | Detail |
|-------|--------|
| **Authors** | Qi Pang (CMU), Jinhao Zhu (UC Berkeley), Helen Möllering (TU Darmstadt), Wenting Zheng (CMU), Thomas Schneider (TU Darmstadt) |
| **Year** | 2024 |
| **Venue** | IEEE S&P 2024 |
| **ePrint** | 2023/1893 |
| **Crypto Scheme** | Secure Multi-Party Computation (MPC) + Homomorphic Encryption (hybrid) |
| **Inference/Training** | Inference only |

**Transformer Modifications:**
- Novel ML optimizations for efficient matrix multiplications and nonlinear computations under MPC.
- Supports both linear and nonlinear Transformer operations in the secure setting.

**Bottlenecks Identified:**
- Extensive model size of Transformers.
- Resource-intensive matrix-matrix multiplications under MPC.
- Communication cost is the dominant bottleneck in prior MPC-based Transformer inference.

**Results:**
- **10.91× reduction in communication cost** vs. prior work.
- **4.8–9.5× faster inference** across various network settings vs. state-of-the-art.
- Accuracy comparable to floating-point models.

**Limitations / What They DON'T Solve:**
- Still adapts existing Transformer architectures rather than designing for MPC/HE from scratch.
- MPC-based (requires interaction between parties), not purely FHE.
- No training under encryption.
- Scalability to very large LLMs (billions of parameters) not demonstrated.
- The architecture itself is not modified — only the computation protocols are optimized.

---

### A3. Iron: Private Inference on Transformers

| Field | Detail |
|-------|--------|
| **Authors** | Wenxuan Zeng, Meng Li, Wenjie Xiong, Tong Tong, Wen-jie Lu, Jin Tan, Runsheng Wang, Ru Huang |
| **Year** | 2022 |
| **Venue** | NeurIPS 2022 |
| **arXiv** | 2201.06654 |
| **Crypto Scheme** | Secure Two-Party Computation (secret-sharing-based MPC) |
| **Inference/Training** | Inference only |

**Transformer Modifications:**
- Custom secure protocols for Transformer-specific operations:
  - **Linear projections**: optimized secure matrix multiplication
  - **Softmax**: polynomial/fixed-point approximation
  - **GELU**: polynomial approximation
  - **LayerNorm**: secure protocol
- Protocol-level optimization targeting Transformer workloads specifically (not treating Transformers as generic NNs).

**Bottlenecks Identified:**
- Non-linear operations (Softmax, GELU, LayerNorm) are the main cost drivers in secure Transformer inference.
- Large matrix multiplications in attention and FFN layers dominate communication.
- Prior MPC frameworks treated Transformers as generic NNs, missing optimization opportunities.

**Results:**
- Negligible accuracy loss vs. plaintext inference.
- Seconds-level private inference latency for BERT-style models.
- Gigabyte-scale communication (vs. much higher in prior work).
- ~2–3× improvement in latency and communication over prior secure inference baselines.

**Limitations / What They DON'T Solve:**
- MPC-based (requires two interacting parties), not purely FHE/single-party.
- Does not redesign the Transformer architecture.
- No training under encryption.
- Semi-honest security model only.
- Scalability to GPT-scale models not demonstrated.

---

### A4. LLMs Can Understand Encrypted Prompt: Towards Privacy-Computing Friendly Transformers

| Field | Detail |
|-------|--------|
| **Authors** | Xuanqi Liu, Zhuotao Liu |
| **Year** | 2023 |
| **Venue** | arXiv preprint (2305.18396) |
| **Crypto Scheme** | Privacy-computing (MPC/HE hybrid, building on Iron's framework) |
| **Inference/Training** | Inference only |

**Transformer Modifications:**
- This is the closest to "FHE-friendly design" in Category A.
- Substitutes computation-heavy and communication-heavy operators in the Transformer architecture with privacy-computing friendly approximations.
- Modifies the architecture itself, not just the protocols.

**Bottlenecks Identified:**
- Original LLM architectures impose significant overhead when private inputs are forward-propagated.
- Computation-heavy and communication-heavy operators are the root cause.

**Results:**
- **5× acceleration in computation** vs. Iron (NeurIPS 2022).
- **80% reduction in communication overhead** vs. Iron.
- Nearly identical accuracy to the original model.

**Limitations / What They DON'T Solve:**
- Still starts from an existing Transformer and modifies it — not designed from scratch for FHE.
- Modifications are operator-level substitutions, not a fundamental rethinking of the architecture.
- No training under encryption.
- Specific operator replacements not detailed in the abstract.

---

### A5. NEXUS — FHE-Optimized Transformer

| Field | Detail |
|-------|--------|
| **Status** | **NOT FOUND** on arXiv, Google Scholar, or IACR ePrint as of 2026-07-27 |

**Notes:**
- Extensive search across arXiv (multiple query formulations), IACR ePrint, and Google Scholar yielded no paper titled "NEXUS" related to FHE-optimized Transformers or privacy-preserving Transformer inference.
- Possible explanations:
  - May be under a different name or acronym.
  - May be an unpublished/internal project.
  - May be confused with another paper (e.g., Orion, FEnc², or a hardware accelerator named NEXUS).
- **Recommendation:** Verify the source of this reference. If it refers to a specific project, additional identifying information is needed.

---

## Category B: "FHE/Secure ML Frameworks"

These are frameworks/protocols for secure ML inference (and sometimes training) that provide general-purpose cryptographic infrastructure.

---

### B1. Concrete ML by Zama

| Field | Detail |
|-------|--------|
| **Developer** | Zama |
| **Type** | Open-source FHE ML framework (not a single paper) |
| **FHE Scheme** | **TFHE** (Torus Fully Homomorphic Encryption) via programmable bootstrapping |
| **Docs** | https://docs.zama.org/concrete-ml |
| **Inference/Training** | **Inference only** |

**Supported Models:**
- Built-in: Linear regression, Logistic regression, Decision trees, Random forests, XGBoost
- Custom: Quantized neural networks via PyTorch/ONNX → FHE circuit compilation
- Small feed-forward NNs, MLPs, some CNNs (if they fit circuit constraints)

**Key Technical Details:**
- Compiles ML models into FHE circuits using integer/fixed-point quantization.
- Non-linear functions evaluated via TFHE's **programmable bootstrapping** (lookup-table-based).
- Client/server deployment: client encrypts, server evaluates on ciphertext, client decrypts.
- Simulation mode for estimating FHE behavior without real encryption.

**Bottlenecks / Limitations:**
- **Inference only**: no encrypted training.
- FHE execution is much slower than plaintext (orders of magnitude for deep models).
- Models must be **quantized** to integers (accuracy trade-off).
- Large/deep models are impractical: Transformers, LLMs, large CNNs not supported.
- Non-linear operations are expensive (each requires programmable bootstrapping).
- Circuit depth, bit-width, and memory constraints limit model complexity.
- No native support for attention mechanisms, Softmax, or LayerNorm.
- Best suited for: linear models, tree ensembles, small quantized NNs.

**Relevance to CipherFormer:**
- Concrete ML demonstrates that TFHE can handle small ML models but cannot scale to Transformers.
- The gap between "small quantized NN on TFHE" and "Transformer-scale model on FHE" is exactly where FHE-native design is needed.
- CipherFormer could potentially target Concrete ML / TFHE as a backend if the architecture is designed for programmable bootstrapping from the start.

---

### B2. CrypTFlow2: Practical 2-Party Secure Inference

| Field | Detail |
|-------|--------|
| **Authors** | Deevashwer Rathee, Mayank Rathee, Nishant Kumar, Nishanth Chandran, Divya Gupta, Aseem Rastogi, Rahul Sharma |
| **Year** | 2020 |
| **Venue** | ACM CCS 2020 |
| **arXiv** | 2010.06457 |
| **Crypto Scheme** | Secure 2-Party Computation (2PC): secret sharing + garbled circuits |
| **Inference/Training** | Inference only |

**Key Contributions:**
- New 2PC protocols for secure comparison and secure division.
- Balances round complexity and communication complexity for DNN inference.
- Outputs are **bitwise equivalent** to cleartext execution (correctness guarantee).

**Results:**
- **First secure inference on ImageNet-scale DNNs** (ResNet50, DenseNet121).
- Models are **≥10× larger** than those in prior 2-party DNN inference work.
- On prior benchmarks: **~10× less communication** and **20–30× less time** than state-of-the-art.

**Bottlenecks Identified:**
- Secure comparison and division are the critical operations for DNN inference (ReLU, BatchNorm, MaxPool).
- Communication cost is the dominant bottleneck in 2PC.

**Limitations / What They DON'T Solve:**
- MPC-based (requires two interacting parties), not FHE.
- CNN-focused (ResNet, DenseNet), not Transformer-specific.
- No training under encryption.
- Does not redesign architectures for secure computation.
- Semi-honest security model.

---

## Category C: "FHE-Native / HE-Friendly Design"

These papers design or significantly restructure neural network architectures with HE/FHE constraints as a first-class consideration. This is the category closest to CipherFormer's vision.

---

### C1. PRISM: Sensitivity-Aware Polynomial Pruning for Efficient Neural Network Encryption

| Field | Detail |
|-------|--------|
| **Authors** | Sahaj Majavdia, Mahdi Taheri |
| **Year** | 2026 |
| **Venue** | arXiv preprint (2607.18342) |
| **FHE Scheme** | **CKKS** |
| **Inference/Training** | Inference only |

**Key Contributions:**
- Systematic reliability characterization of pruned CKKS-encrypted neural networks.
- Introduces **Polynomial-Sensitivity-Aware Pruning (PSAP)**: scores filters by weight magnitude + polynomial activation sensitivity + rotation cost.
- Adaptive mixed-degree polynomial allocation to reduce multiplicative depth.

**Results:**
- Limits catastrophic (>10pp accuracy drop) layers to ≤2 vs. 5–14 for magnitude-pruned baselines.
- Reduces worst-case vulnerability by up to **29×** under bit-flip injection.
- **45.2% reduction in Halevi–Shoup rotations** on ResNet-32.
- Multiplicative depth reduced from **66 → 56 levels**, enabling **leveled inference without bootstrapping**.
- Fault-critical layers are only **1.1% of parameters**: selective hardening at minimal overhead.

**Limitations:**
- Still adapts existing architectures (ResNet), not designed from scratch for CKKS.
- Evaluation limited to two architectures, two datasets.
- Leveled HE constrains maximum depth.
- No Transformer evaluation.
- Inference only.

**Relevance to CipherFormer:**
- Demonstrates that **HE-aware pruning and polynomial allocation** can dramatically reduce FHE cost.
- The insight that only 1.1% of layers are fault-critical suggests FHE-native designs could be much more efficient.
- CipherFormer could incorporate similar sensitivity-aware design but from the ground up.

---

### C2. Towards Deep Encrypted Training: HE-Friendly ResNet with Batched Inference

| Field | Detail |
|-------|--------|
| **Authors** | Nges Brian Njungle, Eric Jahns, Michel A. Kinsy |
| **Year** | 2026 |
| **Venue** | arXiv preprint (2604.16834) |
| **FHE Scheme** | Homomorphic Encryption (specific scheme not stated; likely CKKS) |
| **Inference/Training** | Primarily inference; title suggests training-oriented workloads |

**Key Contributions:**
- Optimized algorithms for batched HE-friendly neural network inference.
- Pipeline architecture for resource-efficient batch execution.
- Uses HE-friendly ResNet-20 and ResNet-34 models (architecture adapted for HE).

**Results:**
- ResNet-20 on CIFAR-10: **8.86 sec/image** amortized (batch of 512), **98.96 GB** peak memory.
  - **1.78× runtime improvement** and **3.74× memory reduction** vs. state-of-the-art.
- ResNet-34 on CIFAR-100: **28.14 sec/image** (batch of 256), **246.78 GB** RAM.

**Limitations:**
- Still uses adapted ResNets, not designed from scratch.
- Extremely high memory requirements (99–247 GB).
- CIFAR-scale only, no ImageNet or Transformer evaluation.
- Inference-focused despite "training" in the title.
- No Transformer or attention mechanism evaluation.

---

### C3. FEnc²: Architecture-Aware Fragment Encoding for CKKS Private CNN Inference

| Field | Detail |
|-------|--------|
| **Authors** | Ran Ran, Zhaoting Gong, Nuo Xu, Yuanchao Xu, Fan Yao, Wujie Wen |
| **Year** | 2026 |
| **Venue** | ISCA 2026 |
| **arXiv** | 2606.16359 |
| **FHE Scheme** | **CKKS** |
| **Inference/Training** | Inference only |

**Key Contributions:**
- Fragment-based encoding framework for CKKS-based private CNN inference.
- **Conv-aware Encoding** + **Architecture-aware ciphertext compression**.
- Optimizes ciphertext slot utilization, rotation complexity, and ciphertext density.

**Results:**
- Speedups over state-of-the-art Orion system for LeNet/MNIST and MobileNet/ImageNet.

**Limitations:**
- CNN-focused, no Transformer support.
- Encoding/packing optimization, not architecture design.
- Inference only.

---

### C4. SFPDML: MKTFHE-Friendly Activation Function Design

| Field | Detail |
|-------|--------|
| **Authors** | Hongxiao Wang, Zoe L. Jiang, Yanmin Zhao, Siu-Ming Yiu, Peng Yang, Man Chen, Zejiu Tan, Bohan Jin |
| **Year** | 2022 (revised 2024) |
| **arXiv** | 2211.09353 |
| **FHE Scheme** | **Multi-Key TFHE (MKTFHE)** |
| **Inference/Training** | Training (logistic regression, NN) |

**Key Contributions:**
- Proposes a new **MKTFHE-friendly activation function** using a "homogenizer" and "compare quads."
- Designed specifically for the FHE scheme's constraints, not adapted from standard activations.

**Results:**
- ~**10× higher efficiency** than 7-order Taylor polynomial approximations of Sigmoid.
- Similar accuracy to high-order polynomial activation schemes.

**Relevance to CipherFormer:**
- This is a genuine example of FHE-native activation function design.
- Demonstrates that designing activations for the FHE scheme (rather than approximating standard ones) yields major efficiency gains.
- CipherFormer's vision extends this principle to the entire architecture, not just activations.

---

## Category D: "FHE Training"

These papers train neural networks entirely (or substantially) under FHE. This is the most ambitious and least mature category.

---

### D1. Glyph: Fast and Accurately Training Deep Neural Networks on Encrypted Data

| Field | Detail |
|-------|--------|
| **Authors** | Qian Lou, Bo Feng, Geoffrey C. Fox, Lei Jiang |
| **Year** | 2019/2020 |
| **Venue** | NeurIPS 2020 |
| **arXiv** | 1911.07101 |
| **FHE Scheme** | **Hybrid TFHE + BGV** |
| **Inference/Training** | **Training** |

**Key Contributions:**
- Switches between TFHE and BGV during encrypted training:
  - **TFHE** for nonlinear activations (logic-operation-friendly).
  - **BGV** for multiply-accumulation (MAC) operations (vectorial-arithmetic-friendly).
- Applies **transfer learning** to reduce ciphertext-to-ciphertext MAC operations.

**Bottlenecks Identified:**
- Prior FHE training used BGV lookup tables for all activations, which is extremely slow.
- Ciphertext-to-ciphertext MAC operations in convolutional layers are costly.

**Results:**
- State-of-the-art test accuracy among FHE-based training methods.
- **99% reduction in training latency** vs. prior FHE-based technique.

**Limitations:**
- Small-scale models and datasets only.
- Hybrid scheme switching adds complexity.
- Transfer learning dependency reduces the "fully encrypted" training scope.
- No Transformer training.
- Still orders of magnitude slower than plaintext training.

---

### D2. Neural Network Training on Encrypted Data with TFHE (Zama)

| Field | Detail |
|-------|--------|
| **Authors** | Luis Montero, Jordan Frery, Celia Kherfallah, Roman Bredehoft, Andrei Stoian (Zama) |
| **Year** | 2024 |
| **Venue** | arXiv preprint (2401.16136) |
| **FHE Scheme** | **TFHE** |
| **Inference/Training** | **Training** |

**Key Contributions:**
- Unified FHE training approach for quantized neural networks on encrypted data.
- Supports horizontal and vertical data splitting between multiple parties.
- Trains logistic regression and multi-layer perceptrons (MLPs).

**Limitations:**
- Only logistic regression and small MLPs, no CNNs, no Transformers.
- No quantitative results in the abstract.
- Quantized models only.
- Scalability to deeper/wider networks not demonstrated.

---

### D3. Privacy-Preserving CNN Training with Transfer Learning: Two Hidden Layers

| Field | Detail |
|-------|--------|
| **Authors** | John Chiang |
| **Year** | 2025 |
| **Venue** | arXiv preprint (2504.12623) |
| **FHE Scheme** | FHE (specific scheme not stated in abstract) |
| **Inference/Training** | **Training** |

**Key Contributions:**
- Trains a **4-layer neural network entirely under FHE** (non-interactive).
- Key insight: replace Softmax with Sigmoid + Binary Cross-Entropy loss for homomorphic classification.
- Shows BCE loss extends naturally to multi-class settings.
- Identifies that prior loss functions (SLE loss, 2019 CVPR Workshop loss) suffer from **vanishing gradients** as depth increases.
- Improved **Double Volley Revolver** data encoding scheme for better compute/memory trade-off.

**Bottlenecks Identified:**
- Loss function design is critical for FHE training depth scalability.
- Data encoding overhead for large-scale encrypted datasets.
- Vanishing gradients in prior FHE-compatible loss functions.

**Limitations:**
- Only 4 layers (2 hidden), very shallow.
- No Transformer or attention mechanism.
- Specific FHE scheme not stated.
- No quantitative accuracy/runtime benchmarks in abstract.

---

### D4. Understanding the Resource Cost of FHE in Quantum Federated Learning

| Field | Detail |
|-------|--------|
| **Authors** | Lukas Böhm, Arjhun Swaminathan, Anika Hannemann, Erik Buchmann |
| **Year** | 2026 |
| **arXiv** | 2603.02799 |
| **FHE Scheme** | **CKKS** |
| **Inference/Training** | **Training** (federated) |

**Key Contributions:**
- First QCNN trained in a federated setting with CKKS-encrypted parameters.
- Evaluates FHE overhead in quantum federated learning.

**Key Finding:**
- Memory and communication overhead remain substantial, making FHE challenging to deploy.
- Reducing model parameters to minimize overhead degrades classification performance.
- Fundamental **trade-off between privacy and model complexity**.

---

## Category E: Survey Papers

### E1. Private Transformer Inference in MLaaS: A Survey

| Field | Detail |
|-------|--------|
| **Authors** | Yang Li, Xinyu Zhou, Yitong Wang, Liangxin Qian, Jun Zhao |
| **Year** | 2025 |
| **arXiv** | 2505.10315 |
| **Scope** | Survey of Private Transformer Inference (PTI) |

**Key Contributions:**
- Structured taxonomy and evaluation framework for PTI.
- Covers MPC-based and HE-based approaches.
- Focuses on balancing resource efficiency with privacy.

**Key Challenges Identified:**
- Privacy risks from centralized MLaaS processing.
- Preserving both user data privacy AND model privacy.
- Balancing resource efficiency with privacy protection.
- Bridging the gap between high-performance inference and data privacy.

---

## Gap Analysis: Where CipherFormer Fits

### What the field has done:
1. **Adapt existing Transformers** (THE-X, BOLT, Iron, Liu & Liu): Replace non-linear ops with polynomial approximations or secure protocols. Achieves 2–10× speedups but still starts from architectures designed for plaintext.
2. **Build general FHE ML frameworks** (Concrete ML, CrypTFlow2): Provide infrastructure but cannot scale to Transformers. Concrete ML tops out at small MLPs/tree models.
3. **HE-aware pruning/encoding** (PRISM, FEnc²): Optimize existing architectures for HE but don't redesign them.
4. **FHE training** (Glyph, Zama, Chiang): Demonstrate feasibility on tiny models (4 layers, MLPs) but nowhere near Transformer scale.

### What NOBODY has done (the CipherFormer gap):
1. No one has designed a Transformer architecture from scratch for FHE. Every paper starts from a standard Transformer and adapts it. CipherFormer's vision of FHE-native design is genuinely novel.
2. No one has addressed the multiplicative depth problem architecturally. Papers deal with depth via bootstrapping (expensive) or leveled HE (limits depth). An architecture designed to minimize multiplicative depth from the start would be fundamentally different.
3. No one has co-designed the attention mechanism for FHE. Softmax replacement is always a post-hoc approximation. An FHE-native attention mechanism (e.g., linear attention, polynomial attention designed for CKKS/TFHE) doesn't exist.
4. No one has designed FHE-native positional encodings, normalization, or feed-forward blocks. These are always adapted from plaintext designs.
5. No one has demonstrated FHE training of anything resembling a Transformer. Training is limited to 4-layer MLPs. FHE-native training of even a small Transformer is completely open.
6. The TFHE vs. CKKS choice is never co-designed with the architecture. CipherFormer could exploit TFHE's programmable bootstrapping for non-linear ops and CKKS for linear algebra, with the architecture designed to maximize each scheme's strengths.

### Positioning statement:
> Existing work treats FHE as a constraint to be satisfied by modifying plaintext-designed architectures. CipherFormer inverts this: FHE is the design target, and the architecture is built to exploit FHE's computational structure. This is analogous to how GPU-native architectures (e.g., Tensor Cores) changed how we design models, not just how we run them.

---

## Quick Reference Table

| Paper | Year | Scheme | Approach | Scale | Training? | Key Result |
|-------|------|--------|----------|-------|-----------|------------|
| THE-X | 2022 | HE (CKKS?) | Polynomial approx. | BERT | No | Negligible accuracy drop |
| BOLT | 2024 | MPC+HE | Protocol optimization | BERT | No | 10.91× comm. reduction |
| Iron | 2022 | MPC (SS) | Custom protocols | BERT | No | 2-3× speedup |
| Liu & Liu | 2023 | MPC/HE | Operator substitution | LLM | No | 5× faster than Iron |
| Concrete ML | 2020+ | TFHE | Framework | Small NN | No | General FHE ML |
| CrypTFlow2 | 2020 | 2PC | Protocol design | ResNet50 | No | First ImageNet-scale 2PC |
| PRISM | 2026 | CKKS | HE-aware pruning | ResNet-32 | No | 45% rotation reduction |
| Njungle+ | 2026 | HE | Batched HE inference | ResNet-34 | ~No | 1.78× speedup |
| FEnc² | 2026 | CKKS | Encoding optimization | CNN | No | ISCA 2026 |
| SFPDML | 2022 | MKTFHE | FHE-native activation | Small NN | Yes | 10× vs Taylor |
| Glyph | 2020 | TFHE+BGV | Hybrid training | Small CNN | Yes | 99% latency reduction |
| Zama TFHE | 2024 | TFHE | Unified training | MLP | Yes | First TFHE NN training |
| Chiang | 2025 | FHE | Sigmoid+BCE training | 4-layer | Yes | Vanishing gradient fix |
| **CipherFormer** | **2026** | **TBD** | **FHE-native design** | **Transformer** | **Goal: Yes** | **Open** |
