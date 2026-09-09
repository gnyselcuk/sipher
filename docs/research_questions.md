# Research Questions — Phase 1

## Primary Question

**RQ1:** Which Transformer operations dominate FHE computational cost,
and by how much?

### Sub-questions

**RQ1a:** What is the FHE cost breakdown of a single Transformer block
(self-attention + FFN + norms + residual)?

**RQ1b:** Which operations require bootstrapping, and how does this
change with model depth?

**RQ1c:** How does ciphertext packing efficiency vary across operations?
(i.e., which operations waste SIMD slots?)

**RQ1d:** What is the multiplicative depth profile of a full forward pass?
Where does the depth budget run out?

## Secondary Questions

**RQ2:** How do existing FHE-friendly Transformer adaptations (THE-X, BOLT, NEXUS)
shift the bottleneck, and what remains unsolved?

**RQ3:** Is there a meaningful difference between CKKS and TFHE cost profiles
for Transformer operations? (CKKS = approximate arithmetic, TFHE = programmable bootstrapping)

**RQ4:** What is the theoretical lower bound on FHE cost for:
- A linear transformation (matrix multiply)?
- A non-linear activation?
- An attention score computation?

## Hypotheses to Test

**H1:** Softmax and LayerNorm together account for >50% of total FHE cost
in a standard Transformer block, despite being <5% of FLOPs.

**H2:** The dominant cost driver is not multiplication count but
*rotation count* and *bootstrapping frequency*.

**H3:** Ciphertext packing utilization in standard attention is <25%
due to the triangular access pattern of attention masks.

**H4:** A polynomial-native replacement for softmax (e.g., truncated Taylor
or Chebyshev approximation) can reduce FHE cost by >10x while preserving
>95% of model accuracy on downstream tasks.
