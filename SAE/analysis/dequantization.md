# Methodology: Functional Weight Recovery for Quantized Interpretability

## 1. Motivation and Problem Statement

Mechanistic interpretability research relies heavily on computational graph abstraction libraries, such as `TransformerLens` (Nanda & Bloom, 2022) and `SAELens`, to instrument model activations. These libraries operate on the assumption that model layers are standard PyTorch modules (e.g., `nn.Linear`) exposing explicit weight matrices $W 
in \mathbb{R}^{d_{out} \times d_{in}}$.

However, state-of-the-art quantized inference kernels—such as AWQ (Lin et al., 2023), GPTQ (Frantar et al., 2023), and HQQ—implement linear layers as opaque operators. These modules store weights in packed discrete formats (e.g., `INT4` packed into `INT32`) and execute forward passes using custom fused CUDA kernels. They do not expose a reconstructible $W$ attribute, rendering them incompatible with standard interpretability instrumentation.

To bridge the gap between efficient inference representations and interpretability tooling, we introduce a method of **Functional Dequantization** (or Projective Weight Recovery).

## 2. Mathematical Formulation

### 2.1 Quantization as an Affine Operator

Let $W \nin \mathbb{R}^{d_{out} \times d_{in}}$ be the original high-precision weight matrix. The quantized linear layer defines an affine function $f_q: \mathbb{R}^{d_{in}} \to \mathbb{R}^{d_{out}}$:

$$ f_q(x) = x \hat{W}^T + b $$

Our objective is to extract the effective weight matrix $\hat{W}$ explicitly.

### 2.2 Universality Across Quantization Schemes

Crucially, this affine definition holds true even for complex quantization methods that utilize calibration data or outlier protection:

1.  **Group-Wise Quantization (GPTQ, HQQ):** Weights are scaled per-block rather than per-tensor.
    $$ \hat{W}_{ij} = s_{block} \cdot (q_{ij} - z_{block}) $$
    This is fully resolved into the static matrix $\hat{W}$.

2.  **Activation-Aware Quantization (AWQ):** AWQ scales input channels ($x$) based on calibration activation magnitude to protect salient weights.
    $$ f(x) = (x \odot s_{chan}) \cdot Q_{int} \cdot s_{weight} $$
    Mathematically, this remains a linear transformation where the effective weight absorbs the channel scaling:
    $$ \hat{W} = s_{chan}^T \odot Q_{int} \odot s_{weight} $$

3.  **Mixed-Precision Decomposition (LLM.int8()):** Some methods separate "outlier" vectors into FP16 and compress the rest to INT8.
    $$ f(x) = x_{out} W_{fp16}^T + x_{reg} W_{int8}^T $$
    Since matrix multiplication is distributive, this is functionally equivalent to a single linear map with a composite weight matrix:
    $$ \hat{W} = W_{fp16} \oplus W_{dequantized\_int8} $$

In all cases, the inference operation collapses to a static linear map $x \mapsto x \hat{W}^T$. Therefore, we do not need access to the original calibration data or the specific decomposition rules to recover $\hat{W}$; we only need to observe the operator's output.

## 3. Algorithm: Functional Dequantization via Basis Projection

We treat the quantized module $f_q(\cdot)$ as a black-box operator. To recover the effective weight matrix $\hat{W}$, we project the standard basis vectors through this operator.

### 3.1 Derivation

Let $I_{d_{in}} \nin \mathbb{R}^{d_{in} \times d_{in}}$ be the identity matrix, which effectively stacks the standard basis vectors $e_1, \dots, e_{d_{in}}$ as rows.

1.  **Bias Extraction:**
    We first identify the bias term by passing a zero vector (or observing the affine offset):
    $$ b = f_q(\mathbf{0}) $$
    *(Note: In practice, if the bias is accessible as an attribute, we read it directly. If implicitly fused, we compute $f_q(\mathbf{0})$).*

2.  **Weight Projection:**
    We perform a forward pass with the identity matrix $I_{d_{in}}$:
    $$ Y = f_q(I_{d_{in}}) $$
    
    Substituting the affine definition:
    $$ Y = I_{d_{in}} \hat{W}^T + b = \hat{W}^T + b $$

3.  **Recovery:**
    We solve for $\hat{W}$:
    $$ \hat{W}^T = Y - b $$
    $$ \hat{W} = (Y - b)^T $$

This yields the **exact** effective weights $\hat{W}$ used by the quantization kernel, represented in the accumulation precision (typically `BF16` or `FP16`).

### 3.2 Implementation: The "Identity Trick"

For a generic quantized module `layer`:

```python
# 1. Construct Identity Matrix (Basis Vectors)
# We use float32 to ensure compatibility with all kernel types (HQQ, AWQ)
I = torch.eye(layer.in_features, dtype=torch.float32, device=layer.device)

# 2. Project Basis through Black-Box Kernel
with torch.no_grad():
    output = layer(I)  # output = W^T + b

# 3. Recover Weights via Transposition and Bias Subtraction
if layer.bias is not None:
    W_eff = (output - layer.bias).T
else:
    W_eff = output.T
    
# 4. Cast to Target Precision (BFloat16) for Analysis
W_eff = W_eff.to(torch.bfloat16)
```

## 4. Validity for Interpretability

Crucially, this process is **lossless** with respect to the quantized function. The surrogate layer constructed from $\hat{W}$ produces activations identical to the original quantized kernel:

$$ f_{surr}(x) \equiv f_q(x) \quad \forall x \nin \mathbb{R}^{d_{in}} $$

For Sparse Autoencoder (SAE) training, this ensures that the SAE learns features from the **actual quantized distribution**, capturing any feature suppression or noise introduced by the quantization process. The memory cost is an increase in VRAM usage (converting `INT4` weights back to `BF16`), but this trade-off is necessary to enable the use of mature graph-based analysis tools.

--- 

# Appendix A: Proof of Applicability for Specific Quantization Algorithms

This appendix details why the Projective Weight Recovery method is valid for specific state-of-the-art quantization algorithms, regardless of their calibration complexity.

### A.1 GPTQ (Generalized Post-Training Quantization)
*   **Mechanism:** GPTQ iteratively quantizes weights column-by-column, using the inverse Hessian of the loss function to update the remaining unquantized weights. This compensates for the error introduced by rounding.
*   **Result:** The final model consists of quantized integer weights $Q$ and group-wise scales $S$.
*   **Recovery Validity:** Although the *process* of finding $Q$ is highly non-linear (involving Hessian inversion), the *result* is a standard linear operator. The identity projection captures the exact output of this Hessian-optimized arithmetic.

### A.2 AWQ (Activation-Aware Weight Quantization)
*   **Mechanism:** AWQ observes calibration activations to identify "salient" channels. It applies a scaling factor $s > 1$ to these channels (magnifying the weights) before quantization, effectively increasing their precision relative to the quantization noise.
*   **Result:** The inference kernel performs an implicit channel-wise scaling: $y = x \cdot (S_{chan} \cdot Q_{int})$.
*   **Recovery Validity:** Functional dequantization recovers the composite weight $\hat{W} = S_{chan} \cdot Q_{int}$. This $\hat{W}$ correctly includes the magnified values for salient weights, ensuring the SAE analyzes the exact distribution where "important" features were preserved.

### A.3 HQQ (Half-Quadratic Quantization)
*   **Mechanism:** HQQ formulates quantization as an optimization problem solved via a half-quadratic splitting algorithm. It does not typically use calibration data but optimizes the quantization parameters ($s, z$) to minimize reconstruction error locally.
*   **Result:** A highly optimized set of group-wise parameters.
*   **Recovery Validity:** Similar to GPTQ, HQQ produces a static affine map. The identity trick captures the precise result of the optimization solver without needing to re-solve the half-quadratic objective.

### A.4 LLM.int8() (Mixed-Precision Decomposition)
*   **Mechanism:** This method dynamically separates "outlier" hidden states (magnitude $> 6.0$) during the forward pass. Matrix multiplication is decomposed into two parts: one performed in FP16 for outliers, and one in INT8 for the rest.
*   **Result:** $y = x_{outlier}W_{outlier} + x_{regular}W_{regular}$.
*   **Recovery Validity:** This operation is mathematically equivalent to multiplying the full input $x$ by a full matrix $\hat{W}$ where specific rows/columns are high-precision and others are low-precision. Because matrix multiplication distributes over addition, the identity projection recovers the single unified matrix $\hat{W}$ that represents the sum of these two operations.