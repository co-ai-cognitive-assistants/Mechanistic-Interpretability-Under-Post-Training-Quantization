# Interpretability Metrics for Quantized SAE Comparison

This document provides the theoretical basis, scientific justification, and interpretation guide for the metrics used to evaluate the impact of model quantization on Sparse Autoencoders (SAEs).

## 1. Feature Fidelity: Bi-Directional Precision & Recall (with Sensitivity Analysis)

To measure whether quantization preserves the semantic "concepts" learned by the model, we compare the decoder feature vectors ($W_{dec}$) of the Baseline SAE (e.g., BFloat16) and the Target SAE (e.g., Int4) using **Maximum Cosine Similarity**.

$$ \text{Sim}(v_{a}, W_{b}) = \max_{j} \left( \frac{v_{a} \cdot v_{b, j}}{\|v_{a}\| \|v_{b, j}\|} \right) $$

We calculate this in both directions to distinguish between "forgetting" features and "hallucinating" artifacts.

### A. Feature Recall (Base $\to$ Target)
*   **Definition:** The proportion of Baseline features that have a match in the Target SAE with Similarity > Threshold $t$.
*   **Scientific Question:** "Did we lose any concepts?"
*   **Sensitivity Analysis (AUC):** We report recall at multiple thresholds ($t \in \{0.7, 0.8, 0.9, 0.95, 0.99\}$) to ensure findings are robust to hyperparameter choice.
*   **Interpretation:**
    *   **High Recall:** The quantized model retains the subtle, high-resolution concepts of the full-precision model.
    *   **Low Recall:** **Feature Forgetting.** Quantization noise has drowned out fine-grained features.

### B. Feature Precision (Target $\to$ Base)
*   **Definition:** The proportion of Target features that match a Baseline feature with Similarity > $t$.
*   **Scientific Question:** "Are the new features real?"
*   **Interpretation:**
    *   **High Precision:** The features found in the quantized model are faithful to the original semantics.
    *   **Low Precision:** **Hallucination.** The SAE is learning artifacts introduced by the quantization process (e.g., grid patterns or rounding errors) rather than true model semantics.

*Reference:* Bricken et al., "Towards Monosemanticity: Decomposing Language Models With Dictionary Learning" (Anthropic, 2023).

---

## 2. Structural Complexity: Effective Rank

### Definition
We analyze the geometry of the feature space using the Singular Value Decomposition (SVD) of the decoder weights. The **Effective Rank** is the Shannon entropy of the normalized singular values ($\sigma_k$).

$$ H = - \sum_{k} p_k \log p_k, \quad \text{where } p_k = \frac{\sigma_k}{\sum_i \sigma_i} $$
$$ \text{Effective Rank} = e^H $$

### Scientific Justification
This metric quantifies the **effective dimensionality** of the learned representation [1].
*   **Hypothesis:** Quantization acts as a bottleneck that simplifies the internal representation.
*   **Rank Retention Ratio:** $\frac{\text{Rank}_{Target}}{\text{Rank}_{Base}}$
    *   **$\\approx 1.0$:** Perfect preservation of structural complexity.
    *   **$< 1.0$:** **Feature Collapse.** Distinct concepts have merged into a lower-dimensional subspace.
    *   **$> 1.0$:** **Noise Inflation.** Quantization noise has "whitened" the feature space, artificially increasing orthogonality.

*Reference:* Roy, O., & Vetterli, M. (2007). "The effective rank: A measure of effective dimensionality." *European Signal Processing Conference*.

---

## 3. Spectral Decay (Singular Value Spectrum)

### Definition
A log-log plot of the singular values of the feature matrix, sorted in descending order.

### Scientific Justification
This visualizes the "long tail" of the feature space.
*   **The Head (Large values):** Represents the dominant, most important features.
*   **The Tail (Small values):** Represents rare, subtle features.
*   **Interpretation:** If the Target SAE's tail "lifts up" significantly above the Baseline's, it indicates a **Noise Floor**. The quantization error makes it impossible for the SAE to distinguish between rare features and random noise.

---

## 4. Mode Collapse: Unique Feature Utilization

### Definition
The number of *unique* Target features that are identified as the "best match" for at least one Baseline feature.

$$ \text{Utilization} = \frac{\text{Count}(\text{Unique Best Matches in Target})}{\text{Total Baseline Features}} $$

### Scientific Justification
A naive high recall score could be achieved if 100 different Baseline features (e.g., "Pug", "Husky", "Labrador") all map to the *same* generic Target feature ("Dog"). This metric detects such **Mode Collapse**.
*   **Low Utilization:** Quantization has blurred fine distinctions, causing multiple specific concepts to collapse into a single coarse feature.

---

## 5. Functional Consistency: Pre-TopK Jaccard & Activation Correlation

Geometric similarity (cosine sim of weights) is necessary but insufficient for interpretability. Two vectors can point in the same direction, but if quantization shifts the bias term or adds noise, the feature might **never fire** (Dead) or **always fire** (Dense). We measure functional consistency using two complementary metrics.

### A. Pre-TopK Jaccard Similarity

#### Definition
For TopK SAEs, the standard `encode()` method applies sparse selection (e.g., keeping only top-32 features). This is **too sensitive** for comparison—even tiny weight differences cause completely different top-32 selections.

Instead, we compare the **pre-TopK activations** (raw encoder outputs before sparsity selection) using a ReLU-like threshold:

$$ J_{pre}(A, B) = \frac{|\{i : a_i > 0\} \cap \{i : b_i > 0\}|}{|\{i : a_i > 0\} \cup \{i : b_i > 0\}|} $$

where $a_i$ and $b_i$ are the raw encoder outputs for the $i$-th feature.

#### Scientific Justification
This measures the overlap of features that **would activate** under ReLU, independent of TopK selection. It answers: "Do the SAEs represent similar concepts?"

*   **High Pre-TopK Jaccard (>0.8):** The underlying feature representations are similar.
*   **Low Pre-TopK Jaccard:** Quantization has fundamentally changed which concepts the SAE represents.

#### Why Not Post-TopK Jaccard?
With TopK-32, the SAE selects only 32 features from ~20,000+ positive candidates. Even if 99% of features would activate similarly, small magnitude differences cause different top-32 selections, resulting in near-zero Jaccard. Pre-TopK Jaccard avoids this sensitivity.

### B. Activation Magnitude Correlation

#### Definition
For geometrically matched feature pairs, we compute the Spearman correlation of their activation magnitudes across all tokens:

$$ \rho = \text{Spearman}(a_{base}, a_{target}) $$

#### Scientific Justification
This captures **how strongly** features activate, not just whether they activate. Two SAEs might have different thresholds but still rank tokens similarly by activation strength.

*   **High Correlation (>0.6):** When a concept fires strongly in the base SAE, it also fires strongly in the quantized SAE.
*   **Low Correlation:** **Magnitude Drift.** The relative importance of features has changed.

### C. Weighted Jaccard (Supplementary)

#### Definition
Instead of binary overlap, weight by activation magnitude:

$$ J_w = \frac{\sum_i \min(a_i, b_i)}{\sum_i \max(a_i, b_i)} $$

This measures "how much" features overlap, not just "whether" they overlap.

---

## 6. Sparsity Shift ($L_0$ Ratio)

### Definition
The ratio of the average number of active features per token ($L_0$) between the Target and Baseline SAEs.

$$ \text{Ratio} = \frac{L_0(\text{Target})}{L_0(\text{Base})} $$

### Scientific Justification
*   **Hypothesis:** Quantization noise acts as a "bias lift," causing many features to activate weakly when they should be off.
*   **Interpretation:**
    *   **Ratio $\\gg 1.0$:** **Polysemantic Drift.** The quantized SAE is "denser" and less specific. It activates for more tokens, reducing interpretability.
    *   **Ratio $\\approx 1.0$:** Sparsity is preserved.

---

## Guide to Charts

### Precision vs. Recall Scatter Plot (`*_precision_recall.png`)
*   **Ideal:** Top-right corner (1.0, 1.0).
*   **Evaluation:**
    *   **Cluster at (0.98, 0.98):** Strong evidence that quantization preserves interpretability.
    *   **Drift Left:** Loss of recall (features vanished).
    *   **Drift Down:** Loss of precision (learning noise).

### Rank Retention Bar Chart (`*_rank_retention.png`)
*   **Ideal:** Bars close to the red dashed line (1.0).
*   **Evaluation:**
    *   **Bar < 1.0:** The quantized model is "dumber" (less complex) than the baseline.
    *   **Bar > 1.0:** The quantized model is "noisier" (higher entropy) than the baseline.

### Spectral Decay Plot (`*_spectral_decay.png`)
*   **Ideal:** Colored lines perfectly overlapping the black dashed line.
*   **Evaluation:** Divergence at the right side (tail) shows where quantization noise overwhelms subtle features.

### Functional Consistency Bar Chart (`*_functional_jaccard.png`)
*   **Metrics Shown:** Pre-TopK Jaccard, Activation Correlation, Weighted Jaccard
*   **Ideal:** All bars close to 1.0.
*   **Evaluation:**
    *   **Pre-TopK Jaccard > 0.8:** Strong functional equivalence.
    *   **Activation Correlation > 0.6:** Features fire with similar relative strengths.
    *   **Divergence between metrics:** May indicate specific failure modes (e.g., threshold shift vs. magnitude drift).

### Global Box Plots (`global_*.png`)
*   **Purpose:** Aggregates metrics across all models (1B to 70B) to show systematic trends.
*   **Interpretation:** If the "Int4" box is significantly lower than "FP16", the degradation is universal, not model-specific.
