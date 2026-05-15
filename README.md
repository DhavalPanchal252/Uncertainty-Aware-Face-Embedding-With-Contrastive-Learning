# Uncertainty-Aware Face Embedding With Contrastive Learning (UCFace & QA-UCFace)

This repository contains the implementation of **UCFace** (Uncertainty-Aware Face Embedding With Contrastive Learning for Open-Set Evaluation) as proposed by Ahn et al. (IEEE TIFS 2024), along with a novel extension: **Quality-Adaptive Weighted Contrastive Learning (QA-UCFace)**.

## Overview

Open-set face recognition in unconstrained, low-resolution, and degraded conditions presents a significant challenge. Mainstream softmax-based face recognition models struggle with the "open-set discrepancy" between prototype-based training and pair-based inference.

**UCFace** addresses this by:
1. Treating each anchor embedding as a probabilistic **von Mises-Fisher (vMF) distribution**.
2. Using the **feature norm** ($L_2$-norm) as an inverse proxy for image uncertainty (i.e., higher norm = lower uncertainty/higher quality).
3. Utilizing an **Uncertainty-Aware Contrastive Loss** (InfoNCE-based) that modulates similarity scores based on the vMF distribution's concentration parameter, enhancing discriminative structure for open-set scenarios.

## Proposed Extension: Quality-Adaptive Weighted Contrastive Learning (QA-UCFace)

While UCFace implicitly modulates pairwise similarity scores, extremely degraded samples still contribute equally to the batch-level loss aggregation, potentially destabilizing training with residual noise. 

To overcome this, we propose **QA-UCFace**, which introduces an explicit sample weighting mechanism at the batch loss aggregation layer:

1. **Batch-Normalised Feature Norm**: The feature norms are normalized relative to the batch's mean and standard deviation to calibrate weights relative to the batch's quality distribution.
2. **Sigmoid Quality Weights**: The normalized norm is mapped to a bounded weight $\omega_i \in (0, 1)$ via a sigmoid function. High-quality images receive weights approaching $1$, while severely degraded images receive weights approaching $0$.
3. **Weighted Batch Loss Aggregation**: The batch-averaged contrastive loss is computed as a quality-weighted average, effectively suppressing the gradient contribution of severely corrupted samples.

The final overall training objective combines the classification loss (e.g., ArcFace) with our weighted uncertainty-aware contrastive loss.

## Implementation Details

The complete pipeline is implemented and evaluated on the **CelebA dataset** using a **ResNet-50** backbone. 
The core implementation and evaluation are available in the included Jupyter Notebook: `UCFace_Implementation.ipynb`.

### Key Findings
- **Feature Norm as Quality Proxy**: Analysis confirms that the $L_2$-norm correlates strongly with image recognizability, exhibiting a meaningful quality distribution even without explicit quality supervision.
- **Verification Performance**: The integration of the uncertainty-aware contrastive loss increases the cosine similarity gap between positive and negative pairs by ~47% on the CelebA validation set.
- **Improved Clustering**: Visualizations (t-SNE) show that high-quality samples cluster tightly near identity centers, while low-quality samples are appropriately positioned at cluster peripheries.

## References
* Ahn, K., Lee, S., Han, S., Low, C. Y., & Cha, M. (2024). Uncertainty-Aware Face Embedding With Contrastive Learning for Open-Set Evaluation. *IEEE Transactions on Information Forensics and Security*.