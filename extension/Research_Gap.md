#Research Gap: Clever Hans Detection in Medical Imaging

**Course:** DS357 Explainable AI
**Phase 3 Extension — Anders et al. 2021 (arXiv 2106.13200)**

## Research Gap

Anders et al. (2021) demonstrate the Zennit/CoRelAy/ViRelAy (ZCV) framework for
detecting "Clever Hans" classifiers on natural image benchmarks (ImageNet, PASCAL VOC).
A significant gap exists: **the framework has not been validated on medical imaging**,
which represents the highest-stakes domain for explainability and is well documented
to suffer from spurious correlations.

Zech et al. (2018) showed that chest X-ray classifiers trained on multi-hospital data
learned institution-specific scanner artefacts rather than pathology, achieving
near-zero generalisation to unseen hospitals despite high in-distribution accuracy.
Oakden-Rayner et al. (2020) further demonstrated that such "hidden stratification"
causes clinically meaningful failures invisible to standard evaluation metrics. These
findings constitute a direct medical analogue of the Clever Hans phenomenon described
by Lapuschkin et al. (2019).

## Proposed Extension

We apply the full ZCV pipeline to a chest X-ray classification task using the
PneumoniaMNIST dataset (Kermany et al. 2018, via MedMNIST; Yang et al. 2023).
A VGG-16-BN model is fine-tuned for binary classification (normal vs. pneumonia),
and LRP heatmaps are computed using the EpsilonGammaBox composite (identical to
Listing 1 of the original paper). SpRAy spectral clustering is then applied to
identify clusters of images sharing similar attribution patterns. We additionally
introduce a quantitative cluster quality metric — Border Relevance Ratio (BRR) —
to objectively identify artefact-focused clusters without manual inspection.

## Hypothesis

At least one SpRAy cluster will show disproportionate relevance at image borders
and corners — regions corresponding to scanner acquisition artefacts and burned-in
text markers — rather than over lung parenchyma where true pathology resides.
This would confirm that Clever Hans behaviour occurs in the medical domain and
that the ZCV framework can detect it without ground-truth artefact labels.



## Limitations and Future Work

The current pipeline has a known robustness limitation: t-SNE embedding is
sensitive to hyperparameters (perplexity, learning rate) and produces
non-deterministic layouts across runs even with a fixed seed. Future work could
improve pipeline stability by replacing t-SNE with UMAP (McInnes et al. 2018),
which better preserves global structure, and replacing k-means with HDBSCAN for
density-based clustering that does not require specifying k in advance.

## References

- Anders et al. 2021. arXiv 2106.13200.
- Lapuschkin et al. 2019. "Unmasking Clever Hans predictors." Nature Commun. 10, 1096.
- Zech et al. 2018. "Variable generalization performance detecting pneumonia."
  PLOS Medicine 15(11).
- Oakden-Rayner et al. 2020. "Hidden stratification causes clinically meaningful
  failures." NPJ Digital Medicine 3, 216.
- Bach et al. 2015. "On Pixel-Wise Explanations." PLOS ONE 10(7).
- McInnes et al. 2018. "UMAP: Uniform Manifold Approximation and Projection."
  arXiv 1802.03426.
- Yang et al. 2023. "MedMNIST v2." Scientific Data 10, 41.
- Kermany et al. 2018. "Identifying Medical Diagnoses by Image-Based Deep
  Learning." Cell 172(5):1122–1131.
