# DEVIATIONS — DNDCMH

| ID | Item | Paper | Ours | Reason |
|----|------|-------|------|--------|
| D1 | YTF | Train/eval on YouTube Faces | Omitted | Dataset not available under `projects/datasets/` |
| D2 | LFW | Attribute→image MAP on LFW | Deferred | No official 40-attr labels co-located; CelebA is primary |
| D3 | NECD | Belief-propagation on Tanner graph for designed (N,K) ECC | Residual MLP soft decoder trained with ≤e bit-flip noise | Faithful ECC matrices not released; MLP preserves e≥m training idea |
| D4 | Backbone | Paper Image-CNN (details in PDF) | ResNet-18 + tanh hash head | Standard public backbone; same role |
| D5 | Hash length | Ablations 31/63/127 | Default 63 (paper main setting) | Matches Table 1/2 primary column |
| D6 | Epochs | Not fully specified | ADCMH 20 + NECD 10 | Practical full-train budget on CelebA |
