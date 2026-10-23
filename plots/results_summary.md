# Evaluation Results Summary

Evaluation Date: 2025-12-02 12:21:30

## Accuracy at epsilon = 0.03

| Model | Natural (%) | FGSM (%) | PGD (%) | One-Pixel (%) |
|-------|-------------|----------|---------|---------------|
| Default (No Defense) | 55.18 | 10.45 | 6.93 | 52.80 |
| FGSM Trained | 54.00 | 23.83 | 21.09 | 54.60 |
| PGD Trained | 46.09 | 27.54 | 25.20 | 45.20 |
| Mixed Trained | 46.58 | 23.73 | 22.46 | 44.60 |
| FGSM Pruned | 58.98 | 14.26 | 10.84 | 55.40 |
| PGD Pruned | 54.10 | 16.11 | 13.28 | 52.80 |
| Mixed Pruned | 54.69 | 15.72 | 12.70 | 51.40 |
| PGD Pruned + Rand | 50.51 | 19.59 | 17.91 | N/A |
| Ensemble | 51.46 | 28.22 | 27.34 | 51.60 |
| Ensemble + Rand | 48.66 | 28.98 | 28.19 | N/A |
