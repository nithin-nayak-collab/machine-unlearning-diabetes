Machine unlearning Paradigms for Responsible Ai  A comparative study on diabetes prediction 
Overview
This project implements and compares three machine unlearning techniques on the PIMA Indians Diabetes Dataset to address privacy concerns and GDPR compliance in healthcare ML systems.
Machine unlearning allows trained models to selectively forget specific data points without full retraining, supporting the Right to be Forgotten in Responsible AI.

## Techniques Compared

| Method | Description |
|--------|-------------|
| **SISA Training** | Sharded, Isolated, Sliced, Aggregated training |
| **LoRA Unlearning** | Low-Rank Adaptation for selective forgetting |
| **Full Retraining** | Gold standard baseline |

---

## Key Results
| Method | Accuracy | F1 Score |
|--------|----------|----------|------|
| Original Model | 81.17% | 0.8121 | 
| SISA Unlearning | 79.22% | 0.7945 |
| LoRA Unlearning | 79.87% |0.7983 |
| Full Retraining | 76.62% |0.7640 |

**Winner: SISA Unlearning**
- Minimal accuracy drop from original
- Fastest unlearning time
- MIA verification proves forget set removed
- GDPR compliant

---

Dataset
PIMA Indians Diabetes Dataset (UCI)

768 samples
8 clinical features
Binary classification (Diabetic / Non-Diabetic)


How to Run

Open Machine_Unlearning_Diabetes.ipynb in Google Colab
Upload diabetes.csv when prompted
Run all cells top to bottom (Cell 1 to Cell 11)
Results and graphs generate automatically


Requirements

scikit-learn
xgboost
imbalanced-learn
torch
pandas
numpy
matplotlib
seaborn


Methodology

Train original model on full dataset
Select forget set (10% of training data)
Apply SISA, LoRA and Full Retraining unlearning
Verify via Membership Inference Attack (MIA)
Compare accuracy, F1 score, speed and MIA gap


Responsible AI Contribution
This work addresses:

GDPR Right to be Forgotten
Healthcare data privacy
Efficient model updates without full retraining
