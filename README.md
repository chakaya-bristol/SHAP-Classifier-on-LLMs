# Explaining LLM Failures on Public Health Claims Using SHAP

A lightweight, black-box pipeline for understanding **when and why** LLMs fail on health claim verification, using a surrogate XGBoost classifier and SHAP explanations.

```
Health Claims → LLM (4-class) → Correct/Wrong → XGBoost → SHAP Explanations
 (PUBHEALTH)   (flan-t5-base)    (binary label)  (surrogate)  (global + local)
```

## Overview
 
Large language models are increasingly used for health-related tasks, yet their failures remain opaque. This project proposes a pipeline to explain LLM correctness on public health claims using post-hoc explainable AI. 500 claims from the PUBHEALTH dataset are classified by flan-t5-base into four veracity categories (true, false, mixture, unproven), and a surrogate XGBoost classifier is trained to predict correctness from claim-level features. SHAP analysis reveals that the model's failures are systematic and predictable.

## Key Findings
 
- **0% accuracy** on mixture and unproven claims — the LLM cannot express epistemic uncertainty
- **Assertiveness bias**: mixture claims are labelled "true" 60% of the time; unproven claims are labelled "false" 74% of the time
- **Veracity category** is the single strongest SHAP predictor of LLM failure (mean |SHAP| ≈ 0.40)
- **Textual complexity** (claim length, word sophistication) independently predicts failure
- Surrogate classifier achieves **AUC-ROC of 0.731**, confirming failures are systematic, not random


