# Explainable Credit Card Default Prediction

**Machine learning and deep learning comparison with SHAP explanations, local LLM narration, and a Streamlit research demo**

This project estimates the probability that a credit card client will default on payment in the following month. It compares conventional machine learning with neural models, examines the effect of probability calibration and decision thresholds, and translates selected SHAP attributions into plain-language explanations. The analysis uses only the UCI Default of Credit Card Clients dataset.

> **Research use only.** The dataset describes historical clients in Taiwan in 2005. The model is a portfolio demonstration and must not be used to approve, deny, price, or otherwise make a real credit decision.

## Project background

Credit default prediction is a classification problem with an important probability-estimation component. An institution may care about the ranking of clients by estimated risk, the reliability of those probabilities, and the consequences of different decision thresholds. A useful research comparison therefore requires more than a single accuracy score. It should also examine class imbalance, calibration, error tradeoffs, and whether explanations accurately reflect the fitted model.

This project extends a conventional credit-risk workflow with tabular deep learning, model-agnostic SHAP, and an optional local language model. The language model receives a restricted set of prediction and SHAP evidence. Its output is checked against that evidence before it is treated as an acceptable explanation.

## Problem statement

Given a client's credit limit, demographics, recent repayment status, bill amounts, and payment amounts, estimate the probability of default in the next month. Compare model families under a consistent data split. Select a model using validation performance, assess it once on a held-out test set, and explain individual predictions without presenting model associations as causes.

## Research and business questions

1. How do conventional classifiers compare with tabular neural models when next-month default is the target?
2. Which model has the strongest validation precision-recall performance, and how does the selected model perform on held-out data?
3. Does post-training probability calibration improve the selected model's Brier score on the calibration set?
4. How do classification thresholds change the balance among precision, recall, and illustrative false-positive versus false-negative costs?
5. Which recorded features contribute most strongly to the fitted model's predictions according to SHAP?
6. Can a local language model describe selected SHAP evidence while preserving feature names, contribution directions, and reported probabilities?
7. Do descriptive performance checks differ across the recorded sex and age groups in the test sample?

## Data

The [UCI Default of Credit Card Clients dataset](https://archive.ics.uci.edu/dataset/350/default+of+credit+card+clients) contains **30,000 records** and **23 original predictors** after the client `ID` is excluded. It records credit card clients in Taiwan and covers repayment and billing information from April to September 2005. The target is `default.payment.next.month`, where `1` indicates default in the following month and `0` indicates no default. The observed default rate in this notebook is **22.12%**.

| Feature group | Variables | Role |
| --- | --- | --- |
| Credit and demographics | `LIMIT_BAL`, `SEX`, `EDUCATION`, `MARRIAGE`, `AGE` | Credit limit and recorded client characteristics |
| Repayment status | `PAY_0`, `PAY_2` to `PAY_6` | Monthly repayment status, from September back to April 2005 |
| Bill statements | `BILL_AMT1` to `BILL_AMT6` | Monthly statement balances |
| Previous payments | `PAY_AMT1` to `PAY_AMT6` | Monthly payment amounts |
| Outcome | `default.payment.next.month` | Next-month default indicator |

The notebook reports **zero missing cells** and **35 duplicate rows** in its initial quality summary. It does not automatically discard those duplicates. It removes `ID`, consolidates uncommon education and marital-status codes, and maps categorical codes to readable labels. Feature engineering adds repayment-history summaries, bill and payment aggregates, trends, utilization measures, a payment-to-bill ratio, and credit headroom. The model-ready table contains **40 predictors**.

**Data source:** Yeh, I. (2009). *Default of Credit Card Clients* [Dataset]. UCI Machine Learning Repository. [https://doi.org/10.24432/C55S3H](https://doi.org/10.24432/C55S3H). The dataset is distributed under CC BY 4.0.

## Study design

The notebook uses a stratified split with a fixed random seed of 42:

| Partition | Records | Purpose |
| --- | ---: | --- |
| Training | 18,000 | Fit models and tune conventional classifiers |
| Validation | 3,000 | Compare models and select the champion |
| Calibration | 3,000 | Fit the probability calibrator and select the F1 threshold |
| Test | 6,000 | Report held-out performance and descriptive checks |

Conventional classifiers use five-fold stratified cross-validation within the training partition and are compared on validation PR-AUC. Neural models use validation PR-AUC for training control and model comparison. They do not have the same five-fold cross-validation estimates in this notebook. The test set is used after the champion is selected. A separate test-set comparison of all fitted models is descriptive and is not used to choose the champion.

### Models

| Family | Implemented models |
| --- | --- |
| Baseline | Dummy classifier |
| Conventional ML | Logistic regression, random forest, XGBoost, LightGBM |
| Optional slower baselines | SVM and k-nearest neighbors, disabled by default |
| Deep learning | MLP, tabular ResNet, FT-Transformer, 1D-CNN, TabNet |

Numerical and categorical preprocessing is fitted on training data. The notebook uses imputation, signed-log transformation and scaling where appropriate, along with categorical encoding. The neural workflow uses its own training-fitted preprocessing. After champion selection, a logistic calibrator is fitted to clipped model logits on the calibration partition. The classification threshold is chosen by calibration-set F1.

### Evaluation

The selected model is assessed with PR-AUC, ROC-AUC, Brier score, precision, recall, F1, and Matthews correlation coefficient. The notebook also produces ROC and precision-recall curves, a calibration plot, a confusion matrix, hypothetical error-cost scenarios, and descriptive subgroup tables.

## Results from the saved notebook run

**LightGBM** was selected because it had the highest validation PR-AUC, **0.5468**. XGBoost followed closely at **0.5466**, and the MLP reached **0.5434**. These small validation differences should not be interpreted as evidence of a statistically significant ranking.

| Held-out metric for selected LightGBM model | Result |
| --- | ---: |
| PR-AUC | 0.5594 |
| ROC-AUC | 0.7797 |
| Brier score | 0.1350 |
| Precision | 0.5414 |
| Recall | 0.5373 |
| F1 | 0.5393 |
| Matthews correlation coefficient | 0.4092 |
| Calibration-selected F1 threshold | 0.32 |

On the **calibration partition**, the Brier score changed from **0.1693 before calibration** to **0.1307 after calibration**. The separate held-out test Brier score was **0.1350**. These values belong to different partitions and should not be interpreted as a direct before-and-after test-set comparison.

The hypothetical cost analysis considers false-negative to false-positive cost ratios of **1:1**, **5:1**, and **10:1**. These ratios are assumptions for sensitivity analysis. The dataset does not contain actual exposure, recovery, or financial loss amounts. The sex and age tables are descriptive checks that do not establish fairness or equal treatment.

## Explainability and language-model evaluation

Permutation SHAP explains calibrated probabilities using a sample of **30 training records** as background and **60 test records** for analysis. Categorical labels are encoded for the SHAP masker and decoded before model prediction. The notebook saves a global beeswarm plot, a local waterfall plot, and mean absolute feature attributions. In the saved run, `PAY_0` had the largest mean absolute SHAP value among the raw features. SHAP describes the behavior of the fitted model relative to the chosen background. It does not establish causal effects.

The optional local language model is **`Qwen/Qwen2.5-1.5B-Instruct`**. Its prompt receives the estimated default probability, the comparison threshold, and three leading SHAP factors. The notebook checks generated text for the requested feature names, contribution directions, completeness, and exact displayed percentages. A deterministic template remains available without an API key or language-model inference.

The saved notebook evaluated **26 selected SHAP cases**. Text was generated for all 26 without generation errors, but **only 5 of 26 passed every automated check**. The direction check passed in **7 of 26** cases. Numeric agreement passed in **24 of 26**. These results limit any claim that the generated explanations are reliably grounded. Automated checks also do not replace manual review for unsupported statements or readability.

## Streamlit demonstration

The notebook exports the champion model, preprocessing components where needed, calibrator, SHAP background, metadata, shared runtime code, and `app.py` to Google Drive. The application accepts client inputs, estimates next-month default probability, displays SHAP factors, and provides a template explanation. Local Qwen narration is optional and disabled unless `ENABLE_LOCAL_LLM=true` is set before launching Streamlit. Model weights are downloaded on first use and are not included in the exported ZIP.

The notebook can start Streamlit and print a temporary Cloudflare preview URL. That link ends when the runtime or tunnel stops. It is a development preview, not permanent hosting. For a lasting public demo, deploy the exported bundle on a suitable Streamlit host with enough memory for any enabled local LLM.

## Reproduce the analysis

1. Download the UCI dataset from the source linked above. Prepare a CSV named `UCI_Credit_Card.csv` that includes `ID` and the target column `default.payment.next.month`.
2. Open `CreditDefaultRisk-2.ipynb` in Google Colab. The setup cell installs the required Python packages.
3. Mount Google Drive. Place the CSV in `MyDrive/Credit_Risk_Prediction/dataset/`, or use the notebook's upload prompt when the file is not found.
4. Run the notebook from the beginning through model evaluation and export. Model training, permutation SHAP, and local LLM inference can take substantial time. A GPU is optional for neural training, subject to Colab availability.
5. Inspect the outputs under `MyDrive/Credit_Risk_Prediction/outputs/uci_primary/`. The deployment bundle is written under `MyDrive/Credit_Risk_Prediction/Credit_Default_Streamlit/`.

To run the exported application locally after extracting `credit_risk_streamlit_bundle.zip`:

```bash
pip install -r requirements.txt
streamlit run app.py
```

For local Qwen narration, set `ENABLE_LOCAL_LLM=true` in the application environment before launching it. The default template mode avoids the language-model download and associated memory demand.

## Limitations and responsible use

- The data reflects one historical setting in Taiwan. External validity to present-day clients, institutions, and regulatory environments is unknown.
- The notebook uses a random stratified split. It does not provide a prospective or time-based validation study.
- The measured cost scenarios are illustrative because actual loan losses and recovery amounts are unavailable.
- SHAP attributions depend on the fitted model and selected background data. They are associations rather than explanations of causal mechanisms.
- The subgroup tables are descriptive. A production fairness assessment would require a defined policy, appropriate protected-attribute governance, uncertainty analysis, and broader validation.
- The local LLM did not consistently preserve the supplied SHAP directions in the saved evaluation. Its prose requires manual review and must not be treated as a decision rationale.
- The Streamlit interface is a research demonstration. It has not been validated or governed for operational lending use.

## Project outputs

The notebook creates reproducible CSV summaries and figures for data quality, exploratory analysis, model comparison, calibration, threshold sensitivity, subgroup checks, SHAP, and LLM grounding checks. It also exports a ZIP containing the Streamlit application and the trained champion's inference artifacts. These files are generated in Google Drive when the notebook is run and are not assumed to be committed to this repository.
