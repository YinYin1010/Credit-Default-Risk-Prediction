# Explainable and Cost-Sensitive Credit Default Risk Prediction

An end-to-end machine-learning project for credit-default prediction using the **UCI Credit Card Default** and **German Credit (Statlog)** datasets. The project combines dataset-specific exploratory analysis, feature engineering, model tuning, probability calibration, cost-sensitive threshold selection, SHAP explanations, methodological replication, reproducibility artefacts, and a Streamlit prediction prototype.

> **Research use only:** This project is an academic prototype and must not be used as the sole basis for real lending decisions.

## Project overview

Credit-risk models are often evaluated only by predictive accuracy. In practice, however, a useful lending model must also produce reliable probabilities, reflect the unequal costs of classification errors, and provide explanations that can be reviewed by human decision-makers.

This project therefore addresses four connected questions:

1. Which machine-learning model provides the strongest credit-risk discrimination for each dataset?
2. Do the selected models generalise from cross-validation to unseen test data?
3. How do probability calibration and cost-sensitive thresholds change operational decisions?
4. Which applicant characteristics contribute most strongly to the models' predictions?

The project treats **UCI Credit Card as the primary behavioural-scoring experiment** and **German Credit as an independent methodological replication**. Because the datasets have different schemas, products, populations, and prediction contexts, German Credit is not used as external validation of the fitted UCI model.

## Main capabilities

- Separate exploratory data analysis for the two credit datasets
- Dataset validation and human-readable recoding of categorical variables
- Dataset-specific financial and behavioural feature engineering
- Leakage-safe preprocessing within scikit-learn pipelines
- Stratified train/test splitting and cross-validated hyperparameter search
- Comparison of seven machine-learning models against a Dummy baseline
- Champion selection using mean cross-validated ROC-AUC
- Held-out ROC, precision-recall, confusion-matrix, and calibration evaluation
- Sigmoid probability calibration for the German Credit champion
- Hypothetical UCI and documented German cost-sensitive analyses
- Global and local SHAP explanations for both champion models
- Cross-dataset comparison of model-family rankings using Kendall's tau
- Persistent experiment outputs and a reproducibility manifest in Google Drive
- Streamlit prototype with dataset-specific predictions and local SHAP explanations

## Datasets

| Dataset | Modelling context | Rows | Default rate | Raw predictors | Engineered predictors | Train/test rows |
|---|---|---:|---:|---:|---:|---:|
| UCI Credit Card Default | Behavioural scoring of existing cardholders | 30,000 | 22.12% | 23 | 40 | 24,000 / 6,000 |
| German Credit (Statlog) | Application scoring of credit applicants | 1,000 | 30.00% | 20 | 27 | 800 / 200 |

### Target definitions

- **UCI Credit Card:** `default.payment.next.month` is renamed to `default`; `1` represents default and `0` represents non-default. The identifier `ID` is removed.
- **German Credit:** the original `credit_class` uses `1 = good` and `2 = bad`. These values are recoded as `default = 0` and `default = 1`, respectively.

The German dataset's documented cost matrix assigns a cost of **5** to a bad applicant predicted as good (false negative) and **1** to a good applicant predicted as bad (false positive).

## Data preparation

### Cleaning

- UCI `EDUCATION` codes `0`, `5`, and `6` are consolidated into `other_or_unknown`.
- UCI `MARRIAGE = 0` is consolidated into `other_or_unknown`.
- German symbolic codes are mapped to human-readable categories using the dataset documentation.
- Targets, row counts, missing values, duplicates, and binary-class validity are checked before modelling.

### Feature engineering

UCI features capture payment delinquency, billing exposure, repayment behaviour, utilisation, and available credit. Examples include:

- `DELAY_MONTH_COUNT`
- `SEVERE_DELAY_MONTH_COUNT`
- `MAX_REPAYMENT_STATUS`
- `TOTAL_BILL_AMT`
- `BILL_AMT_VOLATILITY`
- `TOTAL_PAY_AMT`
- `CURRENT_UTILIZATION`
- `PAYMENT_TO_BILL_RATIO`
- `CREDIT_HEADROOM`

German Credit features include:

- `credit_per_month`
- `duration_years`
- `credit_amount_per_age`
- `installment_burden_proxy`
- `age_band`
- `duration_band`
- `credit_amount_band`

### Preprocessing

The data are divided using a stratified **80/20 train/test split** with `random_state=42`. Preprocessing is fitted inside each model pipeline:

- Numeric variables: median imputation and standardisation
- Categorical variables: most-frequent imputation and one-hot encoding
- Unknown categories at prediction time: ignored safely by the encoder

Keeping preprocessing inside the cross-validation pipeline prevents information from the validation or test data leaking into model fitting.

## Models and experimental design

The following models are evaluated independently on each dataset:

1. Dummy classifier
2. Logistic Regression
3. Support Vector Machine
4. K-Nearest Neighbours
5. Multilayer Perceptron
6. Random Forest
7. XGBoost
8. LightGBM

The recorded experiment used:

- `FAST_MODE=True`
- 3-fold stratified cross-validation
- 8 sampled hyperparameter configurations per non-baseline model
- Randomized search with refitting by ROC-AUC
- A fixed random seed of 42
- An untouched 20% holdout test set

Setting `FAST_MODE=False` changes the notebook to 5-fold cross-validation and 25 search iterations. The numerical results below are from the recorded **fast-mode run**.

Alongside ROC-AUC, the notebook reports PR-AUC, accuracy, balanced accuracy, precision, recall, specificity, F1-score, Matthews correlation coefficient, Brier score, and confusion-matrix counts. The holdout test set is used for final evaluation only; it is not used to choose the champion model.

## Model comparison

### UCI Credit Card

| Model | CV ROC-AUC | CV SD | Test ROC-AUC | Test PR-AUC |
|---|---:|---:|---:|---:|
| **XGBoost** | **0.7888** | 0.0045 | **0.7835** | 0.5621 |
| LightGBM | 0.7877 | 0.0044 | 0.7840 | 0.5640 |
| Random Forest | 0.7867 | 0.0030 | 0.7800 | 0.5613 |
| MLP | 0.7787 | 0.0046 | 0.7745 | 0.5556 |
| Logistic Regression | 0.7675 | 0.0054 | 0.7563 | 0.5191 |
| SVM | 0.7643 | 0.0065 | 0.7541 | 0.4801 |
| KNN | 0.7636 | 0.0091 | 0.7607 | 0.5209 |
| Dummy | 0.5000 | 0.0000 | 0.5000 | 0.2212 |

### German Credit

| Model | CV ROC-AUC | CV SD | Test ROC-AUC | Test PR-AUC |
|---|---:|---:|---:|---:|
| **Random Forest** | **0.7973** | 0.0190 | **0.7876** | 0.6530 |
| XGBoost | 0.7891 | 0.0255 | 0.7874 | 0.6618 |
| LightGBM | 0.7885 | 0.0216 | 0.7975 | 0.6722 |
| SVM | 0.7882 | 0.0144 | 0.7800 | 0.6170 |
| MLP | 0.7858 | 0.0262 | 0.8120 | 0.7045 |
| KNN | 0.7759 | 0.0155 | 0.7417 | 0.6091 |
| Logistic Regression | 0.7593 | 0.0294 | 0.8086 | 0.6458 |
| Dummy | 0.5000 | 0.0000 | 0.5000 | 0.3000 |

Some German models obtained higher holdout scores than Random Forest. Random Forest remains the champion because model selection was performed using training-only cross-validation; selecting a model retrospectively from test performance would leak information from the holdout set.

## Champion models

| Dataset | Champion | CV ROC-AUC | Test ROC-AUC | CV-test gap | Test accuracy | Test MCC |
|---|---|---:|---:|---:|---:|---:|
| UCI Credit Card | XGBoost | 0.7888 | 0.7835 | 0.0053 | 0.8208 | 0.4057 |
| German Credit | Random Forest | 0.7973 | 0.7876 | 0.0097 | 0.7250 | 0.3808 |

The small CV-to-test gaps provide evidence of reasonable generalisation with limited overfitting in the recorded splits. The results also demonstrate that the strongest algorithm is dataset-dependent rather than universal.

## Probability calibration

The German Random Forest is calibrated with sigmoid calibration using stratified cross-validation. Calibration reduced the Brier score from **0.1796** to approximately **0.159–0.160**, indicating that the predicted probabilities became more consistent with observed outcomes. ROC-AUC remained broadly unchanged because calibration improves probability reliability rather than the underlying ranking of applicants.

The calibrated German model is used for operational threshold selection, while the original Random Forest remains the explanation model for SHAP.

## Cost-sensitive decision analysis

### UCI Credit Card: hypothetical scenarios

The UCI dataset does not provide an official error-cost matrix. The notebook therefore reports sensitivity analyses under three hypothetical false-negative to false-positive cost ratios. Thresholds are selected using training-only out-of-fold predictions.

| Assumed FN:FP cost | Selected threshold | Test recall | Test specificity | Test cost per client |
|---|---:|---:|---:|---:|
| 2:1 | 0.36 | 0.483 | 0.900 | 0.3065 |
| 5:1 | 0.15 | 0.797 | 0.592 | 0.5430 |
| 10:1 | 0.10 | 0.909 | 0.359 | 0.7010 |

As the assumed cost of a missed default increases, the selected threshold falls and recall rises. Because each row applies a different cost ratio, its absolute cost value should not be compared directly with the other scenarios.

### German Credit: documented cost matrix

The documented 5:1 cost matrix produced a training-only out-of-fold threshold of **0.20**, with an out-of-fold cost of **0.501 per applicant**.

| Decision rule | Threshold | Accuracy | Precision | Recall | Specificity | FP | FN | Test cost/applicant |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| Raw model, standard | 0.50 | 0.725 | 0.535 | 0.633 | 0.764 | 33 | 22 | 0.715 |
| Calibrated model, standard | 0.50 | 0.765 | 0.638 | 0.500 | 0.879 | 17 | 30 | 0.835 |
| **Calibrated, cost-optimised** | **0.20** | 0.620 | 0.431 | **0.833** | 0.529 | 66 | **10** | **0.580** |

Lowering the calibrated threshold from 0.50 to 0.20 increased default recall from 0.500 to 0.833 and reduced false negatives from 30 to 10. This benefit came with more false positives and lower accuracy. The appropriate threshold therefore depends on whether the institution prioritises missed-default cost, approval precision, or overall classification accuracy.

## SHAP explainability

SHAP is applied independently to each champion model. In the recorded fast-mode run, a stratified sample of up to **500 test observations** is explained using `TreeExplainer`. The notebook produces:

- Global SHAP beeswarm plots
- Global mean absolute SHAP bar plots
- Local waterfall explanations
- Transformed-feature importance tables
- Grouped source-feature importance tables
- Ranked local contributions for five example applicants

One-hot encoded features are regrouped into their original source variables before reporting the main global importance rankings.

### Leading grouped features

| Rank | UCI Credit Card | Mean \|SHAP\| | German Credit | Mean \|SHAP\| |
|---:|---|---:|---|---:|
| 1 | `PAY_0` | 0.3229 | `checking_status` | 0.1000 |
| 2 | `SEVERE_DELAY_MONTH_COUNT` | 0.2831 | `savings_status` | 0.0318 |
| 3 | `DELAY_MONTH_COUNT` | 0.2136 | `credit_history` | 0.0297 |
| 4 | `CREDIT_HEADROOM` | 0.1929 | `purpose` | 0.0227 |
| 5 | `BILL_AMT1` | 0.0778 | `duration_years` | 0.0214 |

The UCI model relies mainly on recent and repeated repayment delays, remaining credit capacity, billing exposure, and payment behaviour. The German model is driven primarily by checking-account status, savings, credit history, loan purpose, and repayment duration.

SHAP explains model behaviour rather than causal relationships. Mean absolute SHAP values show the magnitude of model reliance but do not by themselves indicate whether a feature raises or lowers default risk.

## Cross-dataset methodological replication

The ordering of the eight model families across the two datasets achieved **Kendall's tau = 0.643** with **p = 0.031**. This indicates moderately strong, statistically significant agreement in model-family rankings. Tree-based ensembles occupied the leading positions in both experiments, although XGBoost ranked first for UCI and Random Forest ranked first for German Credit.

This analysis evaluates whether the modelling methodology produces similar algorithm rankings; it does not test whether one fitted model transfers between the two datasets.

## Streamlit prototype

The notebook contains a Streamlit application that:

- Allows selection between UCI Credit Card and German Credit
- Builds input controls from a stored feature schema
- Displays the selected champion, threshold, and calibration status
- Returns default probability and the corresponding risk classification
- Generates a local SHAP contribution chart on request
- Uses red contributions for increased model output and blue for decreased output

The deployment bundle records:

| Dataset | Decision model | Threshold | Calibrated | Input features |
|---|---|---:|---|---:|
| UCI Credit Card | XGBoost | 0.50 | No | 40 |
| German Credit | Random Forest | 0.20 | Yes | 27 |

The deployment cells expect an existing `credit_models.joblib` bundle in:

```text
/content/drive/MyDrive/Credit_Default_Streamlit/
```

They generate `app.py`, run Streamlit on port 8501, verify the local HTTP response, and expose the session temporarily using a Cloudflare Quick Tunnel. The generated `trycloudflare.com` address is temporary and should not be placed in the README as a permanent application URL.

For a self-contained GitHub deployment, export `app.py`, add the code used to build `credit_models.joblib`, and include a pinned `requirements.txt`.

## Output structure

The notebook saves **111 experiment artefacts** to Google Drive:

```text
/content/drive/MyDrive/Credit_Risk_Prediction/
├── dataset/
└── outputs/
    ├── uci_primary/
    │   ├── processed/
    │   ├── models/
    │   ├── results/
    │   ├── figures/
    │   └── shap/
    ├── german_replication/
    │   ├── processed/
    │   ├── models/
    │   ├── results/
    │   ├── figures/
    │   └── shap/
    └── cross_experiment_comparison/
```

Saved artefacts include model-ready data, fitted pipelines, cross-validation results, test predictions, classification reports, ROC and PR curves, confusion matrices, calibration figures, threshold-search tables, SHAP outputs, an artefact index, and a JSON reproducibility manifest.

## Running the notebook

The project is designed for **Google Colab**.

1. Open `Credit_Risk_Prediction_UCI_German.ipynb` in Colab.
2. Place the following files in `/content/drive/MyDrive/Credit_Risk_Prediction/dataset/`:
   - `UCI_Credit_Card.csv`
   - `german.data`
3. If the files are not found, the notebook will open an upload prompt and save the selected file under the required standardised name.
4. Run the notebook cells in order.
5. Review the generated artefacts under the Google Drive output folders.

Install missing dependencies in Colab with:

```bash
pip install pandas scikit-learn xgboost lightgbm shap imbalanced-learn joblib scipy matplotlib seaborn streamlit
```

The recorded execution environment used:

| Component | Version |
|---|---:|
| Python | 3.13.15 |
| pandas | 2.2.3 |
| scikit-learn | 1.6.1 |
| XGBoost | 3.4.1 |
| LightGBM | 4.6.0 |
| SHAP | 0.52.0 |

Model training can take considerable time, particularly for SVM and large hyperparameter searches. The notebook saves fitted pipelines and result files to Google Drive, but rerunning the training cells will train the models again unless explicit loading or caching logic is added.

## Recommended GitHub structure

```text
credit-risk-prediction/
├── Credit_Risk_Prediction_UCI_German.ipynb
├── app.py                         # export from the deployment cell
├── requirements.txt
├── README.md
├── LICENSE
└── .gitignore
```

Datasets, research PDFs, generated outputs, temporary tunnel logs, and model binaries should normally be excluded from Git unless their licences and file sizes permit redistribution.

Suggested `.gitignore` entries:

```gitignore
dataset/
outputs/
*.joblib
*.pkl
*.log
.ipynb_checkpoints/
__pycache__/
```

## Limitations and responsible use

- Both datasets are historical benchmarks and may not represent current lending populations.
- German Credit contains only 1,000 observations, increasing uncertainty in model and threshold estimates.
- The reported search used three-fold fast mode; a final study should repeat the analysis with more folds, more search iterations, and repeated seeds.
- Test results are based on one stratified split and should be supported with external or temporal validation where possible.
- The UCI cost scenarios are hypothetical and should not be interpreted as actual institutional costs.
- The German threshold conclusion depends on the dataset's documented 5:1 cost assumption.
- Calibration, discrimination, and classification utility measure different properties and should be assessed separately.
- SHAP values represent model associations, not causal effects.
- Variables such as sex, marriage, and personal status require explicit fairness and disparate-impact assessment before deployment.
- The Streamlit application is a demonstration prototype and does not provide production security, monitoring, governance, or audit controls.

## Conclusion

The project demonstrates that credit-risk modelling benefits from combining predictive performance with probability quality, operational decision costs, and transparent explanations. XGBoost was selected for UCI Credit Card with a test ROC-AUC of 0.7835, while Random Forest was selected for German Credit with a test ROC-AUC of 0.7876. Their small cross-validation-to-test gaps suggest reasonable generalisation in the recorded experiments.

SHAP analysis showed that repayment behaviour, delinquency frequency, available credit, liquidity, credit history, loan purpose, and repayment duration were the models' principal drivers. For German Credit, calibration improved probability reliability and a cost-sensitive threshold of 0.20 substantially reduced missed defaults. These gains came with more false alarms, illustrating why model deployment requires an explicit choice of operational priorities rather than reliance on a universal 0.50 threshold.

Overall, the project provides a reproducible framework for comparing dataset-specific credit-risk models while integrating discrimination, calibration, cost sensitivity, explainability, and methodological replication.

## References and data licensing

The notebook contains the complete APA-style literature reference list used for the project. The datasets and research articles are not included in this repository. Users should obtain them from their official sources and comply with their respective licences and terms of use.

## Licence

No software licence is assumed by this README. Add an appropriate licence, such as MIT, Apache-2.0, or another licence consistent with the intended use of the code and third-party dependencies.
