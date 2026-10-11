# Titanic Classification Pipeline

End-to-end binary classification project for the Kaggle **Titanic - Machine Learning from Disaster** competition.

The project compares a small PyTorch MLP with several classical ML models, uses leakage-safe 5-fold stratified cross-validation, Optuna tuning, SHAP interpretation, and probability blending.

## Results

### Final public leaderboard

| Model | Public score |
|---|---:|
| **MLP baseline, threshold 0.60** | **0.78229** |
| MLP + tuned XGBoost blend | 0.77751 |
| Tuned XGBoost | 0.77033 |
| XGBoost baseline | 0.75598 |

The MLP remained the best public-leaderboard model, while the blend had the highest OOF accuracy. This is a useful example of the fact that CV improvements do not always transfer directly to a small external test set.

### Cross-validation summary

| Model | OOF Accuracy | F1 | ROC-AUC | Threshold |
|---|---:|---:|---:|---:|
| Logistic Regression | 0.8328 | 0.7616 | 0.8709 | 0.50 |
| Random Forest | 0.8215 | 0.7480 | 0.8743 | 0.50 |
| XGBoost baseline | 0.8361 | 0.7740 | 0.8780 | 0.50 |
| CatBoost baseline | 0.8283 | 0.7591 | 0.8783 | 0.50 |
| MLP baseline | 0.8294 | 0.7580 | 0.8785 | 0.50 |
| MLP with selected threshold | 0.8384 | 0.7584 | 0.8785 | 0.60 |
| Optuna XGBoost | **0.8462** | **0.7896** | **0.8799** | 0.50 |
| Optuna CatBoost | 0.8395 | 0.7776 | 0.8779 | 0.50 |
| MLP + tuned XGBoost blend | **0.8474** | — | — | 0.65 |

## Project structure

```text
ML-classification-problem/
├── classic_pipeline/
│   ├── __init__.py
│   ├── blend.py
│   ├── config.py
│   ├── data.py
│   ├── inference.py
│   ├── interpret.py
│   ├── main.py
│   ├── models.py
│   ├── train.py
│   ├── tune.py
│   └── utils.py
├── deep_learning_pipeline/
│   ├── __init__.py
│   ├── config.py
│   ├── data.py
│   ├── inference.py
│   ├── main.py
│   ├── model.py
│   ├── predict.py
│   ├── train.py
│   └── utils.py
├── notebooks/
│   └── EDA.ipynb
├── data/                 # ignored by git
├── checkpoints/          # ignored by git
├── artifacts/            # generated interpretation/blend artifacts
├── submissions/          # ignored by git
├── environment.yml
├── setup.sh
├── run.sh
└── README.md
```

## Data and EDA

The target is `Survived` with a roughly 61/39 class split. The main EDA covered missing values, class balance, survival rates across important categorical and numerical features, interaction tables, and numerical correlations.

Important raw columns:

- `Pclass` — passenger class
- `Sex`
- `Age`
- `SibSp` — siblings/spouses aboard
- `Parch` — parents/children aboard
- `Fare`
- `Cabin`
- `Embarked`
- `Name`

`PassengerId` is kept only for the submission. Raw `Name`, `Ticket`, and `Cabin` are not passed directly to the OHE pipelines.

## Feature engineering

The reusable preprocessing code creates:

- `FamilySize = SibSp + Parch + 1`
- `HasFamily`
- `Title` extracted from `Name`
  - `Mlle -> Miss`
  - `Ms -> Miss`
  - `Mme -> Mrs`
  - rare titles grouped into `Rare`
- `AgeGroup`
  - kid: `[0, 18)`
  - young adult: `[18, 35)`
  - adult: `[35, 55)`
  - senior: `55+`
- `HasCabin`
- `Deck` extracted from `Cabin`
- `Sex_Pclass`
- `Sex_AgeGroup`

Missing-value handling is fitted inside each fold:

- numerical features: median imputation + missing indicator + standardization
- categorical features: most-frequent imputation + one-hot encoding
- `AgeGroup` / `Deck`: explicit `Unknown` category where appropriate

All sklearn preprocessing is fitted only on the training portion of each fold and then applied to validation/test data.

## Deep learning pipeline

The PyTorch model is a compact MLP suitable for the very small Titanic dataset:

```text
input (~49 after OHE)
→ Linear(64)
→ BatchNorm
→ ReLU
→ Dropout(0.20)
→ Linear(32)
→ BatchNorm
→ ReLU
→ Dropout(0.20)
→ Linear(1)
```

Training setup:

- 5-fold `StratifiedKFold`, seed 42
- `BCEWithLogitsLoss`
- AdamW
- `ReduceLROnPlateau`
- early stopping by validation loss
- MPS on Apple Silicon when available
- fold-specific preprocessors and model checkpoints
- OOF probabilities for evaluation
- probabilities averaged across folds at inference time

Metrics:

- Accuracy
- Precision
- Recall
- F1
- ROC-AUC

A threshold analysis on OOF predictions selected `0.60` for the final MLP submission.

## Classical ML pipeline

Baseline models:

- Logistic Regression
- Random Forest
- XGBoost
- CatBoost

Logistic Regression, Random Forest, and XGBoost use the shared sklearn preprocessing pipeline. CatBoost uses native categorical features instead of OHE.

Each fold is saved as a complete reusable estimator, so inference can be reproduced without retraining.

### Optuna

Optuna was used for XGBoost and CatBoost with 20 trials per model. Every trial is evaluated with the same 5-fold stratified CV and optimizes OOF Accuracy at threshold `0.50`.

Best XGBoost parameters found:

```text
n_estimators: 887
learning_rate: 0.013473634779059902
max_depth: 5
min_child_weight: 2
subsample: 0.7794963304041234
colsample_bytree: 0.6052878399890287
reg_alpha: 0.04899718993341205
reg_lambda: 0.003193410903745766
```

Tuned XGBoost improved OOF Accuracy from `0.8361` to `0.8462` and public Kaggle score from `0.75598` to `0.77033`.

## Model interpretation

The tuned XGBoost model was inspected using:

- built-in XGBoost feature importance
- mean absolute SHAP importance
- SHAP beeswarm plot

The strongest signal came from the cluster of `Title`, `Sex`, `Pclass`, `Fare`, `Age`, family features, and cabin-related features. Interaction features such as `Sex_Pclass` were also useful.

The SHAP beeswarm additionally showed direction of influence: for example, `Title_Mr`, male sex, higher numerical `Pclass`, and very large families generally pushed predictions toward non-survival, while female sex and several higher-class/family-status signals pushed predictions toward survival.

## Blend

The final blend combines probabilities from the saved MLP and tuned XGBoost folds.

The OOF search selected:

```text
MLP weight:      0.15
XGBoost weight:  0.85
Threshold:       0.65
OOF Accuracy:    0.8474
```

The blend improved over tuned XGBoost on the Kaggle public leaderboard (`0.77751` vs `0.77033`) but did not beat the standalone MLP (`0.78229`).

Because PyTorch/MPS and XGBoost caused a native-runtime segmentation fault when used sequentially in one macOS process, `blend.py` generates MLP and XGBoost predictions in separate Python subprocesses and combines only the saved NumPy probabilities in the parent process.

## Quick start

### 1. Clone the repository

```bash
git clone https://github.com/Gloccium/ML-classification-problem.git
cd ML-classification-problem
```

### 2. Download Titanic data

With Kaggle CLI configured:

```bash
mkdir -p data
kaggle competitions download -c titanic -p data
unzip data/titanic.zip -d data
rm data/titanic.zip
```

Required files:

```text
data/train.csv
data/test.csv
```

### 3. Create the environment

```bash
chmod +x setup.sh run.sh
./setup.sh
```

### 4. Reproduce the best public-leaderboard model

```bash
./run.sh
```

`run.sh` trains the 5-fold MLP and then runs inference. The final file is:

```text
submissions/mlp_baseline.csv
```

## Useful commands

Train individual classical baselines:

```bash
python -m classic_pipeline.main --model logistic_regression
python -m classic_pipeline.main --model random_forest
python -m classic_pipeline.main --model xgboost
python -m classic_pipeline.main --model catboost
```

Tune boosting models:

```bash
python -m classic_pipeline.tune --model xgboost --trials 20
python -m classic_pipeline.tune --model catboost --trials 20
```

Train tuned XGBoost from saved Optuna parameters:

```bash
python -m classic_pipeline.main --model xgboost --tuned
```

Run tuned XGBoost inference:

```bash
python -m classic_pipeline.inference --model xgboost --experiment classic_tuned
```

Generate XGBoost interpretation artifacts:

```bash
python -m classic_pipeline.interpret --model xgboost --experiment classic_tuned
```

Generate final MLP + XGBoost blend:

```bash
python -m classic_pipeline.blend
```

Submit a generated prediction file to Kaggle, for example:

```bash
kaggle competitions submit -c titanic -f submissions/mlp_baseline.csv -m "MLP baseline"
```

## Reproducibility notes

- random seed: `42`
- 5-fold stratified CV
- all fold preprocessors/models are saved separately
- inference loads saved artifacts instead of retraining
- `data/`, `checkpoints/`, `submissions/`, and generated artifacts should not be committed
- Apple Silicon uses PyTorch MPS automatically when available
- `llvm-openmp` is included in the Conda environment because XGBoost requires an OpenMP runtime on macOS
