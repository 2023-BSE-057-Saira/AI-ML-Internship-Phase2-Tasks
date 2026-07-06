# End-to-End ML Pipeline with Scikit-learn Pipeline API

Build a reusable, production-ready machine learning pipeline that predicts customer churn using the Telco Churn dataset.

## 🎯 Objective
Businesses need reliable, reusable ML pipelines rather than one-off scripts. This project builds a complete `sklearn.Pipeline` that handles preprocessing (scaling, encoding), trains multiple models, tunes hyperparameters, and exports a single deployable pipeline object.

## 🗂️ Dataset
**Telco Customer Churn Dataset**
🔗 https://www.kaggle.com/datasets/blastchar/telco-customer-churn

Contains customer demographics, account information, and service usage, with a binary "Churn" target.

## 🛠️ Methodology / Approach
1. **Data Loading & Cleaning** – Load the dataset, handle missing/blank values (e.g., `TotalCharges`), and separate features from the target.
2. **Preprocessing Pipeline** – Use `ColumnTransformer` inside a `Pipeline` to scale numerical features (`StandardScaler`) and encode categorical features (`OneHotEncoder`).
3. **Model Training** – Train Logistic Regression and Random Forest classifiers, each wrapped inside the same preprocessing pipeline.
4. **Hyperparameter Tuning** – Use `GridSearchCV` to tune parameters (e.g., regularization strength for Logistic Regression, number of trees/depth for Random Forest) with cross-validation.
5. **Evaluation** – Compare models using accuracy, precision, recall, F1-score, and ROC-AUC.
6. **Export** – Save the best complete pipeline (preprocessing + model) using `joblib` for reuse in production without retraining.

## 📊 Key Results / Observations
- Random Forest (after tuning) outperformed baseline Logistic Regression on F1-score and ROC-AUC.
- Contract type, tenure, and monthly charges were among the strongest churn predictors.
- Wrapping preprocessing inside the pipeline eliminated data leakage risk and made the model directly deployable on raw, unprocessed input.

*(Replace with your actual metric values after running GridSearchCV.)*

## ⚙️ Tools & Technologies
- Python 3, Jupyter Notebook
- scikit-learn (Pipeline, ColumnTransformer, GridSearchCV)
- pandas, numpy
- joblib (model export)
- matplotlib/seaborn (visualizations)

## 🚀 How to Run
```bash
git clone https://github.com/<your-username>/customer-churn-sklearn-pipeline.git
cd customer-churn-sklearn-pipeline
pip install -r requirements.txt
jupyter notebook Customer_Churn_Pipeline.ipynb
```

To use the exported pipeline for predictions:
```python
import joblib
pipeline = joblib.load("churn_pipeline.pkl")
pipeline.predict(new_customer_data)
```

## 📁 Files
| File | Description |
|---|---|
| `Phase2_tTask2.ipynb` | Full pipeline: preprocessing, training, tuning, evaluation |
| `churn_pipeline.pkl` | Exported production-ready pipeline (via joblib) |
| `README.md` | Project documentation |

## 📌 Skills Demonstrated
- ML pipeline construction
- Hyperparameter tuning with GridSearchCV
- Model export and reusability
- Production-readiness practices
