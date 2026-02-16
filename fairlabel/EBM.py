import pandas as pd
import numpy as np
import shap
import matplotlib.pyplot as plt
from xgboost import XGBClassifier
from modAL.models import ActiveLearner
from modAL.uncertainty import uncertainty_sampling
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from fairlearn.metrics import MetricFrame, selection_rate, demographic_parity_difference
from fairlearn.reductions import ExponentiatedGradient, DemographicParity


print("--- Step 1: Loading Data ---")


# file_path = '../data/datasets/architsharma01/loan-approval-prediction-dataset/versions/1/loan_approval_dataset.csv'
file_path = 'data/datasets/architsharma01/loan-approval-prediction-dataset/versions/1/loan_approval_dataset.csv'

try:
    df = pd.read_csv(file_path)
    print(f"Successfully loaded {len(df)} rows.")
except FileNotFoundError:
    print("Error: File not found. Please create 'loan_data.csv' with your data.")
    exit()


df.columns = df.columns.str.strip()

if 'loan_id' in df.columns:
    df = df.drop('loan_id', axis=1)


df['loan_status'] = df['loan_status'].astype(str).str.strip()
df['target'] = df['loan_status'].apply(lambda x: 1 if x == 'Approved' else 0)
df = df.drop('loan_status', axis=1)


categorical_cols = df.select_dtypes(include=['object']).columns.tolist()
print(f"Categorical columns found: {categorical_cols}")

df_encoded = pd.get_dummies(df, columns=categorical_cols, drop_first=True)


sensitive_col_name = [col for col in df_encoded.columns if 'self_employed' in col]
if sensitive_col_name:
    SENSITIVE_COL = sensitive_col_name[0] 
    print(f"Sensitive Attribute identified: {SENSITIVE_COL}")
else:
    print("Warning: 'self_employed' column not found. Fairness check might fail.")
    SENSITIVE_COL = df_encoded.columns[0] 


X = df_encoded.drop('target', axis=1)
y = df_encoded['target']
A = df_encoded[SENSITIVE_COL] 


X_dev, X_test, y_dev, y_test, A_dev, A_test = train_test_split(
    X, y, A, test_size=0.3, random_state=42
)


X_seed, X_pool, y_seed, y_pool = train_test_split(X_dev, y_dev, train_size=0.05, random_state=42)


print("\n--- Step 2: Starting Active Learning Loop ---")

learner = ActiveLearner(
    estimator=XGBClassifier(use_label_encoder=False, eval_metric='logloss', random_state=42),
    query_strategy=uncertainty_sampling,
    X_training=X_seed.values,
    y_training=y_seed.values
)

print(f"Initial Accuracy (Seed only): {learner.score(X_test.values, y_test.values):.2f}")


n_queries = 5
for i in range(n_queries):
    query_idx, query_inst = learner.query(X_pool.values)
    
    human_label = y_pool.iloc[query_idx].values
    
    
    learner.teach(query_inst, human_label)
    
    
    X_pool = X_pool.drop(X_pool.index[query_idx])
    y_pool = y_pool.drop(y_pool.index[query_idx])
    
    acc = learner.score(X_test.values, y_test.values)
    print(f"Round {i+1}: Accuracy -> {acc:.2f}")


print("\n--- Step 3: Applying Fairness Constraints ---")

mitigator = ExponentiatedGradient(
    estimator=XGBClassifier(use_label_encoder=False, eval_metric='logloss', random_state=42),
    constraints=DemographicParity()
)


mitigator.fit(X_dev, y_dev, sensitive_features=A_dev)
y_pred_fair = mitigator.predict(X_test)

mf = MetricFrame(
    metrics={'accuracy': accuracy_score, 'selection_rate': selection_rate},
    y_true=y_test,
    y_pred=y_pred_fair,
    sensitive_features=A_test
)

print("\n--- FAIRNESS REPORT ---")
print(mf.by_group)
print(f"\nFinal Accuracy: {mf.overall['accuracy']:.2f}")
print(f"Demographic Parity Diff: {demographic_parity_difference(y_test, y_pred_fair, sensitive_features=A_test):.4f}")

print("\n--- Step 4: Generating Charts ---")


plt.figure(figsize=(10, 5))
mf.by_group['selection_rate'].plot(kind='bar', color=['#1f77b4', '#ff7f0e'])
plt.title('Approval Rate by Group (Fairness Check)')
plt.ylabel('Approval Rate')
plt.xlabel(f'Sensitive Attribute: {SENSITIVE_COL}')
plt.axhline(y=mf.overall['selection_rate'], color='red', linestyle='--', label='Average')
plt.legend()
plt.tight_layout()
plt.show()

# SHAP Explanation
explainer = shap.Explainer(mitigator.predictor_)
shap_values = explainer(X_test)
shap.plots.beeswarm(shap_values, max_display=10)