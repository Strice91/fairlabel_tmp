# fairlabel

fairlabel_tmp is a Python-based tool for **fair active learning**.  
It provides an interactive web-based interface for data labeling and exploration, while supporting workflows that aim to improve fairness in machine learning through active learning strategies.

The application is built as a full-stack Python service using **NiceGUI**, which serves both the backend logic and the browser-based user interface.

---

## Project Goals

- Support **fair active learning** workflows
- Enable interactive data labeling through a web UI
- Assist in reducing bias during dataset annotation
- Provide a Python-native, reproducible research tool

---

## Features

- Interactive labeling interface
- Full-stack Python application powered by NiceGUI
- Browser-based UI without a separate frontend framework
- Modern Python packaging using `pyproject.toml`
- Suitable for research and experimentation

---

## Installation

### Requirements

- Python 3.9 or newer
- Either:
    - Poetry (recommended), or
    - pip (22.0+) with venv / virtualenv

---

### Option A — Using Poetry (Recommended)
Poetry is recommended for managing environments, dependency resolution, and project tooling.

**Install Poetry (System-Wide)**
```bash
brew install poetry
```

**macOS/Linux (Official Installer):**
```bash
curl -sSL https://install.python-poetry.org | python3 -
```

Verify installation:
```bash
poetry --version
```

**Setup the Environment**
To ensure Poetry creates the virtual environment inside the project (as ./.venv), enable this setting:
```bash
poetry config virtualenvs.in-project true
```

Select the Python interpreter to use (this project supports Python 3.9+; pick the version installed on your machine, e.g. 3.11):
```bash
poetry env use python3.11
```

Install dependencies:
```bash
poetry install
```


### Option B — Using pip + venv (Minimal Setup)
If you prefer a minimal setup without Poetry:

**Create and Activate a Virtual Environment**

**Linux / macOS:**
```bash
python -m venv .venv
source .venv/bin/activate
```

**Windows (PowerShell):**
```bash
python -m venv .venv
.\.venv\Scripts\Activate.ps1
```

Upgrade packaging tools (recommended):
```bash
python -m pip install --upgrade pip setuptools wheel
```

**Install the Project**
```bash
pip install -e .
```

## Run the Application
Make sure the virtual environment is activated!

**Linux / macOS:**
```bash
source .venv/bin/activate
```

**Windows (PowerShell):**
```bash
.\.venv\Scripts\Activate.ps1
```

**Run the Webserver**

```bash
python fairlabel/web/server.py
```

## Overview

- This project simulates a real-world lending scenario where labeled data is expensive and fairness is critical. It uses:

- Active Learning (modAL): To intelligently select the most "confusing" loan applications for human review, reducing the need for massive labeled datasets.

- XGBoost: As the high-performance decision engine.

- Fairlearn: To enforce Demographic Parity, ensuring the model does not discriminate against protected groups (e.g., Self-Employed individuals).

- SHAP: To explain why a specific loan was approved or rejected.

## Tech Stack
- Python 3.9 or newer

- XGBoost (Gradient Boosting)

- modAL (Active Learning Framework)

- Fairlearn (Bias Mitigation)

- SHAP (Model Explainability)

- Pandas / NumPy (Data Manipulation)

## Installation
- Clone the repository
- Create and Activate virtual environment
- Install dependencies: pip install pandas numpy xgboost scikit-learn fairlearn shap modAL-python matplotlib
- Prepare your data:
Place your dataset in the root folder and name it loan_data.csv. The script expects a CSV with columns like:
loan_id, no_of_dependents, education, self_employed, income_annum, loan_amount, loan_term, cibil_score, ... loan_status
- Run the script:  python fairlabel\EBM.py

## 📊 Expected Output
The script will perform the following steps:

- Data Loading: Automatically detects categorical columns and the sensitive attribute (e.g., self_employed).

- Active Learning Loop:

    - The model starts with a small "seed" (5% of data).

    - It queries the "pool" for the most uncertain cases.

    - It simulates human labeling and retrains.

    - Output: You will see the accuracy score update in the terminal for each round.

- Fairness Correction:

    - It applies the ExponentiatedGradient algorithm to enforce Demographic Parity.

    - Output: A "Fairness Report" comparing approval rates between groups.

- Visualization:

    - Bar Chart: Shows the Approval Rate gap between groups (before/after fairness).

    - SHAP Beeswarm Plot: Shows which features (Income, CIBIL, etc.) drove the decisions.
 
- Common Issues
    - ModuleNotFoundError: No module named 'modAL':
You likely installed the wrong package. Run pip uninstall modal and then pip install modAL-python.

    - UserWarning: Parameters: { "use_label_encoder" } are not used:
This is a harmless warning from XGBoost. You can ignore it or remove the parameter from the code.
