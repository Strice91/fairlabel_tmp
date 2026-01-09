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