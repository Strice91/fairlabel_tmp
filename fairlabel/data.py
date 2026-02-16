import os
import re

import kagglehub
import numpy as np
import pandas as pd

from fairlabel.config import settings

os.environ["KAGGLEHUB_CACHE"] = str(settings.data.dir)


def cache_data(data_set_name: str) -> str:
    path = kagglehub.dataset_download(data_set_name)
    print("Path to dataset files:", path)
    return path


def get_dataset(short_name: str) -> pd.DataFrame:
    folder = settings.data.dir / "datasets" / settings.dataset[short_name].name / "versions"
    file = list(folder.glob("*/*.csv"))[-1]
    df = pd.read_csv(file)
    return df


def clean_column_name(name: str) -> str:
    """
    Clean a column name for display:
    - Replace underscores with spaces
    - Capitalize each word
    - Strip extra spaces
    """
    name = re.sub(r"_+", " ", name)  # replace underscores
    name = re.sub(r"\s+", " ", name)  # collapse multiple spaces
    name = name.strip().title()  # title case
    return name


def infer_column_types(df: pd.DataFrame) -> dict[str, str]:
    """
    Infer the type of each column in a DataFrame: 'numerical', 'categorical', or 'boolean'.
    Also cleans up column names (removes underscores, capitalizes words).

    :param df:
    """
    column_types = {}

    for col in df.columns:
        series = df[col].dropna()
        unique_vals = series.unique()

        # --- Determine type ---
        if series.dtype == bool:
            col_type = "boolean"

        elif len(unique_vals) == 2:
            normalized = {str(v).strip().lower() for v in unique_vals}
            if (
                normalized <= {"0", "1"}
                or normalized <= {"true", "false"}
                or normalized <= {"yes", "no"}
                or normalized <= {"y", "n"}
                or normalized <= {"approved", "rejected"}
            ):
                col_type = "boolean"
            else:
                # Sometimes two-value numeric columns may still be categorical (like "Gender")
                col_type = "categorical"

        elif np.issubdtype(series.dtype, np.number):
            if pd.api.types.is_integer_dtype(series) and series.nunique() < 10:
                col_type = "categorical"
            else:
                col_type = "numerical"

        else:
            # Try converting to numeric
            try:
                series_as_num = pd.to_numeric(series)
                if series_as_num.nunique() > 10:
                    col_type = "numerical"
                else:
                    col_type = "categorical"
            except Exception:
                col_type = "categorical"

        column_types[col] = col_type

    return column_types
