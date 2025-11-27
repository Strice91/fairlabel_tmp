import pandas as pd
import numpy as np
from nicegui import ui
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
import random

# --- 1. DATA AND STATE MANAGEMENT ---

# Mock Credit Approval Dataset (Tabular)
# Features: Age, Income (k$), DTI, Score. Sensitive Attribute: Group
data = {
    "Age": [35, 22, 50, 40, 60, 28, 55, 30, 45, 25, 42, 38, 58, 29, 33],
    "Income": [50, 30, 90, 45, 120, 35, 80, 55, 70, 40, 65, 52, 95, 38, 48],
    "DTI": [0.30, 0.50, 0.20, 0.40, 0.10, 0.60, 0.25, 0.35, 0.15, 0.45, 0.33, 0.41, 0.18, 0.55, 0.39],
    "Score": [720, 610, 780, 650, 800, 580, 750, 700, 790, 620, 680, 640, 770, 590, 690],
    "Group": ["M", "F", "F", "M", "M", "F", "M", "F", "F", "M", "F", "M", "M", "F", "F"],
    "True_Label": [1, 0, 1, 0, 1, 0, 1, 1, 1, 0, 1, 0, 1, 0, 1],  # 1=Approved, 0=Rejected (Hidden in AL)
}
df = pd.DataFrame(data)
df["Label"] = np.nan  # This is the column the user will fill
df["Selected"] = False  # True if the item has been selected for labeling


# Application State Class
class AppState:
    def __init__(self):
        self.df = df.copy()
        self.current_index = -1
        self.model = None
        self.scaler = StandardScaler()
        self.FAIRNESS_TARGET = 0.5  # Aim for 50/50 M/F split in selected samples
        self.FEATURES = ["Age", "Income", "DTI", "Score"]


state = AppState()

# --- 2. MACHINE LEARNING AND FAIRNESS LOGIC ---


def train_model():
    """Trains the Logistic Regression model using labeled data."""
    labeled_df = state.df.dropna(subset=["Label"])

    if len(labeled_df) < 5:
        state.model = None
        return "Not enough labeled data (need 5+)."

    X = labeled_df[state.FEATURES]
    y = labeled_df["Label"]

    # Scale features
    state.scaler.fit(X)
    X_scaled = state.scaler.transform(X)

    # Train model
    state.model = LogisticRegression(solver="liblinear", random_state=42)
    state.model.fit(X_scaled, y)
    return f"Model trained with {len(labeled_df)} samples."


def calculate_uncertainty_score(features_df):
    """Calculates uncertainty (distance from 0.5 probability) for unlabeled data."""
    if state.model is None:
        # High uncertainty if model is not trained (encourages random initial sampling)
        # Random initial selection if model is not trained
        return pd.Series(0.5 + np.random.rand(len(features_df)) * 0.1, index=features_df.index)

    X_scaled = state.scaler.transform(features_df)
    # Predict probabilities for the positive class (1: Approved)
    probabilities = state.model.predict_proba(X_scaled)[:, 1]
    # Uncertainty is 1 minus the absolute distance from 0.5 (closer to 0.5 is higher uncertainty)
    uncertainty = 1 - np.abs(probabilities - 0.5)
    return pd.Series(uncertainty, index=features_df.index)


def fair_active_select():
    """Selects the next item using a hybrid Uncertainty + Fairness score."""
    unlabeled_df = state.df[state.df["Label"].isna()]

    if unlabeled_df.empty:
        return -1, "No more unlabeled items."

    selected_df = state.df[state.df["Selected"] == True]
    total_selected = len(selected_df)

    if total_selected == 0:
        # Random initial selection if no items have been selected yet
        return random.choice(unlabeled_df.index), "Random initial selection."

    # 1. Calculate Selection Fairness Metrics
    group_counts = selected_df["Group"].value_counts(normalize=True).reindex(["M", "F"], fill_value=0)
    current_M_ratio = group_counts.get("M", 0)

    # Determine the underrepresented group
    underrepresented_group = "M" if current_M_ratio < state.FAIRNESS_TARGET else "F"
    deviation = abs(current_M_ratio - state.FAIRNESS_TARGET)

    # 2. Calculate Hybrid Score for Unlabeled Items
    uncertainty_scores = calculate_uncertainty_score(unlabeled_df[state.FEATURES])

    hybrid_scores = {}
    for index in unlabeled_df.index:
        group = unlabeled_df.loc[index, "Group"]
        ml_score = uncertainty_scores.loc[index]

        # Fairness Boost: significant if the item is in the underrepresented group
        fairness_boost = 0.0
        if group == underrepresented_group:
            # Scale the boost by the current deviation
            fairness_boost = deviation * 1.5

        # Hybrid Score: prioritize uncertainty, then boost fairness
        hybrid_score = ml_score + fairness_boost
        hybrid_scores[index] = hybrid_score

    # 3. Select the index with the highest hybrid score
    next_index = max(hybrid_scores, key=hybrid_scores.get)

    return next_index, "Fair Active Learning selection."


# --- 3. NICEGUI UI LOGIC (Error-Fixed) ---


def update_ui(status_message: str, selected_card: ui.card, stats_label: ui.label, table: ui.table):
    """Updates all reactive elements on the page."""

    # 1. Update Current Item Card
    with selected_card:
        selected_card.clear()
        if state.current_index != -1:
            item = state.df.loc[state.current_index]
            ui.label(f"➡️ **CURRENT ITEM TO LABEL** (ID: {state.current_index})").classes("text-lg text-primary")
            ui.label(f"Age: {item['Age']}, Income: ${item['Income']}K, DTI: {item['DTI']:.2f}, Score: {item['Score']}")
            ui.label(f"Sensitive Group: **{item['Group']}**").classes("text-xl")
        else:
            ui.label("Dataset fully labeled or initialization needed. Press 'Start'").classes("text-lg")

    # 2. Update Table Display
    table.rows = state.df.reset_index().to_dict("records")

    # 3. Update Stats
    labeled_count = state.df["Label"].count()
    total_count = len(state.df)

    selected_df = state.df[state.df["Selected"] == True]
    total_selected = len(selected_df)

    m_count = selected_df[selected_df["Group"] == "M"].shape[0]
    f_count = selected_df[selected_df["Group"] == "F"].shape[0]
    m_pct = (m_count / total_selected) if total_selected > 0 else 0
    f_pct = (f_count / total_selected) if total_selected > 0 else 0

    stats_label.set_text(f"""
        **Status:** {status_message} | **Labeled:** {labeled_count}/{total_count} | 
        **M Selected:** {m_count} ({m_pct:.1%}) | **F Selected:** {f_count} ({f_pct:.1%})
    """)


def select_next_item(selected_card, stats_label, table):
    """Handles the UI action for selecting the next item."""
    index, message = fair_active_select()

    if index != -1:
        state.current_index = index
        state.df.loc[index, "Selected"] = True
        update_ui(message, selected_card, stats_label, table)
    else:
        state.current_index = -1
        update_ui("No more unlabeled data!", selected_card, stats_label, table)


def label_item(label_value, selected_card, stats_label, table):
    """Applies the label, retrains the model, and selects the next item."""
    if state.current_index != -1 and pd.isna(state.df.loc[state.current_index, "Label"]):
        # 1. Apply the label
        state.df.loc[state.current_index, "Label"] = label_value

        # 2. Retrain the model
        model_status = train_model()

        # 3. Select the next item automatically
        select_next_item(selected_card, stats_label, table)

        # Update the UI with the final status and new selection
        update_ui(f"Item labeled as {label_value}. {model_status}", selected_card, stats_label, table)


@ui.page("/")
def main_page():
    ui.add_head_html("<title>Fair Active Learning MVP</title>")

    # --- 1. INITIALIZE UI ELEMENTS INSIDE THE PAGE FUNCTION ---
    stats_label = ui.label("Loading...").classes("font-mono text-sm mb-4")
    selected_card = ui.card().classes("w-full border-2 border-primary p-4 rounded-lg shadow-lg")
    table = ui.table(
        columns=[
            {"name": "index", "label": "ID", "field": "index", "align": "left"},
            {"name": "Age", "label": "Age", "field": "Age"},
            {"name": "Income", "label": "Income (k$)", "field": "Income"},
            {"name": "Score", "label": "Score", "field": "Score"},
            {"name": "Group", "label": "Group", "field": "Group"},
            {"name": "Label", "label": "Label", "field": "Label"},
        ],
        rows=[],
    ).classes("w-full")

    # --- 2. LAYOUT DEFINITION (using initialized elements) ---
    with ui.header().classes("items-center justify-between"):
        ui.label("Fair Active Learning Dashboard").classes("text-2xl font-bold")

    with ui.row().classes("w-full"):
        # --- LEFT COLUMN: Selection and Labeling ---
        with ui.column().classes("w-1/3"):
            ui.label("Active Selection & Labeling").classes("text-xl font-semibold")

            stats_label
            selected_card

            # Action Buttons: Use lambda to pass the necessary UI elements
            ui.label("Action:").classes("mt-4 font-semibold")
            with ui.row().classes("w-full"):
                ui.button(
                    "➡️ Start/Next Item Selection",
                    on_click=lambda: select_next_item(selected_card, stats_label, table),
                    color="secondary",
                ).classes("w-full")
                ui.button(
                    "✅ APPROVE (Label: 1)",
                    on_click=lambda: label_item(1, selected_card, stats_label, table),
                    color="positive",
                ).classes("w-1/2")
                ui.button(
                    "❌ REJECT (Label: 0)",
                    on_click=lambda: label_item(0, selected_card, stats_label, table),
                    color="negative",
                ).classes("w-1/2")

        # --- RIGHT COLUMN: Dataset View and Model Status ---
        with ui.column().classes("w-2/3"):
            ui.label("Full Dataset and Progress").classes("text-xl font-semibold")

            table

            ui.label("Sensitive Group: **M** / **F** (Not used for prediction, only for fairness)").classes(
                "mt-4 text-xs italic"
            )

    # Initial UI update
    update_ui("Welcome! Press 'Start/Next Item Selection' to begin.", selected_card, stats_label, table)


# Run the NiceGUI app
ui.run()
