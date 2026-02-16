from typing import Any, Callable

from nicegui import ui
import pandas as pd

from fairlabel.config import settings
from fairlabel.data import clean_column_name, get_dataset, infer_column_types
from fairlabel.models import MODELS, ModelDefinition
from fairlabel.web.client import Client


class SetupWizard:
    def __init__(self, on_complete: Callable):
        self.on_complete = on_complete
        self.client = Client.retrieve()

        # State
        self.current_step = 1

        # Pre-select first dataset
        dataset_names = list(settings.dataset.keys())
        self.selected_dataset_name: str | None = dataset_names[0] if dataset_names else None

        # Pre-select first model
        model_names = list(MODELS.keys())
        self.selected_model_name: str | None = model_names[0] if model_names else None

        self.model_params: dict[str, Any] = {}
        if self.selected_model_name:
            model_def = MODELS[self.selected_model_name]
            self.model_params = {p.name: p.default for p in model_def.hyperparameters}

        # UI Elements
        self.container = ui.column().classes("w-full h-full items-center justify-center p-8")
        self.render()

    def render(self):
        self.container.clear()
        with self.container:
            with ui.card().classes("w-full max-w-4xl p-6"):
                # Stepper Header
                with ui.row().classes("w-full justify-between mb-8"):
                    self.render_step_header(1, "Select Dataset")
                    ui.icon("arrow_forward").classes("text-gray-400 mt-2")
                    self.render_step_header(2, "Select Model")
                    ui.icon("arrow_forward").classes("text-gray-400 mt-2")
                    self.render_step_header(3, "Configuration")

                # Step Content
                if self.current_step == 1:
                    self.render_dataset_step()
                elif self.current_step == 2:
                    self.render_model_step()
                elif self.current_step == 3:
                    self.render_config_step()

    def render_step_header(self, step: int, title: str):
        color = "primary" if step <= self.current_step else "gray-300"
        weight = "bold" if step == self.current_step else "normal"
        with ui.column().classes("items-center"):
            with ui.element("div").classes(
                f"w-8 h-8 rounded-full flex items-center justify-center bg-{color} text-white mb-1"
            ):
                ui.label(str(step))
            ui.label(title).classes(f"text-sm font-{weight}")

    # --- Step 1: Dataset Selection ---
    def render_dataset_step(self):
        ui.label("Choose a Dataset").classes("text-2xl font-bold mb-4")

        with ui.column().classes("w-full gap-8"):
            # List
            datasets = list(settings.dataset.keys())
            ui.select(
                datasets, label="Dataset", value=self.selected_dataset_name, on_change=self.on_dataset_select
            ).classes("w-full")

            # Preview
            if self.selected_dataset_name:
                self.render_dataset_preview()
            else:
                ui.label("Select a dataset to view preview").classes("text-gray-500 italic")

        with ui.row().classes("w-full justify-end mt-8"):
            ui.button("Next", on_click=lambda: self.set_step(2)).props("color=primary").bind_enabled_from(
                self, "selected_dataset_name"
            )

    def on_dataset_select(self, e):
        self.selected_dataset_name = e.value
        self.render()

    def render_dataset_preview(self):
        df = get_dataset(self.selected_dataset_name)
        preview_df = df.head(5)

        ui.label(f"Preview: {self.selected_dataset_name}").classes("text-lg font-semibold mb-2")
        ui.label(f"{len(df)} rows, {len(df.columns)} columns").classes("text-sm text-gray-600 mb-4")

        cols = [{"name": col, "label": clean_column_name(col), "field": col} for col in preview_df.columns]
        ui.table(columns=cols, rows=preview_df.to_dict("records")).classes("w-full h-64")

    # --- Step 2: Model Selection ---
    def render_model_step(self):
        ui.label("Choose a Model").classes("text-2xl font-bold mb-4")

        model_names = list(MODELS.keys())
        ui.radio(model_names, value=self.selected_model_name, on_change=self.on_model_select).props("inline").classes(
            "mb-4"
        )

        if self.selected_model_name:
            ui.markdown(f"**{self.selected_model_name}** selected. " "Continue to configure hyperparameters.").classes(
                "text-gray-600"
            )

        with ui.row().classes("w-full justify-between mt-8"):
            ui.button("Back", on_click=lambda: self.set_step(1)).props("outline")
            ui.button("Next", on_click=lambda: self.set_step(3)).props("color=primary").bind_enabled_from(
                self, "selected_model_name"
            )

    def on_model_select(self, e):
        self.selected_model_name = e.value
        # Reset params on model change
        model_def = MODELS[self.selected_model_name]
        self.model_params = {p.name: p.default for p in model_def.hyperparameters}
        self.render()

    # --- Step 3: Configuration ---
    def render_config_step(self):
        ui.label("Configure Hyperparameters").classes("text-2xl font-bold mb-4")

        model_def = MODELS[self.selected_model_name]

        with ui.grid(columns=2).classes("w-full gap-4"):
            for param in model_def.hyperparameters:
                with ui.column():
                    ui.label(f"{param.name}").classes("font-semibold")

                    if param.type == "float":
                        ui.number(
                            value=self.model_params[param.name],
                            min=param.min,
                            max=param.max,
                            step=0.01,
                            on_change=lambda e, name=param.name: self.update_param(name, e.value),
                        ).classes("w-full")

                    elif param.type == "int":
                        ui.number(
                            value=self.model_params[param.name],
                            min=param.min,
                            max=param.max,
                            step=1,
                            on_change=lambda e, name=param.name: self.update_param(name, int(e.value)),
                        ).classes("w-full")

                    elif param.type == "choice":
                        ui.select(
                            options=param.options,
                            value=self.model_params[param.name],
                            on_change=lambda e, name=param.name: self.update_param(name, e.value),
                        ).classes("w-full")

        with ui.row().classes("w-full justify-between mt-8"):
            ui.button("Back", on_click=lambda: self.set_step(2)).props("outline")
            ui.button("Finish Setup", on_click=self.finish_setup).props("color=positive")

    def update_param(self, name, value):
        self.model_params[name] = value

    def set_step(self, step):
        self.current_step = step
        self.render()

    def finish_setup(self):
        # Save configuration to client state
        self.client.dataset = self.selected_dataset_name
        self.client.model_name = self.selected_model_name
        self.client.model_params = self.model_params

        # Initialize model (TODO: Store this properly where the main app can access it, maybe in Client)
        model_def = MODELS[self.selected_model_name]
        self.client.model_instance = model_def.cls(**self.model_params)

        ui.notify("Setup Complete! Loading dataset...", type="positive")
        self.on_complete()
