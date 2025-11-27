from nicegui import background_tasks, run, ui

from fairlabel.config import settings
from fairlabel.data import clean_column_name, get_dataset, infer_column_types
from fairlabel.web.client import Client


class Header(ui.header):
    def __init__(self, on_dataset_change) -> None:
        super().__init__()
        cl = Client.retrieve()

        ui.colors(**settings.colors)
        with self.classes("items-center"):
            ui.image("/static/fair.svg").classes("w-12")
            ui.label("Fair Label").classes("text-xl text-white")
            dataset_names = list(settings.dataset.keys())
            self.dropdown = (
                ui.select(
                    options=dataset_names,
                    label="Select Dataset",
                    on_change=lambda e: setattr(cl, "dataset", e.value),
                )
                .props("outlined dense color=white")
                .classes("w-64")
            )


class Menu(ui.left_drawer):
    def __init__(self, selected_dataset_name: str | None = None) -> None:
        super().__init__(fixed=True, bordered=False)
        self.selected_dataset_name = selected_dataset_name
        self.dataset_info = None
        ui.colors(**settings.colors)

        with Header(self.set_selected_dataset):
            ui.button(on_click=self.toggle).props("flat color=white icon=menu")

        with self.classes(f"bg-[{settings.colors.accent}] p-4"):
            self.info_container = ui.column().classes("mt-4")
            self.update_info()

    def set_selected_dataset(self, dataset_name: str):
        """Called when user selects a new dataset."""
        self.selected_dataset_name = dataset_name
        self.update_info()

    def update_info(self):
        """Refresh dataset overview."""
        self.info_container.clear()
        with self.info_container:
            if not self.selected_dataset_name:
                ui.label("No dataset selected")
                return
            df = get_dataset(self.selected_dataset_name)

            data_cfg = settings.dataset[self.selected_dataset_name]
            ui.label(f"📊 {self.selected_dataset_name}").classes("text-lg font-semibold")
            ui.label(f"Name: {data_cfg.get('name', '-')}")
            ui.label(f"Label: {data_cfg.get('label', '-')}")
            ui.label(f"Excluded: {', '.join(data_cfg.get('exclude', [])) or '-'}")
            ui.label("Columns:").classes("mt-2 font-medium")

            with ui.column().classes("ml-2"):
                for col, dtype in infer_column_types(df).items():
                    ui.label(f"- {clean_column_name(col)}: {dtype}")
