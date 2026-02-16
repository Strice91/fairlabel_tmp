from nicegui import background_tasks, run, ui

from fairlabel.config import settings
from fairlabel.data import clean_column_name, get_dataset, infer_column_types
from fairlabel.web.client import Client


class Header(ui.header):
    def __init__(self) -> None:
        super().__init__()

        ui.colors(**settings.colors)
        with self.classes("items-center gap-4"):
            ui.image("/static/fair.svg").classes("w-12")
            ui.label("Fair Label").classes("text-xl text-white")


class Menu(ui.left_drawer):
    def __init__(self) -> None:
        super().__init__(fixed=True, bordered=False)
        self.app_client = Client.retrieve()

        ui.colors(**settings.colors)

        with Header():
            ui.button(on_click=self.toggle).props("flat color=white icon=menu")
            ui.button("Restart", on_click=self.confirm_restart, icon="restart_alt").props("flat color=white").tooltip(
                "Restart setup process and clear selection"
            )

        with self.classes(f"bg-[{settings.colors.accent}] p-4"):
            self.info_container = ui.column().classes("mt-4")
            self.update_info()

    def update_info(self):
        """Refresh dataset overview based on client selection."""
        self.info_container.clear()

        dataset_name = self.app_client.dataset

        with self.info_container:
            if not dataset_name:
                ui.label("No dataset selected")
                return

            df = get_dataset(dataset_name)

            data_cfg = settings.dataset[dataset_name]
            ui.label(f"📊 {dataset_name}").classes("text-lg font-semibold")
            ui.label(f"Name: {data_cfg.get('name', '-')}")
            ui.label(f"Label: {data_cfg.get('label', '-')}")
            ui.label(f"Excluded: {', '.join(data_cfg.get('exclude', [])) or '-'}")
            ui.label("Columns:").classes("mt-2 font-medium")

            with ui.column().classes("ml-2"):
                for col, dtype in infer_column_types(df).items():
                    ui.label(f"- {clean_column_name(col)}: {dtype}")

    def restart(self):
        """Resets client state and reloads to show wizard."""
        self.app_client.reset()
        ui.navigate.to("/", new_tab=False)

    def confirm_restart(self):
        with ui.dialog() as dialog, ui.card():
            ui.label("Are you sure you want to restart? This will clear your current selection.")
            with ui.row():
                ui.button("Cancel", on_click=dialog.close)
                ui.button("Restart", on_click=self.restart, color="negative")
        dialog.open()
