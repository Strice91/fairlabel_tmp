from nicegui import app, background_tasks, run, ui

from fairlabel.config import FAVICON, PACKAGE_ROOT, settings
from fairlabel.data import cache_data
from fairlabel.log import logger
from fairlabel.web.client import Client
from fairlabel.web.widgets import Menu
from fairlabel.web.wizard import SetupWizard


async def setup_ui():
    client_tab_id = "Unknown"
    try:
        await ui.context.client.connected(timeout=10)
        client_tab_id = ui.context.client.tab_id
        if "client" not in app.storage.tab:
            app.storage.tab["client"] = Client(id=client_tab_id)
        logger.info(f"Client-{client_tab_id[-4:]} connected - page layout setup complete")
    except TimeoutError:
        logger.warning(f"Connection timeout for client {client_tab_id[-4:]}. Please reload the page.")


@ui.page("/")
async def main():
    await setup_ui()
    ui.page_title("Fairlabel")

    client = Client.retrieve()

    # Content container that will hold either the wizard or the main app
    content = ui.column().classes("w-full h-full p-0 gap-0")

    def show_main_app():
        content.clear()
        Menu()

    def show_wizard():
        content.clear()
        with content:
            SetupWizard(on_complete=lambda: ui.navigate.to("/", new_tab=False))

    # Check if dataset is already selected (reloads)
    if client.dataset:
        show_main_app()
    else:
        show_wizard()


if __name__ in {"__main__", "__mp_main__"}:
    for data_set in settings.dataset.values():
        cache_data(data_set.name)
    app.add_static_files("/static", PACKAGE_ROOT / "web/static")
    ui.run(title="fairlabel", favicon=FAVICON)
