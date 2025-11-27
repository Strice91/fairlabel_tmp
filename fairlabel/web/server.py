from nicegui import app, background_tasks, run, ui

from fairlabel.config import FAVICON, PACKAGE_ROOT, settings
from fairlabel.data import cache_data
from fairlabel.log import logger
from fairlabel.web.client import Client
from fairlabel.web.widgets import Menu


async def setup_ui():
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
    Menu()


if __name__ in {"__main__", "__mp_main__"}:
    for data_set in settings.dataset.values():
        cache_data(data_set.name)
    app.add_static_files("/static", PACKAGE_ROOT / "web/static")
    ui.run(title="fairlabel", favicon=FAVICON)
