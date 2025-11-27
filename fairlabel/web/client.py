from nicegui import app


def element_group(elem, obj):
    elem.bind_value(obj, "value")
    elem.on("update:model-value", lambda e: obj.update(e.args), throttle=0.1, leading_events=False)
    return elem


class Client:
    def __init__(self, id: str):
        self._id: str = id  # corresponds to ui.context.client.id
        self._dataset = None

    @staticmethod
    def retrieve() -> "Client":
        """
        Static method to retrieve the current client instance stored in the application session storage.

        This is also the definition of a client (app.storage.tab)
        """
        return app.storage.tab["client"]

    @property
    def dataset(self):
        return self._dataset

    @dataset.setter
    def dataset(self, value):
        print(f"dataset selected: {value}")
        self._dataset = value
