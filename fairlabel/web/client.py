from nicegui import app


def element_group(elem, obj):
    elem.bind_value(obj, "value")
    elem.on("update:model-value", lambda e: obj.update(e.args), throttle=0.1, leading_events=False)
    return elem


class Client:
    def __init__(self, id: str):
        self._id: str = id  # corresponds to ui.context.client.id
        self._dataset = None

    def reset(self):
        """Clears all client state."""
        self._dataset = None
        self._model_name = None
        self._model_params = {}
        self._model_instance = None

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

    @property
    def model_name(self):
        return getattr(self, "_model_name", None)

    @model_name.setter
    def model_name(self, value):
        self._model_name = value

    @property
    def model_params(self):
        return getattr(self, "_model_params", {})

    @model_params.setter
    def model_params(self, value):
        self._model_params = value

    @property
    def model_instance(self):
        return getattr(self, "_model_instance", None)

    @model_instance.setter
    def model_instance(self, value):
        self._model_instance = value
