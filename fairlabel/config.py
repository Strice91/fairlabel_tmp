import importlib.resources as res
from importlib.readers import MultiplexedPath
from pathlib import Path

from dynaconf import Dynaconf, Validator

f = res.files("fairlabel") # Workaround as res.files now returns a MultiplexedPath
PACKAGE_ROOT = next(iter(f._paths)) if isinstance(f, MultiplexedPath) else f
PROJECT_ROOT = PACKAGE_ROOT.parent
CONFIG_ROOT = PROJECT_ROOT / "config"
FAVICON = PACKAGE_ROOT / "web" / "static" / "fair.ico"

settings = Dynaconf(
    envvar_prefix="FAIRLBL",
    merge_enabled=True,
    load_dotenv=False,
    root_path=CONFIG_ROOT,
    settings_files=["settings.toml", ".secrets.toml"],
    validators=[
        Validator("data.dir", default=PROJECT_ROOT / "data", cast=Path),
        Validator("logging.level", default="DEBUG"),
        Validator("logging.stream", default=True),
        Validator("logging.size_kb", default=500),
        Validator("logging.file", default=False),
        Validator("logging.path", default="fairlabel.log", cast=Path),
    ],
)
