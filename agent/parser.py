from pathlib import Path
from textx import metamodel_from_file

from agent.validation import validate_model


PROJECT_ROOT = Path(__file__).resolve().parent.parent
GRAMMAR_PATH = PROJECT_ROOT / "grammar" / "agent.tx"


def load_metamodel():
    return metamodel_from_file(GRAMMAR_PATH)


def parse_model(model_path):
    mm = load_metamodel()
    model = mm.model_from_file(str(model_path))
    validate_model(model)
    return model