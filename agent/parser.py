from pathlib import Path
from textx import metamodel_from_file

from agent.validation import validate_system
from agent.processors import build_system_from_model, process_rule, process_skill


PROJECT_ROOT = Path(__file__).resolve().parent.parent
GRAMMAR_PATH = PROJECT_ROOT / "grammar" / "agent.tx"

_metamodel_cache = None


def load_metamodel():
    """Load the metamodel with object processors registered, cached after first call."""
    global _metamodel_cache
    if _metamodel_cache is None:
        mm = metamodel_from_file(str(GRAMMAR_PATH))
        mm.register_obj_processors({
            "Rule": process_rule,
            "Skill": process_skill,
        })
        _metamodel_cache = mm
    return _metamodel_cache


def parse_model(model_path):
    """
    Parse and validate a .agent file.

    Two-phase approach:
      1. textX parses the file; Rule/Skill processors run during parsing.
      2. build_system_from_model() transforms the Model into a System.
      3. validate_system() validates the System.

    Args:
        model_path: Path to the .agent file.

    Returns:
        System: Parsed and validated System object.
    """
    mm = load_metamodel()
    model = mm.model_from_file(str(model_path))
    return _build_and_validate(model)


def parse_model_text(model_text, source_name="<memory>"):
    """
    Parse and validate in-memory .agent content.

    Args:
        model_text: Full .agent document content.
        source_name: Logical source filename for diagnostics.

    Returns:
        System: Parsed and validated System object.
    """
    mm = load_metamodel()
    model = mm.model_from_str(model_text, file_name=str(source_name))
    return _build_and_validate(model)


def _build_and_validate(model):
    system = build_system_from_model(model)
    validate_system(system)
    return system
