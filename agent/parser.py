from pathlib import Path

from textx import metamodel_from_file

from agent.processors import build_system_from_model, process_rule, process_skill
from agent.validation import validate_system

PROJECT_ROOT = Path(__file__).resolve().parent.parent
GRAMMAR_PATH = PROJECT_ROOT / "grammar" / "agent.tx"

_metamodel_cache = None
_metamodel_grammar_mtime: float | None = None


def load_metamodel():
    """Load the metamodel with object processors registered.

    Cached after the first call. The cache is invalidated automatically if the
    grammar file is modified (detected via mtime), so grammar changes take effect
    without restarting the Python process.
    """
    global _metamodel_cache, _metamodel_grammar_mtime
    current_mtime = GRAMMAR_PATH.stat().st_mtime
    if _metamodel_cache is None or current_mtime != _metamodel_grammar_mtime:
        mm = metamodel_from_file(str(GRAMMAR_PATH))
        mm.register_obj_processors(
            {
                "Rule": process_rule,
                "Skill": process_skill,
            }
        )
        _metamodel_cache = mm
        _metamodel_grammar_mtime = current_mtime
    return _metamodel_cache


def parse_model(model_path):
    """Parse and validate a .agent file into a System."""
    mm = load_metamodel()
    model = mm.model_from_file(str(model_path))
    return _build_and_validate(model)


def parse_model_text(model_text, source_name="<memory>"):
    """Parse and validate in-memory .agent content into a System.

    source_name is the logical filename used in diagnostics.
    """
    mm = load_metamodel()
    model = mm.model_from_str(model_text, file_name=str(source_name))
    return _build_and_validate(model)


def _build_and_validate(model):
    system = build_system_from_model(model)
    validate_system(system)
    return system
