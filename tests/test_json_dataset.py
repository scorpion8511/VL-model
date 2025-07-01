import ast
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
source = (ROOT / "musk" / "json_dataset.py").read_text()
module = ast.parse(source)
fn_source = next(ast.get_source_segment(source, node) for node in module.body if isinstance(node, ast.FunctionDef) and node.name == "_infer_domain")
ns: dict = {}
exec(fn_source, ns)
_infer_domain = ns["_infer_domain"]


def test_infer_domain():
    assert _infer_domain("chest x-ray image") == 0
    assert _infer_domain("histopathology slide") == 1
    assert _infer_domain("endoscopy view") == 2
