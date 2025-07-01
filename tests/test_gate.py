from pathlib import Path
import ast

SRC = Path(__file__).resolve().parents[1] / 'musk' / 'torchscale' / 'component' / 'xmoe' / 'routing.py'
source = SRC.read_text()
module = ast.parse(source)


def _func_params(name):
    for node in module.body:
        if isinstance(node, ast.ClassDef) and node.name == name:
            for f in node.body:
                if isinstance(f, ast.FunctionDef) and f.name == 'forward':
                    return [a.arg for a in f.args.args]
    return []


def test_gate_accepts_domain_ids():
    params1 = _func_params('Top2Gate')
    params2 = _func_params('Top1Gate')
    assert 'domain_ids' in params1
    assert 'domain_ids' in params2
