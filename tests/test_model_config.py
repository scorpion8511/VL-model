from pathlib import Path
import ast

from musk.torchscale.architecture.config import EncoderConfig

# Parse the source to avoid importing heavy dependencies
source = Path('musk/modeling.py').read_text()
module = ast.parse(source)
func = next(node for node in module.body if isinstance(node, ast.FunctionDef) and node.name == '_get_large_config')
code = compile(ast.Module(body=[func], type_ignores=[]), filename='<ast>', mode='exec')
ns = {}
exec(code, {'EncoderConfig': EncoderConfig}, ns)
_get_large_config = ns['_get_large_config']


def test_moe_params_passthrough():
    cfg = _get_large_config(moe_freq=2, moe_expert_count=4, use_xmoe=True)
    assert cfg.moe_freq == 2
    assert cfg.moe_expert_count == 4
    assert cfg.use_xmoe is True
