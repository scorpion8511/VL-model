import sys
import importlib.util
from pathlib import Path

stub_path = Path(__file__).parent / 'torch_stub.py'
spec = importlib.util.spec_from_file_location('torch_stub', stub_path)
torch_stub = importlib.util.module_from_spec(spec)
spec.loader.exec_module(torch_stub)
sys.modules.setdefault('torch', torch_stub)
