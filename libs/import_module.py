import importlib
import sys

def import_module(path, name):
	"""从指定`path`导入名为`name`的模块"""
	try:
		module = importlib.import_module(path)
		module = getattr(module, name, None)
		return module
	except ImportError:
		print(f"Error: Path '{path}' not found.")
		sys.exit(1)