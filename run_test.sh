set -e

rm -rf build dist .pytest_cache
pip uninstall python_prtree -y || true
DEBUG=1 pip install -v -e .
python -m pytest tests -vv --capture=no || rm -rf build dist .pytest_cache

