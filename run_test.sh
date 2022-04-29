set -e

rm -rf build dist .pytest_cache
pip uninstall python_prtree -y || true
pip install -v .
python -m pytest tests -vv --capture=no || rm -rf build dist .pytest_cache

