set -e

rm -rf build dist .pytest_cache
pip uninstall python_prtree -y || true
pip install -v -e .
python -m pytest tests -vv --capture=no
