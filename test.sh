set -e

rm -rf build dist .pytest_cache
pip install . --force-reinstall
python -m pytest tests -vv --capture=no || rm -rf build dist .pytest_cache

