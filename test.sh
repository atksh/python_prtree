set -e

rm -rf build dist .pytest_cache
pip install .
pytest tests -vv --capture=no || rm -rf build dist .pytest_cache

