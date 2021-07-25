set -e

rm -rf build dist .pytest_cache
python setup.py install
pytest tests -vv --capture=no

