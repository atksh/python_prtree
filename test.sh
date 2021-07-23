rm -rf build dist .pytest_cache
CXX=/usr/bin/g++ pip install .
pytest tests -vv --capture=no

