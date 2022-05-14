set -e

rm -rf build dist .pytest_cache
pip uninstall python_prtree -y || true
DEBUG=1 pip install -v -e .
CPUPROFILE_FREQUENCY=10000 python docs/profile.py

so_path=src/python_prtree/PRTree.cpython-310-x86_64-linux-gnu.so
google-pprof --callgrind $so_path build.prof > cg_build.prof
google-pprof --callgrind $so_path find_all.prof > cg_find_all.prof