"""Crash isolation tests using subprocess.

These tests run potentially dangerous operations in isolated subprocesses
to prevent crashes from affecting the test suite. Each test checks if
the subprocess exits cleanly or crashes with a segfault.

Run with: pytest tests/unit/test_crash_isolation.py -v
"""
import subprocess
import sys
import textwrap
from typing import Tuple
import pytest


def run_in_subprocess(code: str) -> Tuple[int, str, str]:
    """Run code in a subprocess and return exit code, stdout, stderr.

    Returns:
        (exit_code, stdout, stderr)
        exit_code: 0 for success, -11 for segfault on Unix, >0 for other errors
    """
    result = subprocess.run(
        [sys.executable, "-c", code],
        capture_output=True,
        text=True,
        timeout=30  # Increased timeout for slower CI environments
    )
    return result.returncode, result.stdout, result.stderr


class TestDoubleFree:
    """Test protection against double-free errors."""

    @pytest.mark.parametrize("dim", [2, 3, 4])
    def test_double_erase_no_crash(self, dim):
        """Verify that double erase of same index does not crash."""
        code = textwrap.dedent(f"""
        import numpy as np
        from python_prtree import PRTree{dim}D

        idx = np.arange(10)
        boxes = np.random.rand(10, {2*dim}) * 100
        for i in range({dim}):
            boxes[:, i + {dim}] += boxes[:, i] + 1

        tree = PRTree{dim}D(idx, boxes)
        tree.erase(5)

        # Try to erase again - should not crash
        try:
            tree.erase(5)
        except (ValueError, RuntimeError, KeyError):
            pass  # Error is OK

        print("SUCCESS")
        """)

        exit_code, stdout, stderr = run_in_subprocess(code)

        # Should not segfault (-11 on Unix)
        assert exit_code != -11, f"Process crashed with segfault. stderr: {stderr}"
        assert exit_code == 0 or "SUCCESS" in stdout, f"Unexpected error: {stderr}"

    @pytest.mark.parametrize("dim", [2, 3, 4])
    def test_erase_after_rebuild_no_crash(self, dim):
        """Verify that erasing old indices after rebuild does not crash."""
        code = textwrap.dedent(f"""
        import numpy as np
        from python_prtree import PRTree{dim}D

        idx = np.arange(100)
        boxes = np.random.rand(100, {2*dim}) * 100
        for i in range({dim}):
            boxes[:, i + {dim}] += boxes[:, i] + 1

        tree = PRTree{dim}D(idx, boxes)

        # Erase half
        for i in range(50):
            tree.erase(i)

        tree.rebuild()

        # Try to erase already-erased indices - should not crash
        try:
            for i in range(25):
                tree.erase(i)
        except (ValueError, RuntimeError, KeyError):
            pass

        print("SUCCESS")
        """)

        exit_code, stdout, stderr = run_in_subprocess(code)
        assert exit_code != -11, f"Process crashed with segfault. stderr: {stderr}"


class TestInvalidMemoryAccess:
    """Test protection against invalid memory access."""

    @pytest.mark.parametrize("dim", [2, 3, 4])
    def test_query_with_massive_coordinates_no_crash(self, dim):
        """Verify that extremely large coordinates do not crash."""
        code = textwrap.dedent(f"""
        import numpy as np
        from python_prtree import PRTree{dim}D

        idx = np.arange(10)
        boxes = np.random.rand(10, {2*dim}) * 100
        for i in range({dim}):
            boxes[:, i + {dim}] += boxes[:, i] + 1

        tree = PRTree{dim}D(idx, boxes)

        # Query with massive coordinates
        query = np.full({2*dim}, 1e308)  # Near max float64

        try:
            result = tree.query(query)
        except (ValueError, RuntimeError, OverflowError):
            pass

        print("SUCCESS")
        """)

        exit_code, stdout, stderr = run_in_subprocess(code)
        assert exit_code != -11, f"Process crashed with segfault. stderr: {stderr}"

    @pytest.mark.parametrize("dim", [2, 3, 4])
    def test_insert_extreme_values_no_crash(self, dim):
        """Verify that inserting extreme values does not crash."""
        code = textwrap.dedent(f"""
        import numpy as np
        from python_prtree import PRTree{dim}D

        tree = PRTree{dim}D()

        # Try inserting with extreme values
        test_cases = [
            (1, np.full({2*dim}, 1e200)),
            (2, np.full({2*dim}, -1e200)),
            (3, np.array([1e100] * {dim} + [1e101] * {dim})),
        ]

        for idx, box in test_cases:
            try:
                tree.insert(idx=idx, bb=box)
            except (ValueError, RuntimeError, OverflowError):
                pass  # Error is acceptable

        print("SUCCESS")
        """)

        exit_code, stdout, stderr = run_in_subprocess(code)
        assert exit_code != -11, f"Process crashed with segfault. stderr: {stderr}"


class TestFileCorruption:
    """Test protection against file corruption crashes."""

    @pytest.mark.parametrize("dim", [2, 3, 4])
    def test_load_random_bytes_no_crash(self, dim):
        """Verify that loading random bytes file does not crash."""
        code = textwrap.dedent(f"""
        import numpy as np
        import tempfile
        import os
        from python_prtree import PRTree{dim}D

        with tempfile.NamedTemporaryFile(delete=False, suffix='.bin') as f:
            # Write random bytes
            f.write(np.random.bytes(10000))
            fname = f.name

        try:
            tree = PRTree{dim}D(fname)
        except (RuntimeError, ValueError, OSError, EOFError):
            pass  # Error is expected
        finally:
            os.unlink(fname)

        print("SUCCESS")
        """)

        exit_code, stdout, stderr = run_in_subprocess(code)
        assert exit_code != -11, f"Process crashed with segfault. stderr: {stderr}"

    @pytest.mark.parametrize("dim", [2, 3, 4])
    def test_load_truncated_file_no_crash(self, dim):
        """Verify that loading truncated file does not crash."""
        code = textwrap.dedent(f"""
        import numpy as np
        import tempfile
        import os
        from python_prtree import PRTree{dim}D

        # Create valid tree and save
        idx = np.arange(100)
        boxes = np.random.rand(100, {2*dim}) * 100
        for i in range({dim}):
            boxes[:, i + {dim}] += boxes[:, i] + 1

        tree = PRTree{dim}D(idx, boxes)

        with tempfile.NamedTemporaryFile(delete=False, suffix='.bin') as f:
            fname = f.name

        tree.save(fname)

        # Truncate file
        with open(fname, 'rb') as f:
            data = f.read()

        # Write only 10% of data
        with open(fname, 'wb') as f:
            f.write(data[:len(data) // 10])

        try:
            tree2 = PRTree{dim}D(fname)
        except (RuntimeError, ValueError, OSError, EOFError):
            pass  # Error is expected
        finally:
            os.unlink(fname)

        print("SUCCESS")
        """)

        exit_code, stdout, stderr = run_in_subprocess(code)
        assert exit_code != -11, f"Process crashed with segfault. stderr: {stderr}"


class TestStressConditions:
    """Test behavior under stress conditions."""

    @pytest.mark.parametrize("dim", [2, 3, 4])
    def test_rapid_insert_erase_no_crash(self, dim):
        """Verify that rapid insert/erase cycles do not crash."""
        code = textwrap.dedent(f"""
        import numpy as np
        from python_prtree import PRTree{dim}D

        tree = PRTree{dim}D()

        # Rapid insert/erase cycles (reduced for CI performance)
        for iteration in range(20):
            for i in range(50):
                box = np.random.rand({2*dim}) * 100
                for d in range({dim}):
                    box[d + {dim}] += box[d] + 1
                tree.insert(idx=i, bb=box)

            for i in range(50):
                try:
                    tree.erase(i)
                except ValueError:
                    pass

        print("SUCCESS")
        """)

        exit_code, stdout, stderr = run_in_subprocess(code)
        assert exit_code != -11, f"Process crashed with segfault. stderr: {stderr}"

    @pytest.mark.parametrize("dim", [2, 3, 4])
    def test_massive_rebuild_cycles_no_crash(self, dim):
        """Verify that rebuild cycles do not crash."""
        code = textwrap.dedent(f"""
        import numpy as np
        from python_prtree import PRTree{dim}D

        idx = np.arange(500)
        boxes = np.random.rand(500, {2*dim}).astype(np.float32) * 100
        for i in range({dim}):
            boxes[:, i + {dim}] += boxes[:, i] + 1

        tree = PRTree{dim}D(idx, boxes)

        # Rebuild cycles (reduced for CI performance)
        for _ in range(10):
            tree.rebuild()

        print("SUCCESS")
        """)

        exit_code, stdout, stderr = run_in_subprocess(code)
        assert exit_code != -11, f"Process crashed with segfault. stderr: {stderr}"


class TestBoundaryConditions:
    """Test boundary condition crashes."""

    @pytest.mark.parametrize("dim", [2, 3, 4])
    def test_query_intersections_on_empty_no_crash(self, dim):
        """Verify that calling query_intersections on empty tree does not crash."""
        code = textwrap.dedent(f"""
        from python_prtree import PRTree{dim}D

        tree = PRTree{dim}D()

        # Should not crash
        pairs = tree.query_intersections()
        assert pairs.shape == (0, 2)

        print("SUCCESS")
        """)

        exit_code, stdout, stderr = run_in_subprocess(code)
        assert exit_code != -11, f"Process crashed with segfault. stderr: {stderr}"
        assert "SUCCESS" in stdout

    @pytest.mark.parametrize("dim", [2, 3, 4])
    def test_batch_query_empty_array_no_crash(self, dim):
        """Verify that calling batch_query with empty array does not crash."""
        code = textwrap.dedent(f"""
        import numpy as np
        from python_prtree import PRTree{dim}D

        idx = np.arange(10)
        boxes = np.random.rand(10, {2*dim}) * 100
        for i in range({dim}):
            boxes[:, i + {dim}] += boxes[:, i] + 1

        tree = PRTree{dim}D(idx, boxes)

        # Empty query array
        queries = np.empty((0, {2*dim}))
        results = tree.batch_query(queries)

        assert len(results) == 0

        print("SUCCESS")
        """)

        exit_code, stdout, stderr = run_in_subprocess(code)
        assert exit_code != -11, f"Process crashed with segfault. stderr: {stderr}"
        assert "SUCCESS" in stdout


class TestObjectPicklingSafety:
    """Test object pickling/unpickling safety."""

    @pytest.mark.parametrize("dim", [2, 3, 4])
    def test_unpicklable_object_no_crash(self, dim):
        """Verify that unpicklable object does not crash."""
        code = textwrap.dedent(f"""
        import numpy as np
        from python_prtree import PRTree{dim}D
        import threading

        tree = PRTree{dim}D()

        box = np.zeros({2*dim})
        for i in range({dim}):
            box[i] = 0.0
            box[i + {dim}] = 1.0

        # Try to insert unpicklable object (threading.Lock)
        try:
            tree.insert(idx=1, bb=box, obj=threading.Lock())
        except (TypeError, AttributeError, RuntimeError):
            pass  # Error is expected

        print("SUCCESS")
        """)

        exit_code, stdout, stderr = run_in_subprocess(code)
        assert exit_code != -11, f"Process crashed with segfault. stderr: {stderr}"

    @pytest.mark.parametrize("dim", [2, 3, 4])
    def test_deeply_nested_object_no_crash(self, dim):
        """Verify that deeply nested object does not crash."""
        code = textwrap.dedent(f"""
        import numpy as np
        from python_prtree import PRTree{dim}D

        tree = PRTree{dim}D()

        box = np.zeros({2*dim})
        for i in range({dim}):
            box[i] = 0.0
            box[i + {dim}] = 1.0

        # Create deeply nested object
        obj = {{"level": 0}}
        current = obj
        for i in range(100):
            current["next"] = {{"level": i + 1}}
            current = current["next"]

        try:
            tree.insert(idx=1, bb=box, obj=obj)
            # Query with return_obj
            result = tree.query(box, return_obj=True)
        except (RecursionError, RuntimeError):
            pass  # Error is acceptable

        print("SUCCESS")
        """)

        exit_code, stdout, stderr = run_in_subprocess(code)
        assert exit_code != -11, f"Process crashed with segfault. stderr: {stderr}"


class TestMultipleTreeInteraction:
    """Test interaction between multiple tree instances."""

    @pytest.mark.parametrize("dim", [2, 3, 4])
    def test_cross_tree_operations_no_crash(self, dim):
        """Verify that operations across multiple trees do not crash."""
        code = textwrap.dedent(f"""
        import numpy as np
        from python_prtree import PRTree{dim}D

        # Create multiple trees
        trees = []
        for _ in range(10):
            idx = np.arange(50)
            boxes = np.random.rand(50, {2*dim}) * 100
            for i in range({dim}):
                boxes[:, i + {dim}] += boxes[:, i] + 1
            trees.append(PRTree{dim}D(idx, boxes))

        # Query all trees
        query_box = np.random.rand({2*dim}) * 100
        for i in range({dim}):
            query_box[i + {dim}] += query_box[i] + 1

        for tree in trees:
            result = tree.query(query_box)

        # Delete some trees
        del trees[::2]

        # Query remaining trees
        for tree in trees:
            result = tree.query(query_box)

        print("SUCCESS")
        """)

        exit_code, stdout, stderr = run_in_subprocess(code)
        assert exit_code != -11, f"Process crashed with segfault. stderr: {stderr}"


class TestRaceConditions:
    """Test potential race condition scenarios (single-threaded)."""

    @pytest.mark.parametrize("dim", [2, 3, 4])
    def test_save_during_iteration_no_crash(self, dim):
        """Verify that save during iteration does not crash."""
        code = textwrap.dedent(f"""
        import numpy as np
        import tempfile
        import os
        from python_prtree import PRTree{dim}D

        idx = np.arange(100)
        boxes = np.random.rand(100, {2*dim}) * 100
        for i in range({dim}):
            boxes[:, i + {dim}] += boxes[:, i] + 1

        tree = PRTree{dim}D(idx, boxes)

        with tempfile.NamedTemporaryFile(delete=False, suffix='.bin') as f:
            fname = f.name

        try:
            # Query while saving
            for i in range(10):
                tree.query(boxes[i])
                if i == 5:
                    tree.save(fname)
                tree.query(boxes[i])
        finally:
            if os.path.exists(fname):
                os.unlink(fname)

        print("SUCCESS")
        """)

        exit_code, stdout, stderr = run_in_subprocess(code)
        assert exit_code != -11, f"Process crashed with segfault. stderr: {stderr}"
