"""Concurrency tests for Python-level threading, multiprocessing, and async.

Tests verify that PRTree works correctly when called from:
- Multiple Python threads
- Multiple Python processes
- Async/await contexts

Note: batch_query is parallelized internally with C++ std::thread.
These tests verify Python-level concurrency safety.
"""
import asyncio
import concurrent.futures
import multiprocessing as mp
import threading
import time
import numpy as np
import pytest

from python_prtree import PRTree2D, PRTree3D, PRTree4D


class TestPythonThreading:
    """Test Python threading safety."""

    @pytest.mark.parametrize("PRTree, dim", [(PRTree2D, 2), (PRTree3D, 3), (PRTree4D, 4)])
    @pytest.mark.parametrize("num_threads", [2, 4, 8])
    def test_concurrent_queries_multiple_threads(self, PRTree, dim, num_threads):
        """複数Pythonスレッドから同時にクエリしても安全であることを確認."""
        np.random.seed(42)
        n = 1000
        idx = np.arange(n)
        boxes = np.random.rand(n, 2 * dim) * 100
        for i in range(dim):
            boxes[:, i + dim] += boxes[:, i] + 1

        tree = PRTree(idx, boxes)

        results = []
        errors = []

        def query_worker(thread_id):
            try:
                # Each thread does multiple queries
                thread_results = []
                for i in range(100):
                    query_box = np.random.rand(2 * dim) * 100
                    for d in range(dim):
                        query_box[d + dim] += query_box[d] + 1

                    result = tree.query(query_box)
                    thread_results.append(result)

                results.append((thread_id, thread_results))
            except Exception as e:
                errors.append((thread_id, e))

        threads = []
        for i in range(num_threads):
            t = threading.Thread(target=query_worker, args=(i,))
            threads.append(t)
            t.start()

        for t in threads:
            t.join()

        assert len(errors) == 0, f"Errors in threads: {errors}"
        assert len(results) == num_threads

    @pytest.mark.parametrize("PRTree, dim", [(PRTree2D, 2), (PRTree3D, 3), (PRTree4D, 4)])
    @pytest.mark.parametrize("num_threads", [2, 4])
    def test_concurrent_batch_queries_multiple_threads(self, PRTree, dim, num_threads):
        """複数Pythonスレッドから同時にbatch_queryしても安全であることを確認."""
        np.random.seed(42)
        n = 1000
        idx = np.arange(n)
        boxes = np.random.rand(n, 2 * dim) * 100
        for i in range(dim):
            boxes[:, i + dim] += boxes[:, i] + 1

        tree = PRTree(idx, boxes)

        results = []
        errors = []

        def batch_query_worker(thread_id):
            try:
                queries = np.random.rand(100, 2 * dim) * 100
                for i in range(dim):
                    queries[:, i + dim] += queries[:, i] + 1

                result = tree.batch_query(queries)
                results.append((thread_id, len(result)))
            except Exception as e:
                errors.append((thread_id, e))

        threads = []
        for i in range(num_threads):
            t = threading.Thread(target=batch_query_worker, args=(i,))
            threads.append(t)
            t.start()

        for t in threads:
            t.join()

        assert len(errors) == 0, f"Errors in threads: {errors}"
        assert len(results) == num_threads
        for thread_id, result_len in results:
            assert result_len == 100

    @pytest.mark.parametrize("PRTree, dim", [(PRTree2D, 2), (PRTree3D, 3), (PRTree4D, 4)])
    def test_read_only_concurrent_access(self, PRTree, dim):
        """読み取り専用の同時アクセスが安全であることを確認."""
        np.random.seed(42)
        n = 500
        idx = np.arange(n)
        boxes = np.random.rand(n, 2 * dim) * 100
        for i in range(dim):
            boxes[:, i + dim] += boxes[:, i] + 1

        tree = PRTree(idx, boxes)

        num_threads = 10
        queries_per_thread = 50

        def read_worker():
            for _ in range(queries_per_thread):
                query_box = np.random.rand(2 * dim) * 100
                for d in range(dim):
                    query_box[d + dim] += query_box[d] + 1
                tree.query(query_box)

        threads = [threading.Thread(target=read_worker) for _ in range(num_threads)]

        for t in threads:
            t.start()
        for t in threads:
            t.join()

        # Should complete without crash or deadlock


class TestPythonMultiprocessing:
    """Test Python multiprocessing safety."""

    @pytest.mark.parametrize("PRTree, dim", [(PRTree2D, 2), (PRTree3D, 3)])
    @pytest.mark.parametrize("num_processes", [2, 4])
    def test_concurrent_queries_multiple_processes(self, PRTree, dim, num_processes):
        """複数Pythonプロセスから同時にクエリしても安全であることを確認."""

        def query_worker(proc_id, return_dict):
            try:
                np.random.seed(proc_id)
                n = 500
                idx = np.arange(n)
                boxes = np.random.rand(n, 2 * dim) * 100
                for i in range(dim):
                    boxes[:, i + dim] += boxes[:, i] + 1

                # Each process creates its own tree
                tree = PRTree(idx, boxes)

                # Do queries
                results = []
                for i in range(50):
                    query_box = np.random.rand(2 * dim) * 100
                    for d in range(dim):
                        query_box[d + dim] += query_box[d] + 1

                    result = tree.query(query_box)
                    results.append(len(result))

                return_dict[proc_id] = sum(results)
            except Exception as e:
                return_dict[proc_id] = f"ERROR: {e}"

        manager = mp.Manager()
        return_dict = manager.dict()
        processes = []

        for i in range(num_processes):
            p = mp.Process(target=query_worker, args=(i, return_dict))
            processes.append(p)
            p.start()

        for p in processes:
            p.join(timeout=30)
            if p.is_alive():
                p.terminate()
                pytest.fail("Process timed out")

        assert len(return_dict) == num_processes
        for proc_id, result in return_dict.items():
            assert not isinstance(result, str) or not result.startswith("ERROR"), f"Process {proc_id} failed: {result}"

    @pytest.mark.parametrize("PRTree, dim", [(PRTree2D, 2)])
    def test_process_pool_queries(self, PRTree, dim):
        """ProcessPoolExecutorでのクエリが安全であることを確認."""
        np.random.seed(42)
        n = 500
        idx = np.arange(n)
        boxes = np.random.rand(n, 2 * dim) * 100
        for i in range(dim):
            boxes[:, i + dim] += boxes[:, i] + 1

        def process_query(query_data):
            tree_class, idx_data, boxes_data, query_box = query_data
            # Recreate tree in subprocess
            tree = tree_class(idx_data, boxes_data)
            return tree.query(query_box)

        # Prepare queries
        queries = []
        for _ in range(20):
            query_box = np.random.rand(2 * dim) * 100
            for d in range(dim):
                query_box[d + dim] += query_box[d] + 1
            queries.append((PRTree, idx, boxes, query_box))

        with concurrent.futures.ProcessPoolExecutor(max_workers=4) as executor:
            results = list(executor.map(process_query, queries))

        assert len(results) == 20
        for result in results:
            assert isinstance(result, list)


class TestAsyncIO:
    """Test async/await compatibility."""

    @pytest.mark.parametrize("PRTree, dim", [(PRTree2D, 2), (PRTree3D, 3), (PRTree4D, 4)])
    @pytest.mark.parametrize("num_tasks", [5, 10])
    def test_async_queries(self, PRTree, dim, num_tasks):
        """asyncコンテキストでクエリが動作することを確認."""
        np.random.seed(42)
        n = 500
        idx = np.arange(n)
        boxes = np.random.rand(n, 2 * dim) * 100
        for i in range(dim):
            boxes[:, i + dim] += boxes[:, i] + 1

        tree = PRTree(idx, boxes)

        async def async_query_worker(task_id):
            results = []
            for i in range(20):
                query_box = np.random.rand(2 * dim) * 100
                for d in range(dim):
                    query_box[d + dim] += query_box[d] + 1

                # Run query in executor to avoid blocking
                loop = asyncio.get_event_loop()
                result = await loop.run_in_executor(None, tree.query, query_box)
                results.append(result)

                # Small delay to interleave tasks
                await asyncio.sleep(0.001)

            return task_id, len(results)

        async def run_async_test():
            tasks = [async_query_worker(i) for i in range(num_tasks)]
            results = await asyncio.gather(*tasks)
            return results

        results = asyncio.run(run_async_test())

        assert len(results) == num_tasks
        for task_id, result_count in results:
            assert result_count == 20

    @pytest.mark.parametrize("PRTree, dim", [(PRTree2D, 2), (PRTree3D, 3)])
    def test_async_batch_queries(self, PRTree, dim):
        """asyncコンテキストでbatch_queryが動作することを確認."""
        np.random.seed(42)
        n = 500
        idx = np.arange(n)
        boxes = np.random.rand(n, 2 * dim) * 100
        for i in range(dim):
            boxes[:, i + dim] += boxes[:, i] + 1

        tree = PRTree(idx, boxes)

        async def async_batch_query_worker(task_id):
            queries = np.random.rand(100, 2 * dim) * 100
            for i in range(dim):
                queries[:, i + dim] += queries[:, i] + 1

            loop = asyncio.get_event_loop()
            result = await loop.run_in_executor(None, tree.batch_query, queries)
            return task_id, len(result)

        async def run_async_batch_test():
            tasks = [async_batch_query_worker(i) for i in range(5)]
            results = await asyncio.gather(*tasks)
            return results

        results = asyncio.run(run_async_batch_test())

        assert len(results) == 5
        for task_id, result_count in results:
            assert result_count == 100


class TestThreadPoolExecutor:
    """Test ThreadPoolExecutor compatibility."""

    @pytest.mark.parametrize("PRTree, dim", [(PRTree2D, 2), (PRTree3D, 3), (PRTree4D, 4)])
    @pytest.mark.parametrize("max_workers", [2, 4, 8])
    def test_thread_pool_queries(self, PRTree, dim, max_workers):
        """ThreadPoolExecutorでのクエリが安全であることを確認."""
        np.random.seed(42)
        n = 1000
        idx = np.arange(n)
        boxes = np.random.rand(n, 2 * dim) * 100
        for i in range(dim):
            boxes[:, i + dim] += boxes[:, i] + 1

        tree = PRTree(idx, boxes)

        def query_task(query_box):
            return tree.query(query_box)

        # Prepare queries
        queries = []
        for _ in range(100):
            query_box = np.random.rand(2 * dim) * 100
            for d in range(dim):
                query_box[d + dim] += query_box[d] + 1
            queries.append(query_box)

        with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as executor:
            results = list(executor.map(query_task, queries))

        assert len(results) == 100
        for result in results:
            assert isinstance(result, list)

    @pytest.mark.parametrize("PRTree, dim", [(PRTree2D, 2), (PRTree3D, 3)])
    @pytest.mark.parametrize("max_workers", [2, 4])
    def test_thread_pool_batch_queries(self, PRTree, dim, max_workers):
        """ThreadPoolExecutorでのbatch_queryが安全であることを確認."""
        np.random.seed(42)
        n = 1000
        idx = np.arange(n)
        boxes = np.random.rand(n, 2 * dim) * 100
        for i in range(dim):
            boxes[:, i + dim] += boxes[:, i] + 1

        tree = PRTree(idx, boxes)

        def batch_query_task(seed):
            np.random.seed(seed)
            queries = np.random.rand(50, 2 * dim) * 100
            for i in range(dim):
                queries[:, i + dim] += queries[:, i] + 1
            return tree.batch_query(queries)

        with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as executor:
            futures = [executor.submit(batch_query_task, i) for i in range(20)]
            results = [f.result() for f in concurrent.futures.as_completed(futures)]

        assert len(results) == 20
        for result in results:
            assert len(result) == 50


class TestConcurrentModification:
    """Test concurrent modification scenarios."""

    @pytest.mark.parametrize("PRTree, dim", [(PRTree2D, 2), (PRTree3D, 3)])
    def test_insert_from_multiple_threads_sequential(self, PRTree, dim):
        """複数スレッドから順次挿入しても安全であることを確認."""
        tree = PRTree()
        lock = threading.Lock()
        errors = []

        def insert_worker(thread_id):
            try:
                for i in range(100):
                    box = np.random.rand(2 * dim) * 100
                    for d in range(dim):
                        box[d + dim] += box[d] + 1

                    with lock:
                        tree.insert(idx=thread_id * 100 + i, bb=box)
            except Exception as e:
                errors.append((thread_id, e))

        threads = []
        for i in range(4):
            t = threading.Thread(target=insert_worker, args=(i,))
            threads.append(t)
            t.start()

        for t in threads:
            t.join()

        assert len(errors) == 0, f"Errors: {errors}"
        assert tree.size() == 400

    @pytest.mark.parametrize("PRTree, dim", [(PRTree2D, 2)])
    def test_query_during_save_load(self, PRTree, dim, tmp_path):
        """保存・読込中のクエリが安全であることを確認."""
        np.random.seed(42)
        n = 500
        idx = np.arange(n)
        boxes = np.random.rand(n, 2 * dim) * 100
        for i in range(dim):
            boxes[:, i + dim] += boxes[:, i] + 1

        tree = PRTree(idx, boxes)
        fname = tmp_path / "concurrent_tree.bin"

        query_results = []
        errors = []

        def query_worker():
            try:
                for _ in range(100):
                    query_box = np.random.rand(2 * dim) * 100
                    for d in range(dim):
                        query_box[d + dim] += query_box[d] + 1
                    result = tree.query(query_box)
                    query_results.append(len(result))
                    time.sleep(0.001)
            except Exception as e:
                errors.append(e)

        def save_load_worker():
            try:
                for i in range(5):
                    tree.save(str(fname))
                    time.sleep(0.01)
                    # Note: Loading creates new tree, doesn't affect original
                    loaded = PRTree(str(fname))
                    time.sleep(0.01)
            except Exception as e:
                errors.append(e)

        query_thread = threading.Thread(target=query_worker)
        save_thread = threading.Thread(target=save_load_worker)

        query_thread.start()
        save_thread.start()

        query_thread.join()
        save_thread.join()

        assert len(errors) == 0, f"Errors: {errors}"
        assert len(query_results) > 0


class TestDataRaceProtection:
    """Test protection against data races."""

    @pytest.mark.parametrize("PRTree, dim", [(PRTree2D, 2), (PRTree3D, 3)])
    def test_simultaneous_read_write_protected(self, PRTree, dim):
        """読み書きの同時実行が保護されていることを確認（GIL依存）."""
        n = 500
        idx = np.arange(n)
        boxes = np.random.rand(n, 2 * dim) * 100
        for i in range(dim):
            boxes[:, i + dim] += boxes[:, i] + 1

        tree = PRTree(idx, boxes)
        lock = threading.Lock()
        errors = []

        def reader():
            try:
                for _ in range(200):
                    query_box = np.random.rand(2 * dim) * 100
                    for d in range(dim):
                        query_box[d + dim] += query_box[d] + 1
                    tree.query(query_box)
                    time.sleep(0.0001)
            except Exception as e:
                errors.append(("reader", e))

        def writer():
            try:
                for i in range(50):
                    box = np.random.rand(2 * dim) * 100
                    for d in range(dim):
                        box[d + dim] += box[d] + 1

                    with lock:
                        tree.insert(idx=n + i, bb=box)
                    time.sleep(0.001)
            except Exception as e:
                errors.append(("writer", e))

        readers = [threading.Thread(target=reader) for _ in range(3)]
        writers = [threading.Thread(target=writer) for _ in range(2)]

        for t in readers + writers:
            t.start()
        for t in readers + writers:
            t.join()

        # Should complete without data race errors
        # (GIL provides some protection, but implementation should be safe)
        assert len(errors) == 0, f"Errors: {errors}"
