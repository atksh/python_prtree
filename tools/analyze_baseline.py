#!/usr/bin/env python3
"""
Phase 0: Baseline Analysis Script
Parses profiling data and generates summary reports
"""

import re
import os
import sys
from pathlib import Path
from collections import defaultdict
import csv

def parse_perf_stat(filename):
    """Parse perf stat output and extract key metrics"""
    metrics = {}

    if not os.path.exists(filename):
        return metrics

    with open(filename, 'r') as f:
        content = f.read()

    patterns = {
        'cycles': r'([\d,]+)\s+cycles',
        'instructions': r'([\d,]+)\s+instructions',
        'cache_references': r'([\d,]+)\s+cache-references',
        'cache_misses': r'([\d,]+)\s+cache-misses',
        'L1_dcache_loads': r'([\d,]+)\s+L1-dcache-loads',
        'L1_dcache_load_misses': r'([\d,]+)\s+L1-dcache-load-misses',
        'LLC_loads': r'([\d,]+)\s+LLC-loads',
        'LLC_load_misses': r'([\d,]+)\s+LLC-load-misses',
        'branch_instructions': r'([\d,]+)\s+branch-instructions',
        'branch_misses': r'([\d,]+)\s+branch-misses',
        'time_seconds': r'([\d.]+)\s+seconds time elapsed',
    }

    for key, pattern in patterns.items():
        match = re.search(pattern, content)
        if match:
            value_str = match.group(1).replace(',', '')
            try:
                metrics[key] = float(value_str)
            except ValueError:
                pass

    # Calculate derived metrics
    if 'cycles' in metrics and 'instructions' in metrics and metrics['cycles'] > 0:
        metrics['ipc'] = metrics['instructions'] / metrics['cycles']

    if 'cache_references' in metrics and 'cache_misses' in metrics and metrics['cache_references'] > 0:
        metrics['cache_miss_rate'] = (metrics['cache_misses'] / metrics['cache_references']) * 100

    if 'L1_dcache_loads' in metrics and 'L1_dcache_load_misses' in metrics and metrics['L1_dcache_loads'] > 0:
        metrics['l1_miss_rate'] = (metrics['L1_dcache_load_misses'] / metrics['L1_dcache_loads']) * 100

    if 'LLC_loads' in metrics and 'LLC_load_misses' in metrics and metrics['LLC_loads'] > 0:
        metrics['llc_miss_rate'] = (metrics['LLC_load_misses'] / metrics['LLC_loads']) * 100

    if 'branch_instructions' in metrics and 'branch_misses' in metrics and metrics['branch_instructions'] > 0:
        metrics['branch_miss_rate'] = (metrics['branch_misses'] / metrics['branch_instructions']) * 100

    return metrics

def parse_callgraph(filename, top_n=10):
    """Parse perf report callgraph and extract top functions"""
    functions = []

    if not os.path.exists(filename):
        return functions

    with open(filename, 'r') as f:
        for line in f:
            # Look for lines with percentage
            match = re.match(r'\s*([\d.]+)%\s+.*\s+\[.\]\s+(.+)', line)
            if match:
                percentage = float(match.group(1))
                function = match.group(2).strip()
                functions.append((function, percentage))

    # Sort by percentage and return top N
    functions.sort(key=lambda x: x[1], reverse=True)
    return functions[:top_n]

def generate_perf_counters_report(reports_dir, output_file):
    """Generate performance counters summary table"""
    workloads = ['small_uniform', 'large_uniform', 'clustered', 'skewed', 'sequential']

    with open(output_file, 'w') as f:
        f.write("# Performance Counter Baseline\n\n")
        f.write("## Construction Phase\n\n")

        # Construction table
        f.write("| Workload | Time (s) | Cycles (M) | IPC | L1 Miss% | LLC Miss% | Branch Miss% |\n")
        f.write("|----------|----------|------------|-----|----------|-----------|-------------|\n")

        for workload in workloads:
            perf_file = os.path.join(reports_dir, f'perf_construction_{workload}.txt')
            metrics = parse_perf_stat(perf_file)

            time_s = metrics.get('time_seconds', 0)
            cycles_m = metrics.get('cycles', 0) / 1e6
            ipc = metrics.get('ipc', 0)
            l1_miss = metrics.get('l1_miss_rate', 0)
            llc_miss = metrics.get('llc_miss_rate', 0)
            branch_miss = metrics.get('branch_miss_rate', 0)

            f.write(f"| {workload:12} | {time_s:8.2f} | {cycles_m:10.1f} | "
                   f"{ipc:3.2f} | {l1_miss:7.2f} | {llc_miss:8.2f} | {branch_miss:11.2f} |\n")

        f.write("\n## Query Phase\n\n")
        f.write("| Workload | Time (s) | L1 Miss% | LLC Miss% | Branch Miss% |\n")
        f.write("|----------|----------|----------|-----------|-------------|\n")

        for workload in workloads:
            perf_file = os.path.join(reports_dir, f'perf_query_{workload}.txt')
            metrics = parse_perf_stat(perf_file)

            time_s = metrics.get('time_seconds', 0)
            l1_miss = metrics.get('l1_miss_rate', 0)
            llc_miss = metrics.get('llc_miss_rate', 0)
            branch_miss = metrics.get('branch_miss_rate', 0)

            f.write(f"| {workload:12} | {time_s:8.2f} | {l1_miss:7.2f} | "
                   f"{llc_miss:8.2f} | {branch_miss:11.2f} |\n")

        f.write("\n*Generated by analyze_baseline.py*\n")

    print(f"✓ Generated: {output_file}")

def generate_hotspots_report(reports_dir, output_file):
    """Generate hotspot analysis from callgraphs"""
    with open(output_file, 'w') as f:
        f.write("# Hotspot Analysis\n\n")

        f.write("## Construction Hotspots\n\n")
        f.write("### large_uniform workload\n\n")
        f.write("| Rank | Function | CPU Time% |\n")
        f.write("|------|----------|----------|\n")

        callgraph_file = os.path.join(reports_dir, 'callgraph_benchmark_construction_large_uniform.txt')
        hotspots = parse_callgraph(callgraph_file, top_n=10)

        for i, (func, pct) in enumerate(hotspots, 1):
            f.write(f"| {i:4} | {func:50} | {pct:8.2f} |\n")

        f.write("\n## Query Hotspots\n\n")
        f.write("### large_uniform workload\n\n")
        f.write("| Rank | Function | CPU Time% |\n")
        f.write("|------|----------|----------|\n")

        callgraph_file = os.path.join(reports_dir, 'callgraph_benchmark_query_large_uniform.txt')
        hotspots = parse_callgraph(callgraph_file, top_n=10)

        for i, (func, pct) in enumerate(hotspots, 1):
            f.write(f"| {i:4} | {func:50} | {pct:8.2f} |\n")

        f.write("\n*Generated by analyze_baseline.py*\n")

    print(f"✓ Generated: {output_file}")

def check_baseline_completeness(baseline_dir):
    """Check if all required artifacts are present"""
    required_files = [
        'reports/perf_construction_large_uniform.txt',
        'reports/perf_query_large_uniform.txt',
        'system_info.txt',
    ]

    required_dirs = [
        'reports',
        'flamegraphs',
    ]

    print("\nBaseline Completeness Check:")
    print("=" * 60)

    all_present = True

    for dirname in required_dirs:
        path = os.path.join(baseline_dir, dirname)
        if os.path.exists(path):
            print(f"✓ Directory exists: {dirname}")
        else:
            print(f"✗ Missing directory: {dirname}")
            all_present = False

    for filename in required_files:
        path = os.path.join(baseline_dir, filename)
        if os.path.exists(path):
            print(f"✓ File exists: {filename}")
        else:
            print(f"✗ Missing file: {filename}")
            all_present = False

    print("=" * 60)

    if all_present:
        print("✓ Baseline artifacts complete")
    else:
        print("✗ Baseline incomplete - run profile_all_workloads.sh")

    return all_present

def main():
    # Find repository root
    script_dir = Path(__file__).parent
    repo_root = script_dir.parent
    baseline_dir = repo_root / "docs" / "baseline"
    reports_dir = baseline_dir / "reports"

    print("PRTree Phase 0: Baseline Analysis")
    print("=" * 60)
    print()

    if not baseline_dir.exists():
        print(f"Error: Baseline directory not found: {baseline_dir}")
        sys.exit(1)

    # Check completeness
    if not check_baseline_completeness(baseline_dir):
        print("\nPlease run profiling first:")
        print("  ./scripts/profile_all_workloads.sh")
        sys.exit(1)

    print("\nGenerating analysis reports...")
    print()

    # Generate reports
    perf_counters_file = baseline_dir / "perf_counters.md"
    generate_perf_counters_report(reports_dir, perf_counters_file)

    hotspots_file = baseline_dir / "hotspots.md"
    generate_hotspots_report(reports_dir, hotspots_file)

    print()
    print("=" * 60)
    print("Analysis complete!")
    print("=" * 60)
    print()
    print("Generated files:")
    print(f"  - {perf_counters_file}")
    print(f"  - {hotspots_file}")
    print()
    print("Next steps:")
    print("  1. Review generated reports")
    print("  2. Open flamegraphs in browser")
    print("  3. Fill out docs/baseline/BASELINE_SUMMARY.md")
    print("  4. Commit baseline data to git")
    print()

if __name__ == '__main__':
    main()
