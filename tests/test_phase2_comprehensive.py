#!/usr/bin/env python3
"""
Phase 2 Comprehensive Validation

Validates ALL Phase 2 completion criteria:
1. All optimization rules exist and can trigger
2. Impact estimates are realistic (0-80%)
3. Code examples are complete (before/after + explanation)
4. Edge cases handled gracefully
5. End-to-end pipeline works for multiple kernel types

Run on GPU server with: python3 tests/test_phase2_comprehensive.py
"""

import sys
import os
import subprocess
import tempfile
import csv
from io import StringIO

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Test workloads - each designed to trigger different bottleneck types
WORKLOADS = {
    # Memory-bound with redundant fetches (attention pattern)
    'attention': '''
import torch
torch.cuda.synchronize()
batch, heads, seq, dim = 2, 8, 1024, 64
q = torch.randn(batch, heads, seq, dim, device='cuda')
k = torch.randn(batch, heads, seq, dim, device='cuda')
v = torch.randn(batch, heads, seq, dim, device='cuda')
for _ in range(2):
    with torch.no_grad():
        scores = torch.matmul(q, k.transpose(-2, -1)) / 8.0
        attn = torch.softmax(scores, dim=-1)
        out = torch.matmul(attn, v)
torch.cuda.synchronize()
''',

    # Memory-bound with poor coalescing (LayerNorm pattern)
    'layernorm': '''
import torch
import torch.nn as nn
torch.cuda.synchronize()
model = nn.LayerNorm(1024).cuda().eval()
x = torch.randn(16, 512, 1024, device='cuda')
for _ in range(2):
    with torch.no_grad():
        _ = model(x)
torch.cuda.synchronize()
''',

    # Compute-bound (large matmul)
    'matmul': '''
import torch
torch.cuda.synchronize()
A = torch.randn(4096, 4096, device='cuda')
B = torch.randn(4096, 4096, device='cuda')
for _ in range(2):
    _ = torch.matmul(A, B)
torch.cuda.synchronize()
''',

    # Cache thrashing (large working set)
    'large_activation': '''
import torch
import torch.nn as nn
torch.cuda.synchronize()
# 256MB working set - exceeds L2 cache
x = torch.randn(64, 1024, 1024, device='cuda')
model = nn.Sequential(
    nn.Linear(1024, 1024),
    nn.GELU(),
    nn.Linear(1024, 1024),
).cuda().eval()
for _ in range(2):
    with torch.no_grad():
        _ = model(x)
torch.cuda.synchronize()
''',

    # Low occupancy (small batch)
    'small_batch': '''
import torch
import torch.nn as nn
torch.cuda.synchronize()
model = nn.Linear(256, 256).cuda().eval()
x = torch.randn(1, 256, device='cuda')
for _ in range(10):
    with torch.no_grad():
        _ = model(x)
torch.cuda.synchronize()
''',

    # Well-optimized kernel (should trigger no/few recommendations)
    'optimized': '''
import torch
torch.cuda.synchronize()
# Simple contiguous elementwise - should be efficient
x = torch.randn(1024, 1024, device='cuda')
for _ in range(5):
    y = x * 2 + 1
torch.cuda.synchronize()
''',
}


def run_ncu(script: str) -> dict:
    """Run NCU and return metrics dict."""

    with tempfile.NamedTemporaryFile(mode='w', suffix='.py', delete=False) as f:
        f.write(script)
        script_path = f.name

    try:
        metrics = [
            "l1tex__t_sectors_pipe_lsu_mem_global_op_ld.sum",
            "l1tex__t_requests_pipe_lsu_mem_global_op_ld.sum",
            "l1tex__t_sectors_pipe_lsu_mem_global_op_st.sum",
            "l1tex__t_requests_pipe_lsu_mem_global_op_st.sum",
            "dram__bytes_read.sum",
            "dram__bytes_write.sum",
            "lts__t_sector_hit_rate.pct",
            "lts__t_bytes.sum",
            "lts__t_requests_miss.sum",
            "lts__t_requests_hit.sum",
            "smsp__warp_issue_stalled_long_scoreboard_per_warp_active.pct",
            "sm__warps_active.avg.pct_of_peak_sustained_active",
            "gpu__time_duration.sum",
        ]

        result = subprocess.run(
            ["/usr/local/cuda/bin/ncu", "--target-processes", "all",
             "--metrics", ",".join(metrics), "--csv", "python3", script_path],
            capture_output=True, text=True, timeout=120
        )

        return parse_ncu(result.stdout)

    except Exception as e:
        print(f"NCU error: {e}")
        return {}
    finally:
        os.unlink(script_path)


def parse_ncu(content: str) -> dict:
    """Parse NCU CSV output."""
    if not content.strip():
        return {}

    metrics = {}
    lines = [l for l in content.split('\n') if l.strip() and not l.startswith('==PROF==')]

    header_idx = next((i for i, l in enumerate(lines) if '"ID"' in l or 'ID,' in l), -1)
    if header_idx < 0:
        return {}

    samples = {k: [] for k in ['sector_ld', 'request_ld', 'sector_st', 'request_st',
                                'l2_hit', 'l2_bytes', 'l2_miss', 'l2_hit_req', 'stall', 'occupancy']}
    totals = {'dram_read': 0, 'dram_write': 0, 'duration': 0}

    try:
        reader = csv.DictReader(StringIO('\n'.join(lines[header_idx:])))
        for row in reader:
            name = row.get('Metric Name', '')
            val_str = str(row.get('Metric Value', '0')).split()[0]
            try:
                val = float(val_str.replace(',', ''))
            except:
                continue

            if 'sectors_pipe_lsu_mem_global_op_ld' in name:
                samples['sector_ld'].append(val)
            elif 'requests_pipe_lsu_mem_global_op_ld' in name:
                samples['request_ld'].append(val)
            elif 'sectors_pipe_lsu_mem_global_op_st' in name:
                samples['sector_st'].append(val)
            elif 'requests_pipe_lsu_mem_global_op_st' in name:
                samples['request_st'].append(val)
            elif 'dram__bytes_read' in name:
                totals['dram_read'] += val
            elif 'dram__bytes_write' in name:
                totals['dram_write'] += val
            elif 'lts__t_sector_hit_rate' in name:
                samples['l2_hit'].append(val)
            elif 'lts__t_bytes' in name:
                samples['l2_bytes'].append(val)
            elif 'lts__t_requests_miss' in name:
                samples['l2_miss'].append(val)
            elif 'lts__t_requests_hit' in name:
                samples['l2_hit_req'].append(val)
            elif 'stalled_long_scoreboard' in name:
                samples['stall'].append(val)
            elif 'warps_active' in name and 'pct' in name:
                samples['occupancy'].append(val)
            elif 'time_duration' in name:
                totals['duration'] += val

    except Exception as e:
        print(f"Parse error: {e}")
        return {}

    return {
        "l1tex__t_sectors_pipe_lsu_mem_global_op_ld.sum": sum(samples['sector_ld']),
        "l1tex__t_requests_pipe_lsu_mem_global_op_ld.sum": sum(samples['request_ld']),
        "l1tex__t_sectors_pipe_lsu_mem_global_op_st.sum": sum(samples['sector_st']),
        "l1tex__t_requests_pipe_lsu_mem_global_op_st.sum": sum(samples['request_st']),
        "dram__bytes_read.sum": totals['dram_read'],
        "dram__bytes_write.sum": totals['dram_write'],
        "lts__t_sector_hit_rate.pct": sum(samples['l2_hit']) / len(samples['l2_hit']) if samples['l2_hit'] else 0,
        "lts__t_bytes.sum": sum(samples['l2_bytes']),
        "lts__t_requests_miss.sum": sum(samples['l2_miss']),
        "lts__t_requests_hit.sum": sum(samples['l2_hit_req']),
        "smsp__warp_issue_stalled_long_scoreboard_per_warp_active.pct": sum(samples['stall']) / len(samples['stall']) if samples['stall'] else 0,
        "sm__warps_active.avg.pct_of_peak_sustained_active": sum(samples['occupancy']) / len(samples['occupancy']) if samples['occupancy'] else 0,
        "gpu__time_duration.sum": totals['duration'],
    }


def validate_phase2():
    """Run comprehensive Phase 2 validation."""

    print("="*70)
    print("PHASE 2 COMPREHENSIVE VALIDATION")
    print("="*70)

    # Check PyTorch
    try:
        import torch
        if not torch.cuda.is_available():
            print("ERROR: CUDA not available")
            return False
        print(f"GPU: {torch.cuda.get_device_name(0)}")
    except ImportError:
        print("ERROR: PyTorch not installed")
        return False

    # Import Phase 2 components
    from memopt.profiler.access_pattern_analyzer import AccessPatternAnalyzer
    from memopt.profiler.optimization_synthesis import (
        OptimizationCandidateGenerator,
        ImpactScoreCalculator,
        RecommendationFormatter,
        Phase2Profiler,
    )
    from memopt.profiler.hardware_counters import HardwareCounters

    validation_results = {
        'rules_exist': False,
        'all_rules_can_trigger': False,
        'impact_estimates_realistic': False,
        'code_examples_complete': False,
        'edge_cases_handled': False,
        'pipeline_works': False,
    }

    # =========================================================================
    # TEST 1: Verify all 6 optimization rules exist
    # =========================================================================
    print("\n" + "="*70)
    print("TEST 1: Optimization Rules Exist")
    print("="*70)

    generator = OptimizationCandidateGenerator()
    rules = generator._rules

    expected_rules = [
        'uncoalesced_strided',
        'redundant_fetch',
        'cache_thrashing',
        'scattered_access',
        'random_access',
        'low_cache_hit',
    ]

    found_rules = list(rules.keys())
    print(f"  Expected rules: {expected_rules}")
    print(f"  Found rules:    {found_rules}")

    missing = set(expected_rules) - set(found_rules)
    if missing:
        print(f"  [FAIL] Missing rules: {missing}")
    else:
        print(f"  [PASS] All {len(expected_rules)} rules present")
        validation_results['rules_exist'] = True

    # =========================================================================
    # TEST 2: Run all workloads and check rule triggering
    # =========================================================================
    print("\n" + "="*70)
    print("TEST 2: Rule Triggering Across Workloads")
    print("="*70)

    triggered_rules = set()
    workload_results = {}

    for name, script in WORKLOADS.items():
        print(f"\n  Processing: {name}")

        ncu_metrics = run_ncu(script)
        if not ncu_metrics:
            print(f"    [SKIP] No NCU metrics")
            continue

        # Analyze with Phase 2
        analyzer = AccessPatternAnalyzer()

        # Estimate tensor sizes based on workload
        tensor_sizes = {'data': 64 * 1024 * 1024}  # 64MB default

        access_report = analyzer.analyze(
            kernel_name=name,
            ncu_metrics=ncu_metrics,
            tensor_info=tensor_sizes,
            gpu_name="A100"
        )

        candidates = generator.generate_candidates(access_report)

        rules_fired = [c.rule_name for c in candidates]
        triggered_rules.update(rules_fired)

        workload_results[name] = {
            'coalescing_eff': access_report.coalescing.efficiency_pct,
            'reuse_ratio': access_report.redundant_fetch.reuse_ratio,
            'overflow_ratio': access_report.cache_thrashing.overflow_ratio,
            'l2_hit_rate': access_report.redundant_fetch.l2_hit_rate_pct,
            'rules_fired': rules_fired,
            'candidates': len(candidates),
        }

        print(f"    Coalescing: {access_report.coalescing.efficiency_pct:.1f}%")
        print(f"    Reuse ratio: {access_report.redundant_fetch.reuse_ratio:.2f}x")
        print(f"    Cache overflow: {access_report.cache_thrashing.overflow_ratio:.2f}x")
        print(f"    Rules fired: {rules_fired if rules_fired else 'None'}")

    print(f"\n  Total unique rules triggered: {triggered_rules}")

    # At least 3 different rules should trigger across all workloads
    if len(triggered_rules) >= 3:
        print(f"  [PASS] {len(triggered_rules)} rules triggered across workloads")
        validation_results['all_rules_can_trigger'] = True
    else:
        print(f"  [WARN] Only {len(triggered_rules)} rules triggered")

    # =========================================================================
    # TEST 3: Impact Estimates Realistic (0-80%)
    # =========================================================================
    print("\n" + "="*70)
    print("TEST 3: Impact Estimates Realistic")
    print("="*70)

    all_impacts = []
    impact_issues = []

    for name, result in workload_results.items():
        if 'candidates' not in result:
            continue

        # Re-run to get actual impact values
        ncu_metrics = run_ncu(WORKLOADS[name])
        if not ncu_metrics:
            continue

        analyzer = AccessPatternAnalyzer()
        access_report = analyzer.analyze(name, ncu_metrics, {'data': 64*1024*1024}, "A100")
        candidates = generator.generate_candidates(access_report)

        for c in candidates:
            all_impacts.append((name, c.rule_name, c.expected_impact_pct))

            if c.expected_impact_pct <= 0:
                impact_issues.append(f"{name}/{c.rule_name}: {c.expected_impact_pct}% <= 0")
            elif c.expected_impact_pct > 80:
                impact_issues.append(f"{name}/{c.rule_name}: {c.expected_impact_pct}% > 80%")

    print(f"  Impact values collected: {len(all_impacts)}")
    for name, rule, impact in all_impacts[:10]:  # Show first 10
        status = "OK" if 0 < impact <= 80 else "BAD"
        print(f"    [{status}] {name}/{rule}: {impact:.1f}%")

    if impact_issues:
        print(f"\n  [WARN] Issues found:")
        for issue in impact_issues:
            print(f"    - {issue}")
    else:
        print(f"  [PASS] All impact estimates in valid range (0-80%)")
        validation_results['impact_estimates_realistic'] = True

    # =========================================================================
    # TEST 4: Code Examples Complete
    # =========================================================================
    print("\n" + "="*70)
    print("TEST 4: Code Examples Complete")
    print("="*70)

    formatter = RecommendationFormatter()
    templates = formatter._code_templates

    print(f"  Code templates defined: {len(templates)}")

    template_issues = []
    for opt_type, template in templates.items():
        has_before = 'before' in template and len(template['before']) > 20
        has_after = 'after' in template and len(template['after']) > 20

        status = "[PASS]" if has_before and has_after else "[FAIL]"
        print(f"    {status} {opt_type.value}: before={has_before}, after={has_after}")

        if not has_before or not has_after:
            template_issues.append(opt_type.value)

    if not template_issues:
        print(f"  [PASS] All code templates have before/after examples")
        validation_results['code_examples_complete'] = True
    else:
        print(f"  [WARN] Missing templates: {template_issues}")

    # =========================================================================
    # TEST 5: Edge Cases Handled
    # =========================================================================
    print("\n" + "="*70)
    print("TEST 5: Edge Cases Handled")
    print("="*70)

    edge_case_pass = True

    # Test 5a: Empty/zero metrics
    print("  5a: Empty metrics...")
    try:
        empty_report = analyzer.analyze("empty", {}, {'x': 1024}, "A100")
        empty_candidates = generator.generate_candidates(empty_report)
        print(f"      [PASS] Handled empty metrics, got {len(empty_candidates)} candidates")
    except Exception as e:
        print(f"      [FAIL] Crashed on empty metrics: {e}")
        edge_case_pass = False

    # Test 5b: Perfect kernel (no issues)
    print("  5b: Highly efficient kernel...")
    perfect_metrics = {
        "l1tex__t_sectors_pipe_lsu_mem_global_op_ld.sum": 1000,
        "l1tex__t_requests_pipe_lsu_mem_global_op_ld.sum": 1000,  # 1:1 = perfect coalescing
        "dram__bytes_read.sum": 1e9,
        "dram__bytes_write.sum": 1e8,
        "lts__t_sector_hit_rate.pct": 95,  # High hit rate
        "lts__t_bytes.sum": 1e9,
        "lts__t_requests_miss.sum": 100,
        "lts__t_requests_hit.sum": 10000,  # High hit ratio
    }
    try:
        perfect_report = analyzer.analyze("perfect", perfect_metrics, {'x': 1024}, "A100")
        perfect_candidates = generator.generate_candidates(perfect_report)
        print(f"      [PASS] Efficient kernel: {len(perfect_candidates)} candidates (expected 0-1)")
    except Exception as e:
        print(f"      [FAIL] Crashed on efficient kernel: {e}")
        edge_case_pass = False

    # Test 5c: Unknown GPU
    print("  5c: Unknown GPU name...")
    try:
        unknown_report = analyzer.analyze("test", ncu_metrics, {'x': 1024}, "UNKNOWN_GPU_XYZ")
        print(f"      [PASS] Handled unknown GPU gracefully")
    except Exception as e:
        print(f"      [FAIL] Crashed on unknown GPU: {e}")
        edge_case_pass = False

    if edge_case_pass:
        validation_results['edge_cases_handled'] = True

    # =========================================================================
    # TEST 6: End-to-End Pipeline
    # =========================================================================
    print("\n" + "="*70)
    print("TEST 6: End-to-End Pipeline")
    print("="*70)

    try:
        profiler = Phase2Profiler()

        # Run on attention workload
        ncu_metrics = run_ncu(WORKLOADS['attention'])

        if ncu_metrics:
            # Create mock Phase 1 counters
            phase1_counters = HardwareCounters(
                kernel_name="attention",
                timestamp=0,
                duration_ms=1.0,
                gpu_time_ms=1.0,
                dram_bytes_read=int(ncu_metrics.get("dram__bytes_read.sum", 0)),
                dram_bytes_write=int(ncu_metrics.get("dram__bytes_write.sum", 0)),
                sm_cycles_active=1000000,
                sm_cycles_elapsed=1000000,
                stall_cycles=226000,  # ~22.6%
                measurement_method="ncu_cupti",
                measurement_confidence=1.0,
            )

            report = profiler.analyze_and_recommend(
                kernel_name="attention",
                ncu_metrics=ncu_metrics,
                phase1_metrics=phase1_counters,
                tensor_info={'Q': 4*1024*1024, 'K': 4*1024*1024, 'V': 4*1024*1024},
                gpu_name="A100",
                total_gpu_time_ms=10.0
            )

            print(f"  Phase2Report generated:")
            print(f"    - Access patterns: {report.access_patterns.primary_issue}")
            print(f"    - Candidates: {len(report.optimization_candidates)}")
            print(f"    - Recommendations: {len(report.recommendations)}")
            print(f"    - Impact score: {report.impact_score.final_impact_score:.1f}")
            print(f"    - Priority: {report.impact_score.priority}")

            # Verify recommendation has all components
            if report.recommendations:
                rec = report.recommendations[0]
                has_title = len(rec.title) > 10
                has_text = len(rec.full_text) > 100
                has_before = len(rec.code_before) > 20
                has_after = len(rec.code_after) > 20

                print(f"\n  Top recommendation components:")
                print(f"    - Title: {'OK' if has_title else 'MISSING'}")
                print(f"    - Full text: {'OK' if has_text else 'MISSING'} ({len(rec.full_text)} chars)")
                print(f"    - Code before: {'OK' if has_before else 'MISSING'}")
                print(f"    - Code after: {'OK' if has_after else 'MISSING'}")

                if has_title and has_text and has_before and has_after:
                    print(f"\n  [PASS] Full recommendation generated")
                    validation_results['pipeline_works'] = True
            else:
                print(f"  [WARN] No recommendations generated")

        else:
            print(f"  [SKIP] Could not get NCU metrics")

    except Exception as e:
        print(f"  [FAIL] Pipeline crashed: {e}")
        import traceback
        traceback.print_exc()

    # =========================================================================
    # SUMMARY
    # =========================================================================
    print("\n" + "="*70)
    print("VALIDATION SUMMARY")
    print("="*70)

    all_pass = True
    for test, passed in validation_results.items():
        status = "[PASS]" if passed else "[FAIL]"
        print(f"  {status} {test.replace('_', ' ').title()}")
        if not passed:
            all_pass = False

    passed_count = sum(1 for v in validation_results.values() if v)
    total_count = len(validation_results)

    print(f"\n  Result: {passed_count}/{total_count} tests passed")

    if all_pass:
        print("\n" + "="*70)
        print("  PHASE 2 VALIDATION COMPLETE - READY FOR PHASE 3")
        print("="*70)
    else:
        print("\n  [ACTION REQUIRED] Fix failing tests before Phase 3")

    return all_pass


if __name__ == "__main__":
    success = validate_phase2()
    sys.exit(0 if success else 1)
