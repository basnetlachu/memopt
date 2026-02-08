"""
Analyze Workflow - Complete Model Analysis Pipeline

Orchestrates the full memopt analysis pipeline:
1. Profile model with Phase 1 (hardware counters, bottleneck detection)
2. Analyze with Phase 2 (access patterns, optimization synthesis)
3. Calculate ROI (business metrics)
4. Format output (text, HTML, JSON)
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional, Union
from pathlib import Path

logger = logging.getLogger("memopt.workflows")


@dataclass
class AnalysisResult:
    """Complete analysis result from the workflow."""
    model_name: str
    phase2_report: Any                     # Phase2Report
    roi_report: Optional[Any] = None       # ROIReport
    output_text: Optional[str] = None      # Formatted text output
    output_html: Optional[str] = None      # HTML report
    output_json: Optional[str] = None      # JSON export
    success: bool = True
    error: Optional[str] = None


class AnalyzeWorkflow:
    """
    High-level workflow for complete model analysis.

    Integrates all memopt components into a single, easy-to-use interface.

    Example:
        workflow = AnalyzeWorkflow(gpu_count=8, gpu_cost=3.50)
        result = workflow.analyze(model, sample_input, format='html')
        if result.success:
            print(result.output_html)
    """

    def __init__(
        self,
        gpu_count: int = 1,
        gpu_cost_per_hour: float = 3.00,
        verbose: bool = True
    ):
        """
        Initialize workflow with fleet parameters.

        Args:
            gpu_count: Number of GPUs in fleet for ROI calculation
            gpu_cost_per_hour: Cost per GPU hour in dollars
            verbose: Whether to print progress messages
        """
        self.gpu_count = gpu_count
        self.gpu_cost_per_hour = gpu_cost_per_hour
        self.verbose = verbose

    def analyze(
        self,
        model: Any,
        sample_input: Any,
        model_name: Optional[str] = None,
        output_format: str = 'text',
        output_path: Optional[str] = None,
        num_iterations: int = 5,
        use_ncu: bool = False
    ) -> AnalysisResult:
        """
        Run complete analysis pipeline on a model.

        Args:
            model: PyTorch model to analyze
            sample_input: Sample input tensor for profiling
            model_name: Optional name for the model
            output_format: Output format ('text', 'html', 'json', 'all')
            output_path: Optional path to save output file
            num_iterations: Number of profiling iterations
            use_ncu: Whether to use NCU for real CUPTI measurements

        Returns:
            AnalysisResult with all outputs
        """
        name = model_name or self._get_model_name(model)

        try:
            # Step 1: Run Phase 1 profiling
            if self.verbose:
                print(f"[1/4] Profiling model: {name}")

            phase1_report, phase1_metrics = self._run_phase1(
                model, sample_input, num_iterations, use_ncu
            )

            # Step 2: Run Phase 2 analysis
            if self.verbose:
                print("[2/4] Analyzing access patterns and generating recommendations")

            phase2_report = self._run_phase2(
                name, phase1_report, phase1_metrics, model
            )

            # Step 3: Calculate ROI
            if self.verbose:
                print("[3/4] Calculating ROI metrics")

            roi_report = self._calculate_roi(phase2_report, name)

            # Step 4: Format output
            if self.verbose:
                print(f"[4/4] Generating {output_format} output")

            result = AnalysisResult(
                model_name=name,
                phase2_report=phase2_report,
                roi_report=roi_report,
            )

            # Generate requested format(s)
            if output_format in ('text', 'all'):
                result.output_text = self._format_text(phase2_report, roi_report)

            if output_format in ('html', 'all'):
                result.output_html = self._format_html(phase2_report, roi_report, name)

            if output_format in ('json', 'all'):
                result.output_json = self._format_json(phase2_report, roi_report, name)

            # Save to file if requested
            if output_path:
                self._save_output(result, output_format, output_path)

            if self.verbose:
                print("\nAnalysis complete!")

            return result

        except Exception as e:
            logger.exception(f"Analysis failed: {e}")
            return AnalysisResult(
                model_name=name,
                phase2_report=None,
                success=False,
                error=str(e)
            )

    def analyze_from_ncu_report(
        self,
        ncu_report_path: str,
        model_name: Optional[str] = None,
        output_format: str = 'text',
        output_path: Optional[str] = None
    ) -> AnalysisResult:
        """
        Run analysis from an existing NCU report file.

        Args:
            ncu_report_path: Path to NCU report (.ncu-rep or .csv)
            model_name: Optional name for the model
            output_format: Output format ('text', 'html', 'json', 'all')
            output_path: Optional path to save output file

        Returns:
            AnalysisResult with all outputs
        """
        name = model_name or Path(ncu_report_path).stem

        try:
            from memopt.profiler import NCUProfiler

            if self.verbose:
                print(f"[1/4] Loading NCU report: {ncu_report_path}")

            ncu_profiler = NCUProfiler()
            ncu_counters = ncu_profiler.load_report(ncu_report_path)

            if self.verbose:
                print("[2/4] Analyzing access patterns and generating recommendations")

            phase2_report = self._run_phase2_from_ncu(name, ncu_counters)

            if self.verbose:
                print("[3/4] Calculating ROI metrics")

            roi_report = self._calculate_roi(phase2_report, name)

            if self.verbose:
                print(f"[4/4] Generating {output_format} output")

            result = AnalysisResult(
                model_name=name,
                phase2_report=phase2_report,
                roi_report=roi_report,
            )

            if output_format in ('text', 'all'):
                result.output_text = self._format_text(phase2_report, roi_report)

            if output_format in ('html', 'all'):
                result.output_html = self._format_html(phase2_report, roi_report, name)

            if output_format in ('json', 'all'):
                result.output_json = self._format_json(phase2_report, roi_report, name)

            if output_path:
                self._save_output(result, output_format, output_path)

            if self.verbose:
                print("\nAnalysis complete!")

            return result

        except Exception as e:
            logger.exception(f"Analysis failed: {e}")
            return AnalysisResult(
                model_name=name,
                phase2_report=None,
                success=False,
                error=str(e)
            )

    def _get_model_name(self, model: Any) -> str:
        """Extract model name from model object."""
        if hasattr(model, 'name'):
            return model.name
        if hasattr(model, '__class__'):
            return model.__class__.__name__
        return 'UnknownModel'

    def _run_phase1(
        self,
        model: Any,
        sample_input: Any,
        num_iterations: int,
        use_ncu: bool
    ) -> tuple:
        """Run Phase 1 profiling."""
        from memopt.profiler import Phase1Profiler

        phase1 = Phase1Profiler()

        # Profile the model
        if callable(sample_input):
            # sample_input is a generator function
            report = phase1.profile_model(model, sample_input, num_iterations=num_iterations)
        else:
            # sample_input is a tensor
            report = phase1.profile_model(
                model,
                lambda: sample_input,
                num_iterations=num_iterations
            )

        # Get hardware metrics for Phase 2
        metrics = None
        if hasattr(report, 'kernel_profiles') and report.kernel_profiles:
            # Get metrics from first kernel
            first_kernel = report.kernel_profiles[0]
            if hasattr(first_kernel, 'counters'):
                metrics = first_kernel.counters

        return report, metrics

    def _run_phase2(
        self,
        kernel_name: str,
        phase1_report: Any,
        phase1_metrics: Any,
        model: Any
    ) -> Any:
        """Run Phase 2 analysis."""
        from memopt.profiler import Phase2Profiler

        phase2 = Phase2Profiler()

        # Extract GPU info
        gpu_name = 'Unknown'
        if hasattr(phase1_report, 'gpu_info'):
            gpu_name = phase1_report.gpu_info.get('name', 'Unknown')

        # Extract total GPU time
        total_gpu_time = 1.0
        if hasattr(phase1_report, 'total_gpu_time_ms'):
            total_gpu_time = phase1_report.total_gpu_time_ms

        # Build tensor info from model
        tensor_info = self._extract_tensor_info(model)

        # Run Phase 2 analysis
        report = phase2.analyze_and_recommend(
            kernel_name=kernel_name,
            ncu_metrics=phase1_metrics,
            phase1_metrics=phase1_metrics,
            tensor_info=tensor_info,
            gpu_name=gpu_name,
            total_gpu_time_ms=total_gpu_time
        )

        return report

    def _run_phase2_from_ncu(self, kernel_name: str, ncu_counters: Any) -> Any:
        """Run Phase 2 analysis from NCU counters."""
        from memopt.profiler import Phase2Profiler, counters_from_ncu

        phase2 = Phase2Profiler()

        # Convert NCU counters to Phase 1 metrics format
        phase1_metrics = counters_from_ncu(ncu_counters)

        # Run analysis
        report = phase2.analyze_and_recommend(
            kernel_name=kernel_name,
            ncu_metrics=ncu_counters,
            phase1_metrics=phase1_metrics,
            tensor_info={},
            gpu_name='Unknown',
            total_gpu_time_ms=1.0
        )

        return report

    def _extract_tensor_info(self, model: Any) -> Dict[str, int]:
        """Extract tensor size information from model parameters."""
        tensor_info = {}

        try:
            for name, param in model.named_parameters():
                tensor_info[name] = param.numel() * param.element_size()
        except Exception:
            pass

        return tensor_info

    def _calculate_roi(self, phase2_report: Any, model_name: str) -> Any:
        """Calculate ROI from Phase 2 report."""
        from memopt.business import ROICalculator

        calculator = ROICalculator(
            gpu_cost_per_hour=self.gpu_cost_per_hour,
            gpu_count=self.gpu_count
        )

        return calculator.calculate_from_phase2_report(phase2_report, model_name)

    def _format_text(self, phase2_report: Any, roi_report: Any) -> str:
        """Format output as text."""
        parts = []

        # Phase 2 report has __str__ method
        if phase2_report:
            parts.append(str(phase2_report))

        # ROI report has __str__ method
        if roi_report:
            parts.append(str(roi_report))

        return '\n'.join(parts)

    def _format_html(self, phase2_report: Any, roi_report: Any, model_name: str) -> str:
        """Format output as HTML."""
        from memopt.formatters import HTMLReportGenerator

        generator = HTMLReportGenerator()
        return generator.generate(phase2_report, roi_report, model_name)

    def _format_json(self, phase2_report: Any, roi_report: Any, model_name: str) -> str:
        """Format output as JSON."""
        from memopt.formatters import JSONExporter

        exporter = JSONExporter(pretty=True)
        return exporter.export(phase2_report, roi_report, model_name)

    def _save_output(self, result: AnalysisResult, output_format: str, output_path: str):
        """Save output to file."""
        path = Path(output_path)

        if output_format == 'text' and result.output_text:
            path.write_text(result.output_text)
            if self.verbose:
                print(f"Saved text report to: {path}")

        elif output_format == 'html' and result.output_html:
            # Add .html extension if not present
            if not path.suffix:
                path = path.with_suffix('.html')
            path.write_text(result.output_html)
            if self.verbose:
                print(f"Saved HTML report to: {path}")

        elif output_format == 'json' and result.output_json:
            # Add .json extension if not present
            if not path.suffix:
                path = path.with_suffix('.json')
            path.write_text(result.output_json)
            if self.verbose:
                print(f"Saved JSON report to: {path}")

        elif output_format == 'all':
            base = path.stem
            parent = path.parent

            if result.output_text:
                (parent / f"{base}.txt").write_text(result.output_text)
            if result.output_html:
                (parent / f"{base}.html").write_text(result.output_html)
            if result.output_json:
                (parent / f"{base}.json").write_text(result.output_json)

            if self.verbose:
                print(f"Saved all formats to: {parent}/{base}.[txt|html|json]")
