"""
memopt.formatters - Output Formatters

Provides multiple output formats for optimization reports:
- HTML: Professional, shareable reports
- JSON: Machine-readable export
- Text: Terminal/console output (uses existing RecommendationFormatter)
"""

from .html_formatter import HTMLReportGenerator
from .json_exporter import JSONExporter

__all__ = ['HTMLReportGenerator', 'JSONExporter']
