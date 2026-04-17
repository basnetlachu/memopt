"""
Savings report exporter for memopt.

Generates per-customer savings reports in HTML (always available)
or PDF (requires reportlab).

Reports are honest:
  - Measured energy: from NVML readings
  - Estimated energy: from hit rate math
  - Unmeasured: no data available

The distinction is always shown. Never claims measured when estimated.
"""
import csv
import io
import logging
import os
import time
from typing import Optional

logger = logging.getLogger(__name__)


class SavingsReport:
    """
    Per-customer savings report. Generated from ledger data.
    Honest about measurement quality.
    """

    def __init__(
        self,
        tenant_id: str,
        period_label: str,
        totals: dict,
        source_breakdown: dict,
        config: dict,
    ):
        self.tenant_id = tenant_id
        self.period_label = period_label
        self.totals = totals
        self.source_breakdown = source_breakdown
        self.config = config
        self.generated_at = time.time()

    def to_html(self) -> str:
        """Generate HTML report. Always available — no dependencies."""
        t = self.totals
        sb = self.source_breakdown

        measured = sb.get("nvml_measured", 0)
        estimated = sb.get("estimated", 0)
        unmeasured = sb.get("unmeasured", 0)
        total_batches = measured + estimated + unmeasured

        measured_pct = (
            measured / total_batches * 100
            if total_batches > 0 else 0)

        tokens = t.get("tokens_total", 0) or t.get("tokens_generated", 0) or 0
        energy_kwh = t.get("energy_saved_kwh") or 0
        co2_kg = t.get("co2_saved_kg") or 0
        cost_usd = t.get("cost_saved_usd") or 0

        energy_note = (
            f"{measured_pct:.0f}% of batches had real NVML measurements."
            if measured_pct > 0
            else
            "Energy savings are estimated. "
            "Install pynvml on GPU nodes for real measurements.")

        cost_note = (
            "Based on configured pricing."
            if cost_usd > 0
            else
            "Set MEMOPT_COST_PER_1K_TOKENS to see dollar savings.")

        grid_intensity = self.config.get("grid_intensity", 0.233)
        co2_note = (
            f"Based on {grid_intensity} kg CO2/kWh. "
            "Configure MEMOPT_GRID_INTENSITY_KG_KWH for your region.")

        cost_display = f"${cost_usd:,.2f}" if cost_usd > 0 else "N/A"

        html = f"""<!DOCTYPE html>
<html>
<head>
<meta charset="utf-8">
<title>memopt Savings Report — {self.tenant_id}</title>
<style>
  body {{ font-family: Arial, sans-serif; margin: 40px; color: #333; }}
  h1 {{ color: #1a1a2e; }}
  h2 {{ color: #444; border-bottom: 1px solid #ddd; padding-bottom: 8px; }}
  .metric {{ display: inline-block; margin: 16px; padding: 24px;
             background: #f5f5f5; border-radius: 8px; min-width: 180px; }}
  .metric .value {{ font-size: 2em; font-weight: bold; color: #1a1a2e; }}
  .metric .label {{ color: #666; font-size: 0.9em; }}
  .note {{ background: #fff3cd; padding: 12px; border-radius: 4px;
           margin: 8px 0; font-size: 0.9em; }}
  .honest {{ background: #d1ecf1; padding: 12px; border-radius: 4px;
             margin: 8px 0; font-size: 0.9em; }}
  table {{ border-collapse: collapse; width: 100%; }}
  td, th {{ border: 1px solid #ddd; padding: 8px; text-align: left; }}
  th {{ background: #f2f2f2; }}
</style>
</head>
<body>

<h1>memopt Savings Report</h1>
<p><strong>Tenant:</strong> {self.tenant_id}</p>
<p><strong>Period:</strong> {self.period_label}</p>
<p><strong>Generated:</strong> {time.strftime('%Y-%m-%d %H:%M UTC', time.gmtime(self.generated_at))}</p>

<div class="honest">
  <strong>Measurement transparency:</strong> {energy_note}
</div>

<h2>Summary</h2>
<div>
  <div class="metric">
    <div class="value">{tokens:,}</div>
    <div class="label">Tokens Processed</div>
  </div>
  <div class="metric">
    <div class="value">{energy_kwh:.4f} kWh</div>
    <div class="label">Energy Saved</div>
  </div>
  <div class="metric">
    <div class="value">{co2_kg:.3f} kg</div>
    <div class="label">CO2 Avoided</div>
  </div>
  <div class="metric">
    <div class="value">{cost_display}</div>
    <div class="label">Compute Cost Saved</div>
  </div>
</div>

<h2>Measurement Quality</h2>
<table>
  <tr><th>Source</th><th>Batches</th><th>Description</th></tr>
  <tr><td>NVML measured</td><td>{measured:,}</td><td>Real GPU power readings via NVML</td></tr>
  <tr><td>Estimated</td><td>{estimated:,}</td><td>Calculated from GKD hit rate</td></tr>
  <tr><td>Unmeasured</td><td>{unmeasured:,}</td><td>No measurement available</td></tr>
</table>

<h2>Notes</h2>
<div class="note"><strong>Cost:</strong> {cost_note}</div>
<div class="note"><strong>Carbon:</strong> {co2_note}</div>
<div class="honest">
  <strong>About this report:</strong>
  All numbers come from real usage counters in the memopt ledger.
  Energy measurements labelled "NVML measured" come from GPU hardware
  power sensors. "Estimated" values are derived from computational models.
  This report does not constitute a certified carbon credit or regulatory
  compliance document.
</div>

<hr>
<p style="color: #999; font-size: 0.8em;">
  Generated by memopt — Sophisticates | Ledger integrity: HMAC-SHA256
</p>
</body>
</html>"""
        return html

    def to_csv_summary(self) -> str:
        """CSV summary — one row with totals."""
        output = io.StringIO()
        writer = csv.writer(output)

        writer.writerow([
            "tenant_id", "period", "tokens_generated",
            "energy_saved_kwh", "co2_saved_kg", "cost_saved_usd",
            "nvml_measured_batches", "estimated_batches",
            "unmeasured_batches", "generated_at",
        ])

        tokens = (self.totals.get("tokens_total", 0)
                  or self.totals.get("tokens_generated", 0) or 0)

        writer.writerow([
            self.tenant_id,
            self.period_label,
            tokens,
            self.totals.get("energy_saved_kwh") or "",
            self.totals.get("co2_saved_kg") or "",
            self.totals.get("cost_saved_usd") or "",
            self.source_breakdown.get("nvml_measured", 0),
            self.source_breakdown.get("estimated", 0),
            self.source_breakdown.get("unmeasured", 0),
            time.strftime("%Y-%m-%d %H:%M UTC",
                          time.gmtime(self.generated_at)),
        ])

        return output.getvalue()

    def to_pdf(self) -> Optional[bytes]:
        """
        Generate PDF report. Requires: pip install reportlab.
        Returns PDF bytes or None if reportlab not installed. Never raises.
        """
        try:
            from reportlab.lib.pagesizes import letter
            from reportlab.lib.styles import getSampleStyleSheet
            from reportlab.lib.units import inch
            from reportlab.platypus import (
                SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle)
            from reportlab.lib import colors

            buffer = io.BytesIO()
            doc = SimpleDocTemplate(buffer, pagesize=letter,
                                    rightMargin=72, leftMargin=72,
                                    topMargin=72, bottomMargin=72)

            styles = getSampleStyleSheet()
            story = []

            story.append(Paragraph("memopt Savings Report", styles["Title"]))
            story.append(Spacer(1, 12))
            story.append(Paragraph(f"Tenant: {self.tenant_id}", styles["Normal"]))
            story.append(Paragraph(f"Period: {self.period_label}", styles["Normal"]))
            story.append(Paragraph(
                f"Generated: {time.strftime('%Y-%m-%d %H:%M UTC', time.gmtime(self.generated_at))}",
                styles["Normal"]))
            story.append(Spacer(1, 24))

            measured = self.source_breakdown.get("nvml_measured", 0)
            total = sum(self.source_breakdown.values()) or 1
            measured_pct = measured / total * 100

            note = (
                f"Measurement quality: {measured_pct:.0f}% of batches used real NVML readings."
                if measured_pct > 0
                else "Measurement quality: Energy savings are estimated.")
            story.append(Paragraph(note, styles["Normal"]))
            story.append(Spacer(1, 24))

            t = self.totals
            tokens = t.get("tokens_total", 0) or t.get("tokens_generated", 0) or 0
            data = [
                ["Metric", "Value"],
                ["Tokens Processed", f"{tokens:,}"],
                ["Energy Saved", f"{t.get('energy_saved_kwh') or 0:.4f} kWh"],
                ["CO2 Avoided", f"{t.get('co2_saved_kg') or 0:.3f} kg"],
                ["Compute Cost Saved",
                 f"${t.get('cost_saved_usd') or 0:,.2f}"
                 if (t.get("cost_saved_usd") or 0) > 0
                 else "Not configured"],
            ]

            table = Table(data, colWidths=[3 * inch, 3 * inch])
            table.setStyle(TableStyle([
                ("BACKGROUND", (0, 0), (-1, 0), colors.grey),
                ("TEXTCOLOR", (0, 0), (-1, 0), colors.whitesmoke),
                ("FONTNAME", (0, 0), (-1, 0), "Helvetica-Bold"),
                ("GRID", (0, 0), (-1, -1), 0.5, colors.black),
            ]))

            story.append(Paragraph("Summary", styles["Heading2"]))
            story.append(table)
            story.append(Spacer(1, 24))

            story.append(Paragraph(
                "This report does not constitute a certified carbon credit "
                "or regulatory compliance document.",
                styles["Normal"]))

            doc.build(story)
            return buffer.getvalue()

        except ImportError:
            logger.debug("reportlab not installed. pip install reportlab for PDF export.")
            return None
        except Exception as e:
            logger.debug("PDF generation failed: %s", e)
            return None


def generate_report(
    ledger,
    tenant_id: str = "_default",
    period_hours: int = 24,
) -> SavingsReport:
    """Generate a SavingsReport from ledger data. Never raises."""
    try:
        totals = ledger.totals(
            tenant_id=tenant_id if tenant_id != "_default" else None)

        source_breakdown = totals.get(
            "energy_source_breakdown", {
                "nvml_measured": 0, "estimated": 0, "unmeasured": 0})

        config = {
            "grid_intensity": float(os.getenv(
                "MEMOPT_GRID_INTENSITY_KG_KWH", "0.233")),
            "cost_per_1k": float(os.getenv(
                "MEMOPT_COST_PER_1K_TOKENS", "0")),
        }

        period_label = (
            f"Last {period_hours}h"
            if period_hours <= 48
            else f"Last {period_hours // 24} days")

        return SavingsReport(
            tenant_id=tenant_id,
            period_label=period_label,
            totals=totals,
            source_breakdown=source_breakdown,
            config=config)

    except Exception as e:
        logger.debug("Report generation failed: %s", e)
        return SavingsReport(
            tenant_id=tenant_id, period_label="unknown",
            totals={}, source_breakdown={}, config={})
