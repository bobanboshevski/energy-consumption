from datetime import datetime
import sys
import json
import great_expectations as gx
from pathlib import Path
from great_expectations.core.expectation_validation_result import (
    ExpectationSuiteValidationResult,
)


# ─────────────────────────────────────────────
# Report generation
# ─────────────────────────────────────────────

def generate_summary_html(checkpoint_result, output_path: Path):
    """
    Renders a self-contained single-file HTML summary from a CheckpointResult.
    Aggregates all validation results from the checkpoint run.
    Saved to reports/ so it's committed to Git and accessible from DagShub.
    """
    # Extract all validation results from the CheckpointResult
    # checkpoint.run() returns CheckpointResult with run_results dict
    validation_results = list(checkpoint_result.run_results.values())

    total = 0
    passed_count = 0
    failed_count = 0
    all_rows = ""
    overall_success = checkpoint_result["success"]

    for validation in validation_results:
        # Each value in run_results is a dict with 'validation_result' key
        result_obj = validation.get("validation_result", validation)
        stats = result_obj.statistics

        total += stats.get("evaluated_expectations", 0)
        passed_count += stats.get("successful_expectations", 0)
        failed_count += stats.get("unsuccessful_expectations", 0)

        for r in result_obj.results:
            exp_type = r.expectation_config.expectation_type
            kwargs = json.dumps(
                {k: v for k, v in r.expectation_config.kwargs.items() if k != "batch_id"},
                indent=None,
            )
            ok = r.success
            row_color = "#14532d" if ok else "#450a0a"
            icon = "✓" if ok else "✗"
            icon_color = "#22c55e" if ok else "#ef4444"

            result_detail = ""
            if not ok and r.result:
                result_detail = (
                    f'<code style="font-size:11px;color:#fca5a5">'
                    f'{json.dumps(r.result)}</code>'
                )

            all_rows += f"""
            <tr style="background:{row_color}22;border-bottom:1px solid #1f2937">
              <td style="padding:10px 12px;color:{icon_color};font-weight:bold">{icon}</td>
              <td style="padding:10px 12px;color:#e5e7eb;font-family:monospace;font-size:12px">{exp_type}</td>
              <td style="padding:10px 12px;color:#9ca3af;font-size:12px">{kwargs}</td>
              <td style="padding:10px 12px">{result_detail}</td>
            </tr>"""

    now = datetime.utcnow().strftime("%Y-%m-%d %H:%M UTC")
    status_color = "#22c55e" if overall_success else "#ef4444"
    status_label = "PASSED" if overall_success else "FAILED"

    html = f"""<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="UTF-8" />
  <meta name="viewport" content="width=device-width, initial-scale=1.0" />
  <title>GX Validation Report</title>
  <style>
    * {{ box-sizing: border-box; margin: 0; padding: 0; }}
    body {{ background: #030712; color: #f9fafb; font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', sans-serif; padding: 32px; }}
    h1 {{ font-size: 24px; font-weight: 700; margin-bottom: 4px; }}
    .meta {{ color: #6b7280; font-size: 13px; margin-bottom: 24px; }}
    .badge {{ display: inline-block; padding: 4px 12px; border-radius: 9999px; font-size: 13px; font-weight: 600; color: {status_color}; background: {status_color}22; border: 1px solid {status_color}55; margin-bottom: 24px; }}
    .stats {{ display: grid; grid-template-columns: repeat(3, 1fr); gap: 16px; max-width: 480px; margin-bottom: 32px; }}
    .stat {{ background: #111827; border: 1px solid #1f2937; border-radius: 12px; padding: 16px; }}
    .stat-val {{ font-size: 28px; font-weight: 700; font-family: monospace; }}
    .stat-label {{ font-size: 11px; color: #6b7280; text-transform: uppercase; letter-spacing: .05em; margin-top: 4px; }}
    table {{ width: 100%; border-collapse: collapse; background: #111827; border: 1px solid #1f2937; border-radius: 12px; overflow: hidden; }}
    th {{ text-align: left; padding: 10px 12px; font-size: 11px; text-transform: uppercase; letter-spacing: .05em; color: #6b7280; background: #0f172a; border-bottom: 1px solid #1f2937; }}
    code {{ background: #1f2937; padding: 2px 6px; border-radius: 4px; }}
  </style>
</head>
<body>
  <h1>⚡ Great Expectations Validation</h1>
  <p class="meta">Generated: {now}</p>
  <div class="badge">{status_label}</div>
  <div class="stats">
    <div class="stat">
      <div class="stat-val" style="color:#f9fafb">{total}</div>
      <div class="stat-label">Total</div>
    </div>
    <div class="stat">
      <div class="stat-val" style="color:#22c55e">{passed_count}</div>
      <div class="stat-label">Passed</div>
    </div>
    <div class="stat">
      <div class="stat-val" style="color:#ef4444">{failed_count}</div>
      <div class="stat-label">Failed</div>
    </div>
  </div>
  <table>
    <thead>
      <tr>
        <th style="width:40px"></th>
        <th>Expectation</th>
        <th>Parameters</th>
        <th>Detail</th>
      </tr>
    </thead>
    <tbody>{all_rows}</tbody>
  </table>
</body>
</html>"""

    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(html, encoding="utf-8")
    print(f"GX summary report saved to {output_path}")


# context = gx.get_context(context_root_dir=str(Path(__file__).parent))
context = gx.get_context()

datasource_name = "energy_demand"
data_asset_name = "energy_demand_data"

# Override base_directory so it works on any machine
# context.sources.add_or_update_pandas_filesystem(
#     name=datasource_name,
#     base_directory=str(Path(__file__).parent.parent / "data")
# )

asset = context.get_datasource(datasource_name).get_asset(data_asset_name)

# Load checkpoint
checkpoint_name = "energy_demand_checkpoint"
checkpoint = context.get_checkpoint(checkpoint_name)

# Run checkpoint
checkpoint_result = checkpoint.run(run_id="energy_demand_run")

# Build data docs (this is local)
context.build_data_docs()

# Save portable single-file summary to reports/ — this gets committed and pushed to DagShub
generate_summary_html(
    checkpoint_result=checkpoint_result,
    output_path=Path(__file__).parent.parent / "reports" / "gx_validation_report.html",
)

# Check if the checkpoint passed
if checkpoint_result["success"]:
    print("Validation passed!")
    sys.exit(0)
else:
    print("Validation failed!")
    sys.exit(1)
