import sys
import os
from pathlib import Path
import pandas as pd
from evidently import Report
from evidently.presets import DataSummaryPreset, DataDriftPreset

CURRENT_PATH = "data/preprocessed/final_dataset.csv"
REFERENCE_PATH = "data/reference/final_dataset.csv"
REPORT_PATH = "reports/data_testing_report.html"


def generate_index_html(reports_dir: Path) -> None:
    report_files = sorted(reports_dir.glob("*.html"))
    index_html = """<!DOCTYPE html>
    <html>
    <head><title>Energy Demand Reports</title>
    <style>body{{font-family:sans-serif;max-width:800px;margin:40px auto;padding:0 20px;}}
    a{{display:block;padding:8px 0;font-size:16px;}}</style>
    </head>
    <body>
    <h1>Energy Demand Data Reports</h1>
    <p>Generated: {date}</p>
    {links}
    </body></html>""".format(
        date=pd.Timestamp.now().strftime("%Y-%m-%d %H:%M"),
        links="\n".join(
            f'<a href="{f.name}">{f.stem}</a>'
            for f in report_files if f.name != "index.html"
        )
    )
    (reports_dir / "index.html").write_text(index_html)
    print("Index page generated.")


def main():
    reports_dir = Path("reports")
    reports_dir.mkdir(parents=True, exist_ok=True)

    current = pd.read_csv(CURRENT_PATH)

    if not os.path.exists(REFERENCE_PATH):
        print(f"Reference file not found. Copying from current data to {REFERENCE_PATH}.")
        os.makedirs(os.path.dirname(REFERENCE_PATH), exist_ok=True)
        current.to_csv(REFERENCE_PATH, index=False)

    reference = pd.read_csv(REFERENCE_PATH)

    # Drop non-numeric/non-comparable columns
    drop_cols = ["Date", "is_forecast"]  # todo: We skip is_forecast right?
    for col in drop_cols:
        if col in reference.columns:
            del reference[col]
        if col in current.columns:
            del current[col]

    # Skip if all data is null
    if current.isnull().all().all():
        print("All current data is null. Skipping.")
        generate_index_html(reports_dir)
        sys.exit(0)

    report = Report([
        DataSummaryPreset(),
        DataDriftPreset(num_threshold=0.05),
    ], include_tests=True)

    result = report.run(reference_data=reference, current_data=current)
    result.save_html(REPORT_PATH)
    print(f"Report saved to {REPORT_PATH}")

    result_dict = result.dict()
    all_tests_passed = True

    if "tests" in result_dict:
        for test in result_dict["tests"]:
            test_name = test.get("name", "unknown")
            status = test.get("status", "unknown")
            p_value = test.get("parameters", {}).get("p_value", None)

            if p_value is not None:
                print(f"  {test_name}: status={status}, p={p_value:.4f} (threshold=0.01)")
            else:
                print(f"  {test_name}: status={status}")

            if status != "SUCCESS":
                all_tests_passed = False
                break

    generate_index_html(reports_dir)

    if all_tests_passed:
        print("Data tests passed. Updating reference file.")
        os.remove(REFERENCE_PATH)
        current_raw = pd.read_csv(CURRENT_PATH)
        current_raw.to_csv(REFERENCE_PATH, index=False)
        sys.exit(0)
    else:
        print("Data tests failed.")
        sys.exit(1)


if __name__ == "__main__":
    main()
