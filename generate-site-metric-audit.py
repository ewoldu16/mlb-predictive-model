"""Write a traceable inventory of every headline metric shown by the site."""
from pathlib import Path
import pandas as pd
from mlb_app.performance import load_performance

ROOT = Path(__file__).resolve().parent
SOURCES = {
    "data": "results/v11_2_cv_metrics.json",
    "efficiency": "results/v11_2_cv_metrics.json + results/v11_2_computational_efficiency.csv",
    "forecast": "results/v11_2_untouched_2025_comparison.csv (V11_2_compact)",
    "confidence": "results/v11_2_confidence_overall_diagnostics.csv + results/v11_2_confidence_cumulative_levels.csv",
}

def main():
    rows = []
    for section, values in load_performance(ROOT).items():
        for metric, value in values.items():
            if isinstance(value, list):
                continue
            rows.append({"site_section": section, "metric": metric, "value": value, "source": SOURCES[section]})
    out = ROOT / "results/site_metric_source_audit.csv"
    pd.DataFrame(rows).to_csv(out, index=False)
    print(f"Saved {len(rows)} audited metrics to {out}")

if __name__ == "__main__":
    main()
