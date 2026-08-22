"""Create final read-only summary from completed official-pitching experiments."""
from pathlib import Path
import pandas as pd
R=Path("results")
def deltas(path,metrics,out):
 d=pd.read_csv(path);b=d[d.experiment.eq("baseline")].set_index("validation_year");rows=[]
 for r in d[~d.experiment.eq("baseline")].itertuples(index=False):
  z={"experiment":r.experiment,"validation_year":r.validation_year}
  for m in metrics:z[f"delta_{m}"]=getattr(r,m)-b.loc[r.validation_year,m]
  rows.append(z)
 q=pd.DataFrame(rows);q.to_csv(R/out,index=False);return q
def main():
 ml=deltas("results/official_pitching_ml_development_folds.csv",["log_loss","brier","auc","accuracy"],"official_pitching_ml_development_deltas.csv")
 tt=deltas("results/official_pitching_totals_development_folds.csv",["rmse","mae","poisson_deviance"],"official_pitching_totals_development_deltas.csv")
 ms=pd.read_csv(R/"official_pitching_ml_development_summary.csv");ts=pd.read_csv(R/"official_pitching_totals_development_summary.csv");m25=pd.read_csv(R/"official_pitching_ml_untouched_2025.csv");t25=pd.read_csv(R/"official_pitching_totals_untouched_2025.csv")
 lines=["# Official MLB pitching feature experiment","","## Frozen development decisions","",
 f"- Moneyline winner: `{ms.iloc[0].experiment}` (mean Log Loss {ms.iloc[0].log_loss_mean:.6f}).",
 f"- Totals winner: `{ts.iloc[0].experiment}` (mean RMSE {ts.iloc[0].rmse_mean:.6f}).",
 "- Therefore neither existing champion is replaced.","","## Untouched 2025","",
 "### Moneyline",m25.to_csv(index=False).strip(),"","### Totals",t25.to_csv(index=False).strip(),"","## Interpretation","",
 "Raw official starter, bullpen, and combined families all worsened the primary development metric. Most official K/BB and several ERA/WHIP variables are strongly correlated with existing Statcast rates, so coefficients are unstable and do not establish incremental value. The ablation result is decisive: retaining the official family did not improve OOS performance.",
 "","The structural run-prevention family was closer, improving moneyline Log Loss in 2024 only and totals MAE in 2024 only. It worsened mean development Log Loss and RMSE, so its small 2025 improvement is diagnostic and cannot reverse the frozen development decision.","",
 "Official starter additions were less harmful than bullpen additions for moneyline; for totals, starter and bullpen additions were similarly small degradations. Neither helped consistently in at least two development years."]
 (R/"official_pitching_experiment_final_report.md").write_text("\n".join(lines),encoding="utf-8")
 print("ML development deltas:\n",ml.to_string(index=False));print("\nTotals development deltas:\n",tt.to_string(index=False));print("\nSaved final report.")
if __name__=="__main__":main()
