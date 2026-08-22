from pathlib import Path
import pandas as pd,matplotlib;matplotlib.use('Agg');import matplotlib.pyplot as plt
R=Path('results');F=R/'figures';F.mkdir(exist_ok=True)
b=pd.read_csv(R/'v11_2_confidence_accuracy_buckets.csv');b=b[(b.scheme=='fixed_probability')&(b.scope=='combined')];plt.figure(figsize=(8,4.5));plt.bar(b.bucket,b.accuracy*100,color='#4da3ff');plt.plot(b.bucket,b.expected_accuracy*100,'o--',color='#f4b860',label='Mean predicted confidence');plt.ylabel('Winner accuracy (%)');plt.xticks(rotation=30,ha='right');plt.legend();plt.tight_layout();plt.savefig(F/'site_winner_accuracy_by_confidence.png',dpi=180);plt.close()
l=pd.read_csv(R/'v11_2_confidence_cumulative_levels.csv');l=l[(l.measure=='favorite_probability')&(l.threshold==.6)&(l.scope!='combined')];plt.figure(figsize=(7,4));plt.bar(l.scope,l.accuracy*100,color='#67d5b5');plt.axhline(60,color='#94a3b8',ls='--');plt.ylim(50,72);plt.ylabel('Accuracy for ≥60% forecasts (%)');plt.tight_layout();plt.savefig(F/'site_high_confidence_accuracy_by_season.png',dpi=180);plt.close();print('Website figures generated.')

