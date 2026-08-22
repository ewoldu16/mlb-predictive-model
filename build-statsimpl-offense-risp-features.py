"""Build leakage-safe traditional and RISP offense features (2021-2025)."""
from pathlib import Path
import numpy as np
import pandas as pd

YEARS = range(2021, 2026)
RAW = Path("data/raw")
OUT = Path("data/processed")
HITS = {"single": 1, "double": 2, "triple": 3, "home_run": 4}
NO_AB = {"walk", "intent_walk", "hit_by_pitch", "sac_fly", "sac_bunt", "catcher_interf"}

def safe(n, d):
    return n / d if d > 0 else np.nan

def pa_table(year):
    use = ["game_pk","game_date","inning_topbot","events",
           "on_2b","on_3b","woba_value","woba_denom"]
    p = pd.read_csv(RAW/f"statcast_enriched_{year}.csv", usecols=use, low_memory=False)
    known = pd.read_csv(RAW/f"games_{year}.csv", usecols=["game_id","home_team","away_team"]).rename(columns={"game_id":"game_pk"})
    p = p.merge(known, on="game_pk", how="inner", validate="many_to_one")
    p = p[p.events.notna()].copy()
    p["date"] = pd.to_datetime(p.game_date)
    p["team"] = np.where(p.inning_topbot.eq("Top"), p.away_team, p.home_team)
    p["risp"] = p.on_2b.notna() | p.on_3b.notna()
    p["h"] = p.events.isin(HITS).astype(int)
    p["tb"] = p.events.map(HITS).fillna(0)
    p["bb"] = p.events.isin(["walk","intent_walk"]).astype(int)
    p["hbp"] = p.events.eq("hit_by_pitch").astype(int)
    p["sf"] = p.events.eq("sac_fly").astype(int)
    p["ab"] = (~p.events.isin(NO_AB)).astype(int)
    p["ob_num"] = p.h + p.bb + p.hbp
    p["ob_den"] = p.ab + p.bb + p.hbp + p.sf
    p["pa"] = 1
    p["woba_num"] = pd.to_numeric(p.woba_value, errors="coerce").fillna(0)
    p["woba_den"] = pd.to_numeric(p.woba_denom, errors="coerce").fillna(0)
    return p

def histories(pa):
    cols = ["pa","h","tb","ab","ob_num","ob_den","woba_num","woba_den"]
    result = {}
    for context, q in (("all", pa), ("risp", pa[pa.risp])):
        d = q.groupby(["team","date"], as_index=False)[cols].sum().sort_values(["team","date"])
        for team, g in d.groupby("team"):
            result[(team, context)] = (g.date.to_numpy(dtype="datetime64[ns]"),
                np.vstack([np.zeros(len(cols)), g[cols].cumsum().to_numpy(float)]), cols)
    return result

def counts(hist, team, context, date, days=None):
    item = hist.get((team, context))
    if item is None: return None
    dates, cum, cols = item; end = np.searchsorted(dates, np.datetime64(date), side="left")
    start = 0 if days is None else np.searchsorted(dates, np.datetime64(date-pd.Timedelta(days=days)), side="left")
    return dict(zip(cols, cum[end]-cum[start]))

def metrics(c, prefix):
    if c is None: return {f"{prefix}_{x}":np.nan for x in ["woba","avg","obp","slg","ops","pa","ab"]}
    obp=safe(c["ob_num"],c["ob_den"]); slg=safe(c["tb"],c["ab"])
    return {f"{prefix}_woba":safe(c["woba_num"],c["woba_den"]), f"{prefix}_avg":safe(c["h"],c["ab"]),
            f"{prefix}_obp":obp, f"{prefix}_slg":slg, f"{prefix}_ops":obp+slg if pd.notna(obp) and pd.notna(slg) else np.nan,
            f"{prefix}_pa":c["pa"], f"{prefix}_ab":c["ab"]}

def build(year):
    games=pd.read_csv(RAW/f"games_{year}.csv"); games["date"]=pd.to_datetime(games.date)
    hist=histories(pa_table(year)); rows=[]; audits=[]
    for r in games.itertuples(index=False):
        row={"game_id":r.game_id}
        for side,team in (("home",r.home_team),("away",r.away_team)):
            vals={}
            vals.update(metrics(counts(hist,team,"all",r.date), "season"))
            vals.update(metrics(counts(hist,team,"all",r.date,30), "l30"))
            # Existing V1/V7 layers already contain these exact wOBA rates.
            vals.pop("season_woba"); vals.pop("l30_woba")
            vals.update(metrics(counts(hist,team,"risp",r.date), "season_risp"))
            vals.update(metrics(counts(hist,team,"risp",r.date,30), "l30_risp"))
            row.update({f"{side}_{k}":v for k,v in vals.items()})
        for k in list(row):
            if k.startswith("home_") and not k.endswith(("_pa","_ab")):
                base=k[5:]; row[f"off_{base}_diff"]=row[k]-row.get(f"away_{base}",np.nan)
        rows.append(row)
        if len(audits)<8:
            used=[]
            for team in (r.home_team,r.away_team):
                for context in ("all","risp"):
                    if (team,context) in hist:
                        ds=hist[(team,context)][0]; pos=np.searchsorted(ds,np.datetime64(r.date),"left")
                        if pos: used.append(pd.Timestamp(ds[pos-1]))
            latest=max(used) if used else pd.NaT
            audits.append({"year":year,"game_id":r.game_id,"target_date":r.date.date(),"latest_source_date":latest.date() if pd.notna(latest) else None,"passed":pd.isna(latest) or latest<r.date})
    out=pd.DataFrame(rows); path=OUT/f"features_statsimpl_offense_risp_{year}.csv"; out.to_csv(path,index=False)
    print(f"{year}: games={len(out)}, duplicates={out.game_id.duplicated().sum()}, infinities={np.isinf(out.select_dtypes('number')).sum().sum()}")
    print(out.filter(regex="(_pa|_ab)$").describe().loc[["min","50%","max"]].round(1).to_string())
    return audits

def main():
    OUT.mkdir(parents=True,exist_ok=True); audit=[]
    for y in YEARS: audit.extend(build(y))
    pd.DataFrame(audit).to_csv("results/statsimpl_offense_risp_temporal_audit.csv",index=False)

if __name__ == "__main__": main()
