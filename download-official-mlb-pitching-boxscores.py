"""Resumably cache MLB Stats API boxscores and normalize official pitching lines."""
import argparse, json, os, random, time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
import pandas as pd
import statsapi

YEARS=range(2021,2026); ATTEMPTS=3; DELAY=.05; RETRY=2
RAW=Path("data/raw/official_mlb_pitching")
FIELDS=["gamesPlayed","gamesStarted","inningsPitched","outs","earnedRuns","runs","hits","baseOnBalls","intentionalWalks","strikeOuts","homeRuns","numberOfPitches","battersFaced","saves","saveOpportunities","holds","blownSaves","inheritedRunners","inheritedRunnersScored"]

def atomic_json(obj,path):
    tmp=path.with_suffix(".tmp");tmp.write_text(json.dumps(obj),encoding="utf-8");os.replace(tmp,path)

def valid_boxscore(b):
    if not isinstance(b,dict) or "teams" not in b:return False
    for side in ("away","home"):
        t=b["teams"].get(side,{})
        if not t.get("pitchers") or not t.get("players"):return False
        for pid in t["pitchers"]:
            if "pitching" not in t["players"].get("ID"+str(pid),{}).get("stats",{}):return False
    return True

def get_game(gid,path):
    if path.exists():
        try:
            b=json.loads(path.read_text(encoding="utf-8"))
            if valid_boxscore(b):return b
        except Exception:pass
        path.rename(path.with_suffix(".invalid.json"))
    for attempt in range(1,ATTEMPTS+1):
        try:
            b=statsapi.get("game_boxscore",{"gamePk":int(gid)})
            if not valid_boxscore(b):raise ValueError("missing teams/pitching stats")
            atomic_json(b,path);time.sleep(DELAY);return b
        except Exception as e:
            print(f"game {gid} attempt {attempt}/{ATTEMPTS}: {e}",flush=True)
            if attempt==ATTEMPTS:raise RuntimeError(f"UNRECOVERABLE GAME {gid}") from e
            time.sleep(RETRY*attempt)

def extract(gid,date,b):
    rows=[]
    for side in ("away","home"):
        t=b["teams"][side]; ids=[int(x) for x in t["pitchers"]]
        for order,pid in enumerate(ids):
            p=t["players"]["ID"+str(pid)]; s=p["stats"]["pitching"]
            r={"game_id":int(gid),"date":date,"team_side":side,"team_id":t["team"].get("id"),"team_name":t["team"].get("name"),
               "pitcher_id":pid,"pitcher_name":p["person"].get("fullName"),"pitching_order":order+1,"is_starter":order==0}
            r.update({k:s.get(k) for k in FIELDS});rows.append(r)
    return rows

def collect(year,limit=None,workers=6):
    games=pd.read_csv(f"data/raw/games_{year}.csv").sort_values(["date","game_id"])
    if limit:games=games.head(limit)
    folder=RAW/str(year);folder.mkdir(parents=True,exist_ok=True);rows=[]
    items=[(int(g.game_id),g.date,folder/f"boxscore_{g.game_id}.json") for g in games.itertuples(index=False)]
    def fetch(item):
        gid,date,path=item;return gid,date,get_game(gid,path)
    with ThreadPoolExecutor(max_workers=workers) as pool:
      for n,(gid,date,b) in enumerate(pool.map(fetch,items),1):
        rows.extend(extract(gid,date,b))
        if n%100==0:print(f"{year}: {n}/{len(games)} games",flush=True)
    d=pd.DataFrame(rows); expected=set(games.game_id); covered=set(d.game_id)
    if covered!=expected or d.duplicated(["game_id","team_side","pitcher_id"]).any():raise ValueError(f"{year} coverage/duplicate validation failed")
    for c in FIELDS:
        if c!="inningsPitched":d[c]=pd.to_numeric(d[c],errors="coerce")
    path=RAW/f"official_pitcher_game_lines_{year}.csv";d.to_csv(path,index=False)
    # Deterministic random cache-vs-normalized audit.
    audit=[]
    for gid in random.Random(1701+year).sample(sorted(expected),min(5,len(expected))):
        b=json.loads((folder/f"boxscore_{gid}.json").read_text(encoding="utf-8")); q=d[d.game_id.eq(gid)]
        api_pitchers=sum(len(b["teams"][s]["pitchers"]) for s in ("away","home"))
        api_er=sum(b["teams"][s]["players"]["ID"+str(pid)]["stats"]["pitching"].get("earnedRuns",0) for s in ("away","home") for pid in b["teams"][s]["pitchers"])
        audit.append({"year":year,"game_id":gid,"normalized_pitchers":len(q),"api_pitchers":api_pitchers,"normalized_er":q.earnedRuns.sum(),"api_er":api_er,"passed":len(q)==api_pitchers and q.earnedRuns.sum()==api_er})
    print(f"{year}: games={len(expected)} pitcher-lines={len(d)} starters={d.is_starter.sum()} missing ER={d.earnedRuns.isna().sum()} saved={path}")
    return audit

def main():
    a=argparse.ArgumentParser();a.add_argument("--year",type=int,choices=YEARS);a.add_argument("--limit",type=int);a.add_argument("--workers",type=int,default=6);x=a.parse_args();aud=[]
    for y in ([x.year] if x.year else YEARS):aud.extend(collect(y,x.limit,x.workers))
    out=RAW/((f"download_validation_{x.year}_sample.csv") if x.limit else "official_boxscore_validation_audit.csv");pd.DataFrame(aud).to_csv(out,index=False)
    if aud and not all(r["passed"] for r in aud):raise ValueError("boxscore audit failed")
if __name__=="__main__":main()
