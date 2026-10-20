import os, io, re, json
from datetime import datetime, timedelta, timezone

import pandas as pd
import gspread
from google.oauth2.service_account import Credentials
from googleapiclient.discovery import build
from googleapiclient.http import MediaIoBaseDownload

# ---------- config ----------
SA_INFO   = json.loads(os.environ["GDRIVE_SA_JSON"])
FOLDER_ID = os.environ["GDRIVE_FOLDER_ID"]
SHEET_ID  = os.environ["GSHEET_ID"]
DAYS_BACK = int(os.environ.get("DAYS_BACK", 180))
WS_NAME   = "DB_Corners"
TZ_SRC    = "UTC"
TZ_DST    = "Europe/Madrid"

T_PERIOD_START = 1033
T_CORNER       = 1048

SCOPES = ["https://www.googleapis.com/auth/drive.readonly",
          "https://www.googleapis.com/auth/spreadsheets"]
creds = Credentials.from_service_account_info(SA_INFO, scopes=SCOPES)
drive = build("drive", "v3", credentials=creds)


# ---------- drive ----------
def list_files():
    q = f"'{FOLDER_ID}' in parents and name contains '_fifa_score.parquet' and trashed=false"
    out, tok = [], None
    while True:
        r = drive.files().list(q=q, fields="nextPageToken, files(id,name)", pageSize=1000,
                               pageToken=tok, supportsAllDrives=True,
                               includeItemsFromAllDrives=True).execute()
        out += r["files"]
        tok = r.get("nextPageToken")
        if not tok:
            return out


def load(fid):
    buf = io.BytesIO()
    dl = MediaIoBaseDownload(buf, drive.files().get_media(fileId=fid, supportsAllDrives=True))
    done = False
    while not done:
        _, done = dl.next_chunk()
    buf.seek(0)
    return pd.read_parquet(buf)


def file_date(name):
    try:
        return datetime.strptime(name[:10], "%Y-%m-%d").date()
    except ValueError:
        return None


cutoff = (datetime.now(timezone.utc) - timedelta(days=DAYS_BACK)).date()
files = [f for f in list_files() if (file_date(f["name"]) or cutoff) >= cutoff]
files.sort(key=lambda f: f["name"])

frames = []
for f in files:
    d = load(f["id"])
    actor = next(c for c in d.columns if c.startswith("actor"))
    d = d[["match_id", "match_date", "match_title", "league",
           "incident_type_id", "incident_timestamp", actor]].rename(columns={actor: "actor"})
    frames.append(d)

df = pd.concat(frames, ignore_index=True)
df["incident_type_id"] = pd.to_numeric(df["incident_type_id"], errors="coerce")
df["ts"] = pd.to_datetime(df["incident_timestamp"], utc=True, errors="coerce", format="ISO8601")
df = (df.drop_duplicates(["match_id", "incident_type_id", "ts", "actor"])
        .sort_values(["match_id", "ts"], kind="stable"))

# ---------- period by 1033 events ----------
df["period"] = (df["incident_type_id"].eq(T_PERIOD_START)
                .groupby(df["match_id"]).cumsum().clip(lower=1))

# ---------- match meta ----------
meta = (df.groupby("match_id")
          .agg(match_date=("match_date", "first"), match_title=("match_title", "first"),
               league=("league", "first"), periods=("period", "max"))
          .reset_index())

pat = r"^\s*(.*?)\s*\(([^()]*)\)\s*-\s*(.*?)\s*\(([^()]*)\)\s*$"
meta[["team1", "p1", "team2", "p2"]] = meta["match_title"].str.extract(pat)

# ---------- corners ----------
c = df[df["incident_type_id"].eq(T_CORNER)].copy()
c["actor"] = pd.to_numeric(c["actor"], errors="coerce").astype("Int64")
c["half"] = c["period"].map(lambda p: "h1" if p == 1 else ("h2" if p == 2 else "et"))
c["key"] = "c" + c["actor"].astype(str) + "_" + c["half"]

cols = [f"c{a}_{h}" for a in (1, 2) for h in ("h1", "h2", "et")]
cnt = (c.pivot_table(index="match_id", columns="key", values="ts", aggfunc="count", fill_value=0)
         .reindex(columns=cols, fill_value=0).reset_index())

out = meta.merge(cnt, on="match_id", how="left")
out[cols] = out[cols].fillna(0).astype(int)
out = out[out["periods"] >= 2]  # only matches reached H2

out["c1"] = out["c1_h1"] + out["c1_h2"]
out["c2"] = out["c2_h1"] + out["c2_h2"]
out["total"] = out["c1"] + out["c2"]

dt = pd.to_datetime(out["match_date"], errors="coerce", format="ISO8601")
dt = dt.dt.tz_localize(TZ_SRC) if dt.dt.tz is None else dt
out["event_dt"] = dt.dt.tz_convert(TZ_DST).dt.strftime("%Y-%m-%d %H:%M:%S")

out["tournament"] = out["league"].str.replace(r"^ESB\s+", "", regex=True).str.strip()
out["pair_key"] = [("|".join(sorted([str(a).lower(), str(b).lower()])))
                   for a, b in zip(out["p1"], out["p2"])]

out = out[["match_id", "event_dt", "tournament", "team1", "p1", "team2", "p2",
           "c1_h1", "c1_h2", "c2_h1", "c2_h2", "c1_et", "c2_et",
           "c1", "c2", "total", "pair_key"]].sort_values("event_dt", ascending=False)

# ---------- write ----------
gc = gspread.authorize(creds)
sh = gc.open_by_key(SHEET_ID)
try:
    ws = sh.worksheet(WS_NAME)
except gspread.WorksheetNotFound:
    ws = sh.add_worksheet(WS_NAME, rows=1, cols=len(out.columns))

values = [out.columns.tolist()] + json.loads(out.to_json(orient="values"))
ws.clear()
ws.resize(rows=len(values), cols=len(out.columns))
ws.update(values, "A1", value_input_option="USER_ENTERED")
print(f"files={len(files)} matches={len(out)}")
