import os, io, re, json, time
from datetime import datetime, timedelta, timezone

import pandas as pd
import gspread
from google.oauth2.service_account import Credentials
from googleapiclient.discovery import build
from googleapiclient.http import MediaIoBaseDownload

# ---------- config ----------
SA_INFO   = json.loads(os.environ["GDRIVE_SA_JSON"])
FOLDER_ID = os.environ["GDRIVE_FOLDER_ID"]
SHEET_ID  = re.search(r"[-\w]{25,}", os.environ["GSHEET_ID"]).group(0)
print("SHEET_ID len:", len(SHEET_ID))
DAYS_BACK = int(os.environ.get("DAYS_BACK", 180))
WS_NAME   = "DB_Corners"
CALC_NAME = "CALC"
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

WANT = {
    "match_id":           ["match_id"],
    "match_date":         ["match_date", "match_start"],
    "match_title":        ["match_title", "match_name"],
    "league":             ["league", "tournament", "competition"],
    "incident_type_id":   ["incident_type_id", "incident_type"],
    "incident_timestamp": ["incident_timestamp", "incident_ts"],
    "actor":              ["actor"],
}


def pick(cols, prefixes):
    low = {c.lower(): c for c in cols}
    for p in prefixes:
        if p in low:
            return low[p]
    for p in prefixes:
        hit = [c for c in cols if c.lower().startswith(p)]
        if hit:
            return sorted(hit, key=len)[0]
    raise KeyError(f"no column for {prefixes}. have: {list(cols)}")


frames, colmap = [], None
for f in files:
    d = load(f["id"])
    if colmap is None:
        print("COLUMNS:", list(d.columns))
        colmap = {k: pick(d.columns, v) for k, v in WANT.items()}
        print("MAP:", colmap)
    d = d[list(colmap.values())].rename(columns={v: k for k, v in colmap.items()})
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

cov = (out.groupby("tournament")["total"]
          .agg(matches="size", avg_total="mean")
          .sort_values("matches", ascending=False))
print(cov.to_string())
dead = cov.index[cov["avg_total"] == 0].tolist()
out = out[~out["tournament"].isin(dead)]
print("dropped:", dead, "| left:", len(out))

# ---------- write DB ----------
gc = gspread.authorize(creds)
sh = gc.open_by_key(SHEET_ID)
try:
    ws = sh.worksheet(WS_NAME)
except gspread.WorksheetNotFound:
    ws = sh.add_worksheet(WS_NAME, rows=1, cols=len(out.columns))

values = [out.columns.tolist()] + json.loads(out.to_json(orient="values"))
ws.clear()
ws.resize(rows=len(values), cols=len(out.columns))

CHUNK = 5000
for i in range(0, len(values), CHUNK):
    part = values[i:i + CHUNK]
    for attempt in range(5):
        try:
            ws.update(part, f"A{i + 1}", value_input_option="USER_ENTERED")
            break
        except gspread.exceptions.APIError:
            if attempt == 4:
                raise
            time.sleep(2 ** attempt * 3)
    print(f"written {i + len(part)}/{len(values)}")

print(f"files={len(files)} matches={len(out)}")


# ---------- CALC (created once; delete tab to rebuild) ----------
H2H = (
    '=IFERROR(ARRAYFORMULA(LET('
    'db,DB_Corners!A2:Q,'
    'p1,LOWER(TRIM($B$2)),p2,LOWER(TRIM($B$4)),'
    'k,INDEX(db,,17),'
    'f,FILTER(db,((k=p1&"|"&p2)+(k=p2&"|"&p1))>0,'
    '(($B$1="*")+(INDEX(db,,3)=$B$1))>0,'
    'INDEX(db,,2)>=$B$6,INDEX(db,,2)<$B$7+1),'
    'a,LOWER(INDEX(f,,5))=p1,'
    'd,INDEX(f,,2),'
    't1,IF(a,INDEX(f,,4),INDEX(f,,6)),'
    't2,IF(a,INDEX(f,,6),INDEX(f,,4)),'
    'r,HSTACK(d,t1,t2,'
    'IF(a,INDEX(f,,8),INDEX(f,,10)),IF(a,INDEX(f,,9),INDEX(f,,11)),'
    'IF(a,INDEX(f,,10),INDEX(f,,8)),IF(a,INDEX(f,,11),INDEX(f,,9)),'
    'EXP(-LN(2)*($B$7+1-d)/$B$8)),'
    'g,FILTER(r,(($B$3="*")+(t1=$B$3))>0,(($B$5="*")+(t2=$B$5))>0),'
    'SORT(g,1,FALSE))),"нет H2H")'
)

MX = ('LET(l1,$C$12,l2,$C$13,'
      'i,MAKEARRAY(31,31,LAMBDA(r,c,r-1)),j,MAKEARRAY(31,31,LAMBDA(r,c,c-1)),'
      'm,POISSON.DIST(i,l1,FALSE)*POISSON.DIST(j,l2,FALSE),1/SUM(m*({cond})))')


def mx(cond):
    return '=IFERROR(ARRAYFORMULA(' + MX.replace("{cond}", cond) + '),"")'


def build_calc(sh):
    ws = sh.add_worksheet(CALC_NAME, rows=200, cols=30, index=0)
    cells = []

    def put(a1, v):
        cells.append({"range": a1, "values": [[v]]})

    # inputs
    for r, (lab, val) in enumerate([
        ("Турнир", "*"), ("Игрок 1", ""), ("Команда 1", "*"),
        ("Игрок 2", ""), ("Команда 2", "*"),
        ("Дата от", "=TODAY()-60"), ("Дата до", "=TODAY()"),
        ("Полураспад, дн", 30)], start=1):
        put(f"A{r}", lab)
        put(f"B{r}", val)

    # results
    put("A10", "Показатель"); put("B10", "Простое"); put("C10", "Затухание")
    S = lambda col: f"${col}$2:${col}"
    J, M, N, O, P, Q = S("J"), S("M"), S("N"), S("O"), S("P"), S("Q")
    res = [
        (11, "H2H матчей / эфф.",
         f"=COUNT({J})",
         f"=IFERROR(SUM({Q})^2/SUMSQ({Q}),0)"),
        (12, "λ1 (Игрок 1)",
         f'=IFERROR((SUM({M})+SUM({N}))/$B$11,"")',
         f'=IFERROR(SUMPRODUCT({Q},{M}+{N})/SUM({Q}),"")'),
        (13, "λ2 (Игрок 2)",
         f'=IFERROR((SUM({O})+SUM({P}))/$B$11,"")',
         f'=IFERROR(SUMPRODUCT({Q},{O}+{P})/SUM({Q}),"")'),
        (14, "М.О.Угл", '=IFERROR(B12+B13,"")', '=IFERROR(C12+C13,"")'),
        (15, "Разница (λ1−λ2)", '=IFERROR(B12-B13,"")', '=IFERROR(C12-C13,"")'),
        (16, "Часть H1",
         f'=IFERROR((SUM({M})+SUM({O}))/(SUM({M})+SUM({N})+SUM({O})+SUM({P})),"")',
         f'=IFERROR(SUMPRODUCT({Q},{M}+{O})/SUMPRODUCT({Q},{M}+{N}+{O}+{P}),"")'),
        (17, "Часть H2", '=IFERROR(1-B16,"")', '=IFERROR(1-C16,"")'),
        (18, "H1 доля Игрок 1",
         f'=IFERROR(SUM({M})/(SUM({M})+SUM({N})),"")',
         f'=IFERROR(SUMPRODUCT({Q},{M})/SUMPRODUCT({Q},{M}+{N}),"")'),
        (19, "H1 доля Игрок 2",
         f'=IFERROR(SUM({O})/(SUM({O})+SUM({P})),"")',
         f'=IFERROR(SUMPRODUCT({Q},{O})/SUMPRODUCT({Q},{O}+{P}),"")'),
        (20, "% ничьих факт",
         f'=IFERROR(SUMPRODUCT(ISNUMBER({J})*({M}+{N}={O}+{P}))/$B$11,"")',
         f'=IFERROR(SUMPRODUCT({Q}*({M}+{N}={O}+{P}))/SUM({Q}),"")'),
        (21, "% ничьих Пуассон",
         '=IFERROR(ARRAYFORMULA(SUMPRODUCT(POISSON.DIST(SEQUENCE(31,1,0),B12,FALSE),'
         'POISSON.DIST(SEQUENCE(31,1,0),B13,FALSE))),"")',
         '=IFERROR(ARRAYFORMULA(SUMPRODUCT(POISSON.DIST(SEQUENCE(31,1,0),C12,FALSE),'
         'POISSON.DIST(SEQUENCE(31,1,0),C13,FALSE))),"")'),
        (22, "ДнаН подсказка", '=IFERROR(B20-B21,"")', '=IFERROR(C20-C21,"")'),
    ]
    for r, lab, fb, fc in res:
        put(f"A{r}", lab); put(f"B{r}", fb); put(f"C{r}", fc)

    # market (Poisson on decayed λ, no margin)
    put("E1", "Маркет: Пуассон, λ затухание, без маржи")
    for col, lab, cond in [("E", "П1", "i>j"), ("F", "Ничья", "i=j"), ("G", "П2", "i<j")]:
        put(f"{col}2", lab)
        put(f"{col}3", mx(cond))

    put("E5", "Ф1"); put("F5", "Кф П1"); put("G5", "Ф2"); put("H5", "Кф П2")
    for r, h in zip(range(6, 10), (-1.5, -0.5, 0.5, 1.5)):
        put(f"E{r}", h)
        put(f"F{r}", mx(f"i+$E${r}>j"))
        put(f"G{r}", f"=-E{r}")
        put(f"H{r}", mx(f"j+$G${r}>i"))

    def totals(row0, title, lam, lines):
        put(f"E{row0}", title); put(f"F{row0}", "Больше"); put(f"G{row0}", "Меньше")
        for r, ln in zip(range(row0 + 1, row0 + 1 + len(lines)), lines):
            put(f"E{r}", ln)
            put(f"F{r}", f'=IFERROR(1/(1-POISSON.DIST(FLOOR(E{r}),{lam},TRUE)),"")')
            put(f"G{r}", f'=IFERROR(1/POISSON.DIST(FLOOR(E{r}),{lam},TRUE),"")')

    totals(11, "Тотал", "$C$12+$C$13", [1.5, 2.5, 3.5, 4.5, 5.5, 6.5, 7.5, 8.5])
    totals(21, "ИТ1", "$C$12", [0.5, 1.5, 2.5, 3.5])
    totals(27, "ИТ2", "$C$13", [0.5, 1.5, 2.5, 3.5])

    # H2H helper
    for col, lab in zip("JKLMNOPQ", ["Дата", "Команда P1", "Команда P2",
                                     "P1 H1", "P1 H2", "P2 H1", "P2 H2", "Вес"]):
        put(f"{col}1", lab)
    put("J2", H2H)

    # dropdown lists
    put("Z1", "Турниры")
    put("Z2", '=VSTACK("*",SORT(UNIQUE(FILTER(DB_Corners!C2:C,DB_Corners!C2:C<>""))))')
    put("AA1", "Игроки")
    put("AA2", '=IFERROR(ARRAYFORMULA(LET(t,$B$1,tr,DB_Corners!C2:C,'
               'p,VSTACK(DB_Corners!E2:E,DB_Corners!G2:G),trr,VSTACK(tr,tr),'
               'SORT(UNIQUE(FILTER(p,p<>"",((t="*")+(trr=t))>0))))),"")')
    for col, pcell in (("AB", "$B$2"), ("AC", "$B$4")):
        put(f"{col}1", "Команды")
        put(f"{col}2", '=IFERROR(ARRAYFORMULA(VSTACK("*",SORT(UNIQUE(FILTER('
                       'VSTACK(DB_Corners!D2:D,DB_Corners!F2:F),'
                       'VSTACK(LOWER(DB_Corners!E2:E),LOWER(DB_Corners!G2:G))=LOWER(TRIM('
                       + pcell + '))))))),"*")')

    ws.batch_update(cells, value_input_option="USER_ENTERED")

    # formats
    ws.batch_format([
        {"range": "B6:B7", "format": {"numberFormat": {"type": "DATE", "pattern": "dd.mm.yyyy"}}},
        {"range": "J2:J", "format": {"numberFormat": {"type": "DATE_TIME", "pattern": "dd.mm.yyyy hh:mm"}}},
        {"range": "Q2:Q", "format": {"numberFormat": {"type": "NUMBER", "pattern": "0.000"}}},
        {"range": "B11:C15", "format": {"numberFormat": {"type": "NUMBER", "pattern": "0.00"}}},
        {"range": "B16:C22", "format": {"numberFormat": {"type": "PERCENT", "pattern": "0.0%"}}},
        {"range": "E3:H31", "format": {"numberFormat": {"type": "NUMBER", "pattern": "0.00"}}},
        {"range": "A1:A22", "format": {"textFormat": {"bold": True}}},
        {"range": "A10:C10", "format": {"textFormat": {"bold": True}}},
        {"range": "E1:H1", "format": {"textFormat": {"bold": True}}},
        {"range": "J1:AC1", "format": {"textFormat": {"bold": True}}},
        {"range": "B14:C15", "format": {"backgroundColor": {"red": 1, "green": 0.85, "blue": 0.4},
                                        "textFormat": {"bold": True}}},
    ])

    # dropdowns
    def dv(row, col, src):
        return {"setDataValidation": {
            "range": {"sheetId": ws.id, "startRowIndex": row - 1, "endRowIndex": row,
                      "startColumnIndex": col - 1, "endColumnIndex": col},
            "rule": {"condition": {"type": "ONE_OF_RANGE",
                                   "values": [{"userEnteredValue": src}]},
                     "showCustomUi": True, "strict": False}}}

    sh.batch_update({"requests": [
        dv(1, 2, f"={CALC_NAME}!$Z$2:$Z"),
        dv(2, 2, f"={CALC_NAME}!$AA$2:$AA"),
        dv(3, 2, f"={CALC_NAME}!$AB$2:$AB"),
        dv(4, 2, f"={CALC_NAME}!$AA$2:$AA"),
        dv(5, 2, f"={CALC_NAME}!$AC$2:$AC"),
    ]})
    print("CALC built")


if CALC_NAME not in [w.title for w in sh.worksheets()]:
    build_calc(sh)
else:
    print("CALC exists, skipped")
