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
# ECF script: Corners_H2H (GSHEET_ID) is not used
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

out["diff"] = out["c1"] - out["c2"]
out["tot_h1"] = out["c1_h1"] + out["c2_h1"]
out["tot_h2"] = out["c1_h2"] + out["c2_h2"]

# A..T  (pair_key must stay in column Q = 17)
out = out[["match_id", "event_dt", "tournament", "team1", "p1", "team2", "p2",
           "c1_h1", "c1_h2", "c2_h1", "c2_h2", "c1_et", "c2_et",
           "c1", "c2", "total", "pair_key", "diff", "tot_h1", "tot_h2"]].sort_values("event_dt", ascending=False)

cov = (out.groupby("tournament")["total"]
          .agg(matches="size", avg_total="mean")
          .sort_values("matches", ascending=False))
print(cov.to_string())
dead = cov.index[cov["avg_total"] == 0].tolist()
out = out[~out["tournament"].isin(dead)]
print("dropped:", dead, "| left:", len(out))

# ---------- pairs (for dropdowns) ----------
a = out[["p1", "p2", "tournament", "team1", "team2"]].set_axis(
    ["player", "opp", "tournament", "team", "opp_team"], axis=1)
b = out[["p2", "p1", "tournament", "team2", "team1"]].set_axis(
    ["player", "opp", "tournament", "team", "opp_team"], axis=1)
pairs = (pd.concat([a, b]).dropna()
           .groupby(["player", "opp", "tournament", "team", "opp_team"]).size()
           .rename("matches").reset_index()
           .sort_values(["player", "opp", "tournament"]))
print("pairs rows:", len(pairs))

# ---------- write ----------
gc = gspread.authorize(creds)
sh = None


def write_ws(name, frame, chunk=5000, book=None):
    book = book or sh
    try:
        ws = book.worksheet(name)
    except gspread.WorksheetNotFound:
        ws = book.add_worksheet(name, rows=1, cols=len(frame.columns))
    values = [frame.columns.tolist()] + json.loads(frame.to_json(orient="values"))
    ws.clear()
    ws.resize(rows=len(values), cols=len(frame.columns))
    for i in range(0, len(values), chunk):
        part = values[i:i + chunk]
        for attempt in range(5):
            try:
                ws.update(part, f"A{i + 1}", value_input_option="USER_ENTERED")
                break
            except gspread.exceptions.APIError:
                if attempt == 4:
                    raise
                time.sleep(2 ** attempt * 3)
    print(f"{name}: written {len(values) - 1} rows")


print(f"files={len(files)} matches={len(out)}")


# ---------- CALC (created once; delete tab to rebuild) ----------
# 1:1 port of "Console 1" (2x6 ESB) to corners.
#   A:E   left table  = H2H all tournaments, raw counts in "Выборка дней"
#   G:K   right table = H2H in tournament A1
#   M:N   window / raw averages / МО (!!!) / Разница (!!!)
#   P:AB  players summary, halves, Poisson / Final result
#   AH:AI settings (rows 9-30) + model chain (rows 36-101)
#   AK+   helper blocks, chart data, dropdown lists

TOUR, T1, T2, PL1, PL2 = "$A$1", "$A$2", "$D$2", "$A$3", "$D$3"
WIN, DFROM, DTO, GWIN = "$M$2", "$N$2", "$N$4", "$M$4"
# settings (same cells as Console 1)
BAY, LIN1, MAR2, LIN2, DSTAR, MAR3 = "$AI$10", "$AH$12", "$AI$12", "$AH$13", "$AH$14", "$AI$14"
PRES, KDEC, TRES, BASE = "$AH$17", "$AI$17", "$AH$19", "$AI$19"
MAXT, NTOUR, INFL, NTEAM, MAXPT, NPT = "$AH$21", "$AI$21", "$AH$23", "$AI$23", "$AH$25", "$AI$25"
UPD, MOPLUS, CRIT, OLD = "$AH$28", "$AI$28", "$AH$30", "$AI$30"
MODEL_DAYS = 100

DB = "DB_Corners"
dbc = lambda c: f"{DB}!${c}$2:${c}"
D_DT, D_TR, D_T1, D_P1, D_T2, D_P2 = dbc("B"), dbc("C"), dbc("D"), dbc("E"), dbc("F"), dbc("G")
D_C1, D_C2, D_TOT, D_KEY, D_DIFF, D_H1, D_H2 = dbc("N"), dbc("O"), dbc("P"), dbc("Q"), dbc("R"), dbc("S"), dbc("T")
KEY = (f'IF(LOWER(TRIM({PL1}))<LOWER(TRIM({PL2})),LOWER(TRIM({PL1}))&"|"&LOWER(TRIM({PL2})),'
       f'LOWER(TRIM({PL2}))&"|"&LOWER(TRIM({PL1})))')


def colnum(s):
    n = 0
    for ch in s:
        n = n * 26 + ord(ch) - 64
    return n


def colstr(n):
    s = ""
    while n:
        n, r = divmod(n - 1, 26)
        s = chr(65 + r) + s
    return s


R = lambda c: f"${c}$2:${c}"


def weights(dd):
    # Console 1 AM/AN: (1-AM)^2, linear tail from Линейность to min after day AH14, Old before Critical Date;
    # wm = w * МО+ before Update Date (applies to corner sums only)
    return (f'dn,INT({DTO})-INT({dd}),'
            f'am,IF(dn<{BASE},0,IF((dn-{BASE})*{KDEC}/100>1,1,(dn-{BASE})*{KDEC}/100)),'
            f'qq,(1-am)*(1-am),'
            f'prc,IF(qq<{LIN1},({MODEL_DAYS - 1}-dn)/({MODEL_DAYS - 1}-{DSTAR})*({LIN1}-{LIN2})+{LIN2},qq),'
            f'ww,IF(INT({dd})<{CRIT},prc*{OLD},prc),'
            f'wm,IF(INT({dd})<{UPD},ww*{MOPLUS},ww),')


def h2h_table(use_tour):
    # raw H2H list for nominal tables, window = Выборка дней
    tour = f'(({TOUR}="*")+(INDEX(db,,3)={TOUR}))>0,' if use_tour else ''
    return (
        '=IFERROR(ARRAYFORMULA(LET('
        f'db,{DB}!A2:Q,'
        f'pa,LOWER(TRIM({PL1})),pb,LOWER(TRIM({PL2})),'
        'kk,INDEX(db,,17),'
        'f,FILTER(db,((kk=pa&"|"&pb)+(kk=pb&"|"&pa))>0,' + tour +
        f'INDEX(db,,2)>={DFROM},INDEX(db,,2)<{DTO}+1),'
        'a,LOWER(INDEX(f,,5))=pa,'
        'SORT(HSTACK(INDEX(f,,2),IF(a,INDEX(f,,4),INDEX(f,,6)),IF(a,INDEX(f,,6),INDEX(f,,4)),'
        'IF(a,INDEX(f,,8),INDEX(f,,10)),IF(a,INDEX(f,,9),INDEX(f,,11)),'
        'IF(a,INDEX(f,,10),INDEX(f,,8)),IF(a,INDEX(f,,11),INDEX(f,,9)),INDEX(f,,3)),1,FALSE))),"нет H2H")'
    )


def h2h_model():
    # weighted H2H, all tournaments, last 100 days: date, tournament, cP1, cP2, w, wm
    return (
        '=IFERROR(ARRAYFORMULA(LET('
        f'db,{DB}!A2:Q,'
        f'pa,LOWER(TRIM({PL1})),pb,LOWER(TRIM({PL2})),'
        'kk,INDEX(db,,17),dt,INDEX(db,,2),'
        'f,FILTER(db,((kk=pa&"|"&pb)+(kk=pb&"|"&pa))>0,'
        f'INT(dt)>INT({DTO})-{MODEL_DAYS},INT(dt)<=INT({DTO})),'
        'a,LOWER(INDEX(f,,5))=pa,dd,INDEX(f,,2),'
        'xa,INDEX(f,,8)+INDEX(f,,9),xb,INDEX(f,,10)+INDEX(f,,11),'
        + weights("dd") +
        'HSTACK(dd,INDEX(f,,3),IF(a,xa,xb),IF(a,xb,xa),ww,wm))),"")'
    )


def tour_model():
    # weighted, all matches of tournament, last 100 days: date, p1, team1, p2, team2, c1, c2, w, wm
    return (
        '=IFERROR(ARRAYFORMULA(LET('
        f'db,{DB}!A2:Q,dt,INDEX(db,,2),tr,INDEX(db,,3),'
        f'f,FILTER(db,(({TOUR}="*")+(tr={TOUR}))>0,'
        f'INT(dt)>INT({DTO})-{MODEL_DAYS},INT(dt)<=INT({DTO})),'
        'dd,INDEX(f,,2),'
        + weights("dd") +
        'HSTACK(dd,INDEX(f,,5),INDEX(f,,4),INDEX(f,,7),INDEX(f,,6),'
        'INDEX(f,,8)+INDEX(f,,9),INDEX(f,,10)+INDEX(f,,11),ww,wm))),"")'
    )


def odds(p, n, mcell):
    # FIX|WEST (bayes a): (1-M) / ((1 - a*M)*p + a*M/n)
    return f"(1-{mcell})/((1-{BAY}*{mcell})*({p})+{BAY}*{mcell}/{n})"


def pois(cond, la, lb):
    return ('IFERROR(ARRAYFORMULA(LET(la,' + la + ',lb,' + lb + ','
            'i,MAKEARRAY(31,31,LAMBDA(r,c,r-1)),j,MAKEARRAY(31,31,LAMBDA(r,c,c-1)),'
            'm,POISSON.DIST(i,la,FALSE)*POISSON.DIST(j,lb,FALSE),SUM(m*(' + cond + ')))),0)')


BLUE = {"red": 0.79, "green": 0.85, "blue": 0.97}
PINK = {"red": 0.96, "green": 0.80, "blue": 0.80}
GREY = {"red": 0.95, "green": 0.95, "blue": 0.95}
MIDC = {"red": 0.86, "green": 0.82, "blue": 0.93}
GOLD = {"red": 1, "green": 0.85, "blue": 0.4}
LGOLD = {"red": 1, "green": 0.95, "blue": 0.8}
GREEN = {"red": 0.72, "green": 0.88, "blue": 0.75}

# helper column starts
HL, HR = "AK", "AT"      # table helpers (8 cols each)
HM = "BC"                # model H2H (6 cols)
HT = "BJ"                # model tournament (9 cols)
HC = "BT"                # chart data (7 cols)
HLS = "CB"               # dropdown lists (5 cols)


MARKER_CELL, MARKER = "AH1", "CALC v2 ok"   # written last -> build complete


def retry(fn, *a, **k):
    for attempt in range(6):
        try:
            return fn(*a, **k)
        except gspread.exceptions.APIError as e:
            code = getattr(getattr(e, "response", None), "status_code", 0)
            if attempt == 5 or code not in (429, 500, 502, 503):
                raise
            wait = 5 * 2 ** attempt
            print(f"API {code}, retry in {wait}s")
            time.sleep(wait)

def chart_requests(sheet_id):
    hc = [colstr(colnum(HC) + i) for i in range(7)]

    def src(col):
        return {"sourceRange": {"sources": [{"sheetId": sheet_id, "startRowIndex": 0, "endRowIndex": 71,
                                             "startColumnIndex": colnum(col) - 1, "endColumnIndex": colnum(col)}]}}

    def ser(col, rgb, width, label_col=None):
        dl = ({"type": "CUSTOM", "customLabelData": src(label_col), "placement": "ABOVE"}
              if label_col else {"type": "NONE"})
        return {"series": src(col), "targetAxis": "LEFT_AXIS",
                "colorStyle": {"rgbColor": rgb}, "color": rgb,
                "lineStyle": {"width": width, "type": "SOLID"},
                "dataLabel": dl}

    def chart(title, ycol, trend, label_col, anchor_col, vmin=None, vmax=None):
        left = {"position": "LEFT_AXIS"}
        if vmin is not None:
            left["viewWindowOptions"] = {"viewWindowMin": vmin, "viewWindowMax": vmax, "viewWindowMode": "EXPLICIT"}
        return {"addChart": {"chart": {
            "spec": {
                "title": title,
                "basicChart": {
                    "chartType": "LINE", "legendPosition": "NO_LEGEND", "headerCount": 1,
                    "axis": [{"position": "BOTTOM_AXIS"}, left],
                    "domains": [{"domain": src(hc[0])}],
                    "series": [ser(ycol, {"red": 0.26, "green": 0.52, "blue": 0.96}, 2, label_col),
                               ser(trend, {"red": 0.62, "green": 0.74, "blue": 0.98}, 2)],
                    "interpolateNulls": True,
                }},
            "position": {"overlayPosition": {
                "anchorCell": {"sheetId": sheet_id, "rowIndex": 18, "columnIndex": colnum(anchor_col) - 1},
                "widthPixels": 470, "heightPixels": 400}}}}}

    return [chart("Средняя разница угловых", hc[3], hc[6], LBL[1], "A", -3, 3),
            chart("Средний тотал угловых", hc[1], hc[5], LBL[0], "G")]


LBL = ("CH", "CI")      # chart point labels: total, diff
LBL_STEP = 7            # one label every N days, counted back from the last point


def label_cells():
    hc = [colstr(colnum(HC) + i) for i in range(7)]
    out = []
    for lc, yc, name in ((LBL[0], hc[1], "Метки тотала"), (LBL[1], hc[3], "Метки разницы")):
        Y = f"{yc}2:{yc}71"
        out += [{"range": f"{lc}1", "values": [[name]]},
                {"range": f"{lc}2", "values": [[
                    f'=ARRAYFORMULA(IF(ISNUMBER({Y})*(MOD(71-ROW({Y}),{LBL_STEP})=0),TEXT({Y},"0.00"),""))']]}]
    return out


def refresh_charts(sh, ws):
    meta = sh.fetch_sheet_metadata({"fields": "sheets(properties.sheetId,charts.chartId)"})
    ids = [c["chartId"] for s in meta["sheets"] if s["properties"]["sheetId"] == ws.id
           for c in s.get("charts", [])]
    reqs = [{"deleteEmbeddedObject": {"objectId": i}} for i in ids]
    retry(ws.batch_update, label_cells(), value_input_option="USER_ENTERED")
    retry(sh.batch_update, {"requests": reqs + chart_requests(ws.id)})
    print(f"CALC charts refreshed ({len(ids)} removed)")


def build_calc(sh, name):
    ws = sh.add_worksheet(name, rows=200, cols=90, index=0)
    ref = f"'{name}'"
    cells, fmts, reqs = [], [], []

    def put(a1, v):
        cells.append({"range": a1, "values": [[v]]})

    def fmt(rng, f):
        fmts.append({"range": rng, "format": f})

    def grid(c1, r1, c2, r2):
        return {"sheetId": ws.id, "startRowIndex": r1 - 1, "endRowIndex": r2,
                "startColumnIndex": colnum(c1) - 1, "endColumnIndex": colnum(c2)}

    def merge(c1, r1, c2, r2):
        reqs.append({"mergeCells": {"mergeType": "MERGE_ALL", "range": grid(c1, r1, c2, r2)}})

    def cs(start, n):
        return [colstr(colnum(start) + i) for i in range(n)]

    # ================= settings AH:AI (as Console 1 / Lists) =================
    for a1, v in (
            ("AH9", "Строка дней"), ("AI9", "Байес (a)"), ("AH10", 1), ("AI10", 0.40),
            ("AH11", "Линейность"), ("AI11", "Маржа 2"), ("AH12", 0.18), ("AI12", 0.075),
            ("AH13", 0.06), ("AI13", "Маржа 3"),
            ("AH14", f"=CEILING({BASE}+(1-SQRT({LIN1}))*100/{KDEC})"), ("AI14", 0.09),
            ("AH16", "Poisson_Res"), ("AI16", "Затухание"), ("AH17", 0.10), ("AI17", 1.5),
            ("AH18", "Tour_Res"), ("AI18", "База дней"), ("AH19", 0.0), ("AI19", 7),
            ("AH20", "MAX Влияние"), ("AI20", "Турнир"), ("AH21", 0.50), ("AI21", 15),
            ("AH22", "Влияние"), ("AI22", "Команды"), ("AH23", 0.10), ("AI23", 200),
            ("AH24", "MAX Влияние"), ("AI24", "И+К"), ("AH25", 0.60), ("AI25", 20),
            ("AH27", "Update Date"), ("AI27", "МО+"), ("AH28", "=DATE(2025,12,4)"), ("AI28", 1.05),
            ("AH29", "Critical Date"), ("AI29", "Old"), ("AH30", "=DATE(2025,11,10)"), ("AI30", 0.50)):
        put(a1, v)
    for rr in (9, 11, 16, 18, 20, 22, 24, 27, 29):
        fmt(f"AH{rr}:AI{rr}", {"backgroundColor": GREEN, "textFormat": {"bold": True},
                               "horizontalAlignment": "CENTER"})
    fmt("AI13", {"backgroundColor": GREEN, "textFormat": {"bold": True}, "horizontalAlignment": "CENTER"})
    fmt("AH10:AI30", {"horizontalAlignment": "CENTER"})
    for a in ("AH10", "AI10", "AH12", "AI12", "AH13", "AI14", "AH17", "AI17", "AH19", "AI19",
              "AH21", "AI21", "AH23", "AI23", "AH25", "AI25", "AH28", "AI28", "AH30", "AI30"):
        fmt(a, {"backgroundColor": LGOLD})
    fmt("AH14", {"backgroundColor": PINK})
    fmt("AI12", {"numberFormat": {"type": "PERCENT", "pattern": "0.0%"}})
    fmt("AI14", {"numberFormat": {"type": "PERCENT", "pattern": "0.0%"}})
    fmt("AH28", {"numberFormat": {"type": "DATE", "pattern": "dd.mm.yyyy"}})
    fmt("AH30", {"numberFormat": {"type": "DATE", "pattern": "dd.mm.yyyy"}})

    # ================= helper blocks =================
    hl, hr, hm, ht = cs(HL, 8), cs(HR, 8), cs(HM, 6), cs(HT, 9)
    for c, lab in zip(hl, ["Дата", "Команда P1", "Команда P2", "P1 H1", "P1 H2", "P2 H1", "P2 H2", "Турнир"]):
        put(f"{c}1", lab)
    for c, lab in zip(hr, ["Дата", "Команда P1", "Команда P2", "P1 H1", "P1 H2", "P2 H1", "P2 H2", "Турнир"]):
        put(f"{c}1", lab)
    for c, lab in zip(hm, ["M Дата", "Турнир", "Угл P1", "Угл P2", "Вес", "Вес МО+"]):
        put(f"{c}1", lab)
    for c, lab in zip(ht, ["T Дата", "P1", "Team1", "P2", "Team2", "C1", "C2", "Вес", "Вес МО+"]):
        put(f"{c}1", lab)
    put(f"{hl[0]}2", h2h_table(False))
    put(f"{hr[0]}2", h2h_table(True))
    put(f"{hm[0]}2", h2h_model())
    put(f"{ht[0]}2", tour_model())
    for blk in (hl, hr, hm, ht):
        fmt(f"{blk[0]}1:{blk[-1]}1", {"textFormat": {"bold": True}, "backgroundColor": GREY})
        fmt(f"{blk[0]}2:{blk[0]}", {"numberFormat": {"type": "DATE_TIME", "pattern": "dd.mm.yyyy hh:mm"}})
    fmt(f"{hm[4]}2:{hm[5]}", {"numberFormat": {"type": "NUMBER", "pattern": "0.0000"}})
    fmt(f"{ht[7]}2:{ht[8]}", {"numberFormat": {"type": "NUMBER", "pattern": "0.0000"}})

    # ================= model chain AH36:AI101 =================
    M_DT, M_TR, M_A, M_B, M_W, M_WM = (R(c) for c in hm)
    T_DT, T_P1, T_T1, T_P2, T_T2, T_C1, T_C2, T_W, T_WM = (R(c) for c in ht)
    tmask = f"((({TOUR}=\"*\")+({M_TR}={TOUR}))>0)"

    def tm(team, side):     # team mask in tournament block ("*" = any)
        col = T_T1 if side == 1 else T_T2
        return f"((({team}=\"*\")+({col}={team}))>0)"

    def pm(pl, side):
        col = T_P1 if side == 1 else T_P2
        return f"({col}=LOWER(TRIM({pl})))"

    chain = [
        (36, "MO_P1", f"=IFERROR(SUMPRODUCT({M_WM},{M_A})/SUM({M_W}),0)"),
        (37, "MO_P2", f"=IFERROR(SUMPRODUCT({M_WM},{M_B})/SUM({M_W}),0)"),
        (39, "G_tour", f"=SUMPRODUCT({M_W}*{tmask})"),
        (40, "a_tour", f"=IF(AI39>{NTOUR},{MAXT},AI39*{MAXT}/{NTOUR})"),
        (41, "a_tour_Res", f"=IF(AI39>{NTOUR},{TRES},AI39*{TRES}/{NTOUR})"),
        (42, "MO_P1_a", f"=IF(AI39>0,AI36*(1-AI40)+SUMPRODUCT({M_WM}*{tmask}*{M_A})/AI39*AI40,AI36)"),
        (43, "MO_P2_a", f"=IF(AI39>0,AI37*(1-AI40)+SUMPRODUCT({M_WM}*{tmask}*{M_B})/AI39*AI40,AI37)"),
        (46, "P_draw", f"=IFERROR(SUMPRODUCT({M_W}*({M_A}={M_B}))/SUM({M_W}),0)"),
        (47, "MO_tour", f"=IFERROR(SUMPRODUCT({T_WM},{T_C1}+{T_C2})/SUM({T_W}),0)"),
    ]
    for base, team in ((48, T1), (51, T2)):
        m1, m2 = tm(team, 1), tm(team, 2)
        n = "1" if base == 48 else "2"
        chain += [
            (base, f"G_T{n}", f"=SUMPRODUCT({T_W}*({m1}+{m2}))"),
            (base + 1, f"MO_S_T{n}", f"=IFERROR(SUMPRODUCT({T_WM}*({m1}*{T_C1}+{m2}*{T_C2}))/AI{base},0)"),
            (base + 2, f"MO_M_T{n}", f"=IFERROR(SUMPRODUCT({T_WM}*({m1}*{T_C2}+{m2}*{T_C1}))/AI{base},0)"),
        ]
    chain += [
        (55, "S_T1", f"=IFERROR(IF(AI48<{NTEAM},1+((AI49/(AI47/2)-1)*AI48/{NTEAM}),AI49/(AI47/2)),1)"),
        (56, "M_T1", f"=IFERROR(IF(AI48<{NTEAM},1+((AI50/(AI47/2)-1)*AI48/{NTEAM}),AI50/(AI47/2)),1)"),
        (57, "S_T2", f"=IFERROR(IF(AI51<{NTEAM},1+((AI52/(AI47/2)-1)*AI51/{NTEAM}),AI52/(AI47/2)),1)"),
        (58, "M_T2", f"=IFERROR(IF(AI51<{NTEAM},1+((AI53/(AI47/2)-1)*AI51/{NTEAM}),AI53/(AI47/2)),1)"),
    ]
    for base, pl, team, n in ((60, PL1, T1, "1"), (65, PL2, T2, "2")):
        p1, p2 = pm(pl, 1), pm(pl, 2)
        k1, k2 = f"{p1}*{tm(team, 1)}", f"{p2}*{tm(team, 2)}"
        chain += [
            (base, f"G_P{n}T{n}", f"=SUMPRODUCT({T_W}*({k1}+{k2}))"),
            (base + 1, f"MO_S_P{n}", f"=IFERROR(SUMPRODUCT({T_WM}*({p1}*{T_C1}+{p2}*{T_C2}))/SUMPRODUCT({T_W}*({p1}+{p2})),0)"),
            (base + 2, f"MO_S_P{n}T{n}", f"=IFERROR(SUMPRODUCT({T_WM}*({k1}*{T_C1}+{k2}*{T_C2}))/AI{base},0)"),
            (base + 3, f"MO_M_P{n}", f"=IFERROR(SUMPRODUCT({T_WM}*({p1}*{T_C2}+{p2}*{T_C1}))/SUMPRODUCT({T_W}*({p1}+{p2})),0)"),
            (base + 4, f"MO_M_P{n}T{n}", f"=IFERROR(SUMPRODUCT({T_WM}*({k1}*{T_C2}+{k2}*{T_C1}))/AI{base},0)"),
        ]
    chain += [
        (71, "S_P1T1", "=IFERROR(AI62/AI61,1)"),
        (72, "M_P1T1", "=IFERROR(AI64/AI63,1)"),
        (73, "S_P2T2", "=IFERROR(AI67/AI66,1)"),
        (74, "M_P2T2", "=IFERROR(AI69/AI68,1)"),
        (76, "a_P1T1", f"=IF(AI60>{NPT},{MAXPT},AI60*{MAXPT}/{NPT})"),
        (77, "a_P2T2", f"=IF(AI65>{NPT},{MAXPT},AI65*{MAXPT}/{NPT})"),
        (79, "S_T1+P1T1", "=IF(AI76>0,AI55*(1-AI76)+AI71*AI76,AI55)"),
        (80, "M_T1+P1T1", "=IF(AI76>0,AI56*(1-AI76)+AI72*AI76,AI56)"),
        (81, "S_T2+P2T2", "=IF(AI77>0,AI57*(1-AI77)+AI73*AI77,AI57)"),
        (82, "M_T2+P2T2", "=IF(AI77>0,AI58*(1-AI77)+AI74*AI77,AI58)"),
        (84, "MO_a_T1P1_a", f"=AI42*(1+(AI79-1)*{INFL})*(1+(AI82-1)*{INFL})"),
        (85, "MO_a_T2P2_a", f"=AI43*(1+(AI80-1)*{INFL})*(1+(AI81-1)*{INFL})"),
        (87, "W1", f"=IFERROR(SUMPRODUCT({M_W}*({M_A}>{M_B}))/SUM({M_W}),0)"),
        (88, "Draw", f"=IFERROR(SUMPRODUCT({M_W}*({M_A}={M_B}))/SUM({M_W}),0)"),
        (89, "W2", f"=IFERROR(SUMPRODUCT({M_W}*({M_A}<{M_B}))/SUM({M_W}),0)"),
        (91, "Tour_W1", f"=IFERROR(SUMPRODUCT({M_W}*{tmask}*({M_A}>{M_B}))/AI39,0)"),
        (92, "Tour_Draw", f"=IFERROR(SUMPRODUCT({M_W}*{tmask}*({M_A}={M_B}))/AI39,0)"),
        (93, "Tour_W2", f"=IFERROR(SUMPRODUCT({M_W}*{tmask}*({M_A}<{M_B}))/AI39,0)"),
        (95, "Pois_W1", "=" + pois("i>j", "$AI$84", "$AI$85")),
        (96, "Pois_Draw", "=" + pois("i=j", "$AI$84", "$AI$85")),
        (97, "Pois_W2", "=" + pois("i<j", "$AI$84", "$AI$85")),
        (99, "Final_W1", f"=(AI87*(1-$AI$41)+AI91*$AI$41)*(1-{PRES})+AI95*{PRES}"),
        (100, "Final_Draw", f"=(AI88*(1-$AI$41)+AI92*$AI$41)*(1-{PRES})+AI96*{PRES}"),
        (101, "Final_W2", f"=(AI89*(1-$AI$41)+AI93*$AI$41)*(1-{PRES})+AI97*{PRES}"),
    ]
    for r_, lab, f_ in chain:
        put(f"AH{r_}", lab)
        put(f"AI{r_}", f_)
    fmt("AH36:AH101", {"textFormat": {"bold": True}})
    fmt("AI36:AI101", {"numberFormat": {"type": "NUMBER", "pattern": "0.0000"}})
    fmt("AH84:AI85", {"backgroundColor": GOLD, "textFormat": {"bold": True}})

    # ================= tables A:E / G:K =================
    def table(c0, hb, all_tours):
        c = cs(c0, 5)
        J, P1H1, P1H2, P2H1, P2H2 = R(hb[0]), R(hb[3]), R(hb[4]), R(hb[5]), R(hb[6])
        ok = f"ISNUMBER({J})"
        D = f"(({P1H1}+{P1H2})-({P2H1}+{P2H2}))"
        T = f"({P1H1}+{P1H2}+{P2H1}+{P2H2})"
        n = f"COUNT({J})"

        def cnt(expr):
            return f"=IF({n}=0,\"\",SUMPRODUCT({ok}*({expr})))"

        def o2(a, e):
            return f'=IFERROR({odds(f"{a}/({a}+{e})", 2, MAR2)},"")'

        if all_tours:
            put(f"{c[0]}1", "*")
            put(f"{c[0]}2", "*"); put(f"{c[3]}2", "*")
            put(f"{c[0]}3", ""); put(f"{c[3]}3", "")
        else:
            put(f"{c[0]}1", f'="* "&{TOUR}&" ("&{PL1}&" - "&{PL2}&")"')
            put(f"{c[0]}2", "*"); put(f"{c[3]}2", "*")
            put(f"{c[0]}3", f"={PL1}"); put(f"{c[3]}3", f"={PL2}")
        put(f"{c[2]}3", "X")
        merge(c[0], 1, c[4], 1)
        for rr in (2, 3, 4, 5):
            merge(c[0], rr, c[1], rr); merge(c[3], rr, c[4], rr)

        # 1X2 counts + odds (Маржа 3)
        s3 = f"(${c[0]}$4+${c[2]}$4+${c[3]}$4)"
        for cc, cond in ((c[0], f"{D}>0"), (c[2], f"{D}=0"), (c[3], f"{D}<0")):
            put(f"{cc}4", cnt(cond))
            put(f"{cc}5", f'=IFERROR({odds(f"${cc}$4/{s3}", 3, MAR3)},"")')

        # handicap rows 6-11
        put(f"{c[0]}6", "Ф1"); put(f"{c[4]}6", "Ф2")
        put(f"{c[1]}6", f'=IF({n}=0,"Разница","Разница   "&TEXT(SUMPRODUCT({ok}*{D})/{n},"0.00"))')
        merge(c[1], 6, c[3], 6)
        for rr, expr in ((7, f"={c[2]}9-1.5"), (8, f"={c[2]}9-1"), (9, 0), (10, f"={c[2]}9+1"), (11, f"={c[2]}9+1.5")):
            L = f"${c[2]}${rr}"
            put(f"{c[2]}{rr}", expr)
            put(f"{c[0]}{rr}", cnt(f"{D}+{L}>0"))
            put(f"{c[4]}{rr}", cnt(f"{D}+{L}<0"))
            put(f"{c[1]}{rr}", o2(f"{c[0]}{rr}", f"{c[4]}{rr}"))
            put(f"{c[3]}{rr}", o2(f"{c[4]}{rr}", f"{c[0]}{rr}"))

        # total rows 12-17 (center line 15 editable)
        put(f"{c[0]}12", ">"); put(f"{c[4]}12", "<")
        put(f"{c[1]}12", f'=IF({n}=0,"Тотал","Тотал   "&TEXT(SUMPRODUCT({ok}*{T})/{n},"0.00"))')
        merge(c[1], 12, c[3], 12)
        put(f"{c[2]}15", f"=IF({n}=0,3.5,MAX(1.5,FLOOR(SUMPRODUCT({ok}*{T})/{n})+0.5))")
        for rr in (13, 14, 16, 17):
            put(f"{c[2]}{rr}", f"={c[2]}15{rr - 15:+d}")
        for rr in range(13, 18):
            L = f"${c[2]}${rr}"
            put(f"{c[0]}{rr}", cnt(f"{T}>{L}"))
            put(f"{c[4]}{rr}", cnt(f"{T}<{L}"))
            put(f"{c[1]}{rr}", o2(f"{c[0]}{rr}", f"{c[4]}{rr}"))
            put(f"{c[3]}{rr}", o2(f"{c[4]}{rr}", f"{c[0]}{rr}"))

        rng = lambda c1, r1, c2, r2: f"{c1}{r1}:{c2}{r2}"
        fmt(rng(c[0], 1, c[4], 17), {"horizontalAlignment": "CENTER"})
        fmt(rng(c[0], 1, c[4], 1), {"backgroundColor": GREY, "textFormat": {"bold": True}})
        fmt(rng(c[0], 2, c[4], 3), {"backgroundColor": BLUE, "textFormat": {"bold": True}})
        fmt(rng(c[0], 4, c[4], 4), {"backgroundColor": PINK, "numberFormat": {"type": "NUMBER", "pattern": "0"}})
        fmt(rng(c[0], 5, c[4], 5), {"backgroundColor": GREY, "textFormat": {"bold": True},
                                    "numberFormat": {"type": "NUMBER", "pattern": "0.00"}})
        for rr in (6, 12):
            fmt(rng(c[0], rr, c[4], rr), {"backgroundColor": BLUE, "textFormat": {"bold": True}})
        for a, b in ((7, 11), (13, 17)):
            for cc, bg, pat in ((c[0], PINK, "0"), (c[4], PINK, "0"), (c[1], GREY, "0.00"), (c[3], GREY, "0.00")):
                fmt(rng(cc, a, cc, b), {"backgroundColor": bg, "numberFormat": {"type": "NUMBER", "pattern": pat}})
            fmt(rng(c[2], a, c[2], b), {"backgroundColor": BLUE})
        fmt(rng(c[2], 7, c[2], 11), {"numberFormat": {"type": "NUMBER", "pattern": "+0.0;-0.0;0"}})
        fmt(rng(c[2], 13, c[2], 17), {"numberFormat": {"type": "NUMBER", "pattern": "0.0"}})
        for rr in (9, 15):
            fmt(rng(c[0], rr, c[4], rr), {"textFormat": {"bold": True}})
            fmt(rng(c[2], rr, c[2], rr), {"backgroundColor": MIDC})

    table("A", hl, True)
    table("G", hr, False)

    # ================= M:N =================
    J, P1H1, P1H2, P2H1, P2H2 = R(hl[0]), R(hl[3]), R(hl[4]), R(hl[5]), R(hl[6])
    nL = f"COUNT({J})"
    for a1, v in (("M1", "Выборка дней"), ("N1", "От"), ("M2", 50), ("N2", f"={DTO}-{WIN}"),
                  ("M3", "Для графика"), ("N3", "До"), ("M4", 30), ("N4", "=TODAY()"),
                  ("N6", "Разница"), ("N7", f'=IFERROR((SUM({P1H1})+SUM({P1H2})-SUM({P2H1})-SUM({P2H2}))/{nL},"")'),
                  ("M7", "Матчи"), ("M8", f"={nL}"),
                  ("N8", "Тотал"), ("N9", f'=IFERROR((SUM({P1H1})+SUM({P1H2})+SUM({P2H1})+SUM({P2H2}))/{nL},"")'),
                  ("M10", "Часть H1"), ("N10", "Часть H2"),
                  ("M11", f'=IFERROR((SUM({P1H1})+SUM({P2H1}))/(SUM({P1H1})+SUM({P1H2})+SUM({P2H1})+SUM({P2H2})),"")'),
                  ("N11", '=IFERROR(1-M11,"")'),
                  ("M13", "МО (!!!)"), ("N13", "Разница (!!!)"),
                  ("M14", "=$AI$84+$AI$85"), ("N14", "=$AI$84-$AI$85")):
        put(a1, v)
    fmt("M1:N14", {"horizontalAlignment": "CENTER"})
    for rr in ("M1:N1", "M3:N3", "N6", "M7", "N8", "M10:N10", "M13:N13"):
        fmt(rr, {"backgroundColor": BLUE, "textFormat": {"bold": True}})
    fmt("M2", {"backgroundColor": GREY}); fmt("M4", {"backgroundColor": GREY})
    for a in ("N2", "N4"):
        fmt(a, {"backgroundColor": PINK, "numberFormat": {"type": "DATE", "pattern": "dd.mm.yyyy"}})
    fmt("N7", {"backgroundColor": PINK, "numberFormat": {"type": "NUMBER", "pattern": "0.00"}})
    fmt("N9", {"backgroundColor": PINK, "numberFormat": {"type": "NUMBER", "pattern": "0.00"}})
    fmt("M8", {"backgroundColor": PINK})
    fmt("M11:N11", {"backgroundColor": PINK, "numberFormat": {"type": "PERCENT", "pattern": "0.0%"}})
    fmt("M14:N14", {"backgroundColor": GOLD, "textFormat": {"bold": True},
                    "numberFormat": {"type": "NUMBER", "pattern": "0.00"}})

    # last H2H matches list N17:T
    put("N17", "Event Date/Time"); put("P17", "Pair")
    put("Q17", "УГ_1"); put("R17", "УГ_2"); put("S17", "H1_1"); put("T17", "H1_2")
    put("N18", f'=IFERROR(ARRAY_CONSTRAIN(FILTER({J},ISNUMBER({J})),50,1),"")')
    put("P18", f'=IFERROR(ARRAY_CONSTRAIN(FILTER(IF(ISNUMBER({J}),{PL1}&" - "&{PL2},""),ISNUMBER({J})),50,1),"")')
    put("Q18", f'=IFERROR(ARRAY_CONSTRAIN(FILTER({P1H1}+{P1H2},ISNUMBER({J})),50,1),"")')
    put("R18", f'=IFERROR(ARRAY_CONSTRAIN(FILTER({P2H1}+{P2H2},ISNUMBER({J})),50,1),"")')
    put("S18", f'=IFERROR(ARRAY_CONSTRAIN(FILTER({P1H1},ISNUMBER({J})),50,1),"")')
    put("T18", f'=IFERROR(ARRAY_CONSTRAIN(FILTER({P2H1},ISNUMBER({J})),50,1),"")')
    fmt("N17:T17", {"backgroundColor": BLUE, "textFormat": {"bold": True}, "horizontalAlignment": "CENTER"})
    fmt("N18:N67", {"numberFormat": {"type": "DATE_TIME", "pattern": "dd.mm.yyyy hh:mm"}})
    fmt("Q18:T67", {"horizontalAlignment": "CENTER"})

    # ================= P:AB summary =================
    put("P1", f"={TOUR}")
    for c, lab in zip(cs("Q", 12), ["И", "В", "Н", "П", "УЗ", "УП", "РЗП", "Т", "ТЗ", "ТП", "Рейт", "РЗП/И"]):
        put(f"{c}1", lab)
    win = f'{D_DT},">="&{DFROM},{D_DT},"<"&{DTO}+1,{D_TR},{TOUR}'

    def prow(r_, pl, team, label):
        put(f"P{r_}", label)
        a = f"{D_P1},{pl},{D_T1},{team},{win}"
        b = f"{D_P2},{pl},{D_T2},{team},{win}"
        put(f"Q{r_}", f"=COUNTIFS({a})+COUNTIFS({b})")
        put(f"R{r_}", f'=COUNTIFS({D_DIFF},">0",{a})+COUNTIFS({D_DIFF},"<0",{b})')
        put(f"S{r_}", f'=COUNTIFS({D_DIFF},"=0",{a})+COUNTIFS({D_DIFF},"=0",{b})')
        put(f"T{r_}", f'=COUNTIFS({D_DIFF},"<0",{a})+COUNTIFS({D_DIFF},">0",{b})')
        put(f"U{r_}", f"=SUMIFS({D_C1},{a})+SUMIFS({D_C2},{b})")
        put(f"V{r_}", f"=SUMIFS({D_C2},{a})+SUMIFS({D_C1},{b})")
        put(f"W{r_}", f"=U{r_}-V{r_}")
        put(f"X{r_}", f'=IFERROR((U{r_}+V{r_})/Q{r_},"")')
        put(f"Y{r_}", f'=IFERROR(U{r_}/Q{r_},"")')
        put(f"Z{r_}", f'=IFERROR(V{r_}/Q{r_},"")')
        put(f"AA{r_}", f'=IFERROR((R{r_}*3+S{r_})/Q{r_},"")')
        put(f"AB{r_}", f'=IFERROR(Y{r_}-Z{r_},"")')

    prow(2, PL1, '"*"', f'={PL1}&" ( * )"')
    prow(3, PL1, T1, f'={PL1}&" ("&{T1}&")"')
    prow(5, PL2, '"*"', f'={PL2}&" ( * )"')
    prow(6, PL2, T2, f'={PL2}&" ("&{T2}&")"')
    fmt("P1:AB1", {"backgroundColor": BLUE, "textFormat": {"bold": True}, "horizontalAlignment": "CENTER"})
    fmt("P2:AB2", {"backgroundColor": BLUE}); fmt("P5:AB5", {"backgroundColor": BLUE})
    fmt("P3:AB3", {"backgroundColor": GREY}); fmt("P6:AB6", {"backgroundColor": GREY})
    fmt("Q2:AB6", {"horizontalAlignment": "CENTER"})
    fmt("X2:AB6", {"numberFormat": {"type": "NUMBER", "pattern": "0.00"}})

    # halves: pair (all tournaments, window) vs all matches in window
    put("P8", f'={PL1}&" - "&{PL2}'); put("Q8", "УГ"); put("R8", "%УГ"); put("S8", "*")
    put("T8", "УГ"); put("U8", "%УГ")
    put("P9", "1-й тайм"); put("P10", "2-й тайм")
    put("Q9", f"=SUM({P1H1})+SUM({P2H1})"); put("Q10", f"=SUM({P1H2})+SUM({P2H2})")
    put("R9", '=IFERROR(Q9/(Q9+Q10),"")'); put("R10", '=IFERROR(Q10/(Q9+Q10),"")')
    allwin = f'{D_DT},">="&{DFROM},{D_DT},"<"&{DTO}+1'
    put("T9", f"=SUMIFS({D_H1},{allwin})"); put("T10", f"=SUMIFS({D_H2},{allwin})")
    put("U9", '=IFERROR(T9/(T9+T10),"")'); put("U10", '=IFERROR(T10/(T9+T10),"")')
    merge("S", 9, "S", 10)
    fmt("P8:U8", {"backgroundColor": GREY, "textFormat": {"bold": True}, "horizontalAlignment": "CENTER"})
    fmt("P9:P10", {"backgroundColor": GREY, "horizontalAlignment": "CENTER"})
    fmt("Q9:R10", {"backgroundColor": BLUE, "horizontalAlignment": "CENTER"})
    fmt("T9:U10", {"backgroundColor": BLUE, "horizontalAlignment": "CENTER"})
    fmt("S9", {"textFormat": {"bold": True, "fontSize": 14}, "horizontalAlignment": "CENTER",
               "verticalAlignment": "MIDDLE"})
    fmt("R9:R10", {"numberFormat": {"type": "NUMBER", "pattern": "0.0000"}})
    fmt("U9:U10", {"numberFormat": {"type": "NUMBER", "pattern": "0.0000"}})

    # Результат матча
    put("P12", "Результат матча"); put("Q12", "W1"); put("R12", "Draw"); put("S12", "W2")
    put("P13", "Poisson Result"); put("P14", "Final Result")
    for c, pr, fr in (("Q", "$AI$95", "$AI$99"), ("R", "$AI$96", "$AI$100"), ("S", "$AI$97", "$AI$101")):
        put(f"{c}13", f'=IFERROR({odds(pr, 3, MAR3)},"")')
        put(f"{c}14", f'=IFERROR({odds(fr, 3, MAR3)},"")')
    fmt("P12:S12", {"backgroundColor": BLUE, "textFormat": {"bold": True}, "horizontalAlignment": "CENTER"})
    fmt("P13:P14", {"backgroundColor": BLUE, "horizontalAlignment": "CENTER"})
    fmt("P14", {"textFormat": {"bold": True}})
    fmt("Q13:S13", {"backgroundColor": LGOLD, "horizontalAlignment": "CENTER",
                    "numberFormat": {"type": "NUMBER", "pattern": "0.00"}})
    fmt("Q14:S14", {"backgroundColor": GOLD, "textFormat": {"bold": True}, "horizontalAlignment": "CENTER",
                    "numberFormat": {"type": "NUMBER", "pattern": "0.00"}})

    # ================= chart data (70 days, rolling "Для графика") =================
    hc = cs(HC, 7)
    put(f"{hc[0]}1", "Дата")
    put(f"{hc[1]}1", f'="Средний тотал угловых за "&{GWIN}&" дней"')
    put(f"{hc[2]}1", f'="Средний тотал угловых за "&{GWIN}&" дней в турнире"')
    put(f"{hc[3]}1", f'="Средняя разница угловых за "&{GWIN}&" дней"')
    put(f"{hc[4]}1", f'="Средняя разница угловых за "&{GWIN}&" дней в турнире"')
    put(f"{hc[5]}1", "Тренд тотала")
    put(f"{hc[6]}1", "Тренд разницы")
    for i in range(70):
        rr = 2 + i
        dcell = f"${hc[0]}{rr}"
        put(f"{hc[0]}{rr}", "=TODAY()" if i == 69 else f"={hc[0]}{rr + 1}-1")
        w = f'{D_KEY},{KEY},{D_DT},"<"&{dcell}+1,{D_DT},">="&{dcell}+1-{GWIN}'
        wt = w + f",{D_TR},{TOUR}"
        for col, ww_ in ((hc[1], w), (hc[2], wt)):
            put(f"{col}{rr}", f'=IFERROR(SUMIFS({D_TOT},{ww_})/COUNTIFS({ww_}),NA())')
        for col, ww_ in ((hc[3], w), (hc[4], wt)):
            put(f"{col}{rr}", f'=IFERROR((SUMIFS({D_DIFF},{ww_},{D_P1},{PL1})-SUMIFS({D_DIFF},{ww_},{D_P2},{PL1}))'
                              f'/COUNTIFS({ww_}),NA())')
    X = f"{hc[0]}2:{hc[0]}71"
    for tcol, ycol in ((hc[5], hc[1]), (hc[6], hc[3])):
        Y = f"{ycol}2:{ycol}71"
        put(f"{tcol}2", f'=IFERROR(ARRAYFORMULA(IF(ISNUMBER({Y}),TREND(FILTER({Y},ISNUMBER({Y})),'
                        f'FILTER({X},ISNUMBER({Y})),{X}),NA())),"")')
    fmt(f"{hc[0]}1:{hc[6]}1", {"textFormat": {"bold": True}, "backgroundColor": GREY})
    fmt(f"{hc[0]}2:{hc[0]}71", {"numberFormat": {"type": "DATE", "pattern": "dd.mm.yyyy"}})
    fmt(f"{hc[1]}2:{hc[6]}71", {"numberFormat": {"type": "NUMBER", "pattern": "0.00"}})

    # ================= dropdown lists =================
    ls = cs(HLS, 5)
    PA, PB, PC, PD = (f"PAIRS!{x}2:{x}" for x in "ABCD")
    put(f"{ls[0]}1", "Турниры")
    put(f"{ls[0]}2", f'=IFERROR(VSTACK("*",SORT(UNIQUE(FILTER({PC},{PA}={PL1},'
                     f'(({PL2}="")+({PB}={PL2}))>0)))),"*")')
    put(f"{ls[1]}1", "Игроки")
    put(f"{ls[1]}2", f'=IFERROR(SORT(UNIQUE(FILTER({PA},{PA}<>""))),"")')
    put(f"{ls[2]}1", "Соперники")
    put(f"{ls[2]}2", f'=IFERROR(SORT(UNIQUE(FILTER({PB},{PA}={PL1}))),"")')
    for col, pl in ((ls[3], PL1), (ls[4], PL2)):
        put(f"{col}1", "Команды " + ("P1" if pl == PL1 else "P2"))
        put(f"{col}2", f'=IFERROR(VSTACK("*",SORT(UNIQUE(FILTER({PD},{PA}={pl},'
                       f'(({TOUR}="*")+({PC}={TOUR}))>0)))),"*")')
    fmt(f"{ls[0]}1:{ls[4]}1", {"textFormat": {"bold": True}, "backgroundColor": GREY})

    # formats first (empty sheet = cheap), values last (heavy recalculation)
    for i in range(0, len(fmts), 25):
        retry(ws.batch_format, fmts[i:i + 25])
    print("CALC formats ok")

    def dv(a1, src):
        cc = "".join(ch for ch in a1 if ch.isalpha())
        rr = int("".join(ch for ch in a1 if ch.isdigit()))
        return {"setDataValidation": {
            "range": grid(cc, rr, cc, rr),
            "rule": {"condition": {"type": "ONE_OF_RANGE", "values": [{"userEnteredValue": src}]},
                     "showCustomUi": True, "strict": False}}}

    widths = [{"updateDimensionProperties": {
        "range": {"sheetId": ws.id, "dimension": "COLUMNS",
                  "startIndex": colnum(a) - 1, "endIndex": colnum(b)},
        "properties": {"pixelSize": px}, "fields": "pixelSize"}}
        for a, b, px in (("A", "K", 85), ("F", "F", 15), ("L", "L", 15), ("O", "O", 15),
                         ("M", "N", 110), ("P", "P", 150), ("Q", "AB", 55),
                         ("AH", "AI", 120))]

    retry(ws.batch_update, label_cells(), value_input_option="USER_ENTERED")
    retry(sh.batch_update, {"requests": reqs + widths + [
        dv("A1", f"={ref}!${ls[0]}$2:${ls[0]}"),
        dv("A3", f"={ref}!${ls[1]}$2:${ls[1]}"),
        dv("D3", f"={ref}!${ls[2]}$2:${ls[2]}"),
        dv("A2", f"={ref}!${ls[3]}$2:${ls[3]}"),
        dv("D2", f"={ref}!${ls[4]}$2:${ls[4]}"),
    ] + chart_requests(ws.id)})
    print("CALC layout ok")

    for i in range(0, len(cells), 80):
        retry(ws.batch_update, cells[i:i + 80], value_input_option="USER_ENTERED")
        time.sleep(1)
    retry(ws.update, [[MARKER]], MARKER_CELL)
    print("CALC built")




# =====================================================================================
# Console file (2x4 ECF): corners block inside "Console 1", directly under the goals tables
#   daily  : DB_Corners (only tournaments + players present in DATA, Kyiv time) -> console file
#   manual : BUILD_CONSOLE=1 -> helper tab "Corners_Model" + corners block in Console 1:
#            tables A18:K34, summary U18:AD31, settings AH104:AI125,
#            goals charts moved down 18 rows, team-vs-team block A86:K102 moved to A104:K120.
#   Nothing is written if the target cells are not empty or the marker AH103 is already set.
# =====================================================================================
X_ID_RAW = os.environ.get("GSHEET_ID_CONSOLE", "").strip()
X_TZ = "Europe/Kyiv"
C1, HS = "Console 1", "Corners_Model"
R0 = 17                                # table row r -> sheet row R0 + r  (row 1 of table = 18)
SHIFT = 18                             # goals charts move down by this many rows
TEAM_SRC, TEAM_DST = ("A", 86, "K", 102), ("A", 104)
X_MARK_CELL, X_MARK = "AH103", "УГЛОВЫЕ v1"
SET_OFF = 95                           # settings AH9:AI30 -> AH104:AI125


def q(sheet, a):
    return f"'{sheet}'!{a}"


def cabs(a):                            # "AL115" -> "$AL$115"
    m = re.fullmatch(r"([A-Z]+)(\d+)", a)
    return f"${m.group(1)}${m.group(2)}"


# settings panel in Console 1: same layout as goals AH9:AI30, shifted down by SET_OFF rows
def sset(a):
    m = re.fullmatch(r"(AH|AI)(\d+)", a)
    return q(C1, cabs(f"{m.group(1)}{int(m.group(2)) + SET_OFF}"))


XS = {k: sset(v) for k, v in dict(
    BAY="AI10", LIN1="AH12", MAR2="AI12", LIN2="AH13", DSTAR="AH14", MAR3="AI14", PRES="AH17", KDEC="AI17",
    TRES="AH19", BASE="AI19", MAXT="AH21", NTOUR="AI21", INFL="AH23", NTEAM="AI23", MAXPT="AH25", NPT="AI25",
    UPD="AH28", MOPLUS="AI28", CRIT="AH30", OLD="AI30").items()}

# inputs mirrored on helper tab (AT1:AU8); TOUR maps "Champions League A" -> "Champions League A 2x6"
XIN = dict(TOUR="$AU$1", T1="$AU$2", T2="$AU$3", P1="$AU$4", P2="$AU$5", DFROM="$AU$6", DTO="$AU$7", GWIN="$AU$8")
XH = dict(L="A", R="J", M="S", T="Z", C="AJ", LBL=("AQ", "AR"), CH="AU")


def x_weights(dd):
    s, i = XS, XIN
    return (f'dn,INT({i["DTO"]})-INT({dd}),'
            f'am,IF(dn<{s["BASE"]},0,IF((dn-{s["BASE"]})*{s["KDEC"]}/100>1,1,(dn-{s["BASE"]})*{s["KDEC"]}/100)),'
            f'qq,(1-am)*(1-am),'
            f'prc,IF(qq<{s["LIN1"]},({MODEL_DAYS - 1}-dn)/({MODEL_DAYS - 1}-{s["DSTAR"]})*({s["LIN1"]}-{s["LIN2"]})+{s["LIN2"]},qq),'
            f'ww,IF(INT({dd})<{s["CRIT"]},prc*{s["OLD"]},prc),'
            f'wm,IF(INT({dd})<{s["UPD"]},ww*{s["MOPLUS"]},ww),')


def x_h2h_table(use_tour):
    i = XIN
    tour = f'(({i["TOUR"]}="*")+(INDEX(db,,3)={i["TOUR"]}))>0,' if use_tour else ''
    return ('=IFERROR(ARRAYFORMULA(LET('
            f'db,{DB}!A2:Q,pa,LOWER(TRIM({i["P1"]})),pb,LOWER(TRIM({i["P2"]})),kk,INDEX(db,,17),'
            'f,FILTER(db,((kk=pa&"|"&pb)+(kk=pb&"|"&pa))>0,' + tour +
            f'INDEX(db,,2)>={i["DFROM"]},INDEX(db,,2)<{i["DTO"]}+1),'
            'a,LOWER(INDEX(f,,5))=pa,'
            'SORT(HSTACK(INDEX(f,,2),IF(a,INDEX(f,,4),INDEX(f,,6)),IF(a,INDEX(f,,6),INDEX(f,,4)),'
            'IF(a,INDEX(f,,8),INDEX(f,,10)),IF(a,INDEX(f,,9),INDEX(f,,11)),'
            'IF(a,INDEX(f,,10),INDEX(f,,8)),IF(a,INDEX(f,,11),INDEX(f,,9)),INDEX(f,,3)),1,FALSE))),"нет H2H")')


def x_h2h_model():
    i = XIN
    return ('=IFERROR(ARRAYFORMULA(LET('
            f'db,{DB}!A2:Q,pa,LOWER(TRIM({i["P1"]})),pb,LOWER(TRIM({i["P2"]})),kk,INDEX(db,,17),dt,INDEX(db,,2),'
            'f,FILTER(db,((kk=pa&"|"&pb)+(kk=pb&"|"&pa))>0,'
            f'INT(dt)>INT({i["DTO"]})-{MODEL_DAYS},INT(dt)<=INT({i["DTO"]})),'
            'a,LOWER(INDEX(f,,5))=pa,dd,INDEX(f,,2),'
            'xa,INDEX(f,,8)+INDEX(f,,9),xb,INDEX(f,,10)+INDEX(f,,11),'
            + x_weights("dd") +
            'HSTACK(dd,INDEX(f,,3),IF(a,xa,xb),IF(a,xb,xa),ww,wm))),"")')


def x_tour_model():
    i = XIN
    return ('=IFERROR(ARRAYFORMULA(LET('
            f'db,{DB}!A2:Q,dt,INDEX(db,,2),tr,INDEX(db,,3),'
            f'f,FILTER(db,(({i["TOUR"]}="*")+(tr={i["TOUR"]}))>0,'
            f'INT(dt)>INT({i["DTO"]})-{MODEL_DAYS},INT(dt)<=INT({i["DTO"]})),'
            'dd,INDEX(f,,2),'
            + x_weights("dd") +
            'HSTACK(dd,INDEX(f,,5),INDEX(f,,4),INDEX(f,,7),INDEX(f,,6),'
            'INDEX(f,,8)+INDEX(f,,9),INDEX(f,,10)+INDEX(f,,11),ww,wm))),"")')


def x_odds(p, n, mcell):
    return f"(1-{mcell})/((1-{XS['BAY']}*{mcell})*({p})+{XS['BAY']}*{mcell}/{n})"


def build_console(book):
    c1 = book.worksheet(C1)
    if (c1.acell(X_MARK_CELL).value or "").strip():
        print(f"{C1}: corners block already exists ({X_MARK_CELL}) -> not touched")
        return
    sc, sr1, ec, sr2 = TEAM_SRC
    dc, dr1 = TEAM_DST
    dr2 = dr1 + (sr2 - sr1)
    need_empty = [f"A{R0 + 1}:K{R0 + 17}", f"U{R0 + 1}:AD{R0 + 14}", f"AH{X_MARK_CELL[2:]}:AI{30 + SET_OFF}",
                  f"{dc}{dr1}:{ec}{dr2}"]
    for rng in need_empty:
        busy = [v for row in c1.get(rng) for v in row if str(v).strip()]
        if busy:
            print(f"{C1}: {rng} is not empty -> abort, nothing written")
            return
    has_team = any(str(v).strip() for row in c1.get(f"{sc}{sr1}:{ec}{sr2}") for v in row)
    for w in book.worksheets():
        if w.title == HS:
            retry(book.del_worksheet, w)          # our own helper tab from a failed build
    hs = book.add_worksheet(HS, rows=200, cols=60)

    hcells, ccells, hf, cf, reqs = [], [], [], [], []

    def hput(a, v): hcells.append({"range": a, "values": [[v]]})
    def cput(a, v): ccells.append({"range": a, "values": [[v]]})
    def cfmt(rng, f): cf.append({"range": rng, "format": f})

    def cs(start, n): return [colstr(colnum(start) + k) for k in range(n)]

    def grid(sid, c1_, r1, c2_, r2):
        return {"sheetId": sid, "startRowIndex": r1 - 1, "endRowIndex": r2,
                "startColumnIndex": colnum(c1_) - 1, "endColumnIndex": colnum(c2_)}

    def merge(c1_, r1, c2_, r2):
        reqs.append({"mergeCells": {"mergeType": "MERGE_ALL", "range": grid(c1.id, c1_, r1, c2_, r2)}})

    s = XS
    CH = XH["CH"]
    HR_ = lambda c: f"${c}$2:${c}"                 # helper range, local
    HQ = lambda c: q(HS, f"${c}$2:${c}")           # helper range, from Console 1
    CHQ = lambda n: q(HS, f"${CH}${n}")            # chain value, from Console 1

    # ---------------- helper tab: inputs mirror ----------------
    for r_, lab, f_ in (
            (1, "Турнир (угловые)", f'=IF({q(C1, "$A$1")}="*","*",IF(COUNTIF({DB}!$C$2:$C,{q(C1, "$A$1")}&" 2x6")>0,'
                                    f'{q(C1, "$A$1")}&" 2x6",{q(C1, "$A$1")}))'),
            (2, "Команда 1", f'={q(C1, "$A$2")}'), (3, "Команда 2", f'={q(C1, "$D$2")}'),
            (4, "Игрок 1", f'={q(C1, "$A$3")}'), (5, "Игрок 2", f'={q(C1, "$D$3")}'),
            (6, "От", f'={q(C1, "$N$2")}'), (7, "До", f'={q(C1, "$N$4")}'), (8, "Для графика", f'={q(C1, "$M$4")}')):
        hput(f"AT{r_}", lab); hput(f"{CH}{r_}", f_)

    hl, hr, hm, ht = cs(XH["L"], 8), cs(XH["R"], 8), cs(XH["M"], 6), cs(XH["T"], 9)
    for blk, labs in ((hl, ["Дата", "Команда P1", "Команда P2", "P1 H1", "P1 H2", "P2 H1", "P2 H2", "Турнир"]),
                      (hr, ["Дата", "Команда P1", "Команда P2", "P1 H1", "P1 H2", "P2 H1", "P2 H2", "Турнир"]),
                      (hm, ["M Дата", "Турнир", "Угл P1", "Угл P2", "Вес", "Вес МО+"]),
                      (ht, ["T Дата", "P1", "Team1", "P2", "Team2", "C1", "C2", "Вес", "Вес МО+"])):
        for c, lab in zip(blk, labs):
            hput(f"{c}1", lab)
    hput(f"{hl[0]}2", x_h2h_table(False))
    hput(f"{hr[0]}2", x_h2h_table(True))
    hput(f"{hm[0]}2", x_h2h_model())
    hput(f"{ht[0]}2", x_tour_model())
    for blk in (hl, hr, hm, ht):
        hf.append({"range": f"{blk[0]}2:{blk[0]}", "format": {"numberFormat": {"type": "DATE_TIME", "pattern": "dd.mm.yyyy hh:mm"}}})
        hf.append({"range": f"{blk[0]}1:{blk[-1]}1", "format": {"textFormat": {"bold": True}}})

    # ---------------- helper tab: model chain (same names/rows as Console 1 AH36:AI101) ----------------
    M_W, M_WM, M_A, M_B, M_TR = HR_(hm[4]), HR_(hm[5]), HR_(hm[2]), HR_(hm[3]), HR_(hm[1])
    T_P1, T_T1, T_P2, T_T2, T_C1, T_C2, T_W, T_WM = (HR_(c) for c in ht[1:])
    TOUR, T1, T2, P1, P2 = XIN["TOUR"], XIN["T1"], XIN["T2"], XIN["P1"], XIN["P2"]
    tmask = f"(((({TOUR}=\"*\")+({M_TR}={TOUR}))>0))"
    tm = lambda team, side: f"((({team}=\"*\")+({T_T1 if side == 1 else T_T2}={team}))>0)"
    pm = lambda pl, side: f"({T_P1 if side == 1 else T_P2}=LOWER(TRIM({pl})))"
    A = lambda n: f"{CH}{n}"

    chain = [
        (36, "MO_P1", f"=IFERROR(SUMPRODUCT({M_WM},{M_A})/SUM({M_W}),0)"),
        (37, "MO_P2", f"=IFERROR(SUMPRODUCT({M_WM},{M_B})/SUM({M_W}),0)"),
        (39, "G_tour", f"=SUMPRODUCT({M_W}*{tmask})"),
        (40, "a_tour", f"=IF({A(39)}>{s['NTOUR']},{s['MAXT']},{A(39)}*{s['MAXT']}/{s['NTOUR']})"),
        (41, "a_tour_Res", f"=IF({A(39)}>{s['NTOUR']},{s['TRES']},{A(39)}*{s['TRES']}/{s['NTOUR']})"),
        (42, "MO_P1_a", f"=IF({A(39)}>0,{A(36)}*(1-{A(40)})+SUMPRODUCT({M_WM}*{tmask}*{M_A})/{A(39)}*{A(40)},{A(36)})"),
        (43, "MO_P2_a", f"=IF({A(39)}>0,{A(37)}*(1-{A(40)})+SUMPRODUCT({M_WM}*{tmask}*{M_B})/{A(39)}*{A(40)},{A(37)})"),
        (46, "P_draw", f"=IFERROR(SUMPRODUCT({M_W}*({M_A}={M_B}))/SUM({M_W}),0)"),
        (47, "MO_tour", f"=IFERROR(SUMPRODUCT({T_WM},{T_C1}+{T_C2})/SUM({T_W}),0)"),
    ]
    for base, team, n in ((48, T1, "1"), (51, T2, "2")):
        m1, m2 = tm(team, 1), tm(team, 2)
        chain += [(base, f"G_T{n}", f"=SUMPRODUCT({T_W}*({m1}+{m2}))"),
                  (base + 1, f"MO_S_T{n}", f"=IFERROR(SUMPRODUCT({T_WM}*({m1}*{T_C1}+{m2}*{T_C2}))/{A(base)},0)"),
                  (base + 2, f"MO_M_T{n}", f"=IFERROR(SUMPRODUCT({T_WM}*({m1}*{T_C2}+{m2}*{T_C1}))/{A(base)},0)")]
    for r_, g, x in ((55, 48, 49), (56, 48, 50), (57, 51, 52), (58, 51, 53)):
        lab = {55: "S_T1", 56: "M_T1", 57: "S_T2", 58: "M_T2"}[r_]
        chain.append((r_, lab, f"=IFERROR(IF({A(g)}<{s['NTEAM']},1+(({A(x)}/({A(47)}/2)-1)*{A(g)}/{s['NTEAM']}),"
                               f"{A(x)}/({A(47)}/2)),1)"))
    for base, pl, team, n in ((60, P1, T1, "1"), (65, P2, T2, "2")):
        p1, p2 = pm(pl, 1), pm(pl, 2)
        k1, k2 = f"{p1}*{tm(team, 1)}", f"{p2}*{tm(team, 2)}"
        chain += [(base, f"G_P{n}T{n}", f"=SUMPRODUCT({T_W}*({k1}+{k2}))"),
                  (base + 1, f"MO_S_P{n}", f"=IFERROR(SUMPRODUCT({T_WM}*({p1}*{T_C1}+{p2}*{T_C2}))/SUMPRODUCT({T_W}*({p1}+{p2})),0)"),
                  (base + 2, f"MO_S_P{n}T{n}", f"=IFERROR(SUMPRODUCT({T_WM}*({k1}*{T_C1}+{k2}*{T_C2}))/{A(base)},0)"),
                  (base + 3, f"MO_M_P{n}", f"=IFERROR(SUMPRODUCT({T_WM}*({p1}*{T_C2}+{p2}*{T_C1}))/SUMPRODUCT({T_W}*({p1}+{p2})),0)"),
                  (base + 4, f"MO_M_P{n}T{n}", f"=IFERROR(SUMPRODUCT({T_WM}*({k1}*{T_C2}+{k2}*{T_C1}))/{A(base)},0)")]
    chain += [
        (71, "S_P1T1", f"=IFERROR({A(62)}/{A(61)},1)"), (72, "M_P1T1", f"=IFERROR({A(64)}/{A(63)},1)"),
        (73, "S_P2T2", f"=IFERROR({A(67)}/{A(66)},1)"), (74, "M_P2T2", f"=IFERROR({A(69)}/{A(68)},1)"),
        (76, "a_P1T1", f"=IF({A(60)}>{s['NPT']},{s['MAXPT']},{A(60)}*{s['MAXPT']}/{s['NPT']})"),
        (77, "a_P2T2", f"=IF({A(65)}>{s['NPT']},{s['MAXPT']},{A(65)}*{s['MAXPT']}/{s['NPT']})"),
        (79, "S_T1+P1T1", f"=IF({A(76)}>0,{A(55)}*(1-{A(76)})+{A(71)}*{A(76)},{A(55)})"),
        (80, "M_T1+P1T1", f"=IF({A(76)}>0,{A(56)}*(1-{A(76)})+{A(72)}*{A(76)},{A(56)})"),
        (81, "S_T2+P2T2", f"=IF({A(77)}>0,{A(57)}*(1-{A(77)})+{A(73)}*{A(77)},{A(57)})"),
        (82, "M_T2+P2T2", f"=IF({A(77)}>0,{A(58)}*(1-{A(77)})+{A(74)}*{A(77)},{A(58)})"),
        (84, "MO_a_T1P1_a", f"={A(42)}*(1+({A(79)}-1)*{s['INFL']})*(1+({A(82)}-1)*{s['INFL']})"),
        (85, "MO_a_T2P2_a", f"={A(43)}*(1+({A(80)}-1)*{s['INFL']})*(1+({A(81)}-1)*{s['INFL']})"),
        (87, "W1", f"=IFERROR(SUMPRODUCT({M_W}*({M_A}>{M_B}))/SUM({M_W}),0)"),
        (88, "Draw", f"=IFERROR(SUMPRODUCT({M_W}*({M_A}={M_B}))/SUM({M_W}),0)"),
        (89, "W2", f"=IFERROR(SUMPRODUCT({M_W}*({M_A}<{M_B}))/SUM({M_W}),0)"),
        (91, "Tour_W1", f"=IFERROR(SUMPRODUCT({M_W}*{tmask}*({M_A}>{M_B}))/{A(39)},0)"),
        (92, "Tour_Draw", f"=IFERROR(SUMPRODUCT({M_W}*{tmask}*({M_A}={M_B}))/{A(39)},0)"),
        (93, "Tour_W2", f"=IFERROR(SUMPRODUCT({M_W}*{tmask}*({M_A}<{M_B}))/{A(39)},0)"),
        (95, "Pois_W1", "=" + pois("i>j", f"${CH}$84", f"${CH}$85")),
        (96, "Pois_Draw", "=" + pois("i=j", f"${CH}$84", f"${CH}$85")),
        (97, "Pois_W2", "=" + pois("i<j", f"${CH}$84", f"${CH}$85")),
    ]
    for r_, w_, t_, p_ in ((99, 87, 91, 95), (100, 88, 92, 96), (101, 89, 93, 97)):
        lab = {99: "Final_W1", 100: "Final_Draw", 101: "Final_W2"}[r_]
        chain.append((r_, lab, f"=({A(w_)}*(1-${CH}$41)+{A(t_)}*${CH}$41)*(1-{s['PRES']})+{A(p_)}*{s['PRES']}"))
    for r_, lab, f_ in chain:
        hput(f"AT{r_}", lab); hput(f"{CH}{r_}", f_)
    hf.append({"range": f"AT1:AT101", "format": {"textFormat": {"bold": True}}})

    # ---------------- helper tab: chart data ----------------
    hc = cs(XH["C"], 7)
    for c, lab in zip(hc, ["Дата", "Тотал", "Тотал турнир", "Разница", "Разница турнир", "Тренд тотала", "Тренд разницы"]):
        hput(f"{c}1", lab)
    KEYX = (f'IF(LOWER(TRIM({P1}))<LOWER(TRIM({P2})),LOWER(TRIM({P1}))&"|"&LOWER(TRIM({P2})),'
            f'LOWER(TRIM({P2}))&"|"&LOWER(TRIM({P1})))')
    for k in range(70):
        rr = 2 + k
        d = f"${hc[0]}{rr}"
        hput(f"{hc[0]}{rr}", f"=INT({XIN['DTO']})" if k == 69 else f"={hc[0]}{rr + 1}-1")
        w = f'{D_KEY},{KEYX},{D_DT},"<"&{d}+1,{D_DT},">="&{d}+1-{XIN["GWIN"]}'
        wt = w + f",{D_TR},{TOUR}"
        hput(f"{hc[1]}{rr}", f'=IFERROR(SUMIFS({D_TOT},{w})/COUNTIFS({w}),NA())')
        hput(f"{hc[2]}{rr}", f'=IFERROR(SUMIFS({D_TOT},{wt})/COUNTIFS({wt}),NA())')
        hput(f"{hc[3]}{rr}", f'=IFERROR((SUMIFS({D_DIFF},{w},{D_P1},{P1})-SUMIFS({D_DIFF},{w},{D_P2},{P1}))/COUNTIFS({w}),NA())')
        hput(f"{hc[4]}{rr}", f'=IFERROR((SUMIFS({D_DIFF},{wt},{D_P1},{P1})-SUMIFS({D_DIFF},{wt},{D_P2},{P1}))/COUNTIFS({wt}),NA())')
    X = f"{hc[0]}2:{hc[0]}71"
    for tcol, ycol in ((hc[5], hc[1]), (hc[6], hc[3])):
        Y = f"{ycol}2:{ycol}71"
        hput(f"{tcol}2", f'=IFERROR(ARRAYFORMULA(IF(ISNUMBER({Y}),TREND(FILTER({Y},ISNUMBER({Y})),'
                         f'FILTER({X},ISNUMBER({Y})),{X}),NA())),"")')
    for lc, yc, name in ((XH["LBL"][0], hc[1], "Метки тотала"), (XH["LBL"][1], hc[3], "Метки разницы")):
        Y = f"{yc}2:{yc}71"
        hput(f"{lc}1", name)
        hput(f"{lc}2", f'=ARRAYFORMULA(IF(ISNUMBER({Y})*(MOD(71-ROW({Y}),{LBL_STEP})=0),TEXT({Y},"0.00"),""))')
    hf.append({"range": f"{hc[0]}2:{hc[0]}71", "format": {"numberFormat": {"type": "DATE", "pattern": "dd.mm.yyyy"}}})

    # ---------------- Console 1: settings panel AL112:AM133 ----------------
    for a1, v in (("AH9", "Строка дней"), ("AI9", "Байес (a)"), ("AH10", 1), ("AI10", 0.40),
                  ("AH11", "Линейность"), ("AI11", "Маржа 2"), ("AH12", 0.18), ("AI12", 0.075),
                  ("AH13", 0.06), ("AI13", "Маржа 3"),
                  ("AH14", f"=CEILING({s['BASE']}+(1-SQRT({s['LIN1']}))*100/{s['KDEC']})"), ("AI14", 0.09),
                  ("AH16", "Poisson_Res"), ("AI16", "Затухание"), ("AH17", 0.10), ("AI17", 1.5),
                  ("AH18", "Tour_Res"), ("AI18", "База дней"), ("AH19", 0.0), ("AI19", 7),
                  ("AH20", "MAX Влияние"), ("AI20", "Турнир"), ("AH21", 0.50), ("AI21", 15),
                  ("AH22", "Влияние"), ("AI22", "Команды"), ("AH23", 0.10), ("AI23", 200),
                  ("AH24", "MAX Влияние"), ("AI24", "И+К"), ("AH25", 0.60), ("AI25", 20),
                  ("AH27", "Update Date"), ("AI27", "МО+"), ("AH28", "=DATE(2025,12,4)"), ("AI28", 1.05),
                  ("AH29", "Critical Date"), ("AI29", "Old"), ("AH30", "=DATE(2025,11,10)"), ("AI30", 0.50)):
        cput(sset(a1).split("!")[1].replace("$", ""), v)
    a_ = lambda a1: sset(a1).split("!")[1].replace("$", "")
    cput(f"AI{X_MARK_CELL[2:]}", "Настройки угловых")
    cfmt(f"AH{X_MARK_CELL[2:]}:AI{X_MARK_CELL[2:]}", {"backgroundColor": GOLD, "textFormat": {"bold": True},
                                                  "horizontalAlignment": "CENTER"})
    for r in (9, 11, 16, 18, 20, 22, 24, 27, 29):
        cfmt(f"{a_(f'AH{r}')}:{a_(f'AI{r}')}", {"backgroundColor": GREEN, "textFormat": {"bold": True}, "horizontalAlignment": "CENTER"})
    cfmt(a_("AI13"), {"backgroundColor": GREEN, "textFormat": {"bold": True}, "horizontalAlignment": "CENTER"})
    cfmt(f"{a_('AH10')}:{a_('AI30')}", {"horizontalAlignment": "CENTER"})
    for a1 in ("AH10", "AI10", "AH12", "AI12", "AH13", "AI14", "AH17", "AI17", "AH19", "AI19", "AH21", "AI21",
               "AH23", "AI23", "AH25", "AI25", "AH28", "AI28", "AH30", "AI30"):
        cfmt(a_(a1), {"backgroundColor": LGOLD})
    cfmt(a_("AH14"), {"backgroundColor": PINK})
    for a1 in ("AI12", "AI14"):
        cfmt(a_(a1), {"numberFormat": {"type": "PERCENT", "pattern": "0.0%"}})
    for a1 in ("AH28", "AH30"):
        cfmt(a_(a1), {"numberFormat": {"type": "DATE", "pattern": "dd.mm.yyyy"}})


    # ---------------- Console 1: tables ----------------
    def table(c0, hb, title):
        c = cs(c0, 5)
        J, P1H1, P1H2, P2H1, P2H2 = HQ(hb[0]), HQ(hb[3]), HQ(hb[4]), HQ(hb[5]), HQ(hb[6])
        ok = f"ISNUMBER({J})"
        D = f"(({P1H1}+{P1H2})-({P2H1}+{P2H2}))"
        T = f"({P1H1}+{P1H2}+{P2H1}+{P2H2})"
        n = f"COUNT({J})"
        row = lambda r: R0 + r
        cnt = lambda e: f'=IF({n}=0,"",SUMPRODUCT({ok}*({e})))'
        o2 = lambda a, e: f'=IFERROR({x_odds(f"{a}/({a}+{e})", 2, s["MAR2"])},"")'

        cput(f"{c[0]}{row(1)}", title)
        cput(f"{c[0]}{row(2)}", "*"); cput(f"{c[3]}{row(2)}", "*")
        cput(f"{c[0]}{row(3)}", "=$A$3"); cput(f"{c[2]}{row(3)}", "X"); cput(f"{c[3]}{row(3)}", "=$D$3")
        merge(c[0], row(1), c[4], row(1))
        for r in (2, 3, 4, 5):
            merge(c[0], row(r), c[1], row(r)); merge(c[3], row(r), c[4], row(r))
        s3 = f"(${c[0]}${row(4)}+${c[2]}${row(4)}+${c[3]}${row(4)})"
        for cc, cond in ((c[0], f"{D}>0"), (c[2], f"{D}=0"), (c[3], f"{D}<0")):
            cput(f"{cc}{row(4)}", cnt(cond))
            cput(f"{cc}{row(5)}", f'=IFERROR({x_odds(f"${cc}${row(4)}/{s3}", 3, s["MAR3"])},"")')
        cput(f"{c[0]}{row(6)}", "Ф1"); cput(f"{c[4]}{row(6)}", "Ф2")
        cput(f"{c[1]}{row(6)}", f'=IF({n}=0,"Разница","Разница   "&TEXT(SUMPRODUCT({ok}*{D})/{n},"0.00"))')
        merge(c[1], row(6), c[3], row(6))
        mid_h = f"{c[2]}{row(9)}"
        for r, e in ((7, f"={mid_h}-1.5"), (8, f"={mid_h}-1"), (9, 0), (10, f"={mid_h}+1"), (11, f"={mid_h}+1.5")):
            L = f"${c[2]}${row(r)}"
            cput(f"{c[2]}{row(r)}", e)
            cput(f"{c[0]}{row(r)}", cnt(f"{D}+{L}>0")); cput(f"{c[4]}{row(r)}", cnt(f"{D}+{L}<0"))
            cput(f"{c[1]}{row(r)}", o2(f"{c[0]}{row(r)}", f"{c[4]}{row(r)}"))
            cput(f"{c[3]}{row(r)}", o2(f"{c[4]}{row(r)}", f"{c[0]}{row(r)}"))
        cput(f"{c[0]}{row(12)}", ">"); cput(f"{c[4]}{row(12)}", "<")
        cput(f"{c[1]}{row(12)}", f'=IF({n}=0,"Тотал","Тотал   "&TEXT(SUMPRODUCT({ok}*{T})/{n},"0.00"))')
        merge(c[1], row(12), c[3], row(12))
        mid_t = f"{c[2]}{row(15)}"
        cput(mid_t, f"=IF({n}=0,3.5,MAX(1.5,FLOOR(SUMPRODUCT({ok}*{T})/{n})+0.5))")
        for r in (13, 14, 16, 17):
            cput(f"{c[2]}{row(r)}", f"={mid_t}{r - 15:+d}")
        for r in range(13, 18):
            L = f"${c[2]}${row(r)}"
            cput(f"{c[0]}{row(r)}", cnt(f"{T}>{L}")); cput(f"{c[4]}{row(r)}", cnt(f"{T}<{L}"))
            cput(f"{c[1]}{row(r)}", o2(f"{c[0]}{row(r)}", f"{c[4]}{row(r)}"))
            cput(f"{c[3]}{row(r)}", o2(f"{c[4]}{row(r)}", f"{c[0]}{row(r)}"))
        g = lambda a, r1, b, r2: f"{a}{row(r1)}:{b}{row(r2)}"
        cfmt(g(c[0], 1, c[4], 17), {"horizontalAlignment": "CENTER"})
        cfmt(g(c[0], 1, c[4], 1), {"backgroundColor": GOLD, "textFormat": {"bold": True}})
        cfmt(g(c[0], 2, c[4], 3), {"backgroundColor": BLUE, "textFormat": {"bold": True}})
        cfmt(g(c[0], 4, c[4], 4), {"backgroundColor": PINK, "numberFormat": {"type": "NUMBER", "pattern": "0"}})
        cfmt(g(c[0], 5, c[4], 5), {"backgroundColor": GREY, "textFormat": {"bold": True},
                                   "numberFormat": {"type": "NUMBER", "pattern": "0.00"}})
        for r in (6, 12):
            cfmt(g(c[0], r, c[4], r), {"backgroundColor": BLUE, "textFormat": {"bold": True}})
        for a, b in ((7, 11), (13, 17)):
            for cc, bg, pat in ((c[0], PINK, "0"), (c[4], PINK, "0"), (c[1], GREY, "0.00"), (c[3], GREY, "0.00")):
                cfmt(g(cc, a, cc, b), {"backgroundColor": bg, "numberFormat": {"type": "NUMBER", "pattern": pat}})
            cfmt(g(c[2], a, c[2], b), {"backgroundColor": BLUE})
        cfmt(g(c[2], 7, c[2], 11), {"numberFormat": {"type": "NUMBER", "pattern": "+0.0;-0.0;0"}})
        cfmt(g(c[2], 13, c[2], 17), {"numberFormat": {"type": "NUMBER", "pattern": "0.0"}})
        for r in (9, 15):
            cfmt(g(c[0], r, c[4], r), {"textFormat": {"bold": True}})
            cfmt(g(c[2], r, c[2], r), {"backgroundColor": MIDC})

    TQ = q(HS, XIN["TOUR"])
    table("A", hl, '="УГЛОВЫЕ  * все турниры ("&$A$3&" - "&$D$3&")"')
    table("G", hr, f'="УГЛОВЫЕ  "&{TQ}&" ("&$A$3&" - "&$D$3&")"')

    # ---------------- Console 1: compact summary U:AD (rows R0+1 .. R0+14) ----------------
    J, P1H1, P1H2, P2H1, P2H2 = HQ(hl[0]), HQ(hl[3]), HQ(hl[4]), HQ(hl[5]), HQ(hl[6])
    nL = f"COUNT({J})"
    r1 = R0 + 1
    cols = cs("U", 10)          # U label, V..AD = И В Н П УЗ УП Т ТЗ ТП
    cput(f"U{r1}", f'="Угловые: "&{TQ}')
    for c, lab in zip(cols[1:], ["И", "В", "Н", "П", "УЗ", "УП", "Т", "ТЗ", "ТП"]):
        cput(f"{c}{r1}", lab)
    win = f'{D_DT},">="&$N$2,{D_DT},"<"&$N$4+1,{D_TR},{TQ}'

    def prow(r_, pl, team, label):
        y = cols
        a = f"{D_P1},{pl},{D_T1},{team},{win}"
        b = f"{D_P2},{pl},{D_T2},{team},{win}"
        cput(f"{y[0]}{r_}", label)
        cput(f"{y[1]}{r_}", f"=COUNTIFS({a})+COUNTIFS({b})")
        cput(f"{y[2]}{r_}", f'=COUNTIFS({D_DIFF},">0",{a})+COUNTIFS({D_DIFF},"<0",{b})')
        cput(f"{y[3]}{r_}", f'=COUNTIFS({D_DIFF},"=0",{a})+COUNTIFS({D_DIFF},"=0",{b})')
        cput(f"{y[4]}{r_}", f'=COUNTIFS({D_DIFF},"<0",{a})+COUNTIFS({D_DIFF},">0",{b})')
        cput(f"{y[5]}{r_}", f"=SUMIFS({D_C1},{a})+SUMIFS({D_C2},{b})")
        cput(f"{y[6]}{r_}", f"=SUMIFS({D_C2},{a})+SUMIFS({D_C1},{b})")
        cput(f"{y[7]}{r_}", f'=IFERROR(({y[5]}{r_}+{y[6]}{r_})/{y[1]}{r_},"")')
        cput(f"{y[8]}{r_}", f'=IFERROR({y[5]}{r_}/{y[1]}{r_},"")')
        cput(f"{y[9]}{r_}", f'=IFERROR({y[6]}{r_}/{y[1]}{r_},"")')

    prow(r1 + 1, "$A$3", '"*"', '=$A$3&" ( * )"')
    prow(r1 + 2, "$A$3", "$A$2", '=$A$3&" ("&$A$2&")"')
    prow(r1 + 3, "$D$3", '"*"', '=$D$3&" ( * )"')
    prow(r1 + 4, "$D$3", "$D$2", '=$D$3&" ("&$D$2&")"')
    cfmt(f"U{r1}:AD{r1}", {"backgroundColor": GOLD, "textFormat": {"bold": True}, "horizontalAlignment": "CENTER"})
    for k, bg in ((1, BLUE), (2, GREY), (3, BLUE), (4, GREY)):
        cfmt(f"U{r1 + k}:AD{r1 + k}", {"backgroundColor": bg, "horizontalAlignment": "CENTER"})
    cfmt(f"AB{r1 + 1}:AD{r1 + 4}", {"numberFormat": {"type": "NUMBER", "pattern": "0.00"}})

    # halves (U:Z) + panel (AB:AC)
    h0 = r1 + 6
    cput(f"U{h0}", '=$A$3&" - "&$D$3'); cput(f"V{h0}", "УГ"); cput(f"W{h0}", "%УГ"); cput(f"X{h0}", "*")
    cput(f"Y{h0}", "УГ"); cput(f"Z{h0}", "%УГ")
    cput(f"U{h0 + 1}", "1-й тайм"); cput(f"U{h0 + 2}", "2-й тайм")
    cput(f"V{h0 + 1}", f"=SUM({P1H1})+SUM({P2H1})"); cput(f"V{h0 + 2}", f"=SUM({P1H2})+SUM({P2H2})")
    cput(f"W{h0 + 1}", f'=IFERROR(V{h0 + 1}/(V{h0 + 1}+V{h0 + 2}),"")')
    cput(f"W{h0 + 2}", f'=IFERROR(V{h0 + 2}/(V{h0 + 1}+V{h0 + 2}),"")')
    allwin = f'{D_DT},">="&$N$2,{D_DT},"<"&$N$4+1'
    cput(f"Y{h0 + 1}", f"=SUMIFS({D_H1},{allwin})"); cput(f"Y{h0 + 2}", f"=SUMIFS({D_H2},{allwin})")
    cput(f"Z{h0 + 1}", f'=IFERROR(Y{h0 + 1}/(Y{h0 + 1}+Y{h0 + 2}),"")')
    cput(f"Z{h0 + 2}", f'=IFERROR(Y{h0 + 2}/(Y{h0 + 1}+Y{h0 + 2}),"")')
    merge("X", h0 + 1, "X", h0 + 2)
    cfmt(f"U{h0}:Z{h0}", {"backgroundColor": GREY, "textFormat": {"bold": True}, "horizontalAlignment": "CENTER"})
    cfmt(f"U{h0 + 1}:U{h0 + 2}", {"backgroundColor": GREY, "horizontalAlignment": "CENTER"})
    cfmt(f"V{h0 + 1}:W{h0 + 2}", {"backgroundColor": BLUE, "horizontalAlignment": "CENTER"})
    cfmt(f"Y{h0 + 1}:Z{h0 + 2}", {"backgroundColor": BLUE, "horizontalAlignment": "CENTER"})
    cfmt(f"W{h0 + 1}:W{h0 + 2}", {"numberFormat": {"type": "NUMBER", "pattern": "0.0000"}})
    cfmt(f"Z{h0 + 1}:Z{h0 + 2}", {"numberFormat": {"type": "NUMBER", "pattern": "0.0000"}})

    for a1, v in ((f"AB{h0}", "Матчи"), (f"AC{h0}", "Разница"),
                  (f"AB{h0 + 1}", f"={nL}"),
                  (f"AC{h0 + 1}", f'=IFERROR((SUM({P1H1})+SUM({P1H2})-SUM({P2H1})-SUM({P2H2}))/{nL},"")'),
                  (f"AB{h0 + 2}", "Тотал"), (f"AC{h0 + 2}", "Часть H1"),
                  (f"AB{h0 + 3}", f'=IFERROR((SUM({P1H1})+SUM({P1H2})+SUM({P2H1})+SUM({P2H2}))/{nL},"")'),
                  (f"AC{h0 + 3}", f'=IFERROR((SUM({P1H1})+SUM({P2H1}))/(SUM({P1H1})+SUM({P1H2})+SUM({P2H1})+SUM({P2H2})),"")'),
                  (f"AB{h0 + 5}", "МО (!!!)"), (f"AC{h0 + 5}", "Разница (!!!)"),
                  (f"AB{h0 + 6}", f"={CHQ(84)}+{CHQ(85)}"), (f"AC{h0 + 6}", f"={CHQ(84)}-{CHQ(85)}")):
        cput(a1, v)
    cfmt(f"AB{h0}:AC{h0 + 6}", {"horizontalAlignment": "CENTER"})
    for k in (0, 2, 5):
        cfmt(f"AB{h0 + k}:AC{h0 + k}", {"backgroundColor": BLUE, "textFormat": {"bold": True}})
    cfmt(f"AB{h0 + 1}", {"backgroundColor": PINK, "numberFormat": {"type": "NUMBER", "pattern": "0"}})
    cfmt(f"AC{h0 + 1}", {"backgroundColor": PINK, "numberFormat": {"type": "NUMBER", "pattern": "0.00"}})
    cfmt(f"AB{h0 + 3}", {"backgroundColor": PINK, "numberFormat": {"type": "NUMBER", "pattern": "0.00"}})
    cfmt(f"AC{h0 + 3}", {"backgroundColor": PINK, "numberFormat": {"type": "PERCENT", "pattern": "0.0%"}})
    cfmt(f"AB{h0 + 6}:AC{h0 + 6}", {"backgroundColor": GOLD, "textFormat": {"bold": True},
                                    "numberFormat": {"type": "NUMBER", "pattern": "0.00"}})

    # result (U:X)
    rr0 = h0 + 4
    cput(f"U{rr0}", "Результат (угл.)"); cput(f"V{rr0}", "W1"); cput(f"W{rr0}", "Draw"); cput(f"X{rr0}", "W2")
    cput(f"U{rr0 + 1}", "Poisson Result"); cput(f"U{rr0 + 2}", "Final Result")
    for c, pr, fr in (("V", 95, 99), ("W", 96, 100), ("X", 97, 101)):
        cput(f"{c}{rr0 + 1}", f'=IFERROR({x_odds(CHQ(pr), 3, s["MAR3"])},"")')
        cput(f"{c}{rr0 + 2}", f'=IFERROR({x_odds(CHQ(fr), 3, s["MAR3"])},"")')
    cfmt(f"U{rr0}:X{rr0}", {"backgroundColor": BLUE, "textFormat": {"bold": True}, "horizontalAlignment": "CENTER"})
    cfmt(f"U{rr0 + 1}:U{rr0 + 2}", {"backgroundColor": BLUE, "horizontalAlignment": "CENTER"})
    cfmt(f"V{rr0 + 1}:X{rr0 + 1}", {"backgroundColor": LGOLD, "horizontalAlignment": "CENTER",
                                    "numberFormat": {"type": "NUMBER", "pattern": "0.00"}})
    cfmt(f"V{rr0 + 2}:X{rr0 + 2}", {"backgroundColor": GOLD, "textFormat": {"bold": True}, "horizontalAlignment": "CENTER",
                                    "numberFormat": {"type": "NUMBER", "pattern": "0.00"}})

    # ---------------- charts (data on helper tab, drawn in Console 1) ----------------
    def src(col):
        return {"sourceRange": {"sources": [grid(hs.id, col, 1, col, 71)]}}

    def ser(col, rgb, label_col=None):
        dl = ({"type": "CUSTOM", "customLabelData": src(label_col), "placement": "ABOVE"}
              if label_col else {"type": "NONE"})
        return {"series": src(col), "targetAxis": "LEFT_AXIS", "colorStyle": {"rgbColor": rgb}, "color": rgb,
                "lineStyle": {"width": 2, "type": "SOLID"}, "dataLabel": dl}

    def chart(title, ycol, trend, lbl, anchor_col, vmin=None, vmax=None, row0=R0 + SHIFT + 3, w=400, h=330):
        left = {"position": "LEFT_AXIS"}
        if vmin is not None:
            left["viewWindowOptions"] = {"viewWindowMin": vmin, "viewWindowMax": vmax, "viewWindowMode": "EXPLICIT"}
        return {"addChart": {"chart": {
            "spec": {"title": title, "basicChart": {
                "chartType": "LINE", "legendPosition": "NO_LEGEND", "headerCount": 1,
                "axis": [{"position": "BOTTOM_AXIS"}, left],
                "domains": [{"domain": src(hc[0])}],
                "series": [ser(ycol, {"red": 0.26, "green": 0.52, "blue": 0.96}, lbl),
                           ser(trend, {"red": 0.62, "green": 0.74, "blue": 0.98})],
                "interpolateNulls": True}},
            "position": {"overlayPosition": {
                "anchorCell": {"sheetId": c1.id, "rowIndex": row0, "columnIndex": colnum(anchor_col) - 1},
                "widthPixels": w, "heightPixels": h}}}}}

    charts = [chart("Средняя разница угловых", hc[3], hc[6], XH["LBL"][1], "P", -3, 3),
              chart("Средний тотал угловых", hc[1], hc[5], XH["LBL"][0], "U")]

    # existing goals charts below the tables -> down SHIFT rows (same column, offsets, size)
    meta = book.fetch_sheet_metadata({"fields": "sheets(properties.sheetId,charts(chartId,position))"})
    moves = []
    for sh_ in meta["sheets"]:
        if sh_["properties"]["sheetId"] != c1.id:
            continue
        for chrt in sh_.get("charts", []):
            ov = chrt.get("position", {}).get("overlayPosition")
            if not ov or ov["anchorCell"].get("rowIndex", 0) < R0:
                continue
            new = dict(ov); new["anchorCell"] = dict(ov["anchorCell"])
            new["anchorCell"]["rowIndex"] = ov["anchorCell"].get("rowIndex", 0) + SHIFT
            moves.append({"updateEmbeddedObjectPosition": {
                "objectId": chrt["chartId"], "newPosition": {"overlayPosition": new},
                "fields": "anchorCell,offsetXPixels,offsetYPixels,widthPixels,heightPixels"}})
    team_move = []
    if has_team:
        team_move = [{"cutPaste": {"source": grid(c1.id, sc, sr1, ec, sr2),
                                   "destination": {"sheetId": c1.id, "rowIndex": dr1 - 1,
                                                   "columnIndex": colnum(dc) - 1},
                                   "pasteType": "PASTE_NORMAL"}}]
    print(f"{C1}: move {len(moves)} charts down {SHIFT} rows, team block moved: {bool(team_move)}")

    # ---------------- write: helper first, then Console 1 formats, layout, values, marker ----------------
    for k in range(0, len(hf), 25):
        retry(hs.batch_format, hf[k:k + 25])
    for k in range(0, len(hcells), 80):
        retry(hs.batch_update, hcells[k:k + 80], value_input_option="USER_ENTERED")
        time.sleep(1)
    print(f"{HS}: built")
    for k in range(0, len(cf), 25):
        retry(c1.batch_format, cf[k:k + 25])
    retry(book.batch_update, {"requests": team_move + moves})          # make room first
    reqs.append({"updateDimensionProperties": {
        "range": {"sheetId": c1.id, "dimension": "COLUMNS", "startIndex": colnum("U") - 1, "endIndex": colnum("U")},
        "properties": {"pixelSize": 150}, "fields": "pixelSize"}})
    retry(book.batch_update, {"requests": reqs + charts})
    for k in range(0, len(ccells), 80):
        retry(c1.batch_update, ccells[k:k + 80], value_input_option="USER_ENTERED")
        time.sleep(1)
    retry(c1.update, [[X_MARK]], X_MARK_CELL)
    print(f"{C1}: corners block built: A{R0 + 1}:K{R0 + 17}, U{R0 + 1}:AD{R0 + 14}, settings AH{X_MARK_CELL[2:]}")


# ---------------- one-off edit of an already built block (no rebuild) ----------------
# v1 -> v2: event list N17:T37 -> U104; corners players summary -> P18 (under goals summary);
#           halves -> P25, result -> P29, panel (Матчи/Тотал/МО) -> M24:N30  (mirrors goals rows 1-17)
#           + fixes AU40/AU41/AU76/AU77 in Corners_Model (no division by zero when Tour_Res/MAX = 0)
X_MARK2 = "УГЛОВЫЕ v2"


def relayout_console(book):
    c1 = book.worksheet(C1)
    mark = (c1.acell(X_MARK_CELL).value or "").strip()
    if mark != X_MARK:
        print(f"{C1}: marker is '{mark}', expected '{X_MARK}' -> relayout skipped")
        return
    # targets inside N17:T37 are freed by moving the event list first -> check only the rest
    for rng in ("U104:AA124", "AK104:AT112", "M24:M30"):
        busy = [v for row in c1.get(rng) for v in row if str(v).strip()]
        if busy:
            print(f"{C1}: {rng} is not empty -> relayout aborted, nothing changed")
            return
    hdr = c1.acell("N17").value or ""
    if "Event" not in hdr:
        print(f"{C1}: N17 is '{hdr}', event list not found -> relayout aborted, nothing changed")
        return

    def g(a, b):
        m1, m2 = re.fullmatch(r"([A-Z]+)(\d+)", a), re.fullmatch(r"([A-Z]+)(\d+)", b)
        return {"sheetId": c1.id, "startRowIndex": int(m1.group(2)) - 1, "endRowIndex": int(m2.group(2)),
                "startColumnIndex": colnum(m1.group(1)) - 1, "endColumnIndex": colnum(m2.group(1))}

    def cut(a, b, dest):
        m = re.fullmatch(r"([A-Z]+)(\d+)", dest)
        return {"cutPaste": {"source": g(a, b), "pasteType": "PASTE_NORMAL",
                             "destination": {"sheetId": c1.id, "rowIndex": int(m.group(2)) - 1,
                                             "columnIndex": colnum(m.group(1)) - 1}}}

    reqs = [
        cut("N17", "T37", "U104"),             # goals event list (QUERY keeps working, refs are absolute)
        cut("U18", "AD22", "AK104"),           # corners players summary -> temp (source/target overlap)
        cut("U24", "Z26", "AK110"),            # corners halves -> temp
        cut("AB24", "AC30", "M24"),            # corners panel
        cut("U28", "X30", "P29"),              # corners result
        cut("AK104", "AT108", "P18"),          # summary -> final
        cut("AK110", "AP112", "P25"),          # halves -> final
        {"updateDimensionProperties": {"range": {"sheetId": c1.id, "dimension": "COLUMNS",
                                                 "startIndex": colnum("U") - 1, "endIndex": colnum("U")},
                                       "properties": {"pixelSize": 120}, "fields": "pixelSize"}},
    ]
    retry(book.batch_update, {"requests": reqs})
    print(f"{C1}: relayout done (event list -> U104, corners summary -> P18, halves -> P25, "
          f"result -> P29, panel -> M24:N30)")

    try:
        hs = book.worksheet(HS)
        s = XS
        fix = [("AU40", f"=IF(AU39>{s['NTOUR']},{s['MAXT']},AU39*{s['MAXT']}/{s['NTOUR']})"),
               ("AU41", f"=IF(AU39>{s['NTOUR']},{s['TRES']},AU39*{s['TRES']}/{s['NTOUR']})"),
               ("AU76", f"=IF(AU60>{s['NPT']},{s['MAXPT']},AU60*{s['MAXPT']}/{s['NPT']})"),
               ("AU77", f"=IF(AU65>{s['NPT']},{s['MAXPT']},AU65*{s['MAXPT']}/{s['NPT']})")]
        retry(hs.batch_update, [{"range": a, "values": [[f]]} for a, f in fix], value_input_option="USER_ENTERED")
        print(f"{HS}: AU40/AU41/AU76/AU77 fixed")
    except gspread.WorksheetNotFound:
        print(f"{HS} not found -> formula fix skipped")
    retry(c1.update, [[X_MARK2]], X_MARK_CELL)


# ---------------- DATA (goals) vs parquet: session dates + diagnostics ----------------
# DATA keeps the SESSION date for night matches (e.g. 23.10 06:50 is stored as 22.10 06:50).
# We detect from the data which clock hours are stored with the previous date and write corners the same
# way, so windows, day weights and H2H lists match the goals block 1:1.
CHECK_DAYS = 30


def load_data_matches(data_ws):
    vals = data_ws.get_values("A2:G", value_render_option="UNFORMATTED_VALUE",
                              date_time_render_option="SERIAL_NUMBER")
    rows = []
    for v in vals:
        v = (list(v) + [""] * 7)[:7]
        try:
            d = float(v[0]) + (float(v[2]) if str(v[2]).strip() else 0.0)
        except (TypeError, ValueError):
            continue
        rows.append((d, str(v[1]).strip(), str(v[5]).strip(), str(v[6]).strip()))
    g = pd.DataFrame(rows, columns=["serial", "tour", "p1", "p2"])
    g["dt"] = (pd.to_datetime("1899-12-30") + pd.to_timedelta(g["serial"], unit="D")).dt.round("min")
    g["key"] = ["|".join(sorted([a.lower(), b.lower()])) for a, b in zip(g.p1, g.p2)]
    return g


def parquet_matches():
    m = meta.copy()                                     # ALL parquet matches, incl. dropped ones
    utc = pd.to_datetime(m["match_date"], errors="coerce", format="ISO8601", utc=True)
    m["loc"] = utc.dt.tz_convert(X_TZ).dt.tz_localize(None).dt.round("min")
    m["key"] = ["|".join(sorted([str(a).lower(), str(b).lower()])) for a, b in zip(m.p1, m.p2)]
    m["tour"] = m["league"].str.replace(r"^ESB\s+", "", regex=True).str.strip()
    m["in_db"] = m["match_id"].isin(set(out["match_id"]))
    return m


def detect_session_hours(g, m):
    recent = g[g["dt"] >= g["dt"].max() - pd.Timedelta(days=CHECK_DAYS)]
    idx = set(zip(m["key"], m["loc"]))
    same, prev = [0] * 24, [0] * 24

    def hit(k, t):
        return any((k, t + pd.Timedelta(minutes=d)) in idx for d in (-1, 0, 1))
    for r in recent.itertuples():
        if hit(r.key, r.dt):
            same[r.dt.hour] += 1
        elif hit(r.key, r.dt + pd.Timedelta(days=1)):
            prev[r.dt.hour] += 1
    hours = {h for h in range(24) if prev[h] > same[h] and prev[h] >= 3}
    print("session check by hour (same date / stored as previous date):",
          {h: (same[h], prev[h]) for h in range(24) if same[h] or prev[h]})
    print("DATA stores previous date for hours:", sorted(hours) or "none")
    return hours


def to_session(ts, hours):
    return ts - pd.to_timedelta(ts.dt.hour.isin(hours).astype(int), unit="D")


def check_console(book, g, m, hours):
    """Sheet 'Corners_Check': DATA matches of the last CHECK_DAYS days (up to the last parquet match)
    that have no corners, with the reason."""
    m = m.copy()
    m["sess"] = to_session(m["loc"], hours)
    last = m["sess"].max()
    g = g[(g["dt"] >= g["dt"].max() - pd.Timedelta(days=CHECK_DAYS)) & (g["dt"] <= last)]
    m = m[m["key"].isin(set(g["key"]))]
    by_key = {k: v for k, v in m.groupby("key")}
    empty = m.iloc[0:0]
    out_rows, ok, cnt = [], 0, {}
    for r in g.itertuples():
        cand = by_key.get(r.key, empty)
        diff = (cand["sess"] - r.dt).abs()
        exact = cand[diff <= pd.Timedelta(minutes=5)]
        c = None
        if len(exact):
            c = exact.iloc[0]
            if c["in_db"] and c["tour"].lower() == r.tour.lower():
                ok += 1
                continue
            reason = ("в паркете есть, но выкинут: не дошёл до 2-го тайма" if not c["in_db"]
                      else f"в паркете другой турнир: {c['tour']}")
        else:
            reason = "матча нет в паркете"
            if len(cand):
                c = cand.loc[diff.idxmin()]
        cnt[reason] = cnt.get(reason, 0) + 1
        out_rows.append([r.dt.strftime("%Y-%m-%d %H:%M"), r.tour, f"{r.p1} - {r.p2}", reason,
                         c["sess"].strftime("%Y-%m-%d %H:%M") if c is not None else "",
                         c["tour"] if c is not None else ""])
    print(f"check: DATA matches {g['dt'].min():%d.%m}-{last:%d.%m %H:%M}: {len(g)}, "
          f"with corners: {ok}, without: {len(out_rows)}")
    print("check reasons:", cnt or "none")
    rep = pd.DataFrame(out_rows, columns=["DATA время", "DATA турнир", "Пара", "Причина",
                                          "Ближайший в паркете", "Турнир в паркете"])
    if rep.empty:
        rep = pd.DataFrame([["все матчи DATA найдены", "", "", "", "", ""]], columns=rep.columns)
    write_ws("Corners_Check", rep, book=book)


if X_ID_RAW:
    X_ID = re.search(r"[-\w]{25,}", X_ID_RAW).group(0)
    xbook = gc.open_by_key(X_ID)
    data = xbook.worksheet("DATA")
    gdata = load_data_matches(data)
    tours = {t.lower() for t in gdata["tour"] if t}
    plist = {x.lower() for x in pd.concat([gdata["p1"], gdata["p2"]]) if x}
    print(f"console DATA: {len(gdata)} matches, {len(tours)} tournaments, {len(plist)} players")
    pm = parquet_matches()
    sess_hours = detect_session_hours(gdata, pm)
    keep = (out["tournament"].str.lower().isin(tours) & out["p1"].str.lower().isin(plist)
            & out["p2"].str.lower().isin(plist))
    db2 = out[keep].copy()
    loc = dt.loc[db2.index].dt.tz_convert(X_TZ).dt.tz_localize(None)
    db2["event_dt"] = to_session(loc, sess_hours).dt.strftime("%Y-%m-%d %H:%M:%S")
    print("console corners by tournament:", db2["tournament"].value_counts().to_dict())
    write_ws(WS_NAME, db2, book=xbook)
    if os.environ.get("CHECK_CONSOLE", "0") == "1":
        check_console(xbook, gdata, pm, sess_hours)
    if os.environ.get("BUILD_CONSOLE", "0") == "1":
        build_console(xbook)
        relayout_console(xbook)              # fresh build -> final layout right away
    elif os.environ.get("RELAYOUT_CONSOLE", "0") == "1":
        relayout_console(xbook)              # edit the existing block in place, no rebuild
    else:
        print(f"console: data only, {C1} not touched")
else:
    print("GSHEET_ID_CONSOLE not set -> console file skipped")
