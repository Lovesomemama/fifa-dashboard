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
sh = gc.open_by_key(SHEET_ID)


def write_ws(name, frame, chunk=5000):
    try:
        ws = sh.worksheet(name)
    except gspread.WorksheetNotFound:
        ws = sh.add_worksheet(name, rows=1, cols=len(frame.columns))
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


write_ws(WS_NAME, out)
write_ws("PAIRS", pairs)
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

    def ser(col, rgb, width):
        return {"series": src(col), "targetAxis": "LEFT_AXIS",
                "colorStyle": {"rgbColor": rgb}, "color": rgb,
                "lineStyle": {"width": width, "type": "SOLID"},
                "dataLabel": {"type": "NONE"}}

    def chart(title, ycol, trend, anchor_col, vmin=None, vmax=None):
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
                    "series": [ser(ycol, {"red": 0.26, "green": 0.52, "blue": 0.96}, 2),
                               ser(trend, {"red": 0.62, "green": 0.74, "blue": 0.98}, 2)],
                    "interpolateNulls": True,
                }},
            "position": {"overlayPosition": {
                "anchorCell": {"sheetId": sheet_id, "rowIndex": 18, "columnIndex": colnum(anchor_col) - 1},
                "widthPixels": 470, "heightPixels": 400}}}}}

    return [chart("Средняя разница угловых", hc[3], hc[6], "A", -3, 3),
            chart("Средний тотал угловых", hc[1], hc[5], "G")]


def refresh_charts(sh, ws):
    meta = sh.fetch_sheet_metadata({"fields": "sheets(properties.sheetId,charts.chartId)"})
    ids = [c["chartId"] for s in meta["sheets"] if s["properties"]["sheetId"] == ws.id
           for c in s.get("charts", [])]
    reqs = [{"deleteEmbeddedObject": {"objectId": i}} for i in ids]
    retry(sh.batch_update, {"requests": reqs + chart_requests(ws.id)})
    print(f"CALC charts refreshed ({len(ids)} removed)")


def build_calc(sh):
    ws = sh.add_worksheet(CALC_NAME, rows=200, cols=90, index=0)
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
        (40, "a_tour", f"=IF(AI39>{NTOUR},{MAXT},AI39/({NTOUR}/{MAXT}))"),
        (41, "a_tour_Res", f"=IF(AI39>{NTOUR},{TRES},AI39/({NTOUR}/{TRES}))"),
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
        (76, "a_P1T1", f"=IF(AI60>{NPT},{MAXPT},AI60/({NPT}/{MAXPT}))"),
        (77, "a_P2T2", f"=IF(AI65>{NPT},{MAXPT},AI65/({NPT}/{MAXPT}))"),
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
        put(f"{tcol}2", f'=IFERROR(ARRAYFORMULA(TREND(FILTER({Y},ISNUMBER({Y})),FILTER({X},ISNUMBER({Y})),{X})),"")')
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

    retry(sh.batch_update, {"requests": reqs + widths + [
        dv("A1", f"={CALC_NAME}!${ls[0]}$2:${ls[0]}"),
        dv("A3", f"={CALC_NAME}!${ls[1]}$2:${ls[1]}"),
        dv("D3", f"={CALC_NAME}!${ls[2]}$2:${ls[2]}"),
        dv("A2", f"={CALC_NAME}!${ls[3]}$2:${ls[3]}"),
        dv("D2", f"={CALC_NAME}!${ls[4]}$2:${ls[4]}"),
    ] + chart_requests(ws.id)})
    print("CALC layout ok")

    for i in range(0, len(cells), 80):
        retry(ws.batch_update, cells[i:i + 80], value_input_option="USER_ENTERED")
        time.sleep(1)
    retry(ws.update, [[MARKER]], MARKER_CELL)
    print("CALC built")


existing = {w.title: w for w in sh.worksheets()}
if CALC_NAME in existing and existing[CALC_NAME].acell(MARKER_CELL).value != MARKER:
    print("CALC incomplete -> rebuild")
    retry(sh.del_worksheet, existing.pop(CALC_NAME))
if CALC_NAME not in existing:
    build_calc(sh)
else:
    print("CALC exists, formulas kept")
    refresh_charts(sh, existing[CALC_NAME])
