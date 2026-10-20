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
# layout 1:1 with Volta sheet:
#   A:E  table 1 = all tournaments H2H (input chips in header)
#   G:K  table 2 = selected tournament + teams
#   M:N  settings, P:T summary, V+ helpers/lists

TOUR, T1, T2, PL1, PL2 = "$A$1", "$A$2", "$D$2", "$A$3", "$D$3"
WIN, DFROM, HALF, DTO, MF, MX = "$M$2", "$N$2", "$M$4", "$N$4", "$M$6", "$N$6"
SR0 = 3   # first summary metric row


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


def h2h(use_tour, use_team):
    # LET names must not look like cell refs (p1, t1, l1 ... are invalid)
    tour = f'(({TOUR}="*")+(INDEX(db,,3)={TOUR}))>0,' if use_tour else ''
    team = (f'g,FILTER(r,(({T1}="*")+(ta={T1}))>0,(({T2}="*")+(tb={T2}))>0),'
            if use_team else 'g,r,')
    return (
        '=IFERROR(ARRAYFORMULA(LET('
        'db,DB_Corners!A2:Q,'
        f'pa,LOWER(TRIM({PL1})),pb,LOWER(TRIM({PL2})),'
        'k,INDEX(db,,17),'
        'f,FILTER(db,((k=pa&"|"&pb)+(k=pb&"|"&pa))>0,' + tour +
        f'INDEX(db,,2)>={DFROM},INDEX(db,,2)<{DTO}+1),'
        'a,LOWER(INDEX(f,,5))=pa,'
        'd,INDEX(f,,2),'
        'ta,IF(a,INDEX(f,,4),INDEX(f,,6)),'
        'tb,IF(a,INDEX(f,,6),INDEX(f,,4)),'
        'r,HSTACK(d,ta,tb,'
        'IF(a,INDEX(f,,8),INDEX(f,,10)),IF(a,INDEX(f,,9),INDEX(f,,11)),'
        'IF(a,INDEX(f,,10),INDEX(f,,8)),IF(a,INDEX(f,,11),INDEX(f,,9)),'
        f'EXP(-LN(2)*({DTO}+1-d)/{HALF})),'
        + team +
        'SORT(g,1,FALSE))),"нет H2H")'
    )


def guard(f, cnt):
    return f'=IF({cnt}=0,"",' + f[1:] + ')'


def _pm(la, lb, tail, cnt, mcell):
    return guard('=IFERROR(ARRAYFORMULA(LET(la,' + la + ',lb,' + lb + ','
                 'i,MAKEARRAY(31,31,LAMBDA(r,c,r-1)),j,MAKEARRAY(31,31,LAMBDA(r,c,c-1)),'
                 'm,POISSON.DIST(i,la,FALSE)*POISSON.DIST(j,lb,FALSE),'
                 '(' + tail + ')/(1+' + mcell + '))),"")', cnt)


def mxa(win, push, la, lb, cnt):
    return _pm(la, lb, '(1-SUM(m*(' + push + ')))/SUM(m*(' + win + '))', cnt, MF)


def mx(cond, la, lb, cnt):
    return _pm(la, lb, '1/SUM(m*(' + cond + '))', cnt, MX)


BLUE = {"red": 0.79, "green": 0.85, "blue": 0.97}
PINK = {"red": 0.96, "green": 0.80, "blue": 0.80}
GREY = {"red": 0.95, "green": 0.95, "blue": 0.95}
MIDC = {"red": 0.86, "green": 0.82, "blue": 0.93}
GOLD = {"red": 1, "green": 0.85, "blue": 0.4}


def build_calc(sh):
    ws = sh.add_worksheet(CALC_NAME, rows=200, cols=50, index=0)
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

    # ---- settings M:N ----
    for a1, v in (("M1", "Выборка дней"), ("N1", "От"), ("M2", 60), ("N2", "=N4-M2"),
                  ("M3", "Полураспад, дн"), ("N3", "До"), ("M4", 30), ("N4", "=TODAY()"),
                  ("M5", "Маржа Ф/Тотал"), ("N5", "Маржа 1X2"), ("M6", 0.075), ("N6", 0.10)):
        put(a1, v)
    fmt("M1:N6", {"horizontalAlignment": "CENTER"})
    for rr in (1, 3, 5):
        fmt(f"M{rr}:N{rr}", {"backgroundColor": BLUE, "textFormat": {"bold": True}})
    fmt("M2:M4", {"backgroundColor": GREY})
    fmt("N2", {"backgroundColor": PINK, "numberFormat": {"type": "DATE", "pattern": "dd.mm.yyyy"}})
    fmt("N4", {"backgroundColor": PINK, "numberFormat": {"type": "DATE", "pattern": "dd.mm.yyyy"}})
    fmt("M6:N6", {"backgroundColor": GOLD, "textFormat": {"bold": True},
                  "numberFormat": {"type": "PERCENT", "pattern": "0.0%"}})

    # ---- summary P:T ----
    labels = ["H2H матчей / эфф.", "λ1 (Игрок 1)", "λ2 (Игрок 2)", "М.О.Угл",
              "Разница (λ1−λ2)", "Часть H1", "Часть H2", "H1 доля Игрок 1",
              "H1 доля Игрок 2", "% ничьих факт", "% ничьих Пуассон", "ДнаН подсказка"]
    put("P1", "Сводка")
    for i, lab in enumerate(labels):
        put(f"P{SR0 + i}", lab)
    fmt("P1:T2", {"backgroundColor": BLUE, "textFormat": {"bold": True}, "horizontalAlignment": "CENTER"})
    fmt(f"P{SR0}:P{SR0 + 11}", {"textFormat": {"bold": True}, "backgroundColor": GREY})

    def block(cs, cd, sum_title, helper0, use_tour, use_team, c0, header):
        hc = [colstr(colnum(helper0) + i) for i in range(8)]
        for c, lab in zip(hc, ["Дата", "Команда P1", "Команда P2",
                               "P1 H1", "P1 H2", "P2 H1", "P2 H2", "Вес"]):
            put(f"{c}1", lab)
        put(f"{hc[0]}2", h2h(use_tour, use_team))
        fmt(f"{hc[0]}2:{hc[0]}", {"numberFormat": {"type": "DATE_TIME", "pattern": "dd.mm.yyyy hh:mm"}})
        fmt(f"{hc[7]}2:{hc[7]}", {"numberFormat": {"type": "NUMBER", "pattern": "0.000"}})
        fmt(f"{hc[0]}1:{hc[7]}1", {"textFormat": {"bold": True}})

        S = lambda c: f"${c}$2:${c}"
        J, M, N, O, P, Q = S(hc[0]), S(hc[3]), S(hc[4]), S(hc[5]), S(hc[6]), S(hc[7])
        cnt = f"${cs}${SR0}"

        # summary columns
        put(f"{cs}1", sum_title); merge(cs, 1, cd, 1)
        put(f"{cs}2", "Простое"); put(f"{cd}2", "Затухание")
        r = {k: SR0 + i for i, k in enumerate(
            ["n", "l1", "l2", "mo", "diff", "h1", "h2", "s1", "s2", "dfact", "dpois", "dnan"])}
        rows = {
            "n":    (f"=COUNT({J})", f"=IFERROR(SUM({Q})^2/SUMSQ({Q}),0)"),
            "l1":   (f'=IFERROR((SUM({M})+SUM({N}))/{cnt},"")',
                     f'=IFERROR(SUMPRODUCT({Q},{M}+{N})/SUM({Q}),"")'),
            "l2":   (f'=IFERROR((SUM({O})+SUM({P}))/{cnt},"")',
                     f'=IFERROR(SUMPRODUCT({Q},{O}+{P})/SUM({Q}),"")'),
            "mo":   (f'=IFERROR({cs}{r["l1"]}+{cs}{r["l2"]},"")',
                     f'=IFERROR({cd}{r["l1"]}+{cd}{r["l2"]},"")'),
            "diff": (f'=IFERROR({cs}{r["l1"]}-{cs}{r["l2"]},"")',
                     f'=IFERROR({cd}{r["l1"]}-{cd}{r["l2"]},"")'),
            "h1":   (f'=IFERROR((SUM({M})+SUM({O}))/(SUM({M})+SUM({N})+SUM({O})+SUM({P})),"")',
                     f'=IFERROR(SUMPRODUCT({Q},{M}+{O})/SUMPRODUCT({Q},{M}+{N}+{O}+{P}),"")'),
            "h2":   (f'=IFERROR(1-{cs}{r["h1"]},"")', f'=IFERROR(1-{cd}{r["h1"]},"")'),
            "s1":   (f'=IFERROR(SUM({M})/(SUM({M})+SUM({N})),"")',
                     f'=IFERROR(SUMPRODUCT({Q},{M})/SUMPRODUCT({Q},{M}+{N}),"")'),
            "s2":   (f'=IFERROR(SUM({O})/(SUM({O})+SUM({P})),"")',
                     f'=IFERROR(SUMPRODUCT({Q},{O})/SUMPRODUCT({Q},{O}+{P}),"")'),
            "dfact": (f'=IFERROR(SUMPRODUCT(ISNUMBER({J})*({M}+{N}={O}+{P}))/{cnt},"")',
                      f'=IFERROR(SUMPRODUCT({Q}*({M}+{N}={O}+{P}))/SUM({Q}),"")'),
            "dpois": tuple(
                f'=IFERROR(ARRAYFORMULA(SUMPRODUCT(POISSON.DIST(SEQUENCE(31,1,0),{c}{r["l1"]},FALSE),'
                f'POISSON.DIST(SEQUENCE(31,1,0),{c}{r["l2"]},FALSE))),"")' for c in (cs, cd)),
            "dnan": (f'=IFERROR({cs}{r["dfact"]}-{cs}{r["dpois"]},"")',
                     f'=IFERROR({cd}{r["dfact"]}-{cd}{r["dpois"]},"")'),
        }
        for k, (fs, fd) in rows.items():
            if k != "n":
                fs, fd = guard(fs, cnt), guard(fd, cnt)
            put(f"{cs}{r[k]}", fs)
            put(f"{cd}{r[k]}", fd)
        fmt(f"{cs}{SR0}:{cd}{r['diff']}", {"numberFormat": {"type": "NUMBER", "pattern": "0.00"}})
        fmt(f"{cs}{r['h1']}:{cd}{r['dnan']}", {"numberFormat": {"type": "PERCENT", "pattern": "0.0%"}})
        fmt(f"{cs}{r['mo']}:{cd}{r['diff']}", {"backgroundColor": GOLD, "textFormat": {"bold": True}})

        # ---- table ----
        diff_c, mo_c = f"${cd}${r['diff']}", f"${cd}${r['mo']}"
        l1, l2 = f"${cd}${r['l1']}", f"${cd}${r['l2']}"
        c = [colstr(colnum(c0) + i) for i in range(5)]
        D = f"(({M}+{N})-({O}+{P}))"
        T = f"({M}+{N}+{O}+{P})"
        ok = f"ISNUMBER({J})"

        def cntf(expr):
            return guard(f"=SUMPRODUCT({ok}*({expr}))", cnt)

        # header rows 1-3
        if header == "inputs":
            put(f"{c[0]}1", "*")                       # tournament dropdown
            put(f"{c[0]}2", "*"); put(f"{c[3]}2", "*")  # team dropdowns
            put(f"{c[0]}3", ""); put(f"{c[3]}3", "")    # player dropdowns
        else:
            put(f"{c[0]}1", f'={TOUR}&" ("&{PL1}&" - "&{PL2}&")"')
            put(f"{c[0]}2", f"={T1}"); put(f"{c[3]}2", f"={T2}")
            put(f"{c[0]}3", f"={PL1}"); put(f"{c[3]}3", f"={PL2}")
        put(f"{c[2]}3", "X")
        merge(c[0], 1, c[4], 1)
        for rr in (2, 3, 4, 5):
            merge(c[0], rr, c[1], rr); merge(c[3], rr, c[4], rr)

        # W/D/L
        for cc, cond, mcond in ((c[0], f"{D}>0", "i>j"), (c[2], f"{D}=0", "i=j"), (c[3], f"{D}<0", "i<j")):
            put(f"{cc}4", cntf(cond))
            put(f"{cc}5", mx(mcond, l1, l2, cnt))

        # handicap rows 6-11
        hr = 6
        put(f"{c[0]}{hr}", "Ф1"); put(f"{c[4]}{hr}", "Ф2")
        put(f"{c[1]}{hr}", guard(f'="Разница   "&TEXT({diff_c},"0.00")', cnt))
        merge(c[1], hr, c[3], hr)
        for i, h in enumerate((-1.5, -1.0, 0, 1.0, 1.5)):
            rr = hr + 1 + i
            L = f"${c[2]}${rr}"
            put(f"{c[2]}{rr}", h)
            put(f"{c[0]}{rr}", cntf(f"{D}+{L}>0"))
            put(f"{c[1]}{rr}", mxa(f"i-j+{L}>0", f"i-j+{L}=0", l1, l2, cnt))
            put(f"{c[3]}{rr}", mxa(f"i-j+{L}<0", f"i-j+{L}=0", l1, l2, cnt))
            put(f"{c[4]}{rr}", cntf(f"{D}+{L}<0"))

        # total rows 12-17
        tr0 = 12
        put(f"{c[0]}{tr0}", ">"); put(f"{c[4]}{tr0}", "<")
        put(f"{c[1]}{tr0}", guard(f'="Тотал   "&TEXT({mo_c},"0.00")', cnt))
        merge(c[1], tr0, c[3], tr0)
        mid = tr0 + 3
        lam = f"({l1}+{l2})"
        for i in range(5):
            rr = tr0 + 1 + i
            L = f"${c[2]}${rr}"
            if rr == mid:
                put(f"{c[2]}{rr}", guard(f"=MAX(2.5,FLOOR({mo_c})+0.5)", cnt))
            else:
                put(f"{c[2]}{rr}", guard(f"={c[2]}{mid}{rr - mid:+d}", cnt))
            push = f"IF({L}=INT({L}),POISSON.DIST({L},{lam},FALSE),0)"
            put(f"{c[0]}{rr}", cntf(f"{T}>{L}"))
            put(f"{c[1]}{rr}", guard(f'=IFERROR((1-{push})/(1-POISSON.DIST(FLOOR({L}),{lam},TRUE))/(1+{MF}),"")', cnt))
            put(f"{c[3]}{rr}", guard(f'=IFERROR((1-{push})/POISSON.DIST(CEILING({L})-1,{lam},TRUE)/(1+{MF}),"")', cnt))
            put(f"{c[4]}{rr}", cntf(f"{T}<{L}"))

        rng = lambda c1, r1, c2, r2: f"{c1}{r1}:{c2}{r2}"
        fmt(rng(c[0], 1, c[4], 17), {"horizontalAlignment": "CENTER"})
        fmt(rng(c[0], 1, c[4], 1), {"backgroundColor": GREY, "textFormat": {"bold": True}})
        fmt(rng(c[0], 2, c[4], 3), {"backgroundColor": BLUE, "textFormat": {"bold": True}})
        fmt(rng(c[0], 4, c[4], 4), {"backgroundColor": PINK, "numberFormat": {"type": "NUMBER", "pattern": "0"}})
        fmt(rng(c[0], 5, c[4], 5), {"backgroundColor": GREY, "textFormat": {"bold": True},
                                    "numberFormat": {"type": "NUMBER", "pattern": "0.00"}})
        fmt(rng(c[0], hr, c[4], hr), {"backgroundColor": BLUE, "textFormat": {"bold": True}})
        fmt(rng(c[0], tr0, c[4], tr0), {"backgroundColor": BLUE, "textFormat": {"bold": True}})
        for a, b in ((hr + 1, hr + 5), (tr0 + 1, tr0 + 5)):
            fmt(rng(c[0], a, c[0], b), {"backgroundColor": PINK, "numberFormat": {"type": "NUMBER", "pattern": "0"}})
            fmt(rng(c[4], a, c[4], b), {"backgroundColor": PINK, "numberFormat": {"type": "NUMBER", "pattern": "0"}})
            fmt(rng(c[1], a, c[1], b), {"backgroundColor": GREY, "numberFormat": {"type": "NUMBER", "pattern": "0.00"}})
            fmt(rng(c[3], a, c[3], b), {"backgroundColor": GREY, "numberFormat": {"type": "NUMBER", "pattern": "0.00"}})
            fmt(rng(c[2], a, c[2], b), {"backgroundColor": BLUE})
        fmt(rng(c[2], hr + 1, c[2], hr + 5), {"numberFormat": {"type": "NUMBER", "pattern": "+0.0;-0.0;0"}})
        fmt(rng(c[2], tr0 + 1, c[2], tr0 + 5), {"numberFormat": {"type": "NUMBER", "pattern": "0.0"}})
        fmt(rng(c[0], hr + 3, c[4], hr + 3), {"textFormat": {"bold": True}})
        fmt(rng(c[2], hr + 3, c[2], hr + 3), {"backgroundColor": MIDC})
        fmt(rng(c[0], mid, c[4], mid), {"textFormat": {"bold": True}})
        fmt(rng(c[2], mid, c[2], mid), {"backgroundColor": MIDC})

    # left: all tournaments (chips = inputs), right: selected tournament + teams
    block("Q", "R", "ВСЕ ТУРНИРЫ", "V", False, False, "A", "inputs")
    block("S", "T", f'="ТУРНИР: "&{TOUR}', "AE", True, True, "G", "tour")

    # ---- dropdown lists AN:AR (from PAIRS: player, opp, tournament, team, opp_team) ----
    PA, PB, PC, PD, PE = (f"PAIRS!{x}2:{x}" for x in "ABCDE")
    put("AN1", "Турниры")
    put("AN2", f'=IFERROR(VSTACK("*",SORT(UNIQUE(FILTER({PC},{PA}={PL1},'
               f'(({PL2}="")+({PB}={PL2}))>0)))),"*")')
    put("AO1", "Игроки")
    put("AO2", f'=IFERROR(SORT(UNIQUE(FILTER({PA},{PA}<>""))),"")')
    put("AP1", "Соперники")
    put("AP2", f'=IFERROR(SORT(UNIQUE(FILTER({PB},{PA}={PL1}))),"")')
    for col, src in (("AQ", PD), ("AR", PE)):
        put(f"{col}1", "Команды P1" if col == "AQ" else "Команды P2")
        put(f"{col}2", f'=IFERROR(VSTACK("*",SORT(UNIQUE(FILTER({src},{PA}={PL1},{PB}={PL2},'
                       f'(({TOUR}="*")+({PC}={TOUR}))>0)))),"*")')
    fmt("AN1:AR1", {"textFormat": {"bold": True}})

    ws.batch_update(cells, value_input_option="USER_ENTERED")
    ws.batch_format(fmts)

    def dv(a1, src):
        cc = "".join(ch for ch in a1 if ch.isalpha())
        rr = int("".join(ch for ch in a1 if ch.isdigit()))
        return {"setDataValidation": {
            "range": grid(cc, rr, cc, rr),
            "rule": {"condition": {"type": "ONE_OF_RANGE",
                                   "values": [{"userEnteredValue": src}]},
                     "showCustomUi": True, "strict": False}}}

    widths = [{"updateDimensionProperties": {
        "range": {"sheetId": ws.id, "dimension": "COLUMNS",
                  "startIndex": colnum(a) - 1, "endIndex": colnum(b)},
        "properties": {"pixelSize": px}, "fields": "pixelSize"}}
        for a, b, px in (("A", "K", 85), ("F", "F", 20), ("L", "L", 20), ("O", "O", 20),
                         ("M", "N", 110), ("P", "P", 150), ("Q", "T", 85))]

    sh.batch_update({"requests": reqs + widths + [
        dv("A1", f"={CALC_NAME}!$AN$2:$AN"),
        dv("A2", f"={CALC_NAME}!$AQ$2:$AQ"),
        dv("D2", f"={CALC_NAME}!$AR$2:$AR"),
        dv("A3", f"={CALC_NAME}!$AO$2:$AO"),
        dv("D3", f"={CALC_NAME}!$AP$2:$AP"),
    ]})
    print("CALC built")


if CALC_NAME not in [w.title for w in sh.worksheets()]:
    build_calc(sh)
else:
    print("CALC exists, skipped")
