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
    tour = '(($B$1="*")+(INDEX(db,,3)=$B$1))>0,' if use_tour else ''
    team = ('g,FILTER(r,(($B$3="*")+(t1=$B$3))>0,(($B$5="*")+(t2=$B$5))>0),'
            if use_team else 'g,r,')
    return (
        '=IFERROR(ARRAYFORMULA(LET('
        'db,DB_Corners!A2:Q,'
        'p1,LOWER(TRIM($B$2)),p2,LOWER(TRIM($B$4)),'
        'k,INDEX(db,,17),'
        'f,FILTER(db,((k=p1&"|"&p2)+(k=p2&"|"&p1))>0,' + tour +
        'INDEX(db,,2)>=$B$6,INDEX(db,,2)<$B$7+1),'
        'a,LOWER(INDEX(f,,5))=p1,'
        'd,INDEX(f,,2),'
        't1,IF(a,INDEX(f,,4),INDEX(f,,6)),'
        't2,IF(a,INDEX(f,,6),INDEX(f,,4)),'
        'r,HSTACK(d,t1,t2,'
        'IF(a,INDEX(f,,8),INDEX(f,,10)),IF(a,INDEX(f,,9),INDEX(f,,11)),'
        'IF(a,INDEX(f,,10),INDEX(f,,8)),IF(a,INDEX(f,,11),INDEX(f,,9)),'
        'EXP(-LN(2)*($B$7+1-d)/$B$8)),'
        + team +
        'SORT(g,1,FALSE))),"нет H2H")'
    )


def guard(f, cnt):
    return f'=IF({cnt}=0,"",' + f[1:] + ')'


def mxa(win, push, l1, l2, cnt):
    return guard('=IFERROR(ARRAYFORMULA(LET(l1,' + l1 + ',l2,' + l2 + ','
                 'i,MAKEARRAY(31,31,LAMBDA(r,c,r-1)),j,MAKEARRAY(31,31,LAMBDA(r,c,c-1)),'
                 'm,POISSON.DIST(i,l1,FALSE)*POISSON.DIST(j,l2,FALSE),'
                 '(1-SUM(m*(' + push + ')))/SUM(m*(' + win + ')))),"")', cnt)


def mx(cond, l1, l2, cnt):
    return guard('=IFERROR(ARRAYFORMULA(LET(l1,' + l1 + ',l2,' + l2 + ','
                 'i,MAKEARRAY(31,31,LAMBDA(r,c,r-1)),j,MAKEARRAY(31,31,LAMBDA(r,c,c-1)),'
                 'm,POISSON.DIST(i,l1,FALSE)*POISSON.DIST(j,l2,FALSE),'
                 '1/SUM(m*(' + cond + ')))),"")', cnt)


R0 = 12          # first results row
MKT_ROW = 26     # first table row


def build_calc(sh):
    ws = sh.add_worksheet(CALC_NAME, rows=200, cols=40, index=0)
    cells, fmts, reqs = [], [], []

    def grid(c1, r1, c2, r2):
        return {"sheetId": ws.id, "startRowIndex": r1 - 1, "endRowIndex": r2,
                "startColumnIndex": colnum(c1) - 1, "endColumnIndex": colnum(c2)}

    def put(a1, v):
        cells.append({"range": a1, "values": [[v]]})

    # ---- inputs ----
    for r, (lab, val) in enumerate([
        ("Турнир (блок 2)", "*"), ("Игрок 1", ""), ("Команда 1 (блок 2)", "*"),
        ("Игрок 2", ""), ("Команда 2 (блок 2)", "*"),
        ("Дата от", "=TODAY()-60"), ("Дата до", "=TODAY()"),
        ("Полураспад, дн", 30)], start=1):
        put(f"A{r}", lab)
        put(f"B{r}", val)

    labels = ["H2H матчей / эфф.", "λ1 (Игрок 1)", "λ2 (Игрок 2)", "М.О.Угл",
              "Разница (λ1−λ2)", "Часть H1", "Часть H2", "H1 доля Игрок 1",
              "H1 доля Игрок 2", "% ничьих факт", "% ничьих Пуассон", "ДнаН подсказка"]
    for i, lab in enumerate(labels):
        put(f"A{R0 + i}", lab)

    def block(title, title_f, cs, cd, helper0, use_tour, use_team, mkt_col0):
        # helper columns
        hc = [colstr(colnum(helper0) + i) for i in range(8)]
        for c, lab in zip(hc, ["Дата", "Команда P1", "Команда P2",
                               "P1 H1", "P1 H2", "P2 H1", "P2 H2", "Вес"]):
            put(f"{c}1", lab)
        put(f"{hc[0]}2", h2h(use_tour, use_team))
        fmts.append({"range": f"{hc[0]}2:{hc[0]}",
                     "format": {"numberFormat": {"type": "DATE_TIME", "pattern": "dd.mm.yyyy hh:mm"}}})
        fmts.append({"range": f"{hc[7]}2:{hc[7]}",
                     "format": {"numberFormat": {"type": "NUMBER", "pattern": "0.000"}}})
        fmts.append({"range": f"{hc[0]}1:{hc[7]}1", "format": {"textFormat": {"bold": True}}})

        S = lambda c: f"${c}$2:${c}"
        J, M, N, O, P, Q = S(hc[0]), S(hc[3]), S(hc[4]), S(hc[5]), S(hc[6]), S(hc[7])
        cnt = f"${cs}${R0}"

        put(f"{cs}{R0 - 2}", title)
        put(f"{cs}{R0 - 1}", "Простое")
        put(f"{cd}{R0 - 1}", "Затухание")

        r = {k: R0 + i for i, k in enumerate(
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

        # ---- table 1:1 (W/D/L, handicap nominals, total nominals) ----
        diff_c, mo_c = f"${cd}${r['diff']}", f"${cd}${r['mo']}"
        l1, l2 = f"${cd}${r['l1']}", f"${cd}${r['l2']}"
        c = [colstr(colnum(mkt_col0) + i) for i in range(5)]
        m0 = MKT_ROW
        D = f"(({M}+{N})-({O}+{P}))"
        T = f"({M}+{N}+{O}+{P})"
        ok = f"ISNUMBER({J})"

        def cntf(expr):
            return guard(f"=SUMPRODUCT({ok}*({expr}))", cnt)

        def merge(c1, r1, c2, r2):
            reqs.append({"mergeCells": {"mergeType": "MERGE_ALL", "range": grid(c1, r1, c2, r2)}})

        # title
        put(f"{c[0]}{m0}", title_f)
        merge(c[0], m0, c[4], m0)
        # names
        put(f"{c[0]}{m0 + 1}", "=$B$2"); merge(c[0], m0 + 1, c[1], m0 + 1)
        put(f"{c[2]}{m0 + 1}", "X")
        put(f"{c[3]}{m0 + 1}", "=$B$4"); merge(c[3], m0 + 1, c[4], m0 + 1)
        # W/D/L counts + odds
        for cc, cond, mcond in ((c[0], f"{D}>0", "i>j"), (c[2], f"{D}=0", "i=j"), (c[3], f"{D}<0", "i<j")):
            put(f"{cc}{m0 + 2}", cntf(cond))
            put(f"{cc}{m0 + 3}", mx(mcond, l1, l2, cnt))
        for rr in (m0 + 2, m0 + 3):
            merge(c[0], rr, c[1], rr); merge(c[3], rr, c[4], rr)

        # handicap
        hr = m0 + 4
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

        # total
        tr0 = m0 + 10
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
            put(f"{c[1]}{rr}", guard(f'=IFERROR((1-{push})/(1-POISSON.DIST(FLOOR({L}),{lam},TRUE)),"")', cnt))
            put(f"{c[3]}{rr}", guard(f'=IFERROR((1-{push})/POISSON.DIST(CEILING({L})-1,{lam},TRUE),"")', cnt))
            put(f"{c[4]}{rr}", cntf(f"{T}<{L}"))

        BLUE = {"red": 0.79, "green": 0.85, "blue": 0.97}
        PINK = {"red": 0.96, "green": 0.80, "blue": 0.80}
        GREY = {"red": 0.95, "green": 0.95, "blue": 0.95}
        MIDC = {"red": 0.86, "green": 0.82, "blue": 0.93}
        rng = lambda c1, r1, c2, r2: f"{c1}{r1}:{c2}{r2}"
        fmts.extend([
            {"range": f"{cs}{R0}:{cd}{r['diff']}",
             "format": {"numberFormat": {"type": "NUMBER", "pattern": "0.00"}}},
            {"range": f"{cs}{r['h1']}:{cd}{r['dnan']}",
             "format": {"numberFormat": {"type": "PERCENT", "pattern": "0.0%"}}},
            {"range": f"{cs}{r['mo']}:{cd}{r['diff']}",
             "format": {"backgroundColor": {"red": 1, "green": 0.85, "blue": 0.4},
                        "textFormat": {"bold": True}}},
            {"range": f"{cs}{R0 - 2}:{cd}{R0 - 1}", "format": {"textFormat": {"bold": True}}},
            # table
            {"range": rng(c[0], m0, c[4], m0 + 15),
             "format": {"horizontalAlignment": "CENTER"}},
            {"range": rng(c[0], m0, c[4], m0), "format": {"backgroundColor": GREY, "textFormat": {"bold": True}}},
            {"range": rng(c[0], m0 + 1, c[4], m0 + 1), "format": {"backgroundColor": BLUE, "textFormat": {"bold": True}}},
            {"range": rng(c[0], m0 + 2, c[4], m0 + 2),
             "format": {"backgroundColor": PINK, "numberFormat": {"type": "NUMBER", "pattern": "0"}}},
            {"range": rng(c[0], m0 + 3, c[4], m0 + 3),
             "format": {"backgroundColor": GREY, "textFormat": {"bold": True},
                        "numberFormat": {"type": "NUMBER", "pattern": "0.00"}}},
            {"range": rng(c[0], hr, c[4], hr), "format": {"backgroundColor": BLUE, "textFormat": {"bold": True}}},
            {"range": rng(c[0], tr0, c[4], tr0), "format": {"backgroundColor": BLUE, "textFormat": {"bold": True}}},
        ])
        for a, b in ((hr + 1, hr + 5), (tr0 + 1, tr0 + 5)):
            fmts.extend([
                {"range": rng(c[0], a, c[0], b),
                 "format": {"backgroundColor": PINK, "numberFormat": {"type": "NUMBER", "pattern": "0"}}},
                {"range": rng(c[4], a, c[4], b),
                 "format": {"backgroundColor": PINK, "numberFormat": {"type": "NUMBER", "pattern": "0"}}},
                {"range": rng(c[1], a, c[1], b),
                 "format": {"backgroundColor": GREY, "numberFormat": {"type": "NUMBER", "pattern": "0.00"}}},
                {"range": rng(c[3], a, c[3], b),
                 "format": {"backgroundColor": GREY, "numberFormat": {"type": "NUMBER", "pattern": "0.00"}}},
                {"range": rng(c[2], a, c[2], b), "format": {"backgroundColor": BLUE}},
            ])
        fmts.extend([
            {"range": rng(c[2], hr + 1, c[2], hr + 5),
             "format": {"numberFormat": {"type": "NUMBER", "pattern": "+0.0;-0.0;0"}}},
            {"range": rng(c[2], tr0 + 1, c[2], tr0 + 5),
             "format": {"numberFormat": {"type": "NUMBER", "pattern": "0.0"}}},
            {"range": rng(c[0], hr + 3, c[4], hr + 3), "format": {"textFormat": {"bold": True}}},
            {"range": rng(c[2], hr + 3, c[2], hr + 3), "format": {"backgroundColor": MIDC}},
            {"range": rng(c[0], mid, c[4], mid), "format": {"textFormat": {"bold": True}}},
            {"range": rng(c[2], mid, c[2], mid), "format": {"backgroundColor": MIDC}},
        ])

    block("ВСЕ ТУРНИРЫ", '="* ВСЕ ТУРНИРЫ ("&$B$2&" - "&$B$4&")"',
          "B", "C", "M", use_tour=False, use_team=False, mkt_col0="A")
    block("ТУРНИР", '=$B$1&" ("&$B$2&" - "&$B$4&")"',
          "D", "E", "V", use_tour=True, use_team=True, mkt_col0="G")
    put(f"D{R0 - 2}", '="ТУРНИР: "&$B$1')

    # ---- dropdown lists ----
    put("AE1", "Турниры")
    put("AE2", '=VSTACK("*",SORT(UNIQUE(FILTER(DB_Corners!C2:C,DB_Corners!C2:C<>""))))')
    put("AF1", "Игроки")
    put("AF2", '=IFERROR(SORT(UNIQUE(FILTER(VSTACK(DB_Corners!E2:E,DB_Corners!G2:G),'
               'VSTACK(DB_Corners!E2:E,DB_Corners!G2:G)<>""))),"")')
    put("AG1", "Соперники")
    put("AG2", '=IFERROR(ARRAYFORMULA(LET(p1,LOWER(TRIM($B$2)),'
               'opp,VSTACK(DB_Corners!G2:G,DB_Corners!E2:E),'
               'me,VSTACK(LOWER(DB_Corners!E2:E),LOWER(DB_Corners!G2:G)),'
               'SORT(UNIQUE(FILTER(opp,me=p1))))),"")')
    for col, mine in (("AH", True), ("AI", False)):
        tm = ("IF(a,DB_Corners!D2:D,DB_Corners!F2:F)" if mine
              else "IF(a,DB_Corners!F2:F,DB_Corners!D2:D)")
        put(f"{col}1", "Команды P1" if mine else "Команды P2")
        put(f"{col}2", '=IFERROR(ARRAYFORMULA(LET(t,$B$1,'
                       'p1,LOWER(TRIM($B$2)),p2,LOWER(TRIM($B$4)),'
                       'k,DB_Corners!Q2:Q,tr,DB_Corners!C2:C,'
                       'a,LOWER(DB_Corners!E2:E)=p1,'
                       'tm,' + tm + ','
                       'VSTACK("*",SORT(UNIQUE(FILTER(tm,((k=p1&"|"&p2)+(k=p2&"|"&p1))>0,'
                       '((t="*")+(tr=t))>0)))))),"*")')

    ws.batch_update(cells, value_input_option="USER_ENTERED")

    fmts += [
        {"range": "B6:B7", "format": {"numberFormat": {"type": "DATE", "pattern": "dd.mm.yyyy"}}},
        {"range": f"A1:A{R0 + 11}", "format": {"textFormat": {"bold": True}}},
        {"range": "AE1:AI1", "format": {"textFormat": {"bold": True}}},
    ]
    ws.batch_format(fmts)

    def dv(row, col, src):
        return {"setDataValidation": {
            "range": {"sheetId": ws.id, "startRowIndex": row - 1, "endRowIndex": row,
                      "startColumnIndex": col - 1, "endColumnIndex": col},
            "rule": {"condition": {"type": "ONE_OF_RANGE",
                                   "values": [{"userEnteredValue": src}]},
                     "showCustomUi": True, "strict": False}}}

    sh.batch_update({"requests": reqs + [
        dv(1, 2, f"={CALC_NAME}!$AE$2:$AE"),
        dv(2, 2, f"={CALC_NAME}!$AF$2:$AF"),
        dv(3, 2, f"={CALC_NAME}!$AH$2:$AH"),
        dv(4, 2, f"={CALC_NAME}!$AG$2:$AG"),
        dv(5, 2, f"={CALC_NAME}!$AI$2:$AI"),
    ]})
    print("CALC built")


if CALC_NAME not in [w.title for w in sh.worksheets()]:
    build_calc(sh)
else:
    print("CALC exists, skipped")
