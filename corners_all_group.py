"""Corners calculator 1:1 as in "2x4 ECF" Console 1 (corners block), but for ALL players from the parquets.
Builds a NEW tab ECF_ALL_vN (+ helper tab ECF_ALL_vN_model) in Corners_H2H. Existing tabs are never touched.
Data: DB_Corners + PAIRS of Corners_H2H (refreshed daily by corners-to-sheet).
Settings: copied once from ECF "Console 1"!AH104:AI125 (calibrated values) if GSHEET_ID_CONSOLE is set,
otherwise defaults. After the build the settings live in the new tab (AH5:AI26) and can be edited there."""
import os, re, json, time

import gspread
from google.oauth2.service_account import Credentials

SA_INFO = json.loads(os.environ["GDRIVE_SA_JSON"])
SHEET_ID = re.search(r"[-\w]{25,}", os.environ["GSHEET_ID"]).group(0)
ECF_RAW = os.environ.get("GSHEET_ID_CONSOLE", "").strip()
PREFIX = "ECF_ALL"

creds = Credentials.from_service_account_info(SA_INFO, scopes=["https://www.googleapis.com/auth/spreadsheets"])
gc = gspread.authorize(creds)
book = gc.open_by_key(SHEET_ID)

MODEL_DAYS = 100
LBL_STEP = 7
R0 = 4                                   # table row r -> sheet row R0 + r (tables A5:K21)
SET_OFF = -4                             # settings layout AH9:AI30 (Console 1) -> AH5:AI26
ECF_SET_OFF = 95                         # calibrated settings in ECF Console 1: AH104:AI125

DB = "DB_Corners"
dbc = lambda c: f"{DB}!${c}$2:${c}"
D_DT, D_TR, D_T1, D_P1, D_T2, D_P2 = dbc("B"), dbc("C"), dbc("D"), dbc("E"), dbc("F"), dbc("G")
D_C1, D_C2, D_TOT, D_KEY, D_DIFF, D_H1, D_H2 = dbc("N"), dbc("O"), dbc("P"), dbc("Q"), dbc("R"), dbc("S"), dbc("T")

BLUE = {"red": 0.79, "green": 0.85, "blue": 0.97}
PINK = {"red": 0.96, "green": 0.80, "blue": 0.80}
GREY = {"red": 0.95, "green": 0.95, "blue": 0.95}
MIDC = {"red": 0.86, "green": 0.82, "blue": 0.93}
GOLD = {"red": 1, "green": 0.85, "blue": 0.4}
LGOLD = {"red": 1, "green": 0.95, "blue": 0.8}
GREEN = {"red": 0.72, "green": 0.88, "blue": 0.75}

# settings panel (Console 1 layout AH9:AI30): label cells, input cells
SET_LABELS = (("AH9", "Строка дней"), ("AI9", "Байес (a)"), ("AH11", "Линейность"), ("AI11", "Маржа 2"),
              ("AI13", "Маржа 3"), ("AH16", "Poisson_Res"), ("AI16", "Затухание"), ("AH18", "Tour_Res"),
              ("AI18", "База дней"), ("AH20", "MAX Влияние"), ("AI20", "Турнир"), ("AH22", "Влияние"),
              ("AI22", "Команды"), ("AH24", "MAX Влияние"), ("AI24", "И+К"), ("AH27", "Update Date"),
              ("AI27", "МО+"), ("AH29", "Critical Date"), ("AI29", "Old"))
SET_DEFAULT = {"AH10": 1, "AI10": 0.40, "AH12": 0.18, "AI12": 0.075, "AH13": 0.06, "AI14": 0.09,
               "AH17": 0.10, "AI17": 1.5, "AH19": 0.0, "AI19": 7, "AH21": 0.50, "AI21": 15, "AH23": 0.10,
               "AI23": 200, "AH25": 0.60, "AI25": 20, "AH28": "=DATE(2025,12,4)", "AI28": 1.05,
               "AH30": "=DATE(2025,11,10)", "AI30": 0.50}


def retry(fn, *a, **k):
    for att in range(6):
        try:
            return fn(*a, **k)
        except gspread.exceptions.APIError as e:
            if att == 5:
                raise
            print(f"retry {att + 1}/6: {str(e)[:120]}")
            time.sleep(3 * 2 ** att)


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


def cs(start, n):
    return [colstr(colnum(start) + k) for k in range(n)]


def shift(a1, off):
    m = re.fullmatch(r"([A-Z]+)(\d+)", a1)
    return f"{m.group(1)}{int(m.group(2)) + off}"


def pois(cond, la, lb):
    return ('IFERROR(ARRAYFORMULA(LET(la,' + la + ',lb,' + lb + ','
            'i,MAKEARRAY(31,31,LAMBDA(r,c,r-1)),j,MAKEARRAY(31,31,LAMBDA(r,c,c-1)),'
            'm,POISSON.DIST(i,la,FALSE)*POISSON.DIST(j,lb,FALSE),SUM(m*(' + cond + ')))),0)')


def read_ecf_settings():
    """Calibrated corner settings from ECF Console 1 (AH104:AI125). Formulas with cell refs are skipped."""
    if not ECF_RAW:
        print("settings: GSHEET_ID_CONSOLE not set -> defaults")
        return {}
    try:
        ecf = gc.open_by_key(re.search(r"[-\w]{25,}", ECF_RAW).group(0))
        vals = retry(ecf.worksheet("Console 1").get, f"AH{9 + ECF_SET_OFF}:AI{30 + ECF_SET_OFF}",
                     value_render_option="FORMULA")
    except Exception as e:
        print(f"settings: ECF not readable ({str(e)[:100]}) -> defaults")
        return {}
    got = {}
    for i, row in enumerate(vals):
        for j, v in enumerate(row):
            a1 = f"{'AH' if j == 0 else 'AI'}{9 + i}"
            if a1 not in SET_DEFAULT or v in ("", None):
                continue
            if isinstance(v, str) and v.startswith("=") and re.search(r"\$?[A-Z]{1,3}\$?\d", v[1:].replace("DATE(", "")):
                continue
            got[a1] = v
    print(f"settings: {len(got)} of {len(SET_DEFAULT)} copied from ECF Console 1: {got}")
    return got


def build(T, H, settings):
    q = lambda sheet, a: f"'{sheet}'!{a}"
    cabs = lambda a: re.sub(r"([A-Z]+)(\d+)", r"$\1$\2", a)
    sset = lambda a: q(T, cabs(shift(a, SET_OFF)))
    S = {k: sset(v) for k, v in dict(
        BAY="AI10", LIN1="AH12", MAR2="AI12", LIN2="AH13", DSTAR="AH14", MAR3="AI14", PRES="AH17", KDEC="AI17",
        TRES="AH19", BASE="AI19", MAXT="AH21", NTOUR="AI21", INFL="AH23", NTEAM="AI23", MAXPT="AH25", NPT="AI25",
        UPD="AH28", MOPLUS="AI28", CRIT="AH30", OLD="AI30").items()}
    XIN = dict(TOUR="$AU$1", T1="$AU$2", T2="$AU$3", P1="$AU$4", P2="$AU$5", DFROM="$AU$6", DTO="$AU$7",
               GWIN="$AU$8")
    CH = "AU"

    def odds(p, n, mcell):
        return f"(1-{mcell})/((1-{S['BAY']}*{mcell})*({p})+{S['BAY']}*{mcell}/{n})"

    def weights(dd):
        i = XIN
        return (f'dn,INT({i["DTO"]})-INT({dd}),'
                f'am,IF(dn<{S["BASE"]},0,IF((dn-{S["BASE"]})*{S["KDEC"]}/100>1,1,(dn-{S["BASE"]})*{S["KDEC"]}/100)),'
                f'qq,(1-am)*(1-am),'
                f'prc,IF(qq<{S["LIN1"]},({MODEL_DAYS - 1}-dn)/({MODEL_DAYS - 1}-{S["DSTAR"]})*({S["LIN1"]}-{S["LIN2"]})+{S["LIN2"]},qq),'
                f'ww,IF(INT({dd})<{S["CRIT"]},prc*{S["OLD"]},prc),'
                f'wm,IF(INT({dd})<{S["UPD"]},ww*{S["MOPLUS"]},ww),')

    def h2h_table(use_tour):
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

    def h2h_model():
        i = XIN
        return ('=IFERROR(ARRAYFORMULA(LET('
                f'db,{DB}!A2:Q,pa,LOWER(TRIM({i["P1"]})),pb,LOWER(TRIM({i["P2"]})),kk,INDEX(db,,17),dt,INDEX(db,,2),'
                'f,FILTER(db,((kk=pa&"|"&pb)+(kk=pb&"|"&pa))>0,'
                f'INT(dt)>INT({i["DTO"]})-{MODEL_DAYS},INT(dt)<=INT({i["DTO"]})),'
                'a,LOWER(INDEX(f,,5))=pa,dd,INDEX(f,,2),'
                'xa,INDEX(f,,8)+INDEX(f,,9),xb,INDEX(f,,10)+INDEX(f,,11),'
                + weights("dd") +
                'HSTACK(dd,INDEX(f,,3),IF(a,xa,xb),IF(a,xb,xa),ww,wm))),"")')

    def tour_model():
        i = XIN
        return ('=IFERROR(ARRAYFORMULA(LET('
                f'db,{DB}!A2:Q,dt,INDEX(db,,2),tr,INDEX(db,,3),'
                f'f,FILTER(db,(({i["TOUR"]}="*")+(tr={i["TOUR"]}))>0,'
                f'INT(dt)>INT({i["DTO"]})-{MODEL_DAYS},INT(dt)<=INT({i["DTO"]})),'
                'dd,INDEX(f,,2),'
                + weights("dd") +
                'HSTACK(dd,INDEX(f,,5),INDEX(f,,4),INDEX(f,,7),INDEX(f,,6),'
                'INDEX(f,,8)+INDEX(f,,9),INDEX(f,,10)+INDEX(f,,11),ww,wm))),"")')

    ws = retry(book.add_worksheet, T, rows=200, cols=40, index=0)
    hs = retry(book.add_worksheet, H, rows=200, cols=65)
    hcells, ccells, hf, cf, reqs = [], [], [], [], []

    def hput(a, v): hcells.append({"range": a, "values": [[v]]})
    def cput(a, v): ccells.append({"range": a, "values": [[v]]})
    def cfmt(rng, f): cf.append({"range": rng, "format": f})

    def grid(sid, c1_, r1, c2_, r2):
        return {"sheetId": sid, "startRowIndex": r1 - 1, "endRowIndex": r2,
                "startColumnIndex": colnum(c1_) - 1, "endColumnIndex": colnum(c2_)}

    def merge(c1_, r1, c2_, r2):
        reqs.append({"mergeCells": {"mergeType": "MERGE_ALL", "range": grid(ws.id, c1_, r1, c2_, r2)}})

    HR_ = lambda c: f"${c}$2:${c}"
    HQ = lambda c: q(H, f"${c}$2:${c}")
    CHQ = lambda n: q(H, f"${CH}${n}")

    # ================= chips (rows 1-4) =================
    cput("A1", "*"); cput("A2", "*"); cput("D2", "*"); cput("C3", "X")
    merge("A1", 1, "E1", 1); merge("A2", 2, "B2", 2); merge("D2", 2, "E2", 2)
    merge("A3", 3, "B3", 3); merge("D3", 3, "E3", 3)
    for a1, v in (("M1", "Выборка дней"), ("N1", "От"), ("M2", 30), ("N2", "=$N$4-$M$2"),
                  ("M3", "Для графика"), ("N3", "До"), ("M4", 30), ("N4", "=TODAY()")):
        cput(a1, v)
    cfmt("A1:E3", {"backgroundColor": BLUE, "textFormat": {"bold": True}, "horizontalAlignment": "CENTER"})
    cfmt("M1:N1", {"backgroundColor": BLUE, "textFormat": {"bold": True}, "horizontalAlignment": "CENTER"})
    cfmt("M3:N3", {"backgroundColor": BLUE, "textFormat": {"bold": True}, "horizontalAlignment": "CENTER"})
    cfmt("M2:N2", {"horizontalAlignment": "CENTER"}); cfmt("M4:N4", {"horizontalAlignment": "CENTER"})
    cfmt("M2", {"backgroundColor": LGOLD}); cfmt("M4", {"backgroundColor": LGOLD})
    cfmt("N2", {"backgroundColor": PINK, "numberFormat": {"type": "DATE", "pattern": "dd.mm.yyyy"}})
    cfmt("N4", {"backgroundColor": PINK, "numberFormat": {"type": "DATE", "pattern": "dd.mm.yyyy"}})

    # ================= helper: inputs mirror =================
    for r_, lab, f_ in ((1, "Турнир", f'={q(T, "$A$1")}'), (2, "Команда 1", f'={q(T, "$A$2")}'),
                        (3, "Команда 2", f'={q(T, "$D$2")}'), (4, "Игрок 1", f'={q(T, "$A$3")}'),
                        (5, "Игрок 2", f'={q(T, "$D$3")}'), (6, "От", f'={q(T, "$N$2")}'),
                        (7, "До", f'={q(T, "$N$4")}'), (8, "Для графика", f'={q(T, "$M$4")}')):
        hput(f"AT{r_}", lab); hput(f"{CH}{r_}", f_)

    hl, hr, hm, ht = cs("A", 8), cs("J", 8), cs("S", 6), cs("Z", 9)
    for blk, labs in ((hl, ["Дата", "Команда P1", "Команда P2", "P1 H1", "P1 H2", "P2 H1", "P2 H2", "Турнир"]),
                      (hr, ["Дата", "Команда P1", "Команда P2", "P1 H1", "P1 H2", "P2 H1", "P2 H2", "Турнир"]),
                      (hm, ["M Дата", "Турнир", "Угл P1", "Угл P2", "Вес", "Вес МО+"]),
                      (ht, ["T Дата", "P1", "Team1", "P2", "Team2", "C1", "C2", "Вес", "Вес МО+"])):
        for c, lab in zip(blk, labs):
            hput(f"{c}1", lab)
    hput(f"{hl[0]}2", h2h_table(False))
    hput(f"{hr[0]}2", h2h_table(True))
    hput(f"{hm[0]}2", h2h_model())
    hput(f"{ht[0]}2", tour_model())
    for blk in (hl, hr, hm, ht):
        hf.append({"range": f"{blk[0]}2:{blk[0]}", "format": {"numberFormat": {"type": "DATE_TIME", "pattern": "dd.mm.yyyy hh:mm"}}})
        hf.append({"range": f"{blk[0]}1:{blk[-1]}1", "format": {"textFormat": {"bold": True}}})

    # ================= helper: model chain (rows as Console 1 AH36:AI101) =================
    s = S
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
    hf.append({"range": "AT1:AT101", "format": {"textFormat": {"bold": True}}})

    # ================= helper: chart data + labels =================
    hc = cs("AJ", 7)
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
    LBL = ("AQ", "AR")
    for lc, yc, name in ((LBL[0], hc[1], "Метки тотала"), (LBL[1], hc[3], "Метки разницы")):
        Y = f"{yc}2:{yc}71"
        hput(f"{lc}1", name)
        hput(f"{lc}2", f'=ARRAYFORMULA(IF(ISNUMBER({Y})*(MOD(71-ROW({Y}),{LBL_STEP})=0),TEXT({Y},"0.00"),""))')
    hf.append({"range": f"{hc[0]}2:{hc[0]}71", "format": {"numberFormat": {"type": "DATE", "pattern": "dd.mm.yyyy"}}})

    # ================= helper: dropdown lists (all players from PAIRS) =================
    ls = cs("BA", 5)
    PA, PB, PC, PD = (f"PAIRS!${x}$2:${x}" for x in "ABCD")
    tP1, tP2, tTR = q(T, "$A$3"), q(T, "$D$3"), q(T, "$A$1")
    hput(f"{ls[0]}1", "Турниры")
    hput(f"{ls[0]}2", f'=IFERROR(VSTACK("*",SORT(UNIQUE(FILTER({PC},{PA}={tP1},(({tP2}="")+({PB}={tP2}))>0)))),'
                      f'VSTACK("*",SORT(UNIQUE(FILTER({PC},{PC}<>"")))))')
    hput(f"{ls[1]}1", "Игроки")
    hput(f"{ls[1]}2", f'=IFERROR(SORT(UNIQUE(FILTER({PA},{PA}<>""))),"")')
    hput(f"{ls[2]}1", "Соперники")
    hput(f"{ls[2]}2", f'=IFERROR(SORT(UNIQUE(FILTER({PB},{PA}={tP1}))),"")')
    for col, pl, lab in ((ls[3], tP1, "Команды P1"), (ls[4], tP2, "Команды P2")):
        hput(f"{col}1", lab)
        hput(f"{col}2", f'=IFERROR(VSTACK("*",SORT(UNIQUE(FILTER({PD},{PA}={pl},(({tTR}="*")+({PC}={tTR}))>0)))),"*")')
    hf.append({"range": f"{ls[0]}1:{ls[4]}1", "format": {"textFormat": {"bold": True}}})

    # ================= settings AH5:AI26 =================
    a_ = lambda a1: shift(a1, SET_OFF)
    for a1, v in SET_LABELS:
        cput(a_(a1), v)
    for a1, v in SET_DEFAULT.items():
        cput(a_(a1), settings.get(a1, v))
    cput(a_("AH14"), f"=CEILING({s['BASE']}+(1-SQRT({s['LIN1']}))*100/{s['KDEC']})")
    cput(a_("AH8"), "Настройки угловых")
    merge("AH", 8 + SET_OFF, "AI", 8 + SET_OFF)
    cfmt(f"{a_('AH8')}:{a_('AI8')}", {"backgroundColor": GOLD, "textFormat": {"bold": True}, "horizontalAlignment": "CENTER"})
    for r in (9, 11, 16, 18, 20, 22, 24, 27, 29):
        cfmt(f"{a_(f'AH{r}')}:{a_(f'AI{r}')}", {"backgroundColor": GREEN, "textFormat": {"bold": True}, "horizontalAlignment": "CENTER"})
    cfmt(a_("AI13"), {"backgroundColor": GREEN, "textFormat": {"bold": True}, "horizontalAlignment": "CENTER"})
    cfmt(f"{a_('AH10')}:{a_('AI30')}", {"horizontalAlignment": "CENTER"})
    for a1 in SET_DEFAULT:
        cfmt(a_(a1), {"backgroundColor": LGOLD})
    cfmt(a_("AH14"), {"backgroundColor": PINK})
    for a1 in ("AI12", "AI14"):
        cfmt(a_(a1), {"numberFormat": {"type": "PERCENT", "pattern": "0.0%"}})
    for a1 in ("AH28", "AH30"):
        cfmt(a_(a1), {"numberFormat": {"type": "DATE", "pattern": "dd.mm.yyyy"}})

    # ================= tables A5:E21 (all tournaments) / G5:K21 (tournament A1) =================
    def table(c0, hb, title):
        c = cs(c0, 5)
        J, P1H1, P1H2, P2H1, P2H2 = HQ(hb[0]), HQ(hb[3]), HQ(hb[4]), HQ(hb[5]), HQ(hb[6])
        ok = f"ISNUMBER({J})"
        D = f"(({P1H1}+{P1H2})-({P2H1}+{P2H2}))"
        Tt = f"({P1H1}+{P1H2}+{P2H1}+{P2H2})"
        n = f"COUNT({J})"
        row = lambda r: R0 + r
        cnt = lambda e: f'=IF({n}=0,"",SUMPRODUCT({ok}*({e})))'
        o2 = lambda a, e: f'=IFERROR({odds(f"{a}/({a}+{e})", 2, s["MAR2"])},"")'
        cput(f"{c[0]}{row(1)}", title)
        cput(f"{c[0]}{row(2)}", "*"); cput(f"{c[3]}{row(2)}", "*")
        cput(f"{c[0]}{row(3)}", "=$A$3"); cput(f"{c[2]}{row(3)}", "X"); cput(f"{c[3]}{row(3)}", "=$D$3")
        merge(c[0], row(1), c[4], row(1))
        for r in (2, 3, 4, 5):
            merge(c[0], row(r), c[1], row(r)); merge(c[3], row(r), c[4], row(r))
        s3 = f"(${c[0]}${row(4)}+${c[2]}${row(4)}+${c[3]}${row(4)})"
        for cc, cond in ((c[0], f"{D}>0"), (c[2], f"{D}=0"), (c[3], f"{D}<0")):
            cput(f"{cc}{row(4)}", cnt(cond))
            cput(f"{cc}{row(5)}", f'=IFERROR({odds(f"${cc}${row(4)}/{s3}", 3, s["MAR3"])},"")')
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
        cput(f"{c[1]}{row(12)}", f'=IF({n}=0,"Тотал","Тотал   "&TEXT(SUMPRODUCT({ok}*{Tt})/{n},"0.00"))')
        merge(c[1], row(12), c[3], row(12))
        mid_t = f"{c[2]}{row(15)}"
        cput(mid_t, f"=IF({n}=0,3.5,MAX(1.5,FLOOR(SUMPRODUCT({ok}*{Tt})/{n})+0.5))")
        for r in (13, 14, 16, 17):
            cput(f"{c[2]}{row(r)}", f"={mid_t}{r - 15:+d}")
        for r in range(13, 18):
            L = f"${c[2]}${row(r)}"
            cput(f"{c[0]}{row(r)}", cnt(f"{Tt}>{L}")); cput(f"{c[4]}{row(r)}", cnt(f"{Tt}<{L}"))
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

    TQ = q(H, XIN["TOUR"])
    table("A", hl, '="УГЛОВЫЕ  * все турниры ("&$A$3&" - "&$D$3&")"')
    table("G", hr, f'="УГЛОВЫЕ  "&{TQ}&" ("&$A$3&" - "&$D$3&")"')

    # ================= players summary P5:Y9 =================
    J, P1H1, P1H2, P2H1, P2H2 = HQ(hl[0]), HQ(hl[3]), HQ(hl[4]), HQ(hl[5]), HQ(hl[6])
    nL = f"COUNT({J})"
    r1 = R0 + 1
    cols = cs("P", 10)                   # P label, Q..Y = И В Н П УЗ УП Т ТЗ ТП
    cput(f"P{r1}", f'="Угловые: "&{TQ}')
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
    cfmt(f"P{r1}:Y{r1}", {"backgroundColor": GOLD, "textFormat": {"bold": True}, "horizontalAlignment": "CENTER"})
    for k, bg in ((1, BLUE), (2, GREY), (3, BLUE), (4, GREY)):
        cfmt(f"P{r1 + k}:Y{r1 + k}", {"backgroundColor": bg, "horizontalAlignment": "CENTER"})
    cfmt(f"W{r1 + 1}:Y{r1 + 4}", {"numberFormat": {"type": "NUMBER", "pattern": "0.00"}})

    # ================= halves P12:U14 =================
    h0 = R0 + 8
    cput(f"P{h0}", '=$A$3&" - "&$D$3'); cput(f"Q{h0}", "УГ"); cput(f"R{h0}", "%УГ"); cput(f"S{h0}", "*")
    cput(f"T{h0}", "УГ"); cput(f"U{h0}", "%УГ")
    cput(f"P{h0 + 1}", "1-й тайм"); cput(f"P{h0 + 2}", "2-й тайм")
    cput(f"Q{h0 + 1}", f"=SUM({P1H1})+SUM({P2H1})"); cput(f"Q{h0 + 2}", f"=SUM({P1H2})+SUM({P2H2})")
    cput(f"R{h0 + 1}", f'=IFERROR(Q{h0 + 1}/(Q{h0 + 1}+Q{h0 + 2}),"")')
    cput(f"R{h0 + 2}", f'=IFERROR(Q{h0 + 2}/(Q{h0 + 1}+Q{h0 + 2}),"")')
    allwin = f'{D_DT},">="&$N$2,{D_DT},"<"&$N$4+1'
    cput(f"T{h0 + 1}", f"=SUMIFS({D_H1},{allwin})"); cput(f"T{h0 + 2}", f"=SUMIFS({D_H2},{allwin})")
    cput(f"U{h0 + 1}", f'=IFERROR(T{h0 + 1}/(T{h0 + 1}+T{h0 + 2}),"")')
    cput(f"U{h0 + 2}", f'=IFERROR(T{h0 + 2}/(T{h0 + 1}+T{h0 + 2}),"")')
    merge("S", h0 + 1, "S", h0 + 2)
    cfmt(f"P{h0}:U{h0}", {"backgroundColor": GREY, "textFormat": {"bold": True}, "horizontalAlignment": "CENTER"})
    cfmt(f"P{h0 + 1}:P{h0 + 2}", {"backgroundColor": GREY, "horizontalAlignment": "CENTER"})
    cfmt(f"Q{h0 + 1}:R{h0 + 2}", {"backgroundColor": BLUE, "horizontalAlignment": "CENTER"})
    cfmt(f"T{h0 + 1}:U{h0 + 2}", {"backgroundColor": BLUE, "horizontalAlignment": "CENTER"})
    cfmt(f"R{h0 + 1}:R{h0 + 2}", {"numberFormat": {"type": "NUMBER", "pattern": "0.0000"}})
    cfmt(f"U{h0 + 1}:U{h0 + 2}", {"numberFormat": {"type": "NUMBER", "pattern": "0.0000"}})

    # ================= panel M11:N17 =================
    p0 = R0 + 7
    for a1, v in ((f"M{p0}", "Матчи"), (f"N{p0}", "Разница"),
                  (f"M{p0 + 1}", f"={nL}"),
                  (f"N{p0 + 1}", f'=IFERROR((SUM({P1H1})+SUM({P1H2})-SUM({P2H1})-SUM({P2H2}))/{nL},"")'),
                  (f"M{p0 + 2}", "Тотал"), (f"N{p0 + 2}", "Часть H1"),
                  (f"M{p0 + 3}", f'=IFERROR((SUM({P1H1})+SUM({P1H2})+SUM({P2H1})+SUM({P2H2}))/{nL},"")'),
                  (f"N{p0 + 3}", f'=IFERROR((SUM({P1H1})+SUM({P2H1}))/(SUM({P1H1})+SUM({P1H2})+SUM({P2H1})+SUM({P2H2})),"")'),
                  (f"M{p0 + 5}", "МО (!!!)"), (f"N{p0 + 5}", "Разница (!!!)"),
                  (f"M{p0 + 6}", f"={CHQ(84)}+{CHQ(85)}"), (f"N{p0 + 6}", f"={CHQ(84)}-{CHQ(85)}")):
        cput(a1, v)
    cfmt(f"M{p0}:N{p0 + 6}", {"horizontalAlignment": "CENTER"})
    for k in (0, 2, 5):
        cfmt(f"M{p0 + k}:N{p0 + k}", {"backgroundColor": BLUE, "textFormat": {"bold": True}})
    cfmt(f"M{p0 + 1}", {"backgroundColor": PINK, "numberFormat": {"type": "NUMBER", "pattern": "0"}})
    cfmt(f"N{p0 + 1}", {"backgroundColor": PINK, "numberFormat": {"type": "NUMBER", "pattern": "0.00"}})
    cfmt(f"M{p0 + 3}", {"backgroundColor": PINK, "numberFormat": {"type": "NUMBER", "pattern": "0.00"}})
    cfmt(f"N{p0 + 3}", {"backgroundColor": PINK, "numberFormat": {"type": "PERCENT", "pattern": "0.0%"}})
    cfmt(f"M{p0 + 6}:N{p0 + 6}", {"backgroundColor": GOLD, "textFormat": {"bold": True},
                                  "numberFormat": {"type": "NUMBER", "pattern": "0.00"}})

    # ================= result P16:S18 =================
    rr0 = R0 + 12
    cput(f"P{rr0}", "Результат (угл.)"); cput(f"Q{rr0}", "W1"); cput(f"R{rr0}", "Draw"); cput(f"S{rr0}", "W2")
    cput(f"P{rr0 + 1}", "Poisson Result"); cput(f"P{rr0 + 2}", "Final Result")
    for c, pr, fr in (("Q", 95, 99), ("R", 96, 100), ("S", 97, 101)):
        cput(f"{c}{rr0 + 1}", f'=IFERROR({odds(CHQ(pr), 3, s["MAR3"])},"")')
        cput(f"{c}{rr0 + 2}", f'=IFERROR({odds(CHQ(fr), 3, s["MAR3"])},"")')
    cfmt(f"P{rr0}:S{rr0}", {"backgroundColor": BLUE, "textFormat": {"bold": True}, "horizontalAlignment": "CENTER"})
    cfmt(f"P{rr0 + 1}:P{rr0 + 2}", {"backgroundColor": BLUE, "horizontalAlignment": "CENTER"})
    cfmt(f"Q{rr0 + 1}:S{rr0 + 1}", {"backgroundColor": LGOLD, "horizontalAlignment": "CENTER",
                                    "numberFormat": {"type": "NUMBER", "pattern": "0.00"}})
    cfmt(f"Q{rr0 + 2}:S{rr0 + 2}", {"backgroundColor": GOLD, "textFormat": {"bold": True}, "horizontalAlignment": "CENTER",
                                    "numberFormat": {"type": "NUMBER", "pattern": "0.00"}})

    # ================= last 20 H2H matches M23:S43 =================
    e0 = R0 + 19
    for c, lab in zip(cs("M", 7), ["Event Date/Time", "", "Pair", "УГ_1", "УГ_2", "H1_1", "H1_2"]):
        cput(f"{c}{e0}", lab)
    cput(f"M{e0 + 1}", '=IFERROR(ARRAYFORMULA(ARRAY_CONSTRAIN(SORT(LET(db,DB_Corners!A2:T,pa,LOWER(TRIM($A$3)),'
                       'pb,LOWER(TRIM($D$3)),kk,INDEX(db,,17),f,FILTER(db,((kk=pa&"|"&pb)+(kk=pb&"|"&pa))>0,'
                       'INDEX(db,,2)<$N$4+1),a,LOWER(INDEX(f,,5))=pa,HSTACK(INDEX(f,,2),IF(LEN(INDEX(f,,1)),"",""),'
                       'IF(LEN(INDEX(f,,1)),$A$3&" - "&$D$3,""),IF(a,INDEX(f,,14),INDEX(f,,15)),'
                       'IF(a,INDEX(f,,15),INDEX(f,,14)),IF(a,INDEX(f,,8),INDEX(f,,10)),'
                       'IF(a,INDEX(f,,10),INDEX(f,,8)))),1,FALSE),20,7)),"")')
    cfmt(f"M{e0}:S{e0}", {"backgroundColor": BLUE, "textFormat": {"bold": True}, "horizontalAlignment": "CENTER"})
    cfmt(f"M{e0 + 1}:M{e0 + 20}", {"numberFormat": {"type": "DATE_TIME", "pattern": "dd.mm.yyyy hh:mm"}})
    cfmt(f"N{e0 + 1}:S{e0 + 20}", {"horizontalAlignment": "CENTER"})

    # ================= charts under the tables (data on helper) =================
    def src(col):
        return {"sourceRange": {"sources": [grid(hs.id, col, 1, col, 71)]}}

    def ser(col, rgb, label_col=None):
        dl = ({"type": "CUSTOM", "customLabelData": src(label_col), "placement": "ABOVE"}
              if label_col else {"type": "NONE"})
        return {"series": src(col), "targetAxis": "LEFT_AXIS", "colorStyle": {"rgbColor": rgb}, "color": rgb,
                "lineStyle": {"width": 2, "type": "SOLID"}, "dataLabel": dl}

    def chart(title, ycol, trend, lbl, anchor_col, vmin=None, vmax=None):
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
                "anchorCell": {"sheetId": ws.id, "rowIndex": R0 + 18, "columnIndex": colnum(anchor_col) - 1},
                "widthPixels": 470, "heightPixels": 330}}}}}

    charts = [chart("Средняя разница угловых", hc[3], hc[6], LBL[1], "A", -3, 3),
              chart("Средний тотал угловых", hc[1], hc[5], LBL[0], "G")]

    def dv(a1, src_):
        m = re.fullmatch(r"([A-Z]+)(\d+)", a1)
        return {"setDataValidation": {
            "range": grid(ws.id, m.group(1), int(m.group(2)), m.group(1), int(m.group(2))),
            "rule": {"condition": {"type": "ONE_OF_RANGE", "values": [{"userEnteredValue": src_}]},
                     "showCustomUi": True, "strict": False}}}

    lst = lambda c: f"={q(H, f'${c}$2:${c}')}"
    dvs = [dv("A1", lst(ls[0])), dv("A3", lst(ls[1])), dv("D3", lst(ls[2])), dv("A2", lst(ls[3])),
           dv("D2", lst(ls[4]))]
    widths = [{"updateDimensionProperties": {
        "range": {"sheetId": ws.id, "dimension": "COLUMNS", "startIndex": colnum(a) - 1, "endIndex": colnum(b)},
        "properties": {"pixelSize": px}, "fields": "pixelSize"}}
        for a, b, px in (("A", "K", 85), ("F", "F", 15), ("L", "L", 15), ("O", "O", 15), ("M", "N", 120),
                         ("P", "P", 160), ("Q", "Y", 60), ("AH", "AI", 120))]

    # ================= write: helper, formats, layout, values, marker =================
    for k in range(0, len(hf), 25):
        retry(hs.batch_format, hf[k:k + 25])
    for k in range(0, len(hcells), 80):
        retry(hs.batch_update, hcells[k:k + 80], value_input_option="USER_ENTERED")
        time.sleep(1)
    print(f"{H}: built ({len(hcells)} cells)")
    for k in range(0, len(cf), 25):
        retry(ws.batch_format, cf[k:k + 25])
    retry(book.batch_update, {"requests": reqs + widths + dvs + charts})
    for k in range(0, len(ccells), 80):
        retry(ws.batch_update, ccells[k:k + 80], value_input_option="USER_ENTERED")
        time.sleep(1)
    retry(ws.update, [[f"{PREFIX} ok"]], "AH1")
    print(f"{T}: built ({len(ccells)} cells): chips A1:E3, tables A5:K21, panel M11:N17, summary P5:Y9, "
          f"halves P12:U14, result P16:S18, last H2H M23:S43, settings AH4:AI26, charts from row {R0 + 19}")


titles = [w.title for w in book.worksheets()]
nums = [int(m.group(1)) for t in titles for m in [re.fullmatch(rf"{PREFIX}_v(\d+)", t)] if m]
n = max(nums, default=0) + 1
T, H = f"{PREFIX}_v{n}", f"{PREFIX}_v{n}_model"
for need in (DB, "PAIRS"):
    if need not in titles:
        raise SystemExit(f"{need} not found in Corners_H2H -> run corners-to-sheet first")
print("building", T, "+", H)
build(T, H, read_ecf_settings())
