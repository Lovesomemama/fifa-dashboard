import json
import re
import pandas as pd
from pathlib import Path

INPUT_GLOB = "*_fifa_score.parquet"
HISTORY = "Players_Groups_History.xlsx"
OUTPUT_JSON = "fifa_data.json"

# Исключаем лиги виртуальных eComp (только боты Virtual_1..17).
# Логика для ботов будет реализована отдельно — пока скипаем.
EXCLUDE_LEAGUE_PATTERN = r"Virtual eComp"

# Альтернативный фильтр (эквивалентен по составу матчей):
# EXCLUDE_PLAYER_PATTERN = r"^Virtual_\d+$"


def load_player_brand(path):
    """Маппинг игрок -> бренд (ESB/ESL/ECF) по последней дате транзакции."""
    h = pd.read_excel(path, sheet_name="Groups")
    h.columns = ["Player", "Group", "Date"]
    h["Date"] = pd.to_datetime(h["Date"])
    h["Brand"] = h["Group"].str.extract(r"^(ESB|ESL|ECF)", expand=False)
    h = h.dropna(subset=["Brand"])
    h = h.sort_values("Date").groupby("Player").tail(1)
    return dict(zip(h["Player"], h["Brand"]))


PLAYER_BRAND = {}  # заполнится в main()


def get_brand(player):
    return PLAYER_BRAND.get(player, "UNK")


def parse_score(s):
    h, a = s.split(":")
    return int(h), int(a)


def time_to_seconds(t):
    """'12:34' -> 754"""
    if pd.isna(t) or not t:
        return 0
    parts = str(t).split(":")
    return int(parts[0]) * 60 + int(parts[1])


def gtime_to_seconds(v):
    """Игровое время гола 'MM:SS' -> секунды. Пусто/битое -> None."""
    if v is None or (isinstance(v, float) and pd.isna(v)) or str(v).strip() == "":
        return None
    parts = str(v).split(":")
    try:
        return int(parts[0]) * 60 + int(parts[1])
    except (ValueError, IndexError):
        return None


def process_match(m_df):
    m_df = m_df.sort_values("incident_timestamp").reset_index(drop=True)
    first = m_df.iloc[0]

    home = first["team_home"]
    away = first["team_away"]
    # Алфавитная нормализация: p1<=p2 всегда
    if home <= away:
        p1, p2, flip = home, away, False
    else:
        p1, p2, flip = away, home, True
    date = first["match_date"][:10]
    fmt = int(first["match_format"])
    brand = get_brand(p1)

    # Объявленный финальный счёт (Match Finished или последний инцидент)
    finished = m_df[m_df.incident_type_id == 1030]
    if len(finished) > 0:
        declared = finished.iloc[0]["current_score"]
    else:
        declared = m_df.iloc[-1]["current_score"]
    d1, d2 = parse_score(declared)

    # Последовательность голов (сортируем по incident_timestamp — реальное время)
    cs = m_df[m_df.incident_type_id == 1042].copy()
    cs = cs.sort_values("incident_timestamp")

    prev_h, prev_a = 0, 0
    seq = []
    non_monotonic = False
    for _, row in cs.iterrows():
        h, a = parse_score(row["current_score"])
        if h == prev_h + 1 and a == prev_a:
            side = 0
        elif a == prev_a + 1 and h == prev_h:
            side = 1
        elif h == prev_h and a == prev_a:
            continue  # дубликат
        else:
            non_monotonic = True
            break
        # Игровое время гола (секунды) — для расчёта времени до след. гола
        t = gtime_to_seconds(row.get("incident_game_time"))
        seq.append([h, a, side, t])
        prev_h, prev_a = h, a

    if non_monotonic:
        return None, (p1, p2, first["match_date"][:10], first["match_title"])

    # Авторитет: последовательность голов (для консистентности UI)
    s1, s2 = prev_h, prev_a

    # Если flip — инвертировать всё под алфавит (время t не меняется)
    if flip:
        s1, s2 = s2, s1
        d1, d2 = d2, d1
        seq = [[a, h, 1 - side, t] for h, a, side, t in seq]

    return [p1, p2, date, brand, fmt, s1, s2, seq], (d1, d2, prev_h, prev_a)


def main():
    base = Path(__file__).resolve().parent

    global PLAYER_BRAND
    PLAYER_BRAND = load_player_brand(base / HISTORY)
    print(f"Loaded brand mapping: {len(PLAYER_BRAND)} players")

    files = sorted(base.glob(INPUT_GLOB))
    if not files:
        raise FileNotFoundError(f"No files matching {INPUT_GLOB}")
    print(f"Input files: {[f.name for f in files]}")
    df = pd.concat([pd.read_parquet(f) for f in files], ignore_index=True)
    n0 = df["match_id"].nunique()

    # Фильтр Virtual eComp (исключаем ботов)
    df = df[~df["league_name"].str.contains(EXCLUDE_LEAGUE_PATTERN, na=False)].copy()
    n1 = df["match_id"].nunique()
    print(f"Loaded: {len(df)} rows, {n1} matches (excluded {n0-n1} Virtual eComp matches)")

    DATA = []
    mismatches = 0
    by_format = {3: 0, 4: 0, 6: 0}

    excluded = []
    for mid, m in df.groupby("match_id"):
        result = process_match(m)
        if result[0] is None:
            excluded.append(result[1])
            continue
        rec, (d1, d2, ph, pa) = result
        if (ph, pa) != (d1, d2):
            mismatches += 1
        DATA.append(rec)
        by_format[rec[4]] = by_format.get(rec[4], 0) + 1

    DATA.sort(key=lambda r: (r[2], r[0], r[1]))

    print(f"Matches in DATA: {len(DATA)}")
    print(f"Format breakdown: {by_format}")
    print(f"Seq/final mismatches: {mismatches}")
    print(f"Excluded (non-monotonic / data hole): {len(excluded)}")
    for p1, p2, date, title in excluded[:20]:
        print(f"  {date} {title}")
    if len(excluded) > 20:
        print(f"  ... +{len(excluded)-20} more")

    out = base / OUTPUT_JSON
    with open(out, "w") as f:
        json.dump(DATA, f, separators=(",", ":"))
    print(f"Written: {out} ({out.stat().st_size} bytes)")


if __name__ == "__main__":
    main()
