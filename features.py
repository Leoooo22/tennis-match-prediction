from collections import defaultdict, deque

from config import N_RECENT, COLS_ORDERED


def compute_win_rate(df):
    overall = defaultdict(lambda: {"wins": 0, "matches": 0})
    win_rate_p1_list = []
    win_rate_p2_list = []

    for _, row in df.iterrows():
        p1 = row["Player1"]
        p2 = row["Player2"]
        winner = p1 if row["y"] == 1 else p2
        loser = p2 if row["y"] == 1 else p1

        wr_p1 = overall[p1]["wins"] / overall[p1]["matches"] if overall[p1]["matches"] > 0 else 0.5
        wr_p2 = overall[p2]["wins"] / overall[p2]["matches"] if overall[p2]["matches"] > 0 else 0.5

        win_rate_p1_list.append(wr_p1)
        win_rate_p2_list.append(wr_p2)

        overall[winner]["wins"] += 1
        overall[winner]["matches"] += 1
        overall[loser]["matches"] += 1

    df["win_rate_p1"] = win_rate_p1_list
    df["win_rate_p2"] = win_rate_p2_list
    return df


def compute_surface_win_rate(df):
    df["_surface"] = (
        df[["Surface_Clay", "Surface_Grass", "Surface_Hard"]]
        .idxmax(axis=1)
        .str.replace("Surface_", "")
    )

    by_surface = defaultdict(lambda: defaultdict(lambda: {"wins": 0, "matches": 0}))
    win_rate_surface_p1_list = []
    win_rate_surface_p2_list = []

    for _, row in df.iterrows():
        p1 = row["Player1"]
        p2 = row["Player2"]
        surface = row["_surface"]
        winner = p1 if row["y"] == 1 else p2
        loser = p2 if row["y"] == 1 else p1

        ws_p1 = by_surface[p1][surface]["wins"] / by_surface[p1][surface]["matches"] if by_surface[p1][surface]["matches"] > 0 else 0.5
        ws_p2 = by_surface[p2][surface]["wins"] / by_surface[p2][surface]["matches"] if by_surface[p2][surface]["matches"] > 0 else 0.5

        win_rate_surface_p1_list.append(ws_p1)
        win_rate_surface_p2_list.append(ws_p2)

        by_surface[winner][surface]["wins"] += 1
        by_surface[winner][surface]["matches"] += 1
        by_surface[loser][surface]["matches"]  += 1

    df["win_rate_surface_p1"] = win_rate_surface_p1_list
    df["win_rate_surface_p2"] = win_rate_surface_p2_list
    df = df.drop(columns=["_surface"])
    return df


def compute_recent_win_rate(df):
    recent = defaultdict(lambda: deque(maxlen=N_RECENT))
    recent_win_rate_p1_list = []
    recent_win_rate_p2_list = []

    for _, row in df.iterrows():
        p1 = row["Player1"]
        p2 = row["Player2"]
        winner = p1 if row["y"] == 1 else p2
        loser = p2 if row["y"] == 1 else p1

        rec_p1 = sum(recent[p1]) / len(recent[p1]) if recent[p1] else 0.5
        rec_p2 = sum(recent[p2]) / len(recent[p2]) if recent[p2] else 0.5

        recent_win_rate_p1_list.append(rec_p1)
        recent_win_rate_p2_list.append(rec_p2)

        recent[winner].append(1)
        recent[loser].append(0)

    df["recent_win_rate_p1"] = recent_win_rate_p1_list
    df["recent_win_rate_p2"] = recent_win_rate_p2_list
    return df


def compute_h2h(df):
    h2h = defaultdict(lambda: defaultdict(int))
    h2h_win_rate_p1_list = []

    for _, row in df.iterrows():
        p1 = row["Player1"]
        p2 = row["Player2"]
        winner = p1 if row["y"] == 1 else p2
        key = tuple(sorted([p1, p2]))

        total = h2h[key]["total"]
        h2h_p1 = h2h[key][p1] / total if total > 0 else 0.5

        h2h_win_rate_p1_list.append(h2h_p1)

        h2h[key]["total"] += 1
        h2h[key][winner] += 1

    df["h2h_win_rate_p1"] = h2h_win_rate_p1_list
    return df


def compute_differentials(df):
    df["h2h_win_rate_p2"] = 1 - df["h2h_win_rate_p1"]
    df["h2h_win_rate_diff"] = df["h2h_win_rate_p1"] - df["h2h_win_rate_p2"]
    df["rank_diff"] = df["P1Rank"] - df["P2Rank"]
    df["pts_diff"] = df["P1Pts"] - df["P2Pts"]
    df["win_rate_diff"] = df["win_rate_p1"] - df["win_rate_p2"]
    df["win_rate_surface_diff"] = df["win_rate_surface_p1"] - df["win_rate_surface_p2"]
    df["recent_win_rate_diff"]  = df["recent_win_rate_p1"] - df["recent_win_rate_p2"]

    df = df[COLS_ORDERED]

    bool_cols = ["Surface_Clay", "Surface_Grass", "Surface_Hard"]
    df[bool_cols] = df[bool_cols].astype(int)
    return df


def compute_all_features(df):
    df = df.sort_values("Date").reset_index(drop=True)
    df = compute_win_rate(df)
    df = compute_surface_win_rate(df)
    df = compute_recent_win_rate(df)
    df = compute_h2h(df)
    df = compute_differentials(df)
    return df
