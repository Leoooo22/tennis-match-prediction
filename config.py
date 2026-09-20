N_RECENT = 20

round_order = {
    "1st Round": 1,
    "2nd Round": 2,
    "3rd Round": 3,
    "4th Round": 4,
    "Quarterfinals": 5,
    "Semifinals": 6,
    "Round Robin": 6,
    "The Final": 7,
}

series_order = {
    "ATP250": 1,
    "ATP500": 2,
    "Masters 1000": 3,
    "Masters Cup": 4,
    "Grand Slam": 5,
}

ODDS_COLS = ["Avg_P1", "Avg_P2"]

COLS_TO_DROP = [
    "ATP",
    "W1", "L1", "W2", "L2", "W3", "L3",
    "W4", "L4", "W5", "L5",
    "Wsets", "Lsets",
    "Comment",
    "BFEW", "BFEL",
    "B365W", "B365L", "PSW", "PSL", "MaxW", "MaxL",
]

COLS_TO_DROP_RAND = [
    "Winner", "Loser",
    "WRank", "LRank", "WPts", "LPts",
    "AvgW", "AvgL",
    "WRank_missing", "LRank_missing", "WPts_missing", "LPts_missing",
]


COLS_ORDERED = [
    "y",
    "Year", "Date",
    "Location", "Tournament", "Series", "Court", "Round", "Best of",
    "Surface_Clay", "Surface_Grass", "Surface_Hard",
    "Player1", "Player2",
    "rank_diff", "pts_diff",
    "win_rate_diff",
    "win_rate_surface_diff",
    "recent_win_rate_diff",
    "h2h_win_rate_diff",
    "Avg_P1", "Avg_P2",
    "Best of_missing", "P1Rank_missing", "P2Rank_missing", "P1Pts_missing", "P2Pts_missing", "odds_missing",
]
