# app.py
# Full, cleaned Streamlit app that:
# - Stores matches.csv and ratings.csv in a GitHub repo via the GitHub API
# - Recomputes ELO ratings (by replaying matches) and saves ratings.csv
# - Provides rating trend charts, team (pair) analysis, player badges, and date filters
#
# Requirements:
# - streamlit, pandas, requests, altair
# - an `elo.py` file in the same repo providing update_elo(...) and predict_win_probability(...)
# - Streamlit secrets must include:
#     GITHUB_TOKEN, REPO_OWNER, REPO_NAME, MATCHES_CSV, RATINGS_CSV
#
# Notes:
# - This code is robust to older matches.csv files that lack a 'date' column.
# - It ensures numeric types before calculations to avoid dtype errors.

import streamlit as st
import pandas as pd
import numpy as np
import requests
import base64
from io import StringIO
import altair as alt
import datetime
from elo import update_elo, predict_win_probability

st.set_page_config(page_title="Badminton Doubles Tracker", layout="wide")
st.title("🏸 Badminton Doubles Tracker")

# --------------------------
# Helper functions
# --------------------------
def normalize(name):
    """Normalize player names to Title case and strip whitespace; handle None."""
    if name is None:
        return ""
    if not isinstance(name, str):
        name = str(name)
    return name.strip().title()

def ensure_column(df, col, default=""):
    """Ensure dataframe has a column 'col' (create with default if missing)."""
    if col not in df.columns:
        df[col] = default
    return df

def df_to_b64_csv(df):
    return base64.b64encode(df.to_csv(index=False).encode()).decode()

# --------------------------
# GitHub helpers (read/write)
# --------------------------
def gh_file_info(path):
    """
    Return (json_response, status_code) from GitHub API for a file path.
    If the file does not exist, returns (None, status_code).
    """
    url = f"https://api.github.com/repos/{st.secrets['REPO_OWNER']}/{st.secrets['REPO_NAME']}/contents/{path}"
    headers = {"Authorization": f"token {st.secrets['GITHUB_TOKEN']}"}
    r = requests.get(url, headers=headers)
    if r.status_code == 200:
        return r.json(), 200
    else:
        return None, r.status_code

def load_csv_from_github(path):
    """
    Load CSV from GitHub repo path.
    Returns (df, sha) or (empty_df, None) if file missing or empty.
    """
    info, status = gh_file_info(path)
    if status != 200 or "content" not in info:
        # missing file, return empty df (no sha)
        return pd.DataFrame(), None
    content = base64.b64decode(info["content"]).decode("utf-8")
    df = pd.read_csv(StringIO(content))
    return df, info.get("sha")

def save_csv_to_github(path, df, sha=None, message="update csv"):
    """
    Save DataFrame to GitHub at path.
    If sha provided, GitHub will update; if None, GitHub will create the file.
    Returns new JSON response from GitHub.
    """
    url = f"https://api.github.com/repos/{st.secrets['REPO_OWNER']}/{st.secrets['REPO_NAME']}/contents/{path}"
    headers = {"Authorization": f"token {st.secrets['GITHUB_TOKEN']}"}
    encoded = df_to_b64_csv(df)
    payload = {"message": message, "content": encoded}
    if sha:
        payload["sha"] = sha
    resp = requests.put(url, headers=headers, json=payload)
    resp.raise_for_status()
    return resp.json()

# --------------------------
# Load CSVs from GitHub (or create empty DataFrames)
# --------------------------
MATCHES_PATH = st.secrets.get("MATCHES_CSV", "matches.csv")
RATINGS_PATH = st.secrets.get("RATINGS_CSV", "ratings.csv")

try:
    matches_df, matches_sha = load_csv_from_github(MATCHES_PATH)
except Exception as e:
    st.error(f"Error loading matches.csv from GitHub: {e}")
    st.stop()

try:
    ratings_df, ratings_sha = load_csv_from_github(RATINGS_PATH)
except Exception as e:
    # treat missing ratings as empty
    ratings_df = pd.DataFrame()
    ratings_sha = None

# --------------------------
# Make sure required columns exist and normalize
# --------------------------
# For matches: expected columns: date (optional), playerA1, playerA2, playerB1, playerB2, scoreA, scoreB
if matches_df.empty:
    matches_df = pd.DataFrame(columns=["date", "playerA1", "playerA2", "playerB1", "playerB2", "scoreA", "scoreB"])

# Ensure columns exist to avoid KeyErrors later
for col in ["date", "playerA1", "playerA2", "playerB1", "playerB2", "scoreA", "scoreB"]:
    matches_df = ensure_column(matches_df, col, "" if col == "date" or col.startswith("player") else 0)

# Normalize player name columns and strip spaces
for c in ["playerA1", "playerA2", "playerB1", "playerB2"]:
    matches_df[c] = matches_df[c].fillna("").astype(str).apply(normalize)

# Coerce score columns to integers safely
matches_df["scoreA"] = pd.to_numeric(matches_df["scoreA"], errors="coerce").fillna(0).astype(int)
matches_df["scoreB"] = pd.to_numeric(matches_df["scoreB"], errors="coerce").fillna(0).astype(int)

# For ratings: expected columns: player, rating, wins, losses, matches
if ratings_df.empty:
    ratings_df = pd.DataFrame(columns=["player", "rating", "wins", "losses", "matches"])

# Normalize ratings_df player names and numeric types
if "player" in ratings_df.columns:
    ratings_df["player"] = ratings_df["player"].astype(str).apply(normalize)
else:
    ratings_df["player"] = []

for col in ["rating", "wins", "losses", "matches"]:
    if col not in ratings_df.columns:
        # add missing columns with defaults
        if col == "rating":
            ratings_df[col] = 1500.0
        else:
            ratings_df[col] = 0

ratings_df["rating"] = pd.to_numeric(ratings_df["rating"], errors="coerce").fillna(1500.0)
ratings_df["wins"] = pd.to_numeric(ratings_df["wins"], errors="coerce").fillna(0).astype(int)
ratings_df["losses"] = pd.to_numeric(ratings_df["losses"], errors="coerce").fillna(0).astype(int)
ratings_df["matches"] = pd.to_numeric(ratings_df["matches"], errors="coerce").fillna(0).astype(int)

# Build quick lookup structures
ratings = {row["player"]: float(row["rating"]) for _, row in ratings_df.iterrows()}
player_stats = {
    row["player"]: {"wins": int(row["wins"]), "losses": int(row["losses"]), "matches": int(row["matches"])}
    for _, row in ratings_df.iterrows()
}

# --------------------------
# UI: Add new match (with date)
# --------------------------
st.header("➕ Add New Match")

with st.form("add_match", clear_on_submit=False):
    c1, c2 = st.columns(2)
    A1 = normalize(c1.text_input("Team A - Player 1", value=""))
    A2 = normalize(c2.text_input("Team A - Player 2", value=""))
    B1 = normalize(c1.text_input("Team B - Player 1", value=""))
    B2 = normalize(c2.text_input("Team B - Player 2", value=""))

    # Typical badminton scoring ends around 21; use it as default example values.
    sA = c1.number_input("Score A", min_value=0, value=21, step=1)
    sB = c2.number_input("Score B", min_value=0, value=19, step=1)

    # date input (defaults to today)
    match_date = c2.date_input("Match date", value=datetime.date.today())

    submitted = st.form_submit_button("Add match")

if submitted:
    # Add new row to matches_df
    new_row = {
        "date": str(match_date),
        "playerA1": A1, "playerA2": A2,
        "playerB1": B1, "playerB2": B2,
        "scoreA": int(sA), "scoreB": int(sB)
    }
    matches_df = pd.concat([matches_df, pd.DataFrame([new_row])], ignore_index=True)

    # Attempt to save matches.csv to GitHub
    try:
        # Get latest sha before save (in case other writes happened)
        info, status = gh_file_info(MATCHES_PATH)
        sha_for_save = info.get("sha") if info else None
        res = save_csv_to_github(MATCHES_PATH, matches_df, sha_for_save, "Add match")
        # update saved sha
        matches_sha = res.get("content", {}).get("sha", matches_sha)
        st.success("Match added and saved to GitHub.")
    except Exception as e:
        st.error(f"Could not save match to GitHub: {e}")

    # Ensure players exist in ratings/stats
    for p in [A1, A2, B1, B2]:
        if p and p not in ratings:
            ratings[p] = 1500.0
        if p and p not in player_stats:
            player_stats[p] = {"wins": 0, "losses": 0, "matches": 0}

    # update matches count
    for p in [A1, A2, B1, B2]:
        if p:
            player_stats[p]["matches"] += 1

    # update wins/losses
    if int(sA) > int(sB):
        if A1: player_stats[A1]["wins"] += 1
        if A2: player_stats[A2]["wins"] += 1
        if B1: player_stats[B1]["losses"] += 1
        if B2: player_stats[B2]["losses"] += 1
    else:
        if B1: player_stats[B1]["wins"] += 1
        if B2: player_stats[B2]["wins"] += 1
        if A1: player_stats[A1]["losses"] += 1
        if A2: player_stats[A2]["losses"] += 1

    # Recompute ratings by replaying all matches in chronological order (safe)
    # First prepare a sorted copy by date
    # If date column missing or unparsable, fallback to file order
    try:
        temp = matches_df.copy()
        temp["date_parsed"] = pd.to_datetime(temp["date"], errors="coerce")
        if temp["date_parsed"].isna().all():
            # no usable dates -> keep original order
            temp = temp.reset_index(drop=True)
        else:
            temp = temp.sort_values("date_parsed").reset_index(drop=True)
    except Exception:
        temp = matches_df.reset_index(drop=True)

    # Start fresh ratings for recompute
    ratings_replay = {}
    for _, row in temp.iterrows():
        pA1 = normalize(row.get("playerA1", ""))
        pA2 = normalize(row.get("playerA2", ""))
        pB1 = normalize(row.get("playerB1", ""))
        pB2 = normalize(row.get("playerB2", ""))
        scA = int(row.get("scoreA", 0))
        scB = int(row.get("scoreB", 0))
        # ensure players exist
        for p in [pA1, pA2, pB1, pB2]:
            if p and p not in ratings_replay:
                ratings_replay[p] = 1500.0
        ratings_replay = update_elo(pA1, pA2, pB1, pB2, scA, scB, ratings_replay)

    # Replace ratings with replayed results
    ratings = ratings_replay

    # Rebuild ratings_df from ratings dict and player_stats
    ratings_rows = []
    for p, r in ratings.items():
        st_w = player_stats.get(p, {}).get("wins", 0)
        st_l = player_stats.get(p, {}).get("losses", 0)
        st_m = player_stats.get(p, {}).get("matches", 0)
        ratings_rows.append({"player": p, "rating": round(float(r), 2), "wins": int(st_w), "losses": int(st_l), "matches": int(st_m)})
    ratings_df = pd.DataFrame(ratings_rows).sort_values("rating", ascending=False).reset_index(drop=True)

    # Save ratings_df to GitHub
    try:
        # get latest sha for ratings if exists
        info_r, status_r = gh_file_info(RATINGS_PATH)
        sha_for_ratings = info_r.get("sha") if info_r else None
        res_r = save_csv_to_github(RATINGS_PATH, ratings_df, sha_for_ratings, "Update ratings after adding match")
        ratings_sha = res_r.get("content", {}).get("sha", ratings_sha)
    except Exception as e:
        st.error(f"Could not save ratings to GitHub: {e}")

# --------------------------
# Prepare matches for display & filtering
# --------------------------
# Ensure date column exists and parse to datetime for filtering
if "date" not in matches_df.columns:
    matches_df["date"] = ""

matches_df["date_parsed"] = pd.to_datetime(matches_df["date"], errors="coerce")
# If all dates are NaT, create synthetic date based on index (so filters work)
if matches_df["date_parsed"].isna().all():
    matches_df = matches_df.reset_index(drop=True)
    base = pd.to_datetime("1970-01-01")
    matches_df["date_parsed"] = base + pd.to_timedelta(matches_df.index, unit="D")

# Provide filters UI
st.header("📜 Match History & Filters")
min_date = matches_df["date_parsed"].min().date()
max_date = matches_df["date_parsed"].max().date()

c1, c2, c3 = st.columns([2,2,1])
with c1:
    start_date = st.date_input("From", value=min_date)
with c2:
    end_date = st.date_input("To", value=max_date)
with c3:
    _ = st.button("Apply")

# Build inclusive datetime range
start_dt = pd.to_datetime(start_date)
end_dt = pd.to_datetime(end_date) + pd.Timedelta(days=1) - pd.Timedelta(seconds=1)

filtered = matches_df[(matches_df["date_parsed"] >= start_dt) & (matches_df["date_parsed"] <= end_dt)].copy()

# Choose display columns that exist
display_cols = [c for c in ["date", "playerA1", "playerA2", "playerB1", "playerB2", "scoreA", "scoreB"] if c in filtered.columns]
st.subheader(f"Showing {len(filtered)} matches between {start_date} and {end_date}")
st.dataframe(filtered[display_cols].reset_index(drop=True))

# --------------------------
# Player ranking badges (top 3)
# --------------------------
st.header("🏆 Top Players (Badges)")
if not ratings_df.empty:
    ratings_df_sorted = ratings_df.copy()
    ratings_df_sorted["rating"] = pd.to_numeric(ratings_df_sorted["rating"], errors="coerce").fillna(1500.0)
    top3 = ratings_df_sorted.sort_values("rating", ascending=False).head(3).reset_index(drop=True)
else:
    top3 = pd.DataFrame(columns=["player", "rating", "wins"])

cols = st.columns(3)
badges = ["🥇 #1", "🥈 #2", "🥉 #3"]
for i in range(3):
    if i < len(top3):
        row = top3.loc[i]
        name = row.get("player", "")
        rating_val = float(row.get("rating", 1500.0))
        wins_val = int(row.get("wins", 0)) if "wins" in row.index else ""
        cols[i].metric(label=f"{badges[i]}  {name}", value=f"{rating_val:.2f}", delta=f"W:{wins_val}")
    else:
        cols[i].write("—")

# --------------------------
# Team (pair) win/loss analysis
# --------------------------
st.header("🧩 Team (Pair) Analysis")

# Create pair keys (sorted names so A&B == B&A)
def pair_key(a, b):
    a = normalize(a); b = normalize(b)
    if not a and not b:
        return None
    return " & ".join(sorted([a, b]))

pair_dict = {}
for _, r in matches_df.iterrows():
    pA = pair_key(r["playerA1"], r["playerA2"])
    pB = pair_key(r["playerB1"], r["playerB2"])
    if pA:
        pair_dict.setdefault(pA, {"matches": 0, "wins": 0, "losses": 0})
    if pB:
        pair_dict.setdefault(pB, {"matches": 0, "wins": 0, "losses": 0})

    if pA:
        pair_dict[pA]["matches"] += 1
    if pB:
        pair_dict[pB]["matches"] += 1

    if int(r["scoreA"]) > int(r["scoreB"]):
        if pA:
            pair_dict[pA]["wins"] += 1
        if pB:
            pair_dict[pB]["losses"] += 1
    else:
        if pB:
            pair_dict[pB]["wins"] += 1
        if pA:
            pair_dict[pA]["losses"] += 1

pairs_rows = []
for team, v in pair_dict.items():
    matches_n = v.get("matches", 0)
    wins_n = v.get("wins", 0)
    losses_n = v.get("losses", 0)
    win_pct = round((wins_n / matches_n * 100) if matches_n else 0.0, 1)
    pairs_rows.append({"team": team, "matches": matches_n, "wins": wins_n, "losses": losses_n, "win_pct": win_pct})

pairs_df = pd.DataFrame(pairs_rows)
if not pairs_df.empty and "win_pct" in pairs_df.columns:
    pairs_df = pairs_df.sort_values("win_pct", ascending=False).reset_index(drop=True)
else:
    # ensure columns exist even if empty
    pairs_df = pairs_df.reindex(columns=["team", "matches", "wins", "losses", "win_pct"])

st.subheader("Top Teams by Win %")
st.dataframe(pairs_df.head(12))

# --------------------------
# Rating trend (replay matches chronologically)
# --------------------------
st.header("📈 Rating Trend")

players_list = sorted(ratings_df["player"].unique()) if not ratings_df.empty else []
selected = st.multiselect("Select players to plot (replay matches in chronological order):", players_list, default=players_list[:3])

# Prepare matches sorted by date_parsed
matches_sorted = matches_df.sort_values("date_parsed").reset_index(drop=True)

def build_timeline(selected_players):
    if len(matches_sorted) == 0:
        # no matches -> return initial points
        today = pd.to_datetime(datetime.date.today())
        rows = []
        for p in selected_players:
            rows.append({"date": today, "player": p, "rating": 1500.0})
        return pd.DataFrame(rows)

    current = {}
    rows = []
    for _, match in matches_sorted.iterrows():
        A1 = normalize(match["playerA1"]); A2 = normalize(match["playerA2"])
        B1 = normalize(match["playerB1"]); B2 = normalize(match["playerB2"])
        scA = int(match["scoreA"]); scB = int(match["scoreB"])
        for p in [A1, A2, B1, B2]:
            if p and p not in current:
                current[p] = 1500.0
        # apply match
        current = update_elo(A1, A2, B1, B2, scA, scB, current)
        # record ratings for selected players at this point
        for p in selected_players:
            rows.append({"date": match["date_parsed"], "player": p, "rating": float(current.get(p, 1500.0))})
    if not rows:
        # fallback single point per selected player
        today = pd.to_datetime(datetime.date.today())
        rows = [{"date": today, "player": p, "rating": 1500.0} for p in selected_players]
    df_t = pd.DataFrame(rows)
    df_t["date"] = pd.to_datetime(df_t["date"])
    return df_t

if selected:
    timeline_df = build_timeline(selected)
    chart = alt.Chart(timeline_df).mark_line(point=True).encode(
        x=alt.X("date:T", title="Date"),
        y=alt.Y("rating:Q", title="Rating"),
        color="player:N",
        tooltip=["player:N", alt.Tooltip("rating:Q", format=".2f"), "date:T"]
    ).interactive()
    st.altair_chart(chart, use_container_width=True)
else:
    st.info("Choose 1+ players to see rating trends.")

# --------------------------
# Prediction section
# --------------------------
st.header("🔮 Predict Match Outcome")
pc1, pc2 = st.columns(2)
pA1_in = normalize(pc1.text_input("Team A - P1"))
pA2_in = normalize(pc1.text_input("Team A - P2"))
pB1_in = normalize(pc2.text_input("Team B - P1"))
pB2_in = normalize(pc2.text_input("Team B - P2"))

if st.button("Predict Win Probability"):
    if not all([pA1_in, pA2_in, pB1_in, pB2_in]):
        st.error("Please enter all 4 player names.")
    elif all(p in ratings for p in [pA1_in, pA2_in, pB1_in, pB2_in]):
        prob = predict_win_probability(ratings, pA1_in, pA2_in, pB1_in, pB2_in)
        st.success(f"Team A win probability: **{prob*100:.2f}%**")
    else:
        st.error("One or more players are missing ratings. Add matches to build ratings.")

# --------------------------
# Footer: quick summary
# --------------------------
st.markdown("---")
st.write(f"Total matches stored: **{len(matches_df)}**")
st.write(f"Players with ratings: **{len(ratings)}**")

