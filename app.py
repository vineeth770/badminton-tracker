# app.py
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

# ------------------------
# Helpers
# ------------------------
def normalize(name):
    """Normalize player names (case-insensitive)."""
    if not isinstance(name, str):
        return ""
    return name.strip().title()

def safe_int_from_text(s):
    """Convert text to int; empty or invalid -> 0."""
    if s is None:
        return 0
    s = str(s).strip()
    if s == "":
        return 0
    try:
        return int(float(s))
    except:
        return 0

# ------------------------
# GitHub helpers (read/write CSV)
# ------------------------
GITHUB_TOKEN = st.secrets["GITHUB_TOKEN"]
OWNER = st.secrets["REPO_OWNER"]
REPO = st.secrets["REPO_NAME"]
MATCHES_PATH = st.secrets["MATCHES_CSV"]
RATINGS_PATH = st.secrets["RATINGS_CSV"]
API_BASE = f"https://api.github.com/repos/{OWNER}/{REPO}/contents"

headers = {"Authorization": f"token {GITHUB_TOKEN}"}

def github_get_csv(path, default_columns=None):
    """
    Return (df, sha). If file not found, return empty df with default_columns and sha=None.
    """
    url = f"{API_BASE}/{path}"
    resp = requests.get(url, headers=headers)
    if resp.status_code == 200:
        info = resp.json()
        content_b64 = info.get("content", "")
        content = base64.b64decode(content_b64).decode("utf-8")
        df = pd.read_csv(StringIO(content))
        return df, info.get("sha")
    elif resp.status_code == 404:
        # file missing — create an empty DF with columns
        if default_columns is None:
            df = pd.DataFrame()
        else:
            df = pd.DataFrame(columns=default_columns)
        return df, None
    else:
        st.error(f"GitHub API error {resp.status_code} while reading {path}: {resp.text}")
        st.stop()

def github_put_csv(path, df, sha=None, message="update csv"):
    """
    Create or update a file in the repo. If sha is None, GitHub treats it as creating a new file.
    Returns new sha.
    """
    url = f"{API_BASE}/{path}"
    content = df.to_csv(index=False)
    encoded = base64.b64encode(content.encode()).decode()
    payload = {"message": message, "content": encoded}
    if sha:
        payload["sha"] = sha
    resp = requests.put(url, headers=headers, json=payload)
    if resp.status_code in (200, 201):
        return resp.json().get("content", {}).get("sha")
    else:
        st.error(f"GitHub API error {resp.status_code} while writing {path}: {resp.text}")
        st.stop()

# ------------------------
# Load CSVs (safe defaults)
# ------------------------
MATCHES_COLS = ["date", "playerA1", "playerA2", "playerB1", "playerB2", "scoreA", "scoreB"]
RATINGS_COLS = ["player", "rating", "wins", "losses", "matches"]

matches, matches_sha = github_get_csv(MATCHES_PATH, default_columns=MATCHES_COLS)
ratings_df, ratings_sha = github_get_csv(RATINGS_PATH, default_columns=RATINGS_COLS)

# Ensure columns exist
for c in MATCHES_COLS:
    if c not in matches.columns:
        matches[c] = ""

for c in RATINGS_COLS:
    if c not in ratings_df.columns:
        ratings_df[c] = []

# Normalize string columns and convert numeric columns
for c in ["playerA1","playerA2","playerB1","playerB2","date"]:
    if c in matches.columns:
        matches[c] = matches[c].fillna("").astype(str).apply(normalize)

# Keep score columns as original text for display (empty when blank)
# Maintain numeric helpers for calculations
matches["scoreA_num"] = pd.to_numeric(matches.get("scoreA", ""), errors="coerce").fillna(0).astype(int)
matches["scoreB_num"] = pd.to_numeric(matches.get("scoreB", ""), errors="coerce").fillna(0).astype(int)

# Make sure ratings_df numeric types (if file existed)
if not ratings_df.empty:
    ratings_df["rating"] = pd.to_numeric(ratings_df["rating"], errors="coerce").fillna(1500.0)
    ratings_df["wins"] = pd.to_numeric(ratings_df["wins"], errors="coerce").fillna(0).astype(int)
    ratings_df["losses"] = pd.to_numeric(ratings_df["losses"], errors="coerce").fillna(0).astype(int)
    ratings_df["matches"] = pd.to_numeric(ratings_df["matches"], errors="coerce").fillna(0).astype(int)

# Build ratings dict and player_stats from ratings_df (for prediction UI)
ratings = {}
player_stats = {}
if not ratings_df.empty:
    for _, r in ratings_df.iterrows():
        p = normalize(r["player"])
        ratings[p] = float(r["rating"])
        player_stats[p] = {
            "wins": int(r["wins"]),
            "losses": int(r["losses"]),
            "matches": int(r["matches"])
        }

# ------------------------
# UI: Add new match
# ------------------------
st.header("➕ Add New Match")

# Use clear_on_submit=True so inputs (including score fields) are cleared after save
with st.form("add_match", clear_on_submit=True):
    left, right = st.columns([1,1])
    A1 = normalize(left.text_input("Team A - Player 1", value=""))
    A2 = normalize(right.text_input("Team A - Player 2", value=""))
    B1 = normalize(left.text_input("Team B - Player 1", value=""))
    B2 = normalize(right.text_input("Team B - Player 2", value=""))

    # Score inputs are text so they start empty and remain empty until user enters
    sA_text = left.text_input("Score A (leave empty = 0)", value="", key="sa_input")
    sB_text = right.text_input("Score B (leave empty = 0)", value="", key="sb_input")

    # Date input default: today's date (must be datetime.date)
    match_date = right.date_input("Match date", value=datetime.date.today())

    add_clicked = st.form_submit_button("Add match")

if add_clicked:
    # Validate players (at least one non-empty name provided)
    if not any([A1, A2, B1, B2]):
        st.error("Please enter at least one player name.")
    else:
        # Convert score strings to ints (empty -> 0)
        sA = safe_int_from_text(sA_text)
        sB = safe_int_from_text(sB_text)

        # Prepare new row: keep score text empty if user left blank (so UI shows empty)
        new_row = {
            "date": match_date.strftime("%Y-%m-%d"),
            "playerA1": A1, "playerA2": A2,
            "playerB1": B1, "playerB2": B2,
            "scoreA": "" if str(sA_text).strip() == "" else str(sA),
            "scoreB": "" if str(sB_text).strip() == "" else str(sB)
        }

        # Append to matches DF
        matches = pd.concat([matches, pd.DataFrame([new_row])], ignore_index=True)

        # update numeric helper columns before saving
        matches["scoreA_num"] = pd.to_numeric(matches.get("scoreA", ""), errors="coerce").fillna(0).astype(int)
        matches["scoreB_num"] = pd.to_numeric(matches.get("scoreB", ""), errors="coerce").fillna(0).astype(int)

        # Save matches.csv (write only the canonical MATCHES_COLS order)
        new_matches_sha = github_put_csv(MATCHES_PATH, matches[MATCHES_COLS], matches_sha, "Add match")
        matches_sha = new_matches_sha  # update sha

        # Recompute ratings + player stats by replaying matches in chronological order
        ratings_replay = {}
        stats_replay = {}

        matches_sorted = matches.copy()
        matches_sorted["date_parsed"] = pd.to_datetime(matches_sorted["date"], errors="coerce")
        # if no valid dates, keep file order
        if matches_sorted["date_parsed"].isnull().all():
            matches_sorted = matches_sorted.reset_index(drop=True)
        else:
            matches_sorted = matches_sorted.sort_values("date_parsed").reset_index(drop=True)

        for _, row in matches_sorted.iterrows():
            pA1 = normalize(row.get("playerA1", ""))
            pA2 = normalize(row.get("playerA2", ""))
            pB1 = normalize(row.get("playerB1", ""))
            pB2 = normalize(row.get("playerB2", ""))
            scA = safe_int_from_text(row.get("scoreA", ""))
            scB = safe_int_from_text(row.get("scoreB", ""))

            # initialize players
            for p in [pA1, pA2, pB1, pB2]:
                if p and p not in ratings_replay:
                    ratings_replay[p] = 1500.0
                if p and p not in stats_replay:
                    stats_replay[p] = {"wins": 0, "losses": 0, "matches": 0}

            # increment matches
            for p in [pA1, pA2, pB1, pB2]:
                if p:
                    stats_replay[p]["matches"] += 1

            # assign wins/losses
            if scA > scB:
                if pA1: stats_replay[pA1]["wins"] += 1
                if pA2: stats_replay[pA2]["wins"] += 1
                if pB1: stats_replay[pB1]["losses"] += 1
                if pB2: stats_replay[pB2]["losses"] += 1
            else:
                if pB1: stats_replay[pB1]["wins"] += 1
                if pB2: stats_replay[pB2]["wins"] += 1
                if pA1: stats_replay[pA1]["losses"] += 1
                if pA2: stats_replay[pA2]["losses"] += 1

            # update elo
            ratings_replay = update_elo(pA1, pA2, pB1, pB2, scA, scB, ratings_replay)

        # Build ratings_df from replayed state
        ratings_rows = []
        for p, r in ratings_replay.items():
            st_w = stats_replay.get(p, {}).get("wins", 0)
            st_l = stats_replay.get(p, {}).get("losses", 0)
            st_m = stats_replay.get(p, {}).get("matches", 0)
            ratings_rows.append({
                "player": p,
                "rating": round(r, 2),
                "wins": int(st_w),
                "losses": int(st_l),
                "matches": int(st_m)
            })

        ratings_df = pd.DataFrame(ratings_rows).sort_values("rating", ascending=False).reset_index(drop=True)

        # Save ratings.csv (update or create)
        new_ratings_sha = github_put_csv(RATINGS_PATH, ratings_df[RATINGS_COLS], ratings_sha, "Update ratings")
        ratings_sha = new_ratings_sha

        # Update in-memory convenience structures
        ratings = {row["player"]: float(row["rating"]) for _, row in ratings_df.iterrows()}
        player_stats = {row["player"]: {"wins": int(row["wins"]), "losses": int(row["losses"]), "matches": int(row["matches"])} for _, row in ratings_df.iterrows()}

        st.success("Match added and ratings updated. (Score inputs were cleared.)")

# ------------------------
# Prepare matches for display (safe)
# ------------------------
for c in MATCHES_COLS:
    if c not in matches.columns:
        matches[c] = ""

matches = matches.reset_index(drop=True)
matches["scoreA_num"] = pd.to_numeric(matches.get("scoreA", ""), errors="coerce").fillna(0).astype(int)
matches["scoreB_num"] = pd.to_numeric(matches.get("scoreB", ""), errors="coerce").fillna(0).astype(int)

# safe parse date column into date_parsed
def parse_date_column(df):
    if "date" in df.columns:
        df["date_parsed"] = pd.to_datetime(df["date"], errors="coerce")
        # if all invalid => fallback to synthetic dates from index
        if df["date_parsed"].isna().all():
            df["date_parsed"] = pd.to_datetime("1970-01-01") + pd.to_timedelta(df.index, unit="D")
    else:
        df["date_parsed"] = pd.to_datetime("1970-01-01") + pd.to_timedelta(df.index, unit="D")
    return df

matches = parse_date_column(matches)

# ------------------------
# Filters and Match History display
# ------------------------
st.header("📜 Match History & Filters")

# ALWAYS ensure date_parsed column exists BEFORE filtering
if "date_parsed" not in matches.columns:
    matches["date_parsed"] = pd.to_datetime(matches["date"], errors="coerce")
    matches["date_parsed"] = matches["date_parsed"].fillna(
        pd.Timestamp("1970-01-01") + pd.to_timedelta(matches.index, unit="D")
    )

# Safe defaults
if matches["date_parsed"].isnull().all():
    matches["date_parsed"] = pd.Timestamp("1970-01-01") + pd.to_timedelta(matches.index, unit="D")

min_date = matches["date_parsed"].min().date()
max_date = matches["date_parsed"].max().date()

col1, col2 = st.columns(2)
start_date = col1.date_input("From", min_date)
end_date = col2.date_input("To", max_date)

start_dt = pd.to_datetime(start_date)
end_dt = pd.to_datetime(end_date) + pd.Timedelta(days=1) - pd.Timedelta(seconds=1)

# Filtering
filtered = matches[
    (matches["date_parsed"] >= start_dt) &
    (matches["date_parsed"] <= end_dt)
].copy()

# Columns that actually exist
cols_to_show = [c for c in ["date","playerA1","playerA2","playerB1","playerB2","scoreA","scoreB"]
                if c in filtered.columns]

# Only sort if column exists
if "date_parsed" in filtered.columns:
    filtered = filtered.sort_values("date_parsed", ascending=False)

st.subheader(f"Showing {len(filtered)} matches from {start_date} to {end_date}")
st.dataframe(filtered[cols_to_show].reset_index(drop=True))

# Only sort by date_parsed if it exists in filtered
if "date_parsed" in filtered.columns:
    st.dataframe(filtered[cols_to_show].sort_values("date_parsed", ascending=False).reset_index(drop=True))
else:
    st.dataframe(filtered[cols_to_show].reset_index(drop=True))

# ------------------------
# Player individual stats (derived from ratings_df)
# ------------------------
st.header("📊 Player Statistics (individual)")

if ratings_df.empty:
    st.info("No player stats yet.")
else:
    stats_df = ratings_df.copy()
    for col in ["rating","wins","losses","matches"]:
        stats_df[col] = pd.to_numeric(stats_df[col], errors="coerce").fillna(0)
    stats_df["Win %"] = ((stats_df["wins"] / stats_df["matches"]) * 100).replace([np.inf, -np.inf], 0).fillna(0).round(1)
    stats_df = stats_df[["player","rating","matches","wins","losses","Win %"]].sort_values("rating", ascending=False)
    st.dataframe(stats_df.reset_index(drop=True))

# ------------------------
# Top 3 badges
# ------------------------
st.header("🏆 Top Players")
if not ratings_df.empty:
    top = ratings_df.sort_values("rating", ascending=False).reset_index(drop=True).head(3)
else:
    top = pd.DataFrame(columns=["player","rating","wins"])

cols = st.columns(3)
medals = ["🥇 #1", "🥈 #2", "🥉 #3"]
for i in range(3):
    if i < len(top):
        row = top.loc[i]
        cols[i].metric(f"{medals[i]} {row['player']}", f"{float(row['rating']):.2f}", delta=f"W:{int(row['wins'])}")
    else:
        cols[i].write("—")

# ------------------------
# Rating trend (replay matches)
# ------------------------
st.header("📈 Rating Trend (replay matches)")

players_all = sorted(ratings_df["player"].unique()) if not ratings_df.empty else []
selected = st.multiselect("Select players to plot", players_all, default=players_all[:3])

def build_rating_timeline(selected_players):
    timeline_rows = []
    if matches.empty:
        return pd.DataFrame(columns=["date","player","rating"])
    matches_sorted = matches.copy().sort_values("date_parsed").reset_index(drop=True)
    current = {}
    for _, row in matches_sorted.iterrows():
        pA1 = normalize(row["playerA1"]); pA2 = normalize(row["playerA2"])
        pB1 = normalize(row["playerB1"]); pB2 = normalize(row["playerB2"])
        sA = safe_int_from_text(row.get("scoreA",""))
        sB = safe_int_from_text(row.get("scoreB",""))
        for p in [pA1, pA2, pB1, pB2]:
            if p and p not in current:
                current[p] = 1500.0
        current = update_elo(pA1, pA2, pB1, pB2, sA, sB, current)
        for p in selected_players:
            timeline_rows.append({"date": row["date_parsed"], "player": p, "rating": current.get(p, 1500.0)})
    if not timeline_rows:
        return pd.DataFrame(columns=["date","player","rating"])
    df_t = pd.DataFrame(timeline_rows).sort_values(["player","date"]).reset_index(drop=True)
    return df_t

if selected:
    df_t = build_rating_timeline(selected)
    if not df_t.empty:
        chart = alt.Chart(df_t).mark_line(point=True).encode(
            x=alt.X("date:T", title="Date"),
            y=alt.Y("rating:Q", title="Rating"),
            color="player:N",
            tooltip=["player:N", alt.Tooltip("rating:Q", format=".2f"), "date:T"]
        ).interactive()
        st.altair_chart(chart, use_container_width=True)
    else:
        st.info("No timeline data to show yet.")
else:
    st.info("Pick players to plot rating trend.")

# ------------------------
# Prediction
# ------------------------
st.header("🔮 Predict Match Outcome")
pcol1, pcol2 = st.columns(2)
pA1 = normalize(pcol1.text_input("Team A - P1", value=""))
pA2 = normalize(pcol1.text_input("Team A - P2", value=""))
pB1 = normalize(pcol2.text_input("Team B - P1", value=""))
pB2 = normalize(pcol2.text_input("Team B - P2", value=""))

if st.button("Predict"):
    if all(x for x in [pA1,pA2,pB1,pB2]) and all(x in ratings for x in [pA1,pA2,pB1,pB2]):
        prob = predict_win_probability(ratings, pA1, pA2, pB1, pB2)
        st.success(f"Team A win probability: **{prob*100:.2f}%**")
    else:
        st.error("Make sure all four players exist and have ratings.")

st.write("App synced with GitHub CSV files.")
