# app.py - Badminton Tracker (GitHub CSV storage + trends + teams + filters)
import streamlit as st
import pandas as pd
import numpy as np
import requests
import base64
from io import StringIO
import altair as alt
import datetime
from elo import update_elo, predict_win_probability

st.set_page_config(page_title="Badminton Tracker", layout="wide")
st.title("🏸 Badminton Doubles Tracker")

# -------------------------
# Config / Helpers
# -------------------------
def normalize(name):
    if not isinstance(name, str):
        return ""
    return name.strip().title()

def ensure_col(df, col, default=""):
    if col not in df.columns:
        df[col] = default
    return df

def to_int_safe(x, default=0):
    try:
        return int(float(x))
    except:
        return default

# -------------------------
# GitHub helpers
# -------------------------
def github_get_file(path):
    """Return (content_str, sha) or (None, None) if not found or error"""
    url = f"https://api.github.com/repos/{st.secrets['REPO_OWNER']}/{st.secrets['REPO_NAME']}/contents/{path}"
    headers = {"Authorization": f"token {st.secrets['GITHUB_TOKEN']}"}
    r = requests.get(url, headers=headers)
    if r.status_code == 200:
        j = r.json()
        content = base64.b64decode(j["content"]).decode("utf-8")
        return content, j.get("sha")
    elif r.status_code == 404:
        return None, None
    else:
        # Unexpected error
        st.error(f"GitHub GET error: {r.status_code} {r.text[:200]}")
        st.stop()
        return None, None

def github_put_file(path, df, sha=None, message="update csv"):
    """Create or update a file at path with df.to_csv(). `sha` can be None for new files."""
    url = f"https://api.github.com/repos/{st.secrets['REPO_OWNER']}/{st.secrets['REPO_NAME']}/contents/{path}"
    headers = {"Authorization": f"token {st.secrets['GITHUB_TOKEN']}"}
    csv_bytes = df.to_csv(index=False).encode("utf-8")
    encoded = base64.b64encode(csv_bytes).decode("utf-8")
    payload = {"message": message, "content": encoded}
    if sha:
        payload["sha"] = sha
    r = requests.put(url, headers=headers, json=payload)
    if r.status_code in (200, 201):
        return r.json().get("content", {}).get("sha")
    else:
        st.error(f"GitHub PUT error ({r.status_code}): {r.text[:500]}")
        st.stop()
        return None

# -------------------------
# Load CSVs (safe)
# -------------------------
def load_csv_path(path, default_columns):
    content, sha = github_get_file(path)
    if content is None:
        # create empty df with default columns
        df = pd.DataFrame(columns=default_columns)
        return df, None
    else:
        df = pd.read_csv(StringIO(content))
        # ensure columns exist
        for c in default_columns:
            if c not in df.columns:
                df[c] = ""
        return df, sha

MATCHES_COLS = ["date", "playerA1", "playerA2", "playerB1", "playerB2", "scoreA", "scoreB"]
RATINGS_COLS = ["player", "rating", "wins", "losses", "matches"]

matches, matches_sha = load_csv_path(st.secrets["MATCHES_CSV"], MATCHES_COLS)
ratings_df, ratings_sha = load_csv_path(st.secrets["RATINGS_CSV"], RATINGS_COLS)

# -------------------------
# Normalize & coerce types
# -------------------------
# Trim whitespace and normalize player names in matches
for col in ["playerA1", "playerA2", "playerB1", "playerB2"]:
    if col in matches.columns:
        matches[col] = matches[col].fillna("").astype(str).apply(normalize)
    else:
        matches[col] = ""

# Ensure date column exists; older CSVs may not have it
if "date" not in matches.columns:
    matches["date"] = ""

# Coerce scores to int safe
matches["scoreA"] = matches.get("scoreA", 0).apply(lambda x: to_int_safe(x, 0))
matches["scoreB"] = matches.get("scoreB", 0).apply(lambda x: to_int_safe(x, 0))

# Parse dates if present; otherwise create synthetic date based on index
def ensure_dates(df):
    if "date" in df.columns and df["date"].notna().any():
        df["date_parsed"] = pd.to_datetime(df["date"], errors="coerce")
        # fallback for rows with NaT: fill with synthetic increasing days
        mask = df["date_parsed"].isna()
        if mask.any():
            # take last valid date or today
            base = df["date_parsed"].dropna().max()
            if pd.isna(base):
                base = pd.to_datetime(datetime.date.today())
            df.loc[mask, "date_parsed"] = [base + pd.Timedelta(days=i+1) for i in range(mask.sum())]
    else:
        # create synthetic dates
        df = df.reset_index(drop=True)
        df["date_parsed"] = pd.to_datetime("1970-01-01") + pd.to_timedelta(df.index, unit="D")
    return df

matches = ensure_dates(matches)

# Ratings DF numeric coercion
if not ratings_df.empty:
    ratings_df["rating"] = pd.to_numeric(ratings_df.get("rating", 1500), errors="coerce").fillna(1500)
    ratings_df["wins"] = pd.to_numeric(ratings_df.get("wins", 0), errors="coerce").fillna(0).astype(int)
    ratings_df["losses"] = pd.to_numeric(ratings_df.get("losses", 0), errors="coerce").fillna(0).astype(int)
    ratings_df["matches"] = pd.to_numeric(ratings_df.get("matches", 0), errors="coerce").fillna(0).astype(int)
else:
    ratings_df = pd.DataFrame(columns=RATINGS_COLS)

# Build ratings dict and player_stats from ratings_df
ratings = {}
player_stats = {}
for _, r in ratings_df.iterrows():
    p = normalize(r["player"])
    ratings[p] = float(r["rating"])
    player_stats[p] = {
        "wins": int(r.get("wins", 0)),
        "losses": int(r.get("losses", 0)),
        "matches": int(r.get("matches", 0))
    }

# -------------------------
# UI: Add match
# -------------------------
st.header("➕ Add New Match")
with st.form("add_match"):
    c1, c2 = st.columns(2)
    a1 = normalize(c1.text_input("Team A - Player 1"))
    a2 = normalize(c2.text_input("Team A - Player 2"))
    b1 = normalize(c1.text_input("Team B - Player 1"))
    b2 = normalize(c2.text_input("Team B - Player 2"))
    sA = c1.number_input("Score A", min_value=0, value=21)
    sB = c2.number_input("Score B", min_value=0, value=19)
    match_date = c2.date_input("Match date", value=datetime.date.today())
    add_clicked = st.form_submit_button("Add match")

if add_clicked:
    # Validate input
    if not (a1 and a2 and b1 and b2):
        st.error("Please provide all four player names.")
    else:
        new = {
            "date": str(match_date),
            "playerA1": a1, "playerA2": a2,
            "playerB1": b1, "playerB2": b2,
            "scoreA": int(sA), "scoreB": int(sB)
        }
        matches = pd.concat([matches, pd.DataFrame([new])], ignore_index=True)
        # Save matches: fetch latest sha before updating to reduce conflicts
        latest_content, latest_sha = github_get_file(st.secrets["MATCHES_CSV"])
        # If file existed, use latest sha; else create new
        _sha_to_use = latest_sha or matches_sha
        matches_sha = github_put_file = None
        try:
            new_sha = github_put_file = github_put_file  # dummy to show later usage
        except:
            pass
        # Put using our helper
        try:
            new_sha = github_put_file(st.secrets["MATCHES_CSV"], matches, _sha_to_use, message="Add match")
            matches_sha = new_sha
        except Exception as e:
            st.error(f"Failed saving matches: {e}")
            st.stop()
        st.success("Match added and saved to GitHub.")

        # Ensure players present in ratings/stats
        for p in [a1, a2, b1, b2]:
            if p and p not in ratings:
                ratings[p] = 1500.0
            if p and p not in player_stats:
                player_stats[p] = {"wins": 0, "losses": 0, "matches": 0}

        # Update player_stats counts & wins/losses
        for p in [a1, a2, b1, b2]:
            if p:
                player_stats[p]["matches"] += 1
        if sA > sB:
            player_stats[a1]["wins"] += 1
            player_stats[a2]["wins"] += 1
            player_stats[b1]["losses"] += 1
            player_stats[b2]["losses"] += 1
        else:
            player_stats[b1]["wins"] += 1
            player_stats[b2]["wins"] += 1
            player_stats[a1]["losses"] += 1
            player_stats[a2]["losses"] += 1

        # Recompute ratings by replaying matches in chronological order
        ratings = {}
        matches_sorted = matches.sort_values("date_parsed").reset_index(drop=True)
        for _, row in matches_sorted.iterrows():
            pA1 = normalize(row.get("playerA1", ""))
            pA2 = normalize(row.get("playerA2", ""))
            pB1 = normalize(row.get("playerB1", ""))
            pB2 = normalize(row.get("playerB2", ""))
            scA = int(row.get("scoreA", 0))
            scB = int(row.get("scoreB", 0))
            ratings = update_elo(pA1, pA2, pB1, pB2, scA, scB, ratings)

        # Build ratings_df
        ratings_df = pd.DataFrame([
            {"player": p,
             "rating": round(ratings.get(p, 1500.0), 2),
             "wins": player_stats.get(p, {}).get("wins", 0),
             "losses": player_stats.get(p, {}).get("losses", 0),
             "matches": player_stats.get(p, {}).get("matches", 0)}
            for p in sorted(ratings.keys())
        ])

        # Save ratings_df to GitHub
        latest_r_content, latest_r_sha = github_get_file(st.secrets["RATINGS_CSV"])
        _r_sha = latest_r_sha or ratings_sha
        try:
            ratings_sha = github_put_file(st.secrets["RATINGS_CSV"], ratings_df, _r_sha, message="Update ratings")
        except Exception as e:
            st.error(f"Failed saving ratings: {e}")
            st.stop()

# -------------------------
# Filter / Match history view
# -------------------------
st.header("📜 Match History & Filters")
# ensure trimmed strings
matches = matches.fillna("").applymap(lambda x: x.strip() if isinstance(x, str) else x)
matches = ensure_dates(matches)

min_date = matches["date_parsed"].min()
max_date = matches["date_parsed"].max()
if pd.isna(min_date): min_date = datetime.date.today()
if pd.isna(max_date): max_date = datetime.date.today()

colf1, colf2, colf3 = st.columns([2,2,1])
with colf1:
    start = st.date_input("From", value=min_date.date() if hasattr(min_date, "date") else min_date)
with colf2:
    end = st.date_input("To", value=max_date.date() if hasattr(max_date, "date") else max_date)
with colf3:
    apply_filter = st.button("Apply")

start_dt = pd.to_datetime(start)
end_dt = pd.to_datetime(end) + pd.Timedelta(days=1) - pd.Timedelta(seconds=1)

filtered = matches[(matches["date_parsed"] >= start_dt) & (matches["date_parsed"] <= end_dt)]

# Choose columns that exist
cols_to_show = [c for c in ["date", "playerA1", "playerA2", "playerB1", "playerB2", "scoreA", "scoreB"] if c in filtered.columns]
st.subheader(f"Showing {len(filtered)} matches between {start} and {end}")
st.dataframe(filtered[cols_to_show])

# -------------------------
# Player ranking badges
# -------------------------
st.header("🏆 Top Players")
if ratings_df.empty:
    st.info("No players yet.")
else:
    ratings_df["rating"] = pd.to_numeric(ratings_df["rating"], errors="coerce").fillna(1500)
    top3 = ratings_df.sort_values("rating", ascending=False).head(3).reset_index(drop=True)
    cols = st.columns(3)
    medals = ["🥇 #1", "🥈 #2", "🥉 #3"]
    for i in range(3):
        if i < len(top3):
            player = top3.loc[i, "player"]
            rating = top3.loc[i, "rating"]
            wins = top3.loc[i, "wins"] if "wins" in top3.columns else ""
            cols[i].metric(label=f"{medals[i]}  {player}", value=f"{rating:.2f}", delta=f"W:{wins}")
        else:
            cols[i].write("—")

# -------------------------
# Team (pair) analysis
# -------------------------
st.header("🧩 Team (Pair) Analysis")
def pair_key(p1, p2):
    return tuple(sorted([normalize(p1), normalize(p2)]))

pair_stats = {}
for _, r in matches.iterrows():
    pA = pair_key(r.get("playerA1", ""), r.get("playerA2", ""))
    pB = pair_key(r.get("playerB1", ""), r.get("playerB2", ""))
    if pA not in pair_stats:
        pair_stats[pA] = {"matches": 0, "wins": 0, "losses": 0}
    if pB not in pair_stats:
        pair_stats[pB] = {"matches": 0, "wins": 0, "losses": 0}
    pair_stats[pA]["matches"] += 1
    pair_stats[pB]["matches"] += 1
    if int(r.get("scoreA", 0)) > int(r.get("scoreB", 0)):
        pair_stats[pA]["wins"] += 1
        pair_stats[pB]["losses"] += 1
    else:
        pair_stats[pB]["wins"] += 1
        pair_stats[pA]["losses"] += 1

pairs_list = []
for pair, s in pair_stats.items():
    pairs_list.append({
        "team": " & ".join(pair) if pair[0] and pair[1] else " & ".join(pair).strip(" & "),
        "matches": s["matches"],
        "wins": s["wins"],
        "losses": s["losses"],
        "win_pct": round((s["wins"] / s["matches"] * 100) if s["matches"] else 0, 1)
    })
pairs_df = pd.DataFrame(pairs_list).sort_values("win_pct", ascending=False) if pairs_list else pd.DataFrame(columns=["team","matches","wins","losses","win_pct"])
st.subheader("Top Teams by Win %")
st.dataframe(pairs_df.head(10))

# -------------------------
# Rating trend graph (replay matches)
# -------------------------
st.header("📈 Rating Trend")
players_all = sorted(list(ratings_df["player"].unique())) if not ratings_df.empty else []
selected_players = st.multiselect("Select players to plot (replay matches chronologically)", players_all, default=players_all[:3] if players_all else [])

matches_sorted = matches.sort_values("date_parsed").reset_index(drop=True)

def compute_rating_timeline(selected_players):
    current = {}
    timeline_rows = []
    # initialize players to 1500
    for p in selected_players:
        current[p] = 1500.0
    # replay matches
    for _, row in matches_sorted.iterrows():
        pA1 = normalize(row.get("playerA1", "")); pA2 = normalize(row.get("playerA2", ""))
        pB1 = normalize(row.get("playerB1", "")); pB2 = normalize(row.get("playerB2", ""))
        scA = int(row.get("scoreA", 0)); scB = int(row.get("scoreB", 0))
        # ensure all players in current
        for p in [pA1, pA2, pB1, pB2]:
            if p and p not in current:
                current[p] = 1500.0
        # update ratings
        current = update_elo(pA1, pA2, pB1, pB2, scA, scB, current)
        # append snapshot for selected players
        for p in selected_players:
            timeline_rows.append({"date": row["date_parsed"], "player": p, "rating": current.get(p, 1500.0)})
    if not timeline_rows and selected_players:
        today = pd.to_datetime(datetime.date.today())
        timeline_rows = [{"date": today, "player": p, "rating": 1500.0} for p in selected_players]
    return pd.DataFrame(timeline_rows)

if selected_players:
    timeline_df = compute_rating_timeline(selected_players)
    if not timeline_df.empty:
        chart = alt.Chart(timeline_df).mark_line(point=True).encode(
            x=alt.X("date:T", title="Date"),
            y=alt.Y("rating:Q", title="Rating"),
            color="player:N",
            tooltip=["player:N", alt.Tooltip("rating:Q", format=".2f"), "date:T"]
        ).interactive()
        st.altair_chart(chart, use_container_width=True)
    else:
        st.info("No timeline data yet.")
else:
    st.info("Pick players to plot rating trends.")

# -------------------------
# Prediction
# -------------------------
st.header("🔮 Predict Match Outcome")
pc1, pc2 = st.columns(2)
pA1 = normalize(pc1.text_input("Team A - P1"))
pA2 = normalize(pc1.text_input("Team A - P2"))
pB1 = normalize(pc2.text_input("Team B - P1"))
pB2 = normalize(pc2.text_input("Team B - P2"))

if st.button("Predict Win Probability"):
    if all(x in ratings for x in [pA1, pA2, pB1, pB2]):
        prob = predict_win_probability(ratings, pA1, pA2, pB1, pB2)
        st.success(f"Team A win probability: **{prob*100:.2f}%**")
    else:
        st.error("One or more players don't have ratings yet.")
