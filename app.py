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

# --------------------------
# Helper: normalize names & date parsing
# --------------------------
def normalize(name):
    if not isinstance(name, str):
        return ""
    return name.strip().title()

def parse_dates(df):
    # Ensure a 'date' column exists and is datetime
    if "date" in df.columns:
        df["date_parsed"] = pd.to_datetime(df["date"], errors="coerce")
    else:
        # No date column: create one from index assuming chronological order
        df = df.reset_index(drop=True)
        df["date_parsed"] = pd.to_datetime("1970-01-01") + pd.to_timedelta(df.index, unit="D")
    return df

# --------------------------
# GitHub read/write helpers
# --------------------------
def load_csv_from_github(path):
    url = f"https://api.github.com/repos/{st.secrets['REPO_OWNER']}/{st.secrets['REPO_NAME']}/contents/{path}"
    headers = {"Authorization": f"token {st.secrets['GITHUB_TOKEN']}"}
    res = requests.get(url, headers=headers)
    res.raise_for_status()
    response = res.json()
    content = base64.b64decode(response["content"]).decode("utf-8")
    df = pd.read_csv(StringIO(content))
    return df, response["sha"]

def save_csv_to_github(path, df, sha, message="update csv"):
    url = f"https://api.github.com/repos/{st.secrets['REPO_OWNER']}/{st.secrets['REPO_NAME']}/contents/{path}"
    headers = {"Authorization": f"token {st.secrets['GITHUB_TOKEN']}"}
    encoded = base64.b64encode(df.to_csv(index=False).encode()).decode()
    data = {"message": message, "content": encoded, "sha": sha}
    res = requests.put(url, headers=headers, json=data)
    res.raise_for_status()
    return res.json()

# --------------------------
# LOAD CSVs
# --------------------------
try:
    matches, matches_sha = load_csv_from_github(st.secrets["MATCHES_CSV"])
except Exception as e:
    st.error("Could not load matches.csv from GitHub. Check secrets and file path.")
    st.stop()

try:
    ratings_df, ratings_sha = load_csv_from_github(st.secrets["RATINGS_CSV"])
except Exception as e:
    # if ratings.csv missing or empty, create empty df
    ratings_df = pd.DataFrame(columns=["player", "rating", "wins", "losses", "matches"])
    ratings_sha = None

# Normalize existing data
for c in ["playerA1", "playerA2", "playerB1", "playerB2"]:
    if c in matches.columns:
        matches[c] = matches[c].fillna("").astype(str).apply(normalize)

# Ensure numeric columns exist
for col in ["scoreA", "scoreB"]:
    if col in matches.columns:
        matches[col] = pd.to_numeric(matches[col], errors="coerce").fillna(0).astype(int)
    else:
        matches[col] = 0

# If ratings_df has columns, coerce types
if not ratings_df.empty:
    ratings_df["rating"] = pd.to_numeric(ratings_df["rating"], errors="coerce").fillna(1500)
    ratings_df["wins"] = pd.to_numeric(ratings_df["wins"], errors="coerce").fillna(0).astype(int)
    ratings_df["losses"] = pd.to_numeric(ratings_df["losses"], errors="coerce").fillna(0).astype(int)
    ratings_df["matches"] = pd.to_numeric(ratings_df["matches"], errors="coerce").fillna(0).astype(int)

# Build ratings dict & player_stats map from ratings_df
ratings = {}
player_stats = {}
for _, r in ratings_df.iterrows():
    p = normalize(r["player"])
    ratings[p] = float(r["rating"])
    player_stats[p] = {"wins": int(r["wins"]), "losses": int(r["losses"]), "matches": int(r["matches"])}

# --------------------------
# UI: Add Match (with date)
# --------------------------
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
    submitted = st.form_submit_button("Add match")

if submitted:
    # Add row (include date)
    new_row = {
        "date": str(match_date),
        "playerA1": a1, "playerA2": a2,
        "playerB1": b1, "playerB2": b2,
        "scoreA": int(sA), "scoreB": int(sB)
    }
    matches = pd.concat([matches, pd.DataFrame([new_row])], ignore_index=True)
    # Save matches and refresh sha
    save_csv_to_github(st.secrets["MATCHES_CSV"], matches, matches_sha or "", "Add match")
    # Reload matches and sha to keep consistent
    matches, matches_sha = load_csv_from_github(st.secrets["MATCHES_CSV"])
    st.success("Match added.")

    # Ensure players exist in ratings / stats
    for p in [a1, a2, b1, b2]:
        if p and p not in ratings:
            ratings[p] = 1500.0
        if p and p not in player_stats:
            player_stats[p] = {"wins": 0, "losses": 0, "matches": 0}

    # Update counts & wins/losses
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

    # Update elo by replaying all matches (safe approach)
    # Recompute ratings from scratch so order consistent
    ratings = {}
    for _, row in matches.iterrows():
        pA1 = normalize(row.get("playerA1", ""))
        pA2 = normalize(row.get("playerA2", ""))
        pB1 = normalize(row.get("playerB1", ""))
        pB2 = normalize(row.get("playerB2", ""))
        scA = int(row.get("scoreA", 0))
        scB = int(row.get("scoreB", 0))
        ratings = update_elo(pA1, pA2, pB1, pB2, scA, scB, ratings)

    # Rebuild ratings_df from current ratings & player_stats
    ratings_df = pd.DataFrame([
        {
            "player": p,
            "rating": round(ratings.get(p, 1500.0), 2),
            "wins": player_stats.get(p, {}).get("wins", 0),
            "losses": player_stats.get(p, {}).get("losses", 0),
            "matches": player_stats.get(p, {}).get("matches", 0)
        } for p in sorted(ratings.keys())
    ])

    # Save ratings.csv (get latest sha first if needed)
    try:
        _, ratings_sha = load_csv_from_github(st.secrets["RATINGS_CSV"])
    except Exception:
        # If ratings file didn't exist initially, set sha to None
        ratings_sha = None
    save_csv_to_github(st.secrets["RATINGS_CSV"], ratings_df, ratings_sha or "", "Update ratings")
    # reload to refresh
    ratings_df, ratings_sha = load_csv_from_github(st.secrets["RATINGS_CSV"])

# --------------------------
# Process matches for display & filtering
# --------------------------
matches = matches.fillna("")
matches = matches.reset_index(drop=True)
matches = matches.applymap(lambda x: x.strip() if isinstance(x, str) else x)

matches = parse_dates(matches)

st.header("📜 Match History & Filters")

# Date filter
min_date = matches["date_parsed"].min()
max_date = matches["date_parsed"].max()
if pd.isna(min_date):
    min_date = datetime.date.today()
if pd.isna(max_date):
    max_date = datetime.date.today()

colf1, colf2, colf3 = st.columns([2,2,1])
with colf1:
    start = st.date_input("From", value=min_date.date() if hasattr(min_date, "date") else min_date)
with colf2:
    end = st.date_input("To", value=max_date.date() if hasattr(max_date, "date") else max_date)
with colf3:
    apply_filter = st.button("Apply")

# filter
start_dt = pd.to_datetime(start)
end_dt = pd.to_datetime(end) + pd.Timedelta(days=1) - pd.Timedelta(seconds=1)  # include end day

filtered = matches[(matches["date_parsed"] >= start_dt) & (matches["date_parsed"] <= end_dt)]

st.subheader(f"Showing {len(filtered)} matches between {start} and {end}")
st.dataframe(filtered[["date", "playerA1", "playerA2", "playerB1", "playerB2", "scoreA", "scoreB"]])

# --------------------------
# Player ranking badges
# --------------------------
st.header("🏆 Top Players")
# ensure numeric and sorted
ratings_df["rating"] = pd.to_numeric(ratings_df["rating"], errors="coerce").fillna(1500)
top3 = ratings_df.sort_values("rating", ascending=False).head(3).reset_index(drop=True)

cols = st.columns(3)
badges = ["🥇 #1", "🥈 #2", "🥉 #3"]
for i in range(3):
    if i < len(top3):
        player = top3.loc[i, "player"]
        rating = top3.loc[i, "rating"]
        wins = top3.loc[i, "wins"] if "wins" in top3.columns else ""
        cols[i].metric(label=f"{badges[i]}  {player}", value=f"{rating:.2f}", delta=f"W:{wins}")
    else:
        cols[i].write("—")

# --------------------------
# Team win/loss analysis (pair stats)
# --------------------------
st.header("🧩 Team (Pair) Analysis")

# Build pair key (sorted tuple)
def pair_key(p1, p2):
    return tuple(sorted([normalize(p1), normalize(p2)]))

pair_stats = {}
for _, r in matches.iterrows():
    pA = pair_key(r["playerA1"], r["playerA2"])
    pB = pair_key(r["playerB1"], r["playerB2"])
    if pA not in pair_stats:
        pair_stats[pA] = {"matches": 0, "wins": 0, "losses": 0}
    if pB not in pair_stats:
        pair_stats[pB] = {"matches": 0, "wins": 0, "losses": 0}

    pair_stats[pA]["matches"] += 1
    pair_stats[pB]["matches"] += 1

    if int(r["scoreA"]) > int(r["scoreB"]):
        pair_stats[pA]["wins"] += 1
        pair_stats[pB]["losses"] += 1
    else:
        pair_stats[pB]["wins"] += 1
        pair_stats[pA]["losses"] += 1

# Convert to DataFrame
pairs_list = []
for pair, s in pair_stats.items():
    pairs_list.append({
        "team": " & ".join(pair),
        "matches": s["matches"],
        "wins": s["wins"],
        "losses": s["losses"],
        "win_pct": round((s["wins"] / s["matches"] * 100) if s["matches"] else 0, 1)
    })
pairs_df = pd.DataFrame(pairs_list).sort_values("win_pct", ascending=False)
st.subheader("Top Teams by Win %")
st.dataframe(pairs_df.head(10))

# --------------------------
# Trend graph: rating change over time (replay matches)
# --------------------------
st.header("📈 Rating Trend")

# Player selector
players_all = sorted(list(ratings_df["player"].unique()))
selected_players = st.multiselect("Select players to plot (replay all matches chronologically)", players_all, default=players_all[:3])

# Prepare timeline by replaying matches in chronological order
matches_sorted = matches.sort_values("date_parsed").reset_index(drop=True)

def compute_rating_timeline(selected_players):
    # Initialize ratings to 1500
    timeline = {p: [] for p in selected_players}
    current = {}  # current ratings for all players
    for _, row in matches_sorted.iterrows():
        A1 = normalize(row["playerA1"]); A2 = normalize(row["playerA2"])
        B1 = normalize(row["playerB1"]); B2 = normalize(row["playerB2"])
        scA = int(row["scoreA"]); scB = int(row["scoreB"])
        # ensure present
        for p in [A1, A2, B1, B2]:
            if p and p not in current:
                current[p] = 1500.0
        # apply this match
        current = update_elo(A1, A2, B1, B2, scA, scB, current)
        # record snapshot for selected players
        for p in selected_players:
            timeline[p].append({
                "date": row["date_parsed"],
                "player": p,
                "rating": current.get(p, 1500.0)
            })
    # If there are no matches, provide a single point at today with initial rating
    if matches_sorted.empty:
        today = pd.to_datetime(datetime.date.today())
        return pd.DataFrame([{"date": today, "player": p, "rating": 1500.0} for p in selected_players])
    # Combine lists into dataframe
    rows = []
    for p in selected_players:
        rows.extend(timeline[p])
    if not rows:
        # fallback
        today = pd.to_datetime(datetime.date.today())
        rows = [{"date": today, "player": p, "rating": 1500.0} for p in selected_players]
    df_t = pd.DataFrame(rows)
    # sort by date
    df_t = df_t.sort_values(["player", "date"]).reset_index(drop=True)
    return df_t

if selected_players:
    timeline_df = compute_rating_timeline(selected_players)
    # Altair line chart
    chart = alt.Chart(timeline_df).mark_line(point=True).encode(
        x=alt.X("date:T", title="Date"),
        y=alt.Y("rating:Q", title="Rating"),
        color="player:N",
        tooltip=["player:N", "date:T", alt.Tooltip("rating:Q", format=".2f")]
    ).interactive()
    st.altair_chart(chart, use_container_width=True)
else:
    st.info("Pick one or more players to see rating trends.")

# --------------------------
# Prediction (re-usable)
# --------------------------
st.header("🔮 Predict Match Outcome")
c1, c2 = st.columns(2)
pA1 = normalize(c1.text_input("Team A - P1"))
pA2 = normalize(c1.text_input("Team A - P2"))
pB1 = normalize(c2.text_input("Team B - P1"))
pB2 = normalize(c2.text_input("Team B - P2"))

if st.button("Predict Win Probability"):
    # ensure ratings is up-to-date: we have ratings dict from latest computation or CSV
    if all(x in ratings for x in [pA1, pA2, pB1, pB2]):
        prob = predict_win_probability(ratings, pA1, pA2, pB1, pB2)
        st.success(f"Team A win probability: **{prob*100:.2f}%**")
    else:
        st.error("One or more players don't have ratings yet.")
