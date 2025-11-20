# app.py
import streamlit as st
import pandas as pd
import numpy as np
import requests
import base64
from io import StringIO
import altair as alt
import datetime
from elo import update_elo, predict_win_probability  # keep elo.py in repo

st.set_page_config(page_title="Badminton Tracker", layout="wide")
st.title("🏸 Badminton Doubles Tracker")

# --------------------------
# Helpers
# --------------------------
def normalize(name):
    if not isinstance(name, str):
        return ""
    return name.strip().title()

def safe_int_from_str(s):
    """Accept int, float, empty string, or string number. Empty -> 0"""
    if s is None:
        return 0
    if isinstance(s, (int, float)):
        return int(s)
    try:
        s2 = str(s).strip()
        if s2 == "":
            return 0
        return int(float(s2))
    except Exception:
        return 0

# --------------------------
# GitHub helpers (no recursion)
# --------------------------
TOKEN = st.secrets.get("GITHUB_TOKEN", "")
OWNER = st.secrets.get("REPO_OWNER", "")
REPO = st.secrets.get("REPO_NAME", "")
MATCHES_PATH = st.secrets.get("MATCHES_CSV", "matches.csv")
RATINGS_PATH = st.secrets.get("RATINGS_CSV", "ratings.csv")

API_BASE = f"https://api.github.com/repos/{OWNER}/{REPO}/contents"

def github_get_file(path):
    """Return (df, sha). If file missing, return (empty df with expected cols, None)."""
    url = f"{API_BASE}/{path}"
    headers = {"Authorization": f"token {TOKEN}"} if TOKEN else {}
    r = requests.get(url, headers=headers)
    if r.status_code == 200:
        j = r.json()
        content_b64 = j.get("content", "")
        content = base64.b64decode(content_b64).decode("utf-8")
        df = pd.read_csv(StringIO(content))
        return df, j.get("sha")
    elif r.status_code == 404:
        # return empty matches/ratings depending on path
        if path == MATCHES_PATH:
            df = pd.DataFrame(columns=["date","playerA1","playerA2","playerB1","playerB2","scoreA","scoreB"])
        else:
            df = pd.DataFrame(columns=["player","rating","wins","losses","matches"])
        return df, None
    else:
        st.error(f"GitHub GET error {r.status_code}: {r.text}")
        st.stop()

def github_put_file(path, df, sha=None, message="update csv"):
    """Create or update file. If sha is None, GitHub creates new file."""
    url = f"{API_BASE}/{path}"
    headers = {"Authorization": f"token {TOKEN}"} if TOKEN else {}
    content_b64 = base64.b64encode(df.to_csv(index=False).encode()).decode()
    payload = {"message": message, "content": content_b64}
    if sha:
        payload["sha"] = sha
    r = requests.put(url, headers=headers, json=payload)
    if r.status_code in (200, 201):
        return r.json().get("content", {}).get("sha")
    else:
        st.error(f"GitHub PUT error {r.status_code}: {r.text}")
        st.stop()

# --------------------------
# Load CSVs from GitHub (or create if missing)
# --------------------------
matches, matches_sha = github_get_file(MATCHES_PATH)
ratings_df, ratings_sha = github_get_file(RATINGS_PATH)

# Normalize and ensure expected columns exist
for col in ["playerA1","playerA2","playerB1","playerB2"]:
    if col in matches.columns:
        matches[col] = matches[col].fillna("").astype(str).apply(normalize)
    else:
        matches[col] = ""

for col in ["scoreA","scoreB"]:
    if col in matches.columns:
        matches[col] = pd.to_numeric(matches[col], errors="coerce").fillna(0).astype(int)
    else:
        matches[col] = 0

if "date" not in matches.columns:
    matches["date"] = ""  # keep backwards compatibility

# Ensure ratings_df columns exist and types
if ratings_df.empty:
    ratings_df = pd.DataFrame(columns=["player","rating","wins","losses","matches"])
else:
    # normalize names
    ratings_df["player"] = ratings_df["player"].astype(str).apply(normalize)
    for c in ["rating","wins","losses","matches"]:
        if c in ratings_df.columns:
            ratings_df[c] = pd.to_numeric(ratings_df[c], errors="coerce").fillna(0)
        else:
            ratings_df[c] = 0

# Turn ratings/players into dicts for quick updates
ratings = {}
player_stats = {}
for _, r in ratings_df.iterrows():
    p = r["player"]
    ratings[p] = float(r["rating"])
    player_stats[p] = {
        "wins": int(r["wins"]),
        "losses": int(r["losses"]),
        "matches": int(r["matches"])
    }

# --------------------------
# Add Match form
# --------------------------
st.header("➕ Add Match")

with st.form("add_match"):
    c1, c2 = st.columns(2)
    a1 = c1.text_input("Team A - Player 1").strip()
    a2 = c2.text_input("Team A - Player 2").strip()
    b1 = c1.text_input("Team B - Player 1").strip()
    b2 = c2.text_input("Team B - Player 2").strip()

    # empty by default; user can leave blank
    sA_raw = c1.text_input("Score A (leave empty = 0)", value="")
    sB_raw = c2.text_input("Score B (leave empty = 0)", value="")

    # date input - default to today
    match_date = c2.date_input("Match date", value=datetime.date.today())

    submitted = st.form_submit_button("Save Match")

if submitted:
    # normalize names and scores
    A1 = normalize(a1)
    A2 = normalize(a2)
    B1 = normalize(b1)
    B2 = normalize(b2)
    sA = safe_int_from_str(sA_raw)
    sB = safe_int_from_str(sB_raw)

    # Build new row with date as ISO (YYYY-MM-DD)
    new_row = {
        "date": match_date.isoformat(),
        "playerA1": A1, "playerA2": A2,
        "playerB1": B1, "playerB2": B2,
        "scoreA": sA, "scoreB": sB
    }

    # Append to matches DataFrame
    matches = pd.concat([matches, pd.DataFrame([new_row])], ignore_index=True)

    # Save matches back to GitHub (create if missing)
    matches_sha = github_put_file(MATCHES_PATH, matches, matches_sha, "Add match")

    # Ensure any new players initialized
    for p in [A1, A2, B1, B2]:
        if p and p not in ratings:
            ratings[p] = 1500.0
        if p and p not in player_stats:
            player_stats[p] = {"wins": 0, "losses": 0, "matches": 0}

    # Increment match counts
    for p in [A1, A2, B1, B2]:
        if p:
            player_stats[p]["matches"] += 1

    # Assign wins/losses
    if sA > sB:
        player_stats[A1]["wins"] += 1
        player_stats[A2]["wins"] += 1
        player_stats[B1]["losses"] += 1
        player_stats[B2]["losses"] += 1
    elif sB > sA:
        player_stats[B1]["wins"] += 1
        player_stats[B2]["wins"] += 1
        player_stats[A1]["losses"] += 1
        player_stats[A2]["losses"] += 1
    else:
        # tie/no-win -> treat as no wins/losses
        pass

    # Recompute ELOs by replaying matches chronologically (safe)
    # Start from fresh ratings map and replay every row
    ratings_replay = {}
    matches_sorted = matches.sort_values(by="date", kind="stable").reset_index(drop=True)
    for _, row in matches_sorted.iterrows():
        pA1 = normalize(row.get("playerA1",""))
        pA2 = normalize(row.get("playerA2",""))
        pB1 = normalize(row.get("playerB1",""))
        pB2 = normalize(row.get("playerB2",""))
        scA = safe_int_from_str(row.get("scoreA", 0))
        scB = safe_int_from_str(row.get("scoreB", 0))
        ratings_replay = update_elo(pA1, pA2, pB1, pB2, scA, scB, ratings_replay)

    # Merge replayed ratings into ratings dict (ensure all players present)
    for p in set(list(ratings.keys()) + list(ratings_replay.keys())):
        ratings[p] = float(ratings_replay.get(p, ratings.get(p, 1500.0)))

    # Build ratings_df and save to GitHub
    ratings_out = pd.DataFrame([
        {
            "player": p,
            "rating": round(ratings[p], 2),
            "wins": player_stats.get(p, {}).get("wins", 0),
            "losses": player_stats.get(p, {}).get("losses", 0),
            "matches": player_stats.get(p, {}).get("matches", 0)
        }
        for p in sorted(ratings.keys())
    ])

    ratings_sha = github_put_file(RATINGS_PATH, ratings_out, ratings_sha, "Update ratings")

    # Rerun app so form clears and latest CSVs are loaded (clears input fields)
    st.experimental_rerun()

# --------------------------
# Prepare matches and ratings for display
# --------------------------
# Reload current copies after potential update (to reflect correct SHAs)
matches, matches_sha = github_get_file(MATCHES_PATH)
ratings_df, ratings_sha = github_get_file(RATINGS_PATH)

# Normalize display
for c in ["playerA1","playerA2","playerB1","playerB2"]:
    if c in matches.columns:
        matches[c] = matches[c].fillna("").astype(str).apply(normalize)

for c in ["scoreA","scoreB"]:
    if c in matches.columns:
        matches[c] = pd.to_numeric(matches[c], errors="coerce").fillna(0).astype(int)
    else:
        matches[c] = 0

if "date" not in matches.columns:
    matches["date"] = ""

# Parse dates (safe)
def parse_dates_column(df):
    if "date" in df.columns:
        df["date_parsed"] = pd.to_datetime(df["date"], errors="coerce")
    else:
        df["date_parsed"] = pd.NaT
    return df

matches = parse_dates_column(matches)

# --------------------------
# Filters & Match History display
# --------------------------
st.header("📜 Match History")

# date filter inputs - ensure valid defaults
min_date = matches["date_parsed"].min()
max_date = matches["date_parsed"].max()
today = pd.to_datetime(datetime.date.today())

if pd.isna(min_date):
    min_date = today
if pd.isna(max_date):
    max_date = today

colf1, colf2, colf3 = st.columns([2,2,1])
with colf1:
    start_date = st.date_input("From", value=min_date.date() if hasattr(min_date, "date") else today.date())
with colf2:
    end_date = st.date_input("To", value=max_date.date() if hasattr(max_date, "date") else today.date())
with colf3:
    _ = st.button("Apply")

start_dt = pd.to_datetime(start_date)
end_dt = pd.to_datetime(end_date) + pd.Timedelta(days=1) - pd.Timedelta(seconds=1)

filtered = matches.copy()
# if date_parsed is NaT for rows, include them only if they fall into the default range? We'll include them when date missing.
mask = ((filtered["date_parsed"].isna()) | ((filtered["date_parsed"] >= start_dt) & (filtered["date_parsed"] <= end_dt)))
filtered = filtered[mask].reset_index(drop=True)

cols_to_show = [c for c in ["date","playerA1","playerA2","playerB1","playerB2","scoreA","scoreB"] if c in filtered.columns]
st.subheader(f"Showing {len(filtered)} matches")
st.dataframe(filtered[cols_to_show])

# --------------------------
# Player individual stats (from ratings_df)
# --------------------------
st.header("📊 Player Statistics")

if ratings_df.empty:
    st.info("No players yet. Add a match to generate player stats.")
else:
    # Ensure numeric types
    for c in ["wins","losses","matches","rating"]:
        if c in ratings_df.columns:
            ratings_df[c] = pd.to_numeric(ratings_df[c], errors="coerce").fillna(0)

    stats_display = ratings_df.copy()
    stats_display["Win %"] = ((stats_display["wins"] / stats_display["matches"]) * 100).replace([np.inf, -np.inf], 0).fillna(0).round(1)
    stats_display = stats_display[["player","rating","matches","wins","losses","Win %"]]
    stats_display = stats_display.rename(columns={"player":"Player","rating":"Rating","matches":"Matches","wins":"Wins","losses":"Losses"})
    st.dataframe(stats_display.sort_values("Rating", ascending=False))

# --------------------------
# Top player badges
# --------------------------
st.header("🏆 Top Players")
if not ratings_df.empty:
    top3 = ratings_df.sort_values("rating", ascending=False).head(3).reset_index(drop=True)
    cols = st.columns(3)
    medals = ["🥇 #1", "🥈 #2", "🥉 #3"]
    for i in range(3):
        if i < len(top3):
            p = top3.loc[i, "player"]
            r = top3.loc[i, "rating"]
            w = int(top3.loc[i, "wins"]) if "wins" in top3.columns else ""
            cols[i].metric(label=f"{medals[i]}  {p}", value=f"{r:.2f}", delta=f"W:{w}")
        else:
            cols[i].write("—")
else:
    st.info("No player ratings yet.")

# --------------------------
# Team (pair) analysis
# --------------------------
st.header("🧩 Team (Pair) Analysis")
pair_stats = {}
for _, r in matches.iterrows():
    A = tuple(sorted([normalize(r.get("playerA1","")), normalize(r.get("playerA2",""))]))
    B = tuple(sorted([normalize(r.get("playerB1","")), normalize(r.get("playerB2",""))]))
    if A not in pair_stats:
        pair_stats[A] = {"matches":0,"wins":0,"losses":0}
    if B not in pair_stats:
        pair_stats[B] = {"matches":0,"wins":0,"losses":0}

    pair_stats[A]["matches"] += 1
    pair_stats[B]["matches"] += 1

    if int(r.get("scoreA",0)) > int(r.get("scoreB",0)):
        pair_stats[A]["wins"] += 1
        pair_stats[B]["losses"] += 1
    elif int(r.get("scoreB",0)) > int(r.get("scoreA",0)):
        pair_stats[B]["wins"] += 1
        pair_stats[A]["losses"] += 1
    else:
        # tie -> no wins
        pass

pairs_list = []
for pair, s in pair_stats.items():
    team_name = " & ".join([p for p in pair if p])
    if team_name == "":
        continue
    matches_count = s["matches"]
    wins = s["wins"]
    losses = s["losses"]
    win_pct = round((wins/matches_count*100) if matches_count else 0, 1)
    pairs_list.append({"team": team_name, "matches": matches_count, "wins": wins, "losses": losses, "win_pct": win_pct})

pairs_df = pd.DataFrame(pairs_list)
if not pairs_df.empty:
    st.dataframe(pairs_df.sort_values("win_pct", ascending=False).head(15))
else:
    st.info("Not enough team data yet.")

# --------------------------
# Rating trend graph (replay matches)
# --------------------------
st.header("📈 Rating Trend")

players_all = sorted(list(ratings_df["player"].unique())) if not ratings_df.empty else []
selected_players = st.multiselect("Players to plot (replay matches chronologically)", players_all, default=players_all[:3])

def compute_timeline(players):
    # replay from scratch
    timeline_rows = []
    current = {}
    matches_sorted = matches.sort_values(by="date", kind="stable").reset_index(drop=True)
    for _, row in matches_sorted.iterrows():
        pA1 = normalize(row.get("playerA1","")); pA2 = normalize(row.get("playerA2",""))
        pB1 = normalize(row.get("playerB1","")); pB2 = normalize(row.get("playerB2",""))
        scA = safe_int_from_str(row.get("scoreA",0)); scB = safe_int_from_str(row.get("scoreB",0))
        # ensure present
        for p in [pA1,pA2,pB1,pB2]:
            if p and p not in current:
                current[p] = 1500.0
        # apply match
        current = update_elo(pA1,pA2,pB1,pB2,scA,scB,current)
        # snapshot for the selected players
        date_point = pd.to_datetime(row.get("date"), errors="coerce")
        for p in players:
            timeline_rows.append({"date": date_point if not pd.isna(date_point) else pd.to_datetime(datetime.date.today()), "player": p, "rating": float(current.get(p,1500.0))})
    if not timeline_rows:
        # return single point per player
        today = pd.to_datetime(datetime.date.today())
        for p in players:
            timeline_rows.append({"date": today, "player": p, "rating": float(ratings.get(p,1500.0))})
    return pd.DataFrame(timeline_rows)

if selected_players:
    tdf = compute_timeline(selected_players)
    chart = alt.Chart(tdf).mark_line(point=True).encode(
        x=alt.X("date:T", title="Date"),
        y=alt.Y("rating:Q", title="Rating"),
        color="player:N",
        tooltip=["player:N", alt.Tooltip("rating:Q", format=".2f"), "date:T"]
    ).interactive()
    st.altair_chart(chart, use_container_width=True)
else:
    st.info("Select players to show rating trends.")

# --------------------------
# Prediction
# --------------------------
st.header("🔮 Predict Match Outcome")
p1, p2 = st.columns(2)
pA1_in = normalize(p1.text_input("Team A - P1"))
pA2_in = normalize(p1.text_input("Team A - P2"))
pB1_in = normalize(p2.text_input("Team B - P1"))
pB2_in = normalize(p2.text_input("Team B - P2"))

if st.button("Predict Win Probability"):
    if all(x in ratings for x in [pA1_in, pA2_in, pB1_in, pB2_in]):
        prob = predict_win_probability(ratings, pA1_in, pA2_in, pB1_in, pB2_in)
        st.success(f"Team A win probability: **{prob*100:.2f}%**")
    else:
        st.error("One or more players don't have ratings yet. Add matches to generate ratings.")
