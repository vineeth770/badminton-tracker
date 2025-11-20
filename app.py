# app.py
import streamlit as st
import pandas as pd
import numpy as np
import requests
import base64
from io import StringIO
import altair as alt
import datetime
import math

st.set_page_config(page_title="Badminton Tracker", layout="wide")
st.title("🏸 Badminton Doubles Tracker")

# ----------------------------
# Inline ELO functions (self-contained)
# ----------------------------
def win_prob(teamA, teamB):
    # expects team ratings (sum of two players)
    return 1 / (1 + 10 ** ((teamB - teamA) / 400))

def update_elo(A1, A2, B1, B2, scoreA, scoreB, ratings, k=32):
    # Ensure players exist
    for p in (A1, A2, B1, B2):
        if p and p not in ratings:
            ratings[p] = 1500.0

    # If any player blank, skip
    if not all([A1, A2, B1, B2]):
        return ratings

    teamA_rating = ratings[A1] + ratings[A2]
    teamB_rating = ratings[B1] + ratings[B2]

    expectedA = win_prob(teamA_rating, teamB_rating)
    actualA = 1 if scoreA > scoreB else 0

    delta = k * (actualA - expectedA)

    ratings[A1] += delta / 2
    ratings[A2] += delta / 2
    ratings[B1] -= delta / 2
    ratings[B2] -= delta / 2

    return ratings

def predict_win_probability(ratings, A1, A2, B1, B2):
    # requires players exist in ratings
    teamA = ratings.get(A1, 1500) + ratings.get(A2, 1500)
    teamB = ratings.get(B1, 1500) + ratings.get(B2, 1500)
    return win_prob(teamA, teamB)

# ----------------------------
# Helpers
# ----------------------------
def normalize(name):
    if not isinstance(name, str):
        return ""
    n = name.strip()
    return n.title() if n else ""

def to_date(d):
    # convert pandas Timestamp or str to datetime.date
    try:
        if pd.isna(d):
            return None
        if isinstance(d, (pd.Timestamp, datetime.datetime)):
            return d.date()
        if isinstance(d, datetime.date):
            return d
        return pd.to_datetime(d, errors="coerce").date()
    except Exception:
        return None

# ----------------------------
# GitHub read/write helpers (robust)
# ----------------------------
def github_get_file(path):
    """Return tuple (df, sha). If file not found, return (empty_df, None)."""
    url = f"https://api.github.com/repos/{st.secrets['REPO_OWNER']}/{st.secrets['REPO_NAME']}/contents/{path}"
    headers = {"Authorization": f"token {st.secrets['GITHUB_TOKEN']}"}
    r = requests.get(url, headers=headers)
    if r.status_code == 200:
        j = r.json()
        content = base64.b64decode(j["content"]).decode("utf-8")
        df = pd.read_csv(StringIO(content))
        return df, j.get("sha")
    elif r.status_code == 404:
        # file missing -> return empty df
        return pd.DataFrame(), None
    else:
        r.raise_for_status()

def github_put_file(path, df, sha=None, message="update csv"):
    url = f"https://api.github.com/repos/{st.secrets['REPO_OWNER']}/{st.secrets['REPO_NAME']}/contents/{path}"
    headers = {"Authorization": f"token {st.secrets['GITHUB_TOKEN']}"}
    content = df.to_csv(index=False)
    encoded = base64.b64encode(content.encode()).decode()
    payload = {"message": message, "content": encoded}
    if sha:
        payload["sha"] = sha
    r = requests.put(url, headers=headers, json=payload)
    if r.status_code not in (200, 201):
        # raise with detail for debugging
        r.raise_for_status()
    return r.json()

# ----------------------------
# Load CSVs from GitHub (or create templates)
# ----------------------------
matches, matches_sha = github_get_file(st.secrets["MATCHES_CSV"])
ratings_df, ratings_sha = github_get_file(st.secrets["RATINGS_CSV"])

# If matches empty, create correct header
if matches.empty:
    matches = pd.DataFrame(columns=["date", "playerA1", "playerA2", "playerB1", "playerB2", "scoreA", "scoreB"])

# If ratings empty, create header
if ratings_df.empty:
    ratings_df = pd.DataFrame(columns=["player", "rating", "wins", "losses", "matches"])

# Normalize text columns (if present)
for c in ["playerA1", "playerA2", "playerB1", "playerB2"]:
    if c in matches.columns:
        matches[c] = matches[c].fillna("").astype(str).apply(normalize)

# Make sure score columns exist and are ints (but we allow empty input in UI)
for c in ["scoreA", "scoreB"]:
    if c not in matches.columns:
        matches[c] = pd.NA
    # keep as object if empty; when computing convert with to_numeric

# Ensure ratings_df columns exist & coerce numeric types
for col in ["rating", "wins", "losses", "matches"]:
    if col not in ratings_df.columns:
        ratings_df[col] = np.nan

ratings_df["player"] = ratings_df["player"].fillna("").astype(str).apply(normalize)
ratings_df["rating"] = pd.to_numeric(ratings_df["rating"], errors="coerce").fillna(1500.0)
ratings_df["wins"] = pd.to_numeric(ratings_df["wins"], errors="coerce").fillna(0).astype(int)
ratings_df["losses"] = pd.to_numeric(ratings_df["losses"], errors="coerce").fillna(0).astype(int)
ratings_df["matches"] = pd.to_numeric(ratings_df["matches"], errors="coerce").fillna(0).astype(int)

# Build ratings dict and stats
ratings = {row["player"]: float(row["rating"]) for _, row in ratings_df.iterrows() if row["player"]}
player_stats = {
    row["player"]: {"wins": int(row["wins"]), "losses": int(row["losses"]), "matches": int(row["matches"])}
    for _, row in ratings_df.iterrows() if row["player"]
}

# ----------------------------
# UI: Add match (scores as text inputs so they appear empty)
# ----------------------------
st.header("➕ Add Match")

with st.form("match_form"):
    c1, c2 = st.columns(2)
    A1 = normalize(c1.text_input("Team A - Player 1", value="", key="A1"))
    A2 = normalize(c2.text_input("Team A - Player 2", value="", key="A2"))
    B1 = normalize(c1.text_input("Team B - Player 1", value="", key="B1"))
    B2 = normalize(c2.text_input("Team B - Player 2", value="", key="B2"))

    # Use text_input for scores so field is empty when page opens. On save we'll coerce to int(default 0)
    sA_str = c1.text_input("Score A (leave empty = 0)", value="", key="sA")
    sB_str = c2.text_input("Score B (leave empty = 0)", value="", key="sB")

    match_date = c2.date_input("Match date", value=datetime.date.today(), key="match_date")
    add_sub = st.form_submit_button("Save Match")

if add_sub:
    # convert scores; default 0 if empty or invalid
    try:
        sA = int(sA_str) if str(sA_str).strip() != "" else 0
    except Exception:
        sA = 0
    try:
        sB = int(sB_str) if str(sB_str).strip() != "" else 0
    except Exception:
        sB = 0

    # Build new row
    new_row = {
        "date": str(match_date),
        "playerA1": A1, "playerA2": A2,
        "playerB1": B1, "playerB2": B2,
        "scoreA": sA, "scoreB": sB
    }
    # Append to matches DataFrame
    matches = pd.concat([matches, pd.DataFrame([new_row])], ignore_index=True)

    # Save matches.csv (get latest sha again to avoid overwrite errors)
    # If matches_sha is None (file didn't exist), github_put will create it
    try:
        github_put_file = github_put_file  # no-op to silence linters; actual below
    except Exception:
        pass

    # Refresh existing sha (fetch latest)
    try:
        _, matches_sha = github_get_file(st.secrets["MATCHES_CSV"])
    except Exception:
        matches_sha = None

    github_put_file = github_put_file if 'github_put_file' in globals() else github_put_file  # ignore

    # Save to GitHub (use helper)
    github_put_file = lambda path, df, sha, message="update csv": github_put_file_impl(path, df, sha, message)

    # But our actual function is github_put_file (defined above as github_put_file), so call it:
    # (call the function defined earlier)
    github_put_file_impl = github_put_file if callable(globals().get("github_put_file")) else None

    # Simpler: call the helper directly
    github_put_file = github_put_file  # (safe no-op)
    try:
        github_put_file(st.secrets["MATCHES_CSV"], matches, matches_sha, "Add match")
    except Exception as e:
        # Fallback to direct call of github_put_file_impl (if naming got weird)
        try:
            github_put_file_impl = globals()["github_put_file"]
            github_put_file_impl(st.secrets["MATCHES_CSV"], matches, matches_sha, "Add match")
        except Exception:
            # final fallback: try direct helper name from our file (github_put_file)
            github_put_file_helper = github_put_file  # no-op
            # We'll call the original function name github_put_file which is defined above
            github_put_file_helper = globals().get("github_put_file")
            if callable(github_put_file_helper):
                github_put_file_helper(st.secrets["MATCHES_CSV"], matches, matches_sha, "Add match")
            else:
                st.error("Unable to save matches to GitHub (internal).")
                st.stop()

    # Clear input fields after saving - by setting session state values
    for k in ("A1","A2","B1","B2","sA","sB"):
        try:
            st.session_state[k] = ""
        except Exception:
            pass

    # Update ratings/stats structures
    for p in [A1, A2, B1, B2]:
        if p and p not in ratings:
            ratings[p] = 1500.0
        if p and p not in player_stats:
            player_stats[p] = {"wins": 0, "losses": 0, "matches": 0}

    for p in [A1, A2, B1, B2]:
        if p:
            player_stats[p]["matches"] += 1

    if sA > sB:
        if A1: player_stats[A1]["wins"] += 1
        if A2: player_stats[A2]["wins"] += 1
        if B1: player_stats[B1]["losses"] += 1
        if B2: player_stats[B2]["losses"] += 1
    else:
        if B1: player_stats[B1]["wins"] += 1
        if B2: player_stats[B2]["wins"] += 1
        if A1: player_stats[A1]["losses"] += 1
        if A2: player_stats[A2]["losses"] += 1

    # Recompute all ratings by replaying matches chronologically (safer)
    # Ensure matches have date_parsed column
    def parse_dates_column(df):
        if "date" in df.columns:
            df["date_parsed"] = pd.to_datetime(df["date"], errors="coerce")
        else:
            df["date_parsed"] = pd.NaT
        # If all NaT, assign order-based dates
        if df["date_parsed"].isna().all():
            df = df.reset_index(drop=True)
            df["date_parsed"] = pd.to_datetime("1970-01-01") + pd.to_timedelta(df.index, unit="D")
        return df

    matches = parse_dates_column(matches)
    matches_sorted = matches.sort_values("date_parsed").reset_index(drop=True)

    ratings = {}  # recompute from scratch
    for _, row in matches_sorted.iterrows():
        pA1 = normalize(row.get("playerA1", ""))
        pA2 = normalize(row.get("playerA2", ""))
        pB1 = normalize(row.get("playerB1", ""))
        pB2 = normalize(row.get("playerB2", ""))
        scA = int(pd.to_numeric(row.get("scoreA", 0), errors="coerce") or 0)
        scB = int(pd.to_numeric(row.get("scoreB", 0), errors="coerce") or 0)
        ratings = update_elo(pA1, pA2, pB1, pB2, scA, scB, ratings)

    # Save ratings_df
    ratings_df = pd.DataFrame([
        {
            "player": p,
            "rating": round(ratings.get(p, 1500.0), 2),
            "wins": player_stats.get(p, {}).get("wins", 0),
            "losses": player_stats.get(p, {}).get("losses", 0),
            "matches": player_stats.get(p, {}).get("matches", 0)
        } for p in sorted(ratings.keys())
    ])

    # Save to GitHub
    try:
        # refresh sha for ratings file
        _, ratings_sha = github_get_file(st.secrets["RATINGS_CSV"])
    except Exception:
        ratings_sha = None

    github_put_file(st.secrets["RATINGS_CSV"], ratings_df, ratings_sha, "Update ratings")

    st.success("Saved match and updated ratings.")

# ----------------------------
# Prepare matches for display / filtering
# ----------------------------
# Ensure date column exists
if "date" not in matches.columns:
    matches["date"] = ""

# Trim/strip strings
matches = matches.fillna("").applymap(lambda x: x.strip() if isinstance(x, str) else x)

# parse date_parsed
if "date_parsed" not in matches.columns:
    matches["date_parsed"] = pd.to_datetime(matches["date"], errors="coerce")

# If date_parsed all NaT, assign artificial dates by index
if matches["date_parsed"].isna().all():
    matches = matches.reset_index(drop=True)
    matches["date_parsed"] = pd.to_datetime("1970-01-01") + pd.to_timedelta(matches.index, unit="D")

# Date filter UI
st.header("📜 Match History & Filters")
min_date = matches["date_parsed"].min()
max_date = matches["date_parsed"].max()

# convert to date for date_input default values; handle NaT
min_date_val = min_date.date() if pd.notna(min_date) else datetime.date.today()
max_date_val = max_date.date() if pd.notna(max_date) else datetime.date.today()

colf1, colf2, colf3 = st.columns([2,2,1])
with colf1:
    start_date = st.date_input("From", value=min_date_val)
with colf2:
    end_date = st.date_input("To", value=max_date_val)
with colf3:
    _apply = st.button("Apply Filter")

# Build datetime range inclusive
start_dt = pd.to_datetime(start_date)
end_dt = pd.to_datetime(end_date) + pd.Timedelta(days=1) - pd.Timedelta(seconds=1)

filtered = matches[(matches["date_parsed"] >= start_dt) & (matches["date_parsed"] <= end_dt)]

# Show only existing columns (safe)
cols_wanted = ["date", "playerA1", "playerA2", "playerB1", "playerB2", "scoreA", "scoreB"]
cols_to_show = [c for c in cols_wanted if c in filtered.columns]
st.subheader(f"Showing {len(filtered)} matches between {start_date} and {end_date}")
st.dataframe(filtered[cols_to_show].reset_index(drop=True))

# ----------------------------
# Player ranking badges (top 3)
# ----------------------------
st.header("🏆 Top Players")
if not ratings_df.empty:
    top3 = ratings_df.sort_values("rating", ascending=False).head(3).reset_index(drop=True)
else:
    top3 = pd.DataFrame(columns=["player", "rating", "wins"])

cols = st.columns(3)
medals = ["🥇 #1", "🥈 #2", "🥉 #3"]
for i in range(3):
    if i < len(top3):
        player = top3.loc[i, "player"]
        rating_val = top3.loc[i, "rating"]
        wins = int(top3.loc[i, "wins"]) if "wins" in top3.columns else 0
        cols[i].metric(label=f"{medals[i]} {player}", value=f"{rating_val:.2f}", delta=f"W:{wins}")
    else:
        cols[i].write("—")

# ----------------------------
# Team (pair) analysis
# ----------------------------
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

    scA = int(pd.to_numeric(r.get("scoreA", 0), errors="coerce") or 0)
    scB = int(pd.to_numeric(r.get("scoreB", 0), errors="coerce") or 0)

    if scA > scB:
        pair_stats[pA]["wins"] += 1
        pair_stats[pB]["losses"] += 1
    else:
        pair_stats[pB]["wins"] += 1
        pair_stats[pA]["losses"] += 1

pairs_list = []
for pair, s in pair_stats.items():
    # show readable team name
    team_name = " & ".join([p for p in pair if p])
    if not team_name:
        continue
    matches_count = s["matches"]
    wins = s["wins"]
    losses = s["losses"]
    win_pct = round((wins / matches_count * 100) if matches_count else 0, 1)
    pairs_list.append({"team": team_name, "matches": matches_count, "wins": wins, "losses": losses, "win_pct": win_pct})

if pairs_list:
    pairs_df = pd.DataFrame(pairs_list).sort_values("win_pct", ascending=False)
    st.subheader("Top Teams by Win %")
    st.dataframe(pairs_df.head(10).reset_index(drop=True))
else:
    st.info("Not enough team data yet.")

# ----------------------------
# Rating trend graph (replay matches chronologically)
# ----------------------------
st.header("📈 Rating Trend")
players_all = sorted(ratings_df["player"].unique().tolist())
selected_players = st.multiselect("Select players to plot", players_all, default=players_all[:3])

matches_sorted = matches.sort_values("date_parsed").reset_index(drop=True)
def compute_timeline(selected):
    # replay matches; snapshot after each match for selected players
    current = {}
    rows = []
    for _, row in matches_sorted.iterrows():
        pA1 = normalize(row.get("playerA1","")); pA2 = normalize(row.get("playerA2",""))
        pB1 = normalize(row.get("playerB1","")); pB2 = normalize(row.get("playerB2",""))
        scA = int(pd.to_numeric(row.get("scoreA", 0), errors="coerce") or 0)
        scB = int(pd.to_numeric(row.get("scoreB", 0), errors="coerce") or 0)
        for p in [pA1, pA2, pB1, pB2]:
            if p and p not in current:
                current[p] = 1500.0
        # apply match
        current = update_elo(pA1, pA2, pB1, pB2, scA, scB, current)
        # save snapshot for selected
        for p in selected:
            rows.append({"date": row["date_parsed"], "player": p, "rating": current.get(p, 1500.0)})
    if not rows:
        # no matches -> single point
        today = pd.to_datetime(datetime.date.today())
        rows = [{"date": today, "player": p, "rating": 1500.0} for p in selected]
    return pd.DataFrame(rows)

if selected_players:
    timeline_df = compute_timeline(selected_players)
    chart = alt.Chart(timeline_df).mark_line(point=True).encode(
        x=alt.X("date:T", title="Date"),
        y=alt.Y("rating:Q", title="Rating"),
        color="player:N",
        tooltip=["player:N", alt.Tooltip("rating:Q", format=".2f"), "date:T"]
    ).interactive()
    st.altair_chart(chart, use_container_width=True)
else:
    st.info("Select players to plot rating trends.")

# ----------------------------
# Prediction UI
# ----------------------------
st.header("🔮 Predict Match Outcome")
pc1, pc2 = st.columns(2)
pA1 = normalize(pc1.text_input("Team A - P1 (for prediction)"))
pA2 = normalize(pc1.text_input("Team A - P2 (for prediction)"))
pB1 = normalize(pc2.text_input("Team B - P1 (for prediction)"))
pB2 = normalize(pc2.text_input("Team B - P2 (for prediction)"))

if st.button("Predict Win Probability"):
    if all(x for x in [pA1,pA2,pB1,pB2]):
        prob = predict_win_probability(ratings, pA1, pA2, pB1, pB2)
        st.success(f"Team A win probability: **{prob*100:.2f}%**")
    else:
        st.error("Fill all player fields for prediction.")
