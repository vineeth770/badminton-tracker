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

# --------------------------
# Helper functions
# --------------------------
def normalize(name):
    """Normalize player names: strip, collapse spaces, title-case (case-insensitive)."""
    if not isinstance(name, str):
        return ""
    return " ".join(name.split()).title()

def safe_to_int(x, default=0):
    try:
        return int(x)
    except Exception:
        try:
            return int(float(x))
        except Exception:
            return default

# --------------------------
# GitHub helpers (robust)
# --------------------------
def github_get_file(path):
    """Return (df, sha). If file not found return (empty_df, None)."""
    api_url = f"https://api.github.com/repos/{st.secrets['REPO_OWNER']}/{st.secrets['REPO_NAME']}/contents/{path}"
    headers = {"Authorization": f"token {st.secrets['GITHUB_TOKEN']}"}
    r = requests.get(api_url, headers=headers)
    if r.status_code == 404:
        # file missing -> return empty appropriate DF later by caller
        return None, None
    r.raise_for_status()
    j = r.json()
    content_b64 = j.get("content", "")
    if not content_b64:
        return None, j.get("sha")
    s = base64.b64decode(content_b64).decode("utf-8")
    df = pd.read_csv(StringIO(s))
    return df, j.get("sha")

def github_put_file(path, df, sha=None, message="update"):
    """Create or update a file at path with df (pandas). If sha is None, GitHub creates file."""
    api_url = f"https://api.github.com/repos/{st.secrets['REPO_OWNER']}/{st.secrets['REPO_NAME']}/contents/{path}"
    headers = {"Authorization": f"token {st.secrets['GITHUB_TOKEN']}"}
    csv_bytes = df.to_csv(index=False).encode("utf-8")
    content_b64 = base64.b64encode(csv_bytes).decode("utf-8")
    payload = {"message": message, "content": content_b64}
    if sha:
        payload["sha"] = sha
    r = requests.put(api_url, headers=headers, json=payload)
    r.raise_for_status()
    return r.json().get("content", {}).get("sha")

# --------------------------
# Load CSVs from GitHub (or create empty DataFrames)
# --------------------------
# matches.csv should have columns: date (optional), playerA1, playerA2, playerB1, playerB2, scoreA, scoreB
matches_path = st.secrets.get("MATCHES_CSV", "matches.csv")
ratings_path = st.secrets.get("RATINGS_CSV", "ratings.csv")

matches_df, matches_sha = github_get_file(matches_path)
if matches_df is None:
    # create empty matches dataframe with expected columns
    matches_df = pd.DataFrame(columns=["date", "playerA1", "playerA2", "playerB1", "playerB2", "scoreA", "scoreB"])
    matches_sha = None

ratings_df, ratings_sha = github_get_file(ratings_path)
if ratings_df is None:
    ratings_df = pd.DataFrame(columns=["player", "rating", "wins", "losses", "matches"])
    ratings_sha = None

# --------------------------
# Clean & normalize loaded data
# --------------------------
# Ensure expected columns exist in matches_df
for col in ["date", "playerA1", "playerA2", "playerB1", "playerB2", "scoreA", "scoreB"]:
    if col not in matches_df.columns:
        matches_df[col] = "" if col == "date" else 0 if col.startswith("score") else ""

# Normalize string columns
for c in ["playerA1", "playerA2", "playerB1", "playerB2", "date"]:
    if c in matches_df.columns:
        matches_df[c] = matches_df[c].fillna("").astype(str).apply(normalize if c != "date" else lambda d: d.strip())

# Ensure numeric score columns
for c in ["scoreA", "scoreB"]:
    matches_df[c] = pd.to_numeric(matches_df[c], errors="coerce").fillna(0).astype(int)

# Ensure ratings_df columns
for col in ["player", "rating", "wins", "losses", "matches"]:
    if col not in ratings_df.columns:
        if col == "player":
            ratings_df[col] = []
        elif col == "rating":
            ratings_df[col] = []
        else:
            ratings_df[col] = []

# Normalize ratings_df player column and numeric columns
if not ratings_df.empty:
    ratings_df["player"] = ratings_df["player"].astype(str).apply(normalize)
ratings_df["rating"] = pd.to_numeric(ratings_df.get("rating", pd.Series(dtype=float)), errors="coerce").fillna(1500.0)
for col in ["wins", "losses", "matches"]:
    ratings_df[col] = pd.to_numeric(ratings_df.get(col, pd.Series(dtype=int)), errors="coerce").fillna(0).astype(int)

# Build ratings dict & player_stats map from ratings_df
ratings = {}
player_stats = {}
for _, r in ratings_df.iterrows():
    p = normalize(r["player"])
    ratings[p] = float(r["rating"])
    player_stats[p] = {"wins": int(r["wins"]), "losses": int(r["losses"]), "matches": int(r["matches"])}

# --------------------------
# UI: Add New Match
# --------------------------
st.header("➕ Add New Match")

with st.form("add_match", clear_on_submit=True):
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
    # validate
    if not (a1 and a2 and b1 and b2):
        st.error("Please enter all 4 player names.")
    else:
        # append to matches_df
        new_row = {
            "date": match_date.isoformat(),
            "playerA1": a1, "playerA2": a2,
            "playerB1": b1, "playerB2": b2,
            "scoreA": int(sA), "scoreB": int(sB)
        }
        matches_df = pd.concat([matches_df, pd.DataFrame([new_row])], ignore_index=True)

        # save matches to GitHub (get current sha before put)
        try:
            # fetch current sha (in case someone else updated)
            _, latest_matches_sha = github_get_file(matches_path)
            sha_to_use = latest_matches_sha or matches_sha
        except Exception:
            sha_to_use = matches_sha
        try:
            new_sha = github_put_file(matches_path, matches_df, sha=sha_to_use, message="Add match")
            matches_sha = new_sha
        except Exception as e:
            st.error(f"Failed to save match to GitHub: {e}")
            st.stop()

        st.success("Match saved to repository.")

        # Ensure players exist in ratings/stats
        for p in [a1, a2, b1, b2]:
            if p and p not in ratings:
                ratings[p] = 1500.0
            if p and p not in player_stats:
                player_stats[p] = {"wins": 0, "losses": 0, "matches": 0}

        # update player_stats counts
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

        # To be consistent and safe, recompute ratings from scratch by replaying matches in chronological order
        # Build chronological ordering by date column if valid, otherwise by row order
        temp_matches = matches_df.copy()
        # parse dates
        def parse_date_safe(d):
            try:
                return pd.to_datetime(d)
            except Exception:
                return pd.NaT
        temp_matches["__dt"] = temp_matches["date"].apply(parse_date_safe)
        if temp_matches["__dt"].isna().all():
            temp_matches = temp_matches.reset_index(drop=True)
        else:
            temp_matches = temp_matches.sort_values("__dt").reset_index(drop=True)

        # recompute ELOs
        ratings_recomputed = {}
        for _, row in temp_matches.iterrows():
            pA1 = normalize(row["playerA1"]); pA2 = normalize(row["playerA2"])
            pB1 = normalize(row["playerB1"]); pB2 = normalize(row["playerB2"])
            scA = safe_to_int(row["scoreA"])
            scB = safe_to_int(row["scoreB"])
            ratings_recomputed = update_elo(pA1, pA2, pB1, pB2, scA, scB, ratings_recomputed)

        # merge recomputed ratings into ratings & rebuild ratings_df
        for p, r in ratings_recomputed.items():
            ratings[p] = float(r)

        # Build new ratings_df from ratings + player_stats
        new_ratings_rows = []
        for p in sorted(ratings.keys()):
            stats = player_stats.get(p, {"wins": 0, "losses": 0, "matches": 0})
            new_ratings_rows.append({
                "player": p,
                "rating": round(ratings.get(p, 1500.0), 2),
                "wins": int(stats.get("wins", 0)),
                "losses": int(stats.get("losses", 0)),
                "matches": int(stats.get("matches", 0))
            })
        ratings_df = pd.DataFrame(new_ratings_rows)

        # save ratings_df to GitHub
        try:
            _, latest_ratings_sha = github_get_file(ratings_path)
            sha_to_use_r = latest_ratings_sha or ratings_sha
        except Exception:
            sha_to_use_r = ratings_sha
        try:
            new_ratings_sha = github_put_file(ratings_path, ratings_df, sha=sha_to_use_r, message="Update ratings")
            ratings_sha = new_ratings_sha
        except Exception as e:
            st.error(f"Failed to save ratings to GitHub: {e}")
            st.stop()

# --------------------------
# Prepare matches (date parsing + filtering)
# --------------------------
# Guarantee date column exists
if "date" not in matches_df.columns:
    matches_df["date"] = ""

# parse date into date_parsed column; if parse fails -> NaT
def parse_date_col(x):
    try:
        return pd.to_datetime(x)
    except Exception:
        return pd.NaT

matches_df["date_parsed"] = matches_df["date"].apply(parse_date_col)

# If all date_parsed are NaT, create synthetic incremental dates so filtering still works
if matches_df["date_parsed"].isna().all():
    # create synthetic dates starting today stepping negative days (so older rows are older)
    n = len(matches_df)
    base = pd.to_datetime(datetime.date.today())
    synthetic = [base - pd.Timedelta(days=(n - i - 1)) for i in range(n)]
    matches_df["date_parsed"] = synthetic

# UI: date filters
st.header("📜 Match History & Filters")
min_date = matches_df["date_parsed"].min().date()
max_date = matches_df["date_parsed"].max().date()

col1, col2, col3 = st.columns([2,2,1])
with col1:
    start = st.date_input("Start date", value=min_date)
with col2:
    end = st.date_input("End date", value=max_date)
with col3:
    _ = st.write("")  # placeholder for layout

# filter inclusive
start_dt = pd.to_datetime(start)
end_dt = pd.to_datetime(end) + pd.Timedelta(days=1) - pd.Timedelta(seconds=1)
filtered = matches_df[(matches_df["date_parsed"] >= start_dt) & (matches_df["date_parsed"] <= end_dt)]

# Only show existing columns to avoid KeyError
cols_to_show = [c for c in ["date", "playerA1", "playerA2", "playerB1", "playerB2", "scoreA", "scoreB"] if c in filtered.columns]
st.subheader(f"Showing {len(filtered)} matches between {start} and {end}")
st.dataframe(filtered[cols_to_show].reset_index(drop=True))

# --------------------------
# Player ranking badges (top 3)
# --------------------------
st.header("🏆 Top Players")
if ratings_df.empty:
    st.write("No players yet.")
else:
    # ensure numeric
    ratings_df["rating"] = pd.to_numeric(ratings_df["rating"], errors="coerce").fillna(1500.0)
    top3 = ratings_df.sort_values("rating", ascending=False).head(3).reset_index(drop=True)
    cols = st.columns(3)
    medals = ["🥇 #1", "🥈 #2", "🥉 #3"]
    for i in range(3):
        if i < len(top3):
            p = top3.loc[i, "player"]
            r = top3.loc[i, "rating"]
            wins = int(top3.loc[i, "wins"]) if "wins" in top3.columns else ""
            cols[i].metric(label=f"{medals[i]}  {p}", value=f"{r:.2f}", delta=f"W:{wins}")
        else:
            cols[i].write("—")

# --------------------------
# Team pair analysis
# --------------------------
st.header("🧩 Team (Pair) Analysis")
# build pair stats
pair_stats = {}
for _, row in matches_df.iterrows():
    A1 = normalize(row["playerA1"]); A2 = normalize(row["playerA2"])
    B1 = normalize(row["playerB1"]); B2 = normalize(row["playerB2"])
    # skip invalid entries
    if not (A1 and A2 and B1 and B2):
        continue
    keyA = tuple(sorted([A1, A2]))
    keyB = tuple(sorted([B1, B2]))
    for k in [keyA, keyB]:
        if k not in pair_stats:
            pair_stats[k] = {"matches": 0, "wins": 0, "losses": 0}
    pair_stats[keyA]["matches"] += 1
    pair_stats[keyB]["matches"] += 1
    # decide winner
    scA = safe_to_int(row["scoreA"]); scB = safe_to_int(row["scoreB"])
    if scA > scB:
        pair_stats[keyA]["wins"] += 1
        pair_stats[keyB]["losses"] += 1
    else:
        pair_stats[keyB]["wins"] += 1
        pair_stats[keyA]["losses"] += 1

# convert to df if pair_stats not empty, handle missing gracefully
pairs_list = []
for pair, s in pair_stats.items():
    matches_count = s.get("matches", 0)
    wins = s.get("wins", 0)
    losses = s.get("losses", 0)
    win_pct = round((wins / matches_count * 100) if matches_count else 0.0, 1)
    pairs_list.append({
        "team": " & ".join(pair),
        "matches": matches_count,
        "wins": wins,
        "losses": losses,
        "win_pct": win_pct
    })

if pairs_list:
    pairs_df = pd.DataFrame(pairs_list).sort_values("win_pct", ascending=False).reset_index(drop=True)
    st.subheader("Top Teams by Win %")
    st.dataframe(pairs_df.head(20))
else:
    st.write("No complete teams found yet.")

# --------------------------
# Rating trend (replay matches)
# --------------------------
st.header("📈 Rating Trend (replay matches chronologically)")
# get players list
players_all = sorted(ratings_df["player"].unique()) if not ratings_df.empty else []
selected = st.multiselect("Select players to show", players_all, default=players_all[:3])

# Replay matches in chronological order and capture snapshots
def build_timeline(selected_players):
    # starting ratings
    current = {}
    for p in selected_players:
        current[p] = 1500.0
    rows = []
    sorted_matches = matches_df.sort_values("date_parsed").reset_index(drop=True)
    for _, row in sorted_matches.iterrows():
        pA1 = normalize(row["playerA1"]); pA2 = normalize(row["playerA2"])
        pB1 = normalize(row["playerB1"]); pB2 = normalize(row["playerB2"])
        scA = safe_to_int(row["scoreA"]); scB = safe_to_int(row["scoreB"])
        # ensure all players present in current
        for p in [pA1, pA2, pB1, pB2]:
            if p and p not in current and p in players_all:
                current[p] = 1500.0
        # update ratings for this match
        current = update_elo(pA1, pA2, pB1, pB2, scA, scB, current)
        dt = row["date_parsed"]
        for p in selected_players:
            rows.append({"date": dt, "player": p, "rating": current.get(p, 1500.0)})
    if not rows and selected_players:
        rows = [{"date": pd.to_datetime(datetime.date.today()), "player": p, "rating": 1500.0} for p in selected_players]
    return pd.DataFrame(rows)

if selected:
    timeline_df = build_timeline(selected)
    if not timeline_df.empty:
        chart = alt.Chart(timeline_df).mark_line(point=True).encode(
            x=alt.X("date:T", title="Date"),
            y=alt.Y("rating:Q", title="Rating"),
            color="player:N",
            tooltip=["player:N", "date:T", alt.Tooltip("rating:Q", format=".2f")]
        ).interactive()
        st.altair_chart(chart, use_container_width=True)
    else:
        st.info("No timeline data to show.")
else:
    st.info("Select 1+ players to plot rating trend.")

# --------------------------
# Prediction UI (use latest ratings dict)
# --------------------------
st.header("🔮 Predict Match Outcome")
pc1, pc2 = st.columns(2)
pA1 = normalize(pc1.text_input("Team A - P1"))
pA2 = normalize(pc1.text_input("Team A - P2"))
pB1 = normalize(pc2.text_input("Team B - P1"))
pB2 = normalize(pc2.text_input("Team B - P2"))

if st.button("Predict Win Probability"):
    # ensure rating dict exists - recompute from ratings_df if needed
    if not ratings:
        # try to populate ratings from ratings_df
        for _, row in ratings_df.iterrows():
            ratings[normalize(row["player"])] = float(row["rating"])
    if all(x for x in [pA1, pA2, pB1, pB2]) and all(p in ratings for p in [pA1, pA2, pB1, pB2]):
        prob = predict_win_probability(ratings, pA1, pA2, pB1, pB2)
        st.success(f"Team A win probability: **{prob*100:.2f}%**")
    else:
        st.error("One or more players do not have ratings yet or inputs are missing.")

# --------------------------
# Footer: quick actions
# --------------------------
st.markdown("---")
st.caption("Notes: CSV files are stored in your GitHub repo. If multiple users write simultaneously GitHub SHA conflicts can happen — the app fetches latest SHA before saving but in high-concurrency cases a retry/merge strategy would be needed.")
