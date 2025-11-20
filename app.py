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
# GitHub helpers
# ------------------------
GITHUB_TOKEN = st.secrets["GITHUB_TOKEN"]
OWNER = st.secrets["REPO_OWNER"]
REPO = st.secrets["REPO_NAME"]
MATCHES_PATH = st.secrets["MATCHES_CSV"]
RATINGS_PATH = st.secrets["RATINGS_CSV"]
API_BASE = f"https://api.github.com/repos/{OWNER}/{REPO}/contents"
HEADERS = {"Authorization": f"token {GITHUB_TOKEN}"}

MATCHES_COLS = ["date", "playerA1", "playerA2", "playerB1", "playerB2", "scoreA", "scoreB"]
RATINGS_COLS = ["player", "rating", "wins", "losses", "matches"]

def github_get_csv(path, default_columns=None):
    url = f"{API_BASE}/{path}"
    resp = requests.get(url, headers=HEADERS)
    if resp.status_code == 200:
        info = resp.json()
        content = base64.b64decode(info.get("content", "")).decode("utf-8")
        df = pd.read_csv(StringIO(content))
        return df, info.get("sha")
    elif resp.status_code == 404:
        if default_columns:
            return pd.DataFrame(columns=default_columns), None
        return pd.DataFrame(), None
    else:
        st.error(f"GitHub API error {resp.status_code} while reading {path}: {resp.text}")
        st.stop()

def github_put_csv(path, df, sha=None, message="update csv"):
    url = f"{API_BASE}/{path}"
    content = df.to_csv(index=False)
    encoded = base64.b64encode(content.encode()).decode()
    payload = {"message": message, "content": encoded}
    if sha:
        payload["sha"] = sha
    resp = requests.put(url, headers=HEADERS, json=payload)
    if resp.status_code in (200, 201):
        return resp.json().get("content", {}).get("sha")
    else:
        st.error(f"GitHub API error {resp.status_code} while writing {path}: {resp.text}")
        st.stop()

# ------------------------
# Load CSVs (safe)
# ------------------------
matches, matches_sha = github_get_csv(MATCHES_PATH, default_columns=MATCHES_COLS)
ratings_df, ratings_sha = github_get_csv(RATINGS_PATH, default_columns=RATINGS_COLS)

# Ensure columns exist
for c in MATCHES_COLS:
    if c not in matches.columns:
        matches[c] = ""

for c in RATINGS_COLS:
    if c not in ratings_df.columns:
        ratings_df[c] = []

# Normalize names (strings) in matches and keep score text columns (so they appear empty in UI)
for c in ["playerA1", "playerA2", "playerB1", "playerB2", "date"]:
    if c in matches.columns:
        matches[c] = matches[c].fillna("").astype(str).apply(normalize)

# numeric helper columns for computations
matches["scoreA_num"] = pd.to_numeric(matches.get("scoreA", ""), errors="coerce").fillna(0).astype(int)
matches["scoreB_num"] = pd.to_numeric(matches.get("scoreB", ""), errors="coerce").fillna(0).astype(int)

# Ensure ratings_df numeric types
if not ratings_df.empty:
    ratings_df["rating"] = pd.to_numeric(ratings_df["rating"], errors="coerce").fillna(1500.0)
    ratings_df["wins"] = pd.to_numeric(ratings_df["wins"], errors="coerce").fillna(0).astype(int)
    ratings_df["losses"] = pd.to_numeric(ratings_df["losses"], errors="coerce").fillna(0).astype(int)
    ratings_df["matches"] = pd.to_numeric(ratings_df["matches"], errors="coerce").fillna(0).astype(int)

# Build ratings dict for quick lookups (may be recalculated below)
ratings = {}
player_stats = {}
if not ratings_df.empty:
    for _, r in ratings_df.iterrows():
        p = normalize(r["player"])
        ratings[p] = float(r["rating"])
        player_stats[p] = {"wins": int(r["wins"]), "losses": int(r["losses"]), "matches": int(r["matches"])}

# ------------------------
# Add Match Form (clear_on_submit=True so inputs clear after save)
# ------------------------
st.header("➕ Add New Match")

with st.form("add_match", clear_on_submit=True):
    left, right = st.columns(2)
    a1 = normalize(left.text_input("Team A - Player 1", value=""))
    a2 = normalize(right.text_input("Team A - Player 2", value=""))
    b1 = normalize(left.text_input("Team B - Player 1", value=""))
    b2 = normalize(right.text_input("Team B - Player 2", value=""))

    # use text_input so it appears empty by default — treat blank as 0 when saving
    sa_text = left.text_input("Score A (leave empty = 0)", value="", key="sa")
    sb_text = right.text_input("Score B (leave empty = 0)", value="", key="sb")

    match_date = right.date_input("Match date", value=datetime.date.today())
    add_clicked = st.form_submit_button("Add match")

if add_clicked:
    if not any([a1, a2, b1, b2]):
        st.error("Please enter at least one player name.")
    else:
        sA = safe_int_from_text(sa_text)
        sB = safe_int_from_text(sb_text)

        # Store blank string in CSV if user left the score blank
        scoreA_csv = "" if sa_text.strip() == "" else str(sA)
        scoreB_csv = "" if sb_text.strip() == "" else str(sB)

        new = {
            "date": match_date.strftime("%Y-%m-%d"),
            "playerA1": a1, "playerA2": a2,
            "playerB1": b1, "playerB2": b2,
            "scoreA": scoreA_csv, "scoreB": scoreB_csv
        }
        matches = pd.concat([matches, pd.DataFrame([new])], ignore_index=True)

        # update numeric helper columns
        matches["scoreA_num"] = pd.to_numeric(matches.get("scoreA", ""), errors="coerce").fillna(0).astype(int)
        matches["scoreB_num"] = pd.to_numeric(matches.get("scoreB", ""), errors="coerce").fillna(0).astype(int)

        # Save matches.csv (fetch latest sha on each write)
        try:
            # get latest sha fresh to reduce conflicts
            _, current_sha = github_get_csv(MATCHES_PATH, default_columns=MATCHES_COLS)
            matches_sha = github_put_csv(MATCHES_PATH, matches[MATCHES_COLS], current_sha, "Add match")
        except Exception as e:
            st.error("Failed to save matches to GitHub.")
            st.stop()

        st.success("Match saved — now updating ratings...")

        # Recompute all ratings & stats by replaying matches in chronological order
        # Parse date column and sort; if dates invalid, use current order
        m_copy = matches.copy()
        m_copy["date_parsed"] = pd.to_datetime(m_copy["date"], errors="coerce")
        if m_copy["date_parsed"].isna().all():
            m_sorted = m_copy.reset_index(drop=True)
        else:
            m_sorted = m_copy.sort_values("date_parsed").reset_index(drop=True)

        ratings_replay = {}
        stats_replay = {}

        for _, row in m_sorted.iterrows():
            pA1 = normalize(row.get("playerA1", ""))
            pA2 = normalize(row.get("playerA2", ""))
            pB1 = normalize(row.get("playerB1", ""))
            pB2 = normalize(row.get("playerB2", ""))
            scA = safe_int_from_text(row.get("scoreA", ""))
            scB = safe_int_from_text(row.get("scoreB", ""))

            # initialize players
            for p in [pA1, pA2, pB1, pB2]:
                if not p:
                    continue
                ratings_replay.setdefault(p, 1500.0)
                stats_replay.setdefault(p, {"wins": 0, "losses": 0, "matches": 0})

            # increment matches
            for p in [pA1, pA2, pB1, pB2]:
                if p:
                    stats_replay[p]["matches"] += 1

            # update wins/losses (ties treated as B win — consistent with previous logic)
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

        # Build ratings_df and save
        rows = []
        for p, r in ratings_replay.items():
            st_w = stats_replay.get(p, {}).get("wins", 0)
            st_l = stats_replay.get(p, {}).get("losses", 0)
            st_m = stats_replay.get(p, {}).get("matches", 0)
            rows.append({"player": p, "rating": round(r,2), "wins": int(st_w), "losses": int(st_l), "matches": int(st_m)})
        ratings_df = pd.DataFrame(rows).sort_values("rating", ascending=False).reset_index(drop=True)

        try:
            _, current_ratings_sha = github_get_csv(RATINGS_PATH, default_columns=RATINGS_COLS)
            ratings_sha = github_put_csv(RATINGS_PATH, ratings_df[RATINGS_COLS], current_ratings_sha, "Update ratings")
        except Exception:
            st.error("Failed to save ratings to GitHub.")
            st.stop()

        # refresh local ratings and matches helpers
        ratings = {r["player"]: float(r["rating"]) for _, r in ratings_df.iterrows()}
        player_stats = {r["player"]: {"wins": int(r["wins"]), "losses": int(r["losses"]), "matches": int(r["matches"])} for _, r in ratings_df.iterrows()}

        # reload matches to ensure fresh state
        matches, matches_sha = github_get_csv(MATCHES_PATH, default_columns=MATCHES_COLS)
        for c in MATCHES_COLS:
            if c not in matches.columns:
                matches[c] = ""
        matches["scoreA_num"] = pd.to_numeric(matches.get("scoreA",""), errors="coerce").fillna(0).astype(int)
        matches["scoreB_num"] = pd.to_numeric(matches.get("scoreB",""), errors="coerce").fillna(0).astype(int)

# ------------------------
# Prepare matches for display
# ------------------------
for c in MATCHES_COLS:
    if c not in matches.columns:
        matches[c] = ""

matches = matches.reset_index(drop=True)
matches["scoreA_num"] = pd.to_numeric(matches.get("scoreA",""), errors="coerce").fillna(0).astype(int)
matches["scoreB_num"] = pd.to_numeric(matches.get("scoreB",""), errors="coerce").fillna(0).astype(int)

# parse dates safely
def parse_dates_safe(df):
    df = df.copy()
    if "date" in df.columns:
        df["date_parsed"] = pd.to_datetime(df["date"], errors="coerce")
        if df["date_parsed"].isna().all():
            df["date_parsed"] = pd.to_datetime("1970-01-01") + pd.to_timedelta(df.index, unit="D")
    else:
        df["date_parsed"] = pd.to_datetime("1970-01-01") + pd.to_timedelta(df.index, unit="D")
    return df

matches = parse_dates_safe(matches)

# ------------------------
# Filters and Match History display
# ------------------------
st.header("📜 Match History & Filters")

min_date = matches["date_parsed"].min()
max_date = matches["date_parsed"].max()
if pd.isna(min_date):
    min_date = datetime.date.today()
if pd.isna(max_date):
    max_date = datetime.date.today()

col1, col2, col3 = st.columns([2,2,1])
with col1:
    start_date = st.date_input("From", value=min_date.date() if hasattr(min_date, "date") else datetime.date.today())
with col2:
    end_date = st.date_input("To", value=max_date.date() if hasattr(max_date, "date") else datetime.date.today())
with col3:
    if st.button("Apply filter"):
        pass

start_dt = pd.to_datetime(start_date)
end_dt = pd.to_datetime(end_date) + pd.Timedelta(days=1) - pd.Timedelta(seconds=1)
filtered = matches[(matches["date_parsed"] >= start_dt) & (matches["date_parsed"] <= end_dt)]

cols_to_show = [c for c in ["date","playerA1","playerA2","playerB1","playerB2","scoreA","scoreB","scoreA_num","scoreB_num"] if c in filtered.columns]

st.subheader(f"Showing {len(filtered)} matches between {start_date} and {end_date}")
# ensure date_parsed exists before sorting
if "date_parsed" not in filtered.columns:
    filtered = parse_dates_safe(filtered)
st.dataframe(filtered[cols_to_show].sort_values("date_parsed", ascending=False).reset_index(drop=True))

# ------------------------
# Player individual stats
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
        cols[i].metric(f"{medals[i]} {row['player']}", f"{float(row['rating']):.2f}", delta=f"W:{int(row.get('wins',0))}")
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
    ms = matches.copy().sort_values("date_parsed").reset_index(drop=True)
    current = {}
    for _, row in ms.iterrows():
        pA1 = normalize(row.get("playerA1","")); pA2 = normalize(row.get("playerA2",""))
        pB1 = normalize(row.get("playerB1","")); pB2 = normalize(row.get("playerB2",""))
        sA = safe_int_from_text(row.get("scoreA","")); sB = safe_int_from_text(row.get("scoreB",""))
        for p in [pA1,pA2,pB1,pB2]:
            if p and p not in current:
                current[p] = 1500.0
        current = update_elo(pA1,pA2,pB1,pB2,sA,sB,current)
        for p in selected_players:
            timeline_rows.append({"date": row["date_parsed"], "player": p, "rating": float(current.get(p,1500.0))})
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
        st.info("No timeline data to show.")
else:
    st.info("Select players to show rating trend.")

# ------------------------
# Prediction
# ------------------------
st.header("🔮 Predict Match Outcome")
pc1, pc2 = st.columns(2)
ppA1 = normalize(pc1.text_input("Team A - P1", value=""))
ppA2 = normalize(pc1.text_input("Team A - P2", value=""))
ppB1 = normalize(pc2.text_input("Team B - P1", value=""))
ppB2 = normalize(pc2.text_input("Team B - P2", value=""))

if st.button("Predict"):
    if all(pp in ratings for pp in [ppA1,ppA2,ppB1,ppB2]) and all([ppA1,ppA2,ppB1,ppB2]):
        prob = predict_win_probability(ratings, ppA1, ppA2, ppB1, ppB2)
        st.success(f"Team A win probability: **{prob*100:.2f}%**")
    else:
        st.error("Make sure all four players exist with ratings.")

st.write("App synced with GitHub CSV files.")
