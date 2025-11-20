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

#
# -------------------------
# Helpers
# -------------------------
#
def normalize(name):
    """Normalize player names to Title case, safe for non-string input."""
    if not isinstance(name, str):
        return ""
    return name.strip().title()

def safe_int(x, default=0):
    try:
        return int(x)
    except:
        return default

#
# -------------------------
# GitHub helpers (read/write CSV robustly)
# -------------------------
#
def github_get_file(path):
    """Return tuple (df, sha). If file missing return (empty_df, None)."""
    api_url = f"https://api.github.com/repos/{st.secrets['REPO_OWNER']}/{st.secrets['REPO_NAME']}/contents/{path}"
    headers = {"Authorization": f"token {st.secrets['GITHUB_TOKEN']}"}
    res = requests.get(api_url, headers=headers)
    if res.status_code == 404:
        # Return empty df with correct columns depending on file
        if path == st.secrets["MATCHES_CSV"]:
            cols = ["date", "playerA1", "playerA2", "playerB1", "playerB2", "scoreA", "scoreB"]
        else:
            cols = ["player", "rating", "wins", "losses", "matches"]
        return pd.DataFrame(columns=cols), None
    res.raise_for_status()
    data = res.json()
    content = base64.b64decode(data["content"]).decode("utf-8")
    df = pd.read_csv(StringIO(content))
    return df, data.get("sha")

def github_put_file(path, df, sha=None, message="update csv"):
    """Save df to GitHub; if sha is None create new file, else update."""
    api_url = f"https://api.github.com/repos/{st.secrets['REPO_OWNER']}/{st.secrets['REPO_NAME']}/contents/{path}"
    headers = {"Authorization": f"token {st.secrets['GITHUB_TOKEN']}"}
    encoded = base64.b64encode(df.to_csv(index=False).encode()).decode()
    payload = {"message": message, "content": encoded}
    if sha:
        payload["sha"] = sha
    res = requests.put(api_url, headers=headers, json=payload)
    res.raise_for_status()
    return res.json()

#
# -------------------------
# Load CSVs from GitHub
# -------------------------
#
try:
    matches, matches_sha = github_get_file(st.secrets["MATCHES_CSV"])
except Exception as e:
    st.error("Unable to load matches.csv from GitHub. Check secrets/path/token.")
    st.stop()

try:
    ratings_df, ratings_sha = github_get_file(st.secrets["RATINGS_CSV"])
except Exception:
    ratings_df = pd.DataFrame(columns=["player", "rating", "wins", "losses", "matches"])
    ratings_sha = None

# Normalize matches columns (fill missing columns if any)
for c in ["date", "playerA1", "playerA2", "playerB1", "playerB2", "scoreA", "scoreB"]:
    if c not in matches.columns:
        matches[c] = "" if c == "date" else ""

# Clean whitespace and normalize player names for existing data
for c in ["playerA1", "playerA2", "playerB1", "playerB2"]:
    matches[c] = matches[c].fillna("").astype(str).apply(normalize)

# Score columns: coerce to int where possible, but keep zeros for missing
matches["scoreA"] = pd.to_numeric(matches["scoreA"], errors="coerce").fillna(0).astype(int)
matches["scoreB"] = pd.to_numeric(matches["scoreB"], errors="coerce").fillna(0).astype(int)

# Ratings df types and fallback
if ratings_df.empty:
    ratings_df = pd.DataFrame(columns=["player", "rating", "wins", "losses", "matches"])
else:
    ratings_df["player"] = ratings_df["player"].astype(str).apply(normalize)
    ratings_df["rating"] = pd.to_numeric(ratings_df["rating"], errors="coerce").fillna(1500.0)
    ratings_df["wins"] = pd.to_numeric(ratings_df["wins"], errors="coerce").fillna(0).astype(int)
    ratings_df["losses"] = pd.to_numeric(ratings_df["losses"], errors="coerce").fillna(0).astype(int)
    ratings_df["matches"] = pd.to_numeric(ratings_df["matches"], errors="coerce").fillna(0).astype(int)

# Build in-memory dicts
ratings = {row["player"]: float(row["rating"]) for _, row in ratings_df.iterrows()}
player_stats = {
    row["player"]: {"wins": int(row["wins"]), "losses": int(row["losses"]), "matches": int(row["matches"])}
    for _, row in ratings_df.iterrows()
}

#
# -------------------------
# UI: Add Match (inputs)
# -------------------------
#
st.header("➕ Add New Match")

with st.form("add_match_form"):
    c1, c2 = st.columns(2)
    a1 = c1.text_input("Team A - Player 1").strip()
    a2 = c2.text_input("Team A - Player 2").strip()
    b1 = c1.text_input("Team B - Player 1").strip()
    b2 = c2.text_input("Team B - Player 2").strip()

    # Scores as numeric inputs default 0 (user requested)
    sA = c1.number_input("Score A", min_value=0, value=0, step=1)
    sB = c2.number_input("Score B", min_value=0, value=0, step=1)

    match_date = c2.date_input("Match date", value=datetime.date.today())
    submitted = st.form_submit_button("Save Match")

if submitted:
    # Normalize names
    pA1 = normalize(a1)
    pA2 = normalize(a2)
    pB1 = normalize(b1)
    pB2 = normalize(b2)
    sA = safe_int(sA, 0)
    sB = safe_int(sB, 0)
    date_str = str(match_date)

    # Append to matches DataFrame
    new_row = {
        "date": date_str,
        "playerA1": pA1,
        "playerA2": pA2,
        "playerB1": pB1,
        "playerB2": pB2,
        "scoreA": sA,
        "scoreB": sB
    }
    matches = pd.concat([matches, pd.DataFrame([new_row])], ignore_index=True)

    # Save to GitHub (update matches)
    try:
        github_put_file(st.secrets["MATCHES_CSV"], matches, sha=matches_sha, message="Add match")
        # reload to refresh sha & content
        matches, matches_sha = github_get_file(st.secrets["MATCHES_CSV"])
    except Exception as e:
        st.error(f"Failed to save match to GitHub: {e}")
        st.stop()

    # Ensure players exist in ratings/stats
    for p in [pA1, pA2, pB1, pB2]:
        if p and p not in ratings:
            ratings[p] = 1500.0
        if p and p not in player_stats:
            player_stats[p] = {"wins": 0, "losses": 0, "matches": 0}

    # Update player match counts and wins/losses
    for p in [pA1, pA2, pB1, pB2]:
        if p:
            player_stats[p]["matches"] += 1

    if sA > sB:
        player_stats[pA1]["wins"] += 1
        player_stats[pA2]["wins"] += 1
        player_stats[pB1]["losses"] += 1
        player_stats[pB2]["losses"] += 1
    else:
        player_stats[pB1]["wins"] += 1
        player_stats[pB2]["wins"] += 1
        player_stats[pA1]["losses"] += 1
        player_stats[pA2]["losses"] += 1

    # Recompute ratings by replaying all matches in chronological order
    # (This avoids drift if you changed K or rating code)
    def get_parsed_matches(df):
        dfc = df.copy()
        if "date" in dfc.columns:
            dfc["date_parsed"] = pd.to_datetime(dfc["date"], errors="coerce")
        else:
            dfc["date_parsed"] = pd.NaT
        # for rows without dates, set monotonic increasing synthetic dates
        if dfc["date_parsed"].isna().all():
            dfc = dfc.reset_index(drop=True)
            dfc["date_parsed"] = pd.to_datetime("1970-01-01") + pd.to_timedelta(dfc.index, unit="D")
        else:
            # fill missing with end-of-data synthetic times
            nan_mask = dfc["date_parsed"].isna()
            if nan_mask.any():
                last = dfc["date_parsed"].max()
                fillers = pd.date_range(start=last + pd.Timedelta(days=1), periods=nan_mask.sum())
                dfc.loc[nan_mask, "date_parsed"] = fillers
        return dfc.sort_values("date_parsed").reset_index(drop=True)

    matches_sorted = get_parsed_matches(matches)

    # Recompute ratings from scratch
    ratings = {}
    for _, row in matches_sorted.iterrows():
        pA1 = normalize(row.get("playerA1", ""))
        pA2 = normalize(row.get("playerA2", ""))
        pB1 = normalize(row.get("playerB1", ""))
        pB2 = normalize(row.get("playerB2", ""))
        scA = safe_int(row.get("scoreA", 0), 0)
        scB = safe_int(row.get("scoreB", 0), 0)
        ratings = update_elo(pA1, pA2, pB1, pB2, scA, scB, ratings)

    # Build ratings_df and save to GitHub
    new_ratings_rows = []
    for p, r_rating in ratings.items():
        stats = player_stats.get(p, {"wins": 0, "losses": 0, "matches": 0})
        new_ratings_rows.append({
            "player": p,
            "rating": round(float(r_rating), 2),
            "wins": int(stats.get("wins", 0)),
            "losses": int(stats.get("losses", 0)),
            "matches": int(stats.get("matches", 0))
        })
    ratings_df = pd.DataFrame(new_ratings_rows).sort_values("rating", ascending=False).reset_index(drop=True)

    try:
        github_put_file(st.secrets["RATINGS_CSV"], ratings_df, sha=ratings_sha, message="Update ratings")
        ratings_df, ratings_sha = github_get_file(st.secrets["RATINGS_CSV"])
    except Exception as e:
        st.error(f"Failed to save ratings to GitHub: {e}")
        st.stop()

    st.success("Match saved and ratings updated.")

#
# -------------------------
# Match history display & filters
# -------------------------
#
# Normalize matches and parse dates for filtering
matches = matches.fillna("")
for c in ["playerA1", "playerA2", "playerB1", "playerB2"]:
    matches[c] = matches[c].astype(str).apply(normalize)

# Parse/ensure dates
if "date" not in matches.columns:
    matches["date"] = ""
matches["date_parsed"] = pd.to_datetime(matches["date"], errors="coerce")
# Fill fully-missing dates with synthetic timeline
if matches["date_parsed"].isna().all():
    matches = matches.reset_index(drop=True)
    matches["date_parsed"] = pd.to_datetime("1970-01-01") + pd.to_timedelta(matches.index, unit="D")
else:
    # fill missing ones after the max date
    nan_mask = matches["date_parsed"].isna()
    if nan_mask.any():
        last = matches["date_parsed"].max()
        fillers = pd.date_range(start=last + pd.Timedelta(days=1), periods=nan_mask.sum())
        matches.loc[nan_mask, "date_parsed"] = fillers

st.header("📜 Match History & Filters")

# Date filter widget: supply Python date objects
min_date = matches["date_parsed"].min()
max_date = matches["date_parsed"].max()
if pd.isna(min_date):
    min_date_py = datetime.date.today()
else:
    min_date_py = min_date.date()
if pd.isna(max_date):
    max_date_py = datetime.date.today()
else:
    max_date_py = max_date.date()

colf1, colf2, colf3 = st.columns([2,2,1])
with colf1:
    start_date = st.date_input("From", value=min_date_py)
with colf2:
    end_date = st.date_input("To", value=max_date_py)
with colf3:
    _ = st.button("Apply filter")

# convert to datetime range (include end day fully)
start_dt = pd.to_datetime(start_date)
end_dt = pd.to_datetime(end_date) + pd.Timedelta(days=1) - pd.Timedelta(seconds=1)

filtered = matches[(matches["date_parsed"] >= start_dt) & (matches["date_parsed"] <= end_dt)].copy()

# Display scores as empty strings when 0
display = filtered.copy()
display["scoreA"] = display["scoreA"].replace(0, "")
display["scoreB"] = display["scoreB"].replace(0, "")

cols_to_show = [c for c in ["date", "playerA1", "playerA2", "playerB1", "playerB2", "scoreA", "scoreB"] if c in display.columns]
st.subheader(f"Showing {len(display)} matches between {start_date} and {end_date}")
st.dataframe(display[cols_to_show].sort_values("date", ascending=False))

#
# -------------------------
# Player individual stats (from ratings_df)
# -------------------------
#
st.header("📊 Player Individual Stats")

# Ensure numeric types
if not ratings_df.empty:
    ratings_df["rating"] = pd.to_numeric(ratings_df["rating"], errors="coerce").fillna(1500.0)
    for col in ["wins", "losses", "matches"]:
        ratings_df[col] = pd.to_numeric(ratings_df[col], errors="coerce").fillna(0).astype(int)
else:
    ratings_df = pd.DataFrame(columns=["player", "rating", "wins", "losses", "matches"])

ratings_df["Win %"] = ((ratings_df["wins"] / ratings_df["matches"]).replace([np.inf, -np.inf], 0).fillna(0) * 100).round(1)
st.dataframe(ratings_df.sort_values("rating", ascending=False).reset_index(drop=True))

#
# -------------------------
# Top players badges
# -------------------------
#
st.header("🏆 Top Players")
top3 = ratings_df.sort_values("rating", ascending=False).head(3).reset_index(drop=True)
cols = st.columns(3)
badges = ["🥇 #1", "🥈 #2", "🥉 #3"]
for i in range(3):
    if i < len(top3):
        p = top3.loc[i, "player"]
        r = top3.loc[i, "rating"]
        w = top3.loc[i, "wins"]
        cols[i].metric(label=f"{badges[i]}  {p}", value=f"{r:.2f}", delta=f"W:{w}")
    else:
        cols[i].write("—")

#
# -------------------------
# Team (pair) win/loss analysis
# -------------------------
#
st.header("🧩 Team (Pair) Analysis")

def pair_key(p1, p2):
    a = normalize(p1); b = normalize(p2)
    return tuple(sorted([a, b]))

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

pairs_list = []
for pair, s in pair_stats.items():
    matches_cnt = s["matches"]
    wins_cnt = s["wins"]
    pairs_list.append({
        "team": " & ".join(pair),
        "matches": matches_cnt,
        "wins": wins_cnt,
        "losses": s["losses"],
        "win_pct": round((wins_cnt / matches_cnt * 100) if matches_cnt else 0, 1)
    })

pairs_df = pd.DataFrame(pairs_list) if pairs_list else pd.DataFrame(columns=["team", "matches", "wins", "losses", "win_pct"])
st.subheader("Top Teams by Win %")
st.dataframe(pairs_df.sort_values("win_pct", ascending=False).reset_index(drop=True))

#
# -------------------------
# Rating trend graph (replay matches)
# -------------------------
#
st.header("📈 Rating Trend (replay matches chronologically)")

players_all = sorted(list(ratings_df["player"].unique()))
selected_players = st.multiselect("Select players to plot", players_all, default=players_all[:3] if players_all else [])

matches_sorted = matches.sort_values("date_parsed").reset_index(drop=True)

def compute_rating_timeline(selected_players):
    cur = {}
    timeline_rows = []
    for _, row in matches_sorted.iterrows():
        pA1 = normalize(row["playerA1"]); pA2 = normalize(row["playerA2"])
        pB1 = normalize(row["playerB1"]); pB2 = normalize(row["playerB2"])
        scA = safe_int(row["scoreA"], 0); scB = safe_int(row["scoreB"], 0)
        # ensure players exist
        for p in [pA1, pA2, pB1, pB2]:
            if p and p not in cur:
                cur[p] = 1500.0
        cur = update_elo(pA1, pA2, pB1, pB2, scA, scB, cur)
        for p in selected_players:
            timeline_rows.append({"date": row["date_parsed"], "player": p, "rating": cur.get(p, 1500.0)})
    if not timeline_rows:
        # fallback single point
        today = pd.to_datetime(datetime.date.today())
        for p in selected_players:
            timeline_rows.append({"date": today, "player": p, "rating": 1500.0})
    df_t = pd.DataFrame(timeline_rows)
    df_t = df_t.sort_values(["player", "date"]).reset_index(drop=True)
    return df_t

if selected_players:
    tdf = compute_rating_timeline(selected_players)
    chart = alt.Chart(tdf).mark_line(point=True).encode(
        x=alt.X("date:T", title="Date"),
        y=alt.Y("rating:Q", title="Rating"),
        color="player:N",
        tooltip=["player:N", alt.Tooltip("rating:Q", format=".2f"), "date:T"]
    ).interactive()
    st.altair_chart(chart, use_container_width=True)
else:
    st.info("Choose players to plot rating trends.")

#
# -------------------------
# Predictor
# -------------------------
#
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
        st.error("One or more players don't have ratings yet (add some matches first).")
