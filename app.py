# app.py (final, mobile-optimized)
import streamlit as st
import pandas as pd
import numpy as np
import requests
import base64
from io import StringIO
import altair as alt
import datetime
from elo import update_elo, predict_win_probability

# ---------- Page config ----------
st.set_page_config(page_title="Badminton Doubles Tracker", layout="wide")
st.markdown("<h1 style='text-align:left'>🏸 Badminton Doubles Tracker</h1>", unsafe_allow_html=True)

# ---------- Small responsive CSS ----------
st.markdown(
    """
    <style>
    /* Make table wrap on small screens */
    @media (max-width: 600px) {
      .stDataFrame table {font-size: 12px;}
    }
    .center {display:flex; justify-content: center; align-items: center}
    </style>
    """,
    unsafe_allow_html=True,
)

# ---------- Helpers ----------
def normalize(name):
    if not isinstance(name, str):
        return ""
    return name.strip().title()

def safe_int_from_text(s):
    """Text input may be empty — treat empty as 0."""
    if s is None:
        return 0
    s = str(s).strip()
    if s == "":
        return 0
    try:
        return int(float(s))
    except:
        return 0

# ---------- GitHub helpers ----------
GITHUB_TOKEN = st.secrets["GITHUB_TOKEN"]
OWNER = st.secrets["REPO_OWNER"]
REPO = st.secrets["REPO_NAME"]
MATCHES_PATH = st.secrets["MATCHES_CSV"]
RATINGS_PATH = st.secrets["RATINGS_CSV"]

API_BASE = f"https://api.github.com/repos/{OWNER}/{REPO}/contents"
HEADERS = {"Authorization": f"token {GITHUB_TOKEN}"}

def github_get_csv(path, default_columns=None):
    url = f"{API_BASE}/{path}"
    resp = requests.get(url, headers=HEADERS)
    if resp.status_code == 200:
        info = resp.json()
        content_b64 = info.get("content", "")
        content = base64.b64decode(content_b64).decode("utf-8")
        df = pd.read_csv(StringIO(content))
        return df, info.get("sha")
    elif resp.status_code == 404:
        # return empty df with default columns
        if default_columns:
            return pd.DataFrame(columns=default_columns), None
        return pd.DataFrame(), None
    else:
        st.error(f"GitHub read error {resp.status_code}: {resp.text}")
        st.stop()

def github_put_csv(path, df, sha=None, message="update csv"):
    url = f"{API_BASE}/{path}"
    content = df.to_csv(index=False)
    encoded = base64.b64encode(content.encode()).decode()
    payload = {"message": message, "content": encoded}
    if sha:
        payload["sha"] = sha
    resp = requests.put(url, headers=HEADERS, json=payload)
    if resp.status_code in (200,201):
        return resp.json().get("content", {}).get("sha")
    else:
        st.error(f"GitHub write error {resp.status_code}: {resp.text}")
        st.stop()

# ---------- Load CSVs (safe defaults) ----------
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

# Normalize text columns
for c in ["playerA1","playerA2","playerB1","playerB2","date"]:
    if c in matches.columns:
        matches[c] = matches[c].fillna("").astype(str).apply(normalize)

# Numeric helper versions
matches["scoreA_num"] = pd.to_numeric(matches["scoreA"], errors="coerce").fillna(0).astype(int)
matches["scoreB_num"] = pd.to_numeric(matches["scoreB"], errors="coerce").fillna(0).astype(int)

# Ratings numeric coercion
if not ratings_df.empty:
    ratings_df["rating"] = pd.to_numeric(ratings_df["rating"], errors="coerce").fillna(1500.0)
    ratings_df["wins"] = pd.to_numeric(ratings_df["wins"], errors="coerce").fillna(0).astype(int)
    ratings_df["losses"] = pd.to_numeric(ratings_df["losses"], errors="coerce").fillna(0).astype(int)
    ratings_df["matches"] = pd.to_numeric(ratings_df["matches"], errors="coerce").fillna(0).astype(int)

# Build runtime dicts
ratings = {}
player_stats = {}
if not ratings_df.empty:
    for _, r in ratings_df.iterrows():
        p = normalize(r["player"])
        ratings[p] = float(r["rating"])
        player_stats[p] = {"wins": int(r["wins"]), "losses": int(r["losses"]), "matches": int(r["matches"])}

# ---------- UI: Add Match (collapsible to save vertical space on mobile) ----------
with st.expander("➕ Add New Match", expanded=True):
    # Use columns but in mobile expanders collapse naturally
    with st.form("add_match", clear_on_submit=False):
        col1, col2 = st.columns(2)
        a1 = normalize(col1.text_input("Team A - Player 1", value=""))
        a2 = normalize(col2.text_input("Team A - Player 2", value=""))
        b1 = normalize(col1.text_input("Team B - Player 1", value=""))
        b2 = normalize(col2.text_input("Team B - Player 2", value=""))
        # score as text so it starts empty; we'll convert on save
        sa_key = "sa_input"
        sb_key = "sb_input"
        if sa_key not in st.session_state:
            st.session_state[sa_key] = ""
        if sb_key not in st.session_state:
            st.session_state[sb_key] = ""
        sA_text = col1.text_input("Score A (leave empty = 0)", value=st.session_state[sa_key], key=sa_key)
        sB_text = col2.text_input("Score B (leave empty = 0)", value=st.session_state[sb_key], key=sb_key)
        match_date = col2.date_input("Match date", value=datetime.date.today())
        submitted = st.form_submit_button("Save Match")

    if submitted:
        # Basic validation
        if not any([a1, a2, b1, b2]):
            st.error("Please enter at least one player name.")
        else:
            # Convert scores
            sA = safe_int_from_text(sA_text)
            sB = safe_int_from_text(sB_text)
            # Prepare row: keep score text empty if user left blank (for UI)
            scoreA_val = "" if str(sA_text).strip()=="" else str(sA)
            scoreB_val = "" if str(sB_text).strip()=="" else str(sB)
            new_row = {
                "date": match_date.strftime("%Y-%m-%d"),
                "playerA1": a1,
                "playerA2": a2,
                "playerB1": b1,
                "playerB2": b2,
                "scoreA": scoreA_val,
                "scoreB": scoreB_val
            }
            # Append and update numeric helper
            matches = pd.concat([matches, pd.DataFrame([new_row])], ignore_index=True)
            matches["scoreA_num"] = pd.to_numeric(matches["scoreA"], errors="coerce").fillna(0).astype(int)
            matches["scoreB_num"] = pd.to_numeric(matches["scoreB"], errors="coerce").fillna(0).astype(int)

            # Save matches CSV to GitHub and update sha
            new_sha = github_put_csv(MATCHES_PATH, matches[MATCHES_COLS], matches_sha, "Add match")
            matches_sha = new_sha

            # Recompute ratings & stats by replaying matches in chronological order
            # Parse dates safely and sort; fallback to original order if dates invalid
            df_sorted = matches.copy()
            df_sorted["date_parsed"] = pd.to_datetime(df_sorted["date"], errors="coerce")
            if df_sorted["date_parsed"].isna().all():
                df_sorted = df_sorted.reset_index(drop=True)
            else:
                df_sorted = df_sorted.sort_values("date_parsed").reset_index(drop=True)

            ratings_replay = {}
            stats_replay = {}

            for _, r in df_sorted.iterrows():
                pA1 = normalize(r.get("playerA1",""))
                pA2 = normalize(r.get("playerA2",""))
                pB1 = normalize(r.get("playerB1",""))
                pB2 = normalize(r.get("playerB2",""))
                scA = safe_int_from_text(r.get("scoreA",""))
                scB = safe_int_from_text(r.get("scoreB",""))

                # init
                for p in [pA1,pA2,pB1,pB2]:
                    if not p: 
                        continue
                    ratings_replay.setdefault(p, 1500.0)
                    stats_replay.setdefault(p, {"wins":0,"losses":0,"matches":0})

                # matches count
                for p in [pA1,pA2,pB1,pB2]:
                    if p:
                        stats_replay[p]["matches"] += 1

                # wins/losses
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

            # Save ratings.csv
            rows = []
            for p, r in ratings_replay.items():
                rows.append({
                    "player": p,
                    "rating": round(r, 2),
                    "wins": stats_replay.get(p, {}).get("wins", 0),
                    "losses": stats_replay.get(p, {}).get("losses", 0),
                    "matches": stats_replay.get(p, {}).get("matches", 0)
                })
            new_ratings_df = pd.DataFrame(rows).sort_values("rating", ascending=False).reset_index(drop=True)
            new_ratings_sha = github_put_csv(RATINGS_PATH, new_ratings_df[RATINGS_COLS], ratings_sha, "Update ratings")
            ratings_sha = new_ratings_sha

            # Update local runtime structures
            ratings = {r["player"]: float(r["rating"]) for _, r in new_ratings_df.iterrows()}
            player_stats = {r["player"]: {"wins":int(r["wins"]), "losses":int(r["losses"]), "matches":int(r["matches"])} for _, r in new_ratings_df.iterrows()}
            ratings_df = new_ratings_df

            # Clear score inputs (so they appear empty)
            st.session_state[sa_key] = ""
            st.session_state[sb_key] = ""

            st.success("Match added and ratings updated!")

# ---------- Prepare matches for display ----------
# Ensure expected columns exist
for c in MATCHES_COLS:
    if c not in matches.columns:
        matches[c] = ""

matches = matches.reset_index(drop=True)
matches["scoreA_num"] = pd.to_numeric(matches["scoreA"], errors="coerce").fillna(0).astype(int)
matches["scoreB_num"] = pd.to_numeric(matches["scoreB"], errors="coerce").fillna(0).astype(int)

# Parse date column for sorting — fallback to synthetic index-based dates
def parse_date_column(df):
    if "date" in df.columns:
        df["date_parsed"] = pd.to_datetime(df["date"], errors="coerce")
        if df["date_parsed"].isna().all():
            df["date_parsed"] = pd.to_datetime("1970-01-01") + pd.to_timedelta(df.index, unit="D")
    else:
        df["date_parsed"] = pd.to_datetime("1970-01-01") + pd.to_timedelta(df.index, unit="D")
    return df

matches = parse_date_column(matches)

# ---------- Match Filters & History (expander mobile-friendly) ----------
with st.expander("📜 Match History & Filters", expanded=True):
    min_date = matches["date_parsed"].min()
    max_date = matches["date_parsed"].max()
    if pd.isna(min_date): min_date = datetime.date.today()
    if pd.isna(max_date): max_date = datetime.date.today()

    c1, c2, c3 = st.columns([2,2,1])
    with c1:
        start_date = st.date_input("From", value=min_date.date() if hasattr(min_date,"date") else datetime.date.today())
    with c2:
        end_date = st.date_input("To", value=max_date.date() if hasattr(max_date,"date") else datetime.date.today())
    with c3:
        st.write("")  # placeholder for spacing

    start_dt = pd.to_datetime(start_date)
    end_dt = pd.to_datetime(end_date) + pd.Timedelta(days=1) - pd.Timedelta(seconds=1)

    filtered = matches[(matches["date_parsed"] >= start_dt) & (matches["date_parsed"] <= end_dt)]

    # Columns to show (explicitly exclude internal 'date_parsed')
    cols_to_show = [c for c in ["date","playerA1","playerA2","playerB1","playerB2","scoreA","scoreB"] if c in filtered.columns]

    st.subheader(f"Showing {len(filtered)} matches between {start_date} and {end_date}")
    # Sort by date_parsed for display, but don't include it in the visible table (avoids KeyError)
    if "date_parsed" in filtered.columns:
        display_df = filtered.sort_values("date_parsed", ascending=False).reset_index(drop=True)
    else:
        display_df = filtered.reset_index(drop=True)
    st.dataframe(display_df[cols_to_show])

# ---------- Player individual stats ----------
with st.expander("📊 Player Statistics (individual)", expanded=True):
    if (ratings_df is None) or ratings_df.empty:
        st.info("No player stats yet.")
    else:
        stats_df = ratings_df.copy()
        for col in ["rating","wins","losses","matches"]:
            stats_df[col] = pd.to_numeric(stats_df[col], errors="coerce").fillna(0)
        stats_df["Win %"] = ((stats_df["wins"] / stats_df["matches"]) * 100).replace([np.inf, -np.inf], 0).fillna(0).round(1)
        stats_df = stats_df[["player","rating","matches","wins","losses","Win %"]].sort_values("rating", ascending=False).reset_index(drop=True)
        st.dataframe(stats_df)

# ---------- Top 3 badges ----------
with st.expander("🏆 Top Players", expanded=True):
    top = ratings_df.sort_values("rating", ascending=False).reset_index(drop=True) if not ratings_df.empty else pd.DataFrame()
    cols = st.columns(3)
    medals = ["🥇 #1", "🥈 #2", "🥉 #3"]
    for i in range(3):
        if i < len(top):
            row = top.loc[i]
            cols[i].metric(f"{medals[i]} {row['player']}", f"{float(row['rating']):.2f}", delta=f"W:{int(row['wins'])}")
        else:
            cols[i].write("—")

# ---------- Rating trend ----------
with st.expander("📈 Rating Trend (replay matches)", expanded=False):
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
            for p in [pA1,pA2,pB1,pB2]:
                if p and p not in current:
                    current[p] = 1500.0
            current = update_elo(pA1,pA2,pB1,pB2,sA,sB,current)
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
            st.info("No timeline data yet.")
    else:
        st.info("Pick players to plot rating trend.")

# ---------- Predictor (small expander) ----------
with st.expander("🔮 Predict Match Outcome", expanded=False):
    pcol1, pcol2 = st.columns(2)
    pA1 = normalize(pcol1.text_input("Team A - P1", value=""))
    pA2 = normalize(pcol1.text_input("Team A - P2", value=""))
    pB1 = normalize(pcol2.text_input("Team B - P1", value=""))
    pB2 = normalize(pcol2.text_input("Team B - P2", value=""))

    if st.button("Predict"):
        # ensure ratings up-to-date (ratings dict built from latest ratings_df)
        ratings_local = {r["player"]: float(r["rating"]) for _, r in ratings_df.iterrows()} if not ratings_df.empty else {}
        if all(x and x in ratings_local for x in [pA1,pA2,pB1,pB2]):
            prob = predict_win_probability(ratings_local, pA1, pA2, pB1, pB2)
            st.success(f"Team A win probability: **{prob*100:.2f}%**")
        else:
            st.error("Make sure all players exist and have ratings.")

# ---------- Footer ----------
st.markdown("---")
st.caption("App synced to GitHub CSV files (matches.csv & ratings.csv).")
