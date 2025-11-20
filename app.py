# app.py
import streamlit as st
import pandas as pd
import numpy as np
import requests
import base64
from io import StringIO
import altair as alt
import datetime
import pytz
from elo import update_elo, predict_win_probability
# -------------------------------------------------------
#  PERSISTENT LOGIN USING SESSION_STATE + QUERY PARAMS
# -------------------------------------------------------

# Load saved login state from query params if available
params = st.query_params
if "auth" in params and params["auth"] == "1":
    st.session_state.logged_in = True

# Initialize session state on first run
if "logged_in" not in st.session_state:
    st.session_state.logged_in = False

# ---------------- LOGIN PAGE ---------------------------
if not st.session_state.logged_in:

    st.title("🔐 Login Required")

    username = st.text_input("Username")
    password = st.text_input("Password", type="password")

    if st.button("Login"):
        if (
            username == st.secrets["LOGIN"]["APP_USERNAME"]
            and password == st.secrets["LOGIN"]["APP_PASSWORD"]
        ):
            st.session_state.logged_in = True

            # 🔥 This makes login survive page refresh
            st.query_params["auth"] = "1"

            st.success("Login successful!")
            st.rerun()
        else:
            st.error("Invalid username or password")

    st.stop()


# Hide Streamlit UI elements
# ------------------------
hide_streamlit_style = """
    <style>
    /* Hide Main Menu (≡) */
    #MainMenu {visibility: hidden;}

    /* Hide footer (Made with Streamlit) */
    footer {visibility: hidden;}

    /* Hide GitHub repo link inside header if visible */
    header {visibility: hidden;}

    /* Reduce top padding to remove blank space */
    .block-container {
        padding-top: 1rem;
    }
    </style>
"""

import streamlit as st
st.markdown(hide_streamlit_style, unsafe_allow_html=True)

IST = pytz.timezone("Asia/Kolkata")
today_ist = datetime.datetime.now(IST).date()

# -------------------------
# Page config & small CSS for mobile friendliness
# -------------------------
st.set_page_config(page_title="Badminton Doubles Tracker", layout="wide")
st.markdown(
    """
    <style>
    /* make elements more compact on mobile */
    @media (max-width: 600px) {
        .streamlit-expanderHeader {
            font-size: 16px;
        }
    }
    .small { font-size:12px; color: #666; }
    </style>
    """,
    unsafe_allow_html=True,
)

# -------------------------
# Helpers
# -------------------------
def normalize(name):
    if not isinstance(name, str):
        return ""
    return name.strip().title()

def safe_int_from_text(s):
    if s is None:
        return 0
    s = str(s).strip()
    if s == "":
        return 0
    try:
        return int(float(s))
    except:
        return 0

# -------------------------
# GitHub helpers (read/write CSV)
# -------------------------
GITHUB_TOKEN = st.secrets["GITHUB_TOKEN"]
OWNER = st.secrets["REPO_OWNER"]
REPO = st.secrets["REPO_NAME"]
MATCHES_PATH = st.secrets["MATCHES_CSV"]
RATINGS_PATH = st.secrets["RATINGS_CSV"]

API_BASE = f"https://api.github.com/repos/{OWNER}/{REPO}/contents"
headers = {"Authorization": f"token {GITHUB_TOKEN}"}

def github_get_csv(path, default_columns=None):
    url = f"{API_BASE}/{path}"
    resp = requests.get(url, headers=headers)
    if resp.status_code == 200:
        info = resp.json()
        content_b64 = info.get("content", "")
        content = base64.b64decode(content_b64).decode("utf-8")
        df = pd.read_csv(StringIO(content))
        return df, info.get("sha")
    elif resp.status_code == 404:
        if default_columns:
            return pd.DataFrame(columns=default_columns), None
        else:
            return pd.DataFrame(), None
    else:
        st.error(f"GitHub API error {resp.status_code}: {resp.text}")
        st.stop()

def github_put_csv(path, df, sha=None, message="update csv"):
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

# -------------------------
# Load CSVs (safe defaults)
# -------------------------
MATCHES_COLS = ["date","playerA1","playerA2","playerB1","playerB2","scoreA","scoreB"]
RATINGS_COLS = ["player","rating","wins","losses","matches"]

matches, matches_sha = github_get_csv(MATCHES_PATH, default_columns=MATCHES_COLS)
ratings_df, ratings_sha = github_get_csv(RATINGS_PATH, default_columns=RATINGS_COLS)

# Ensure columns exist
for c in MATCHES_COLS:
    if c not in matches.columns:
        matches[c] = ""
for c in RATINGS_COLS:
    if c not in ratings_df.columns:
        ratings_df[c] = []

# Normalize strings
for c in ["playerA1","playerA2","playerB1","playerB2","date"]:
    if c in matches.columns:
        matches[c] = matches[c].fillna("").astype(str).apply(normalize)

# numeric helpers
matches["scoreA_num"] = pd.to_numeric(matches["scoreA"], errors="coerce").fillna(0).astype(int)
matches["scoreB_num"] = pd.to_numeric(matches["scoreB"], errors="coerce").fillna(0).astype(int)

if not ratings_df.empty:
    ratings_df["rating"] = pd.to_numeric(ratings_df["rating"], errors="coerce").fillna(1500.0)
    ratings_df["wins"] = pd.to_numeric(ratings_df["wins"], errors="coerce").fillna(0).astype(int)
    ratings_df["losses"] = pd.to_numeric(ratings_df["losses"], errors="coerce").fillna(0).astype(int)
    ratings_df["matches"] = pd.to_numeric(ratings_df["matches"], errors="coerce").fillna(0).astype(int)

# Build ratings dict & stats from ratings.csv if present
ratings = {}
player_stats = {}
if not ratings_df.empty:
    for _, r in ratings_df.iterrows():
        pname = normalize(r["player"])
        ratings[pname] = float(r["rating"])
        player_stats[pname] = {
            "wins": int(r["wins"]),
            "losses": int(r["losses"]),
            "matches": int(r["matches"])
        }

# Layout: collapsible sections (good for mobile)
# -------------------------
st.title("🏸 Badminton Doubles Tracker")
st.write("Logged in.")

# -------------------------
# Add Match - in an expander (collapsible)
# -------------------------
with st.expander("➕ Add New Match", expanded=True):
    left, right = st.columns([1,1])
    # keep score inputs as text so they appear empty by default on page load
    with st.form("add_match_form", clear_on_submit=True):
        A1 = normalize(left.text_input("Team A - Player 1", value=""))
        A2 = normalize(right.text_input("Team A - Player 2", value=""))
        B1 = normalize(left.text_input("Team B - Player 1", value=""))
        B2 = normalize(right.text_input("Team B - Player 2", value=""))
        sA_text = left.text_input("Score A (leave empty = 0)", value="", key="sa")
        sB_text = right.text_input("Score B (leave empty = 0)", value="", key="sb")
        match_date = right.date_input("Match date", value=today_ist)
        submitted = st.form_submit_button("Save Match")

        if submitted:
            if not any([A1, A2, B1, B2]):
                st.error("Please enter at least one player name.")
            else:
                sA = safe_int_from_text(sA_text)
                sB = safe_int_from_text(sB_text)

                new_row = {
                    "date": match_date.strftime("%Y-%m-%d"),
                    "playerA1": A1, "playerA2": A2,
                    "playerB1": B1, "playerB2": B2,
                    "scoreA": "" if sA_text.strip() == "" else str(sA),
                    "scoreB": "" if sB_text.strip() == "" else str(sB)
                }

                matches = pd.concat([matches, pd.DataFrame([new_row])], ignore_index=True)
                # update helpers
                matches["scoreA_num"] = pd.to_numeric(matches["scoreA"], errors="coerce").fillna(0).astype(int)
                matches["scoreB_num"] = pd.to_numeric(matches["scoreB"], errors="coerce").fillna(0).astype(int)

                # Save matches CSV
                matches_sha = github_put_csv(MATCHES_PATH, matches[MATCHES_COLS], matches_sha, "Add match")

                # Recompute ratings & stats by replaying matches (chronological)
                # parse/handle dates safely
                matches_proc = matches.copy()
                matches_proc["date_parsed"] = pd.to_datetime(matches_proc["date"], errors="coerce")
                if matches_proc["date_parsed"].isna().all():
                    matches_proc = matches_proc.reset_index(drop=True)
                else:
                    matches_proc = matches_proc.sort_values("date_parsed").reset_index(drop=True)

                ratings_replay = {}
                stats_replay = {}

                for _, row in matches_proc.iterrows():
                    pA1 = normalize(row.get("playerA1",""))
                    pA2 = normalize(row.get("playerA2",""))
                    pB1 = normalize(row.get("playerB1",""))
                    pB2 = normalize(row.get("playerB2",""))
                    scA = safe_int_from_text(row.get("scoreA",""))
                    scB = safe_int_from_text(row.get("scoreB",""))

                    # init
                    for p in [pA1,pA2,pB1,pB2]:
                        if not p:
                            continue
                        if p not in ratings_replay:
                            ratings_replay[p] = 1500.0
                        if p not in stats_replay:
                            stats_replay[p] = {"wins":0,"losses":0,"matches":0}

                    for p in [pA1,pA2,pB1,pB2]:
                        if p:
                            stats_replay[p]["matches"] += 1

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

                    ratings_replay = update_elo(pA1,pA2,pB1,pB2,scA,scB,ratings_replay)

                # build ratings_df and save
                ratings_rows = []
                for p, r in ratings_replay.items():
                    ratings_rows.append({
                        "player": p,
                        "rating": round(r,2),
                        "wins": stats_replay.get(p,{}).get("wins",0),
                        "losses": stats_replay.get(p,{}).get("losses",0),
                        "matches": stats_replay.get(p,{}).get("matches",0)
                    })
                ratings_df = pd.DataFrame(ratings_rows).sort_values("rating", ascending=False).reset_index(drop=True)
                ratings_sha = github_put_csv(RATINGS_PATH, ratings_df[RATINGS_COLS], ratings_sha, "Update ratings")

                st.success("Match saved and ratings updated.")

# -------------------------
# Prepare matches for display & filters
# -------------------------
# ensure safe columns
for c in MATCHES_COLS:
    if c not in matches.columns:
        matches[c] = ""

matches = matches.reset_index(drop=True)
matches["scoreA_num"] = pd.to_numeric(matches["scoreA"], errors="coerce").fillna(0).astype(int)
matches["scoreB_num"] = pd.to_numeric(matches["scoreB"], errors="coerce").fillna(0).astype(int)

# parse dates safely; create date_parsed used only for sorting (hidden from display)
def parse_date_column(df):
    df = df.copy()
    if "date" in df.columns:
        df["date_parsed"] = pd.to_datetime(df["date"], errors="coerce")
        if df["date_parsed"].isna().all():
            df["date_parsed"] = pd.to_datetime("1970-01-01") + pd.to_timedelta(df.index, unit="D")
    else:
        df["date_parsed"] = pd.to_datetime("1970-01-01") + pd.to_timedelta(df.index, unit="D")
    return df

matches = parse_date_column(matches)

# -------------------------
# Match History & Filters (collapsible)
# -------------------------
with st.expander("📜 Match History & Filters", expanded=True):
    min_date = matches["date_parsed"].min()
    max_date = matches["date_parsed"].max()
    if pd.isna(min_date):
        min_date = datetime.date.today()
    if pd.isna(max_date):
        max_date = datetime.date.today()

    c1,c2,c3 = st.columns([2,2,1])
    with c1:
        start_date = st.date_input("From", value=min_date.date() if hasattr(min_date,"date") else datetime.date.today())
    with c2:
        end_date = st.date_input("To", value=max_date.date() if hasattr(max_date,"date") else datetime.date.today())
    with c3:
        st.write("")  # spacer
        # apply filter implicitly by reading start_date/end_date below

    start_dt = pd.to_datetime(start_date)
    end_dt = pd.to_datetime(end_date) + pd.Timedelta(days=1) - pd.Timedelta(seconds=1)

    filtered = matches[(matches["date_parsed"] >= start_dt) & (matches["date_parsed"] <= end_dt)].copy()

    cols_to_show = [c for c in ["date","playerA1","playerA2","playerB1","playerB2","scoreA","scoreB"] if c in filtered.columns]

    st.subheader(f"Showing {len(filtered)} matches between {start_date} and {end_date}")

    # Sort by date_parsed if it exists, else fallback to index
    if "date_parsed" in filtered.columns:
        display_df = filtered[cols_to_show].assign(_sort=filtered["date_parsed"]).sort_values("_sort", ascending=False).drop(columns=["_sort"])
    else:
        display_df = filtered[cols_to_show].reset_index(drop=True)

    st.dataframe(display_df.reset_index(drop=True))

# -------------------------
# Player individual stats & weekly/monthly summaries
# -------------------------
with st.expander("📊 Player Statistics & Summaries", expanded=True):
    if ratings_df.empty:
        st.info("No player stats yet.")
    else:
        stats_df = ratings_df.copy()
        for col in ["rating","wins","losses","matches"]:
            stats_df[col] = pd.to_numeric(stats_df[col], errors="coerce").fillna(0)
        stats_df["Win %"] = ((stats_df["wins"] / stats_df["matches"]) * 100).replace([np.inf, -np.inf], 0).fillna(0).round(1)
        stats_view = stats_df[["player","rating","matches","wins","losses","Win %"]].sort_values("rating", ascending=False)
        st.dataframe(stats_view.reset_index(drop=True))

        # Weekly / Monthly auto-summary
        now = pd.to_datetime(today_ist)
        last7 = matches[matches["date_parsed"] >= (now - pd.Timedelta(days=7))]
        last30 = matches[matches["date_parsed"] >= (now - pd.Timedelta(days=30))]

        def summary(df):
            # compute wins per player in df
            s = {}
            for _, r in df.iterrows():
                pA1 = normalize(r["playerA1"]); pA2 = normalize(r["playerA2"])
                pB1 = normalize(r["playerB1"]); pB2 = normalize(r["playerB2"])
                scA = safe_int_from_text(r.get("scoreA",""))
                scB = safe_int_from_text(r.get("scoreB",""))
                # ensure entries
                for p in [pA1,pA2,pB1,pB2]:
                    if not p: continue
                    s.setdefault(p, {"wins":0,"losses":0,"matches":0})
                    s[p]["matches"] += 1
                if scA > scB:
                    if pA1: s[pA1]["wins"] += 1
                    if pA2: s[pA2]["wins"] += 1
                    if pB1: s[pB1]["losses"] += 1
                    if pB2: s[pB2]["losses"] += 1
                else:
                    if pB1: s[pB1]["wins"] += 1
                    if pB2: s[pB2]["wins"] += 1
                    if pA1: s[pA1]["losses"] += 1
                    if pA2: s[pA2]["losses"] += 1
            rows = []
            for p,v in s.items():
                rows.append({"player":p,"matches":v["matches"],"wins":v["wins"],"losses":v["losses"], "win_pct": round((v["wins"]/v["matches"]*100) if v["matches"] else 0,1)})
            return pd.DataFrame(rows).sort_values("win_pct", ascending=False)

        st.subheader("Last 7 days summary")
        st.dataframe(summary(last7).head(10).reset_index(drop=True))

        st.subheader("Last 30 days summary")
        st.dataframe(summary(last30).head(10).reset_index(drop=True))

# -------------------------
# Top-3 badges
# -------------------------
with st.expander("🏆 Top Players", expanded=True):
    if not ratings_df.empty:
        top = ratings_df.sort_values("rating", ascending=False).reset_index(drop=True).head(3)
    else:
        top = pd.DataFrame(columns=["player","rating","wins"])
    cols = st.columns(3)
    medals = ["🥇 #1","🥈 #2","🥉 #3"]
    for i in range(3):
        if i < len(top):
            row = top.loc[i]
            cols[i].metric(f"{medals[i]} {row['player']}", f"{float(row['rating']):.2f}", delta=f"W:{int(row.get('wins',0))}")
        else:
            cols[i].write("—")

# -------------------------
# Rating trend (replay matches)
# -------------------------
with st.expander("📈 Rating Trend (replay matches)", expanded=True):
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
                timeline_rows.append({"date": row["date_parsed"], "player": p, "rating": current.get(p,1500.0)})
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

# -------------------------
# Player profile cards (select player)
# -------------------------
with st.expander("👤 Player Profile", expanded=False):
    psel = st.selectbox("Choose player", sorted(ratings_df["player"].unique()) if not ratings_df.empty else [])
    if psel:
        psel = normalize(psel)
        rs = ratings_df[ratings_df["player"]==psel]
        if not rs.empty:
            row = rs.iloc[0]
            st.subheader(f"{psel} — {float(row['rating']):.2f}")
            st.write(f"Matches: {int(row['matches'])} • Wins: {int(row['wins'])} • Losses: {int(row['losses'])}")
            # Last 10 matches for the player
            player_matches = matches[
                (matches["playerA1"]==psel) | (matches["playerA2"]==psel) | (matches["playerB1"]==psel) | (matches["playerB2"]==psel)
            ].sort_values("date_parsed", ascending=False).head(10)
            if not player_matches.empty:
                st.subheader("Last 10 matches")
                cols_show = [c for c in ["date","playerA1","playerA2","playerB1","playerB2","scoreA","scoreB"] if c in player_matches.columns]
                st.dataframe(player_matches[cols_show].reset_index(drop=True))
            else:
                st.write("No matches yet for this player.")
        else:
            st.info("Player not found in ratings.csv.")

# -------------------------
# Predict match
# -------------------------
with st.expander("🔮 Predict Match Outcome", expanded=False):
    pa, pb = st.columns(2)
    pA1 = normalize(pa.text_input("Team A - P1", ""))
    pA2 = normalize(pa.text_input("Team A - P2", ""))
    pB1 = normalize(pb.text_input("Team B - P1", ""))
    pB2 = normalize(pb.text_input("Team B - P2", ""))
    if st.button("Predict"):
        # compute or reload latest ratings dict
        ratings_curr = {}
        if not ratings_df.empty:
            for _, r in ratings_df.iterrows():
                ratings_curr[normalize(r["player"])] = float(r["rating"])
        if all(x for x in [pA1,pA2,pB1,pB2]) and all(x in ratings_curr for x in [pA1,pA2,pB1,pB2]):
            prob = predict_win_probability(ratings_curr, pA1, pA2, pB1, pB2)
            st.success(f"Team A win probability: **{prob*100:.2f}%**")
        else:
            st.error("Ensure all four players have ratings (check ratings.csv).")

#st.info("App synced with GitHub CSV files.")
st.markdown("---")
if st.button("Logout"):
    st.session_state.logged_in = False
    # Remove query param → logs user out permanently
    st.query_params.clear()
    st.rerun()
