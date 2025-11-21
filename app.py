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

# -------------------------------------------------
# Page config + simple mobile-friendly tweaks
# -------------------------------------------------
st.set_page_config(page_title="Badminton Doubles Tracker", layout="wide")
st.markdown(
    """
    <style>
    @media (max-width: 600px) {
        .block-container { padding: 0.5rem; }
        .streamlit-expanderHeader { font-size: 16px; }
    }
    .small { font-size:12px; color: #888; }
    </style>
    """,
    unsafe_allow_html=True,
)

# -------------------------------------------------
# Time (IST)
# -------------------------------------------------
IST = pytz.timezone("Asia/Kolkata")
today_ist = datetime.datetime.now(IST).date()

# -------------------------------------------------
# Simple login using st.secrets[LOGIN]
# -------------------------------------------------
if "logged_in" not in st.session_state:
    st.session_state.logged_in = False

if not st.session_state.logged_in:
    st.title("🔐 Login")

    username = st.text_input("Username")
    password = st.text_input("Password", type="password")

    if st.button("Login"):
        try:
            ok_user = username == st.secrets["LOGIN"]["APP_USERNAME"]
            ok_pass = password == st.secrets["LOGIN"]["APP_PASSWORD"]
        except Exception:
            st.error("LOGIN section missing / misconfigured in secrets.toml")
            st.stop()

        if ok_user and ok_pass:
            st.session_state.logged_in = True
            st.success("Login successful — loading app...")
            st.rerun()
        else:
            st.error("Invalid username or password.")

    st.stop()  # Do not run the rest of the app if not logged in

# -------------------------------------------------
# GitHub CSV helpers
# -------------------------------------------------
GITHUB_TOKEN = st.secrets.get("GITHUB_TOKEN")
OWNER = st.secrets.get("REPO_OWNER")
REPO = st.secrets.get("REPO_NAME")
MATCHES_PATH = st.secrets.get("MATCHES_CSV", "matches.csv")
RATINGS_PATH = st.secrets.get("RATINGS_CSV", "ratings.csv")

API_BASE = f"https://api.github.com/repos/{OWNER}/{REPO}/contents"
headers = {"Authorization": f"token {GITHUB_TOKEN}"} if GITHUB_TOKEN else {}


def github_get_csv(path, default_columns=None):
    """Read CSV from GitHub, return (df, sha). If missing -> empty df with given columns."""
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
        return pd.DataFrame(), None
    else:
        st.error(f"GitHub API error {resp.status_code} while reading {path}: {resp.text}")
        st.stop()


def github_put_csv(path, df, sha=None, message="update csv"):
    """Create / update CSV in GitHub, return new sha."""
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


# -------------------------------------------------
# Helpers
# -------------------------------------------------
def normalize(name: str) -> str:
    if not isinstance(name, str):
        return ""
    return name.strip().title()


def safe_int_from_text(s) -> int:
    if s is None:
        return 0
    s = str(s).strip()
    if s == "":
        return 0
    try:
        return int(float(s))
    except Exception:
        return 0


def parse_date_column(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    if "date" in df.columns:
        df["date_parsed"] = pd.to_datetime(df["date"], errors="coerce")
        if df["date_parsed"].isna().all():
            df["date_parsed"] = pd.to_datetime("1970-01-01") + pd.to_timedelta(df.index, unit="D")
    else:
        df["date_parsed"] = pd.to_datetime("1970-01-01") + pd.to_timedelta(df.index, unit="D")
    return df


def recompute_ratings(matches_df: pd.DataFrame) -> pd.DataFrame:
    """Replay all matches and rebuild ratings table."""
    matches_proc = parse_date_column(matches_df)
    matches_proc = matches_proc.sort_values("date_parsed").reset_index(drop=True)

    ratings_replay = {}
    stats_replay = {}

    for _, row in matches_proc.iterrows():
        pA1 = normalize(row.get("playerA1", ""))
        pA2 = normalize(row.get("playerA2", ""))
        pB1 = normalize(row.get("playerB1", ""))
        pB2 = normalize(row.get("playerB2", ""))
        scA = safe_int_from_text(row.get("scoreA", ""))
        scB = safe_int_from_text(row.get("scoreB", ""))

        # init
        for p in [pA1, pA2, pB1, pB2]:
            if not p:
                continue
            if p not in ratings_replay:
                ratings_replay[p] = 1500.0
            if p not in stats_replay:
                stats_replay[p] = {"wins": 0, "losses": 0, "matches": 0}

        # matches
        for p in [pA1, pA2, pB1, pB2]:
            if p:
                stats_replay[p]["matches"] += 1

        # wins / losses
        if scA > scB:
            if pA1:
                stats_replay[pA1]["wins"] += 1
            if pA2:
                stats_replay[pA2]["wins"] += 1
            if pB1:
                stats_replay[pB1]["losses"] += 1
            if pB2:
                stats_replay[pB2]["losses"] += 1
        else:
            if pB1:
                stats_replay[pB1]["wins"] += 1
            if pB2:
                stats_replay[pB2]["wins"] += 1
            if pA1:
                stats_replay[pA1]["losses"] += 1
            if pA2:
                stats_replay[pA2]["losses"] += 1

        ratings_replay = update_elo(pA1, pA2, pB1, pB2, scA, scB, ratings_replay)

    rows = []
    for p, r in ratings_replay.items():
        st_w = stats_replay.get(p, {}).get("wins", 0)
        st_l = stats_replay.get(p, {}).get("losses", 0)
        st_m = stats_replay.get(p, {}).get("matches", 0)
        rows.append(
            {
                "player": p,
                "rating": round(r, 2),
                "wins": st_w,
                "losses": st_l,
                "matches": st_m,
            }
        )

    if not rows:
        return pd.DataFrame(columns=["player", "rating", "wins", "losses", "matches"])

    ratings_df_new = (
        pd.DataFrame(rows)
        .sort_values("rating", ascending=False)
        .reset_index(drop=True)
    )
    return ratings_df_new


def add_match_and_update_all(
    matches_df: pd.DataFrame,
    matches_sha: str,
    ratings_df: pd.DataFrame,
    ratings_sha: str,
    A1: str,
    A2: str,
    B1: str,
    B2: str,
    scoreA: int,
    scoreB: int,
    match_date: datetime.date,
):
    """Append one match, save matches.csv and ratings.csv, return updated dfs + shas."""
    new_row = {
        "date": match_date.strftime("%Y-%m-%d"),
        "playerA1": normalize(A1),
        "playerA2": normalize(A2),
        "playerB1": normalize(B1),
        "playerB2": normalize(B2),
        "scoreA": str(int(scoreA)),
        "scoreB": str(int(scoreB)),
    }
    matches_new = pd.concat([matches_df, pd.DataFrame([new_row])], ignore_index=True)

    # save matches
    matches_sha_new = github_put_csv(MATCHES_PATH, matches_new[MATCHES_COLS], matches_sha, "Add match")

    # recompute ratings & save
    ratings_new = recompute_ratings(matches_new)
    ratings_sha_new = github_put_csv(RATINGS_PATH, ratings_new[RATINGS_COLS], ratings_sha, "Update ratings")

    return matches_new, matches_sha_new, ratings_new, ratings_sha_new


# -------------------------------------------------
# Load CSVs
# -------------------------------------------------
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

# Normalize
for c in ["playerA1", "playerA2", "playerB1", "playerB2", "date"]:
    if c in matches.columns:
        matches[c] = matches[c].fillna("").astype(str).apply(normalize)

matches["scoreA_num"] = pd.to_numeric(matches["scoreA"], errors="coerce").fillna(0).astype(int)
matches["scoreB_num"] = pd.to_numeric(matches["scoreB"], errors="coerce").fillna(0).astype(int)

if not ratings_df.empty:
    ratings_df["rating"] = pd.to_numeric(ratings_df["rating"], errors="coerce").fillna(1500.0)
    ratings_df["wins"] = pd.to_numeric(ratings_df["wins"], errors="coerce").fillna(0).astype(int)
    ratings_df["losses"] = pd.to_numeric(ratings_df["losses"], errors="coerce").fillna(0).astype(int)
    ratings_df["matches"] = pd.to_numeric(ratings_df["matches"], errors="coerce").fillna(0).astype(int)

# -------------------------------------------------
# Session helpers (Rematch + Live Score)
# -------------------------------------------------
# Rematch defaults: last match players
if "rematch_defaults" not in st.session_state:
    st.session_state.rematch_defaults = {"A1": "", "A2": "", "B1": "", "B2": ""}


def update_rematch_defaults_from_last():
    if not matches.empty:
        last = matches.tail(1).iloc[0]
        st.session_state.rematch_defaults = {
            "A1": normalize(last.get("playerA1", "")),
            "A2": normalize(last.get("playerA2", "")),
            "B1": normalize(last.get("playerB1", "")),
            "B2": normalize(last.get("playerB2", "")),
        }
    else:
        st.session_state.rematch_defaults = {"A1": "", "A2": "", "B1": "", "B2": ""}


# Live score state
for key, value in {
    "live_active": False,
    "live_A1": "",
    "live_A2": "",
    "live_B1": "",
    "live_B2": "",
    "live_scoreA": 0,
    "live_scoreB": 0,
    "live_server": None,        # "A1" / "A2" / "B1" / "B2"
    "live_last_server_A": "A1",
    "live_last_server_B": "B1",
    "live_target": 21,
}.items():
    if key not in st.session_state:
        st.session_state[key] = value


def reset_live_match():
    st.session_state.live_active = False
    st.session_state.live_A1 = ""
    st.session_state.live_A2 = ""
    st.session_state.live_B1 = ""
    st.session_state.live_B2 = ""
    st.session_state.live_scoreA = 0
    st.session_state.live_scoreB = 0
    st.session_state.live_server = None
    st.session_state.live_last_server_A = "A1"
    st.session_state.live_last_server_B = "B1"
    st.session_state.live_target = 21


def handle_live_point(winner_side: str):
    """winner_side: 'A' or 'B'"""
    # update score
    if winner_side == "A":
        st.session_state.live_scoreA += 1
    else:
        st.session_state.live_scoreB += 1

    server_code = st.session_state.live_server
    if not server_code:
        return

    server_side = "A" if server_code in ("A1", "A2") else "B"

    # if server's team lost, serve goes to other team and alternates player
    if winner_side != server_side:
        if winner_side == "A":
            # next server in A
            last = st.session_state.live_last_server_A
            new = "A2" if last == "A1" else "A1"
            st.session_state.live_last_server_A = new
            st.session_state.live_server = new
        else:
            last = st.session_state.live_last_server_B
            new = "B2" if last == "B1" else "B1"
            st.session_state.live_last_server_B = new
            st.session_state.live_server = new
    # else: same server continues


# -------------------------------------------------
# Header
# -------------------------------------------------
st.title("🏸 Badminton Doubles Tracker")
st.write("You are logged in. Use **Live Score Mode** for rally-by-rally tracking or **Add Match** for quick entry.")
st.markdown("---")

# -------------------------------------------------
# 🎯 Live Score Mode (rally-by-rally)
# -------------------------------------------------
with st.expander("🎯 Live Score Mode (rally-by-rally)", expanded=True):
    if not st.session_state.live_active:
        with st.form("live_setup_form"):
            colA, colB = st.columns(2)
            A1 = normalize(colA.text_input("Team A - Player 1", value=""))
            A2 = normalize(colA.text_input("Team A - Player 2", value=""))
            B1 = normalize(colB.text_input("Team B - Player 1", value=""))
            B2 = normalize(colB.text_input("Team B - Player 2", value=""))

            first_server_option = st.selectbox(
                "Who serves first?",
                [
                    "Team A - Player 1",
                    "Team A - Player 2",
                    "Team B - Player 1",
                    "Team B - Player 2",
                ],
            )
            target = st.number_input("Play to (points)", min_value=1, max_value=30, value=21, step=1)

            start_live = st.form_submit_button("▶️ Start Live Match")

        if start_live:
            if not all([A1, A2, B1, B2]):
                st.error("Please fill all four player names to start live match.")
            else:
                st.session_state.live_A1 = A1
                st.session_state.live_A2 = A2
                st.session_state.live_B1 = B1
                st.session_state.live_B2 = B2
                st.session_state.live_scoreA = 0
                st.session_state.live_scoreB = 0
                st.session_state.live_target = int(target)

                if first_server_option == "Team A - Player 1":
                    st.session_state.live_server = "A1"
                    st.session_state.live_last_server_A = "A1"
                elif first_server_option == "Team A - Player 2":
                    st.session_state.live_server = "A2"
                    st.session_state.live_last_server_A = "A2"
                elif first_server_option == "Team B - Player 1":
                    st.session_state.live_server = "B1"
                    st.session_state.live_last_server_B = "B1"
                else:
                    st.session_state.live_server = "B2"
                    st.session_state.live_last_server_B = "B2"

                st.session_state.live_active = True
                st.success("Live match started!")
                st.rerun()
    else:
        # Active live match UI
        A1 = st.session_state.live_A1
        A2 = st.session_state.live_A2
        B1 = st.session_state.live_B1
        B2 = st.session_state.live_B2
        sA = st.session_state.live_scoreA
        sB = st.session_state.live_scoreB
        target = st.session_state.live_target

        colA, colB = st.columns(2)
        colA.subheader(f"Team A: {A1} & {A2}")
        colB.subheader(f"Team B: {B1} & {B2}")

        score_colA, score_colB = st.columns(2)
        score_colA.markdown(f"## 🅰️ {sA}")
        score_colB.markdown(f"## 🅱️ {sB}")

        # Current server
        server_code = st.session_state.live_server
        if server_code in ("A1", "A2"):
            server_team = "Team A"
            server_name = A1 if server_code == "A1" else A2
        elif server_code in ("B1", "B2"):
            server_team = "Team B"
            server_name = B1 if server_code == "B1" else B2
        else:
            server_team = "?"
            server_name = "?"

        st.markdown(f"**Current server:** {server_name} ({server_team})")
        st.caption(f"Target: first to {target} (you decide when to end)")

        # Point buttons
        btnA, btnB = st.columns(2)
        pointA = btnA.button("➕ Point Team A", use_container_width=True)
        pointB = btnB.button("➕ Point Team B", use_container_width=True)

        if pointA:
            handle_live_point("A")
            st.rerun()
        if pointB:
            handle_live_point("B")
            st.rerun()

        # Game status
        if max(sA, sB) >= target:
            if sA > sB:
                st.success("🎉 Team A has reached the target. You can end and save the match.")
            elif sB > sA:
                st.success("🎉 Team B has reached the target. You can end and save the match.")

        st.markdown("---")
        end_col, cancel_col = st.columns(2)
        end_clicked = end_col.button("✅ End & Save Match")
        cancel_clicked = cancel_col.button("🛑 Cancel Live Match (no save)")

        if end_clicked:
            if sA == 0 and sB == 0:
                st.error("No points played yet. Play at least one point before saving.")
            else:
                matches, matches_sha, ratings_df, ratings_sha = add_match_and_update_all(
                    matches,
                    matches_sha,
                    ratings_df,
                    ratings_sha,
                    A1,
                    A2,
                    B1,
                    B2,
                    sA,
                    sB,
                    today_ist,
                )
                update_rematch_defaults_from_last()
                reset_live_match()
                st.success("Live match saved and ratings updated.")
                st.rerun()

        if cancel_clicked:
            reset_live_match()
            st.info("Live match cancelled (not saved).")
            st.rerun()

st.markdown("---")

# -------------------------------------------------
# ➕ Add Match (manual entry) with Rematch button
# -------------------------------------------------
with st.expander("➕ Add Match (Team names + Scores)", expanded=False):
    defaults = st.session_state.rematch_defaults

    with st.form("add_match_form", clear_on_submit=True):
        left, right = st.columns(2)
        A1 = normalize(left.text_input("Team A - Player 1", value=defaults.get("A1", "")))
        A2 = normalize(left.text_input("Team A - Player 2", value=defaults.get("A2", "")))
        B1 = normalize(right.text_input("Team B - Player 1", value=defaults.get("B1", "")))
        B2 = normalize(right.text_input("Team B - Player 2", value=defaults.get("B2", "")))

        sA_text = left.text_input("Score A", value="", key="manual_scoreA")
        sB_text = right.text_input("Score B", value="", key="manual_scoreB")

        st.caption(f"Match date will be saved as **{today_ist.strftime('%Y-%m-%d')}**")

        col_save, col_rematch = st.columns(2)
        save_clicked = col_save.form_submit_button("💾 Save Match")
        rematch_clicked = col_rematch.form_submit_button("♻️ Use Last Match Players")

    # Rematch outside the form submit scope
    if rematch_clicked:
        update_rematch_defaults_from_last()
        st.rerun()

    # Save match logic (only if both scores present)
    if save_clicked:
        if not any([A1, A2, B1, B2]):
            st.error("Please enter at least one player name before saving.")
        elif sA_text.strip() == "" or sB_text.strip() == "":
            st.error("Please enter BOTH scores before saving.")
        else:
            sA = safe_int_from_text(sA_text)
            sB = safe_int_from_text(sB_text)

            matches, matches_sha, ratings_df, ratings_sha = add_match_and_update_all(
                matches,
                matches_sha,
                ratings_df,
                ratings_sha,
                A1,
                A2,
                B1,
                B2,
                sA,
                sB,
                today_ist,
            )

            # After saving, clear rematch defaults so the form starts empty.
            st.session_state.rematch_defaults = {"A1": "", "A2": "", "B1": "", "B2": ""}

            st.success("Match saved and ratings updated.")
            st.rerun()

st.markdown("---")

# -------------------------------------------------
# Prepare matches for display / filters
# -------------------------------------------------
matches = parse_date_column(matches)
matches["scoreA_num"] = pd.to_numeric(matches["scoreA"], errors="coerce").fillna(0).astype(int)
matches["scoreB_num"] = pd.to_numeric(matches["scoreB"], errors="coerce").fillna(0).astype(int)

# -------------------------------------------------
# 📜 Match history & filters
# -------------------------------------------------
with st.expander("📜 Match History & Filters", expanded=False):
    min_date = matches["date_parsed"].min()
    max_date = matches["date_parsed"].max()
    if pd.isna(min_date):
        min_date = today_ist
    if pd.isna(max_date):
        max_date = today_ist

    c1, c2, _ = st.columns([2, 2, 1])
    with c1:
        start_date = st.date_input("From", value=min_date.date() if hasattr(min_date, "date") else today_ist)
    with c2:
        end_date = st.date_input("To", value=max_date.date() if hasattr(max_date, "date") else today_ist)

    start_dt = pd.to_datetime(start_date)
    end_dt = pd.to_datetime(end_date) + pd.Timedelta(days=1) - pd.Timedelta(seconds=1)

    filtered = matches[(matches["date_parsed"] >= start_dt) & (matches["date_parsed"] <= end_dt)].copy()
    cols_to_show = [
        c
        for c in ["date", "playerA1", "playerA2", "playerB1", "playerB2", "scoreA", "scoreB"]
        if c in filtered.columns
    ]

    st.subheader(f"Showing {len(filtered)} matches between {start_date} and {end_date}")

    if "date_parsed" in filtered.columns:
        display_df = (
            filtered[cols_to_show]
            .assign(_sort=filtered["date_parsed"])
            .sort_values("_sort", ascending=False)
            .drop(columns=["_sort"])
        )
    else:
        display_df = filtered[cols_to_show].reset_index(drop=True)

    st.dataframe(display_df.reset_index(drop=True))

st.markdown("---")

# -------------------------------------------------
# 📊 Player statistics & weekly / monthly summaries
# -------------------------------------------------
with st.expander("📊 Player Statistics & Summaries", expanded=False):
    if ratings_df.empty:
        st.info("No player stats yet.")
    else:
        stats_df = ratings_df.copy()
        for col in ["rating", "wins", "losses", "matches"]:
            stats_df[col] = pd.to_numeric(stats_df[col], errors="coerce").fillna(0)
        stats_df["Win %"] = (
            (stats_df["wins"] / stats_df["matches"]) * 100
        ).replace([np.inf, -np.inf], 0).fillna(0).round(1)
        stats_view = stats_df[
            ["player", "rating", "matches", "wins", "losses", "Win %"]
        ].sort_values("rating", ascending=False)
        st.dataframe(stats_view.reset_index(drop=True))

        now_dt = pd.to_datetime(today_ist)
        last7 = matches[matches["date_parsed"] >= (now_dt - pd.Timedelta(days=7))]
        last30 = matches[matches["date_parsed"] >= (now_dt - pd.Timedelta(days=30))]

        def summary(df):
            s = {}
            for _, r in df.iterrows():
                pA1 = normalize(r["playerA1"])
                pA2 = normalize(r["playerA2"])
                pB1 = normalize(r["playerB1"])
                pB2 = normalize(r["playerB2"])
                scA = safe_int_from_text(r.get("scoreA", ""))
                scB = safe_int_from_text(r.get("scoreB", ""))
                for p in [pA1, pA2, pB1, pB2]:
                    if not p:
                        continue
                    s.setdefault(p, {"wins": 0, "losses": 0, "matches": 0})
                    s[p]["matches"] += 1
                if scA > scB:
                    if pA1:
                        s[pA1]["wins"] += 1
                    if pA2:
                        s[pA2]["wins"] += 1
                    if pB1:
                        s[pB1]["losses"] += 1
                    if pB2:
                        s[pB2]["losses"] += 1
                else:
                    if pB1:
                        s[pB1]["wins"] += 1
                    if pB2:
                        s[pB2]["wins"] += 1
                    if pA1:
                        s[pA1]["losses"] += 1
                    if pA2:
                        s[pA2]["losses"] += 1
            rows = []
            for p, v in s.items():
                m = v["matches"]
                w = v["wins"]
                rows.append(
                    {
                        "player": p,
                        "matches": m,
                        "wins": w,
                        "losses": v["losses"],
                        "win_pct": round((w / m * 100) if m else 0, 1),
                    }
                )
            if not rows:
                return pd.DataFrame(columns=["player", "matches", "wins", "losses", "win_pct"])
            return pd.DataFrame(rows).sort_values("win_pct", ascending=False)

        st.subheader("Last 7 days summary")
        st.dataframe(summary(last7).head(10).reset_index(drop=True))

        st.subheader("Last 30 days summary")
        st.dataframe(summary(last30).head(10).reset_index(drop=True))

st.markdown("---")

# -------------------------------------------------
# 🏆 Top players
# -------------------------------------------------
with st.expander("🏆 Top Players", expanded=False):
    if not ratings_df.empty:
        top = ratings_df.sort_values("rating", ascending=False).reset_index(drop=True).head(3)
    else:
        top = pd.DataFrame(columns=["player", "rating", "wins"])
    cols = st.columns(3)
    medals = ["🥇 #1", "🥈 #2", "🥉 #3"]
    for i in range(3):
        if i < len(top):
            row = top.loc[i]
            cols[i].metric(
                f"{medals[i]} {row['player']}",
                f"{float(row['rating']):.2f}",
                delta=f"W:{int(row.get('wins', 0))}",
            )
        else:
            cols[i].write("—")

st.markdown("---")

# -------------------------------------------------
# 📈 Rating trend
# -------------------------------------------------
with st.expander("📈 Rating Trend (replay matches)", expanded=False):
    players_all = sorted(ratings_df["player"].unique()) if not ratings_df.empty else []
    selected = st.multiselect("Select players to plot", players_all, default=players_all[:3])

    def build_rating_timeline(selected_players):
        timeline_rows = []
        if matches.empty:
            return pd.DataFrame(columns=["date", "player", "rating"])
        matches_sorted = matches.copy().sort_values("date_parsed").reset_index(drop=True)
        current = {}
        for _, row in matches_sorted.iterrows():
            pA1 = normalize(row["playerA1"])
            pA2 = normalize(row["playerA2"])
            pB1 = normalize(row["playerB1"])
            pB2 = normalize(row["playerB2"])
            sA = safe_int_from_text(row.get("scoreA", ""))
            sB = safe_int_from_text(row.get("scoreB", ""))
            for p in [pA1, pA2, pB1, pB2]:
                if p and p not in current:
                    current[p] = 1500.0
            current = update_elo(pA1, pA2, pB1, pB2, sA, sB, current)
            for p in selected_players:
                timeline_rows.append(
                    {"date": row["date_parsed"], "player": p, "rating": current.get(p, 1500.0)}
                )
        if not timeline_rows:
            return pd.DataFrame(columns=["date", "player", "rating"])
        return (
            pd.DataFrame(timeline_rows)
            .sort_values(["player", "date"])
            .reset_index(drop=True)
        )

    if selected:
        df_t = build_rating_timeline(selected)
        if not df_t.empty:
            chart = (
                alt.Chart(df_t)
                .mark_line(point=True)
                .encode(
                    x=alt.X("date:T", title="Date"),
                    y=alt.Y("rating:Q", title="Rating"),
                    color="player:N",
                    tooltip=["player:N", alt.Tooltip("rating:Q", format=".2f"), "date:T"],
                )
                .interactive()
            )
            st.altair_chart(chart, use_container_width=True)
        else:
            st.info("No timeline data yet.")
    else:
        st.info("Pick players to plot rating trend.")

st.markdown("---")

# -------------------------------------------------
# 👤 Player Profile
# -------------------------------------------------
with st.expander("👤 Player Profile", expanded=False):
    psel = st.selectbox(
        "Choose player",
        sorted(ratings_df["player"].unique()) if not ratings_df.empty else [],
    )
    if psel:
        psel_norm = normalize(psel)
        rs = ratings_df[ratings_df["player"] == psel_norm]
        if not rs.empty:
            row = rs.iloc[0]
            st.subheader(f"{psel_norm} — {float(row['rating']):.2f}")
            st.write(
                f"Matches: {int(row['matches'])} • Wins: {int(row['wins'])} • Losses: {int(row['losses'])}"
            )
            player_matches = matches[
                (matches["playerA1"] == psel_norm)
                | (matches["playerA2"] == psel_norm)
                | (matches["playerB1"] == psel_norm)
                | (matches["playerB2"] == psel_norm)
            ].sort_values("date_parsed", ascending=False).head(10)
            if not player_matches.empty:
                cols_show = [
                    c
                    for c in [
                        "date",
                        "playerA1",
                        "playerA2",
                        "playerB1",
                        "playerB2",
                        "scoreA",
                        "scoreB",
                    ]
                    if c in player_matches.columns
                ]
                st.dataframe(player_matches[cols_show].reset_index(drop=True))
            else:
                st.write("No matches yet for this player.")
        else:
            st.info("Player not found in ratings.csv.")

st.markdown("---")

# -------------------------------------------------
# 🔮 Predict Match Outcome
# -------------------------------------------------
with st.expander("🔮 Predict Match Outcome", expanded=False):
    pa, pb = st.columns(2)
    pA1 = normalize(pa.text_input("Team A - P1", value=""))
    pA2 = normalize(pa.text_input("Team A - P2", value=""))
    pB1 = normalize(pb.text_input("Team B - P1", value=""))
    pB2 = normalize(pb.text_input("Team B - P2", value=""))

    if st.button("Predict"):
        ratings_curr = {}
        if not ratings_df.empty:
            for _, r in ratings_df.iterrows():
                ratings_curr[normalize(r["player"])] = float(r["rating"])
        if all(x for x in [pA1, pA2, pB1, pB2]) and all(
            x in ratings_curr for x in [pA1, pA2, pB1, pB2]
        ):
            prob = predict_win_probability(ratings_curr, pA1, pA2, pB1, pB2)
            st.success(f"Team A win probability: **{prob*100:.2f}%**")
        else:
            st.error("Ensure all four players have ratings (check ratings.csv).")

st.markdown("---")

# -------------------------------------------------
# Logout
# -------------------------------------------------
if st.button("Logout"):
    st.session_state.logged_in = False
    st.success("Logged out.")
    st.rerun()
