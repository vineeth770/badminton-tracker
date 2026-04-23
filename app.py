# app.py — Badminton Doubles Tracker (Improved)
# Improvements:
#   1. @st.cache_data on GitHub reads (60s TTL) + cache-busting after writes
#   2. Player autocomplete via selectbox (+ "New player" fallback)
#   3. Rating history CSV + Altair trend chart in Player Profile
#   4. Head-to-Head stats section
#   5. Undo last match (with confirmation)
#   6. Tournament state persisted to GitHub JSON
#   7. Retry logic + better error handling on GitHub writes
#   8. Duplicate-player validation in Live Score

import time
import json
import datetime
import pytz
import base64
from io import StringIO

import requests
import numpy as np
import pandas as pd
import altair as alt
import streamlit as st

from elo import update_elo, predict_win_probability

# ─── Page config ─────────────────────────────────────────────────────────────
st.set_page_config(page_title="Badminton Doubles Tracker", layout="wide")
st.markdown("""
<style>
@media (max-width: 600px) {
    .block-container { padding: 0.5rem; }
    .streamlit-expanderHeader { font-size: 16px; }
}
.small { font-size:12px; color:#888; }
</style>
""", unsafe_allow_html=True)

# ─── Time ─────────────────────────────────────────────────────────────────────
IST = pytz.timezone("Asia/Kolkata")
today_ist = datetime.datetime.now(IST).date()

# ─── Login ────────────────────────────────────────────────────────────────────
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
            st.success("Login successful — loading app…")
            st.rerun()
        else:
            st.error("Invalid username or password.")
    st.stop()

# ─── GitHub config ────────────────────────────────────────────────────────────
GITHUB_TOKEN  = st.secrets.get("GITHUB_TOKEN")
OWNER         = st.secrets.get("REPO_OWNER")
REPO          = st.secrets.get("REPO_NAME")
MATCHES_PATH  = st.secrets.get("MATCHES_CSV",  "matches.csv")
RATINGS_PATH  = st.secrets.get("RATINGS_CSV",  "ratings.csv")
HISTORY_PATH  = "rating_history.csv"
TOURNAMENT_PATH = "tournament.json"

API_BASE  = f"https://api.github.com/repos/{OWNER}/{REPO}/contents"
GH_HEADS  = {"Authorization": f"token {GITHUB_TOKEN}"} if GITHUB_TOKEN else {}

MATCHES_COLS  = ["date", "playerA1", "playerA2", "playerB1", "playerB2", "scoreA", "scoreB"]
RATINGS_COLS  = ["player", "rating", "wins", "losses", "matches"]
HISTORY_COLS  = ["date", "player", "rating"]


# ═══════════════════════════════════════════════════════════════════════════════
# GitHub helpers
# ═══════════════════════════════════════════════════════════════════════════════

@st.cache_data(ttl=60, show_spinner=False)
def fetch_github_csv(path: str, default_columns: tuple = ()):
    """
    Cached read of a CSV from GitHub (60-second TTL).
    Call fetch_github_csv.clear() after any write to bust the cache.
    Returns (DataFrame, sha_or_None).
    """
    url  = f"{API_BASE}/{path}"
    resp = requests.get(url, headers=GH_HEADS, timeout=10)
    if resp.status_code == 200:
        info    = resp.json()
        content = base64.b64decode(info["content"]).decode("utf-8")
        df      = pd.read_csv(StringIO(content))
        return df, info.get("sha")
    elif resp.status_code == 404:
        return pd.DataFrame(columns=list(default_columns)), None
    else:
        st.error(f"GitHub API error {resp.status_code} reading {path}: {resp.text}")
        st.stop()


def github_put_csv(path: str, df: pd.DataFrame, sha=None, message="update csv"):
    """
    Write / update a CSV on GitHub with up to 3 retries on 409 conflicts.
    Clears the cache after a successful write. Returns new sha.
    """
    url     = f"{API_BASE}/{path}"
    encoded = base64.b64encode(df.to_csv(index=False).encode()).decode()
    payload = {"message": message, "content": encoded}
    if sha:
        payload["sha"] = sha

    for attempt in range(3):
        resp = requests.put(url, headers=GH_HEADS, json=payload, timeout=15)
        if resp.status_code in (200, 201):
            fetch_github_csv.clear()          # bust cache so next read is fresh
            return resp.json().get("content", {}).get("sha")
        elif resp.status_code == 409 and attempt < 2:
            # Conflict — fetch latest sha and retry
            time.sleep(1)
            fresh_resp = requests.get(url, headers=GH_HEADS, timeout=10)
            if fresh_resp.status_code == 200:
                payload["sha"] = fresh_resp.json().get("sha")
        else:
            st.error(f"GitHub write error {resp.status_code} on {path}: {resp.text}")
            st.stop()


def fetch_github_json(path: str):
    """Read JSON blob from GitHub. Returns (data_or_None, sha_or_None)."""
    url  = f"{API_BASE}/{path}"
    resp = requests.get(url, headers=GH_HEADS, timeout=10)
    if resp.status_code == 200:
        info    = resp.json()
        content = base64.b64decode(info["content"]).decode("utf-8")
        return json.loads(content), info.get("sha")
    return None, None


def github_put_json(path: str, data, sha=None, message="update json"):
    """Write JSON blob to GitHub. Returns new sha."""
    url     = f"{API_BASE}/{path}"
    encoded = base64.b64encode(json.dumps(data, indent=2).encode()).decode()
    payload = {"message": message, "content": encoded}
    if sha:
        payload["sha"] = sha
    resp = requests.put(url, headers=GH_HEADS, json=payload, timeout=15)
    if resp.status_code in (200, 201):
        return resp.json().get("content", {}).get("sha")
    else:
        st.error(f"GitHub write error {resp.status_code} on {path}: {resp.text}")
        st.stop()


# ═══════════════════════════════════════════════════════════════════════════════
# Pure data helpers
# ═══════════════════════════════════════════════════════════════════════════════

def normalize(name) -> str:
    if not isinstance(name, str):
        return ""
    return name.strip().title()


def safe_int(s) -> int:
    try:
        return int(float(str(s).strip()))
    except Exception:
        return 0


def parse_dates(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df["date_parsed"] = pd.to_datetime(df.get("date", pd.Series(dtype=str)), errors="coerce")
    df["date_parsed"] = df["date_parsed"].fillna(pd.Timestamp("1970-01-01"))
    return df


def recompute_ratings(matches_df: pd.DataFrame) -> pd.DataFrame:
    """Replay every match chronologically and rebuild ratings + stats."""
    proc = parse_dates(matches_df).sort_values("date_parsed").reset_index(drop=True)
    ratings: dict[str, float] = {}
    stats:   dict[str, dict]  = {}

    for _, row in proc.iterrows():
        pA1 = normalize(row.get("playerA1", ""))
        pA2 = normalize(row.get("playerA2", ""))
        pB1 = normalize(row.get("playerB1", ""))
        pB2 = normalize(row.get("playerB2", ""))
        scA = safe_int(row.get("scoreA", 0))
        scB = safe_int(row.get("scoreB", 0))

        for p in [pA1, pA2, pB1, pB2]:
            if not p:
                continue
            ratings.setdefault(p, 1500.0)
            stats.setdefault(p, {"wins": 0, "losses": 0, "matches": 0})
            stats[p]["matches"] += 1

        winners = [pA1, pA2] if scA > scB else [pB1, pB2]
        losers  = [pB1, pB2] if scA > scB else [pA1, pA2]
        for p in winners:
            if p: stats[p]["wins"]   += 1
        for p in losers:
            if p: stats[p]["losses"] += 1

        ratings = update_elo(pA1, pA2, pB1, pB2, scA, scB, ratings)

    rows = [
        {"player": p, "rating": round(r, 2),
         "wins": stats[p]["wins"], "losses": stats[p]["losses"],
         "matches": stats[p]["matches"]}
        for p, r in ratings.items()
    ]
    if not rows:
        return pd.DataFrame(columns=RATINGS_COLS)
    return (pd.DataFrame(rows)
              .sort_values("rating", ascending=False)
              .reset_index(drop=True))


def make_history_snapshot(ratings_df: pd.DataFrame, date: datetime.date) -> pd.DataFrame:
    """One row per player with their current rating — appended after each match."""
    rows = [
        {"date": date.strftime("%Y-%m-%d"),
         "player": r["player"],
         "rating": round(float(r["rating"]), 2)}
        for _, r in ratings_df.iterrows()
    ]
    return pd.DataFrame(rows, columns=HISTORY_COLS) if rows else pd.DataFrame(columns=HISTORY_COLS)


def period_summary(matches_df: pd.DataFrame, days: int) -> pd.DataFrame:
    """Win/loss summary for the last N days."""
    cutoff = pd.Timestamp(today_ist) - pd.Timedelta(days=days)
    sub    = matches_df[matches_df["date_parsed"] >= cutoff]
    s: dict[str, dict] = {}
    for _, r in sub.iterrows():
        pA1 = normalize(r["playerA1"]); pA2 = normalize(r["playerA2"])
        pB1 = normalize(r["playerB1"]); pB2 = normalize(r["playerB2"])
        scA = safe_int(r.get("scoreA", 0)); scB = safe_int(r.get("scoreB", 0))
        for p in [pA1, pA2, pB1, pB2]:
            if p:
                s.setdefault(p, {"wins": 0, "losses": 0, "matches": 0})
                s[p]["matches"] += 1
        winners = [pA1, pA2] if scA > scB else [pB1, pB2]
        losers  = [pB1, pB2] if scA > scB else [pA1, pA2]
        for p in winners:
            if p: s[p]["wins"]   += 1
        for p in losers:
            if p: s[p]["losses"] += 1
    rows = [
        {"player": p, "matches": v["matches"], "wins": v["wins"],
         "losses": v["losses"],
         "win_%": round(v["wins"] / v["matches"] * 100 if v["matches"] else 0, 1)}
        for p, v in s.items()
    ]
    if not rows:
        return pd.DataFrame(columns=["player", "matches", "wins", "losses", "win_%"])
    return pd.DataFrame(rows).sort_values("win_%", ascending=False).reset_index(drop=True)


# ═══════════════════════════════════════════════════════════════════════════════
# Load data
# ═══════════════════════════════════════════════════════════════════════════════

matches,    matches_sha  = fetch_github_csv(MATCHES_PATH,  tuple(MATCHES_COLS))
ratings_df, ratings_sha  = fetch_github_csv(RATINGS_PATH,  tuple(RATINGS_COLS))
history_df, history_sha  = fetch_github_csv(HISTORY_PATH,  tuple(HISTORY_COLS))

# Ensure all expected columns exist
for c in MATCHES_COLS:
    if c not in matches.columns:
        matches[c] = ""
for c in RATINGS_COLS:
    if c not in ratings_df.columns:
        ratings_df[c] = pd.Series(dtype="object")
for c in HISTORY_COLS:
    if c not in history_df.columns:
        history_df[c] = pd.Series(dtype="object")

# Normalize player name columns
for c in ["playerA1", "playerA2", "playerB1", "playerB2"]:
    matches[c] = matches[c].fillna("").astype(str).apply(normalize)
matches["date"] = matches["date"].fillna("").astype(str).str.strip()

if not ratings_df.empty:
    ratings_df["rating"]  = pd.to_numeric(ratings_df["rating"],  errors="coerce").fillna(1500.0)
    for c in ["wins", "losses", "matches"]:
        ratings_df[c] = pd.to_numeric(ratings_df[c], errors="coerce").fillna(0).astype(int)

matches = parse_dates(matches)
matches["scoreA_num"] = pd.to_numeric(matches.get("scoreA", 0), errors="coerce").fillna(0).astype(int)
matches["scoreB_num"] = pd.to_numeric(matches.get("scoreB", 0), errors="coerce").fillna(0).astype(int)

# Known players list — used for autocomplete dropdowns everywhere
known_players: list[str] = (
    sorted(ratings_df["player"].dropna().unique().tolist())
    if not ratings_df.empty else []
)


# ═══════════════════════════════════════════════════════════════════════════════
# Autocomplete widget
# ═══════════════════════════════════════════════════════════════════════════════

def player_selectbox(label: str, key: str, default: str = "", container=None):
    """
    Selectbox showing all known players + a "✏️ New player…" option.
    Falls back to a plain text_input when no players exist yet.
    Works both directly on `st` and inside column containers.
    """
    target = container if container is not None else st
    if not known_players:
        return normalize(target.text_input(label, value=default, key=f"{key}_txt"))
    options = [""] + known_players + ["✏️ New player…"]
    idx     = options.index(default) if default in options else 0
    choice  = target.selectbox(label, options, index=idx, key=f"{key}_sel")
    if choice == "✏️ New player…":
        return normalize(target.text_input("Enter new player name", value="", key=f"{key}_new"))
    return normalize(choice)


# ═══════════════════════════════════════════════════════════════════════════════
# Core save / delete functions
# ═══════════════════════════════════════════════════════════════════════════════

def save_match(A1: str, A2: str, B1: str, B2: str,
               scoreA: int, scoreB: int, match_date: datetime.date):
    """Append a match, recompute ratings, update history snapshot — all in GitHub."""
    global matches, matches_sha, ratings_df, ratings_sha, history_df, history_sha, known_players

    new_row = {
        "date":     match_date.strftime("%Y-%m-%d"),
        "playerA1": normalize(A1), "playerA2": normalize(A2),
        "playerB1": normalize(B1), "playerB2": normalize(B2),
        "scoreA":   str(int(scoreA)), "scoreB": str(int(scoreB)),
    }
    matches      = pd.concat([matches, pd.DataFrame([new_row])], ignore_index=True)
    matches_sha  = github_put_csv(MATCHES_PATH, matches[MATCHES_COLS], matches_sha, "Add match")

    ratings_df   = recompute_ratings(matches)
    ratings_sha  = github_put_csv(RATINGS_PATH, ratings_df[RATINGS_COLS], ratings_sha, "Update ratings")

    snap         = make_history_snapshot(ratings_df, match_date)
    history_df   = pd.concat([history_df, snap], ignore_index=True)
    history_sha  = github_put_csv(HISTORY_PATH, history_df[HISTORY_COLS], history_sha, "Update history")

    known_players = sorted(ratings_df["player"].dropna().unique().tolist())


def delete_last_match():
    """Remove the most recent match row and recompute ratings."""
    global matches, matches_sha, ratings_df, ratings_sha, known_players
    if matches.empty:
        return
    matches      = matches.iloc[:-1].reset_index(drop=True)
    matches_sha  = github_put_csv(MATCHES_PATH, matches[MATCHES_COLS], matches_sha, "Undo last match")
    ratings_df   = recompute_ratings(matches)
    ratings_sha  = github_put_csv(RATINGS_PATH, ratings_df[RATINGS_COLS], ratings_sha, "Recompute ratings")
    known_players = sorted(ratings_df["player"].dropna().unique().tolist())


# ═══════════════════════════════════════════════════════════════════════════════
# Session-state initialisation
# ═══════════════════════════════════════════════════════════════════════════════

_ss_defaults = {
    "rematch_defaults":     {"A1": "", "A2": "", "B1": "", "B2": ""},
    "live_active":          False,
    "live_A1": "", "live_A2": "", "live_B1": "", "live_B2": "",
    "live_scoreA":          0,
    "live_scoreB":          0,
    "live_server":          None,
    "live_last_server_A":   "A1",
    "live_last_server_B":   "B1",
    "live_target":          21,
    "confirm_delete":       False,
    "tournament_loaded":    False,
    "tournament_active":    False,
    "tournament_players":   [],
    "tournament_matches":   [],
    "tournament_results":   {},
    "tournament_sha":       None,
}
for _k, _v in _ss_defaults.items():
    if _k not in st.session_state:
        st.session_state[_k] = _v


def reset_live():
    st.session_state.live_active        = False
    st.session_state.live_A1            = ""
    st.session_state.live_A2            = ""
    st.session_state.live_B1            = ""
    st.session_state.live_B2            = ""
    st.session_state.live_scoreA        = 0
    st.session_state.live_scoreB        = 0
    st.session_state.live_server        = None
    st.session_state.live_last_server_A = "A1"
    st.session_state.live_last_server_B = "B1"
    st.session_state.live_target        = 21


def handle_live_point(winner_side: str):
    if winner_side == "A":
        st.session_state.live_scoreA += 1
    else:
        st.session_state.live_scoreB += 1

    code = st.session_state.live_server
    if not code:
        return
    server_side = "A" if code in ("A1", "A2") else "B"
    if winner_side != server_side:
        if winner_side == "A":
            last = st.session_state.live_last_server_A
            new  = "A2" if last == "A1" else "A1"
            st.session_state.live_last_server_A = new
            st.session_state.live_server        = new
        else:
            last = st.session_state.live_last_server_B
            new  = "B2" if last == "B1" else "B1"
            st.session_state.live_last_server_B = new
            st.session_state.live_server        = new


# ═══════════════════════════════════════════════════════════════════════════════
# Header
# ═══════════════════════════════════════════════════════════════════════════════

st.title("🏸 Badminton Doubles Tracker")
st.caption(f"Today: {today_ist.strftime('%d %b %Y')} (IST)")
st.markdown("---")


# ═══════════════════════════════════════════════════════════════════════════════
# 🎯 LIVE SCORE MODE
# ═══════════════════════════════════════════════════════════════════════════════
with st.expander("🎯 Live Score Mode", expanded=True):

    if not st.session_state.live_active:
        colA, colB = st.columns(2)
        live_A1 = player_selectbox("Team A — Player 1", "live_A1", container=colA)
        live_A2 = player_selectbox("Team A — Player 2", "live_A2", container=colA)
        live_B1 = player_selectbox("Team B — Player 1", "live_B1", container=colB)
        live_B2 = player_selectbox("Team B — Player 2", "live_B2", container=colB)

        srv_opt = st.selectbox("Who serves first?", [
            "Team A — Player 1", "Team A — Player 2",
            "Team B — Player 1", "Team B — Player 2",
        ])
        target  = st.number_input("Play to (points)", min_value=1, max_value=30, value=21, step=1)

        if st.button("▶️ Start Live Match", type="primary"):
            names = [live_A1, live_A2, live_B1, live_B2]
            if not all(names):
                st.error("Fill in all four player names.")
            elif len(set(names)) < 4:
                st.error("All four players must be different.")
            else:
                st.session_state.live_A1 = live_A1
                st.session_state.live_A2 = live_A2
                st.session_state.live_B1 = live_B1
                st.session_state.live_B2 = live_B2
                st.session_state.live_scoreA = 0
                st.session_state.live_scoreB = 0
                st.session_state.live_target = int(target)

                srv_map = {
                    "Team A — Player 1": ("A1", "A1", "B1"),
                    "Team A — Player 2": ("A2", "A2", "B1"),
                    "Team B — Player 1": ("B1", "A1", "B1"),
                    "Team B — Player 2": ("B2", "A1", "B2"),
                }
                srv, la, lb = srv_map[srv_opt]
                st.session_state.live_server        = srv
                st.session_state.live_last_server_A = la
                st.session_state.live_last_server_B = lb
                st.session_state.live_active        = True
                st.rerun()

    else:
        A1, A2 = st.session_state.live_A1, st.session_state.live_A2
        B1, B2 = st.session_state.live_B1, st.session_state.live_B2
        sA, sB = st.session_state.live_scoreA, st.session_state.live_scoreB
        target = st.session_state.live_target

        # Scoreboard
        colA, colB = st.columns(2)
        colA.markdown(f"### 🔵 {A1} & {A2}")
        colB.markdown(f"### 🔴 {B1} & {B2}")
        sc1, sc2 = st.columns(2)
        sc1.markdown(f"<h1 style='text-align:center;color:#1f77b4'>{sA}</h1>", unsafe_allow_html=True)
        sc2.markdown(f"<h1 style='text-align:center;color:#d62728'>{sB}</h1>", unsafe_allow_html=True)

        # Current server
        code        = st.session_state.live_server
        name_map    = {"A1": A1, "A2": A2, "B1": B1, "B2": B2}
        srv_name    = name_map.get(code, "?")
        srv_team    = "Team A 🔵" if code in ("A1", "A2") else "Team B 🔴"
        st.info(f"🏸 Serving: **{srv_name}** ({srv_team})  ·  First to **{target}**")

        # Pre-match win probability (using current ELO)
        if not ratings_df.empty:
            rc = dict(zip(ratings_df["player"], ratings_df["rating"].astype(float)))
            if all(p in rc for p in [A1, A2, B1, B2]):
                prob = predict_win_probability(rc, A1, A2, B1, B2)
                st.caption(
                    f"ELO prediction — 🔵 Team A: {prob*100:.1f}%  ·  🔴 Team B: {(1-prob)*100:.1f}%"
                )

        # Point buttons
        btnA, btnB = st.columns(2)
        if btnA.button("➕ Point — Team A", use_container_width=True, type="primary"):
            handle_live_point("A"); st.rerun()
        if btnB.button("➕ Point — Team B", use_container_width=True):
            handle_live_point("B"); st.rerun()

        if max(sA, sB) >= target:
            leader = "🔵 Team A" if sA > sB else "🔴 Team B"
            st.success(f"🎉 {leader} has reached {target}! Save to confirm.")

        st.markdown("---")
        end_col, cancel_col = st.columns(2)
        if end_col.button("✅ End & Save", type="primary"):
            if sA == 0 and sB == 0:
                st.error("No points played yet.")
            else:
                with st.spinner("Saving match and updating ratings…"):
                    save_match(A1, A2, B1, B2, sA, sB, today_ist)
                reset_live()
                st.success("Match saved! Ratings updated.")
                st.rerun()
        if cancel_col.button("🛑 Cancel"):
            reset_live()
            st.info("Live match cancelled.")
            st.rerun()

st.markdown("---")


# ═══════════════════════════════════════════════════════════════════════════════
# ➕ ADD MATCH (manual entry)
# ═══════════════════════════════════════════════════════════════════════════════
with st.expander("➕ Add Match", expanded=False):
    rd   = st.session_state.rematch_defaults
    colL, colR = st.columns(2)
    am_A1 = player_selectbox("Team A — Player 1", "am_A1", rd.get("A1", ""), colL)
    am_A2 = player_selectbox("Team A — Player 2", "am_A2", rd.get("A2", ""), colL)
    am_B1 = player_selectbox("Team B — Player 1", "am_B1", rd.get("B1", ""), colR)
    am_B2 = player_selectbox("Team B — Player 2", "am_B2", rd.get("B2", ""), colR)

    s1, s2 = st.columns(2)
    am_sA  = s1.number_input("Score A", min_value=0, max_value=100, value=0, key="am_sA")
    am_sB  = s2.number_input("Score B", min_value=0, max_value=100, value=0, key="am_sB")
    st.caption(f"Date saved as **{today_ist.strftime('%Y-%m-%d')}**")

    btn1, btn2, btn3 = st.columns(3)
    save_btn    = btn1.button("💾 Save Match",       type="primary", key="am_save")
    rematch_btn = btn2.button("♻️ Use Last Players", key="am_rematch")
    undo_btn    = btn3.button("↩️ Undo Last Match",  key="am_undo",
                               disabled=matches.empty)

    # Undo confirmation
    if undo_btn:
        st.session_state.confirm_delete = True

    if st.session_state.confirm_delete:
        if not matches.empty:
            last = matches.tail(1).iloc[0]
            st.warning(
                f"⚠️ Delete: **{last['playerA1']} & {last['playerA2']}** vs "
                f"**{last['playerB1']} & {last['playerB2']}**  "
                f"({last['scoreA']}–{last['scoreB']}) on {last['date']}? "
                f"This will recompute all ratings."
            )
        c1, c2 = st.columns(2)
        if c1.button("Yes, delete it", type="primary"):
            with st.spinner("Deleting match and recomputing…"):
                delete_last_match()
            st.session_state.confirm_delete = False
            st.success("Last match deleted. Ratings recomputed.")
            st.rerun()
        if c2.button("Cancel deletion"):
            st.session_state.confirm_delete = False
            st.rerun()

    if rematch_btn:
        if not matches.empty:
            last = matches.tail(1).iloc[0]
            st.session_state.rematch_defaults = {
                "A1": normalize(last.get("playerA1", "")),
                "A2": normalize(last.get("playerA2", "")),
                "B1": normalize(last.get("playerB1", "")),
                "B2": normalize(last.get("playerB2", "")),
            }
        st.rerun()

    if save_btn:
        names = [am_A1, am_A2, am_B1, am_B2]
        if not all(names):
            st.error("Fill in all four player names.")
        elif len(set(names)) < 4:
            st.error("All four players must be different.")
        elif am_sA == 0 and am_sB == 0:
            st.warning("Both scores are 0 — is that correct? Edit if not.")
        else:
            with st.spinner("Saving match…"):
                save_match(am_A1, am_A2, am_B1, am_B2, am_sA, am_sB, today_ist)
            st.session_state.rematch_defaults = {"A1": "", "A2": "", "B1": "", "B2": ""}
            st.success("Match saved and ratings updated!")
            st.rerun()

st.markdown("---")


# ═══════════════════════════════════════════════════════════════════════════════
# 📜 MATCH HISTORY
# ═══════════════════════════════════════════════════════════════════════════════
with st.expander("📜 Match History & Filters", expanded=False):
    if matches.empty:
        st.info("No matches recorded yet.")
    else:
        min_d = matches["date_parsed"].min()
        max_d = matches["date_parsed"].max()
        if pd.isna(min_d): min_d = pd.Timestamp(today_ist)
        if pd.isna(max_d): max_d = pd.Timestamp(today_ist)

        c1, c2 = st.columns(2)
        start_d = c1.date_input("From", value=min_d.date())
        end_d   = c2.date_input("To",   value=max_d.date())

        filt = matches[
            (matches["date_parsed"] >= pd.Timestamp(start_d)) &
            (matches["date_parsed"] <= pd.Timestamp(end_d) + pd.Timedelta(days=1, seconds=-1))
        ].copy()

        disp_cols = [c for c in MATCHES_COLS if c in filt.columns]
        st.subheader(f"{len(filt)} matches — {start_d} to {end_d}")
        st.dataframe(
            filt[disp_cols].sort_values("date", ascending=False).reset_index(drop=True)
        )

st.markdown("---")


# ═══════════════════════════════════════════════════════════════════════════════
# 📊 PLAYER STATISTICS & SUMMARIES
# ═══════════════════════════════════════════════════════════════════════════════
with st.expander("📊 Player Statistics & Summaries", expanded=False):
    if ratings_df.empty:
        st.info("No player stats yet.")
    else:
        stats_df = ratings_df.copy()
        stats_df["Win %"] = (
            stats_df["wins"] / stats_df["matches"].replace(0, 1) * 100
        ).round(1)
        st.dataframe(
            stats_df[["player", "rating", "matches", "wins", "losses", "Win %"]]
            .sort_values("rating", ascending=False).reset_index(drop=True)
        )

        col7, col30 = st.columns(2)
        with col7:
            st.subheader("Last 7 days")
            st.dataframe(period_summary(matches, 7).head(8))
        with col30:
            st.subheader("Last 30 days")
            st.dataframe(period_summary(matches, 30).head(8))

st.markdown("---")


# ═══════════════════════════════════════════════════════════════════════════════
# 🏆 TOP PLAYERS
# ═══════════════════════════════════════════════════════════════════════════════
with st.expander("🏆 Top Players", expanded=False):
    if ratings_df.empty:
        st.info("No data yet.")
    else:
        top     = ratings_df.sort_values("rating", ascending=False).head(3).reset_index(drop=True)
        medals  = ["🥇", "🥈", "🥉"]
        cols    = st.columns(3)
        for i in range(3):
            if i < len(top):
                r = top.loc[i]
                cols[i].metric(
                    f"{medals[i]} {r['player']}",
                    f"{float(r['rating']):.1f}",
                    delta=f"W {int(r['wins'])}  L {int(r['losses'])}",
                )
            else:
                cols[i].write("—")

st.markdown("---")


# ═══════════════════════════════════════════════════════════════════════════════
# 👤 PLAYER PROFILE + RATING TREND CHART
# ═══════════════════════════════════════════════════════════════════════════════
with st.expander("👤 Player Profile", expanded=False):
    if not known_players:
        st.info("No players yet.")
    else:
        psel = st.selectbox("Choose player", known_players, key="profile_sel")
        if psel:
            psel_n = normalize(psel)
            row    = ratings_df[ratings_df["player"] == psel_n]
            if not row.empty:
                r = row.iloc[0]
                m1, m2, m3, m4 = st.columns(4)
                m1.metric("ELO Rating",  f"{float(r['rating']):.1f}")
                m2.metric("Matches",     int(r["matches"]))
                m3.metric("Wins",        int(r["wins"]))
                m4.metric("Losses",      int(r["losses"]))

                # ── Rating history line chart ──────────────────────────────
                if not history_df.empty:
                    ph = history_df[history_df["player"] == psel_n].copy()
                    ph["date"]   = pd.to_datetime(ph["date"],   errors="coerce")
                    ph["rating"] = pd.to_numeric(ph["rating"],  errors="coerce")
                    ph           = ph.dropna().sort_values("date")

                    if len(ph) >= 2:
                        chart = (
                            alt.Chart(ph)
                            .mark_line(point=True, color="#1f77b4", strokeWidth=2)
                            .encode(
                                x=alt.X("date:T",   title="Date"),
                                y=alt.Y("rating:Q", title="ELO Rating",
                                        scale=alt.Scale(zero=False)),
                                tooltip=[
                                    alt.Tooltip("date:T",   title="Date"),
                                    alt.Tooltip("rating:Q", title="Rating", format=".1f"),
                                ],
                            )
                            .properties(title=f"{psel_n} — ELO over time", height=260)
                        )
                        st.altair_chart(chart, use_container_width=True)
                    else:
                        st.caption("Play more matches to see the rating trend.")

                # ── Recent matches ─────────────────────────────────────────
                pm = matches[
                    (matches["playerA1"] == psel_n) | (matches["playerA2"] == psel_n) |
                    (matches["playerB1"] == psel_n) | (matches["playerB2"] == psel_n)
                ].sort_values("date_parsed", ascending=False).head(10)

                if not pm.empty:
                    st.subheader("Recent matches")
                    disp = [c for c in MATCHES_COLS if c in pm.columns]
                    st.dataframe(pm[disp].reset_index(drop=True))

st.markdown("---")


# ═══════════════════════════════════════════════════════════════════════════════
# ⚔️ HEAD-TO-HEAD
# ═══════════════════════════════════════════════════════════════════════════════
with st.expander("⚔️ Head-to-Head", expanded=False):
    if len(known_players) < 2:
        st.info("Need at least 2 rated players.")
    else:
        hh1, hh2 = st.columns(2)
        p1 = hh1.selectbox("Player 1", known_players, key="h2h_p1")
        p2 = hh2.selectbox("Player 2",
                            [x for x in known_players if x != p1],
                            key="h2h_p2")

        if p1 and p2:
            p1n, p2n = normalize(p1), normalize(p2)
            all_cols = {"playerA1", "playerA2", "playerB1", "playerB2"}

            # Matches where BOTH players appear
            def has_both(r):
                ps = {r.playerA1, r.playerA2, r.playerB1, r.playerB2}
                return p1n in ps and p2n in ps

            h2h = matches[matches.apply(has_both, axis=1)].copy()
            st.markdown(f"### {p1n}  vs  {p2n}")

            if h2h.empty:
                st.info("These two players have never shared a match.")
            else:
                same_team  = 0; same_wins  = 0; same_losses = 0
                opp_team   = 0; p1_wins    = 0; p2_wins     = 0

                for _, r in h2h.iterrows():
                    tA  = {r.playerA1, r.playerA2}
                    tB  = {r.playerB1, r.playerB2}
                    scA = safe_int(r.scoreA); scB = safe_int(r.scoreB)

                    if (p1n in tA and p2n in tA) or (p1n in tB and p2n in tB):
                        same_team += 1
                        team_won = (p1n in tA and scA > scB) or (p1n in tB and scB > scA)
                        if team_won: same_wins   += 1
                        else:        same_losses += 1
                    else:
                        opp_team += 1
                        p1_in_A  = p1n in tA
                        p1_won   = (p1_in_A and scA > scB) or (not p1_in_A and scB > scA)
                        if p1_won: p1_wins += 1
                        else:      p2_wins += 1

                m1, m2, m3, m4, m5 = st.columns(5)
                m1.metric("Shared matches",          len(h2h))
                m2.metric("Same team",               same_team,
                           delta=f"W{same_wins} L{same_losses}")
                m3.metric("Head-to-head matches",    opp_team)
                m4.metric(f"{p1n} wins",             p1_wins)
                m5.metric(f"{p2n} wins",             p2_wins)

                disp = [c for c in MATCHES_COLS if c in h2h.columns]
                st.dataframe(h2h[disp].sort_values("date", ascending=False).reset_index(drop=True))

st.markdown("---")


# ═══════════════════════════════════════════════════════════════════════════════
# 🏆 TOURNAMENT MODE (persisted to GitHub JSON)
# ═══════════════════════════════════════════════════════════════════════════════
with st.expander("🏆 Tournament Mode (Round-robin)", expanded=False):

    # Load persisted tournament once per session
    if not st.session_state.tournament_loaded:
        t_data, t_sha = fetch_github_json(TOURNAMENT_PATH)
        if t_data and t_data.get("active"):
            st.session_state.tournament_active  = True
            st.session_state.tournament_players = t_data["players"]
            st.session_state.tournament_matches = [tuple(m) for m in t_data["matches"]]
            st.session_state.tournament_results = t_data["results"]
        st.session_state.tournament_sha    = t_sha
        st.session_state.tournament_loaded = True

    def persist_tournament():
        data = {
            "active":  st.session_state.tournament_active,
            "players": st.session_state.tournament_players,
            "matches": [list(m) for m in st.session_state.tournament_matches],
            "results": st.session_state.tournament_results,
        }
        new_sha = github_put_json(
            TOURNAMENT_PATH, data,
            st.session_state.tournament_sha,
            "Update tournament",
        )
        st.session_state.tournament_sha = new_sha

    if not st.session_state.tournament_active:
        st.markdown("### Start a new tournament")
        players_text = st.text_area("Player names (one per line)",
                                     placeholder="Alice\nBob\nCarol\nDave")
        if st.button("Create Tournament", type="primary"):
            players = [normalize(p) for p in players_text.split("\n") if p.strip()]
            players = list(dict.fromkeys(players))   # deduplicate, preserve order
            if len(players) < 2:
                st.error("Add at least 2 players.")
            else:
                fixtures = [
                    (players[i], players[j])
                    for i in range(len(players))
                    for j in range(i + 1, len(players))
                ]
                st.session_state.tournament_active  = True
                st.session_state.tournament_players = players
                st.session_state.tournament_matches = fixtures
                st.session_state.tournament_results = {
                    f"{a} vs {b}": {"A": 0, "B": 0, "done": False}
                    for a, b in fixtures
                }
                with st.spinner("Saving tournament…"):
                    persist_tournament()
                st.success("Tournament created and saved!")
                st.rerun()
    else:
        st.success(f"Active tournament · {len(st.session_state.tournament_players)} players")
        st.write("Players:", ", ".join(st.session_state.tournament_players))

        st.markdown("### Fixtures")
        changed = False
        for a, b in st.session_state.tournament_matches:
            key = f"{a} vs {b}"
            res = st.session_state.tournament_results[key]
            cA, cB, cC = st.columns([3, 3, 2])
            cA.write(f"**{a}** vs **{b}**")
            if not res["done"]:
                sA_ = cB.number_input(f"{a} score", 0, 30, key=f"t_{key}_A")
                sB_ = cB.number_input(f"{b} score", 0, 30, key=f"t_{key}_B")
                if cC.button("Save", key=f"t_{key}_sv"):
                    res["A"] = int(sA_); res["B"] = int(sB_); res["done"] = True
                    changed = True
            else:
                cB.write(f"**{res['A']} – {res['B']}**")
                if cC.button("Edit", key=f"t_{key}_ed"):
                    res["done"] = False
                    changed = True

        if changed:
            with st.spinner("Saving…"):
                persist_tournament()
            st.rerun()

        # Standings
        st.markdown("### Standings")
        table = {p: {"Played": 0, "Won": 0, "Lost": 0, "Points": 0}
                 for p in st.session_state.tournament_players}
        for a, b in st.session_state.tournament_matches:
            res = st.session_state.tournament_results[f"{a} vs {b}"]
            if not res["done"]:
                continue
            table[a]["Played"] += 1; table[b]["Played"] += 1
            if res["A"] > res["B"]:
                table[a]["Won"] += 1; table[b]["Lost"] += 1; table[a]["Points"] += 2
            else:
                table[b]["Won"] += 1; table[a]["Lost"] += 1; table[b]["Points"] += 2

        st.dataframe(
            pd.DataFrame([{"Player": p, **v} for p, v in table.items()])
            .sort_values(["Points", "Won"], ascending=False)
            .reset_index(drop=True)
        )

        if st.button("🏁 End Tournament"):
            st.session_state.tournament_active = False
            with st.spinner("Saving…"):
                persist_tournament()
            st.success("Tournament ended.")
            st.rerun()

st.markdown("---")


# ═══════════════════════════════════════════════════════════════════════════════
# 🔮 PREDICT MATCH OUTCOME
# ═══════════════════════════════════════════════════════════════════════════════
with st.expander("🔮 Predict Match Outcome", expanded=False):
    if len(known_players) < 4:
        st.info("Need at least 4 rated players to make a prediction.")
    else:
        pa, pb = st.columns(2)
        pA1 = pa.selectbox("Team A — P1", known_players, key="pred_A1")
        pA2 = pa.selectbox("Team A — P2",
                            [x for x in known_players if x != pA1], key="pred_A2")
        pB1 = pb.selectbox("Team B — P1",
                            [x for x in known_players if x not in (pA1, pA2)], key="pred_B1")
        pB2 = pb.selectbox("Team B — P2",
                            [x for x in known_players if x not in (pA1, pA2, pB1)], key="pred_B2")

        if st.button("🔮 Predict", type="primary"):
            rc   = dict(zip(ratings_df["player"], ratings_df["rating"].astype(float)))
            prob = predict_win_probability(rc, normalize(pA1), normalize(pA2),
                                           normalize(pB1), normalize(pB2))
            st.markdown(f"""
| Team | Players | Win Probability |
|------|---------|----------------|
| 🔵 Team A | {pA1} & {pA2} | **{prob*100:.1f}%** |
| 🔴 Team B | {pB1} & {pB2} | **{(1-prob)*100:.1f}%** |
""")

st.markdown("---")


# ═══════════════════════════════════════════════════════════════════════════════
# ⚡ PLAYER PICK SUGGESTIONS
# ═══════════════════════════════════════════════════════════════════════════════
with st.expander("⚡ Player Pick Suggestions", expanded=False):
    if len(known_players) < 2:
        st.info("Need at least 2 rated players.")
    else:
        rd_sorted    = ratings_df.sort_values("rating", ascending=False)
        players_list = list(rd_sorted["player"])
        ratings_map  = dict(zip(rd_sorted["player"], rd_sorted["rating"].astype(float)))

        col1, col2, col3 = st.columns(3)

        with col1:
            st.markdown("### 🥇 Top Rated Pair")
            if len(players_list) >= 2:
                p_a, p_b = players_list[0], players_list[1]
                st.write(f"**{p_a}** & **{p_b}**")
                st.caption(f"Avg rating: {(ratings_map[p_a]+ratings_map[p_b])/2:.1f}")

        with col2:
            st.markdown("### ⚖️ Most Balanced Pair")
            ba, bb, min_d = None, None, float("inf")
            for i in range(len(players_list)):
                for j in range(i + 1, len(players_list)):
                    d = abs(ratings_map[players_list[i]] - ratings_map[players_list[j]])
                    if d < min_d:
                        ba, bb, min_d = players_list[i], players_list[j], d
            if ba:
                st.write(f"**{ba}** & **{bb}**")
                st.caption(f"Rating gap: {min_d:.1f}")

        with col3:
            st.markdown("### 🔥 Hot Streak (last 5)")
            wc: dict[str, int] = {}
            for _, r in matches.tail(5).iterrows():
                scA = safe_int(r.scoreA); scB = safe_int(r.scoreB)
                ws  = [r.playerA1, r.playerA2] if scA > scB else [r.playerB1, r.playerB2]
                for p in ws:
                    if p: wc[p] = wc.get(p, 0) + 1
            for p, w in sorted(wc.items(), key=lambda x: x[1], reverse=True)[:3]:
                st.write(f"**{p}** — {w} wins")

st.markdown("---")

# ─── Logout ───────────────────────────────────────────────────────────────────
if st.button("Logout"):
    st.session_state.logged_in = False
    st.success("Logged out.")
    st.rerun()
