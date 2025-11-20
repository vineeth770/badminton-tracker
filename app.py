import streamlit as st
import pandas as pd
import numpy as np
from elo import update_elo, predict_win_probability

st.set_page_config(page_title="Badminton Tracker", layout="centered")

st.title("🏸 Badminton Doubles Tracker")

# Normalize name (case insensitive)
def normalize(name):
    return name.strip().title()  # "amit", "AMIT" → "Amit"

# Initialize local session matches storage
if "matches" not in st.session_state:
    st.session_state.matches = pd.DataFrame(columns=[
        "playerA1", "playerA2", "playerB1", "playerB2", "scoreA", "scoreB"
    ])

# ==========================
# ADD MATCH
# ==========================
st.header("➕ Add Match")

with st.form("match_form"):
    col1, col2 = st.columns(2)

    playerA1 = normalize(col1.text_input("Team A - Player 1"))
    playerA2 = normalize(col2.text_input("Team A - Player 2"))
    playerB1 = normalize(col1.text_input("Team B - Player 1"))
    playerB2 = normalize(col2.text_input("Team B - Player 2"))

    scoreA = col1.number_input("Score A", min_value=0)
    scoreB = col2.number_input("Score B", min_value=0)

    submit = st.form_submit_button("Save Match")

if submit:
    new_match = pd.DataFrame([{
        "playerA1": playerA1,
        "playerA2": playerA2,
        "playerB1": playerB1,
        "playerB2": playerB2,
        "scoreA": scoreA,
        "scoreB": scoreB
    }])
    st.session_state.matches = pd.concat([st.session_state.matches, new_match], ignore_index=True)
    st.success("Match added!")

# ==========================
# MATCH HISTORY
# ==========================
st.header("📜 Match History")

matches = st.session_state.matches
st.dataframe(matches)
st.subheader(f"Total Matches Played: **{len(matches)}**")

# ==========================
# PLAYER INDIVIDUAL STATS
# ==========================
st.header("📊 Player Statistics")

def compute_player_stats(df):
    stats = {}

    for _, row in df.iterrows():
        A1, A2 = row["playerA1"], row["playerA2"]
        B1, B2 = row["playerB1"], row["playerB2"]
        scoreA, scoreB = row["scoreA"], row["scoreB"]

        # Initialize dict entries
        for p in [A1, A2, B1, B2]:
            stats.setdefault(p, {"matches": 0, "wins": 0, "losses": 0})

        # Everyone played a match
        for p in [A1, A2, B1, B2]:
            stats[p]["matches"] += 1

        # Update win/loss
        if scoreA > scoreB:
            stats[A1]["wins"] += 1
            stats[A2]["wins"] += 1
            stats[B1]["losses"] += 1
            stats[B2]["losses"] += 1
        else:
            stats[B1]["wins"] += 1
            stats[B2]["wins"] += 1
            stats[A1]["losses"] += 1
            stats[A2]["losses"] += 1

    # Compute win %
    for p in stats:
        m = stats[p]["matches"]
        w = stats[p]["wins"]
        stats[p]["win_pct"] = round((w / m) * 100, 1) if m else 0

    return stats

player_stats = compute_player_stats(matches)

stats_df = pd.DataFrame([
    {
        "Player": p,
        "Matches": player_stats[p]["matches"],
        "Wins": player_stats[p]["wins"],
        "Losses": player_stats[p]["losses"],
        "Win %": player_stats[p]["win_pct"]
    }
    for p in player_stats
])

# Ensure columns always exist even if dataframe is empty
if stats_df.empty:
    stats_df = pd.DataFrame(columns=["Player", "Matches", "Wins", "Losses", "Win %"])

st.dataframe(stats_df.sort_values("Win %", ascending=False))


# ==========================
# ELO RATINGS
# ==========================
st.header("⭐ Player Ratings")

ratings = {}
for _, m in matches.iterrows():
    ratings = update_elo(
        m["playerA1"], m["playerA2"],
        m["playerB1"], m["playerB2"],
        m["scoreA"], m["scoreB"],
        ratings
    )

rating_df = pd.DataFrame(
    [(player, round(rating, 2)) for player, rating in ratings.items()],
    columns=["Player", "Rating"]
)

st.dataframe(rating_df.sort_values("Rating", ascending=False))

# ==========================
# PREDICTION
# ==========================
st.header("🔮 Predict Match Outcome")

col3, col4 = st.columns(2)
pA1 = normalize(col3.text_input("Team A P1"))
pA2 = normalize(col4.text_input("Team A P2"))
pB1 = normalize(col3.text_input("Team B P1"))
pB2 = normalize(col4.text_input("Team B P2"))

if st.button("Predict"):
    if all(p in ratings for p in [pA1, pA2, pB1, pB2]):
        prob = predict_win_probability(ratings, pA1, pA2, pB1, pB2)
        st.success(f"Team A Win Probability: **{prob*100:.2f}%**")
    else:
        st.error("One or more players don't have ratings yet.")
