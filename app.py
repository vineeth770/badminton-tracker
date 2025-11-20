import streamlit as st
import pandas as pd
import numpy as np
import requests
import base64
from io import StringIO
from elo import update_elo, predict_win_probability

st.set_page_config(page_title="Badminton Tracker", layout="centered")
st.title("🏸 Badminton Doubles Tracker")

# --------------------------
# Helper: normalize names
# --------------------------
def normalize(name):
    return name.strip().title()

# --------------------------
# GitHub read/write helpers
# --------------------------
def load_csv_from_github(path):
    url = f"https://api.github.com/repos/{st.secrets['REPO_OWNER']}/{st.secrets['REPO_NAME']}/contents/{path}"
    headers = {"Authorization": f"token {st.secrets['GITHUB_TOKEN']}"}

    response = requests.get(url, headers=headers).json()
    content = base64.b64decode(response["content"]).decode("utf-8")

    return pd.read_csv(StringIO(content)), response["sha"]


def save_csv_to_github(path, df, sha, message="update csv"):
    url = f"https://api.github.com/repos/{st.secrets['REPO_OWNER']}/{st.secrets['REPO_NAME']}/contents/{path}"
    headers = {"Authorization": f"token {st.secrets['GITHUB_TOKEN']}"}

    encoded = base64.b64encode(df.to_csv(index=False).encode()).decode()

    data = {
        "message": message,
        "content": encoded,
        "sha": sha
    }

    requests.put(url, headers=headers, json=data)


# --------------------------
# LOAD MATCHES + PLAYER RATINGS FROM GITHUB
# --------------------------
matches, matches_sha = load_csv_from_github(st.secrets["MATCHES_CSV"])
ratings_df, ratings_sha = load_csv_from_github(st.secrets["RATINGS_CSV"])

# Convert ratings to dict
ratings = {
    row["player"]: row["rating"]
    for _, row in ratings_df.iterrows()
}

player_stats = {
    row["player"]: {
        "wins": row["wins"],
        "losses": row["losses"],
        "matches": row["matches"]
    }
    for _, row in ratings_df.iterrows()
}


# --------------------------
# ADD MATCH SECTION
# --------------------------
st.header("➕ Add Match")

with st.form("match_form"):
    col1, col2 = st.columns(2)

    playerA1 = normalize(col1.text_input("Team A - Player 1"))
    playerA2 = normalize(col2.text_input("Team A - Player 2"))
    playerB1 = normalize(col1.text_input("Team B - Player 1"))
    playerB2 = normalize(col2.text_input("Team B - Player 2"))

    scoreA = col1.number_input("Score A", min_value=0)
    scoreB = col2.number_input("Score B", min_value=0)

    submitted = st.form_submit_button("Save Match")

if submitted:
    # Add to matches.csv
    new_row = pd.DataFrame([{
        "playerA1": playerA1,
        "playerA2": playerA2,
        "playerB1": playerB1,
        "playerB2": playerB2,
        "scoreA": scoreA,
        "scoreB": scoreB
    }])

    matches = pd.concat([matches, new_row], ignore_index=True)
    save_csv_to_github(st.secrets["MATCHES_CSV"], matches, matches_sha, "Added match")

    # --------------------------
    # Update ratings and stats
    # --------------------------
    for p in [playerA1, playerA2, playerB1, playerB2]:
        if p not in ratings:
            ratings[p] = 1500
            player_stats[p] = {"wins": 0, "losses": 0, "matches": 0}

    # Update match counts
    for p in [playerA1, playerA2, playerB1, playerB2]:
        player_stats[p]["matches"] += 1

    # Assign wins & losses
    if scoreA > scoreB:
        player_stats[playerA1]["wins"] += 1
        player_stats[playerA2]["wins"] += 1
        player_stats[playerB1]["losses"] += 1
        player_stats[playerB2]["losses"] += 1
    else:
        player_stats[playerB1]["wins"] += 1
        player_stats[playerB2]["wins"] += 1
        player_stats[playerA1]["losses"] += 1
        player_stats[playerA2]["losses"] += 1

    # Update ELO
    ratings = update_elo(playerA1, playerA2, playerB1, playerB2, scoreA, scoreB, ratings)

    # Save ratings.csv
    ratings_df = pd.DataFrame([
        {
            "player": p,
            "rating": round(ratings[p], 2),
            "wins": player_stats[p]["wins"],
            "losses": player_stats[p]["losses"],
            "matches": player_stats[p]["matches"]
        }
        for p in ratings
    ])

    save_csv_to_github(st.secrets["RATINGS_CSV"], ratings_df, ratings_sha, "Updated ratings")

    st.success("Match & ratings updated!")


# --------------------------
# MATCH HISTORY
# --------------------------
st.header("📜 Match History")
st.dataframe(matches)
st.subheader(f"Total Matches Played: **{len(matches)}**")


# --------------------------
# PLAYER STATISTICS
# --------------------------
st.header("📊 Player Statistics")
stats_display = ratings_df.copy()
stats_display["Win %"] = (stats_display["wins"] / stats_display["matches"] * 100).round(1)
st.dataframe(stats_display.sort_values("Win %", ascending=False))


# --------------------------
# ELO RATINGS TABLE
# --------------------------
st.header("⭐ Player Ratings")
st.dataframe(ratings_df.sort_values("rating", ascending=False))


# --------------------------
# PREDICTION
# --------------------------
st.header("🔮 Predict Match Outcome")

colA, colB = st.columns(2)

A1 = normalize(colA.text_input("Team A - P1"))
A2 = normalize(colA.text_input("Team A - P2"))
B1 = normalize(colB.text_input("Team B - P1"))
B2 = normalize(colB.text_input("Team B - P2"))

if st.button("Predict"):
    if all(p in ratings for p in [A1, A2, B1, B2]):
        prob = predict_win_probability(ratings, A1, A2, B1, B2)
        st.success(f"Team A win probability: **{prob*100:.2f}%**")
    else:
        st.error("One or more players do not exist in the system!")
