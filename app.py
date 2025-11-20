import streamlit as st
import pandas as pd
import numpy as np
from elo import update_elo, predict_win_probability

st.set_page_config(page_title="Badminton Tracker", layout="centered")

st.title("🏸 Badminton Doubles Tracker")

if "matches" not in st.session_state:
    st.session_state.matches = pd.DataFrame(columns=[
        "playerA1", "playerA2", "playerB1", "playerB2", "scoreA", "scoreB"
    ])

st.header("➕ Add Match")

with st.form("match_form"):
    col1, col2 = st.columns(2)
    playerA1 = col1.text_input("Team A - Player 1")
    playerA2 = col2.text_input("Team A - Player 2")
    playerB1 = col1.text_input("Team B - Player 1")
    playerB2 = col2.text_input("Team B - Player 2")
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

st.header("📜 Match History")
st.dataframe(st.session_state.matches)

# Ratings
st.header("⭐ Ratings")
ratings = {}
for _, m in st.session_state.matches.iterrows():
    ratings = update_elo(m.playerA1, m.playerA2, m.playerB1, m.playerB2,
                         m.scoreA, m.scoreB, ratings)

rating_df = pd.DataFrame(ratings.items(), columns=["Player", "Rating"])
st.dataframe(rating_df.sort_values("Rating", ascending=False))

# Predictions
st.header("🔮 Predict Match")

col3, col4 = st.columns(2)
pA1 = col3.text_input("Team A P1")
pA2 = col4.text_input("Team A P2")
pB1 = col3.text_input("Team B P1")
pB2 = col4.text_input("Team B P2")

if st.button("Predict"):
    if all(p in ratings for p in [pA1, pA2, pB1, pB2]):
        prob = predict_win_probability(ratings, pA1, pA2, pB1, pB2)
        st.success(f"Team A Win Probability: **{prob*100:.2f}%**")
    else:
        st.error("One or more players don't have ratings yet.")

