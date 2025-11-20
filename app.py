import streamlit as st
import pandas as pd
import gspread
from google.oauth2.service_account import Credentials
from elo import update_elo, predict_win_probability
import datetime

st.set_page_config(page_title="Badminton Tracker", layout="centered")

st.title("🏸 Badminton Doubles Tracker")

# ========== GOOGLE SHEETS AUTH ==========

scope = [
    "https://www.googleapis.com/auth/spreadsheets",
    "https://www.googleapis.com/auth/drive"
]

creds = Credentials.from_service_account_info(
    st.secrets["google_credentials"], scopes=scope
)

client = gspread.authorize(creds)

# Load tabs by URL
matches_sheet = client.open_by_url(st.secrets["sheet_matches"]).sheet1
players_sheet = client.open_by_url(st.secrets["sheet_players"]).sheet1


def load_matches():
    data = matches_sheet.get_all_records()
    if not data:
        return pd.DataFrame(columns=[
            "date", "playerA1", "playerA2", "playerB1", "playerB2", "scoreA", "scoreB"
        ])
    return pd.DataFrame(data)


def add_match(row):
    matches_sheet.append_row(row)


# ========== ADD MATCH FORM ==========

st.header("➕ Add New Match")

with st.form("match_form"):
    col1, col2 = st.columns(2)

    playerA1 = col1.text_input("Team A - Player 1")
    playerA2 = col2.text_input("Team A - Player 2")
    playerB1 = col1.text_input("Team B - Player 1")
    playerB2 = col2.text_input("Team B - Player 2")

    scoreA = col1.number_input("Score A", min_value=0)
    scoreB = col2.number_input("Score B", min_value=0)

    submitted = st.form_submit_button("Save Match")

if submitted:
    row = [
        str(datetime.date.today()),
        playerA1, playerA2, playerB1, playerB2,
        int(scoreA), int(scoreB)
    ]
    add_match(row)
    st.success("Match added successfully!")


# ========== MATCH HISTORY ==========

st.header("📜 Match History")
matches = load_matches()
st.dataframe(matches)

st.subheader(f"Total Matches Played: **{len(matches)}**")


# ========== PLAYER STATS ==========

def compute_player_stats(df):
    stats = {}

    for _, row in df.iterrows():
        A1, A2 = row["playerA1"], row["playerA2"]
        B1, B2 = row["playerB1"], row["playerB2"]
        scoreA, scoreB = row["scoreA"], row["scoreB"]

        for p in [A1, A2, B1, B2]:
            if p not in stats:
                stats[p] = {"matches": 0, "wins": 0, "losses": 0}

        stats[A1]["matches"] += 1
        stats[A2]["matches"] += 1
        stats[B1]["matches"] += 1
        stats[B2]["matches"] += 1

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

    for p in stats:
        m = stats[p]["matches"]
        w = stats[p]["wins"]
        stats[p]["win_pct"] = round((w / m) * 100, 1) if m > 0 else 0

    return stats


st.header("📊 Player Statistics")

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

st.dataframe(stats_df.sort_values("Win %", ascending=False))


# ========== ELO RATINGS ==========

st.header("⭐ Player Ratings")

ratings = {}
for _, m in matches.iterrows():
    ratings = update_elo(
        m["playerA1"], m["playerA2"], m["playerB1"], m["playerB2"],
        m["scoreA"], m["scoreB"], ratings
    )

rating_df = pd.DataFrame(ratings.items(), columns=["Player", "Rating"])
st.dataframe(rating_df.sort_values("Rating", ascending=False))


# ========== PREDICTION ==========

st.header("🔮 Predict Match Outcome")

col3, col4 = st.columns(2)
pA1 = col3.text_input("Team A - Player 1")
pA2 = col4.text_input("Team A - Player 2")
pB1 = col3.text_input("Team B - Player 1")
pB2 = col4.text_input("Team B - Player 2")

if st.button("Predict"):
    if all(p in ratings for p in [pA1, pA2, pB1, pB2]):
        prob = predict_win_probability(ratings, pA1, pA2, pB1, pB2)
        st.success(f"Team A Win Probability: **{prob*100:.2f}%**")
    else:
        st.error("One or more players do not have a rating yet.")
