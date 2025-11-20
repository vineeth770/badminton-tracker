import math

def win_prob(teamA, teamB):
    return 1 / (1 + 10 ** ((teamB - teamA) / 400))

def update_elo(A1, A2, B1, B2, scoreA, scoreB, ratings, k=32):
    for p in [A1, A2, B1, B2]:
        if p not in ratings:
            ratings[p] = 1500

    teamA_rating = ratings[A1] + ratings[A2]
    teamB_rating = ratings[B1] + ratings[B2]

    expectedA = win_prob(teamA_rating, teamB_rating)
    actualA = 1 if scoreA > scoreB else 0

    delta = k * (actualA - expectedA)

    ratings[A1] += delta / 2
    ratings[A2] += delta / 2
    ratings[B1] -= delta / 2
    ratings[B2] -= delta / 2

    return ratings

def predict_win_probability(ratings, A1, A2, B1, B2):
    teamA = ratings[A1] + ratings[A2]
    teamB = ratings[B1] + ratings[B2]
    return win_prob(teamA, teamB)

