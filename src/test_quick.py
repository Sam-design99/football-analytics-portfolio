from statsbombpy import sb

# Test 1 - compétitions
print("=== COMPETITIONS ===")
comps = sb.competitions()
print(comps[["competition_name", "season_name"]].head(10))

# Test 2 - matchs Coupe du Monde 2018
print("\n=== MATCHS CDM 2018 ===")
matches = sb.matches(competition_id=43, season_id=3)
print(matches[["home_team", "away_team", "home_score", "away_score"]].head(5))

# Test 3 - tirs d'un match
print("\n=== TIRS PREMIER MATCH ===")
match_id = matches["match_id"].iloc[0]
events = sb.events(match_id=match_id)
shots = events[events["type"] == "Shot"]
print(f"Nombre de tirs : {len(shots)}")
print(shots[["minute", "player", "team", "shot_statsbomb_xg", "shot_outcome"]].head(10))