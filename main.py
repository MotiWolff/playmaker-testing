attribute_explanations = {
    'FTHG': 'Average goals scored by home team (last 5 games)',
    'FTAG': 'Average goals scored by away team (last 5 games)',
    'home_avg_conceded': 'Average goals conceded by home team',
    'away_avg_conceded': 'Average goals conceded by away team',
    'home_form': 'Recent form of home team (points from last 5 games)',
    'away_form': 'Recent form of away team (points from last 5 games)',
    'last5_matchup': 'Points from last 5 head-to-head matchups',
    'B365H': 'Bet365 odds for home win',
    'B365D': 'Bet365 odds for draw',
    'B365A': 'Bet365 odds for away win',
    'HS': 'Average home team shots per game',
    'AS': 'Average away team shots per game',
    'HC': 'Average home team corners per game',
    'AC': 'Average away team corners per game',
    'HY': 'Average home team yellow cards per game',
    'AY': 'Average away team yellow cards per game',
    'HR': 'Average home team red cards per game',
    'AR': 'Average away team red cards per game',
    'B365BTSY': 'Bet365 odds for both teams to score (Yes)',
    'B365BTSN': 'Bet365 odds for both teams to score (No)'
}

print("\nAttribute Explanations:")
for attr, expl in attribute_explanations.items():
    print(f"{attr}: {expl}")

# --- 1. Load and Merge La Liga CSVs ---
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

csv_files = ["E0.csv", "E0 (1).csv"]  # Premier League CSVs
dfs = [pd.read_csv(f) for f in csv_files]
df = pd.concat(dfs, ignore_index=True)

# --- 2. Extract Features ---
# Use FTHG (Full Time Home Goals), FTAG (Full Time Away Goals), FTR (Full Time Result)
# FTR: 'H' = home win, 'A' = away win, 'D' = draw
def result_to_binary(ftr):
    if ftr == 'H':
        return 1
    elif ftr == 'A':
        return 0
    else:
        return -1

df['result'] = df['FTR'].apply(result_to_binary)
# Omit draws for binary classification
df = df[df['result'] != -1]


# --- Add opponent strength features ---
def get_avg_conceded(team, df, n=5):
    home = df[df['HomeTeam'] == team].sort_values('Date', ascending=False).head(n)
    away = df[df['AwayTeam'] == team].sort_values('Date', ascending=False).head(n)
    avg_conceded = (home['FTAG'].sum() + away['FTHG'].sum()) / max(len(home) + len(away), 1)
    return avg_conceded

# Recent form: average points in last 5 matches
def get_recent_form(team, df, n=5):
    matches = pd.concat([
        df[df['HomeTeam'] == team],
        df[df['AwayTeam'] == team]
    ]).sort_values('Date', ascending=False).head(n)
    points = 0
    for _, m in matches.iterrows():
        if m['HomeTeam'] == team:
            if m['FTR'] == 'H':
                points += 3
            elif m['FTR'] == 'D':
                points += 1
        elif m['AwayTeam'] == team:
            if m['FTR'] == 'A':
                points += 3
            elif m['FTR'] == 'D':
                points += 1
    return points / n if n > 0 else 0

# Last 5 matchups: average points home team earned vs away team
def get_last5_matchup_points(home, away, df, n=5):
    matchups = df[(df['HomeTeam'] == home) & (df['AwayTeam'] == away)].sort_values('Date', ascending=False).head(n)
    points = 0
    for _, m in matchups.iterrows():
        if m['FTR'] == 'H':
            points += 3
        elif m['FTR'] == 'D':
            points += 1
    return points / n if n > 0 else 0

# Build training features with opponent strength
# Build training features with all enhancements
X_rows = []
for idx, row in df.iterrows():
    home = row['HomeTeam']
    away = row['AwayTeam']
    home_avg_conceded = get_avg_conceded(home, df)
    away_avg_conceded = get_avg_conceded(away, df)
    home_form = get_recent_form(home, df)
    away_form = get_recent_form(away, df)
    last5_matchup = get_last5_matchup_points(home, away, df)
    # Bookmaker odds
    b365h = row.get('B365H', None)
    b365d = row.get('B365D', None)
    b365a = row.get('B365A', None)
    # Extra stats (shots, corners, yellow/red cards)
    hs = row.get('HS', None)
    as_ = row.get('AS', None)
    hc = row.get('HC', None)
    ac = row.get('AC', None)
    hy = row.get('HY', None)
    ay = row.get('AY', None)
    hr = row.get('HR', None)
    ar = row.get('AR', None)
    X_rows.append([
        row['FTHG'],
        row['FTAG'],
        home_avg_conceded,
        away_avg_conceded,
        home_form,
        away_form,
        last5_matchup,
        b365h,
        b365d,
        b365a,
        hs,
        as_,
        hc,
        ac,
        hy,
        ay,
        hr,
        ar
    ])
X = pd.DataFrame(X_rows, columns=[
    'FTHG', 'FTAG', 'home_avg_conceded', 'away_avg_conceded', 'home_form', 'away_form', 'last5_matchup',
    'B365H', 'B365D', 'B365A', 'HS', 'AS', 'HC', 'AC', 'HY', 'AY', 'HR', 'AR'])
# Fill NaNs in features with league averages
for col in X.columns:
    league_avg = X[col].mean()
    X[col] = X[col].fillna(league_avg)

# Print class distribution
import collections
print("Class distribution (home=1, away=0):", collections.Counter(df['result']))
y = df['result']

# --- 3. Train/Test Split and Model ---
if len(df) < 2:
    print(f"Not enough matches with results for training. Found {len(df)} match(es). Need at least 2.")
    exit(1)


# Feature scaling
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)
X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)

# Random Forest model
model = RandomForestClassifier(n_estimators=100, random_state=42, min_samples_leaf=2, max_features='sqrt')
model.fit(X_train, y_train)
y_pred = model.predict(X_test)
print("Test Accuracy:", accuracy_score(y_test, y_pred))
importances = model.feature_importances_
feature_names = X.columns
print("Feature importances:")
for name, imp in zip(feature_names, importances):
    print(f"{name}: {imp:.3f}")


# --- 4. Predict Winner of a Sample Future Match ---
sample_home = 'Burnley FC'
sample_away = 'Liverpool FC'
sample_home_avg = df[df['HomeTeam'] == sample_home]['FTHG'].mean()
sample_away_avg = df[df['AwayTeam'] == sample_away]['FTAG'].mean()
sample_home_conceded = get_avg_conceded(sample_home, df)
sample_away_conceded = get_avg_conceded(sample_away, df)

# Add recent form, last 5 matchup, bookmaker odds, and extra stats to sample prediction
sample_home_form = get_recent_form(sample_home, df)
sample_away_form = get_recent_form(sample_away, df)
sample_last5_matchup = get_last5_matchup_points(sample_home, sample_away, df)
sample_b365h = df[df['HomeTeam'] == sample_home]['B365H'].mean()
sample_b365d = df[df['HomeTeam'] == sample_home]['B365D'].mean()
sample_b365a = df[df['HomeTeam'] == sample_home]['B365A'].mean()
sample_hs = df[df['HomeTeam'] == sample_home]['HS'].mean()
sample_as = df[df['AwayTeam'] == sample_away]['AS'].mean()
sample_hc = df[df['HomeTeam'] == sample_home]['HC'].mean()
sample_ac = df[df['AwayTeam'] == sample_away]['AC'].mean()
sample_hy = df[df['HomeTeam'] == sample_home]['HY'].mean()
sample_ay = df[df['AwayTeam'] == sample_away]['AY'].mean()
sample_hr = df[df['HomeTeam'] == sample_home]['HR'].mean()
sample_ar = df[df['AwayTeam'] == sample_away]['AR'].mean()
sample_data = pd.DataFrame({
    'FTHG': [sample_home_avg],
    'FTAG': [sample_away_avg],
    'home_avg_conceded': [sample_home_conceded],
    'away_avg_conceded': [sample_away_conceded],
    'home_form': [sample_home_form],
    'away_form': [sample_away_form],
    'last5_matchup': [sample_last5_matchup],
    'B365H': [sample_b365h],
    'B365D': [sample_b365d],
    'B365A': [sample_b365a],
    'HS': [sample_hs],
    'AS': [sample_as],
    'HC': [sample_hc],
    'AC': [sample_ac],
    'HY': [sample_hy],
    'AY': [sample_ay],
    'HR': [sample_hr],
    'AR': [sample_ar]
})
# Fill NaNs in sample_data with league averages
for col in sample_data.columns:
    league_avg = X[col].mean()
    sample_data[col] = sample_data[col].fillna(league_avg)
# Scale sample features
sample_scaled = scaler.transform(sample_data)
sample_pred = model.predict(sample_scaled)
print(f"Predicted winner for {sample_home} vs {sample_away} (1=home, 0=away):", sample_pred)



# --- 6. Predict Results for Next Three Matchdays Using API Fixtures ---
import requests
from difflib import get_close_matches
print("\nPredictions for Next Three Matchdays (using API fixtures):")

# football-data.org API setup
API_KEY = "2edd2d77128440d686e303c145feeef2"  # Replace with your actual API key
competition_id = "2021"  # Premier League competition code
url = f"https://api.football-data.org/v4/competitions/{competition_id}/matches?status=SCHEDULED"
headers = {"X-Auth-Token": API_KEY}
response = requests.get(url, headers=headers)
data = response.json()

if 'matches' not in data:
    print("Error: Could not fetch upcoming fixtures from API.")
else:
    # Get next 3 matchdays worth of fixtures
    matches = data['matches']
    # Group by matchday
    from collections import defaultdict
    matchdays = defaultdict(list)
    for m in matches:
        md = m.get('matchday', None)
        if md is not None:
            matchdays[md].append(m)
    # Sort matchdays and get next 3
    next_mds = sorted(matchdays.keys())[:3]
    for md in next_mds:
        print(f"\nMatchday {md} predictions:")
        for m in matchdays[md]:
            # Fuzzy match API team names to CSV team names
            api_home = m['homeTeam']['name']
            api_away = m['awayTeam']['name']
            csv_teams = set(df['HomeTeam']).union(set(df['AwayTeam']))
            home = get_close_matches(api_home, csv_teams, n=1, cutoff=0.7)
            away = get_close_matches(api_away, csv_teams, n=1, cutoff=0.7)
            home = home[0] if home else api_home
            away = away[0] if away else api_away
            # Estimate team strengths and opponent defense from historical data
            def get_team_stats(team, df, n=5):
                home_df = df[df['HomeTeam'] == team].sort_values('Date', ascending=False).head(n)
                away_df = df[df['AwayTeam'] == team].sort_values('Date', ascending=False).head(n)
                avg_scored = (home_df['FTHG'].sum() + away_df['FTAG'].sum()) / max(len(home_df) + len(away_df), 1)
                return avg_scored
            home_avg = get_team_stats(home, df)
            away_avg = get_team_stats(away, df)
            home_conceded = get_avg_conceded(home, df)
            away_conceded = get_avg_conceded(away, df)
            home_form = get_recent_form(home, df)
            away_form = get_recent_form(away, df)
            last5_matchup = get_last5_matchup_points(home, away, df)
            b365h = df[df['HomeTeam'] == home]['B365H'].mean()
            b365d = df[df['HomeTeam'] == home]['B365D'].mean()
            b365a = df[df['HomeTeam'] == home]['B365A'].mean()
            # For double chance, 1st half, tie no bet, both teams to score, use placeholders or similar odds if not available
            dc_1x = b365h
            dc_x2 = b365a
            dc_12 = b365h
            fh_1 = b365h
            fh_x = b365d
            fh_2 = b365a
            tnb_1 = b365h
            tnb_2 = b365a
            # Both teams to score odds if present
            btts_yes = df[df['HomeTeam'] == home]['B365BTSY'].mean() if 'B365BTSY' in df.columns else None
            btts_no = df[df['HomeTeam'] == home]['B365BTSN'].mean() if 'B365BTSN' in df.columns else None
            hs = df[df['HomeTeam'] == home]['HS'].mean()
            as_ = df[df['AwayTeam'] == away]['AS'].mean()
            hc = df[df['HomeTeam'] == home]['HC'].mean()
            ac = df[df['AwayTeam'] == away]['AC'].mean()
            hy = df[df['HomeTeam'] == home]['HY'].mean()
            ay = df[df['AwayTeam'] == away]['AY'].mean()
            hr = df[df['HomeTeam'] == home]['HR'].mean()
            ar = df[df['AwayTeam'] == away]['AR'].mean()
            sample = pd.DataFrame({
                'FTHG': [round(home_avg, 2)],
                'FTAG': [round(away_avg, 2)],
                'home_avg_conceded': [home_conceded],
                'away_avg_conceded': [away_conceded],
                'home_form': [home_form],
                'away_form': [away_form],
                'last5_matchup': [last5_matchup],
                'B365H': [b365h],
                'B365D': [b365d],
                'B365A': [b365a],
                'HS': [hs],
                'AS': [as_],
                'HC': [hc],
                'AC': [ac],
                'HY': [hy],
                'AY': [ay],
                'HR': [hr],
                'AR': [ar]
            })
            # Fill NaNs in sample with league averages
            for col in sample.columns:
                league_avg = X[col].mean()
                sample[col] = sample[col].fillna(league_avg)
            # Scale sample features
            sample_scaled = scaler.transform(sample)
            pred = model.predict(sample_scaled)[0]
            result = 'Home win' if pred == 1 else 'Away win'
            # Helper to format odds
            def fmt_odds(val):
                if val is None or pd.isna(val):
                    return 'N/A'
                # Convert decimal to American odds
                if val >= 2:
                    return f"+{int((val-1)*100)}"
                else:
                    return f"-{int(100/(val-1))}"

            print(f"\n{api_home} vs {api_away} (CSV match: {home} vs {away})")
            print("--------------------------------------------------")
            print(f"Full-time odds: Home: {fmt_odds(b365h)} | Draw: {fmt_odds(b365d)} | Away: {fmt_odds(b365a)}")
            print(f"Double chance: 1X: {fmt_odds(dc_1x)} | X2: {fmt_odds(dc_x2)} | 12: {fmt_odds(dc_12)}")
            print(f"1st half odds: Home: {fmt_odds(fh_1)} | Draw: {fmt_odds(fh_x)} | Away: {fmt_odds(fh_2)}")
            print(f"Tie no bet: Home: {fmt_odds(tnb_1)} | Away: {fmt_odds(tnb_2)}")
            print(f"Both teams to score: Yes: {fmt_odds(btts_yes)} | No: {fmt_odds(btts_no)}")
            print(f"Model prediction: {result}")
            print("--------------------------------------------------")

