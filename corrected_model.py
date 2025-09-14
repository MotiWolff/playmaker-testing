"""
Corrected La Liga Prediction Model
Fixes data leakage and other issues from the original code
"""

import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')

def load_and_prepare_data(csv_files):
    """Load and prepare data with proper date handling."""
    print("Loading CSV files...")
    dfs = []
    for file in csv_files:
        try:
            df = pd.read_csv(file)
            # Convert date column
            df['Date'] = pd.to_datetime(df['Date'], format='%d/%m/%Y', errors='coerce')
            df = df.dropna(subset=['Date'])  # Remove rows with invalid dates
            dfs.append(df)
            print(f"Loaded {file}: {len(df)} matches")
        except Exception as e:
            print(f"Error loading {file}: {e}")
    
    if not dfs:
        raise ValueError("No valid CSV files loaded")
    
    df = pd.concat(dfs, ignore_index=True)
    df = df.sort_values('Date').reset_index(drop=True)
    print(f"Total matches loaded: {len(df)}")
    return df

def calculate_historical_features(df, match_idx, lookback=5):
    """
    Calculate features using ONLY historical data (no data leakage).
    For each match, only use data from matches that happened BEFORE it.
    """
    current_match = df.iloc[match_idx]
    current_date = current_match['Date']
    
    # Get historical data (only matches before current match)
    historical_data = df[df['Date'] < current_date].copy()
    
    if len(historical_data) < 2:
        # Not enough historical data
        return {
            'home_avg_conceded': 1.5,  # League average
            'away_avg_conceded': 1.5,
            'home_form': 1.5,  # League average points
            'away_form': 1.5,
            'last5_matchup': 1.5,
            'home_goals_avg': 1.5,
            'away_goals_avg': 1.5
        }
    
    home_team = current_match['HomeTeam']
    away_team = current_match['AwayTeam']
    
    # Calculate home team features
    home_matches = historical_data[
        (historical_data['HomeTeam'] == home_team) | 
        (historical_data['AwayTeam'] == home_team)
    ].tail(lookback * 2)  # Get more matches to ensure we have enough
    
    home_goals_scored = []
    home_goals_conceded = []
    home_points = []
    
    for _, match in home_matches.iterrows():
        if match['HomeTeam'] == home_team:
            home_goals_scored.append(match['FTHG'])
            home_goals_conceded.append(match['FTAG'])
            if match['FTR'] == 'H':
                home_points.append(3)
            elif match['FTR'] == 'D':
                home_points.append(1)
            else:
                home_points.append(0)
        elif match['AwayTeam'] == home_team:
            home_goals_scored.append(match['FTAG'])
            home_goals_conceded.append(match['FTHG'])
            if match['FTR'] == 'A':
                home_points.append(3)
            elif match['FTR'] == 'D':
                home_points.append(1)
            else:
                home_points.append(0)
    
    # Calculate away team features
    away_matches = historical_data[
        (historical_data['HomeTeam'] == away_team) | 
        (historical_data['AwayTeam'] == away_team)
    ].tail(lookback * 2)
    
    away_goals_scored = []
    away_goals_conceded = []
    away_points = []
    
    for _, match in away_matches.iterrows():
        if match['HomeTeam'] == away_team:
            away_goals_scored.append(match['FTHG'])
            away_goals_conceded.append(match['FTAG'])
            if match['FTR'] == 'H':
                away_points.append(3)
            elif match['FTR'] == 'D':
                away_points.append(1)
            else:
                away_points.append(0)
        elif match['AwayTeam'] == away_team:
            away_goals_scored.append(match['FTAG'])
            away_goals_conceded.append(match['FTHG'])
            if match['FTR'] == 'A':
                away_points.append(3)
            elif match['FTR'] == 'D':
                away_points.append(1)
            else:
                away_points.append(0)
    
    # Calculate head-to-head
    h2h_matches = historical_data[
        (historical_data['HomeTeam'] == home_team) & 
        (historical_data['AwayTeam'] == away_team)
    ].tail(lookback)
    
    h2h_points = []
    for _, match in h2h_matches.iterrows():
        if match['FTR'] == 'H':
            h2h_points.append(3)
        elif match['FTR'] == 'D':
            h2h_points.append(1)
        else:
            h2h_points.append(0)
    
    # Return features with fallbacks
    return {
        'home_avg_conceded': np.mean(home_goals_conceded) if home_goals_conceded else 1.5,
        'away_avg_conceded': np.mean(away_goals_conceded) if away_goals_conceded else 1.5,
        'home_form': np.mean(home_points) if home_points else 1.5,
        'away_form': np.mean(away_points) if away_points else 1.5,
        'last5_matchup': np.mean(h2h_points) if h2h_points else 1.5,
        'home_goals_avg': np.mean(home_goals_scored) if home_goals_scored else 1.5,
        'away_goals_avg': np.mean(away_goals_scored) if away_goals_scored else 1.5
    }

def create_training_data(df):
    """Create training data with proper feature engineering."""
    print("Creating training features...")
    
    # Only use matches with valid results
    df_clean = df.dropna(subset=['FTHG', 'FTAG', 'FTR', 'HomeTeam', 'AwayTeam']).copy()
    
    # Convert result to binary (exclude draws for binary classification)
    df_clean['result'] = df_clean['FTR'].apply(lambda x: 1 if x == 'H' else 0 if x == 'A' else -1)
    df_clean = df_clean[df_clean['result'] != -1]  # Remove draws
    
    print(f"Matches with valid results: {len(df_clean)}")
    
    if len(df_clean) < 10:
        raise ValueError("Not enough matches for training")
    
    # Calculate features for each match
    features = []
    targets = []
    
    for idx in range(len(df_clean)):
        if idx % 100 == 0:
            print(f"Processing match {idx}/{len(df_clean)}")
        
        match = df_clean.iloc[idx]
        
        # Calculate historical features
        hist_features = calculate_historical_features(df_clean, idx)
        
        # Add betting odds if available
        b365h = match.get('B365H', 0)
        b365d = match.get('B365D', 0)
        b365a = match.get('B365A', 0)
        
        # Add statistics if available
        hs = match.get('HS', 0)
        as_ = match.get('AS', 0)
        hc = match.get('HC', 0)
        ac = match.get('AC', 0)
        
        feature_vector = [
            hist_features['home_goals_avg'],
            hist_features['away_goals_avg'],
            hist_features['home_avg_conceded'],
            hist_features['away_avg_conceded'],
            hist_features['home_form'],
            hist_features['away_form'],
            hist_features['last5_matchup'],
            b365h if b365h > 0 else 0,
            b365d if b365d > 0 else 0,
            b365a if b365a > 0 else 0,
            hs if hs > 0 else 0,
            as_ if as_ > 0 else 0,
            hc if hc > 0 else 0,
            ac if ac > 0 else 0
        ]
        
        features.append(feature_vector)
        targets.append(match['result'])
    
    feature_names = [
        'home_goals_avg', 'away_goals_avg', 'home_avg_conceded', 'away_avg_conceded',
        'home_form', 'away_form', 'last5_matchup', 'b365h', 'b365d', 'b365a',
        'hs', 'as', 'hc', 'ac'
    ]
    
    X = pd.DataFrame(features, columns=feature_names)
    y = pd.Series(targets)
    
    # Fill any remaining NaNs
    X = X.fillna(X.mean())
    
    return X, y

def train_model(X, y):
    """Train the Random Forest model."""
    print("Training model...")
    
    # Split data
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    
    # Scale features
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    # Train model
    model = RandomForestClassifier(
        n_estimators=100,
        random_state=42,
        min_samples_leaf=5,
        max_features='sqrt',
        class_weight='balanced'
    )
    
    model.fit(X_train_scaled, y_train)
    
    # Evaluate
    y_pred = model.predict(X_test_scaled)
    accuracy = accuracy_score(y_test, y_pred)
    
    print(f"Model accuracy: {accuracy:.3f}")
    print("\nClassification Report:")
    print(classification_report(y_test, y_pred, target_names=['Away Win', 'Home Win']))
    
    # Feature importance
    feature_importance = pd.DataFrame({
        'feature': X.columns,
        'importance': model.feature_importances_
    }).sort_values('importance', ascending=False)
    
    print("\nTop 10 Most Important Features:")
    print(feature_importance.head(10))
    
    return model, scaler, X.columns

def predict_future_match(model, scaler, feature_names, df, home_team, away_team):
    """Predict a future match using historical data."""
    # Get all historical data for feature calculation
    historical_data = df.copy()
    
    # Calculate features for home team
    home_matches = historical_data[
        (historical_data['HomeTeam'] == home_team) | 
        (historical_data['AwayTeam'] == home_team)
    ].tail(10)
    
    home_goals_scored = []
    home_goals_conceded = []
    home_points = []
    
    for _, match in home_matches.iterrows():
        if match['HomeTeam'] == home_team:
            home_goals_scored.append(match['FTHG'])
            home_goals_conceded.append(match['FTAG'])
            if match['FTR'] == 'H':
                home_points.append(3)
            elif match['FTR'] == 'D':
                home_points.append(1)
            else:
                home_points.append(0)
        elif match['AwayTeam'] == home_team:
            home_goals_scored.append(match['FTAG'])
            home_goals_conceded.append(match['FTHG'])
            if match['FTR'] == 'A':
                home_points.append(3)
            elif match['FTR'] == 'D':
                home_points.append(1)
            else:
                home_points.append(0)
    
    # Calculate features for away team
    away_matches = historical_data[
        (historical_data['HomeTeam'] == away_team) | 
        (historical_data['AwayTeam'] == away_team)
    ].tail(10)
    
    away_goals_scored = []
    away_goals_conceded = []
    away_points = []
    
    for _, match in away_matches.iterrows():
        if match['HomeTeam'] == away_team:
            away_goals_scored.append(match['FTHG'])
            away_goals_conceded.append(match['FTAG'])
            if match['FTR'] == 'H':
                away_points.append(3)
            elif match['FTR'] == 'D':
                away_points.append(1)
            else:
                away_points.append(0)
        elif match['AwayTeam'] == away_team:
            away_goals_scored.append(match['FTAG'])
            away_goals_conceded.append(match['FTHG'])
            if match['FTR'] == 'A':
                away_points.append(3)
            elif match['FTR'] == 'D':
                away_points.append(1)
            else:
                away_points.append(0)
    
    # Calculate head-to-head
    h2h_matches = historical_data[
        (historical_data['HomeTeam'] == home_team) & 
        (historical_data['AwayTeam'] == away_team)
    ].tail(5)
    
    h2h_points = []
    for _, match in h2h_matches.iterrows():
        if match['FTR'] == 'H':
            h2h_points.append(3)
        elif match['FTR'] == 'D':
            h2h_points.append(1)
        else:
            h2h_points.append(0)
    
    # Get average betting odds
    home_odds = home_matches[['B365H', 'B365D', 'B365A']].mean()
    away_odds = away_matches[['B365H', 'B365D', 'B365A']].mean()
    
    # Get average statistics
    home_stats = home_matches[['HS', 'HC']].mean()
    away_stats = away_matches[['AS', 'AC']].mean()
    
    # Create feature vector
    feature_vector = [
        np.mean(home_goals_scored) if home_goals_scored else 1.5,
        np.mean(away_goals_scored) if away_goals_scored else 1.5,
        np.mean(home_goals_conceded) if home_goals_conceded else 1.5,
        np.mean(away_goals_conceded) if away_goals_conceded else 1.5,
        np.mean(home_points) if home_points else 1.5,
        np.mean(away_points) if away_points else 1.5,
        np.mean(h2h_points) if h2h_points else 1.5,
        home_odds.get('B365H', 0) if not home_odds.isna().all() else 0,
        home_odds.get('B365D', 0) if not home_odds.isna().all() else 0,
        home_odds.get('B365A', 0) if not home_odds.isna().all() else 0,
        home_stats.get('HS', 0) if not home_stats.isna().all() else 0,
        away_stats.get('AS', 0) if not away_stats.isna().all() else 0,
        home_stats.get('HC', 0) if not home_stats.isna().all() else 0,
        away_stats.get('AC', 0) if not away_stats.isna().all() else 0
    ]
    
    # Scale features
    X_pred = pd.DataFrame([feature_vector], columns=feature_names)
    X_pred_scaled = scaler.transform(X_pred)
    
    # Make prediction
    prediction = model.predict(X_pred_scaled)[0]
    probabilities = model.predict_proba(X_pred_scaled)[0]
    
    return prediction, probabilities

def main():
    """Main execution function."""
    print("=== Corrected La Liga Prediction Model ===\n")
    
    # Load data
    csv_files = ["E0.csv", "E0 (1).csv"]
    df = load_and_prepare_data(csv_files)
    
    # Create training data
    X, y = create_training_data(df)
    
    # Train model
    model, scaler, feature_names = train_model(X, y)
    
    # Test prediction
    print("\n=== Sample Prediction ===")
    home_team = "Arsenal"
    away_team = "Chelsea"
    
    try:
        prediction, probabilities = predict_future_match(
            model, scaler, feature_names, df, home_team, away_team
        )
        
        result = "Home Win" if prediction == 1 else "Away Win"
        home_prob = probabilities[1]
        away_prob = probabilities[0]
        
        print(f"{home_team} vs {away_team}")
        print(f"Prediction: {result}")
        print(f"Home Win Probability: {home_prob:.3f}")
        print(f"Away Win Probability: {away_prob:.3f}")
        
    except Exception as e:
        print(f"Error making prediction: {e}")

if __name__ == "__main__":
    main()
