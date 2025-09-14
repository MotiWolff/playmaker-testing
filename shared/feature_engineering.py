"""Feature engineering utilities for Playmaker."""

import pandas as pd
from typing import Dict, Any
from sqlalchemy.orm import Session

def get_avg_conceded(team_id: int, df: pd.DataFrame, n: int = 5) -> float:
    """Calculate average goals conceded by a team in last n matches."""
    home_matches = df[df['home_team_id'] == team_id].sort_values('match_date', ascending=False).head(n)
    away_matches = df[df['away_team_id'] == team_id].sort_values('match_date', ascending=False).head(n)
    
    total_conceded = home_matches['away_goals'].sum() + away_matches['home_goals'].sum()
    total_matches = len(home_matches) + len(away_matches)
    
    return total_conceded / max(total_matches, 1)

def get_recent_form(team_id: int, df: pd.DataFrame, n: int = 5) -> float:
    """Calculate recent form (average points) for a team in last n matches."""
    matches = pd.concat([
        df[df['home_team_id'] == team_id],
        df[df['away_team_id'] == team_id]
    ]).sort_values('match_date', ascending=False).head(n)
    
    points = 0
    for _, match in matches.iterrows():
        if match['home_team_id'] == team_id:
            if match['result'] == 'H':
                points += 3
            elif match['result'] == 'D':
                points += 1
        elif match['away_team_id'] == team_id:
            if match['result'] == 'A':
                points += 3
            elif match['result'] == 'D':
                points += 1
    
    return points / n if n > 0 else 0

def get_last5_matchup_points(home_team_id: int, away_team_id: int, df: pd.DataFrame, n: int = 5) -> float:
    """Calculate average points home team earned vs away team in last n matchups."""
    matchups = df[
        (df['home_team_id'] == home_team_id) & 
        (df['away_team_id'] == away_team_id)
    ].sort_values('match_date', ascending=False).head(n)
    
    points = 0
    for _, match in matchups.iterrows():
        if match['result'] == 'H':
            points += 3
        elif match['result'] == 'D':
            points += 1
    
    return points / n if n > 0 else 0

def create_features_for_match(home_team_id: int, away_team_id: int, db: Session) -> Dict[str, Any]:
    """Create features for a specific match."""
    from shared.database import MatchClean
    
    # Get all historical matches
    matches = db.query(MatchClean).filter(
        MatchClean.home_goals.isnot(None),
        MatchClean.away_goals.isnot(None)
    ).all()
    
    if not matches:
        return {}
    
    # Convert to DataFrame for easier processing
    df = pd.DataFrame([{
        'match_id': m.match_id,
        'match_date': m.match_date,
        'home_team_id': m.home_team_id,
        'away_team_id': m.away_team_id,
        'home_goals': m.home_goals,
        'away_goals': m.away_goals,
        'result': m.result,
        'b365h': m.b365h,
        'b365d': m.b365d,
        'b365a': m.b365a,
        'hs': m.hs,
        'away_shots': m.away_shots,
        'hc': m.hc,
        'ac': m.ac,
        'hy': m.hy,
        'ay': m.ay,
        'hr': m.hr,
        'ar': m.ar
    } for m in matches])
    
    # Calculate features
    home_avg_conceded = get_avg_conceded(home_team_id, df)
    away_avg_conceded = get_avg_conceded(away_team_id, df)
    home_form = get_recent_form(home_team_id, df)
    away_form = get_recent_form(away_team_id, df)
    last5_matchup = get_last5_matchup_points(home_team_id, away_team_id, df)
    
    # Get average betting odds for teams
    home_odds = df[df['home_team_id'] == home_team_id][['b365h', 'b365d', 'b365a']].mean()
    away_odds = df[df['away_team_id'] == away_team_id][['b365h', 'b365d', 'b365a']].mean()
    
    # Get average statistics
    home_stats = df[df['home_team_id'] == home_team_id][['hs', 'hc', 'hy', 'hr']].mean()
    away_stats = df[df['away_team_id'] == away_team_id][['away_shots', 'ac', 'ay', 'ar']].mean()
    
    return {
        'home_avg_conceded': float(home_avg_conceded),
        'away_avg_conceded': float(away_avg_conceded),
        'home_form': float(home_form),
        'away_form': float(away_form),
        'last5_matchup': float(last5_matchup),
        'b365h': float(home_odds.get('b365h', 0)) if not home_odds.isna().all() else 0,
        'b365d': float(home_odds.get('b365d', 0)) if not home_odds.isna().all() else 0,
        'b365a': float(home_odds.get('b365a', 0)) if not home_odds.isna().all() else 0,
        'hs': float(home_stats.get('hs', 0)) if not home_stats.isna().all() else 0,
        'away_shots': float(away_stats.get('away_shots', 0)) if not away_stats.isna().all() else 0,
        'hc': float(home_stats.get('hc', 0)) if not home_stats.isna().all() else 0,
        'ac': float(away_stats.get('ac', 0)) if not away_stats.isna().all() else 0,
        'hy': float(home_stats.get('hy', 0)) if not home_stats.isna().all() else 0,
        'ay': float(away_stats.get('ay', 0)) if not away_stats.isna().all() else 0,
        'hr': float(home_stats.get('hr', 0)) if not home_stats.isna().all() else 0,
        'ar': float(away_stats.get('ar', 0)) if not away_stats.isna().all() else 0,
    }
