"""Model Service - Trains models and generates predictions."""

import os
import sys
import pandas as pd
import numpy as np
import joblib
from datetime import datetime
from typing import Dict, Any, List, Tuple
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, log_loss
from sqlalchemy.orm import Session

# Add shared to path
sys.path.append('/app')
from shared.database import get_db, MatchClean, Fixture, ModelVersion, Prediction, Team
from shared.kafka_client import KafkaConsumerClient, KafkaProducerClient
from shared.elasticsearch_logger import get_logger

class ModelService:
    def __init__(self):
        self.db = next(get_db())
        self.kafka_consumer = KafkaConsumerClient(['features.soccer'], 'model-service')
        self.kafka_producer = KafkaProducerClient()
        self.scaler = StandardScaler()
        self.model = None
        self.model_id = None
        self.logger = get_logger("model-service")
    
    def prepare_training_data(self) -> Tuple[pd.DataFrame, pd.Series]:
        """Prepare training data from clean matches."""
        print("Preparing training data...")
        
        # Get all clean matches with results
        matches = self.db.query(MatchClean).filter(
            MatchClean.home_goals.isnot(None),
            MatchClean.away_goals.isnot(None),
            MatchClean.result.isnot(None)
        ).all()
        
        if len(matches) < 10:
            print(f"Not enough matches for training: {len(matches)}")
            return pd.DataFrame(), pd.Series()
        
        # Convert to DataFrame
        data = []
        for match in matches:
            data.append({
                'home_goals': match.home_goals,
                'away_goals': match.away_goals,
                'home_avg_conceded': match.home_avg_conceded or 0,
                'away_avg_conceded': match.away_avg_conceded or 0,
                'home_form': match.home_form or 0,
                'away_form': match.away_form or 0,
                'last5_matchup': match.last5_matchup or 0,
                'b365h': match.b365h or 0,
                'b365d': match.b365d or 0,
                'b365a': match.b365a or 0,
                'hs': match.hs or 0,
                'as_': match.away_shots or 0,
                'hc': match.hc or 0,
                'ac': match.ac or 0,
                'hy': match.hy or 0,
                'ay': match.ay or 0,
                'hr': match.hr or 0,
                'ar': match.ar or 0,
                'result': match.result
            })
        
        df = pd.DataFrame(data)
        
        # Convert result to binary (1 for home win, 0 for away win, exclude draws for binary classification)
        df['result_binary'] = df['result'].apply(lambda x: 1 if x == 'H' else 0 if x == 'A' else -1)
        df = df[df['result_binary'] != -1]  # Remove draws
        
        if len(df) < 5:
            print(f"Not enough matches after removing draws: {len(df)}")
            return pd.DataFrame(), pd.Series()
        
        # Prepare features
        feature_columns = [
            'home_goals', 'away_goals', 'home_avg_conceded', 'away_avg_conceded',
            'home_form', 'away_form', 'last5_matchup', 'b365h', 'b365d', 'b365a',
            'hs', 'as_', 'hc', 'ac', 'hy', 'ay', 'hr', 'ar'
        ]
        
        X = df[feature_columns].fillna(0)
        y = df['result_binary']
        
        return X, y
    
    def train_model(self) -> Dict[str, Any]:
        """Train the model and save it."""
        print("Training model...")
        
        X, y = self.prepare_training_data()
        
        if X.empty or len(y) == 0:
            print("No training data available")
            return {}
        
        # Split data
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42, stratify=y
        )
        
        # Scale features
        X_train_scaled = self.scaler.fit_transform(X_train)
        X_test_scaled = self.scaler.transform(X_test)
        
        # Train Random Forest
        self.model = RandomForestClassifier(
            n_estimators=100,
            random_state=42,
            min_samples_leaf=2,
            max_features='sqrt'
        )
        
        self.model.fit(X_train_scaled, y_train)
        
        # Evaluate model
        y_pred = self.model.predict(X_test_scaled)
        y_pred_proba = self.model.predict_proba(X_test_scaled)
        
        accuracy = accuracy_score(y_test, y_pred)
        logloss = log_loss(y_test, y_pred_proba)
        
        print(f"Model accuracy: {accuracy:.3f}")
        print(f"Model log loss: {logloss:.3f}")
        
        # Save model
        model_name = f"rf_v{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        artifact_path = f"/app/models/{model_name}.joblib"
        
        os.makedirs("/app/models", exist_ok=True)
        joblib.dump({
            'model': self.model,
            'scaler': self.scaler,
            'feature_columns': X.columns.tolist()
        }, artifact_path)
        
        # Save model version to database
        model_version = ModelVersion(
            model_name=model_name,
            trained_on_dset=f"matches_{len(X)}",
            metrics={
                'accuracy': float(accuracy),
                'log_loss': float(logloss),
                'training_samples': len(X_train),
                'test_samples': len(X_test)
            },
            artifact_uri=artifact_path
        )
        
        self.db.add(model_version)
        self.db.commit()
        
        self.model_id = model_version.model_id
        
        return {
            'model_id': self.model_id,
            'model_name': model_name,
            'accuracy': accuracy,
            'log_loss': logloss
        }
    
    def predict_fixtures(self):
        """Generate predictions for upcoming fixtures."""
        print("Generating predictions for fixtures...")
        
        if not self.model or not self.model_id:
            print("No trained model available")
            return
        
        print(f"Model available: {self.model is not None}")
        print(f"Model ID: {self.model_id}")
        
        # Get upcoming fixtures
        fixtures = self.db.query(Fixture).filter(
            Fixture.status.in_(['SCHEDULED', 'TIMED'])
        ).all()
        
        print(f"Found {len(fixtures)} fixtures to predict")
        predictions_generated = 0
        
        for fixture in fixtures:
            try:
                # Check if prediction already exists
                existing = self.db.query(Prediction).filter(
                    Prediction.model_id == self.model_id,
                    Prediction.fixture_id == fixture.fixture_id
                ).first()
                
                if existing:
                    continue
                
                # Get team names for feature calculation
                home_team = self.db.query(Team).filter(Team.team_id == fixture.home_team_id).first()
                away_team = self.db.query(Team).filter(Team.team_id == fixture.away_team_id).first()
                
                if not home_team or not away_team:
                    continue
                
                # Calculate features for this fixture
                features = self._calculate_fixture_features(fixture)
                
                if not features:
                    continue
                
                # Prepare feature vector
                feature_vector = np.array([[
                    features.get('home_goals', 0),
                    features.get('away_goals', 0),
                    features.get('home_avg_conceded', 0),
                    features.get('away_avg_conceded', 0),
                    features.get('home_form', 0),
                    features.get('away_form', 0),
                    features.get('last5_matchup', 0),
                    features.get('b365h', 0),
                    features.get('b365d', 0),
                    features.get('b365a', 0),
                    features.get('hs', 0),
                    features.get('as_', 0),
                    features.get('hc', 0),
                    features.get('ac', 0),
                    features.get('hy', 0),
                    features.get('ay', 0),
                    features.get('hr', 0),
                    features.get('ar', 0)
                ]])
                
                # Scale features
                feature_vector_scaled = self.scaler.transform(feature_vector)
                
                # Make prediction
                prediction_proba = self.model.predict_proba(feature_vector_scaled)[0]
                
                # Convert binary prediction to three-way
                home_prob = prediction_proba[1]  # Home win
                away_prob = prediction_proba[0]  # Away win
                draw_prob = 0.1  # Simple heuristic for draw probability
                
                # Normalize probabilities
                total_prob = home_prob + away_prob + draw_prob
                home_prob /= total_prob
                away_prob /= total_prob
                draw_prob /= total_prob
                
                # Calculate expected goals (simple heuristic)
                expected_home_goals = 1.5 * home_prob + 0.5 * draw_prob
                expected_away_goals = 1.5 * away_prob + 0.5 * draw_prob
                
                # Convert features to JSON-serializable format
                json_features = {}
                for key, value in features.items():
                    if hasattr(value, '__float__'):
                        json_features[key] = float(value)
                    else:
                        json_features[key] = value
                
                # Create prediction record
                prediction = Prediction(
                    model_id=self.model_id,
                    fixture_id=fixture.fixture_id,
                    p_home=home_prob,
                    p_draw=draw_prob,
                    p_away=away_prob,
                    expected_home_goals=expected_home_goals,
                    expected_away_goals=expected_away_goals,
                    feature_snapshot=json_features
                )
                
                self.db.add(prediction)
                predictions_generated += 1
                
            except Exception as e:
                print(f"Error predicting fixture {fixture.fixture_id}: {e}")
                continue
        
        self.db.commit()
        print(f"Generated {predictions_generated} predictions")
        
        # Send completion message
        self.kafka_producer.send_message(
            "predictions.soccer",
            {"action": "predictions_generated", "count": predictions_generated}
        )
    
    def _safe_mean(self, values, default=0.0):
        """Calculate mean with NaN handling."""
        if not values:
            return default
        valid_values = []
        for v in values:
            if v is not None:
                try:
                    # Convert to float and check if it's NaN
                    float_val = float(v)
                    if not np.isnan(float_val):
                        valid_values.append(float_val)
                except (ValueError, TypeError):
                    # Skip non-numeric values
                    continue
        if not valid_values:
            return default
        return np.mean(valid_values)
    
    def _calculate_fixture_features(self, fixture: Fixture) -> Dict[str, Any]:
        """Calculate features for a specific fixture."""
        # Get team names
        home_team = self.db.query(Team).filter(Team.team_id == fixture.home_team_id).first()
        away_team = self.db.query(Team).filter(Team.team_id == fixture.away_team_id).first()
        
        if not home_team or not away_team:
            return {}
        
        # Find historical teams by name matching (more flexible matching)
        home_team_name = home_team.name.replace(' FC', '').replace(' AFC', '').replace(' & Hove Albion', '').replace(' Hotspur', '')
        away_team_name = away_team.name.replace(' FC', '').replace(' AFC', '').replace(' & Hove Albion', '').replace(' Hotspur', '')
        
        # Find teams in historical data that match
        home_historical_team = self.db.query(Team).filter(
            Team.name == home_team_name
        ).first()
        
        away_historical_team = self.db.query(Team).filter(
            Team.name == away_team_name
        ).first()
        
        # If no exact match, try partial matching
        if not home_historical_team:
            home_historical_team = self.db.query(Team).filter(
                Team.name.like(f'%{home_team_name.split()[0]}%')
            ).first()
        
        if not away_historical_team:
            away_historical_team = self.db.query(Team).filter(
                Team.name.like(f'%{away_team_name.split()[0]}%')
            ).first()
        
        if not home_historical_team or not away_historical_team:
            return {}
        
        # Get historical data for teams
        home_matches = self.db.query(MatchClean).filter(
            MatchClean.home_team_id == home_historical_team.team_id,
            MatchClean.home_goals.isnot(None)
        ).order_by(MatchClean.match_date.desc()).limit(10).all()
        
        away_matches = self.db.query(MatchClean).filter(
            MatchClean.away_team_id == away_historical_team.team_id,
            MatchClean.away_goals.isnot(None)
        ).order_by(MatchClean.match_date.desc()).limit(10).all()
        
        # Use default values if no historical data
        if not home_matches and not away_matches:
            return {
                'home_goals': 1.5, 'away_goals': 1.5,
                'home_avg_conceded': 1.2, 'away_avg_conceded': 1.2,
                'home_form': 1.0, 'away_form': 1.0,
                'last5_matchup': 0.5,
                'b365h': 2.5, 'b365d': 3.2, 'b365a': 2.8,
                'b365btsy': 1.8, 'b365btsn': 2.0,
                'hs': 12.0, 'as_': 12.0, 'hc': 5.0, 'ac': 5.0,
                'hy': 2.0, 'ay': 2.0, 'hr': 0.1, 'ar': 0.1
            }
        
        # Calculate basic features with NaN handling
        home_goals = self._safe_mean([m.home_goals for m in home_matches], 1.5)
        away_goals = self._safe_mean([m.away_goals for m in away_matches], 1.5)
        
        home_conceded = self._safe_mean([m.away_goals for m in home_matches], 1.2)
        away_conceded = self._safe_mean([m.home_goals for m in away_matches], 1.2)
        
        # Calculate form (simplified)
        home_form = 0
        away_form = 0
        
        for match in home_matches[:5]:
            if match.result == 'H':
                home_form += 3
            elif match.result == 'D':
                home_form += 1
        
        for match in away_matches[:5]:
            if match.result == 'A':
                away_form += 3
            elif match.result == 'D':
                away_form += 1
        
        home_form = home_form / min(len(home_matches), 5) if home_matches else 1.0
        away_form = away_form / min(len(away_matches), 5) if away_matches else 1.0
        
        # Get average betting odds with NaN handling
        home_odds = self._safe_mean([m.b365h for m in home_matches if m.b365h], 2.5)
        draw_odds = self._safe_mean([m.b365d for m in home_matches if m.b365d], 3.2)
        away_odds = self._safe_mean([m.b365a for m in away_matches if m.b365a], 2.8)
        
        # Get average statistics with NaN handling
        home_stats = {
            'hs': self._safe_mean([m.hs for m in home_matches if m.hs], 12.0),
            'as_': self._safe_mean([m.away_shots for m in home_matches if m.away_shots], 12.0),
            'hc': self._safe_mean([m.hc for m in home_matches if m.hc], 5.0),
            'ac': self._safe_mean([m.ac for m in home_matches if m.ac], 5.0),
            'hy': self._safe_mean([m.hy for m in home_matches if m.hy], 2.0),
            'ay': self._safe_mean([m.ay for m in home_matches if m.ay], 2.0),
            'hr': self._safe_mean([m.hr for m in home_matches if m.hr], 0.1),
            'ar': self._safe_mean([m.ar for m in home_matches if m.ar], 0.1)
        }
        
        away_stats = {
            'hs': self._safe_mean([m.hs for m in away_matches if m.hs], 12.0),
            'as_': self._safe_mean([m.away_shots for m in away_matches if m.away_shots], 12.0),
            'hc': self._safe_mean([m.hc for m in away_matches if m.hc], 5.0),
            'ac': self._safe_mean([m.ac for m in away_matches if m.ac], 5.0),
            'hy': self._safe_mean([m.hy for m in away_matches if m.hy], 2.0),
            'ay': self._safe_mean([m.ay for m in away_matches if m.ay], 2.0),
            'hr': self._safe_mean([m.hr for m in away_matches if m.hr], 0.1),
            'ar': self._safe_mean([m.ar for m in away_matches if m.ar], 0.1)
        }
        
        return {
            'home_goals': home_goals,
            'away_goals': away_goals,
            'home_avg_conceded': home_conceded,
            'away_avg_conceded': away_conceded,
            'home_form': home_form,
            'away_form': away_form,
            'last5_matchup': 0,  # Simplified for now
            'b365h': home_odds,
            'b365d': draw_odds,
            'b365a': away_odds,
            'hs': home_stats['hs'],
            'as_': away_stats['as_'],
            'hc': home_stats['hc'],
            'ac': away_stats['ac'],
            'hy': home_stats['hy'],
            'ay': away_stats['ay'],
            'hr': home_stats['hr'],
            'ar': away_stats['ar']
        }
    
    def handle_kafka_message(self, topic: str, key: str, message: Dict[str, Any]):
        """Handle incoming Kafka messages."""
        print(f"Received message on topic {topic}: {message}")
        
        if message.get('action') == 'features_calculated':
            print("Features calculated, training model...")
            self.train_model()
            print("Model trained, generating predictions...")
            self.predict_fixtures()
    
    def run(self):
        """Main execution method."""
        print("Starting Model Service...")
        
        # Train model and generate predictions
        self.train_model()
        self.predict_fixtures()
        
        # Then listen for new data
        print("Listening for new data...")
        self.kafka_consumer.consume_messages(self.handle_kafka_message)

if __name__ == "__main__":
    service = ModelService()
    service.run()
