"""Data Cleaner Service - Cleans raw data and creates features."""

import os
import sys
from datetime import datetime
from typing import Dict, Any
from sqlalchemy.orm import Session

# Add shared to path
sys.path.append('/app')
from shared.database import get_db, MatchRaw, MatchClean, Team
from shared.kafka_client import KafkaConsumerClient, KafkaProducerClient
from shared.feature_engineering import create_features_for_match

class DataCleaner:
    def __init__(self):
        self.db = next(get_db())
        self.kafka_consumer = KafkaConsumerClient(['fixtures.soccer'], 'data-cleaner')
        self.kafka_producer = KafkaProducerClient()
    
    def clean_and_process_data(self):
        """Clean raw data and create features."""
        print("Starting data cleaning process...")
        
        # Get all raw matches that haven't been processed
        raw_matches = self.db.query(MatchRaw).all()
        
        processed_count = 0
        for raw_match in raw_matches:
            try:
                # Skip if no team names
                if not raw_match.home_team or not raw_match.away_team:
                    continue
                
                # Get team IDs
                home_team = self.db.query(Team).filter(Team.name == raw_match.home_team).first()
                away_team = self.db.query(Team).filter(Team.name == raw_match.away_team).first()
                
                if not home_team or not away_team:
                    print(f"Teams not found: {raw_match.home_team} vs {raw_match.away_team}")
                    continue
                
                # Determine season
                season = self._get_season(raw_match.date)
                
                # Check if already processed
                existing = self.db.query(MatchClean).filter(
                    MatchClean.season == season,
                    MatchClean.match_date == raw_match.date,
                    MatchClean.home_team_id == home_team.team_id,
                    MatchClean.away_team_id == away_team.team_id
                ).first()
                
                if existing:
                    continue
                
                # Create clean match record
                clean_match = MatchClean(
                    season=season,
                    match_date=raw_match.date,
                    home_team_id=home_team.team_id,
                    away_team_id=away_team.team_id,
                    home_goals=raw_match.fthg,
                    away_goals=raw_match.ftag,
                    result=raw_match.ftr,
                    b365h=raw_match.b365h,
                    b365d=raw_match.b365d,
                    b365a=raw_match.b365a,
                    b365btsy=raw_match.b365btsy,
                    b365btsn=raw_match.b365btsn,
                    hs=raw_match.hs,
                    away_shots=raw_match.as_,
                    hc=raw_match.hc,
                    ac=raw_match.ac,
                    hy=raw_match.hy,
                    ay=raw_match.ay,
                    hr=raw_match.hr,
                    ar=raw_match.ar
                )
                
                self.db.add(clean_match)
                processed_count += 1
                
            except Exception as e:
                print(f"Error processing match {raw_match.raw_id}: {e}")
                continue
        
        # Commit all clean matches
        self.db.commit()
        print(f"Processed {processed_count} matches")
        
        # Now calculate features for all matches
        self._calculate_features()
        
        # Send completion message
        self.kafka_producer.send_message(
            "features.soccer",
            {"action": "features_calculated", "count": processed_count}
        )
    
    def _get_season(self, match_date) -> str:
        """Determine season from match date."""
        year = match_date.year
        month = match_date.month
        
        if month >= 8:  # August onwards
            return f"{year}-{str(year + 1)[2:]}"
        else:  # January to July
            return f"{year - 1}-{str(year)[2:]}"
    
    def _calculate_features(self):
        """Calculate features for all clean matches."""
        print("Calculating features...")
        
        # Get all clean matches
        clean_matches = self.db.query(MatchClean).all()
        
        for match in clean_matches:
            try:
                # Calculate features
                features = create_features_for_match(
                    match.home_team_id, 
                    match.away_team_id, 
                    self.db
                )
                
                if features:
                    # Update match with features
                    match.home_avg_conceded = features.get('home_avg_conceded', 0)
                    match.away_avg_conceded = features.get('away_avg_conceded', 0)
                    match.home_form = features.get('home_form', 0)
                    match.away_form = features.get('away_form', 0)
                    match.last5_matchup = features.get('last5_matchup', 0)
                    
                    # Update betting odds if not already set
                    if not match.b365h:
                        match.b365h = features.get('b365h', 0)
                    if not match.b365d:
                        match.b365d = features.get('b365d', 0)
                    if not match.b365a:
                        match.b365a = features.get('b365a', 0)
                    
                    # Update statistics if not already set
                    if not match.hs:
                        match.hs = features.get('hs', 0)
                    if not match.away_shots:
                        match.away_shots = features.get('as_', 0)
                    if not match.hc:
                        match.hc = features.get('hc', 0)
                    if not match.ac:
                        match.ac = features.get('ac', 0)
                    if not match.hy:
                        match.hy = features.get('hy', 0)
                    if not match.ay:
                        match.ay = features.get('ay', 0)
                    if not match.hr:
                        match.hr = features.get('hr', 0)
                    if not match.ar:
                        match.ar = features.get('ar', 0)
                
            except Exception as e:
                print(f"Error calculating features for match {match.match_id}: {e}")
                continue
        
        self.db.commit()
        print("Features calculated successfully")
    
    def handle_kafka_message(self, topic: str, key: str, message: Dict[str, Any]):
        """Handle incoming Kafka messages."""
        print(f"Received message on topic {topic}: {message}")
        
        if message.get('action') == 'fixtures_loaded':
            print("Fixtures loaded, starting data cleaning...")
            self.clean_and_process_data()
    
    def run(self):
        """Main execution method."""
        print("Starting Data Cleaner Service...")
        
        # Process any existing data first
        self.clean_and_process_data()
        
        # Then listen for new data
        print("Listening for new data...")
        self.kafka_consumer.consume_messages(self.handle_kafka_message)

if __name__ == "__main__":
    cleaner = DataCleaner()
    cleaner.run()
