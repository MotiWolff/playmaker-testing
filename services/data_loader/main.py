"""Data Loader Service - Loads CSV data and API fixtures into database."""

import os
import sys
import pandas as pd
import requests
import hashlib
import json
from datetime import datetime
from typing import List, Dict, Any
from sqlalchemy.orm import Session

# Add shared to path
sys.path.append('/app')
from shared.database import get_db, create_tables, Team, MatchRaw, Fixture
from shared.kafka_client import KafkaProducerClient
from shared.elasticsearch_logger import get_logger

class DataLoader:
    def __init__(self):
        self.db = next(get_db())
        self.kafka_producer = KafkaProducerClient()
        self.api_key = os.getenv("FOOTBALL_API_KEY", "")
        self.logger = get_logger("data-loader")
        
    def load_csv_files(self):
        """Load CSV files from football-data.co.uk."""
        csv_files = [
            "/app/data/E0.csv",
            "/app/data/E0_1.csv", 
            "/app/data/SP1.csv",
            "/app/data/SP1_1.csv",
            "/app/data/SP1_2.csv"
        ]
        
        self.logger.info("Starting CSV file loading process")
        
        for csv_file in csv_files:
            if os.path.exists(csv_file):
                self.logger.info(f"Loading CSV file: {csv_file}")
                self._load_single_csv(csv_file)
            else:
                self.logger.warning(f"CSV file not found: {csv_file}")
    
    def _load_single_csv(self, file_path: str):
        """Load a single CSV file."""
        try:
            df = pd.read_csv(file_path)
            
            for _, row in df.iterrows():
                # Create row hash for deduplication
                row_str = str(row.to_dict())
                row_hash = hashlib.sha256(row_str.encode()).hexdigest()
                
                # Check if already exists
                existing = self.db.query(MatchRaw).filter(MatchRaw.row_hash == row_hash).first()
                if existing:
                    continue
                
                # Parse date
                try:
                    match_date = pd.to_datetime(row['Date'], format='%d/%m/%Y').date()
                except:
                    try:
                        match_date = pd.to_datetime(row['Date']).date()
                    except:
                        print(f"Could not parse date: {row['Date']}")
                        continue
                
                # Create match raw record
                match_raw = MatchRaw(
                    src_file=os.path.basename(file_path),
                    row_hash=row_hash,
                    date=match_date,
                    home_team=row.get('HomeTeam'),
                    away_team=row.get('AwayTeam'),
                    fthg=row.get('FTHG'),
                    ftag=row.get('FTAG'),
                    ftr=row.get('FTR'),
                    hs=row.get('HS'),
                    as_=row.get('AS'),
                    hc=row.get('HC'),
                    ac=row.get('AC'),
                    hy=row.get('HY'),
                    ay=row.get('AY'),
                    hr=row.get('HR'),
                    ar=row.get('AR'),
                    b365h=row.get('B365H'),
                    b365d=row.get('B365D'),
                    b365a=row.get('B365A'),
                    b365btsy=row.get('B365BTSY'),
                    b365btsn=row.get('B365BTSN'),
                    extras=json.dumps(row.to_dict())
                )
                
                self.db.add(match_raw)
            
            self.db.commit()
            self.logger.success(f"CSV file loaded successfully: {file_path} - {len(df)} records")
            
        except Exception as e:
            self.logger.error(f"Error loading CSV file: {file_path}", 
                            error=str(e), 
                            file_path=file_path)
            print(f"DEBUG: Error details: {e}")
            print(f"DEBUG: Error type: {type(e)}")
            import traceback
            print(f"DEBUG: Traceback: {traceback.format_exc()}")
            self.db.rollback()
    
    def load_teams(self):
        """Extract and load unique teams from match data and API."""
        self.logger.info("Starting team loading process")
        
        # First load teams from CSV data
        home_teams = self.db.query(MatchRaw.home_team).distinct().all()
        away_teams = self.db.query(MatchRaw.away_team).distinct().all()
        
        all_teams = set()
        for team in home_teams + away_teams:
            if team[0]:  # team[0] is the team name
                all_teams.add(team[0])
        
        # Insert teams from CSV
        new_teams = 0
        for team_name in all_teams:
            existing = self.db.query(Team).filter(Team.name == team_name).first()
            if not existing:
                team = Team(name=team_name)
                self.db.add(team)
                new_teams += 1
        
        # Also load teams from Premier League API
        self.load_premier_league_teams()
        
        self.db.commit()
        self.logger.success(f"Teams loaded successfully: {len(all_teams)} total, {new_teams} new teams")
    
    def load_premier_league_teams(self):
        """Load Premier League teams from API."""
        try:
            url = "https://api.football-data.org/v4/competitions/2021/teams"
            headers = {"X-Auth-Token": self.api_key}
            
            response = requests.get(url, headers=headers)
            response.raise_for_status()
            data = response.json()
            
            if 'teams' not in data:
                return
            
            new_teams = 0
            for team_data in data['teams']:
                team_name = team_data['name']
                existing = self.db.query(Team).filter(Team.name == team_name).first()
                if not existing:
                    team = Team(name=team_name)
                    self.db.add(team)
                    new_teams += 1
            
            print(f"Loaded {new_teams} new Premier League teams from API")
            
        except Exception as e:
            print(f"Error loading Premier League teams: {e}")
    
    def load_fixtures_from_api(self):
        """Load upcoming fixtures from football-data.org API."""
        print("Loading fixtures from API...")
        
        # Premier League competition ID
        competition_id = "2021"  # Premier League
        url = f"https://api.football-data.org/v4/competitions/{competition_id}/matches?status=SCHEDULED"
        headers = {"X-Auth-Token": self.api_key}
        
        try:
            response = requests.get(url, headers=headers)
            response.raise_for_status()
            data = response.json()
            
            if 'matches' not in data:
                print("No matches found in API response")
                return
            
            fixtures_loaded = 0
            for match_data in data['matches']:
                # Get team IDs
                home_team_name = match_data['homeTeam']['name']
                away_team_name = match_data['awayTeam']['name']
                
                home_team = self.db.query(Team).filter(Team.name == home_team_name).first()
                away_team = self.db.query(Team).filter(Team.name == away_team_name).first()
                
                if not home_team or not away_team:
                    print(f"Teams not found: {home_team_name} vs {away_team_name}")
                    continue
                
                # Parse match date
                match_utc = datetime.fromisoformat(match_data['utcDate'].replace('Z', '+00:00'))
                
                # Check if fixture already exists
                existing = self.db.query(Fixture).filter(
                    Fixture.provider_match_id == str(match_data['id'])
                ).first()
                
                if existing:
                    continue
                
                # Create fixture
                fixture = Fixture(
                    provider_match_id=str(match_data['id']),
                    competition_code=competition_id,
                    matchday=match_data.get('matchday'),
                    match_utc=match_utc,
                    home_team_id=home_team.team_id,
                    away_team_id=away_team.team_id,
                    status=match_data['status']
                )
                
                self.db.add(fixture)
                fixtures_loaded += 1
            
            self.db.commit()
            print(f"Loaded {fixtures_loaded} fixtures")
            
            # Send Kafka message
            self.kafka_producer.send_message(
                "fixtures.soccer",
                {"action": "fixtures_loaded", "count": fixtures_loaded}
            )
            
        except Exception as e:
            print(f"Error loading fixtures from API: {e}")
            self.db.rollback()
    
    def run(self):
        """Main execution method."""
        print("Starting Data Loader Service...")
        
        # Create tables
        create_tables()
        
        # Load data
        self.load_csv_files()
        self.load_teams()
        self.load_fixtures_from_api()
        
        print("Data Loader Service completed successfully")
        self.kafka_producer.close()

if __name__ == "__main__":
    loader = DataLoader()
    loader.run()
