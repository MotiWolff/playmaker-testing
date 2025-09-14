"""Database models and connection utilities for Playmaker services."""

import os
from sqlalchemy import create_engine, Column, Integer, String, Date, DateTime, Text, Numeric, Boolean, ForeignKey, BigInteger, CHAR, REAL, JSON
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import sessionmaker, relationship
from sqlalchemy.dialects.postgresql import JSONB
from sqlalchemy import TIMESTAMP
from datetime import datetime

# Database connection
DATABASE_URL = os.getenv("DATABASE_URL", "postgresql://playmaker:playmaker123@localhost:5432/playmaker")
engine = create_engine(DATABASE_URL)
SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)
Base = declarative_base()

# Database models
class Team(Base):
    __tablename__ = "team"
    
    team_id = Column(Integer, primary_key=True, index=True)
    name = Column(Text, nullable=False, unique=True)

class MatchRaw(Base):
    __tablename__ = "match_raw"
    
    raw_id = Column(BigInteger, primary_key=True, index=True)
    src_file = Column(Text, nullable=False)
    row_hash = Column(Text, nullable=False, unique=True)
    
    # CSV columns - using quoted names to match database schema
    date = Column("Date", Date)
    home_team = Column("HomeTeam", Text)
    away_team = Column("AwayTeam", Text)
    fthg = Column("FTHG", Integer)  # Full Time Home Goals
    ftag = Column("FTAG", Integer)  # Full Time Away Goals
    ftr = Column("FTR", CHAR(1))   # Full Time Result
    hs = Column("HS", Integer)    # Home Shots
    as_ = Column("AS", Integer)   # Away Shots
    hc = Column("HC", Integer)    # Home Corners
    ac = Column("AC", Integer)    # Away Corners
    hy = Column("HY", Integer)    # Home Yellows
    ay = Column("AY", Integer)    # Away Yellows
    hr = Column("HR", Integer)    # Home Reds
    ar = Column("AR", Integer)    # Away Reds
    
    # Betting odds
    b365h = Column("B365H", Numeric(8, 3))      # Home win odds
    b365d = Column("B365D", Numeric(8, 3))      # Draw odds
    b365a = Column("B365A", Numeric(8, 3))      # Away win odds
    b365btsy = Column("B365BTSY", Numeric(8, 3))   # Both teams to score - Yes
    b365btsn = Column("B365BTSN", Numeric(8, 3))   # Both teams to score - No
    
    extras = Column(JSONB, default={})
    ingested_at = Column(TIMESTAMP, default=datetime.utcnow)

class MatchClean(Base):
    __tablename__ = "match_clean"
    
    match_id = Column(BigInteger, primary_key=True, index=True)
    season = Column(Text, nullable=False)
    match_date = Column(Date, nullable=False)
    
    home_team_id = Column(Integer, ForeignKey("team.team_id"), nullable=False)
    away_team_id = Column(Integer, ForeignKey("team.team_id"), nullable=False)
    
    home_goals = Column(Integer)
    away_goals = Column(Integer)
    result = Column(CHAR(1))
    
    # Features
    home_avg_conceded = Column(REAL)
    away_avg_conceded = Column(REAL)
    home_form = Column(REAL)
    away_form = Column(REAL)
    last5_matchup = Column(REAL)
    
    # Betting odds and stats
    b365h = Column(Numeric(8, 3))
    b365d = Column(Numeric(8, 3))
    b365a = Column(Numeric(8, 3))
    b365btsy = Column(Numeric(8, 3))
    b365btsn = Column(Numeric(8, 3))
    hs = Column(Integer)
    away_shots = Column(Integer)
    hc = Column(Integer)
    ac = Column(Integer)
    hy = Column(Integer)
    ay = Column(Integer)
    hr = Column(Integer)
    ar = Column(Integer)
    
    created_at = Column(TIMESTAMP, default=datetime.utcnow)
    
    # Relationships
    home_team = relationship("Team", foreign_keys=[home_team_id])
    away_team = relationship("Team", foreign_keys=[away_team_id])

class Fixture(Base):
    __tablename__ = "fixture"
    
    fixture_id = Column(BigInteger, primary_key=True, index=True)
    provider_match_id = Column(Text, unique=True)
    competition_code = Column(Text, default='PD')
    matchday = Column(Integer)
    match_utc = Column(TIMESTAMP)
    home_team_id = Column(Integer, ForeignKey("team.team_id"))
    away_team_id = Column(Integer, ForeignKey("team.team_id"))
    status = Column(Text)
    inserted_at = Column(TIMESTAMP, default=datetime.utcnow)
    
    # Relationships
    home_team = relationship("Team", foreign_keys=[home_team_id])
    away_team = relationship("Team", foreign_keys=[away_team_id])

class ModelVersion(Base):
    __tablename__ = "model_version"
    
    model_id = Column(Integer, primary_key=True, index=True)
    model_name = Column(Text, nullable=False)
    trained_on_dset = Column(Text)
    metrics = Column(JSONB)
    artifact_uri = Column(Text)
    created_at = Column(TIMESTAMP, default=datetime.utcnow)

class Prediction(Base):
    __tablename__ = "prediction"
    
    pred_id = Column(BigInteger, primary_key=True, index=True)
    model_id = Column(Integer, ForeignKey("model_version.model_id"), nullable=False)
    fixture_id = Column(BigInteger, ForeignKey("fixture.fixture_id"), nullable=False)
    
    p_home = Column(REAL)
    p_draw = Column(REAL)
    p_away = Column(REAL)
    expected_home_goals = Column(REAL)
    expected_away_goals = Column(REAL)
    feature_snapshot = Column(JSONB, default={})
    generated_at = Column(TIMESTAMP, default=datetime.utcnow)
    
    # Relationships
    model = relationship("ModelVersion")
    fixture = relationship("Fixture")

# Database functions
def get_db():
    """Get database session."""
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()

def create_tables():
    """Create all database tables."""
    Base.metadata.create_all(bind=engine)
