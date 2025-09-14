"""API Service - FastAPI REST endpoints for Playmaker."""

import os
import sys
from datetime import datetime, timedelta
from typing import List, Optional
from fastapi import FastAPI, Depends, HTTPException
from sqlalchemy.orm import Session
from pydantic import BaseModel

# Add shared to path
sys.path.append('/app')
from shared.database import get_db, Fixture, Prediction, ModelVersion, Team
from shared.elasticsearch_logger import get_logger

app = FastAPI(
    title="Playmaker API",
    description="La Liga prediction API",
    version="1.0.0"
)

# Initialize logger
logger = get_logger("api-service")

# Pydantic models
class PredictionResponse(BaseModel):
    fixture_id: int
    home_team: str
    away_team: str
    match_date: datetime
    p_home: float
    p_draw: float
    p_away: float
    expected_home_goals: float
    expected_away_goals: float
    model_name: str
    generated_at: datetime

class UpcomingMatchResponse(BaseModel):
    fixture_id: int
    home_team: str
    away_team: str
    match_date: datetime
    matchday: Optional[int]
    prediction: Optional[PredictionResponse]

class HealthResponse(BaseModel):
    status: str
    timestamp: datetime
    database_connected: bool

@app.get("/healthz", response_model=HealthResponse)
async def health_check(db: Session = Depends(get_db)):
    """Health check endpoint."""
    try:
        # Test database connection
        from sqlalchemy import text
        db.execute(text("SELECT 1"))
        database_connected = True
    except Exception as e:
        database_connected = False
        logger.error(f"Database connection failed: {e}")
    
    return HealthResponse(
        status="healthy" if database_connected else "unhealthy",
        timestamp=datetime.utcnow(),
        database_connected=database_connected
    )

@app.get("/upcoming", response_model=List[UpcomingMatchResponse])
async def get_upcoming_matches(
    days: int = 7,
    db: Session = Depends(get_db)
):
    """Get upcoming matches with predictions."""
    # Calculate date range
    start_date = datetime.utcnow()
    end_date = start_date + timedelta(days=days)
    
    # Query upcoming fixtures
    fixtures = db.query(Fixture).filter(
        Fixture.status.in_(['SCHEDULED', 'TIMED']),
        Fixture.match_utc >= start_date,
        Fixture.match_utc <= end_date
    ).all()
    
    results = []
    
    for fixture in fixtures:
        # Get team names
        home_team = db.query(Team).filter(Team.team_id == fixture.home_team_id).first()
        away_team = db.query(Team).filter(Team.team_id == fixture.away_team_id).first()
        
        if not home_team or not away_team:
            continue
        
        # Get latest prediction
        prediction = db.query(Prediction).filter(
            Prediction.fixture_id == fixture.fixture_id
        ).order_by(Prediction.generated_at.desc()).first()
        
        prediction_response = None
        if prediction:
            model = db.query(ModelVersion).filter(
                ModelVersion.model_id == prediction.model_id
            ).first()
            
            prediction_response = PredictionResponse(
                fixture_id=fixture.fixture_id,
                home_team=home_team.name,
                away_team=away_team.name,
                match_date=fixture.match_utc,
                p_home=float(prediction.p_home),
                p_draw=float(prediction.p_draw),
                p_away=float(prediction.p_away),
                expected_home_goals=float(prediction.expected_home_goals),
                expected_away_goals=float(prediction.expected_away_goals),
                model_name=model.model_name if model else "unknown",
                generated_at=prediction.generated_at
            )
        
        results.append(UpcomingMatchResponse(
            fixture_id=fixture.fixture_id,
            home_team=home_team.name,
            away_team=away_team.name,
            match_date=fixture.match_utc,
            matchday=fixture.matchday,
            prediction=prediction_response
        ))
    
    return results

@app.get("/predictions/{fixture_id}", response_model=PredictionResponse)
async def get_prediction(fixture_id: int, db: Session = Depends(get_db)):
    """Get prediction for a specific fixture."""
    # Get fixture
    fixture = db.query(Fixture).filter(Fixture.fixture_id == fixture_id).first()
    if not fixture:
        raise HTTPException(status_code=404, detail="Fixture not found")
    
    # Get team names
    home_team = db.query(Team).filter(Team.team_id == fixture.home_team_id).first()
    away_team = db.query(Team).filter(Team.team_id == fixture.away_team_id).first()
    
    if not home_team or not away_team:
        raise HTTPException(status_code=404, detail="Teams not found")
    
    # Get latest prediction
    prediction = db.query(Prediction).filter(
        Prediction.fixture_id == fixture_id
    ).order_by(Prediction.generated_at.desc()).first()
    
    if not prediction:
        raise HTTPException(status_code=404, detail="Prediction not found")
    
    # Get model info
    model = db.query(ModelVersion).filter(
        ModelVersion.model_id == prediction.model_id
    ).first()
    
    return PredictionResponse(
        fixture_id=fixture.fixture_id,
        home_team=home_team.name,
        away_team=away_team.name,
        match_date=fixture.match_utc,
        p_home=float(prediction.p_home),
        p_draw=float(prediction.p_draw),
        p_away=float(prediction.p_away),
        expected_home_goals=float(prediction.expected_home_goals),
        expected_away_goals=float(prediction.expected_away_goals),
        model_name=model.model_name if model else "unknown",
        generated_at=prediction.generated_at
    )

@app.get("/teams")
async def get_teams(db: Session = Depends(get_db)):
    """Get all teams."""
    teams = db.query(Team).all()
    return [{"team_id": team.team_id, "name": team.name} for team in teams]

@app.get("/models")
async def get_models(db: Session = Depends(get_db)):
    """Get all model versions."""
    models = db.query(ModelVersion).order_by(ModelVersion.created_at.desc()).all()
    return [
        {
            "model_id": model.model_id,
            "model_name": model.model_name,
            "trained_on_dset": model.trained_on_dset,
            "metrics": model.metrics,
            "created_at": model.created_at
        }
        for model in models
    ]

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
