"""UI Service - Streamlit dashboard for Playmaker."""

import streamlit as st
import requests
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from datetime import datetime, timedelta
import os
from PIL import Image
import io

# Configuration
API_BASE_URL = os.getenv("API_BASE_URL", "http://api-service:8000")

def fetch_data(endpoint: str):
    """Fetch data from API."""
    try:
        response = requests.get(f"{API_BASE_URL}{endpoint}")
        response.raise_for_status()
        return response.json()
    except Exception as e:
        st.error(f"Error fetching data: {e}")
        return None

def format_odds(prob: float) -> str:
    """Format probability as odds."""
    if prob == 0:
        return "N/A"
    odds = 1 / prob
    if odds >= 2:
        return f"+{int((odds-1)*100)}"
    else:
        return f"-{int(100/(odds-1))}"

def format_percentage(prob: float) -> str:
    """Format probability as percentage."""
    return f"{prob*100:.1f}%"

def get_team_logo(team_name: str) -> str:
    """Get team logo URL or emoji based on team name."""
    # Simple mapping of team names to emojis/logos
    team_logos = {
        "Real Madrid": "👑",
        "Barcelona": "🔵",
        "Ath Madrid": "🔴",
        "Sevilla": "⚪",
        "Valencia": "🦇",
        "Villarreal": "🟡",
        "Betis": "🟢",
        "Sociedad": "🔵",
        "Ath Bilbao": "🔴",
        "Girona": "🔴",
        "Mallorca": "🔴",
        "Osasuna": "🔴",
        "Celta": "🔵",
        "Getafe": "🔵",
        "Las Palmas": "🟡",
        "Alaves": "🔵",
        "Vallecano": "🔴",
        "Cadiz": "🟡",
        "Almeria": "🔴",
        "Elche": "🟢",
        "Valladolid": "🟣",
        "Leganes": "🔵",
        "Espanol": "🔵",
        "Arsenal": "🔴",
        "Chelsea": "🔵",
        "Liverpool": "🔴",
        "Manchester City": "🔵",
        "Manchester United": "🔴",
        "Tottenham": "🔵",
        "Newcastle": "⚫",
        "Brighton": "🔵",
        "West Ham": "🔴",
        "Crystal Palace": "🔵",
        "Everton": "🔵",
        "Aston Villa": "🔵",
        "Brentford": "🔴",
        "Fulham": "⚪",
        "Wolves": "🟡",
        "Burnley": "🔴",
        "Leeds United": "⚪",
        "Nottingham Forest": "🔴",
        "Sheffield United": "🔴",
        "Southampton": "🔴",
        "Leicester": "🔵",
        "Bournemouth": "🔴",
        "Ipswich": "🔵",
        "Sunderland": "🔴"
    }
    
    # Try to find exact match first
    if team_name in team_logos:
        return team_logos[team_name]
    
    # Try to find partial match
    for team, logo in team_logos.items():
        if team.lower() in team_name.lower() or team_name.lower() in team.lower():
            return logo
    
    # Default football emoji
    return "⚽"

def display_match_card(match):
    """Display a single match in the requested format."""
    pred = match.get("prediction")
    match_date = pd.to_datetime(match["match_date"])
    
    # Create match card container
    with st.container():
        st.markdown("---")
        
        # Main match info row
        col1, col2, col3 = st.columns([2, 1, 2])
        
        with col1:
            home_logo = get_team_logo(match["home_team"])
            st.markdown(f"### {home_logo} {match['home_team']}")
        
        with col2:
            st.markdown(f"### {match_date.strftime('%d/%m')}")
            st.markdown(f"**{match_date.strftime('%H:%M')}**")
            if match.get("matchday"):
                st.markdown(f"*Matchday {match['matchday']}*")
        
        with col3:
            away_logo = get_team_logo(match["away_team"])
            st.markdown(f"### {match['away_team']} {away_logo}")
        
        # Predictions section
        if pred:
            st.markdown("#### 📊 Predictions")
            
            # Prediction metrics
            col1, col2, col3, col4, col5 = st.columns(5)
            
            with col1:
                st.metric(
                    "Home Win",
                    format_percentage(pred["p_home"]),
                    format_odds(pred["p_home"])
                )
            
            with col2:
                st.metric(
                    "Draw",
                    format_percentage(pred["p_draw"]),
                    format_odds(pred["p_draw"])
                )
            
            with col3:
                st.metric(
                    "Away Win",
                    format_percentage(pred["p_away"]),
                    format_odds(pred["p_away"])
                )
            
            with col4:
                st.metric("Expected Home Goals", f"{pred['expected_home_goals']:.2f}")
            
            with col5:
                st.metric("Expected Away Goals", f"{pred['expected_away_goals']:.2f}")
            
            # Prediction chart
            fig = go.Figure(data=[
                go.Bar(
                    x=["Home Win", "Draw", "Away Win"],
                    y=[pred["p_home"], pred["p_draw"], pred["p_away"]],
                    marker_color=["#1f77b4", "#ff7f0e", "#2ca02c"],
                    text=[format_percentage(pred["p_home"]), format_percentage(pred["p_draw"]), format_percentage(pred["p_away"])],
                    textposition='auto'
                )
            ])
            fig.update_layout(
                title=f"Prediction Probabilities",
                yaxis_title="Probability",
                showlegend=False,
                height=300
            )
            st.plotly_chart(fig, use_container_width=True)
            
            # Model info
            st.caption(f"Model: {pred['model_name']} | Generated: {pred['generated_at']}")
        else:
            st.info("No prediction available for this match")
        
        st.markdown("")

def main():
    st.set_page_config(
        page_title="Playmaker - La Liga Predictions",
        page_icon="⚽",
        layout="wide"
    )
    
    st.title("⚽ Playmaker - La Liga Predictions")
    st.markdown("---")
    
    # Sidebar
    st.sidebar.title("Settings")
    days_ahead = st.sidebar.slider("Days ahead to show", 1, 30, 7)
    
    # Health check
    with st.spinner("Checking system health..."):
        health_data = fetch_data("/healthz")
    
    if health_data:
        if health_data["database_connected"]:
            st.sidebar.success("✅ System Healthy")
        else:
            st.sidebar.error("❌ Database Connection Issue")
    else:
        st.sidebar.error("❌ API Unavailable")
    
    # Main content
    tab1, tab2, tab3 = st.tabs(["📅 Upcoming Matches", "📊 Predictions", "ℹ️ System Info"])
    
    with tab1:
        st.header("Upcoming Matches")
        
        # Fetch upcoming matches
        upcoming_data = fetch_data(f"/upcoming?days={days_ahead}")
        
        if upcoming_data is not None:
            if not upcoming_data:
                st.info("No upcoming matches found for the selected period.")
            else:
                # Display matches in the requested format
                st.markdown(f"**Found {len(upcoming_data)} upcoming matches**")
                
                for match in upcoming_data:
                    display_match_card(match)
        else:
            st.error("Failed to load upcoming matches")
    
    with tab2:
        st.header("Prediction Analysis")
        
        # Fetch upcoming matches for analysis
        upcoming_data = fetch_data(f"/upcoming?days={days_ahead}")
        
        if upcoming_data is not None:
            predictions = [match["prediction"] for match in upcoming_data if match["prediction"]]
            
            if predictions:
                # Create prediction analysis
                pred_df = pd.DataFrame([
                    {
                        "Match": f"{match['home_team']} vs {match['away_team']}",
                        "Home Win %": pred["p_home"] * 100,
                        "Draw %": pred["p_draw"] * 100,
                        "Away Win %": pred["p_away"] * 100,
                        "Expected Home Goals": pred["expected_home_goals"],
                        "Expected Away Goals": pred["expected_away_goals"]
                    }
                    for match, pred in zip(upcoming_data, predictions) if pred
                ])
                
                # Summary statistics
                col1, col2, col3 = st.columns(3)
                
                with col1:
                    st.metric("Total Predictions", len(predictions))
                
                with col2:
                    avg_home_goals = pred_df["Expected Home Goals"].mean()
                    st.metric("Avg Expected Home Goals", f"{avg_home_goals:.2f}")
                
                with col3:
                    avg_away_goals = pred_df["Expected Away Goals"].mean()
                    st.metric("Avg Expected Away Goals", f"{avg_away_goals:.2f}")
                
                # Distribution of predictions
                fig = px.histogram(
                    pred_df,
                    x=["Home Win %", "Draw %", "Away Win %"],
                    title="Distribution of Prediction Probabilities",
                    labels={"value": "Probability (%)", "variable": "Outcome"}
                )
                st.plotly_chart(fig, use_container_width=True)
                
                # Expected goals scatter plot
                fig = px.scatter(
                    pred_df,
                    x="Expected Home Goals",
                    y="Expected Away Goals",
                    hover_data=["Match"],
                    title="Expected Goals Scatter Plot"
                )
                st.plotly_chart(fig, use_container_width=True)
                
                # Detailed predictions table
                st.subheader("Detailed Predictions Table")
                st.dataframe(pred_df, use_container_width=True)
            else:
                st.info("No predictions available for the selected period.")
        else:
            st.error("Failed to load prediction data")
    
    with tab3:
        st.header("System Information")
        
        # Health status
        if health_data:
            col1, col2 = st.columns(2)
            
            with col1:
                st.metric("API Status", health_data["status"].title())
            
            with col2:
                st.metric("Database", "Connected" if health_data["database_connected"] else "Disconnected")
        
        # Model information
        models_data = fetch_data("/models")
        if models_data:
            st.subheader("Available Models")
            
            if models_data:
                models_df = pd.DataFrame(models_data)
                st.dataframe(models_df, use_container_width=True)
                
                # Model metrics visualization
                if len(models_data) > 0:
                    latest_model = models_data[0]
                    if latest_model.get("metrics"):
                        metrics = latest_model["metrics"]
                        
                        col1, col2, col3, col4 = st.columns(4)
                        
                        with col1:
                            st.metric("Accuracy", f"{metrics.get('accuracy', 0):.3f}")
                        
                        with col2:
                            st.metric("Log Loss", f"{metrics.get('log_loss', 0):.3f}")
                        
                        with col3:
                            st.metric("Training Samples", metrics.get('training_samples', 0))
                        
                        with col4:
                            st.metric("Test Samples", metrics.get('test_samples', 0))
            else:
                st.info("No models available")
        else:
            st.error("Failed to load model information")
        
        # Teams information
        teams_data = fetch_data("/teams")
        if teams_data:
            st.subheader("Available Teams")
            teams_df = pd.DataFrame(teams_data)
            st.dataframe(teams_df, use_container_width=True)
        else:
            st.error("Failed to load teams information")

if __name__ == "__main__":
    main()
