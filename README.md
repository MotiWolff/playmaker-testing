# Playmaker - La Liga Forecast System

A comprehensive microservices-based system for predicting La Liga football match outcomes using historical data and machine learning.

## 🏗️ Architecture

The system consists of 6 microservices:

1. **Data Loader** - Loads CSV data and API fixtures
2. **Data Cleaner** - Cleans data and calculates features
3. **Model Service** - Trains ML models and generates predictions
4. **API Service** - REST API endpoints
5. **UI Service** - Streamlit dashboard
6. **DevOps** - Docker Compose orchestration

## 🚀 Quick Start

### Prerequisites

- Docker and Docker Compose
- Python 3.11+ (for local development)

### Running the System

1. **Clone and setup:**
   ```bash
   git clone <repository>
   cd playmaker-test
   cp env.example .env
   # Edit .env with your API key
   ```

2. **Start all services:**
   ```bash
   docker-compose up --build
   ```

3. **Access the application:**
   - UI Dashboard: http://localhost:8501
   - API Documentation: http://localhost:8000/docs
   - Health Check: http://localhost:8000/healthz

### Services Overview

| Service | Port | Description |
|---------|------|-------------|
| UI Service | 8501 | Streamlit dashboard |
| API Service | 8000 | FastAPI REST endpoints |
| PostgreSQL | 5432 | Database |
| Kafka | 9092 | Message broker |

## 📊 Features

### Data Pipeline
- **CSV Loading**: Automatic loading of football-data.co.uk CSV files
- **API Integration**: Real-time fixture loading from football-data.org
- **Data Cleaning**: Feature engineering and data validation
- **Event Streaming**: Kafka-based inter-service communication

### Machine Learning
- **Random Forest**: Binary classification (Home/Away win)
- **Feature Engineering**: Form, head-to-head, betting odds, statistics
- **Model Versioning**: Track model performance and versions
- **Prediction Storage**: Persistent prediction storage

### API Endpoints
- `GET /healthz` - System health check
- `GET /upcoming?days=7` - Upcoming matches with predictions
- `GET /predictions/{fixture_id}` - Specific match prediction
- `GET /teams` - Available teams
- `GET /models` - Model versions and metrics

### Dashboard Features
- **Match Predictions**: Visual probability displays
- **Odds Formatting**: American odds conversion
- **Expected Goals**: xG analysis
- **System Monitoring**: Health status and model metrics

## 🔧 Development

### Local Development

1. **Database setup:**
   ```bash
   # Start only database and Kafka
   docker-compose up postgres kafka zookeeper
   ```

2. **Run services locally:**
   ```bash
   # Install dependencies
   pip install -r requirements.txt
   
   # Run individual services
   python services/data_loader/main.py
   python services/data_cleaner/main.py
   python services/model_service/main.py
   python services/api_service/main.py
   streamlit run services/ui_service/main.py
   ```

### Adding New Features

1. **New CSV files**: Place in project root, update data_loader
2. **New features**: Modify `shared/feature_engineering.py`
3. **New models**: Extend `services/model_service/main.py`
4. **New endpoints**: Add to `services/api_service/main.py`

## 📈 Monitoring

### Health Checks
- API health: `GET /healthz`
- Database connectivity monitoring
- Service dependency tracking

### Metrics
- Model accuracy and log loss
- Prediction generation counts
- Data processing statistics

## 🗄️ Database Schema

### Core Tables
- `team` - Team information
- `match_raw` - Raw CSV data
- `match_clean` - Processed matches with features
- `fixture` - Upcoming matches
- `model_version` - ML model metadata
- `prediction` - Match predictions

### Key Features
- Automatic team mapping
- Feature calculation and storage
- Prediction versioning
- Comprehensive indexing

## 🔄 Data Flow

1. **Data Ingestion**: CSV files → match_raw table
2. **Data Processing**: Raw data → Clean data + features
3. **Model Training**: Historical data → Trained model
4. **Prediction Generation**: Fixtures + Model → Predictions
5. **API Exposure**: Predictions → REST endpoints
6. **UI Display**: API data → Interactive dashboard

## 🛠️ Troubleshooting

### Common Issues

1. **Database connection errors:**
   - Check PostgreSQL is running
   - Verify DATABASE_URL in .env

2. **Kafka connection errors:**
   - Ensure Kafka and Zookeeper are running
   - Check KAFKA_BOOTSTRAP_SERVERS

3. **API key issues:**
   - Get free API key from football-data.org
   - Update FOOTBALL_API_KEY in .env

4. **No predictions showing:**
   - Check data loader completed successfully
   - Verify model training completed
   - Check fixture data availability

### Logs
```bash
# View service logs
docker-compose logs data-loader
docker-compose logs data-cleaner
docker-compose logs model-service
docker-compose logs api-service
docker-compose logs ui-service
```

## 📝 License

This project is for educational purposes as part of IDF 8200 training.

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Test thoroughly
5. Submit a pull request

## 📞 Support

For issues and questions, please check the troubleshooting section or create an issue in the repository.
