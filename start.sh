#!/bin/bash

echo "🚀 Starting Playmaker - La Liga Forecast System"
echo "================================================"

# Check if Docker is running
if ! docker info > /dev/null 2>&1; then
    echo "❌ Docker is not running. Please start Docker and try again."
    exit 1
fi

# Check if Docker Compose is available
if ! command -v docker-compose &> /dev/null; then
    echo "❌ Docker Compose is not installed. Please install Docker Compose and try again."
    exit 1
fi

# Create data directory if it doesn't exist
mkdir -p data

# Copy CSV files to data directory
if [ -f "E0.csv" ]; then
    cp E0.csv data/
    echo "✅ Copied E0.csv to data directory"
fi

if [ -f "E0 (1).csv" ]; then
    cp "E0 (1).csv" data/E0_1.csv
    echo "✅ Copied E0 (1).csv to data directory"
fi

if [ -f "SP1.csv" ]; then
    cp SP1.csv data/
    echo "✅ Copied SP1.csv to data directory"
fi

if [ -f "SP1 (1).csv" ]; then
    cp "SP1 (1).csv" data/SP1_1.csv
    echo "✅ Copied SP1 (1).csv to data directory"
fi

if [ -f "SP1 (2).csv" ]; then
    cp "SP1 (2).csv" data/SP1_2.csv
    echo "✅ Copied SP1 (2).csv to data directory"
fi

# Start services
echo "🐳 Starting Docker services..."
docker-compose up --build -d

# Wait for services to be ready
echo "⏳ Waiting for services to start..."
sleep 30

# Check service health
echo "🔍 Checking service health..."

# Check ElasticSearch health
if curl -s http://localhost:9200/_cluster/health > /dev/null; then
    echo "✅ ElasticSearch is healthy"
else
    echo "❌ ElasticSearch is not responding"
fi

# Check Kibana
if curl -s http://localhost:5601 > /dev/null; then
    echo "✅ Kibana is healthy"
else
    echo "❌ Kibana is not responding"
fi

# Check API health
if curl -s http://localhost:8000/healthz > /dev/null; then
    echo "✅ API Service is healthy"
else
    echo "❌ API Service is not responding"
fi

# Check UI
if curl -s http://localhost:8501 > /dev/null; then
    echo "✅ UI Service is healthy"
else
    echo "❌ UI Service is not responding"
fi

echo ""
echo "🎉 Playmaker is starting up!"
echo ""
echo "📊 Access the application:"
echo "   • UI Dashboard: http://localhost:8501"
echo "   • API Documentation: http://localhost:8000/docs"
echo "   • Health Check: http://localhost:8000/healthz"
echo "   • Kibana Logs: http://localhost:5601"
echo "   • ElasticSearch: http://localhost:9200"
echo ""
echo "📝 View logs:"
echo "   docker-compose logs -f"
echo ""
echo "🛑 Stop the system:"
echo "   docker-compose down"
echo ""
echo "The system will take a few minutes to fully initialize and process data."
echo "Check the logs for progress updates."
