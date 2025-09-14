# Playmaker Logging with ElasticSearch

## 📊 **Overview**

The Playmaker system now includes comprehensive logging using ElasticSearch and Kibana for centralized log management and monitoring.

## 🏗️ **Architecture**

```
Services → ElasticSearch Logger → ElasticSearch → Kibana Dashboard
```

## 🔧 **Components**

### **ElasticSearch**
- **Port**: 9200
- **URL**: http://localhost:9200
- **Purpose**: Centralized log storage and search

### **Kibana**
- **Port**: 5601
- **URL**: http://localhost:5601
- **Purpose**: Log visualization and dashboard

### **ElasticSearch Logger**
- **Location**: `shared/elasticsearch_logger.py`
- **Purpose**: Structured logging utility for all services

## 📝 **Log Types**

### **1. Application Logs**
- **Index Pattern**: `playmaker-{service}-logs-{date}`
- **Content**: Standard application logs (INFO, ERROR, WARNING, DEBUG)

### **2. Metrics**
- **Index Pattern**: `playmaker-{service}-metrics-{date}`
- **Content**: Performance metrics, model accuracy, processing counts

### **3. Predictions**
- **Index Pattern**: `playmaker-{service}-predictions-{date}`
- **Content**: Match predictions with probabilities and team information

### **4. Data Processing**
- **Index Pattern**: `playmaker-{service}-data_processing-{date}`
- **Content**: CSV loading, data cleaning, feature calculation events

### **5. Model Training**
- **Index Pattern**: `playmaker-{service}-model_training-{date}`
- **Content**: Model training results, accuracy metrics, feature importance

### **6. API Requests**
- **Index Pattern**: `playmaker-{service}-api_requests-{date}`
- **Content**: API endpoint calls, response times, status codes

### **7. Kafka Messages**
- **Index Pattern**: `playmaker-{service}-kafka_messages-{date}`
- **Content**: Kafka message processing events

## 🚀 **Usage**

### **In Your Service Code**

```python
from shared.elasticsearch_logger import get_logger

# Initialize logger
logger = get_logger("your-service-name")

# Basic logging
logger.info("Service started")
logger.error("Something went wrong", error_code=500)
logger.warning("Low disk space", free_space="10GB")

# Structured logging
logger.log_metric("accuracy", 0.85, model_version="v1.2")
logger.log_prediction(123, "Arsenal", "Chelsea", "Home Win", 
                     {"home": 0.6, "draw": 0.2, "away": 0.2})
logger.log_data_processing("csv_loaded", 1000, file_name="E0.csv")
logger.log_model_training("rf_v1", 0.85, training_samples=5000)
logger.log_api_request("/predictions/123", "GET", 200, 150.5)
logger.log_kafka_message("fixtures.soccer", "fixtures_loaded")
```

## 📊 **Kibana Dashboards**

### **Access Kibana**
1. Open http://localhost:5601
2. Go to **Discover** to view logs
3. Use **Dashboard** for visualizations
4. Create **Index Patterns** for different log types

### **Useful Queries**

```json
# All logs from data-loader service
service: "data-loader"

# Error logs across all services
level: "ERROR"

# Model training events
log_type: "model_training"

# API requests with high response time
log_type: "api_requests" AND response_time_ms: >1000

# Predictions for specific teams
log_type: "predictions" AND home_team: "Arsenal"
```

## 🔍 **Monitoring**

### **Key Metrics to Monitor**
- **Error Rate**: `level: "ERROR"`
- **Response Times**: `response_time_ms` field
- **Data Processing**: `record_count` field
- **Model Performance**: `accuracy` field
- **Prediction Volume**: `log_type: "predictions"`

### **Alerts Setup**
1. Go to **Stack Management** → **Watcher**
2. Create alerts for:
   - High error rates
   - Slow API responses
   - Model accuracy drops
   - Service failures

## 🛠️ **Configuration**

### **Environment Variables**
```bash
ELASTICSEARCH_HOST=elasticsearch:9200
```

### **Index Management**
- **Retention**: 30 days (configurable)
- **Shards**: 1 per index
- **Replicas**: 0 (single node setup)

## 📈 **Performance**

### **Log Volume**
- **Estimated**: ~1000 logs/hour during active processing
- **Storage**: ~1GB/month for full system
- **Memory**: 512MB allocated to ElasticSearch

### **Optimization**
- Use structured logging (JSON format)
- Avoid logging sensitive data
- Use appropriate log levels
- Batch log sends when possible

## 🚨 **Troubleshooting**

### **Common Issues**

1. **ElasticSearch Connection Failed**
   ```bash
   # Check if ElasticSearch is running
   curl http://localhost:9200/_cluster/health
   
   # Check logs
   docker-compose logs elasticsearch
   ```

2. **No Logs in Kibana**
   ```bash
   # Check index patterns
   curl http://localhost:9200/_cat/indices
   
   # Refresh index patterns in Kibana
   ```

3. **High Memory Usage**
   ```bash
   # Reduce ElasticSearch memory
   # Edit docker-compose.yml
   "ES_JAVA_OPTS=-Xms256m -Xmx256m"
   ```

### **Log Analysis Commands**

```bash
# View all logs
docker-compose logs -f

# View specific service logs
docker-compose logs -f data-loader
docker-compose logs -f model-service

# Check ElasticSearch health
curl http://localhost:9200/_cluster/health?pretty

# List all indices
curl http://localhost:9200/_cat/indices?v
```

## 📚 **Best Practices**

1. **Use Structured Logging**: Always include relevant context
2. **Log Levels**: Use appropriate levels (DEBUG, INFO, WARNING, ERROR)
3. **Performance**: Don't log in tight loops
4. **Security**: Never log passwords or sensitive data
5. **Consistency**: Use consistent field names across services
6. **Monitoring**: Set up alerts for critical errors
7. **Retention**: Configure appropriate log retention policies

## 🔗 **Useful Links**

- **ElasticSearch Docs**: https://www.elastic.co/guide/en/elasticsearch/reference/current/
- **Kibana Docs**: https://www.elastic.co/guide/en/kibana/current/
- **Python ElasticSearch Client**: https://elasticsearch-py.readthedocs.io/
