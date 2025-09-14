"""ElasticSearch logging utility for Playmaker services."""

import os
import json
import logging
from datetime import datetime
from typing import Dict, Any, Optional
from elasticsearch import Elasticsearch
from elasticsearch.exceptions import ConnectionError, RequestError

class ElasticSearchLogger:
    """ElasticSearch logger for structured logging."""
    
    def __init__(self, service_name: str, index_prefix: str = "playmaker"):
        self.service_name = service_name
        self.index_prefix = index_prefix
        self.es_host = os.getenv("ELASTICSEARCH_HOST", "localhost:9200")
        self.es_client = None
        self._setup_elasticsearch()
        self._setup_logger()
    
    def _setup_elasticsearch(self):
        """Initialize ElasticSearch client."""
        try:
            # Ensure the host has the proper format
            if not self.es_host.startswith(('http://', 'https://')):
                self.es_host = f"http://{self.es_host}"
            
            self.es_client = Elasticsearch([self.es_host])
            # Test connection
            if self.es_client.ping():
                print(f"✅ Connected to ElasticSearch at {self.es_host}")
            else:
                print(f"❌ Failed to connect to ElasticSearch at {self.es_host}")
                self.es_client = None
        except Exception as e:
            print(f"❌ ElasticSearch connection error: {e}")
            self.es_client = None
    
    def _setup_logger(self):
        """Setup Python logger."""
        self.logger = logging.getLogger(f"playmaker.{self.service_name}")
        self.logger.setLevel(logging.INFO)
        
        # Create console handler
        console_handler = logging.StreamHandler()
        console_handler.setLevel(logging.INFO)
        
        # Create formatter
        formatter = logging.Formatter(
            '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
        )
        console_handler.setFormatter(formatter)
        
        # Add handler to logger
        if not self.logger.handlers:
            self.logger.addHandler(console_handler)
    
    def _create_index_name(self, log_type: str) -> str:
        """Create index name with date suffix."""
        today = datetime.now().strftime("%Y.%m.%d")
        return f"{self.index_prefix}-{self.service_name}-{log_type}-{today}"
    
    def _send_to_elasticsearch(self, log_data: Dict[str, Any], log_type: str = "logs"):
        """Send log data to ElasticSearch."""
        if not self.es_client:
            return
        
        try:
            index_name = self._create_index_name(log_type)
            
            # Add metadata
            log_data.update({
                "@timestamp": datetime.utcnow().isoformat(),
                "service": self.service_name,
                "log_type": log_type
            })
            
            # Send to ElasticSearch
            self.es_client.index(
                index=index_name,
                body=log_data
            )
            
        except (ConnectionError, RequestError) as e:
            print(f"❌ ElasticSearch error: {e}")
        except Exception as e:
            print(f"❌ Unexpected error sending to ElasticSearch: {e}")
    
    def info(self, message: str, **kwargs):
        """Log info message (console only, not sent to ElasticSearch)."""
        self.logger.info(message)
        # Don't send INFO messages to ElasticSearch
    
    def error(self, message: str, **kwargs):
        """Log error message."""
        log_data = {
            "level": "ERROR",
            "message": message,
            **kwargs
        }
        
        self.logger.error(message)
        self._send_to_elasticsearch(log_data)
    
    def success(self, message: str, **kwargs):
        """Log success message."""
        log_data = {
            "level": "SUCCESS",
            "message": message,
            **kwargs
        }
        
        self.logger.info(f"✅ {message}")
        self._send_to_elasticsearch(log_data)
    
    def warning(self, message: str, **kwargs):
        """Log warning message (console only, not sent to ElasticSearch)."""
        self.logger.warning(message)
        # Don't send WARNING messages to ElasticSearch
    
    def debug(self, message: str, **kwargs):
        """Log debug message (console only, not sent to ElasticSearch)."""
        self.logger.debug(message)
        # Don't send DEBUG messages to ElasticSearch
    
    def log_metric(self, metric_name: str, value: float, **kwargs):
        """Log a metric."""
        log_data = {
            "level": "METRIC",
            "message": f"Metric: {metric_name} = {value}",
            "metric_name": metric_name,
            "metric_value": value,
            **kwargs
        }
        
        self.logger.info(f"Metric: {metric_name} = {value}")
        self._send_to_elasticsearch(log_data, "metrics")
    
    def log_prediction(self, fixture_id: int, home_team: str, away_team: str, 
                      prediction: str, probabilities: Dict[str, float], **kwargs):
        """Log a prediction."""
        log_data = {
            "level": "PREDICTION",
            "message": f"Prediction: {home_team} vs {away_team} -> {prediction}",
            "fixture_id": fixture_id,
            "home_team": home_team,
            "away_team": away_team,
            "prediction": prediction,
            "probabilities": probabilities,
            **kwargs
        }
        
        self.logger.info(f"Prediction: {home_team} vs {away_team} -> {prediction}")
        self._send_to_elasticsearch(log_data, "predictions")
    
    def log_data_processing(self, operation: str, count: int, **kwargs):
        """Log data processing operation."""
        log_data = {
            "level": "DATA_PROCESSING",
            "message": f"Data processing: {operation} - {count} records",
            "operation": operation,
            "record_count": count,
            **kwargs
        }
        
        self.logger.info(f"Data processing: {operation} - {count} records")
        self._send_to_elasticsearch(log_data, "data_processing")
    
    def log_model_training(self, model_name: str, accuracy: float, **kwargs):
        """Log model training results."""
        log_data = {
            "level": "MODEL_TRAINING",
            "message": f"Model training: {model_name} - Accuracy: {accuracy:.3f}",
            "model_name": model_name,
            "accuracy": accuracy,
            **kwargs
        }
        
        self.logger.info(f"Model training: {model_name} - Accuracy: {accuracy:.3f}")
        self._send_to_elasticsearch(log_data, "model_training")
    
    def log_api_request(self, endpoint: str, method: str, status_code: int, 
                       response_time: float, **kwargs):
        """Log API request."""
        log_data = {
            "level": "API_REQUEST",
            "message": f"API {method} {endpoint} - {status_code} ({response_time}ms)",
            "endpoint": endpoint,
            "method": method,
            "status_code": status_code,
            "response_time_ms": response_time,
            **kwargs
        }
        
        self.logger.info(f"API {method} {endpoint} - {status_code} ({response_time}ms)")
        self._send_to_elasticsearch(log_data, "api_requests")
    
    def log_kafka_message(self, topic: str, action: str, **kwargs):
        """Log Kafka message processing."""
        log_data = {
            "level": "KAFKA_MESSAGE",
            "message": f"Kafka message: {topic} - {action}",
            "topic": topic,
            "action": action,
            **kwargs
        }
        
        self.logger.info(f"Kafka message: {topic} - {action}")
        self._send_to_elasticsearch(log_data, "kafka_messages")

# Global logger instance
def get_logger(service_name: str) -> ElasticSearchLogger:
    """Get logger instance for a service."""
    return ElasticSearchLogger(service_name)
