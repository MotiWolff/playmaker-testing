"""Kafka client utilities for Playmaker services."""

from kafka import KafkaProducer, KafkaConsumer
import json
import os
from typing import Dict, Any

KAFKA_BOOTSTRAP_SERVERS = os.getenv("KAFKA_BOOTSTRAP_SERVERS", "kafka:9092")

class KafkaProducerClient:
    """Kafka producer for sending messages."""
    
    def __init__(self):
        self.producer = KafkaProducer(
            bootstrap_servers=KAFKA_BOOTSTRAP_SERVERS,
            value_serializer=lambda v: json.dumps(v).encode('utf-8'),
            key_serializer=lambda k: k.encode('utf-8') if k else None,
            retries=5,
            retry_backoff_ms=1000,
            request_timeout_ms=30000,
            api_version=(0, 10, 1)
        )
    
    def send_message(self, topic: str, message: Dict[Any, Any], key: str = None):
        """Send a message to a Kafka topic."""
        try:
            future = self.producer.send(topic, value=message, key=key)
            future.get(timeout=10)
            print(f"Message sent to topic {topic}")
        except Exception as e:
            print(f"Error sending message to topic {topic}: {e}")
    
    def close(self):
        """Close the producer."""
        self.producer.close()

class KafkaConsumerClient:
    """Kafka consumer for receiving messages."""
    
    def __init__(self, topics: list, group_id: str):
        self.consumer = KafkaConsumer(
            *topics,
            bootstrap_servers=KAFKA_BOOTSTRAP_SERVERS,
            group_id=group_id,
            value_deserializer=lambda m: json.loads(m.decode('utf-8')),
            key_deserializer=lambda k: k.decode('utf-8') if k else None,
            auto_offset_reset='earliest',
            enable_auto_commit=True,
            request_timeout_ms=30000,
            api_version=(0, 10, 1)
        )
    
    def consume_messages(self, callback):
        """Consume messages and call callback for each message."""
        try:
            for message in self.consumer:
                callback(message.topic, message.key, message.value)
        except KeyboardInterrupt:
            print("Consumer stopped")
        finally:
            self.consumer.close()
    
    def close(self):
        """Close the consumer."""
        self.consumer.close()
