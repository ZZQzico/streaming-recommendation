# Recommendation System Monitoring

This directory contains the configuration for monitoring the recommendation system using Prometheus and Grafana.

## Overview

The monitoring system consists of:

1. **Prometheus** - a time-series database for storing metrics from the system
2. **Grafana** - a visualization tool for creating dashboards to display the metrics

The monitoring system is configured to collect metrics from:
- The FastAPI service (API endpoints)
- The model service (ML models)

## Metrics Collected

The system collects the following metrics:

### API Service Metrics
- Request counts and rates
- Request latencies (by endpoint)
- Active user counts
- Kafka message counts
- Redis write counts
- Spark consumed counts

### Model Service Metrics
- Model loading status
- Model inference counts and rates
- Model inference latencies
- Model error rates
- Recommendation item counts
- Redis key counts (total, user profiles, recommendations)

## Running the Monitoring System

To start the monitoring system:

```bash
cd monitoring
docker-compose up -d
```

This will start both Prometheus and Grafana services:
- Prometheus UI will be available at http://localhost:9090
- Grafana UI will be available at http://localhost:3000 (login with admin/admin)

The recommendation dashboard will be automatically provisioned and available in Grafana.

## Adding Custom Metrics

To add custom metrics, modify the following files:
- `data_pipeline/metrics.py` for API service metrics
- `model_service/models.py` for model-related metrics

Then update the Grafana dashboard to display the new metrics.

## Troubleshooting

If metrics are not showing up in Grafana:

1. Check if the metrics endpoints are accessible:
   - API service: http://localhost:8000/metrics
   - Model service: http://localhost:8001/metrics

2. Check if Prometheus can scrape the metrics:
   - Go to http://localhost:9090/targets to see if targets are up

3. Check if the metrics are available in Prometheus:
   - Go to http://localhost:9090/graph and search for the metric names 