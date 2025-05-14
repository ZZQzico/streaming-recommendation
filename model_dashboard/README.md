# Recommendation System Model Evaluation Dashboard

This dashboard visualizes the evaluation metrics of three recommendation system models (LightGCN, DIN, and RankNet) during their evaluation phase.

## Features

- Displays LightGCN recall model evaluation metrics: Precision@K, Recall@K, NDCG@K
- Displays DIN ranking model evaluation metrics: AUC, Accuracy, LogLoss, Loss
- Displays RankNet re-ranking model evaluation metrics: Accuracy, NDCG, MRR, Loss
- Provides performance comparison between the three models
- Visualizes evaluation results using various chart types
- Includes detailed descriptions of models and evaluation metrics

## Installation and Usage

### Install Dependencies

```bash
pip install -r requirements.txt
```

### Launch Dashboard

```bash
python run.py
```

The service will be available at http://localhost:8080.

## Data Sources

The dashboard uses real datasets for model evaluation:

1. **LightGCN Model**: Uses `lightgcn_data.csv` (user-item interaction data)
2. **DIN Model**: Uses `train_data.csv` and `item_embeddings.csv`
3. **RankNet Model**: Uses `train_data.csv` and `item_embeddings.csv`

All model evaluations use an 80/20 training/testing data split.

## Project Structure

```
model_dashboard/
├── app.py                  # Flask application entry point
├── model_evaluation.py     # Model evaluation logic
├── run.py                  # Startup script
├── requirements.txt        # Dependencies
├── templates/              # HTML templates
│   └── dashboard.html      # Dashboard main page
└── static/                 # Static resources (CSS, JS, etc.)
```

## Notes

1. This is a standalone application that does not affect existing recommendation systems and frontend.
2. To improve performance, the number of users and items is limited during evaluation.
3. The current version uses trained model weights from the model_output directory. 