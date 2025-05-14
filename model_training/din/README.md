# DIN (Deep Interest Network) Model

This directory contains the implementation of the DIN model, used for the ranking stage in recommendation systems. The DIN model effectively models users' dynamic interest changes by capturing the intensity of user interest in different historical items through an attention mechanism.

## Model Architecture

The core of the DIN model is the user interest representation based on attention mechanism:

1. **Attention Layer**: Calculates the relevance weights between candidate items and user history items, thereby obtaining the user interest representation for the current candidate item.
2. **Multi-Layer Perceptron**: Concatenates the attention-weighted user interest vector, candidate item vector, and average history item vector, then predicts the probability of user clicking on the candidate item through a multi-layer neural network.

## File Description

- `model.py`: DIN model architecture implementation
- `utils.py`: Data loading and processing tools
- `train.py`: Training and evaluation script

## Data Format

The DIN model requires the following two data files:

1. **train_data.csv**: Contains user historical behavior and candidate items
   - `user_id`: User ID
   - `history_items`: User's historical interaction item IDs, separated by `|`
   - `candidate_item`: Candidate item ID
   - `label`: Label (1 indicates click, 0 indicates no click)

2. **item_embeddings.csv**: Item feature data
   - `item_id`: Item ID
   - Features such as `category_hash`, `brand_hash`, `price_scaled`, etc.

## Training Command

```bash
python train.py \
    --train_data_path ../../data_processing/train_data.csv \
    --item_embeddings_path ../../data_processing/item_embeddings.csv \
    --output_dir ../../model_output/din \
    --embedding_dim 64 \
    --attention_dim 64 \
    --mlp_hidden_dims 128,64,32 \
    --batch_size 1024 \
    --lr 0.001 \
    --n_epochs 20 \
    --device cuda
```

## Parameter Description

- `--train_data_path`: Training data path
- `--item_embeddings_path`: Item feature data path
- `--output_dir`: Model output path
- `--max_samples`: Maximum sample count (for debugging)
- `--embedding_dim`: Embedding dimension
- `--attention_dim`: Attention layer hidden dimension
- `--mlp_hidden_dims`: MLP hidden layer dimensions, comma separated
- `--dropout`: Dropout rate
- `--batch_size`: Batch size
- `--lr`: Learning rate
- `--weight_decay`: Weight decay
- `--n_epochs`: Number of training epochs
- `--early_stop`: Early stopping rounds
- `--num_workers`: Number of data loading worker threads
- `--experiment_name`: MLflow experiment name
- `--device`: Computation device

## Model Output

After training, the model and related configuration will be saved in the `output_dir` directory:

- `best_model.pth`: Best model parameters and configuration

## Performance Metrics

The DIN model uses the following metrics to evaluate performance:

- **AUC**: Measures the model's ability to distinguish between positive and negative samples
- **Log Loss**: Measures the accuracy of predicted probabilities
- **Accuracy**: Binary classification accuracy using 0.5 as the threshold

## Training Process

1. Load training data and item features
2. Create model and optimizer
3. Training loop:
   - Forward propagation to calculate loss
   - Backpropagation to update parameters
   - Validation set evaluation
   - Learning rate adjustment
   - Early stopping check
4. Save best model and configuration

## MLflow Integration

During the training process, MLflow records the following:
- Model parameter configuration
- Training and validation metrics
- Best model metadata

## Using the Model

```python
from model import DIN
from utils import load_model

# Load model
model = load_model(
    model_path='../../model_output/din/best_model.pth',
    item_feat_dim=3,  # Item feature dimension
    device='cuda'
)

# Prediction
model.eval()
with torch.no_grad():
    preds = model(candidate_features, history_features, history_lengths)
```

## Notes

- Ensure that the item IDs in the `history_items` field are all included in `item_embeddings.csv`
- For new items, the model will use a default zero vector representation
- History sequence lengths exceeding the maximum length (default 50) will be truncated
- When training on large datasets, it is recommended to adjust `batch_size` and `num_workers` to accommodate available memory 