# Offline Training Data Description

This directory contains all preprocessed data required for offline model training, prepared for the model training owner.

## 1. `train_data.csv`

### Description:
- Used for training **ranking models** (e.g., DIN, RankNet).
- Contains user interaction history, candidate items, and labels for positive/negative samples.

### Fields:
| Field            | Description                                                        |
|------------------|--------------------------------------------------------------------|
| `user_id`        | Unique user identifier                                             |
| `history_items`  | User's historical item IDs, separated by `|` (e.g., `B001|B005`)    |
| `candidate_item` | Candidate item ID                                                  |
| `label`          | 1 = positive sample (clicked), 0 = negative sample (not clicked)   |

### Example:
```
user_id,history_items,candidate_item,label
A1,B001|B005,B010,1
A1,B001|B005,B015,0
```

---

## 2. `item_embeddings.csv`

### Description:
- Normalized features for each item, used as input for ranking models.
- Features include: category hash, brand hash, normalized price.

### Fields:
| Field          | Description                                       |
|----------------|---------------------------------------------------|
| `item_id`      | Unique item identifier                            |
| `category_hash`| Encoded category feature (float, range [0, 1])    |
| `brand_hash`   | Encoded brand feature (float, range [0, 1])       |
| `price_scaled` | Normalized price feature (float, range [0, 1])    |

### Example:
```
item_id,category_hash,brand_hash,price_scaled
B001,0.23,0.45,0.88
B005,0.67,0.12,0.50
```

---

## 3. `lightgcn_data.csv`

### Description:
- Used for training **LightGCN recall model**, based on user-item interaction graph.
- Only retains interactions with rating ≥ 4 as positive samples.

### Fields:
| Field        | Description                       |
|--------------|-----------------------------------|
| `user_id`    | Unique user identifier            |
| `item_id`    | Item ID with positive interaction |

### Example:
```
user_id,item_id
A1,B001
A2,B005
```

---

## Data Usage Tips:

- Use `train_data.csv` together with `item_embeddings.csv` for training **ranking models** such as **DIN** and **RankNet**.
  - `train_data.csv` provides user behavior sequences and candidate item pairs (positive/negative samples).
  - `item_embeddings.csv` supplies normalized item features for model input.

- Use `lightgcn_data.csv` for training the **LightGCN** recall model, leveraging user-item interaction graphs.

### Important:
- `history_items` in `train_data.csv` are separated by `|`, representing sequential item interactions.
- `embedding` values in `item_embeddings.csv` are **comma-separated floats**, representing the item's normalized features: category hash, brand hash, and normalized price.

### Training Flow Suggestion:
1. Train **LightGCN** with `lightgcn_data.csv` to generate initial user/item embeddings for recall.
2. Use recalled items and user histories from `train_data.csv` to fine-tune ranking with **DIN** or **RankNet**, utilizing `item_embeddings.csv` for item-level features.

