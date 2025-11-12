## Getting Started

### Prerequisites

-   Python 3.9, 3.10, 3.11, or 3.12
-   Git
-   CUDA-compatible GPU (recommended for faster training)

### Installation Steps

1. **Fork and Clone the Repository**

    ```bash
    git clone https://github.com/YOUR_USERNAME/YOUR_REPO_NAME.git
    cd YOUR_REPO_NAME
    ```

2. **Create a Virtual Environment**

    ```bash
    python -m venv venv
    ```

3. **Activate the Virtual Environment**

    ```bash
    venv\Scripts\activate
    ```

4. **Install Dependencies**
    ```bash
    pip install -r requirements.txt
    ```

## Dataset

The project uses the English-Bengali Dataset (20k_4) containing 20,000 samples for sentiment analysis tasks, split over 4 categories.

### Data Preprocessing

Run the preprocessing notebook `(preprocess.ipynb)` to clean and prepare the data:

**Output:** The cleaned dataset will be saved at `data/En-Ba-Dataset(20k_4)/cleaned_data.csv`

**Important:** Use this cleaned dataset for all subsequent experiments and model training.

## Usage

### 1. Data Preprocessing

```bash
jupyter notebook preprocess.ipynb
```

Follow the notebook instructions to generate `cleaned_data.csv`

### 2. Training and Testing

```bash
jupyter notebook train-test.ipynb
```

This notebook provides the main workflow for training and evaluating models.

### 3. Exploring Individual Embeddings

Navigate to any embedding directory and run the respective `main.ipynb`:

```bash
cd embeddings/7-geminiApi-m/
jupyter notebook main.ipynb
```

### 4. Final Embeddings

The final consolidated embeddings to be used for all further activities are saved at:

```
embeddings/embedding_final.csv
```

## Contributing

1. Fork the repository
2. Commit your changes (`git commit -m 'Add some commit'`)
3. Push to the branch (`git push origin main`)
4. Open a Pull Request
5. If you are a contributor, merge the commit with main
