# UCI LSTM implementation

The UCI Default of Credit Card Clients dataset contains 30,000 clients and six genuine monthly observations: repayment status, bill statement amount and previous payment amount from April–September 2005. The target is `default.payment.next.month`.

The sequence is constructed chronologically (oldest month to newest month) as **6 timesteps × 3 behavioral features**. The model follows the proposal's two-layer LSTM structure with 64 and 32 hidden units, Adam optimization and binary cross-entropy equivalent loss.

> **Important:** this dataset provides 6 months, not 24 months.

## Run
```bash
python src/download_uci.py
python src/train_lstm_uci.py --data data/raw/UCI_Credit_Card.csv --epochs 30 --out results
```

Outputs: trained model, metrics JSON, training history, ROC curve and confusion matrix.
