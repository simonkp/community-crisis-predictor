import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset


class LSTMNet(nn.Module):
    def __init__(
        self,
        input_size: int,
        hidden_size: int = 64,
        num_layers: int = 2,
        num_classes: int = 4,
        dropout: float = 0.2,
    ):
        super().__init__()
        self.lstm = nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0.0,
        )
        self.dropout = nn.Dropout(dropout)
        self.fc = nn.Linear(hidden_size, num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out, _ = self.lstm(x)
        out = self.dropout(out[:, -1, :])  # last timestep only
        return self.fc(out)


class LSTMCrisisModel:
    """
    PyTorch LSTM for 4-class crisis state prediction.

    Interface mirrors XGBCrisisModel:
      .train(X, y)         — X: DataFrame, y: Series of int labels (0/1/2/3)
      .predict_proba(X)    — returns P(crisis) = P(state 2) + P(state 3)
      .predict_state(X)    — returns argmax class 0/1/2/3
    """

    def __init__(self, config: dict):
        lstm_cfg = config.get("modeling", {}).get("lstm", {})
        self.sequence_length: int = lstm_cfg.get("sequence_length", 8)
        self.hidden_size: int = lstm_cfg.get("hidden_size", 64)
        self.num_layers: int = lstm_cfg.get("num_layers", 2)
        self.dropout: float = lstm_cfg.get("dropout", 0.2)
        self.epochs: int = lstm_cfg.get("epochs", 50)
        self.walk_forward_epochs: int = lstm_cfg.get("walk_forward_epochs", self.epochs)
        self.lr: float = lstm_cfg.get("learning_rate", 0.001)
        self.batch_size: int = lstm_cfg.get("batch_size", 16)
        self.num_classes: int = 4
        self.seed: int = config.get("random_seed", 42)
        self.model: LSTMNet | None = None
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self._feature_size: int | None = None

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _make_sequences(
        self, X: np.ndarray, y: np.ndarray | None = None
    ) -> tuple[np.ndarray, np.ndarray] | np.ndarray:
        """Build sliding windows of length sequence_length."""
        seqs, labels = [], []
        for i in range(self.sequence_length, len(X) + 1):
            seqs.append(X[i - self.sequence_length : i])
            if y is not None:
                labels.append(y[i - 1])
        seqs_arr = np.array(seqs, dtype=np.float32) if seqs else np.empty(
            (0, self.sequence_length, X.shape[1]), dtype=np.float32
        )
        if y is not None:
            labels_arr = np.array(labels, dtype=np.int64) if labels else np.empty(0, dtype=np.int64)
            return seqs_arr, labels_arr
        return seqs_arr

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def train(
        self,
        X: pd.DataFrame,
        y: pd.Series,
        walk_forward: bool = False,
    ) -> None:
        torch.manual_seed(self.seed)
        X_arr = X.values.astype(np.float32)
        y_arr = y.values.astype(np.int64)

        seqs, labels = self._make_sequences(X_arr, y_arr)
        if len(seqs) == 0:
            raise ValueError("Not enough data to form LSTM sequences")

        self._feature_size = seqs.shape[2]
        self.model = LSTMNet(
            input_size=self._feature_size,
            hidden_size=self.hidden_size,
            num_layers=self.num_layers,
            num_classes=self.num_classes,
            dropout=self.dropout,
        ).to(self.device)

        dataset = TensorDataset(torch.tensor(seqs), torch.tensor(labels))
        loader = DataLoader(dataset, batch_size=self.batch_size, shuffle=True)
        optimizer = torch.optim.Adam(self.model.parameters(), lr=self.lr)
        class_counts = np.bincount(labels, minlength=self.num_classes).astype(np.float32)
        class_counts = np.maximum(class_counts, 1.0)
        class_weights = len(labels) / (self.num_classes * class_counts)
        class_weights = class_weights / class_weights.mean()
        criterion = nn.CrossEntropyLoss(
            weight=torch.tensor(class_weights, dtype=torch.float32, device=self.device)
        )

        n_epochs = self.walk_forward_epochs if walk_forward else self.epochs
        self.model.train()
        for _ in range(n_epochs):
            for xb, yb in loader:
                xb, yb = xb.to(self.device), yb.to(self.device)
                optimizer.zero_grad()
                loss = criterion(self.model(xb), yb)
                loss.backward()
                optimizer.step()

    def predict_proba(self, X: pd.DataFrame) -> np.ndarray:
        """Returns P(crisis) = P(state 2) + P(state 3) per sample."""
        if self.model is None:
            raise ValueError("Must train before predicting")
        X_arr = X.values.astype(np.float32)
        if len(X_arr) < self.sequence_length:
            return np.zeros(len(X_arr))

        seqs = self._make_sequences(X_arr)
        t = torch.tensor(seqs).to(self.device)
        self.model.eval()
        with torch.no_grad():
            probs = torch.softmax(self.model(t), dim=1).cpu().numpy()

        crisis_probs = probs[:, 2] + probs[:, 3]
        result = np.zeros(len(X_arr))
        result[self.sequence_length - 1 :] = crisis_probs
        return result

    def predict_state(self, X: pd.DataFrame) -> np.ndarray:
        """Returns predicted state 0/1/2/3 per sample."""
        if self.model is None:
            raise ValueError("Must train before predicting")
        X_arr = X.values.astype(np.float32)
        if len(X_arr) < self.sequence_length:
            return np.zeros(len(X_arr), dtype=int)

        seqs = self._make_sequences(X_arr)
        t = torch.tensor(seqs).to(self.device)
        self.model.eval()
        with torch.no_grad():
            preds = self.model(t).argmax(dim=1).cpu().numpy()

        result = np.zeros(len(X_arr), dtype=int)
        result[self.sequence_length - 1 :] = preds
        return result
