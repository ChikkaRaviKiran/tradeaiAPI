"""AI Prediction Engine — ML models for market direction prediction.

SRS Module 3.3: AI models analyze data and predict market direction,
stock probability score, and volatility expectation.

Models:
  - Random Forest (trend classification)
  - XGBoost (probability scoring)
  - LSTM placeholder (time series — requires tensorflow, deferred)

Output:
  Market Bias: Bullish / Bearish / Neutral
  Confidence: 0-100%
"""

from __future__ import annotations

import logging
import pickle
from datetime import datetime
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd
import pytz

from app.core.models import GlobalBias, MarketPrediction

logger = logging.getLogger(__name__)
_IST = pytz.timezone("Asia/Kolkata")
_MODEL_DIR = Path(__file__).parent / "trained_models"


class MarketPredictor:
    """Predicts market direction using ML models."""

    def __init__(self) -> None:
        self._rf_model = None  # RandomForestClassifier
        self._xgb_model = None  # XGBClassifier
        self._feature_columns: list[str] = []

    def predict(
        self,
        instrument: str,
        features: pd.DataFrame,
    ) -> MarketPrediction:
        """Generate market prediction from technical features.

        Args:
            instrument: Symbol name
            features: DataFrame with technical indicator columns

        Returns:
            MarketPrediction with bias and confidence
        """
        if features.empty:
            return MarketPrediction(
                instrument=instrument,
                bias=GlobalBias.NEUTRAL,
                confidence=0,
                model="none",
            )

        # If no trained model exists, use rule-based prediction
        if self._rf_model is None:
            return self._rule_based_prediction(instrument, features)

        try:
            X = self._prepare_features(features)
            if X is None:
                return self._rule_based_prediction(instrument, features)

            # Random Forest prediction
            rf_pred = self._rf_model.predict(X[-1:])
            rf_proba = self._rf_model.predict_proba(X[-1:])[0]
            confidence = max(rf_proba) * 100

            bias_map = {0: GlobalBias.BEARISH, 1: GlobalBias.NEUTRAL, 2: GlobalBias.BULLISH}
            bias = bias_map.get(rf_pred[0], GlobalBias.NEUTRAL)

            return MarketPrediction(
                instrument=instrument,
                bias=bias,
                confidence=round(confidence, 1),
                model="random_forest",
                timestamp=datetime.now(_IST),
            )

        except Exception:
            logger.exception("ML prediction failed for %s", instrument)
            return self._rule_based_prediction(instrument, features)

    def _rule_based_prediction(
        self, instrument: str, features: pd.DataFrame
    ) -> MarketPrediction:
        """Fallback: rule-based prediction using technical indicators.

        Used when ML models aren't trained yet.
        """
        if features.empty:
            return MarketPrediction(instrument=instrument, bias=GlobalBias.NEUTRAL, confidence=0, model="rules")

        last = features.iloc[-1]
        score = 0  # -100 to +100

        # RSI signal
        rsi = last.get("rsi", 50)
        if pd.notna(rsi):
            if rsi > 60:
                score += 15
            elif rsi < 40:
                score -= 15

        # EMA alignment
        ema9 = last.get("ema9") or last.get("ema_9")
        ema20 = last.get("ema20") or last.get("ema_20")
        ema50 = last.get("ema50") or last.get("ema_50")
        close = last.get("close") or last.get("Close", 0)

        if ema9 and ema20 and pd.notna(ema9) and pd.notna(ema20):
            if ema9 > ema20:
                score += 15
            else:
                score -= 15

        if ema20 and ema50 and pd.notna(ema20) and pd.notna(ema50):
            if ema20 > ema50:
                score += 10
            else:
                score -= 10

        if close and ema50 and pd.notna(ema50):
            if close > ema50:
                score += 10
            else:
                score -= 10

        # MACD
        macd_hist = last.get("macd_hist") or last.get("macd_diff")
        if macd_hist and pd.notna(macd_hist):
            if macd_hist > 0:
                score += 10
            else:
                score -= 10

        # ADX strength
        adx = last.get("adx")
        if adx and pd.notna(adx) and adx > 25:
            score = int(score * 1.3)  # Amplify in trending market

        # Convert to bias
        confidence = min(abs(score), 100)
        if score > 15:
            bias = GlobalBias.BULLISH
        elif score < -15:
            bias = GlobalBias.BEARISH
        else:
            bias = GlobalBias.NEUTRAL

        return MarketPrediction(
            instrument=instrument,
            bias=bias,
            confidence=confidence,
            model="rules",
            timestamp=datetime.now(_IST),
        )

    def _prepare_features(self, df: pd.DataFrame) -> Optional[np.ndarray]:
        """Prepare feature matrix for ML model."""
        if not self._feature_columns:
            return None

        available = [c for c in self._feature_columns if c in df.columns]
        if len(available) < len(self._feature_columns) * 0.7:
            return None

        X = df[available].fillna(0).values
        return X

    def train(self, X: np.ndarray, y: np.ndarray) -> dict:
        """Train Random Forest and XGBoost models.

        Args:
            X: Feature matrix (n_samples, n_features)
            y: Labels (0=bearish, 1=neutral, 2=bullish)

        Returns:
            Training metrics dict
        """
        try:
            from sklearn.ensemble import RandomForestClassifier
            from sklearn.model_selection import cross_val_score

            # Random Forest
            rf = RandomForestClassifier(
                n_estimators=100,
                max_depth=10,
                random_state=42,
                n_jobs=-1,
            )
            rf_scores = cross_val_score(rf, X, y, cv=5, scoring="accuracy")
            rf.fit(X, y)
            self._rf_model = rf

            metrics = {
                "rf_accuracy": round(rf_scores.mean(), 4),
                "rf_std": round(rf_scores.std(), 4),
                "samples": len(y),
                "features": X.shape[1],
            }

            # XGBoost (if available)
            try:
                from xgboost import XGBClassifier

                xgb = XGBClassifier(
                    n_estimators=100,
                    max_depth=6,
                    learning_rate=0.1,
                    use_label_encoder=False,
                    eval_metric="mlogloss",
                )
                xgb_scores = cross_val_score(xgb, X, y, cv=5, scoring="accuracy")
                xgb.fit(X, y)
                self._xgb_model = xgb
                metrics["xgb_accuracy"] = round(xgb_scores.mean(), 4)
            except ImportError:
                logger.warning("XGBoost not installed — skipping")

            logger.info("Model training complete: %s", metrics)
            return metrics

        except ImportError:
            logger.warning("scikit-learn not installed — ML models unavailable")
            return {"error": "scikit-learn not installed"}

    def save_models(self, path: Optional[Path] = None) -> None:
        """Save trained models to disk."""
        save_dir = path or _MODEL_DIR
        save_dir.mkdir(parents=True, exist_ok=True)

        if self._rf_model:
            with open(save_dir / "rf_model.pkl", "wb") as f:
                pickle.dump(self._rf_model, f)
        if self._xgb_model:
            with open(save_dir / "xgb_model.pkl", "wb") as f:
                pickle.dump(self._xgb_model, f)
        if self._feature_columns:
            with open(save_dir / "feature_columns.pkl", "wb") as f:
                pickle.dump(self._feature_columns, f)

        logger.info("Models saved to %s", save_dir)

    def load_models(self, path: Optional[Path] = None) -> bool:
        """Load trained models from disk. Returns True if loaded."""
        load_dir = path or _MODEL_DIR
        if not load_dir.exists():
            return False

        try:
            rf_path = load_dir / "rf_model.pkl"
            if rf_path.exists():
                with open(rf_path, "rb") as f:
                    self._rf_model = pickle.load(f)

            xgb_path = load_dir / "xgb_model.pkl"
            if xgb_path.exists():
                with open(xgb_path, "rb") as f:
                    self._xgb_model = pickle.load(f)

            cols_path = load_dir / "feature_columns.pkl"
            if cols_path.exists():
                with open(cols_path, "rb") as f:
                    self._feature_columns = pickle.load(f)

            loaded = bool(self._rf_model or self._xgb_model)
            if loaded:
                logger.info("ML models loaded from %s", load_dir)
            return loaded

        except Exception:
            logger.exception("Error loading ML models")
            return False
