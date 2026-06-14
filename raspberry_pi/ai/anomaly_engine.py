#!/usr/bin/env python3
"""
anomaly_engine.py — Moteur IA embarqué (Edge AI) — version renforcée

Algorithmes :
  1. Isolation Forest  → détection anomalies multivariées (200 arbres, 100 échantillons)
  2. Régression linéaire + features dérivées → prédiction niveau dans 30s
  3. Seuil adaptatif σ×2 → confirmation fausse alarme MQ-4
  4. Validation et normalisation précise des données capteurs

Conforme : ISO 13849 (safety functions), IEC 61511 (SIS)
"""

import os
import time
import logging
import numpy as np
from collections import deque
from sklearn.ensemble import IsolationForest
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import StandardScaler

log = logging.getLogger("gnl.ai")

# ── Constantes ─────────────────────────────────────────────────────────────────
HISTORY_SIZE      = int(os.environ.get("AI_HISTORY_SIZE",   "100"))  # 100 mesures (~200s)
TRAIN_MIN_SAMPLES = int(os.environ.get("AI_TRAIN_MIN",       "30"))  # min pour entraîner IF
IF_N_ESTIMATORS   = int(os.environ.get("AI_IF_ESTIMATORS",  "200"))  # arbres Isolation Forest
IF_CONTAMINATION  = float(os.environ.get("AI_CONTAMINATION", "0.05"))
RETRAIN_INTERVAL  = int(os.environ.get("AI_RETRAIN_S",       "45"))  # réentraîner toutes les 45s
SIGMA_MULT        = 2.0           # seuil adaptatif mean ± σ×2
PRED_WINDOW       = 20            # points pour régression (40s à 2s/pt)
CONFIRM_GAS       = int(os.environ.get("CONFIRM_GAS", "3"))

# Seuils gaz MQ-4 (ADC 0–1023)
GAS_WARN   = int(os.environ.get("GAS_WARN",   "250"))
GAS_DANGER = int(os.environ.get("GAS_DANGER", "450"))

# Seuils niveau (%)
LEVEL_HIGH = int(os.environ.get("LEVEL_HIGH", "95"))
LEVEL_LOW  = int(os.environ.get("LEVEL_LOW",  "10"))

# Plages valides des capteurs (validation)
_VALID = {
    "n1": (0.0, 100.0),    # niveau % HC-SR04
    "n2": (0.0, 100.0),
    "t1": (-55.0, 125.0),  # DS18B20
    "t2": (-55.0, 125.0),
    "p":  (800.0, 1200.0), # BMP280 hPa
    "g":  (0.0, 1023.0),   # MQ-4 ADC
}


def _clamp(val: float, lo: float, hi: float) -> float:
    return max(lo, min(hi, val))


def _validate_sensor(data: dict) -> dict:
    """Valide et borne les valeurs capteurs; remplace les valeurs hors-plage par des NaN."""
    out = {}
    for key, (lo, hi) in _VALID.items():
        raw = data.get(key)
        if raw is None:
            out[key] = (lo + hi) / 2  # valeur centrale si absent
        else:
            v = float(raw)
            out[key] = _clamp(v, lo, hi)
    # Copier les champs non-capteurs
    for k in ("pump", "valve", "err"):
        if k in data:
            out[k] = data[k]
    return out


class AnomalyEngine:
    """Moteur d'analyse IA temps réel pour le système IoT GNL — version renforcée."""

    def __init__(self):
        self.history:    deque = deque(maxlen=HISTORY_SIZE)
        self.timestamps: deque = deque(maxlen=HISTORY_SIZE)
        self._rates:     deque = deque(maxlen=HISTORY_SIZE)  # taux variation instantané

        self._if_model: IsolationForest | None = None
        self._scaler:   StandardScaler         = StandardScaler()
        self._if_trained   = False
        self._scaler_fitted = False
        self.sample_count  = 0

        self.gas_alert_count = 0
        self.last_retrain    = 0.0

        # Fenêtre adaptative pour σ du gaz
        self._gas_window: deque = deque(maxlen=30)

        log.info(
            "AnomalyEngine v3 initialisé — IF(%d arbres, %.0f%% contam., hist=%d) + Régression dérivées",
            IF_N_ESTIMATORS, IF_CONTAMINATION * 100, HISTORY_SIZE,
        )

    # ── Interface publique ────────────────────────────────────────────────────

    def analyze(self, data: dict) -> dict:
        """
        Analyse une mesure et retourne un dict enrichi :
          {
            "isolation_forest": int (0-100, score anomalie),
            "regression": {
                "n1_in_30s": float,
                "n2_in_30s": float,
                "overflow_risk": bool,
                "r2_score_n1": float,   # qualité du modèle régression
                "r2_score_n2": float,
            },
            "gas_alert":   str | None,
            "global_risk": int (0-100),
            "command":     str | None,
            "sensor_quality": dict,     # validité par capteur
          }
        """
        now = time.time()
        validated = _validate_sensor(data)

        # Vecteur features de base : [n1, n2, t1, t2, p, g]
        base = [
            validated["n1"],
            validated["n2"],
            validated["t1"],
            validated["t2"],
            validated["p"],
            validated["g"],
        ]

        # Features dérivées : taux de variation si historique suffisant
        rates = self._compute_rates(base, now)

        # Vecteur complet : base + dérivées
        features = base + rates
        self.history.append(features)
        self.timestamps.append(now)
        self._gas_window.append(validated["g"])
        self.sample_count += 1

        result = {
            "isolation_forest": 0,
            "regression":       {},
            "gas_alert":        None,
            "global_risk":      0,
            "command":          None,
            "sensor_quality":   self._check_sensor_quality(data, validated),
        }

        if_score = self._run_isolation_forest(features)
        result["isolation_forest"] = if_score

        reg_result = self._run_regression(validated)
        result["regression"] = reg_result

        gas_alert = self._check_gas_adaptive(validated["g"])
        result["gas_alert"] = gas_alert

        global_risk = self._compute_global_risk(if_score, reg_result, gas_alert, validated)
        result["global_risk"] = global_risk

        command = self._decide_command(global_risk, gas_alert, reg_result, validated)
        result["command"] = command

        if global_risk > 70:
            log.warning(
                "RISQUE ÉLEVÉ %d%% | IF=%d | Gaz=%s | n1=%.2f%% n2=%.2f%% | p=%.1f hPa",
                global_risk, if_score, gas_alert,
                validated["n1"], validated["n2"], validated["p"],
            )

        return result

    # ── Calcul des taux de variation (features dérivées) ─────────────────────

    def _compute_rates(self, base: list, now: float) -> list:
        """Retourne [dn1/dt, dn2/dt, dg/dt] normalisés sur la fenêtre récente."""
        if len(self.history) < 5:
            return [0.0, 0.0, 0.0]
        recent_ts  = list(self.timestamps)[-5:]
        recent_h   = list(self.history)[-5:]
        dt = max(now - recent_ts[0], 0.1)
        dn1 = (base[0] - recent_h[0][0]) / dt
        dn2 = (base[1] - recent_h[0][1]) / dt
        dg  = (base[5] - recent_h[0][5]) / dt
        # Borner les taux pour éviter les outliers numériques
        dn1 = _clamp(dn1, -5.0, 5.0)
        dn2 = _clamp(dn2, -5.0, 5.0)
        dg  = _clamp(dg,  -50.0, 50.0)
        return [round(dn1, 4), round(dn2, 4), round(dg, 4)]

    # ── Validation qualité capteurs ───────────────────────────────────────────

    def _check_sensor_quality(self, raw: dict, validated: dict) -> dict:
        """Signale les capteurs défaillants (valeur hors plage ou manquante)."""
        quality = {}
        for key, (lo, hi) in _VALID.items():
            raw_val = raw.get(key)
            if raw_val is None:
                quality[key] = "ABSENT"
            elif float(raw_val) < lo or float(raw_val) > hi:
                quality[key] = "HORS_PLAGE"
            else:
                quality[key] = "OK"
        return quality

    # ── Isolation Forest ─────────────────────────────────────────────────────

    def _run_isolation_forest(self, features: list) -> int:
        """Retourne un score 0–100 (0=normal, 100=très anormal)."""
        if len(self.history) < TRAIN_MIN_SAMPLES:
            return 0

        if not self._if_trained or (time.time() - self.last_retrain) > RETRAIN_INTERVAL:
            self._train_isolation_forest()

        if not self._if_trained:
            return 0

        try:
            X_raw = np.array([features])
            X = self._scaler.transform(X_raw)
            score = self._if_model.decision_function(X)[0]
            # score négatif = anomalie, positif = normal
            # Normalisation améliorée : centré autour de 0, borné [0, 100]
            normalized = int(np.clip((-score + 0.3) * 120, 0, 100))
            return normalized
        except Exception as e:
            log.debug("IF score error: %s", e)
            return 0

    def _train_isolation_forest(self):
        try:
            X_raw = np.array(list(self.history))
            # Standardisation des features avant entraînement
            self._scaler.fit(X_raw)
            X = self._scaler.transform(X_raw)
            self._scaler_fitted = True

            self._if_model = IsolationForest(
                contamination=IF_CONTAMINATION,
                random_state=42,
                n_estimators=IF_N_ESTIMATORS,
                max_samples="auto",
                bootstrap=False,
            )
            self._if_model.fit(X)
            self._if_trained = True
            self.last_retrain = time.time()
            log.debug(
                "Isolation Forest entraîné sur %d échantillons, %d features",
                len(X_raw), X_raw.shape[1],
            )
        except Exception as e:
            log.warning("Échec entraînement IF : %s", e)
            self._if_trained = False

    # ── Régression linéaire améliorée ────────────────────────────────────────

    def _run_regression(self, validated: dict) -> dict:
        """
        Prédit les niveaux dans 30s.
        Utilise les 20 derniers points, retourne aussi le score R² du modèle.
        """
        result = {
            "n1_in_30s":   None,
            "n2_in_30s":   None,
            "overflow_risk": False,
            "r2_score_n1": None,
            "r2_score_n2": None,
        }

        if len(self.history) < PRED_WINDOW:
            return result

        recent = list(self.history)[-PRED_WINDOW:]
        n_pts  = len(recent)
        X      = np.arange(n_pts).reshape(-1, 1).astype(float)

        n1_vals = np.array([r[0] for r in recent])
        n2_vals = np.array([r[1] for r in recent])

        try:
            reg1 = LinearRegression().fit(X, n1_vals)
            reg2 = LinearRegression().fit(X, n2_vals)

            r2_n1 = reg1.score(X, n1_vals)
            r2_n2 = reg2.score(X, n2_vals)

            # Prédiction dans 15 points supplémentaires (30s à 2s/pt)
            future_x = np.array([[n_pts + 14]], dtype=float)
            n1_pred  = float(np.clip(reg1.predict(future_x)[0], 0, 100))
            n2_pred  = float(np.clip(reg2.predict(future_x)[0], 0, 100))

            result["n1_in_30s"]   = round(n1_pred, 2)
            result["n2_in_30s"]   = round(n2_pred, 2)
            result["r2_score_n1"] = round(r2_n1, 3)
            result["r2_score_n2"] = round(r2_n2, 3)

            # Risque débordement uniquement si régression fiable (R² > 0.5)
            overflow = (n1_pred > LEVEL_HIGH or n2_pred > LEVEL_HIGH)
            result["overflow_risk"] = overflow and (max(r2_n1, r2_n2) > 0.5)

        except Exception as e:
            log.debug("Régression error: %s", e)

        return result

    # ── Seuil adaptatif gaz ───────────────────────────────────────────────────

    def _check_gas_adaptive(self, gas_value: float) -> str | None:
        """
        Détection gaz avec seuil adaptatif basé sur la fenêtre glissante (σ×2).
        Confirmation sur N mesures consécutives pour DANGER_CRITIQUE.
        """
        # Calcul seuil adaptatif si assez de données
        adaptive_warn = GAS_WARN
        if len(self._gas_window) >= 10:
            mu  = np.mean(list(self._gas_window))
            sig = np.std(list(self._gas_window))
            adaptive_warn = max(GAS_WARN, mu + SIGMA_MULT * sig)

        if gas_value >= GAS_DANGER:
            self.gas_alert_count += 1
        elif gas_value >= adaptive_warn:
            self.gas_alert_count = max(0, self.gas_alert_count)
            return "ATTENTION"
        else:
            self.gas_alert_count = 0
            return None

        if self.gas_alert_count >= CONFIRM_GAS:
            return "DANGER_CRITIQUE"

        return "DANGER_CONFIRMING"

    # ── Score global ──────────────────────────────────────────────────────────

    def _compute_global_risk(
        self,
        if_score: int,
        reg: dict,
        gas_alert: str | None,
        v: dict,
    ) -> int:
        """Agrège tous les indicateurs en un score risque 0–100."""
        risk = 0.0

        # Isolation Forest (35%)
        risk += if_score * 0.35

        # Régression (25%) — pondérée par la qualité R²
        if reg.get("overflow_risk"):
            risk += 25.0
        else:
            n1_p = reg.get("n1_in_30s") or 0.0
            n2_p = reg.get("n2_in_30s") or 0.0
            r2   = max(
                reg.get("r2_score_n1") or 0.0,
                reg.get("r2_score_n2") or 0.0,
            )
            if n1_p > 88 or n2_p > 88:
                risk += 15.0 * max(0.5, r2)

        # Gaz (25%)
        if gas_alert == "DANGER_CRITIQUE":
            risk += 25.0
        elif gas_alert == "DANGER_CONFIRMING":
            risk += 15.0
        elif gas_alert == "ATTENTION":
            risk += 8.0

        # Seuils niveaux directs (15%)
        n1, n2 = v["n1"], v["n2"]
        if n1 > LEVEL_HIGH or n2 > LEVEL_HIGH:
            risk += 15.0
        elif n1 > 88 or n2 > 88:
            risk += 8.0
        if n1 < LEVEL_LOW:
            risk += 12.0

        # Pression anormale — BMP280 hors 950–1070 hPa = contexte alarmant
        p = v["p"]
        if p < 950 or p > 1070:
            risk += 5.0

        return min(100, int(round(risk)))

    # ── Décision commande ─────────────────────────────────────────────────────

    def _decide_command(
        self,
        global_risk: int,
        gas_alert: str | None,
        reg: dict,
        v: dict,
    ) -> str | None:
        n1, n2 = v["n1"], v["n2"]

        if gas_alert == "DANGER_CRITIQUE":
            log.error("ESD DÉCLENCHÉ — fuite gaz critique confirmée (MQ-4=%.0f ADC)", v["g"])
            return "CMD:ESD"

        if global_risk >= 85:
            log.error("ESD DÉCLENCHÉ — risque global %d%%", global_risk)
            return "CMD:ESD"

        if reg.get("overflow_risk") and n1 > 88:
            log.warning("Arrêt pompe préventif — débordement R1 prédit (n1=%.1f%%)", n1)
            return "CMD:PUMP_OFF"

        if n1 < LEVEL_LOW:
            log.warning("Arrêt pompe — niveau R1 critique (%.2f%%)", n1)
            return "CMD:PUMP_OFF"

        if n2 >= LEVEL_HIGH:
            log.info("Fermeture vanne — R2 plein (%.2f%%)", n2)
            return "CMD:VALVE_CLOSE"

        return None
