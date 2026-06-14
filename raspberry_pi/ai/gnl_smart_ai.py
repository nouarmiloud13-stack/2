#!/usr/bin/env python3
"""
gnl_smart_ai.py — IA autonome pour le système IoT GNL
Fournisseur unique : Gemma4 local (llama.cpp)
Fallback : règles fixes si Gemma4 indisponible
"""

import os
import re
import json
import time
import logging
from datetime import datetime, timezone
from collections import deque

import requests

log = logging.getLogger("gnl.smart_ai")

# ── Configuration ─────────────────────────────────────────────────────────────
GEMMA4_HOST        = os.environ.get("GEMMA4_HOST",        "localhost")
GEMMA4_PORT        = os.environ.get("GEMMA4_SERVER_PORT", "8080")
GEMMA4_URL         = f"http://{GEMMA4_HOST}:{GEMMA4_PORT}"
GEMMA4_TIMEOUT     = int(os.environ.get("GEMMA4_TIMEOUT",     "45"))
GEMMA4_TEMPERATURE = float(os.environ.get("GEMMA4_TEMPERATURE", "0.15"))

AI_ROUTINE_INTERVAL = int(os.environ.get("GNL_AI_INTERVAL",     "30"))
AI_RISK_THRESHOLD   = int(os.environ.get("GNL_AI_RISK_TRIGGER", "60"))

# ── Prompt système ────────────────────────────────────────────────────────────
_SYSTEM = """Tu es un système d'IA embarqué expert en surveillance et contrôle de réservoirs GNL (Gaz Naturel Liquéfié).

RÈGLES DE SÉCURITÉ ABSOLUES :
- MQ-4 > 450 ADC → commande CMD:ESD OBLIGATOIRE
- Niveau R1 < 10% ET pompe active → CMD:PUMP_OFF immédiat
- Niveau R2 ≥ 95% ET vanne ouverte → CMD:VALVE_CLOSE immédiat
- Risque global ≥ 85% → évaluer CMD:ESD
- Ne jamais activer pompe si gaz > 250 ADC

LOGIQUE DISTRIBUTION :
- R1 ≥ 90% ET R2 < 95% → CMD:PUMP_ON + CMD:VALVE_OPEN
- R2 ≥ 95% → CMD:VALVE_CLOSE + CMD:PUMP_OFF
- R1 ≤ 15% → CMD:PUMP_OFF

Commandes Arduino valides : CMD:PUMP_ON, CMD:PUMP_OFF, CMD:VALVE_OPEN, CMD:VALVE_CLOSE, CMD:ESD
Réponds UNIQUEMENT en JSON valide. Aucun texte en dehors du JSON."""

_JSON_SCHEMA = """{
  "severity": "INFO|ATTENTION|DANGER|CRITIQUE",
  "diagnostic": "résumé situation en 1 phrase",
  "commands": ["CMD:PUMP_ON"],
  "predictions": {
    "r1_5min": 75.0,
    "r2_5min": 45.0,
    "temps_avant_plein_r2_min": 15,
    "temps_avant_vide_r1_min": -1,
    "tendance_gaz": "stable|hausse|baisse"
  },
  "recommendations": ["action 1"],
  "autonomie_h": 4.5,
  "details": "analyse technique"
}"""


# ═══════════════════════════════════════════════════════════════════════════════
# Fournisseur Gemma4 (llama.cpp HTTP)
# ═══════════════════════════════════════════════════════════════════════════════

class GemmaProvider:
    """Utilise le serveur llama.cpp hébergeant Gemma4."""

    def __init__(self):
        self._available = self._check()

    def _check(self) -> bool:
        try:
            r = requests.get(f"{GEMMA4_URL}/health", timeout=4)
            ok = r.status_code == 200
            if ok:
                log.info("Gemma4 disponible : %s", GEMMA4_URL)
            else:
                log.warning("Gemma4 inaccessible (HTTP %d) : %s", r.status_code, GEMMA4_URL)
            return ok
        except Exception as e:
            log.warning("Gemma4 non joignable : %s — %s", GEMMA4_URL, e)
            return False

    @property
    def available(self) -> bool:
        return self._available

    def analyze(self, prompt_body: str) -> dict | None:
        if not self._available:
            self._available = self._check()
            if not self._available:
                return None

        full_prompt = (
            f"<start_of_turn>user\n{_SYSTEM}\n\n{prompt_body}\n\n"
            f"Réponds UNIQUEMENT avec ce JSON (sans commentaire) :\n{_JSON_SCHEMA}"
            f"<end_of_turn>\n<start_of_turn>model\n"
        )

        try:
            resp = requests.post(
                f"{GEMMA4_URL}/completion",
                json={
                    "prompt":      full_prompt,
                    "n_predict":   512,
                    "temperature": GEMMA4_TEMPERATURE,
                    "top_p":       0.9,
                    "stop":        ["</s>", "<end_of_turn>", "<start_of_turn>"],
                    "stream":      False,
                },
                timeout=GEMMA4_TIMEOUT,
            )
            resp.raise_for_status()
            text = resp.json().get("content", "").strip()
            return _parse_json_response(text)
        except requests.Timeout:
            log.warning("Gemma4 timeout (%ds)", GEMMA4_TIMEOUT)
            self._available = False
            return None
        except Exception as e:
            log.warning("Gemma4 erreur : %s", e)
            self._available = False
            return None


# ═══════════════════════════════════════════════════════════════════════════════
# Utilitaire parsing JSON
# ═══════════════════════════════════════════════════════════════════════════════

def _parse_json_response(text: str) -> dict | None:
    match = re.search(r"\{[\s\S]*\}", text)
    if not match:
        log.warning("Gemma4 : réponse sans JSON valide : %s", text[:120])
        return None
    try:
        data = json.loads(match.group())
        data["source"] = "gemma4"
        return data
    except json.JSONDecodeError as e:
        log.warning("Gemma4 : JSON malformé : %s", e)
        return None


# ═══════════════════════════════════════════════════════════════════════════════
# Moteur principal
# ═══════════════════════════════════════════════════════════════════════════════

class GNLSmartAI:
    """IA autonome GNL — Gemma4 local en primaire, règles fixes en fallback."""

    def __init__(self):
        self._gemma = GemmaProvider()
        self._last_call_ts:  float = 0.0
        self._last_result:   dict  = {}
        self._last_pump:     int   = -1
        self._last_valve:    int   = -1
        self._history: deque       = deque(maxlen=30)

        if self._gemma.available:
            log.info("GNLSmartAI prêt (fournisseur : Gemma4 @ %s)", GEMMA4_URL)
        else:
            log.warning(
                "GNLSmartAI : Gemma4 non disponible — mode règles fixes\n"
                "  → Lancer Docker : docker compose --profile ai up -d gemma4\n"
                "  → Ou : make start-gemma4"
            )

    @property
    def is_available(self) -> bool:
        return self._gemma.available

    # ── Interface publique ────────────────────────────────────────────────────

    def analyze(self, sensor_data: dict, anomaly_scores: dict) -> dict:
        self._history.append({
            "ts": time.time(),
            "n1": sensor_data.get("n1", 0),
            "n2": sensor_data.get("n2", 0),
            "g":  sensor_data.get("g",  0),
        })

        if not self._should_call(sensor_data, anomaly_scores):
            result = dict(self._last_result)
            result["from_cache"] = True
            return result

        if not self._gemma.available:
            return self._fallback(sensor_data, anomaly_scores)

        prompt = self._build_prompt(sensor_data, anomaly_scores)

        try:
            raw = self._gemma.analyze(prompt)
        except Exception as e:
            log.error("Gemma4 exception : %s", e)
            raw = None

        if raw is None:
            return self._fallback(sensor_data, anomaly_scores)

        result = self._normalise(raw)
        self._last_call_ts = time.time()
        self._last_result  = result

        if result.get("commands"):
            log.warning(
                "Gemma4 commandes : %s | %s",
                result["commands"],
                result.get("diagnostic", "")[:60],
            )
        else:
            log.info(
                "Gemma4 %s — %s",
                result.get("severity", "?"),
                result.get("diagnostic", "")[:80],
            )

        return result

    # ── Déclenchement ─────────────────────────────────────────────────────────

    def _should_call(self, data: dict, scores: dict) -> bool:
        now     = time.time()
        elapsed = now - self._last_call_ts

        if self._last_call_ts == 0:
            return True

        p, v = data.get("pump", 0), data.get("valve", 0)
        if p != self._last_pump or v != self._last_valve:
            self._last_pump, self._last_valve = p, v
            return True

        risk = scores.get("global_risk", 0)
        gas  = data.get("g", 0)

        if (risk >= AI_RISK_THRESHOLD or gas >= 250) and elapsed >= 5:
            return True

        return elapsed >= AI_ROUTINE_INTERVAL

    # ── Construction du prompt ────────────────────────────────────────────────

    def _build_prompt(self, data: dict, scores: dict) -> str:
        reg       = scores.get("regression", {})
        gas_alert = scores.get("gas_alert") or "Aucune"
        err       = data.get("err", 0)
        trends    = self._trends()

        lines = [
            f"=== MESURES CAPTEURS — {datetime.now().strftime('%Y-%m-%d %H:%M:%S')} ===",
            "",
            "NIVEAUX :",
            f"  R1 : {data.get('n1','?')}%  (prédit 30s : {reg.get('n1_in_30s','?')}%)",
            f"  R2 : {data.get('n2','?')}%  (prédit 30s : {reg.get('n2_in_30s','?')}%)",
            f"  Risque débordement : {'OUI ⚠' if reg.get('overflow_risk') else 'non'}",
            "",
            "TEMPÉRATURES :",
            f"  R1 : {data.get('t1','?')}°C   R2 : {data.get('t2','?')}°C",
            "",
            "ENVIRONNEMENT :",
            f"  Pression BMP280 : {data.get('p','?')} hPa",
            f"  Gaz MQ-4 : {data.get('g','?')} ADC  (WARN=250, DANGER=450)",
            f"  Alerte gaz : {gas_alert}",
            "",
            "ACTIONNEURS :",
            f"  Pompe : {'EN MARCHE' if data.get('pump') else 'ARRÊTÉE'}",
            f"  Vanne : {'OUVERTE'  if data.get('valve') else 'FERMÉE'}",
            "",
            "SCORES IA :",
            f"  Risque global : {scores.get('global_risk',0)}%",
            f"  Score anomalie : {scores.get('isolation_forest',0)}%",
            "",
            f"TENDANCES ({len(self._history)} mesures) :",
            trends,
        ]

        if err:
            lines += ["", f"⚠ ERREURS CAPTEURS (0x{err:02X}) :"]
            if err & 0x01: lines.append("  - HC-SR04 R1 défaillant")
            if err & 0x02: lines.append("  - HC-SR04 R2 défaillant")
            if err & 0x04: lines.append("  - DS18B20 R1 défaillant")
            if err & 0x08: lines.append("  - DS18B20 R2 défaillant")
            if err & 0x10: lines.append("  - BMP280 défaillant")

        lines.append("\nAnalyse et décide les commandes nécessaires.")
        return "\n".join(lines)

    def _trends(self) -> str:
        if len(self._history) < 3:
            return "  Historique insuffisant"
        recent  = list(self._history)[-min(5, len(self._history)):]
        elapsed = max(recent[-1]["ts"] - recent[0]["ts"], 1)
        dn1 = recent[-1]["n1"] - recent[0]["n1"]
        dn2 = recent[-1]["n2"] - recent[0]["n2"]
        dg  = recent[-1]["g"]  - recent[0]["g"]

        def fmt(v): return f"↑+{v:.1f}" if v > 1 else (f"↓{v:.1f}" if v < -1 else f"→{v:.1f}")
        return (
            f"  R1 : {fmt(dn1)}% | R2 : {fmt(dn2)}% | Gaz : {fmt(dg)} ADC"
            f"  (sur {elapsed:.0f}s)"
        )

    # ── Normalisation ──────────────────────────────────────────────────────────

    def _normalise(self, raw: dict) -> dict:
        valid_cmds = {"CMD:PUMP_ON", "CMD:PUMP_OFF", "CMD:VALVE_OPEN", "CMD:VALVE_CLOSE", "CMD:ESD"}
        cmds = [c for c in raw.get("commands", []) if c in valid_cmds]
        return {
            "commands":        cmds,
            "diagnostic":      raw.get("diagnostic",      "Analyse complétée"),
            "severity":        raw.get("severity",        "INFO"),
            "details":         raw.get("details",         ""),
            "predictions":     raw.get("predictions",     {}),
            "recommendations": raw.get("recommendations", []),
            "autonomie_h":     raw.get("autonomie_h",     -1),
            "source":          raw.get("source",          "gemma4"),
            "from_cache":      False,
            "timestamp":       datetime.now(timezone.utc).isoformat(),
        }

    # ── Fallback intelligent avec prédictions réelles ─────────────────────────

    def _fallback(self, data: dict, scores: dict) -> dict:
        risk  = scores.get("global_risk", 0)
        gas   = data.get("g",  0)
        n1    = data.get("n1", 0)
        n2    = data.get("n2", 0)
        pump  = data.get("pump", 0)
        valve = data.get("valve", 0)
        cmds: list[str] = []
        recs:  list[str] = []

        # ── Prédictions basées sur l'historique ──────────────────────────────
        predictions = self._compute_predictions(n1, n2, gas)

        # ── Détermination sévérité ──────────────────────────────────────────
        if gas >= 450:
            severity = "CRITIQUE"
            diag = f"FUITE GAZ CONFIRMÉE — MQ-4={gas} ADC dépasse seuil danger (450). Arrêt d'urgence obligatoire."
            cmds.append("CMD:ESD")
            recs.append("Évacuer la zone immédiatement")
            recs.append("Vérifier l'intégrité des conduites GNL")
        elif risk >= 85:
            severity = "CRITIQUE"
            diag = f"RISQUE SYSTÈME CRITIQUE {risk}% — Paramètres multiples hors normes. ESD déclenché."
            cmds.append("CMD:ESD")
            recs.append("Inspection immédiate de tous les capteurs")
        elif gas >= 250:
            severity = "DANGER"
            diag = f"Concentration méthane élevée ({gas} ADC). Seuil d'alerte dépassé. Surveillance renforcée."
            recs.append(f"Taux méthane actuel : {gas} ADC (seuil danger : 450)")
            recs.append("Vérifier ventilation de la zone")
        elif n2 >= 93:
            severity = "ATTENTION"
            diag = f"Réservoir R2 quasi-plein ({n2}%). Fermeture vanne préventive."
            cmds.append("CMD:VALVE_CLOSE")
            cmds.append("CMD:PUMP_OFF")
            recs.append("R2 atteint capacité maximale — arrêt transfert")
        elif n1 <= 12:
            severity = "ATTENTION"
            diag = f"Niveau R1 critique bas ({n1}%). Risque cavitation pompe — arrêt préventif."
            cmds.append("CMD:PUMP_OFF")
            recs.append("Remplir R1 avant de relancer la pompe (minimum 20%)")
        elif n1 >= 90 and n2 < 92 and not pump:
            severity = "ATTENTION"
            diag = f"R1 quasi-plein ({n1}%), R2 disponible ({n2}%). Démarrage transfert automatique."
            cmds.append("CMD:PUMP_ON")
            cmds.append("CMD:VALVE_OPEN")
            recs.append("Transfert automatique démarré")
        elif risk >= 60:
            severity = "ATTENTION"
            diag = f"Risque modéré détecté ({risk}%). R1={n1}% R2={n2}% Gaz={gas} ADC."
            recs.append("Surveiller l'évolution des niveaux sur les 10 prochaines minutes")
        else:
            severity = "INFO"
            preds = predictions
            t_plein = preds.get("temps_avant_plein_r2_min")
            t_vide  = preds.get("temps_avant_vide_r1_min")
            if t_plein and t_plein > 0:
                diag = f"Système nominal. R1={n1}% R2={n2}% Gaz={gas} ADC. R2 plein estimé dans {t_plein:.0f} min."
            elif t_vide and t_vide > 0:
                diag = f"Système nominal. R1={n1}% R2={n2}% Gaz={gas} ADC. R1 vide estimé dans {t_vide:.0f} min."
            else:
                diag = f"Système nominal. R1={n1}% R2={n2}% Gaz={gas} ADC. Risque global : {risk}%."
            recs.append("Continuer surveillance nominale")

        # Recommandations basées sur les prédictions
        t_plein = predictions.get("temps_avant_plein_r2_min")
        t_vide  = predictions.get("temps_avant_vide_r1_min")
        if t_plein and 0 < t_plein < 10:
            recs.insert(0, f"⚠ R2 sera plein dans {t_plein:.0f} min — prévoir fermeture vanne")
        if t_vide and 0 < t_vide < 10:
            recs.insert(0, f"⚠ R1 sera vide dans {t_vide:.0f} min — arrêter pompe bientôt")

        # Calcul autonomie (heures restantes de fonctionnement)
        autonomie = self._compute_autonomie(n1, n2, pump)

        return {
            "commands":        cmds,
            "diagnostic":      diag,
            "severity":        severity,
            "details":         (
                f"Analyse locale — Moteur règles + prédictions linéaires\n"
                f"Sources: AnomalyEngine (IF={scores.get('isolation_forest',0)}%), "
                f"tendances {len(self._history)} mesures\n"
                f"Pompe: {'EN MARCHE' if pump else 'ARRÊTÉE'} | "
                f"Vanne: {'OUVERTE' if valve else 'FERMÉE'} | "
                f"Pression: {data.get('p',0):.1f} hPa"
            ),
            "predictions":     predictions,
            "recommendations": recs,
            "autonomie_h":     autonomie,
            "source":          "fallback_intelligent",
            "from_cache":      False,
            "timestamp":       datetime.now(timezone.utc).isoformat(),
        }

    def _compute_predictions(self, n1: float, n2: float, gas: int) -> dict:
        """Calcule les prédictions basées sur les tendances de l'historique."""
        predictions = {
            "r1_5min":                 None,
            "r2_5min":                 None,
            "temps_avant_plein_r2_min": -1,
            "temps_avant_vide_r1_min":  -1,
            "tendance_gaz":            "stable",
        }
        if len(self._history) < 4:
            return predictions

        recent = list(self._history)[-min(10, len(self._history)):]
        elapsed_s = max(recent[-1]["ts"] - recent[0]["ts"], 1)

        # Taux de variation (% par seconde)
        dn1 = (recent[-1]["n1"] - recent[0]["n1"]) / elapsed_s
        dn2 = (recent[-1]["n2"] - recent[0]["n2"]) / elapsed_s
        dg  = (recent[-1]["g"]  - recent[0]["g"])  / elapsed_s

        # Prédiction dans 5 minutes (300s)
        n1_5 = max(0, min(100, n1 + dn1 * 300))
        n2_5 = max(0, min(100, n2 + dn2 * 300))
        predictions["r1_5min"] = round(n1_5, 1)
        predictions["r2_5min"] = round(n2_5, 1)

        # Temps avant R2 plein (seconds → minutes)
        if dn2 > 1e-4:
            t_plein_s = (95 - n2) / dn2
            predictions["temps_avant_plein_r2_min"] = round(t_plein_s / 60, 1) if t_plein_s > 0 else -1
        elif dn2 < -1e-4:
            predictions["temps_avant_plein_r2_min"] = -1

        # Temps avant R1 vide
        if dn1 < -1e-4:
            t_vide_s = (n1 - 10) / (-dn1)
            predictions["temps_avant_vide_r1_min"] = round(t_vide_s / 60, 1) if t_vide_s > 0 else -1

        # Tendance gaz
        if dg > 0.5:
            predictions["tendance_gaz"] = "hausse"
        elif dg < -0.5:
            predictions["tendance_gaz"] = "baisse"
        else:
            predictions["tendance_gaz"] = "stable"

        return predictions

    # ── Chatbot : question libre avec contexte capteurs ───────────────────────

    def chat(self, question: str, sensor_data: dict | None = None) -> str:
        """
        Répond à une question libre de l'opérateur/admin en français.
        Inclut le contexte capteurs si disponible.
        Retourne une chaîne de texte (pas de JSON).
        """
        ctx_lines = []
        if sensor_data:
            n1  = sensor_data.get("n1", "?")
            n2  = sensor_data.get("n2", "?")
            t1  = sensor_data.get("t1", "?")
            t2  = sensor_data.get("t2", "?")
            p   = sensor_data.get("p",  "?")
            g   = sensor_data.get("g",  "?")
            pump  = sensor_data.get("pump",  0)
            valve = sensor_data.get("valve", 0)
            ai    = sensor_data.get("ai", {})
            smart = sensor_data.get("smart_ai", {})
            ctx_lines = [
                "=== ÉTAT ACTUEL DU SYSTÈME ===",
                f"  R1={n1}%  R2={n2}%",
                f"  Températures: R1={t1}°C  R2={t2}°C",
                f"  Pression BMP280: {p} hPa",
                f"  Gaz MQ-4: {g} ADC",
                f"  Pompe: {'EN MARCHE' if pump else 'ARRÊTÉE'}  Vanne: {'OUVERTE' if valve else 'FERMÉE'}",
                f"  Risque global IA: {ai.get('global_risk', 0)}%",
                f"  Alerte gaz: {ai.get('gas_alert') or 'Aucune'}",
                f"  Dernier diagnostic: {smart.get('diagnostic', 'N/A')}",
                "",
            ]

        prompt = "\n".join([
            "Tu es un assistant expert en surveillance de réservoirs GNL (Gaz Naturel Liquéfié).",
            "Réponds en français, de façon concise et précise.",
            "Si la question concerne la sécurité, sois particulièrement prudent et factuel.",
            "",
        ] + ctx_lines + [
            f"Question de l'opérateur : {question}",
            "",
            "Réponds directement en texte clair (pas de JSON). Maximum 5 phrases.",
        ])

        if not self._gemma.available:
            self._gemma._available = self._gemma._check()

        if not self._gemma.available:
            return self._fallback_chat(question, sensor_data)

        try:
            full_prompt = (
                f"<start_of_turn>user\n{prompt}<end_of_turn>\n"
                f"<start_of_turn>model\n"
            )
            resp = requests.post(
                f"{GEMMA4_URL}/completion",
                json={
                    "prompt":      full_prompt,
                    "n_predict":   300,
                    "temperature": 0.3,
                    "top_p":       0.9,
                    "stop":        ["</s>", "<end_of_turn>", "<start_of_turn>"],
                    "stream":      False,
                },
                timeout=GEMMA4_TIMEOUT,
            )
            resp.raise_for_status()
            text = resp.json().get("content", "").strip()
            return text if text else self._fallback_chat(question, sensor_data)
        except Exception as e:
            log.warning("Gemma4 chat error : %s", e)
            self._gemma._available = False
            return self._fallback_chat(question, sensor_data)

    def _fallback_chat(self, question: str, sensor_data: dict | None) -> str:
        """Réponse de secours basée sur les règles quand Gemma4 est indisponible."""
        q = question.lower()
        sd = sensor_data or {}
        ai = sd.get("ai", {})
        smart = sd.get("smart_ai", {})

        if any(w in q for w in ["niveau", "réservoir", "r1", "r2", "tank"]):
            n1 = sd.get("n1", "?"); n2 = sd.get("n2", "?")
            return (
                f"Réservoir R1 : {n1}% | Réservoir R2 : {n2}%. "
                f"Seuil haut : {95}% (débordement) | Seuil bas : {10}% (cavitation pompe). "
                f"{'⚠ R1 critique bas !' if isinstance(n1, (int,float)) and n1 < 15 else ''} "
                f"{'⚠ R2 quasi-plein !' if isinstance(n2, (int,float)) and n2 > 88 else ''}"
            ).strip()

        if any(w in q for w in ["gaz", "méthane", "fuite", "mq-4", "mq4"]):
            g = sd.get("g", "?")
            alert = ai.get("gas_alert") or "Aucune"
            return (
                f"Concentration gaz MQ-4 : {g} ADC. "
                f"Seuil attention : 250 ADC | Seuil danger ESD : 450 ADC. "
                f"Alerte actuelle : {alert}."
            )

        if any(w in q for w in ["température", "temperature", "chaud", "froid", "ds18b20"]):
            t1 = sd.get("t1", "?"); t2 = sd.get("t2", "?")
            return f"Température R1 (DS18B20) : {t1}°C | Température R2 : {t2}°C. Plage normale : -10°C à 50°C."

        if any(w in q for w in ["risque", "anomalie", "ia", "isolation forest", "score"]):
            risk = ai.get("global_risk", 0)
            if_s  = ai.get("isolation_forest", 0)
            return (
                f"Risque global IA : {risk}% | Score Isolation Forest : {if_s}%. "
                f"{'⚠ RISQUE ÉLEVÉ' if risk >= 70 else 'Système nominal' if risk < 40 else 'Surveillance recommandée'}. "
                f"Déclenchement ESD automatique au-delà de 85%."
            )

        if any(w in q for w in ["pompe", "vanne", "actionneur", "commande"]):
            pump  = sd.get("pump",  0)
            valve = sd.get("valve", 0)
            return (
                f"Pompe : {'EN MARCHE ▶' if pump else 'ARRÊTÉE ■'} | "
                f"Vanne : {'OUVERTE ▲' if valve else 'FERMÉE ▼'}. "
                f"Commandes disponibles : PUMP_ON/OFF, VALVE_OPEN/CLOSE, ESD (admin uniquement)."
            )

        if any(w in q for w in ["pression", "bmp280", "bmp", "hpa"]):
            p = sd.get("p", "?")
            return f"Pression atmosphérique BMP280 : {p} hPa. Plage normale : 950–1070 hPa."

        if any(w in q for w in ["esd", "urgence", "arrêt d'urgence"]):
            return (
                "L'arrêt d'urgence ESD coupe immédiatement la pompe et ferme la vanne. "
                "Il se déclenche automatiquement si : gaz > 450 ADC (3 confirmations) ou risque global ≥ 85%. "
                "Redémarrage manuel requis après un ESD."
            )

        diag = smart.get("diagnostic", "")
        if diag:
            return f"Diagnostic IA actuel : {diag} (Source : {smart.get('source','?')})"

        return (
            "Je suis l'assistant IA du système GNL. Je surveille les niveaux R1/R2, "
            "la température, la pression, le gaz méthane et les actionneurs. "
            "Vous pouvez me demander des informations sur les capteurs, les alertes ou les commandes. "
            "(Gemma4 indisponible — réponses basées sur les règles de sécurité.)"
        )

    def _compute_autonomie(self, n1: float, n2: float, pump: int) -> float:
        """Estime l'autonomie en heures (combien de temps avant intervention requise)."""
        if not pump:
            return -1.0
        if len(self._history) < 4:
            return -1.0
        recent = list(self._history)[-min(10, len(self._history)):]
        elapsed_s = max(recent[-1]["ts"] - recent[0]["ts"], 1)
        dn1 = (recent[-1]["n1"] - recent[0]["n1"]) / elapsed_s
        if dn1 < -1e-4:
            t_s = (n1 - 10) / (-dn1)
            return round(t_s / 3600, 1)
        return -1.0
