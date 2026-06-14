#!/usr/bin/env python3
"""
mqtt_client.py — Client MQTT avec support TLS optionnel pour le système GNL

TLS activé via variables d'environnement :
  MQTT_TLS=true   → connexion chiffrée sur port 8883
  MQTT_TLS=false  → connexion plain sur port 1883 (Docker interne)

Broker : Mosquitto sur port 1883 (plain) ou 8883 (TLS)
Compatible : paho-mqtt 2.0.0 (CallbackAPIVersion.VERSION2 obligatoire)
"""

import json
import logging
import os
import ssl
import time
from datetime import datetime, timezone

import paho.mqtt.client as mqtt

log = logging.getLogger("gnl.mqtt")

# ── Configuration ──────────────────────────────────────────────────────────────
BROKER_HOST = os.environ.get("MQTT_HOST", "localhost")
BROKER_PORT = int(os.environ.get("MQTT_PORT", "1883"))
KEEPALIVE   = 60
CLIENT_ID   = "gnl_rpi4_edge"

MQTT_USER = os.environ.get("MQTT_USER_PUBLISHER", "nouar")
MQTT_PASS = os.environ.get("MQTT_PASS_PUBLISHER", "hamel")

# ── Configuration TLS ──────────────────────────────────────────────────────────
# MQTT_TLS=true  → TLS activé (port 8883, connexion externe RPi)
# MQTT_TLS=false → plain MQTT (port 1883, Docker interne)
MQTT_TLS  = os.environ.get("MQTT_TLS", "false").lower() == "true"
CERT_DIR  = os.environ.get("CERT_DIR", "/opt/gnl/certs")

# Chemins des certificats (générés par make setup-certs)
CA_CERT     = os.path.join(CERT_DIR, "ca.crt")
CLIENT_CERT = os.path.join(CERT_DIR, "client.crt")
CLIENT_KEY  = os.path.join(CERT_DIR, "client.key")

# Topics avec leur QoS par défaut
TOPICS = {
    "niveau_r1": ("gnl/niveau/r1",      1),
    "niveau_r2": ("gnl/niveau/r2",      1),
    "temp_r1":   ("gnl/temperature/r1", 1),
    "temp_r2":   ("gnl/temperature/r2", 1),
    "gaz":       ("gnl/gaz/mq4",        2),
    "pression":  ("gnl/pression",       0),
    "ia_score":  ("gnl/ia/score",       1),
    "alerte":    ("gnl/alerte",         2),
    "cmd_pompe": ("gnl/cmd/pompe",      2),
    "cmd_vanne": ("gnl/cmd/vanne",      2),
    "cmd_esd":   ("gnl/cmd/esd",        2),
}

RECONNECT_DELAY = 5


class GNLMQTTClient:
    """
    Client MQTT avec reconnexion automatique et support TLS optionnel.

    Modes :
      MQTT_TLS=false → port 1883 plain  (Docker interne, par défaut)
      MQTT_TLS=true  → port 8883 TLS    (connexion externe RPi physique)

    Utilise paho-mqtt 2.0.0 (CallbackAPIVersion.VERSION2).
    """

    def __init__(self) -> None:
        self._client = mqtt.Client(
            callback_api_version=mqtt.CallbackAPIVersion.VERSION2,
            client_id=CLIENT_ID,
            protocol=mqtt.MQTTv5,
        )
        self._connected = False
        self._setup_client()

        mode = f"TLS (port {BROKER_PORT})" if MQTT_TLS else f"plain (port {BROKER_PORT})"
        log.info(
            "GNLMQTTClient initialisé — broker=%s:%d mode=%s",
            BROKER_HOST, BROKER_PORT, mode,
        )

    def _setup_client(self) -> None:
        # Authentification username/password
        self._client.username_pw_set(MQTT_USER, MQTT_PASS)

        # ── TLS optionnel ──────────────────────────────────────────────────────
        if MQTT_TLS:
            self._setup_tls()

        # Callbacks
        self._client.on_connect    = self._on_connect
        self._client.on_disconnect = self._on_disconnect
        self._client.on_message    = self._on_message

        # Last Will Testament
        self._client.will_set(
            "gnl/status",
            json.dumps({"status": "offline", "node": CLIENT_ID}),
            qos=1,
            retain=True,
        )

    def _setup_tls(self) -> None:
        """Configure TLS avec les certificats générés par make setup-certs."""

        # Vérification existence des fichiers
        missing = []
        for f in [CA_CERT, CLIENT_CERT, CLIENT_KEY]:
            if not os.path.exists(f):
                missing.append(f)

        if missing:
            log.error(
                "TLS demandé mais certificats manquants : %s\n"
                "  → Lancer : make setup-certs",
                missing,
            )
            log.warning("Basculement en mode plain MQTT (port 1883)")
            return

        try:
            self._client.tls_set(
                ca_certs    = CA_CERT,
                certfile    = CLIENT_CERT,
                keyfile     = CLIENT_KEY,
                tls_version = ssl.PROTOCOL_TLS_CLIENT,
                cert_reqs   = ssl.CERT_REQUIRED,
            )
            # False = vérification stricte du nom d'hôte
            self._client.tls_insecure_set(False)
            log.info(
                "TLS configuré — ca=%s cert=%s",
                CA_CERT, CLIENT_CERT,
            )
        except Exception as exc:
            log.error("Erreur configuration TLS : %s", exc)
            log.warning("Basculement en mode plain MQTT")

    # ── Connexion / déconnexion ────────────────────────────────────────────────

    def connect(self) -> None:
        """Connexion bloquante avec retry jusqu'au succès."""
        while True:
            try:
                self._client.connect(BROKER_HOST, BROKER_PORT, KEEPALIVE)
                self._client.loop_start()

                mode = "TLS ✅" if MQTT_TLS else "plain"
                log.info(
                    "MQTT connecté : %s:%d [%s]",
                    BROKER_HOST, BROKER_PORT, mode,
                )
                time.sleep(0.5)
                self._publish_raw(
                    "gnl/status",
                    json.dumps({"status": "online", "node": CLIENT_ID}),
                    qos=1,
                    retain=True,
                )
                return
            except Exception as exc:
                log.warning(
                    "MQTT connexion échouée (%s) — retry dans %ds",
                    exc, RECONNECT_DELAY,
                )
                time.sleep(RECONNECT_DELAY)

    def disconnect(self) -> None:
        """Déconnexion propre."""
        self._publish_raw(
            "gnl/status",
            json.dumps({"status": "offline", "node": CLIENT_ID}),
            qos=1,
            retain=True,
        )
        self._client.loop_stop()
        self._client.disconnect()
        log.info("MQTT déconnecté proprement")

    # ── Publication des mesures ────────────────────────────────────────────────

    def publish_all(self, data: dict) -> None:
        """Publie toutes les mesures depuis le dict Arduino enrichi par l'IA."""
        ts = datetime.now(timezone.utc).isoformat()
        ai = data.get("ai", {})

        payloads: dict[str, tuple[dict, int]] = {
            "niveau_r1": (
                {"valeur": data.get("n1"), "unite": "%", "timestamp": ts},
                TOPICS["niveau_r1"][1],
            ),
            "niveau_r2": (
                {"valeur": data.get("n2"), "unite": "%", "timestamp": ts},
                TOPICS["niveau_r2"][1],
            ),
            "temp_r1": (
                {"valeur": data.get("t1"), "unite": "°C", "timestamp": ts},
                TOPICS["temp_r1"][1],
            ),
            "temp_r2": (
                {"valeur": data.get("t2"), "unite": "°C", "timestamp": ts},
                TOPICS["temp_r2"][1],
            ),
            "gaz": (
                {
                    "valeur":    data.get("g"),
                    "unite":     "ADC",
                    "niveau":    self._gas_level(data.get("g", 0)),
                    "timestamp": ts,
                },
                TOPICS["gaz"][1],
            ),
            "pression": (
                {"valeur": data.get("p"), "unite": "hPa", "timestamp": ts},
                TOPICS["pression"][1],
            ),
            "ia_score": (
                {
                    "isolation_forest": ai.get("isolation_forest", 0),
                    "global_risk":      ai.get("global_risk", 0),
                    "gas_alert":        ai.get("gas_alert"),
                    "overflow_risk":    ai.get("regression", {}).get("overflow_risk", False),
                    "timestamp":        ts,
                },
                TOPICS["ia_score"][1],
            ),
        }

        for key, (payload, qos) in payloads.items():
            topic = TOPICS[key][0]
            self._publish(topic, payload, qos)

        # Alertes conditionnelles
        gas_alert = ai.get("gas_alert")
        if gas_alert and gas_alert != "ATTENTION":
            self._publish_alert(gas_alert, data.get("g", 0), ts)

        global_risk = ai.get("global_risk", 0)
        if global_risk >= 70:
            self._publish_alert(f"RISQUE_{global_risk}", global_risk, ts)

    # ── Helpers de publication ─────────────────────────────────────────────────

    def _publish(self, topic: str, payload: dict, qos: int = 1) -> None:
        if not self._connected:
            log.debug("MQTT non connecté — message ignoré : %s", topic)
            return
        try:
            self._client.publish(topic, json.dumps(payload), qos=qos)
        except Exception as exc:
            log.warning("Erreur publication %s : %s", topic, exc)

    def _publish_raw(
        self, topic: str, payload: str, qos: int = 1, retain: bool = False
    ) -> None:
        try:
            self._client.publish(topic, payload, qos=qos, retain=retain)
        except Exception as exc:
            log.warning("Erreur publication raw %s : %s", topic, exc)

    def _publish_alert(
        self, alert_type: str, value: object, timestamp: str
    ) -> None:
        severity = "CRITIQUE" if "DANGER" in str(alert_type) else "ÉLEVÉ"
        payload = {
            "type":      alert_type,
            "severity":  severity,
            "valeur":    value,
            "timestamp": timestamp,
            "node":      CLIENT_ID,
        }
        topic, qos = TOPICS["alerte"]
        self._publish(topic, payload, qos)
        log.warning("ALERTE publiée : %s (valeur=%s)", alert_type, value)

    @staticmethod
    def _gas_level(gas: int) -> str:
        if gas < 250:
            return "OK"
        if gas < 450:
            return "ATTENTION"
        return "DANGER"

    # ── Callbacks MQTT ────────────────────────────────────────────────────────

    def _on_connect(
        self,
        client: mqtt.Client,
        userdata: object,
        connect_flags: mqtt.ConnectFlags,
        reason_code: mqtt.ReasonCode,
        properties: mqtt.Properties,
    ) -> None:
        if reason_code.is_failure:
            log.error("MQTT connexion refusée : %s", reason_code)
            return

        self._connected = True
        mode = "TLS ✅" if MQTT_TLS else "plain"
        log.info("MQTT connecté [%s] (reason_code=%s)", mode, reason_code)

        for cmd_key in ("cmd_pompe", "cmd_vanne", "cmd_esd"):
            topic, qos = TOPICS[cmd_key]
            client.subscribe(topic, qos=qos)
            log.info("Souscrit : %s (QoS %d)", topic, qos)

    def _on_disconnect(
        self,
        client: mqtt.Client,
        userdata: object,
        disconnect_flags: mqtt.DisconnectFlags,
        reason_code: mqtt.ReasonCode,
        properties: mqtt.Properties,
    ) -> None:
        self._connected = False
        log.warning(
            "MQTT déconnecté (reason_code=%s) — reconnexion automatique…",
            reason_code,
        )

    def _on_message(
        self,
        client: mqtt.Client,
        userdata: object,
        message: mqtt.MQTTMessage,
    ) -> None:
        topic   = message.topic
        payload = message.payload.decode("utf-8", errors="replace").strip()
        log.info("Commande reçue [%s] : %s", topic, payload)
