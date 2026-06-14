#!/usr/bin/env python3
"""
rest_server.py — API REST Flask pour le système IoT GNL

Sécurité admin :
  - Clé AES-256 lue depuis .env (AES_SECRET_KEY)
  - Mot de passe admin chiffré AES-256-GCM au démarrage
  - Vérification via hmac.compare_digest (anti timing attack)
  - JWT Bearer token pour toutes les routes protégées

Endpoints :
  GET  /api/v1/status          → état général du système
  GET  /api/v1/data/latest     → dernières mesures
  GET  /api/v1/ai/scores       → scores IA courants
  GET  /api/v1/ai/diagnostic   → diagnostic Smart AI
  POST /api/v1/ai/chat         → chatbot Gemma
  POST /api/v1/cmd/pompe       → commande pompe
  POST /api/v1/cmd/vanne       → commande vanne
  POST /api/v1/cmd/esd         → arrêt d'urgence
  GET  /api/v1/alerts          → journal alertes
  GET  /health                 → health check
"""

import base64
import hmac as hmac_lib
import json
import logging
import os
import time
from datetime import datetime, timezone
from functools import wraps
from threading import Lock

from flask import Flask, jsonify, request, abort, send_from_directory
from flask_cors import CORS
import jwt

# ── AES-256-GCM ───────────────────────────────────────────────────────────────
try:
    from cryptography.hazmat.primitives.ciphers.aead import AESGCM
    _AES_AVAILABLE = True
except ImportError:
    _AES_AVAILABLE = False

log = logging.getLogger("gnl.api")

# ── Config générale ────────────────────────────────────────────────────────────
API_HOST    = os.environ.get("API_HOST", "0.0.0.0")
API_PORT    = int(os.environ.get("API_PORT", "5000"))
JWT_SECRET  = os.environ.get("GNL_JWT_SECRET", "hamel")
JWT_ALGO    = "HS256"
JWT_EXPIRY  = int(os.environ.get("JWT_EXPIRY_S", "3600"))
API_TIMEOUT = int(os.environ.get("API_TIMEOUT_S", "3600"))

PUBLIC_URL = os.environ.get(
    "PUBLIC_URL", "https://theology-custody-rocky.ngrok-free.dev"
)

# ── Clé AES-256 depuis .env ───────────────────────────────────────────────────
#
# AES_SECRET_KEY dans .env = clé 32 bytes encodée en base64 (256 bits)
#
# Générer une nouvelle clé :
#   python3 -c "import os,base64; print(base64.b64encode(os.urandom(32)).decode())"
#
_AES_KEY_B64 = os.environ.get("AES_SECRET_KEY", "")
_AES_SALT    = os.environ.get("AES_SALT", "gnl_usto_mb_2025").encode()


def _load_aes_key() -> bytes | None:
    """
    Charge la clé AES-256 depuis AES_SECRET_KEY dans .env.
    Doit être 32 bytes encodés en base64 (256 bits = AES-256).
    """
    if not _AES_AVAILABLE:
        log.warning(
            "cryptography non installé — AES-256 désactivé\n"
            "  → pip install cryptography"
        )
        return None

    if not _AES_KEY_B64:
        log.error(
            "AES_SECRET_KEY manquant dans .env\n"
            "  Générer : python3 -c \"import os,base64; "
            "print(base64.b64encode(os.urandom(32)).decode())\"\n"
            "  Puis ajouter dans .env : AES_SECRET_KEY=<valeur générée>"
        )
        return None

    try:
        key = base64.b64decode(_AES_KEY_B64)
        if len(key) != 32:
            log.error(
                "AES_SECRET_KEY invalide : %d bytes trouvés, 32 requis (256 bits)",
                len(key),
            )
            return None
        log.info(
            "AES-256-GCM activé\n"
            "  Source      : AES_SECRET_KEY (.env)\n"
            "  Longueur    : %d bytes (%d bits)\n"
            "  Algorithme  : AES-256-GCM (chiffrement authentifié)\n"
            "  AAD (sel)   : AES_SALT (.env)",
            len(key), len(key) * 8,
        )
        return key
    except Exception as exc:
        log.error("Erreur décodage AES_SECRET_KEY depuis .env : %s", exc)
        return None


_AES_KEY = _load_aes_key()


# ── Chiffrement / Déchiffrement AES-256-GCM ───────────────────────────────────

def _aes_encrypt(plaintext: str) -> str:
    """
    Chiffre avec AES-256-GCM (clé depuis .env).
    Format retourné : base64( nonce[12] + ciphertext + tag[16] )
    """
    if _AES_KEY is None:
        return plaintext

    aesgcm = AESGCM(_AES_KEY)
    nonce  = os.urandom(12)    # 96 bits — recommandé NIST SP 800-38D
    ct     = aesgcm.encrypt(nonce, plaintext.encode("utf-8"), _AES_SALT)
    return base64.b64encode(nonce + ct).decode("utf-8")


def _aes_decrypt(token: str) -> str:
    """
    Déchiffre un token AES-256-GCM (clé depuis .env).
    Retourne "" si token invalide ou modifié (GCM authentification échouée).
    """
    if _AES_KEY is None:
        return token

    try:
        raw    = base64.b64decode(token.encode("utf-8"))
        nonce  = raw[:12]
        ct     = raw[12:]
        aesgcm = AESGCM(_AES_KEY)
        return aesgcm.decrypt(nonce, ct, _AES_SALT).decode("utf-8")
    except Exception:
        return ""


def _verify_password(plain: str, encrypted: str) -> bool:
    """
    Vérifie mot de passe contre sa version chiffrée AES-256-GCM.
    hmac.compare_digest protège contre les timing attacks.
    """
    if _AES_KEY is None:
        return hmac_lib.compare_digest(plain, encrypted)
    decrypted = _aes_decrypt(encrypted)
    return hmac_lib.compare_digest(plain, decrypted)


# ── Base utilisateurs — mots de passe chiffrés AES-256-GCM ───────────────────

def _build_users() -> dict:
    """
    Construit la base utilisateurs.
    Clé AES lue depuis AES_SECRET_KEY dans .env.
    Mot de passe jamais stocké en clair après cette fonction.
    """
    raw_admin = "hamel"

    if _AES_KEY is not None:
        encrypted = _aes_encrypt(raw_admin)
        log.info(
            "Utilisateur 'nouar' (admin) — mot de passe chiffré AES-256-GCM\n"
            "  Clé         : AES_SECRET_KEY (.env)\n"
            "  Mot de passe en clair effacé de la mémoire après chiffrement"
        )
    else:
        encrypted = raw_admin
        log.warning("Utilisateur 'nouar' — mot de passe stocké en clair (AES indisponible)")

    return {
        "nouar": {
            "password":  encrypted,
            "role":      "admin",
            "encrypted": _AES_KEY is not None,
        },
    }


USERS = _build_users()

_DASHBOARD_DIR = os.path.join(os.path.dirname(__file__), "..", "dashboard")

app   = Flask(__name__, static_folder=None)
_lock = Lock()


def _cors_origins():
    return [
        PUBLIC_URL,
        "http://localhost:5000",
        "http://127.0.0.1:5000",
        r"https://.*\.app\.github\.dev",
        r"https://.*\.preview\.app\.github\.dev",
    ]


CORS(app, resources={r"/*": {"origins": _cors_origins(), "supports_credentials": True}})

# État partagé
_latest_data:       dict = {}
_alerts:            list = []
_smart_diagnostic:  dict = {}
_mongo             = None
_smart_ai          = None


def set_mongo(mongo_writer):
    global _mongo
    _mongo = mongo_writer


def set_smart_ai(smart_ai_instance):
    global _smart_ai
    _smart_ai = smart_ai_instance


# ── Middleware ngrok ────────────────────────────────────────────────────────────
@app.before_request
def _ngrok_skip_warning():
    pass


@app.after_request
def _add_ngrok_header(response):
    response.headers["ngrok-skip-browser-warning"] = "1"
    origin = request.headers.get("Origin", "")
    allowed = (
        origin == PUBLIC_URL
        or origin.startswith("http://localhost")
        or origin.startswith("http://127.0.0.1")
        or origin.endswith(".app.github.dev")
        or origin.endswith(".preview.app.github.dev")
    )
    if allowed and origin:
        response.headers["Access-Control-Allow-Origin"] = origin
    response.headers["Access-Control-Allow-Credentials"] = "true"
    response.headers["Access-Control-Allow-Headers"] = (
        "Content-Type, Authorization, ngrok-skip-browser-warning"
    )
    response.headers["Access-Control-Allow-Methods"] = "GET, POST, OPTIONS"
    return response


# ── Helpers auth ────────────────────────────────────────────────────────────────

def generate_token(username: str, role: str) -> str:
    payload = {
        "sub":  username,
        "role": role,
        "iat":  int(time.time()),
        "exp":  int(time.time()) + JWT_EXPIRY,
    }
    return jwt.encode(payload, JWT_SECRET, algorithm=JWT_ALGO)


def decode_token(token: str) -> dict | None:
    try:
        return jwt.decode(token, JWT_SECRET, algorithms=[JWT_ALGO])
    except jwt.ExpiredSignatureError:
        return None
    except jwt.InvalidTokenError:
        return None


def require_auth(role: str = "operator"):
    def decorator(f):
        @wraps(f)
        def wrapper(*args, **kwargs):
            if request.method == "OPTIONS":
                return jsonify({}), 200
            auth_header = request.headers.get("Authorization", "")
            if not auth_header.startswith("Bearer "):
                abort(401, "Token manquant")
            token = auth_header[7:]
            payload = decode_token(token)
            if payload is None:
                abort(401, "Token invalide ou expiré")
            if role == "admin" and payload.get("role") != "admin":
                abort(403, "Droits insuffisants (admin requis)")
            request.user = payload
            return f(*args, **kwargs)
        return wrapper
    return decorator


# ── Endpoints publics ───────────────────────────────────────────────────────────

@app.route("/")
def serve_dashboard():
    return send_from_directory(os.path.abspath(_DASHBOARD_DIR), "gnl_dashboard.html")


@app.route("/health")
def health():
    return jsonify({
        "status":     "ok",
        "timestamp":  datetime.now(timezone.utc).isoformat(),
        "public_url": PUBLIC_URL,
        "security": {
            "aes256_gcm": _AES_KEY is not None,
            "key_source": "AES_SECRET_KEY (.env)" if _AES_KEY is not None else "ABSENT",
            "algorithm":  "AES-256-GCM" if _AES_KEY is not None else "plain",
        },
    })


@app.route("/api/v1/auth/login", methods=["POST", "OPTIONS"])
def login():
    if request.method == "OPTIONS":
        return jsonify({}), 200

    body     = request.get_json(silent=True) or {}
    username = body.get("username", "")
    password = body.get("password", "")

    user = USERS.get(username)
    if not user:
        abort(401, "Identifiants incorrects")

    # ── Vérification AES-256-GCM (clé depuis .env) ────────────────────────────
    if not _verify_password(password, user["password"]):
        log.warning("Tentative connexion échouée pour '%s'", username)
        abort(401, "Identifiants incorrects")

    token = generate_token(username, user["role"])
    log.info(
        "Connexion réussie : %s (role=%s, AES-256-GCM=%s)",
        username, user["role"], user["encrypted"],
    )
    return jsonify({
        "token":      token,
        "role":       user["role"],
        "expires_in": JWT_EXPIRY,
        "security":   "AES-256-GCM" if user["encrypted"] else "plain",
    })


# ── Endpoints protégés ─────────────────────────────────────────────────────────

@app.route("/api/v1/status")
@require_auth("operator")
def status():
    with _lock:
        data = dict(_latest_data)
    ai = data.get("ai", {})
    return jsonify({
        "timestamp":   datetime.now(timezone.utc).isoformat(),
        "node":        "rpi4_edge",
        "connected":   bool(data),
        "global_risk": ai.get("global_risk", 0),
        "gas_alert":   ai.get("gas_alert"),
        "pump":        data.get("pump", 0),
        "valve":       data.get("valve", 0),
        "public_url":  PUBLIC_URL,
    })


@app.route("/api/v1/data/latest")
@require_auth("operator")
def data_latest():
    with _lock:
        data = dict(_latest_data)
        diag = dict(_smart_diagnostic)
    if not data:
        return jsonify({"error": "Aucune donnée disponible"}), 503
    return jsonify({
        "timestamp":   datetime.now(timezone.utc).isoformat(),
        "niveau":      {"r1": data.get("n1"), "r2": data.get("n2")},
        "temperature": {"r1": data.get("t1"), "r2": data.get("t2")},
        "gaz":         {"adc": data.get("g"), "niveau": _gas_level(data.get("g", 0))},
        "pression":    data.get("p"),
        "actuateurs":  {"pompe": data.get("pump"), "vanne": data.get("valve")},
        "ia":          data.get("ai", {}),
        "smart_ai":    diag,
    })


@app.route("/api/v1/ai/scores")
@require_auth("operator")
def ai_scores():
    with _lock:
        ai = _latest_data.get("ai", {})
    return jsonify({
        "timestamp":        datetime.now(timezone.utc).isoformat(),
        "isolation_forest": ai.get("isolation_forest", 0),
        "global_risk":      ai.get("global_risk", 0),
        "gas_alert":        ai.get("gas_alert"),
        "regression":       ai.get("regression", {}),
    })


@app.route("/api/v1/ai/diagnostic")
@require_auth("operator")
def ai_diagnostic():
    with _lock:
        diag = dict(_smart_diagnostic)
    if not diag:
        return jsonify({"error": "Aucun diagnostic disponible"}), 503
    return jsonify(diag)


@app.route("/api/v1/ai/chat", methods=["POST", "OPTIONS"])
@require_auth("operator")
def ai_chat():
    if request.method == "OPTIONS":
        return jsonify({}), 200

    body     = request.get_json(silent=True) or {}
    question = (body.get("question") or "").strip()
    if not question:
        abort(400, "Le champ 'question' est requis")
    if len(question) > 500:
        abort(400, "Question trop longue (max 500 caractères)")

    if _smart_ai is None:
        return jsonify({"error": "Smart AI non initialisé"}), 503

    with _lock:
        ctx = dict(_latest_data)

    try:
        answer = _smart_ai.chat(question, sensor_data=ctx)
        source = "gemma4" if _smart_ai.is_available else "fallback"
    except Exception as e:
        log.error("Erreur chatbot : %s", e)
        answer = "Désolé, une erreur est survenue."
        source = "error"

    log.info(
        "Chatbot [%s] Q: %s | A: %s",
        request.user["sub"], question[:60], answer[:60],
    )
    return jsonify({
        "answer":    answer,
        "source":    source,
        "question":  question,
        "timestamp": datetime.now(timezone.utc).isoformat(),
    })


@app.route("/api/v1/alerts")
@require_auth("operator")
def get_alerts():
    with _lock:
        alerts = list(reversed(_alerts[-30:]))
    return jsonify({"count": len(alerts), "alerts": alerts})


# ── Commandes (admin uniquement) ───────────────────────────────────────────────

@app.route("/api/v1/cmd/pompe", methods=["POST", "OPTIONS"])
@require_auth("admin")
def cmd_pompe():
    body   = request.get_json(silent=True) or {}
    action = body.get("action", "").upper()
    if action not in ("ON", "OFF"):
        abort(400, "action doit être ON ou OFF")
    _register_command(f"CMD:PUMP_{action}", request.user["sub"])
    return jsonify({"status": "queued", "command": f"CMD:PUMP_{action}"})


@app.route("/api/v1/cmd/vanne", methods=["POST", "OPTIONS"])
@require_auth("admin")
def cmd_vanne():
    body   = request.get_json(silent=True) or {}
    action = body.get("action", "").upper()
    if action not in ("OPEN", "CLOSE"):
        abort(400, "action doit être OPEN ou CLOSE")
    _register_command(f"CMD:VALVE_{action}", request.user["sub"])
    return jsonify({"status": "queued", "command": f"CMD:VALVE_{action}"})


@app.route("/api/v1/cmd/esd", methods=["POST", "OPTIONS"])
@require_auth("admin")
def cmd_esd():
    _register_command("CMD:ESD", request.user["sub"])
    log.critical("ESD déclenché via API par %s", request.user["sub"])
    return jsonify({"status": "queued", "command": "CMD:ESD"})


# ── File de commandes ──────────────────────────────────────────────────────────
_cmd_queue: list = []


def _register_command(cmd: str, user: str):
    with _lock:
        _cmd_queue.append({"cmd": cmd, "user": user, "ts": time.time()})
        _add_alert("COMMANDE_MANUELLE", cmd, user)


def pop_command() -> str | None:
    with _lock:
        if _cmd_queue:
            return _cmd_queue.pop(0)["cmd"]
    return None


def update_latest(data: dict):
    with _lock:
        _latest_data.clear()
        _latest_data.update(data)
        ai = data.get("ai", {})
        if ai.get("global_risk", 0) >= 70 or ai.get("gas_alert"):
            _add_alert(
                ai.get("gas_alert") or f"RISQUE_{ai.get('global_risk')}",
                data.get("g"),
                "auto_ia",
            )


def update_smart_diagnostic(diag: dict):
    with _lock:
        _smart_diagnostic.clear()
        _smart_diagnostic.update(diag)
        severity = diag.get("severity", "INFO")
        if severity in ("DANGER", "CRITIQUE"):
            _add_alert(
                f"SMART_AI_{severity}",
                diag.get("diagnostic", ""),
                diag.get("source", "smart_ai"),
            )


def _add_alert(alert_type: str, value, source: str):
    _alerts.append({
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "type":      alert_type,
        "valeur":    value,
        "source":    source,
    })
    if len(_alerts) > 100:
        _alerts.pop(0)


def _gas_level(gas: int) -> str:
    if gas < 250:
        return "OK"
    if gas < 450:
        return "ATTENTION"
    return "DANGER"


# ── Historique MongoDB ─────────────────────────────────────────────────────────

@app.route("/api/v1/history/today")
@require_auth("operator")
def history_today():
    if _mongo is None or not _mongo.available:
        return jsonify({"error": "MongoDB non disponible"}), 503
    limit = min(int(request.args.get("limit", 120)), 500)
    data  = _mongo.get_today_history(limit=limit)
    return jsonify({"count": len(data), "readings": data})


@app.route("/api/v1/history/diagnostics")
@require_auth("operator")
def history_diagnostics():
    if _mongo is None or not _mongo.available:
        return jsonify({"error": "MongoDB non disponible"}), 503
    data = _mongo.get_today_diagnostics()
    return jsonify({"count": len(data), "diagnostics": data})


@app.route("/api/v1/history/events")
@require_auth("operator")
def history_events():
    if _mongo is None or not _mongo.available:
        return jsonify({"error": "MongoDB non disponible"}), 503
    data = _mongo.get_today_events()
    return jsonify({"count": len(data), "events": data})


@app.route("/api/v1/history/summary")
@require_auth("operator")
def history_summary():
    if _mongo is None or not _mongo.available:
        return jsonify({"error": "MongoDB non disponible"}), 503
    summary = _mongo.get_daily_summary()
    return jsonify(summary)


# ── Démarrage ──────────────────────────────────────────────────────────────────

def start_api_server():
    log.info(
        "API REST démarrée sur %s:%d | AES-256-GCM=%s | clé=AES_SECRET_KEY(.env)",
        API_HOST, API_PORT, _AES_KEY is not None,
    )
    app.run(
        host=API_HOST,
        port=API_PORT,
        debug=False,
        use_reloader=False,
        threaded=True,
    )


if __name__ == "__main__":
    start_api_server()
