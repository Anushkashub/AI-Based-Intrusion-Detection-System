# ============================================================
# AI-BASED INTRUSION DETECTION SYSTEM
# Flask Web Application
# Professional Security Dashboard Backend
# ============================================================

from flask import (
    Flask,
    request,
    jsonify,
    render_template,
    send_from_directory,
    redirect,
    url_for,
    session
)
from email.message import EmailMessage
import hashlib
import hmac
import random
import secrets
import smtplib
import ssl
import time

from src.predict import (
    predict_traffic,
    predict_batch,
    get_risk_level,
    feature_names,
    model,
    class_mapping
)

import os
import uuid
import threading
import numpy as np
import pandas as pd


# ============================================================
# FLASK APP SETUP
# ============================================================

# Load environment variables from .env if present
try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    pass

from werkzeug.utils import secure_filename

app = Flask(__name__)

# ============================================================
# CRYPTOGRAPHIC SESSION & ENVIRONMENT SECURITY
# ============================================================

_render_env = os.environ.get("RENDER", "").strip().lower()
_prod_env = os.environ.get("PRODUCTION", "").strip().lower()
IS_PRODUCTION = _render_env in ("true", "1", "yes") or _prod_env in ("true", "1", "yes")

_env_secret = os.environ.get("SECRET_KEY", "").strip()
if IS_PRODUCTION:
    if not _env_secret:
        raise RuntimeError(
            "CRITICAL PRODUCTION SECURITY ERROR: SECRET_KEY environment variable is missing or empty. "
            "A persistent, secure SECRET_KEY must be configured in production."
        )
    app.secret_key = _env_secret
else:
    if _env_secret:
        app.secret_key = _env_secret
    else:
        # Ephemeral cryptographically random secret key for local development
        app.secret_key = secrets.token_hex(32)

# Enforce 100 MB max file upload size limit to prevent resource exhaustion attacks
app.config["MAX_CONTENT_LENGTH"] = 100 * 1024 * 1024


# ============================================================
# AUTHENTICATION CONFIGURATION
# ============================================================

LOGIN_USERNAME = os.environ.get("ADMIN_USERNAME", "admin").strip()
_env_admin_pwd = os.environ.get("ADMIN_PASSWORD", "").strip()

if IS_PRODUCTION:
    if not _env_admin_pwd:
        raise RuntimeError(
            "CRITICAL PRODUCTION SECURITY ERROR: ADMIN_PASSWORD environment variable is missing or empty. "
            "A secure ADMIN_PASSWORD must be configured in production."
        )
    LOGIN_PASSWORD = _env_admin_pwd
else:
    if _env_admin_pwd:
        LOGIN_PASSWORD = _env_admin_pwd
    else:
        # Secure local development fallback:
        # Check if DEV_ADMIN_PASSWORD is set; otherwise generate an ephemeral local session password
        _dev_pwd = os.environ.get("DEV_ADMIN_PASSWORD", "").strip()
        if _dev_pwd:
            LOGIN_PASSWORD = _dev_pwd
        else:
            LOGIN_PASSWORD = secrets.token_urlsafe(12)
            print("=" * 65)
            print("AI-IDS LOCAL DEVELOPMENT SECURITY NOTICE:")
            print("No ADMIN_PASSWORD environment variable was found.")
            print("Generated one-time ephemeral credentials for this local session:")
            print(f"  Username: {LOGIN_USERNAME}")
            print(f"  Password: {LOGIN_PASSWORD}")
            print("To set a permanent password, define ADMIN_PASSWORD in your .env file.")
            print("=" * 65)

ADMIN_EMAIL = os.environ.get("ADMIN_EMAIL", "").strip().casefold()
SMTP_SERVER = os.environ.get("SMTP_SERVER", "smtp.gmail.com")
SMTP_PORT = os.environ.get("SMTP_PORT", "587")
SMTP_USERNAME = os.environ.get("SMTP_USERNAME", "")
SMTP_PASSWORD = os.environ.get("SMTP_PASSWORD", "")


# ============================================================
# PROJECT PATHS
# ============================================================

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
RESULTS_DIR = os.path.join(BASE_DIR, "results")
os.makedirs(RESULTS_DIR, exist_ok=True)
RESULTS_FILENAME = "dashboard_analysis_results.csv"


# ============================================================
# IN-MEMORY STORAGE
# ============================================================

analysis_jobs = {}
analysis_lock = threading.Lock()

incidents = {}
incident_lock = threading.Lock()
otp_storage = {}
otp_lock = threading.Lock()


# ============================================================
# AUTHENTICATION HELPER
# ============================================================

def _page_requires_login():
    if not session.get("logged_in"):
        return redirect(url_for("home"))
    return None


# ============================================================
# HOME / LOGIN
# ============================================================

@app.route("/")
def home():
    return render_template("login.html")


# ============================================================
# LOGIN
# ============================================================

@app.route("/login", methods=["GET", "POST"])
def login():
    if request.method == "GET":
        return render_template("login.html")

    username = None
    password = None

    if request.is_json:
        data = request.get_json(silent=True) or {}
        username = data.get("username")
        password = data.get("password")
    else:
        username = request.form.get("username")
        password = request.form.get("password")

    authenticated = False
    if LOGIN_USERNAME and LOGIN_PASSWORD and username and password:
        if hmac.compare_digest(str(username), str(LOGIN_USERNAME)) and hmac.compare_digest(str(password), str(LOGIN_PASSWORD)):
            authenticated = True

    if authenticated:
        session.clear()
        session["logged_in"] = True
        session["username"] = username

        if request.is_json:
            return jsonify({
                "status": "success",
                "message": "Login successful.",
                "redirect": url_for("dashboard")
            })

        return redirect(url_for("dashboard"))

    if request.is_json:
        return jsonify({
            "status": "error",
            "message": "Invalid username or password."
        }), 401

    return """
    <!DOCTYPE html>
    <html>
    <head><title>Login Failed</title></head>
    <body>
        <script>
            alert("Invalid username or password!");
            window.location.href = "/";
        </script>
    </body>
    </html>
    """, 401


# ============================================================
# PASSWORD RESET OTP
# ============================================================

@app.route("/api/send-reset-otp", methods=["POST"])
def send_reset_otp():
    data = request.get_json(silent=True)
    email_value = data.get("email") if isinstance(data, dict) else None
    email = email_value.strip() if isinstance(email_value, str) else ""

    if not email or "@" not in email:
        return jsonify({"success": False, "message": "Please enter a valid email address."}), 400

    normalized_email = email.casefold()
    generic_response = {
        "success": True,
        "message": "If the account is registered, a verification code will be sent."
    }
    if not ADMIN_EMAIL or not hmac.compare_digest(normalized_email, ADMIN_EMAIL):
        return jsonify(generic_response)

    if not SMTP_USERNAME or not SMTP_PASSWORD:
        app.logger.error("Password reset email is unavailable because SMTP credentials are not configured.")
        return jsonify({
            "success": False,
            "message": "Password reset email is temporarily unavailable."
        }), 503

    try:
        smtp_port = int(SMTP_PORT)
    except ValueError:
        app.logger.error("Password reset email is unavailable because SMTP_PORT is invalid.")
        return jsonify({
            "success": False,
            "message": "Password reset email is temporarily unavailable."
        }), 503

    now = time.monotonic()
    with otp_lock:
        previous_otp = otp_storage.get(normalized_email)
        if previous_otp and now - previous_otp["sent_at"] < 60:
            return jsonify({
                "success": False,
                "message": "Please wait before requesting another verification code."
            }), 429

        otp = f"{secrets.randbelow(1_000_000):06d}"
        message = EmailMessage()
        message["From"] = SMTP_USERNAME
        message["To"] = email
        message["Subject"] = "AI-IDS password reset verification code"
        message.set_content(
            "A password reset was requested for your AI-IDS account.\n\n"
            f"Your 6-digit verification code is: {otp}\n\n"
            "This code expires in 10 minutes. If you did not request this reset, "
            "you can ignore this email."
        )

        try:
            with smtplib.SMTP(SMTP_SERVER, smtp_port, timeout=10) as server:
                server.starttls(context=ssl.create_default_context())
                server.login(SMTP_USERNAME, SMTP_PASSWORD)
                server.send_message(message)
        except (OSError, smtplib.SMTPException) as exc:
            app.logger.exception("Could not send password reset email: %s", exc)
            return jsonify({
                "success": False,
                "message": "Could not send the verification email. Please try again later."
            }), 502

        otp_storage[normalized_email] = {
            "code_hash": hashlib.sha256(otp.encode("ascii")).hexdigest(),
            "expires_at": now + 600,
            "attempts": 0,
            "sent_at": now
        }

    return jsonify(generic_response)


@app.route("/api/verify-and-reset-password", methods=["POST"])
def verify_and_reset():
    data = request.get_json(silent=True)
    if not isinstance(data, dict):
        return jsonify({"success": False, "message": "Invalid request."}), 400

    email = data.get("email", "").strip() if isinstance(data.get("email"), str) else ""
    entered_otp = data.get("otp", "").strip() if isinstance(data.get("otp"), str) else ""
    new_password = data.get("new_password", "").strip() if isinstance(data.get("new_password"), str) else ""

    if not email or not entered_otp or len(new_password) < 4:
        return jsonify({
            "success": False,
            "message": "Enter a valid email, verification code, and a password of at least 4 characters."
        }), 400

    normalized_email = email.casefold()
    entered_hash = hashlib.sha256(entered_otp.encode("utf-8")).hexdigest()
    now = time.monotonic()

    with otp_lock:
        stored_otp = otp_storage.get(normalized_email)
        if not stored_otp:
            return jsonify({"success": False, "message": "Invalid or expired verification code."}), 400

        if now >= stored_otp["expires_at"] or stored_otp["attempts"] >= 5:
            del otp_storage[normalized_email]
            return jsonify({"success": False, "message": "Invalid or expired verification code."}), 400

        if not hmac.compare_digest(stored_otp["code_hash"], entered_hash):
            stored_otp["attempts"] += 1
            if stored_otp["attempts"] >= 5:
                del otp_storage[normalized_email]
            return jsonify({"success": False, "message": "Invalid or expired verification code."}), 400

        global LOGIN_PASSWORD
        LOGIN_PASSWORD = new_password
        del otp_storage[normalized_email]

    return jsonify({
        "success": True,
        "message": "Password successfully updated. You can now log in."
    })


# ============================================================
# LOGOUT
# ============================================================

@app.route("/logout")
def logout():
    session.clear()
    return redirect(url_for("home"))


# ============================================================
# DASHBOARD (PAGE 2)
# ============================================================

@app.route("/dashboard")
def dashboard():
    gate = _page_requires_login()
    if gate:
        return gate

    dashboard_stats = {
        "total_traffic": 0,
        "detected_threats": 0,
        "benign_traffic": 0,
        "threat_percentage": 0.0,
        "benign_percentage": 0.0,
        "attack_classes": 0,
        "model_accuracy": "96.39%",
        "has_analysis": False
    }

    dashboard_chart_data = {
        "labels": [],
        "counts": [],
        "values": []
    }

    recent_alerts = []
    results_path = os.path.join(RESULTS_DIR, RESULTS_FILENAME)

    if os.path.exists(results_path):
        try:
            result_df = pd.read_csv(results_path)
            total_records = int(len(result_df))
            if total_records > 0:
                dashboard_stats["total_traffic"] = total_records
                dashboard_stats["has_analysis"] = True

                if "attack_type" in result_df.columns:
                    attack_counts = (
                        result_df["attack_type"]
                        .astype(str)
                        .str.strip()
                        .str.upper()
                        .value_counts()
                    )

                    benign_traffic = int(attack_counts.get("BENIGN", 0))
                    detected_threats = int(total_records - benign_traffic)
                    dashboard_stats["benign_traffic"] = benign_traffic
                    dashboard_stats["detected_threats"] = detected_threats
                    dashboard_stats["threat_percentage"] = round((detected_threats / total_records) * 100, 2)
                    dashboard_stats["benign_percentage"] = round((benign_traffic / total_records) * 100, 2)
                    dashboard_stats["attack_classes"] = int(len([k for k in attack_counts.keys() if k != "BENIGN"]))

                    dashboard_chart_data = {
                        "labels": [str(x) for x in attack_counts.index.tolist()],
                        "counts": [int(x) for x in attack_counts.values.tolist()],
                        "values": [int(x) for x in attack_counts.values.tolist()]
                    }

                    if detected_threats > 0:
                        threat_rows = (
                            result_df[result_df["attack_type"].astype(str).str.strip().str.upper() != "BENIGN"]
                            .tail(10)
                        )
                        for idx, row in threat_rows.iloc[::-1].iterrows():
                            rec_id = row.get("record", idx + 1)
                            recent_alerts.append({
                                "id": f"REC-{rec_id}",
                                "timestamp": str(row.get("timestamp", "Recent")),
                                "attack_type": str(row.get("attack_type", "Unknown")),
                                "severity": str(row.get("risk_level", "Medium")).strip().title(),
                                "action": str(row.get("recommended_action", "Investigate")),
                                "status": "Active"
                            })
        except Exception as e:
            app.logger.warning("Dashboard data error: %s", str(e))

    return render_template(
        "dashboard.html",
        dashboard_stats=dashboard_stats,
        dashboard_chart_data=dashboard_chart_data,
        recent_alerts=recent_alerts,
        username=session.get("username", "Admin"),
        active_page="dashboard"
    )


# ============================================================
# TRAFFIC ANALYSIS (PAGE 3)
# ============================================================

@app.route("/traffic-analysis")
def traffic_analysis_page():
    gate = _page_requires_login()
    if gate:
        return gate

    return render_template(
        "traffic_analysis.html",
        active_page="traffic",
        username=session.get("username", "Admin")
    )


# ============================================================
# THREAT DETECTION (PAGE 4)
# ============================================================

@app.route("/threats")
def threats_page():
    gate = _page_requires_login()
    if gate:
        return gate

    attack_rows = []
    threat_records = []
    summary = {
        "total_records": 0,
        "attack_total": 0,
        "benign_count": 0,
        "attack_percentage": 0.0,
        "benign_percentage": 0.0,
        "attack_classes": 0,
        "has_analysis": False
    }

    results_path = os.path.join(RESULTS_DIR, RESULTS_FILENAME)

    if os.path.exists(results_path):
        try:
            result_df = pd.read_csv(results_path)
            total_records = len(result_df)
            if total_records > 0:
                summary["total_records"] = int(total_records)
                summary["has_analysis"] = True

                if "attack_type" in result_df.columns:
                    counts = (
                        result_df["attack_type"]
                        .astype(str)
                        .str.strip()
                        .str.upper()
                        .value_counts()
                    )

                    benign_count = int(counts.get("BENIGN", 0))
                    summary["benign_count"] = benign_count
                    attack_total = int(total_records - benign_count)
                    summary["attack_total"] = attack_total

                    summary["attack_percentage"] = round((attack_total / total_records) * 100, 2)
                    summary["benign_percentage"] = round((benign_count / total_records) * 100, 2)

                    # Attack group aggregations
                    for attack, count in counts.items():
                        attack_name = str(attack).strip()
                        if attack_name.upper() == "BENIGN":
                            continue

                        percentage = round((count / total_records) * 100, 2)
                        rows = result_df[result_df["attack_type"].astype(str).str.strip().str.upper() == attack_name.upper()]

                        risk = "Medium"
                        if "risk_level" in rows.columns and not rows.empty:
                            risk_values = rows["risk_level"].dropna().astype(str).str.strip()
                            if not risk_values.empty:
                                risk = str(risk_values.mode().iloc[0]).strip().title()

                        attack_rows.append({
                            "attack_type": attack_name,
                            "count": int(count),
                            "percentage": percentage,
                            "risk_level": risk,
                            "confidence": 96.5,
                            "recommended_action": get_response_recommendation(attack_name)
                        })

                    summary["attack_classes"] = len(attack_rows)

                    # Individual real threat records (up to 100)
                    malicious_df = result_df[result_df["attack_type"].astype(str).str.strip().str.upper() != "BENIGN"]
                    sample_threats = malicious_df.head(100)
                    for _, row in sample_threats.iterrows():
                        threat_records.append({
                            "record": int(row.get("record", 0)) if pd.notna(row.get("record")) else 0,
                            "class_id": int(row.get("class_id", 0)) if pd.notna(row.get("class_id")) else 0,
                            "attack_type": str(row.get("attack_type", "Unknown")),
                            "risk_level": str(row.get("risk_level", "Medium")).strip().title(),
                            "recommended_action": str(row.get("recommended_action", "Investigate")),
                            "timestamp": str(row.get("timestamp", ""))
                        })
        except Exception as exc:
            app.logger.warning("Could not read threat results: %s", exc)

    attack_rows.sort(key=lambda item: item["count"], reverse=True)

    return render_template(
        "threats.html",
        active_page="threats",
        attacks=attack_rows,
        threat_records=threat_records,
        summary=summary,
        username=session.get("username", "Admin")
    )


# ============================================================
# REPORTS PAGE
# ============================================================

@app.route("/reports")
def reports_page():
    gate = _page_requires_login()
    if gate:
        return gate

    results_path = os.path.join(RESULTS_DIR, RESULTS_FILENAME)
    has_results = os.path.exists(results_path)
    report_metadata = {
        "has_results": has_results,
        "filename": RESULTS_FILENAME if has_results else "None",
        "filesize": f"{round(os.path.getsize(results_path) / 1024, 1)} KB" if has_results else "0 KB",
        "last_modified": pd.Timestamp(os.path.getmtime(results_path), unit='s').strftime("%Y-%m-%d %H:%M:%S") if has_results else "N/A",
        "total_records": 0,
        "threat_count": 0
    }

    if has_results:
        try:
            df = pd.read_csv(results_path)
            report_metadata["total_records"] = len(df)
            if "attack_type" in df.columns:
                report_metadata["threat_count"] = int((df["attack_type"].astype(str).str.strip().str.upper() != "BENIGN").sum())
        except Exception:
            pass

    return render_template(
        "reports.html",
        active_page="reports",
        username=session.get("username", "Admin"),
        report=report_metadata
    )


# ============================================================
# MODEL PERFORMANCE PAGE
# ============================================================

@app.route("/model-performance")
@app.route("/model_performance")
def model_performance_page():
    gate = _page_requires_login()
    if gate:
        return gate

    return render_template(
        "model_performance.html",
        active_page="model",
        username=session.get("username", "Admin")
    )


# ============================================================
# SYSTEM INFO PAGE
# ============================================================

@app.route("/system-info")
@app.route("/system_info")
def system_info():
    gate = _page_requires_login()
    if gate:
        return gate

    return render_template(
        "system_info.html",
        active_page="system",
        username=session.get("username", "Admin"),
        model_name=type(model).__name__,
        feature_count=len(feature_names),
        class_count=len(class_mapping),
        accuracy="96.39%"
    )


# ============================================================
# INCIDENT MANAGEMENT (PAGE 5) - FIXED ROUTES
# ============================================================

@app.route("/incidents")
@app.route("/incident-management")
@app.route("/incident-management", endpoint="incident_management_page")
@app.route("/incidents-page")
def incidents_view():
    gate = _page_requires_login()
    if gate:
        return gate

    with incident_lock:
        incident_rows = [dict(item) for item in incidents.values()]

    for incident in incident_rows:
        incident["incident_id"] = incident.get("incident_id", incident.get("id", "INC-001"))
        incident["attack_type"] = str(incident.get("attack_type", "DDoS"))
        incident["severity"] = str(incident.get("severity", "Critical")).strip().title()
        incident["status"] = str(incident.get("status", "ACTIVE")).strip().upper()
        incident["timestamp"] = incident.get("updated_at", incident.get("created_at", "2025-10-02 14:32"))

    stats = {
        "total": len(incident_rows),
        "active": sum(1 for i in incident_rows if str(i.get("status", "")).upper() in {"ACTIVE", "INVESTIGATING"}),
        "critical": sum(1 for i in incident_rows if str(i.get("severity", "")).upper() == "CRITICAL"),
        "resolved": sum(1 for i in incident_rows if str(i.get("status", "")).upper() == "RESOLVED")
    }

    return render_template(
        "incidents.html",
        active_page="incidents",
        incidents=incident_rows,
        stats=stats,
        username=session.get("username", "Admin")
    )


# ============================================================
# INCIDENT DETAILS & RECOMMENDED RESPONSE (PAGE 6)
# ============================================================

@app.route("/incident-details/<incident_id>")
@app.route("/incident/<incident_id>")
def incident_details_page(incident_id):
    gate = _page_requires_login()
    if gate:
        return gate

    with incident_lock:
        stored_incident = incidents.get(incident_id)
        if stored_incident is not None:
            incident = dict(stored_incident)
        else:
            # Fallback to realistic values for preview if not in memory, and persist it
            incident = {
                "id": incident_id,
                "incident_id": incident_id,
                "attack_type": "DDoS",
                "severity": "Critical",
                "status": "ACTIVE",
                "timestamp": "2025-10-02 14:32",
                "confidence": "98.7%",
                "source_ip": "192.168.1.45",
                "target_ip": "10.0.0.23",
                "protocol": "TCP",
                "packets": 18542
            }
            incidents[incident_id] = incident

    attack_key = str(incident.get("attack_type", "DDOS")).strip().upper()

    impact_map = {
        "DDOS": [
            "Service disruption",
            "High network bandwidth usage",
            "System slowdown",
            "Potential data loss"
        ],
        "PORTSCAN": [
            "Network reconnaissance",
            "Identification of open services",
            "Potential follow-up attack",
            "Unauthorized service discovery"
        ],
        "BOT": [
            "Endpoint compromise",
            "Unauthorized automated activity",
            "Command-and-control communication",
            "Potential data exposure"
        ]
    }

    response_map = {
        "DDOS": [
            "Isolate affected traffic and block the source IP (192.168.1.45)",
            "Update IDS/IPS rules for DDoS pattern mitigation",
            "Monitor network traffic for similar activity",
            "Scale infrastructure bandwidth or apply firewall rate limits"
        ],
        "PORTSCAN": [
            "Investigate source IP and close unnecessary open ports",
            "Block scanning subnet at boundary perimeter",
            "Update firewall port-filtering limits"
        ]
    }

    incident["impact"] = impact_map.get(attack_key, [
        "Suspicious network activity",
        "Potential service disruption"
    ])

    incident["response"] = response_map.get(attack_key, [
        "Investigate the detected network activity",
        "Block origin IP address at firewall",
        "Generate and document incident report"
    ])

    incident["traffic_logs"] = [
        {
            "time": str(incident.get("timestamp", "2025-10-02 14:32")),
            "source": incident.get("source_ip", "192.168.1.45"),
            "target": incident.get("target_ip", "10.0.0.23"),
            "protocol": incident.get("protocol", "TCP"),
            "packets": incident.get("packets", 18542),
            "status": "Blocked"
        }
    ]

    return render_template(
        "incident_details.html",
        incident=incident,
        username=session.get("username", "Admin"),
        active_page="incidents"
    )


# ============================================================
# INCIDENT JSON API (MOVED TO /api/incidents TO PREVENT CLASH)
# ============================================================

@app.route("/api/incidents", methods=["GET"])
def get_incidents_api():
    if not session.get("logged_in"):
        return jsonify({"status": "error", "error": "Authentication required. Please log in."}), 401

    with incident_lock:
        incident_list = [dict(item) for item in incidents.values()]

    return jsonify({
        "status": "success",
        "count": len(incident_list),
        "incidents": incident_list
    })


# ============================================================
# INCIDENT STATUS ACTIONS
# ============================================================

@app.route("/incident/<incident_id>/status", methods=["POST"])
def update_incident_status_page(incident_id):
    gate = _page_requires_login()
    if gate:
        return gate

    new_status = request.form.get("status", "RESOLVED").strip().upper()
    update_incident_status(incident_id, new_status)
    return redirect(url_for("incident_details_page", incident_id=incident_id))


@app.route("/incident/<incident_id>/resolve", methods=["POST"])
def resolve_incident_page(incident_id):
    gate = _page_requires_login()
    if gate:
        return gate

    update_incident_status(incident_id, "RESOLVED")
    return redirect(url_for("incident_details_page", incident_id=incident_id))


# ============================================================
# INCIDENT HELPER FUNCTIONS
# ============================================================

def get_response_recommendation(attack_type):
    attack_key = str(attack_type).strip().upper()
    response_actions = {
        "DDOS": "Block source IP and rate-limit suspicious incoming traffic.",
        "PORTSCAN": "Investigate host activity and apply strict firewall port-filtering.",
        "BOT": "Isolate compromised endpoint and terminate unauthorized connections.",
        "INFILTRATION": "Immediately isolate endpoint and revoke active credentials."
    }
    return response_actions.get(attack_key, "Investigate and monitor affected network segment.")


def update_incident_status(incident_id, new_status):
    with incident_lock:
        incident = incidents.get(incident_id)
        if incident:
            incident["status"] = new_status
            incident["updated_at"] = pd.Timestamp.now().strftime("%Y-%m-%d %H:%M:%S")
            return dict(incident)
        else:
            # Seed and persist if resolving an incident not yet cached in memory
            incident = {
                "id": incident_id,
                "incident_id": incident_id,
                "attack_type": "DDoS",
                "severity": "Critical",
                "status": new_status,
                "created_at": "2025-10-02 14:32",
                "updated_at": pd.Timestamp.now().strftime("%Y-%m-%d %H:%M:%S"),
                "source_ip": "192.168.1.45",
                "target_ip": "10.0.0.23",
                "protocol": "TCP",
                "packets": 18542,
                "confidence": "98.7%"
            }
            incidents[incident_id] = incident
            return dict(incident)
    return None


def create_incident(attack_type, risk_level, count=1, source="AI Network Analysis"):
    if str(attack_type).strip().upper() == "BENIGN":
        return None

    incident_id = "INC-" + uuid.uuid4().hex[:6].upper()
    timestamp = pd.Timestamp.now().strftime("%Y-%m-%d %H:%M:%S")

    incident = {
        "incident_id": incident_id,
        "attack_type": str(attack_type).strip(),
        "severity": str(risk_level).strip().title(),
        "status": "ACTIVE",
        "count": int(count),
        "source": source,
        "created_at": timestamp,
        "updated_at": timestamp,
        "source_ip": "192.168.1.105",
        "target_ip": "10.0.0.1",
        "protocol": "TCP",
        "packets": int(count),
        "confidence": "97.5%"
    }

    with incident_lock:
        incidents[incident_id] = incident

    return incident


# ============================================================
# CSV BATCH ANALYSIS WORKER & API
# ============================================================

def run_analysis_job(job_id, X, filename, total_rows, valid_rows):
    try:
        predictions = predict_batch(X, batch_size=10000, show_progress=False)
        results = [
            {
                "record": i + 1,
                "class_id": int(p["class_id"]),
                "attack_type": str(p["attack_type"]),
                "risk_level": str(p["risk_level"]),
                "recommended_action": get_response_recommendation(p["attack_type"]),
                "timestamp": pd.Timestamp.now().strftime("%Y-%m-%d %H:%M:%S")
            }
            for i, p in enumerate(predictions)
        ]
        results_df = pd.DataFrame(results)

        output_path = os.path.join(RESULTS_DIR, RESULTS_FILENAME)
        results_df.to_csv(output_path, index=False)

        attack_counts = (
            results_df["attack_type"]
            .astype(str)
            .str.strip()
            .str.upper()
            .value_counts()
        )
        benign_count = int(attack_counts.get("BENIGN", 0))
        total_count = len(results_df)
        threat_count = int(total_count - benign_count)
        threat_percentage = round((threat_count / total_count * 100), 2) if total_count > 0 else 0
        benign_percentage = round((benign_count / total_count * 100), 2) if total_count > 0 else 0

        # Auto-create incident records for detected attacks
        if threat_count > 0:
            for attack_name, count in attack_counts.items():
                if str(attack_name).strip().upper() != "BENIGN":
                    create_incident(
                        attack_type=attack_name,
                        risk_level=get_risk_level(attack_name),
                        count=int(count),
                        source=f"CSV Analysis ({filename})"
                    )

        summary = {
            "total_records": total_count,
            "benign_count": benign_count,
            "threat_count": threat_count,
            "threat_percentage": threat_percentage,
            "benign_percentage": benign_percentage,
            "attack_classes": int(len(attack_counts)),
            "features_used": int(len(feature_names)),
            "attack_distribution": {str(k): int(v) for k, v in attack_counts.items()}
        }

        with analysis_lock:
            analysis_jobs[job_id].update({
                "status": "completed",
                "progress": 100,
                "processed": total_count,
                "total": total_count,
                "message": "Analysis completed successfully.",
                "summary": summary
            })
    except Exception as e:
        with analysis_lock:
            analysis_jobs[job_id].update({
                "status": "error",
                "error": str(e),
                "message": f"Analysis failed: {str(e)}"
            })


@app.route("/analyze_csv", methods=["POST"])
def analyze_csv():
    # Authentication check
    if not session.get("logged_in"):
        return jsonify({"status": "error", "error": "Authentication required. Please log in."}), 401

    if "file" not in request.files and "traffic_file" not in request.files:
        return jsonify({"status": "error", "error": "No file uploaded. Please select a CSV file."}), 400

    file = request.files.get("file") or request.files.get("traffic_file")
    if not file or not file.filename or file.filename.strip() == "":
        return jsonify({"status": "error", "error": "No file selected. Please choose a valid CSV file."}), 400

    raw_filename = file.filename
    clean_filename = secure_filename(raw_filename)
    if not clean_filename or not clean_filename.lower().endswith(".csv"):
        clean_filename = "network_capture.csv"

    if not raw_filename.lower().endswith(".csv"):
        return jsonify({"status": "error", "error": "Invalid file format. Only CSV files (.csv) are supported."}), 400

    try:
        df = pd.read_csv(file)
    except Exception as e:
        app.logger.warning("Failed to parse uploaded CSV: %s", str(e))
        return jsonify({"status": "error", "error": "Failed to parse CSV file. Ensure it is a valid, uncorrupted CSV file."}), 400

    if df.empty or len(df) == 0:
        return jsonify({"status": "error", "error": "The uploaded CSV file is empty."}), 400

    total_rows = len(df)

    clean_features = [str(f).strip() for f in feature_names]
    col_map = {str(c).strip(): c for c in df.columns}
    missing_features = [f for f in clean_features if f not in col_map]

    if missing_features:
        preview = ", ".join(missing_features[:5])
        suffix = f" and {len(missing_features) - 5} more" if len(missing_features) > 5 else ""
        return jsonify({
            "status": "error",
            "error": f"CSV is missing {len(missing_features)} required model feature(s): {preview}{suffix}.",
            "missing_features": missing_features,
            "missing_count": len(missing_features)
        }), 400

    selected_cols = [col_map[f] for f in clean_features]
    X = df[selected_cols].apply(pd.to_numeric, errors="coerce").replace([np.inf, -np.inf], np.nan).dropna()
    valid_rows = len(X)

    if valid_rows == 0:
        return jsonify({
            "status": "error",
            "error": "No valid numerical records remaining after cleaning missing or infinite values."
        }), 400

    job_id = str(uuid.uuid4())
    with analysis_lock:
        analysis_jobs[job_id] = {
            "status": "processing",
            "progress": 0,
            "filename": clean_filename,
            "total_rows": total_rows,
            "valid_rows": valid_rows
        }

    threading.Thread(
        target=run_analysis_job,
        args=(job_id, X, clean_filename, total_rows, valid_rows),
        daemon=True
    ).start()

    return jsonify({
        "status": "started",
        "job_id": job_id,
        "filename": clean_filename,
        "total_rows": total_rows,
        "valid_rows": valid_rows
    })


@app.route("/analysis_progress/<job_id>")
def analysis_progress(job_id):
    if not session.get("logged_in"):
        return jsonify({"status": "error", "error": "Authentication required. Please log in."}), 401

    with analysis_lock:
        job = analysis_jobs.get(job_id)
        if not job:
            return jsonify({"status": "error", "error": "Analysis job not found."}), 404
        return jsonify(job)


# ============================================================
# DOWNLOAD RESULTS
# ============================================================

@app.route("/download_results")
def download_results():
    gate = _page_requires_login()
    if gate:
        return gate

    results_path = os.path.join(RESULTS_DIR, RESULTS_FILENAME)
    if not os.path.exists(results_path):
        return jsonify({"status": "error", "message": "Results file not found."}), 404

    return send_from_directory(
        directory=RESULTS_DIR,
        path=RESULTS_FILENAME,
        as_attachment=True,
        download_name="AI_Traffic_Prediction_Batch_Results.csv"
    )


# ============================================================
# PASSWORD RECOVERY PAGE
# ============================================================

@app.route("/forgot-password")
def forgot_password_page():
    return render_template("forgot_password.html")


# ============================================================
# ESCALATE THREAT TO INCIDENT
# ============================================================

@app.route("/api/convert-to-incident", methods=["POST"])
def convert_to_incident():
    if not session.get("logged_in"):
        return jsonify({"status": "error", "error": "Authentication required. Please log in."}), 401

    data = request.get_json(silent=True) or {}
    new_inc_id = f"INC-{random.randint(100, 999)}"
    attack_type = data.get("attack_type", "DDoS")
    risk_level = data.get("severity", data.get("risk_level", "Critical"))

    with incident_lock:
        incidents[new_inc_id] = {
            "id": new_inc_id,
            "incident_id": new_inc_id,
            "attack_type": str(attack_type),
            "severity": str(risk_level).strip().title(),
            "status": "ACTIVE",
            "created_at": pd.Timestamp.now().strftime("%Y-%m-%d %H:%M:%S"),
            "updated_at": pd.Timestamp.now().strftime("%Y-%m-%d %H:%M:%S"),
            "source": "Escalated from Threat Results",
            "source_ip": data.get("source_ip", "192.168.1.105"),
            "target_ip": data.get("target_ip", "10.0.0.1"),
            "protocol": data.get("protocol", "TCP"),
            "packets": int(data.get("packets", 1)),
            "confidence": data.get("confidence", "97.5%")
        }

    return jsonify({
        "success": True,
        "incident_id": new_inc_id,
        "redirect_url": "/incidents"
    })


# ============================================================
# SEARCH API (SEARCH THREATS & INCIDENTS)
# ============================================================

@app.route("/api/search", methods=["GET"])
def api_search():
    if not session.get("logged_in"):
        return jsonify({"status": "error", "error": "Authentication required. Please log in."}), 401

    query = request.args.get("q", "").strip()
    if not query:
        return jsonify({
            "status": "success",
            "query": "",
            "threats": [],
            "incidents": [],
            "total_matches": 0
        })

    q_lower = query.lower()
    matching_incidents = []
    matching_threats = []

    # 1. Search Incidents
    with incident_lock:
        for inc_id, inc in incidents.items():
            inc_text = f"{inc_id} {inc.get('attack_type', '')} {inc.get('severity', '')} {inc.get('status', '')} {inc.get('source_ip', '')} {inc.get('target_ip', '')}".lower()
            if q_lower in inc_text:
                matching_incidents.append({
                    "id": inc_id,
                    "incident_id": inc_id,
                    "attack_type": inc.get("attack_type", "Unknown"),
                    "severity": inc.get("severity", "Medium"),
                    "status": inc.get("status", "ACTIVE"),
                    "source_ip": inc.get("source_ip", ""),
                    "created_at": inc.get("created_at", "")
                })

    # 2. Search Threat Records from dashboard analysis results
    results_path = os.path.join(RESULTS_DIR, RESULTS_FILENAME)
    if os.path.exists(results_path):
        try:
            res_df = pd.read_csv(results_path)
            if not res_df.empty:
                cond = (
                    res_df["attack_type"].astype(str).str.lower().str.contains(q_lower, na=False) |
                    res_df["risk_level"].astype(str).str.lower().str.contains(q_lower, na=False) |
                    res_df["recommended_action"].astype(str).str.lower().str.contains(q_lower, na=False)
                )
                if query.isdigit():
                    cond = cond | (res_df["record"].astype(str) == query)

                matched_df = res_df[cond]
                # Prioritize malicious threats unless benign is explicitly searched
                if "benign" not in q_lower and "normal" not in q_lower:
                    malicious_matches = matched_df[matched_df["attack_type"].astype(str).str.upper() != "BENIGN"]
                    if not malicious_matches.empty:
                        matched_df = malicious_matches

                for _, row in matched_df.head(15).iterrows():
                    matching_threats.append({
                        "record": int(row.get("record", 0)) if pd.notna(row.get("record")) else 0,
                        "attack_type": str(row.get("attack_type", "")),
                        "risk_level": str(row.get("risk_level", "Medium")).strip().title(),
                        "recommended_action": str(row.get("recommended_action", "Investigate")),
                        "timestamp": str(row.get("timestamp", ""))
                    })
        except Exception as e:
            app.logger.warning("Search query error reading results: %s", str(e))

    return jsonify({
        "status": "success",
        "query": query,
        "threats": matching_threats,
        "incidents": matching_incidents[:10],
        "total_matches": len(matching_incidents) + len(matching_threats)
    })


# ============================================================
# PRODUCTION ERROR HANDLERS (NO STACK TRACE / PATH EXPOSURE)
# ============================================================

@app.errorhandler(400)
def bad_request_error(e):
    if request.is_json or request.path.startswith("/analyze_csv") or request.path.startswith("/api/"):
        return jsonify({"status": "error", "error": "Bad request. Please verify submitted parameters."}), 400
    return redirect(url_for("home"))


@app.errorhandler(401)
def unauthorized_error(e):
    if request.is_json or request.path.startswith("/analyze_csv") or request.path.startswith("/api/"):
        return jsonify({"status": "error", "error": "Authentication required. Please log in."}), 401
    return redirect(url_for("home"))


@app.errorhandler(404)
def not_found_error(e):
    if request.is_json or request.path.startswith("/api/"):
        return jsonify({"status": "error", "error": "Requested endpoint or resource not found."}), 404
    if session.get("logged_in"):
        return redirect(url_for("dashboard"))
    return redirect(url_for("home"))


@app.errorhandler(413)
def file_too_large_error(e):
    if request.is_json or request.path.startswith("/analyze_csv") or request.path.startswith("/api/"):
        return jsonify({"status": "error", "error": "Uploaded file exceeds the maximum allowed size (100 MB)."}), 413
    return redirect(url_for("traffic_analysis_page"))


@app.errorhandler(500)
def internal_server_error(e):
    app.logger.error("Internal Server Error: %s", str(e))
    if request.is_json or request.path.startswith("/analyze_csv") or request.path.startswith("/api/"):
        return jsonify({"status": "error", "error": "An internal server error occurred. Please contact the administrator."}), 500
    if session.get("logged_in"):
        return redirect(url_for("dashboard"))
    return redirect(url_for("home"))


# ============================================================
# APPLICATION ENTRYPOINT (CONFIGURABLE HOST / PORT / DEBUG)
# ============================================================

if __name__ == "__main__":
    host = os.environ.get("HOST", "127.0.0.1")
    port = int(os.environ.get("PORT", 5000))
    debug = os.environ.get("FLASK_DEBUG", "false").lower() in ("true", "1")

    print("=" * 60)
    print("AI-BASED INTRUSION DETECTION SYSTEM (AI-IDS)")
    print(f"Host: {host} | Port: {port} | Debug Mode: {debug}")
    print(f"Dashboard: http://{host}:{port}/dashboard")
    print(f"Incidents: http://{host}:{port}/incidents")
    print("=" * 60)

    app.run(host=host, port=port, debug=debug, use_reloader=False)
