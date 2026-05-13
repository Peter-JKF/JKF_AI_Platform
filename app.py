"""JKF Chatbot & Dashboard Platform – combined Flask application.

Database: SQLite (migrate to PostgreSQL/MySQL by changing DATABASE_URL in .env).
Chatbot: RAG pipeline with hybrid Qdrant search, HyDE and Jina reranking.
"""
from dotenv import load_dotenv
load_dotenv()

import os
import re
import json
import uuid
import time
import hmac
import hashlib
import logging
import threading
from collections import defaultdict
from datetime import datetime, timedelta
from logging.handlers import RotatingFileHandler
from urllib.parse import urljoin, urlparse, urldefrag
from xml.etree import ElementTree as ET

import pytz
import requests
from flask import (
    Flask, render_template, request, redirect, url_for,
    flash, session, jsonify, Response, stream_with_context, send_file,
)
from flask_cors import CORS
from flask_limiter import Limiter
from flask_limiter.util import get_remote_address
from flask_wtf import FlaskForm
from flask_wtf.csrf import CSRFProtect
from werkzeug.security import generate_password_hash, check_password_hash as wz_check_password_hash
from werkzeug.utils import secure_filename
from wtforms import StringField, PasswordField, SubmitField, BooleanField, TextAreaField, SelectField
from wtforms.validators import DataRequired
from sqlalchemy import func, text, desc, case
from tenacity import retry, stop_after_attempt, wait_exponential

from config import Config
from extensions import db, openai_client as client, openrouter_client

# ── Qdrant ───────────────────────────────────────────────────────────────────
try:
    from qdrant_client import QdrantClient
    from qdrant_client.models import (
        Filter, FieldCondition, MatchValue,
        SparseVector, Prefetch, FusionQuery, Fusion,
    )
    QDRANT_AVAILABLE = True
except ImportError:
    QDRANT_AVAILABLE = False

# ─────────────────────────────────────────────────────────────────────────────
# App setup
# ─────────────────────────────────────────────────────────────────────────────
app = Flask(__name__)
app.config.from_object(Config)
CORS(app, resources={r"/chat*": {"origins": "*"}, r"/init_thread": {"origins": "*"},
                     r"/clear_thread": {"origins": "*"}, r"/track_click": {"origins": "*"},
                     r"/feedback": {"origins": "*"}, r"/submit_feedback": {"origins": "*"},
                     r"/static/*": {"origins": "*"}})
db.init_app(app)
csrf = CSRFProtect(app)
limiter = Limiter(get_remote_address, app=app, default_limits=[], storage_uri="memory://")

if app.config['SECRET_KEY'] == 'jkf-platform-secret-change-in-production':
    logging.getLogger(__name__).critical(
        "SECURITY: Default FLASK_SECRET_KEY is in use — set a strong random key in .env before deploying!"
    )

# Exempt chatbot API routes from CSRF (they're called from embedded widgets)
@csrf.exempt
def _noop(): pass  # noqa — individual exemptions done on routes below

# ── Logging ──────────────────────────────────────────────────────────────────
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)
if not app.debug:
    os.makedirs('logs', exist_ok=True)
    fh = RotatingFileHandler('logs/jkf.log', maxBytes=10240, backupCount=10)
    fh.setFormatter(logging.Formatter('%(asctime)s %(levelname)s: %(message)s'))
    fh.setLevel(logging.INFO)
    app.logger.addHandler(fh)

# ── Qdrant client ─────────────────────────────────────────────────────────────
QDRANT_COLLECTION = "jkf_kb"
_qdrant_client = None


def get_qdrant_client():
    global _qdrant_client
    if _qdrant_client is not None:
        return _qdrant_client
    if not QDRANT_AVAILABLE:
        return None
    url = os.environ.get("QDRANT_URL", "")
    key = os.environ.get("QDRANT_API_KEY", "")
    if url and "your-cluster" not in url:
        try:
            _qdrant_client = QdrantClient(url=url, api_key=key or None, prefer_grpc=False)
            logger.info("Qdrant client initialised")
        except Exception as e:
            logger.warning(f"Qdrant init failed: {e}")
    return _qdrant_client


# ── Timezone ──────────────────────────────────────────────────────────────────
danish_tz = pytz.timezone('Europe/Copenhagen')


def get_danish_time():
    return datetime.now(danish_tz)


def utc_iso(dt: "datetime | None") -> "str | None":
    """Return an ISO-8601 string with explicit UTC offset (+00:00) so that
    JavaScript ``new Date()`` correctly interprets the value as UTC rather than
    treating it as local time (which would be 2 h off in CEST)."""
    if dt is None:
        return None
    if dt.tzinfo is None:
        dt = dt.replace(tzinfo=pytz.utc)
    return dt.isoformat()


def get_datetime_context():
    t = get_danish_time()
    weekdays = ['Mandag', 'Tirsdag', 'Onsdag', 'Torsdag', 'Fredag', 'Lørdag', 'Søndag']
    months = ['januar', 'februar', 'marts', 'april', 'maj', 'juni',
              'juli', 'august', 'september', 'oktober', 'november', 'december']
    return (f"[SYSTEM CONTEXT - Dags dato og tid: {weekdays[t.weekday()]} den {t.day}. "
            f"{months[t.month - 1]} {t.year} kl. {t.strftime('%H:%M')}]\n\n")


# ─────────────────────────────────────────────────────────────────────────────
# Database models
# ─────────────────────────────────────────────────────────────────────────────
class User(db.Model):
    __tablename__ = 'users'
    id = db.Column(db.Integer, primary_key=True)
    username = db.Column(db.String(64), unique=True, nullable=False)
    password_hash = db.Column(db.String(256))
    is_admin = db.Column(db.Boolean, default=False)
    can_access_sales_chatbot = db.Column(db.Boolean, default=False)
    sales_only = db.Column(db.Boolean, default=False)
    can_access_budget_agent = db.Column(db.Boolean, default=False)
    budget_only = db.Column(db.Boolean, default=False)
    can_access_master_agent = db.Column(db.Boolean, default=False)
    agents_only = db.Column(db.Boolean, default=False)
    created_at = db.Column(db.DateTime, default=datetime.utcnow)

    def set_password(self, password):
        self.password_hash = generate_password_hash(password)

    def check_password(self, password):
        stored = self.password_hash or ''
        # Seamlessly migrate legacy SHA-256 hashes (64 hex chars, no prefix) to scrypt
        if len(stored) == 64 and re.fullmatch(r'[0-9a-f]{64}', stored):
            if stored == hashlib.sha256(password.encode()).hexdigest():
                self.set_password(password)  # upgrade in-place; login route commits
                return True
            return False
        return wz_check_password_hash(stored, password)


class Message(db.Model):
    __tablename__ = 'messages'
    id = db.Column(db.Integer, primary_key=True)
    thread_id = db.Column(db.String(255), nullable=False, index=True)
    role = db.Column(db.String(50), nullable=False)
    content = db.Column(db.Text, nullable=False)
    timestamp = db.Column(db.DateTime, default=datetime.utcnow, index=True)
    feedback = db.Column(db.Integer)           # 1/2/3 rating
    language = db.Column(db.String(50))
    category = db.Column(db.String(100))
    knowledge_gaps = db.Column(db.Integer, default=0)
    clicked_link = db.Column(db.String(500))


class KnowledgeBase(db.Model):
    __tablename__ = 'knowledge_base'
    id = db.Column(db.Integer, primary_key=True)
    question = db.Column(db.Text, nullable=False)
    answer = db.Column(db.Text, nullable=False)
    category = db.Column(db.String(100))
    source_url = db.Column(db.Text)  # Optional: URL of the source page for this entry
    created_at = db.Column(db.DateTime, default=datetime.utcnow)
    updated_at = db.Column(db.DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)
    created_by = db.Column(db.Integer, db.ForeignKey('users.id'))
    is_active = db.Column(db.Boolean, default=True)
    qdrant_synced = db.Column(db.Boolean, default=False)

    creator = db.relationship('User', foreign_keys=[created_by])


class Document(db.Model):
    """Uploaded documents (PDF, TXT) chunked and synced to Qdrant."""
    __tablename__ = 'documents'
    id            = db.Column(db.Integer, primary_key=True)
    original_name = db.Column(db.String(255), nullable=False)
    stored_name   = db.Column(db.String(255), nullable=False)
    file_type     = db.Column(db.String(20))
    file_size     = db.Column(db.Integer, default=0)
    num_chunks    = db.Column(db.Integer, default=0)
    status        = db.Column(db.String(20), default='processing')  # processing/ready/error
    error_message = db.Column(db.Text)
    qdrant_synced = db.Column(db.Boolean, default=False)
    is_active     = db.Column(db.Boolean, default=True)
    created_at    = db.Column(db.DateTime, default=datetime.utcnow)
    created_by    = db.Column(db.Integer, db.ForeignKey('users.id'))


class JKFSettings(db.Model):
    """Single-row settings table for JKF chatbot configuration."""
    __tablename__ = 'jkf_settings'
    id = db.Column(db.Integer, primary_key=True)

    # Chatbot appearance
    chatbot_name = db.Column(db.String(100), default='JKF AI-assistent')
    welcome_message = db.Column(db.Text, default='Hej! Hvordan kan jeg hjælpe dig i dag?')
    disclaimer_text = db.Column(db.String(255), default='Dette er vejledende svar')
    primary_color = db.Column(db.String(7), default='#1a56db')
    toggle_color = db.Column(db.String(7), default='#1a56db')
    chatbot_position = db.Column(db.String(30), default='bottom-right')
    logo_url = db.Column(db.String(500))
    show_powered_by = db.Column(db.Boolean, default=True)
    quick_questions = db.Column(db.Text)  # JSON array
    speech_to_text_enabled = db.Column(db.Boolean, default=False)

    # RAG / AI
    model = db.Column(db.String(50), default='gpt-5.4-mini')
    system_prompt = db.Column(db.Text)

    # Website scraping
    website_url = db.Column(db.String(500))
    last_scraped_date = db.Column(db.DateTime)


class Integration(db.Model):
    """Third-party integrations (Business Central, etc.)."""
    __tablename__ = 'integrations'
    id               = db.Column(db.Integer, primary_key=True)
    integration_type = db.Column(db.String(50), nullable=False)  # 'business_central'
    name             = db.Column(db.String(100), default='')
    enabled          = db.Column(db.Boolean, default=False)
    config           = db.Column(db.Text)  # JSON blob: credentials + settings
    created_at       = db.Column(db.DateTime, default=datetime.utcnow)
    updated_at       = db.Column(db.DateTime, default=datetime.utcnow)


class BCItem(db.Model):
    """Cached Business Central item for description-based search."""
    __tablename__ = 'bc_items'
    id           = db.Column(db.Integer, primary_key=True)
    item_no      = db.Column(db.String(50), unique=True, nullable=False, index=True)
    description  = db.Column(db.String(250), default='')
    description2 = db.Column(db.String(250), default='')
    inventory    = db.Column(db.Float, default=0.0)
    lead_time    = db.Column(db.String(50), default='')   # e.g. "5D" from salesLeadTimeEVM
    synced_at    = db.Column(db.DateTime, default=datetime.utcnow)


class Comment(db.Model):
    """Per-thread comment left by dashboard users."""
    __tablename__ = 'comments'
    id = db.Column(db.Integer, primary_key=True)
    thread_id = db.Column(db.String(255), nullable=False, index=True)
    user_id = db.Column(db.Integer, db.ForeignKey('users.id'))
    content = db.Column(db.Text, nullable=False)
    timestamp = db.Column(db.DateTime, default=datetime.utcnow)
    user = db.relationship('User', foreign_keys=[user_id])


class KnowledgeGapAnalysis(db.Model):
    """Stored AI analysis of knowledge-gap conversations."""
    __tablename__ = 'knowledge_gap_analyses'
    id = db.Column(db.Integer, primary_key=True)
    created_at = db.Column(db.DateTime, default=datetime.utcnow)
    interval = db.Column(db.String(20))
    conversations_analyzed = db.Column(db.Integer)
    total_conversations = db.Column(db.Integer)
    recommendations = db.Column(db.Text)  # JSON array


class WebsiteScrapeRule(db.Model):
    """Manual include/exclude rules for website scraping."""
    __tablename__ = 'website_scrape_rules'
    id = db.Column(db.Integer, primary_key=True)
    url = db.Column(db.String(1000), unique=True, nullable=False, index=True)
    rule_type = db.Column(db.String(20), nullable=False, default='exclude')  # include | exclude
    created_at = db.Column(db.DateTime, default=datetime.utcnow)
    created_by = db.Column(db.Integer, db.ForeignKey('users.id'))


class WebsiteIndexedPage(db.Model):
    """Bookkeeping for the pages currently indexed from the website."""
    __tablename__ = 'website_indexed_pages'
    id = db.Column(db.Integer, primary_key=True)
    url = db.Column(db.String(1000), unique=True, nullable=False, index=True)
    title = db.Column(db.String(500))
    chunk_count = db.Column(db.Integer, default=0)
    indexed_at = db.Column(db.DateTime, default=datetime.utcnow, index=True)


class SalesConversation(db.Model):
    """Top-level record for a sales-chatbot conversation."""
    __tablename__ = 'sales_conversations'
    id         = db.Column(db.Integer, primary_key=True)
    user_id    = db.Column(db.Integer, db.ForeignKey('users.id'), nullable=False, index=True)
    title      = db.Column(db.String(255), nullable=False, default='Ny samtale')
    created_at = db.Column(db.DateTime, default=datetime.utcnow, index=True)
    updated_at = db.Column(db.DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)

    user     = db.relationship('User', foreign_keys=[user_id])
    messages = db.relationship('SalesMessage', backref='conversation',
                               cascade='all, delete-orphan', order_by='SalesMessage.id')


class SalesMessage(db.Model):
    """A single turn (user or assistant) within a SalesConversation."""
    __tablename__ = 'sales_messages'
    id              = db.Column(db.Integer, primary_key=True)
    conversation_id = db.Column(db.Integer, db.ForeignKey('sales_conversations.id'),
                                nullable=False, index=True)
    role            = db.Column(db.String(20), nullable=False)   # 'user' | 'assistant'
    content         = db.Column(db.Text, nullable=False)
    sql_query       = db.Column(db.Text)                          # only on assistant turns
    row_count       = db.Column(db.Integer)
    created_at      = db.Column(db.DateTime, default=datetime.utcnow)


class BudgetConversation(db.Model):
    """Top-level record for a budget-agent conversation."""
    __tablename__ = 'budget_conversations'
    id         = db.Column(db.Integer, primary_key=True)
    user_id    = db.Column(db.Integer, db.ForeignKey('users.id'), nullable=False, index=True)
    title      = db.Column(db.String(255), nullable=False, default='Ny samtale')
    created_at = db.Column(db.DateTime, default=datetime.utcnow, index=True)
    updated_at = db.Column(db.DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)

    user     = db.relationship('User', foreign_keys=[user_id])
    messages = db.relationship('BudgetMessage', backref='conversation',
                               cascade='all, delete-orphan', order_by='BudgetMessage.id')


class BudgetMessage(db.Model):
    """A single turn (user or assistant) within a BudgetConversation."""
    __tablename__ = 'budget_messages'
    id              = db.Column(db.Integer, primary_key=True)
    conversation_id = db.Column(db.Integer, db.ForeignKey('budget_conversations.id'),
                                nullable=False, index=True)
    role            = db.Column(db.String(20), nullable=False)   # 'user' | 'assistant'
    content         = db.Column(db.Text, nullable=False)
    sql_query       = db.Column(db.Text)                          # only on assistant turns
    row_count       = db.Column(db.Integer)
    created_at      = db.Column(db.DateTime, default=datetime.utcnow)


class MasterConversation(db.Model):
    """Top-level record for a master-agent conversation."""
    __tablename__ = 'master_conversations'
    id         = db.Column(db.Integer, primary_key=True)
    user_id    = db.Column(db.Integer, db.ForeignKey('users.id'), nullable=False, index=True)
    title      = db.Column(db.String(255), nullable=False, default='Ny samtale')
    created_at = db.Column(db.DateTime, default=datetime.utcnow, index=True)
    updated_at = db.Column(db.DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)

    user     = db.relationship('User', foreign_keys=[user_id])
    messages = db.relationship('MasterMessage', backref='conversation',
                               cascade='all, delete-orphan', order_by='MasterMessage.id')


class MasterMessage(db.Model):
    """A single turn (user or assistant) within a MasterConversation."""
    __tablename__ = 'master_messages'
    id              = db.Column(db.Integer, primary_key=True)
    conversation_id = db.Column(db.Integer, db.ForeignKey('master_conversations.id'),
                                nullable=False, index=True)
    role            = db.Column(db.String(20), nullable=False)   # 'user' | 'assistant'
    content         = db.Column(db.Text, nullable=False)
    tools_used      = db.Column(db.Text, default='[]')            # JSON list of tool names
    created_at      = db.Column(db.DateTime, default=datetime.utcnow)


# ─────────────────────────────────────────────────────────────────────────────
# Init DB
# ─────────────────────────────────────────────────────────────────────────────
def init_db():
    with app.app_context():
        db.create_all()
        # Add new columns to existing tables that SQLAlchemy won't auto-migrate
        for migration_sql, migration_msg in [
            ("ALTER TABLE users ADD COLUMN can_access_sales_chatbot BOOLEAN DEFAULT 0 NOT NULL",
             "Migrated: added can_access_sales_chatbot column to users table"),
            ("ALTER TABLE users ADD COLUMN sales_only BOOLEAN DEFAULT 0 NOT NULL",
             "Migrated: added sales_only column to users table"),
            ("ALTER TABLE users ADD COLUMN can_access_budget_agent BOOLEAN DEFAULT 0 NOT NULL",
             "Migrated: added can_access_budget_agent column to users table"),
            ("ALTER TABLE users ADD COLUMN budget_only BOOLEAN DEFAULT 0 NOT NULL",
             "Migrated: added budget_only column to users table"),
            ("ALTER TABLE users ADD COLUMN can_access_master_agent BOOLEAN DEFAULT 0 NOT NULL",
             "Migrated: added can_access_master_agent column to users table"),
            ("ALTER TABLE users ADD COLUMN agents_only BOOLEAN DEFAULT 0 NOT NULL",
             "Migrated: added agents_only column to users table"),
        ]:
            with db.engine.connect() as conn:
                try:
                    conn.execute(text(migration_sql))
                    conn.commit()
                    logger.info(migration_msg)
                except Exception:
                    pass  # Column already exists
        os.makedirs(app.config.get('UPLOAD_FOLDER', 'uploads'), exist_ok=True)
        # Ensure at least one admin user exists
        if not User.query.filter_by(username='admin').first():
            admin = User(username='admin', is_admin=True)
            admin.set_password('jkf2024')  # change immediately after first login!
            db.session.add(admin)
            db.session.commit()
            logger.info("Created default admin user (admin / jkf2024) – CHANGE THIS PASSWORD!")
        # Ensure settings row exists
        if not JKFSettings.query.first():
            db.session.add(JKFSettings())
            db.session.commit()
            logger.info("Created default JKF settings row")

        # Re-queue any documents left stuck in 'processing' from a previous
        # server restart that killed their background threads.
        stuck = Document.query.filter_by(status='processing').all()
        if stuck:
            logger.info(f"Re-queuing {len(stuck)} stuck document(s) from previous run")
            for doc in stuck:
                t = threading.Thread(target=_process_document, args=(doc.id,), daemon=True)
                t.start()


def _normalise_rule_url(url: str) -> str:
    return _web_normalise_url(url.strip())


def _get_website_rules() -> tuple[set[str], set[str]]:
    include_urls = set()
    exclude_urls = set()
    for rule in WebsiteScrapeRule.query.all():
        if rule.rule_type == 'include':
            include_urls.add(rule.url)
        elif rule.rule_type == 'exclude':
            exclude_urls.add(rule.url)
    return include_urls, exclude_urls


def _record_indexed_page(url: str, title: str, chunk_count: int) -> None:
    page = WebsiteIndexedPage.query.filter_by(url=url).first()
    if not page:
        page = WebsiteIndexedPage(url=url)
        db.session.add(page)
    page.title = title[:500] if title else url
    page.chunk_count = chunk_count
    page.indexed_at = datetime.utcnow()


def _delete_website_page_from_qdrant(url: str) -> None:
    qc = get_qdrant_client()
    if not qc:
        return
    try:
        from qdrant_client.models import Filter as QFilter, FieldCondition as QFC, MatchValue as QMV
        qc.delete(
            collection_name=QDRANT_COLLECTION,
            points_selector=QFilter(must=[
                QFC(key="source_type", match=QMV(value="website")),
                QFC(key="source_url", match=QMV(value=url)),
            ]),
        )
    except Exception as e:
        logger.warning(f"_delete_website_page_from_qdrant failed for {url}: {e}")


def _sync_website_indexed_pages_from_qdrant() -> tuple[int, int]:
    """Rebuild website page bookkeeping from the live Qdrant website payloads."""
    qc = get_qdrant_client()
    if not qc:
        return 0, 0

    try:
        from qdrant_client.models import Filter as QFilter, FieldCondition as QFC, MatchValue as QMV

        page_map: dict[str, dict] = {}
        offset = None
        chunk_total = 0

        while True:
            points, offset = qc.scroll(
                collection_name=QDRANT_COLLECTION,
                scroll_filter=QFilter(must=[QFC(key="source_type", match=QMV(value="website"))]),
                limit=256,
                offset=offset,
                with_payload=["source_url", "section_title"],
                with_vectors=False,
            )

            if not points:
                break

            for point in points:
                payload = point.payload or {}
                raw_url = (payload.get("source_url") or "").strip()
                if not raw_url:
                    continue

                url = _web_normalise_url(raw_url)
                title = (payload.get("section_title") or "").strip()
                row = page_map.setdefault(url, {"title": title or url, "chunk_count": 0})
                row["chunk_count"] += 1
                chunk_total += 1

                if title and (not row["title"] or row["title"] == url):
                    row["title"] = title

            if offset is None:
                break

        WebsiteIndexedPage.query.delete()
        for url, row in sorted(page_map.items()):
            db.session.add(WebsiteIndexedPage(
                url=url,
                title=(row["title"] or url)[:500],
                chunk_count=row["chunk_count"],
                indexed_at=datetime.utcnow(),
            ))
        db.session.commit()
        return len(page_map), chunk_total
    except Exception as e:
        db.session.rollback()
        logger.warning(f"_sync_website_indexed_pages_from_qdrant failed: {e}")
        return WebsiteIndexedPage.query.count(), 0


# ─────────────────────────────────────────────────────────────────────────────
# Auth helpers
# ─────────────────────────────────────────────────────────────────────────────
def login_required(f):
    from functools import wraps
    @wraps(f)
    def decorated(*args, **kwargs):
        if 'user_id' not in session:
            return redirect(url_for('login', next=request.url))
        user = User.query.get(session['user_id'])
        if user and not user.is_admin:
            if user.budget_only:
                return redirect(url_for('budget_agent'))
            if user.sales_only:
                return redirect(url_for('sales_chatbot'))
            if user.agents_only:
                if user.can_access_master_agent:
                    return redirect(url_for('master_agent'))
                if user.can_access_sales_chatbot:
                    return redirect(url_for('sales_chatbot'))
                if user.can_access_budget_agent:
                    return redirect(url_for('budget_agent'))
                # No agent assigned yet — fall through to avoid redirect loop
        return f(*args, **kwargs)
    return decorated


def admin_required(f):
    from functools import wraps
    @wraps(f)
    def decorated(*args, **kwargs):
        if 'user_id' not in session:
            return redirect(url_for('login'))
        user = User.query.get(session['user_id'])
        if not user or not user.is_admin:
            flash('Du har ikke adgang til denne side.', 'danger')
            return redirect(url_for('dashboard'))
        return f(*args, **kwargs)
    return decorated


def sales_chatbot_required(f):
    from functools import wraps
    @wraps(f)
    def decorated(*args, **kwargs):
        if 'user_id' not in session:
            return redirect(url_for('login'))
        user = User.query.get(session['user_id'])
        if not user or (not user.can_access_sales_chatbot and not user.is_admin):
            flash('Du har ikke adgang til salgs-assistenten.', 'danger')
            return redirect(url_for('dashboard'))
        return f(*args, **kwargs)
    return decorated


def budget_agent_required(f):
    from functools import wraps
    @wraps(f)
    def decorated(*args, **kwargs):
        if 'user_id' not in session:
            return redirect(url_for('login'))
        user = User.query.get(session['user_id'])
        if not user or (not user.can_access_budget_agent and not user.is_admin):
            flash('Du har ikke adgang til budget-assistenten.', 'danger')
            return redirect(url_for('dashboard'))
        return f(*args, **kwargs)
    return decorated


def master_agent_required(f):
    from functools import wraps
    @wraps(f)
    def decorated(*args, **kwargs):
        if 'user_id' not in session:
            return redirect(url_for('login'))
        user = User.query.get(session['user_id'])
        if not user or (not user.can_access_master_agent and not user.is_admin):
            flash('Du har ikke adgang til Master Agenten.', 'danger')
            return redirect(url_for('dashboard'))
        return f(*args, **kwargs)
    return decorated


# ─────────────────────────────────────────────────────────────────────────────
# Template context
# ─────────────────────────────────────────────────────────────────────────────
@app.context_processor
def inject_globals():
    user = None
    if 'user_id' in session:
        user = User.query.get(session['user_id'])
    return dict(user=user)


@app.template_filter('from_json')
def from_json_filter(s):
    if not s:
        return []
    try:
        return json.loads(s)
    except Exception:
        return []


@app.template_filter('danish_date')
def danish_date_filter(date):
    if isinstance(date, datetime):
        if date.tzinfo is None:
            date = date.replace(tzinfo=pytz.utc)
        return date.astimezone(danish_tz).strftime('%d-%m-%Y')
    return date


@app.template_filter('danish_datetime')
def danish_datetime_filter(dt):
    if isinstance(dt, datetime):
        if dt.tzinfo is None:
            dt = dt.replace(tzinfo=pytz.utc)
        return dt.astimezone(danish_tz).strftime('%d/%m/%Y %H:%M')
    return dt


# ─────────────────────────────────────────────────────────────────────────────
# Forms
# ─────────────────────────────────────────────────────────────────────────────
class LoginForm(FlaskForm):
    username = StringField('Brugernavn', validators=[DataRequired()])
    password = PasswordField('Adgangskode', validators=[DataRequired()])
    submit = SubmitField('Log ind')


class UserForm(FlaskForm):
    username = StringField('Brugernavn', validators=[DataRequired()])
    password = PasswordField('Adgangskode', validators=[DataRequired()])
    is_admin = BooleanField('Administrator')
    can_access_sales_chatbot = BooleanField('Salgs-assistent adgang')
    sales_only = BooleanField('Kun salgs-assistent')
    can_access_budget_agent = BooleanField('Budget-assistent adgang')
    budget_only = BooleanField('Kun budget-assistent')
    can_access_master_agent = BooleanField('Master Agent adgang')
    agents_only = BooleanField('Kun agenter')
    submit = SubmitField('Gem bruger')


# In-memory training history – never persisted to DB, cleared on reset
_training_history: dict = {}


# ─────────────────────────────────────────────────────────────────────────────
# RAG pipeline helpers
# ─────────────────────────────────────────────────────────────────────────────

DEFAULT_SYSTEM_PROMPT = """## Your role / Din rolle
You are JKF's friendly AI assistant, integrated directly on JKF's website. You help visitors quickly and easily with questions about products, delivery times, orders and anything else related to JKF.

Your goal is to make it easy to be a JKF customer. Be welcoming, clear and concrete, like a knowledgeable colleague who knows the company from the inside.

## Language / Sprog
This is the most important rule — it overrides everything else.

Detect the language of the user's LATEST message and reply in that exact language. Every single time. Even if the conversation so far has been in a different language, switch immediately to match the user's most recent message.

- "what can you help with?" -> reply in English
- "hvad kan du hjælpe med?" -> reply in Danish
- "was kannst du mir helfen?" -> reply in German
- "que peux-tu faire?" -> reply in French
- "okay mange tak" -> reply in Danish, even if the previous exchange was in English
- "thanks" after a Danish exchange -> reply in English

Supported languages: Danish, English, German, French (and others as needed).
If the language of the latest message is genuinely ambiguous, default to Danish.
Never mix languages in a single response.
Translate product names and technical terms naturally into the reply language.

## How to answer / Sådan svarer du
- Write in a natural, friendly and fluent tone. Neither too formal nor too casual.
- Never use em dashes (— or –). Rewrite sentences so they flow naturally without them.
- Keep answers short and readable. Avoid large blocks of text. Use bullet points when it helps clarity.
- Do not end with generic phrases like "Contact us for more info" unless that is the only relevant answer.

## URLs and links — HARD RULES (never break these)
Rule 1: You MUST NEVER produce a URL that does not appear literally, character-for-character, in [RELEVANT VIDEN]. Not for any reason. Not even if you "know" the page exists. Not even if the path looks obvious. If it is not in [RELEVANT VIDEN], it does not exist for you.

Rule 2: When [RELEVANT VIDEN] contains a markdown link [text](url), reproduce it EXACTLY as written — never as a plain URL string.

Rule 3: Website chunks end with [Læs mere](url). Embed this link naturally INSIDE your answer sentence — do not append it as a separate line with a label. Use descriptive link text that fits the sentence, in the reply language.

WRONG: "Du kan finde artiklerne på JKF Ecademy.\n\nLæs mere her: [Læs mere](url)"
RIGHT: "Du kan finde artiklerne i [JKF Ecademys ressourcebibliotek](url)."

WRONG: "You can find pipe systems here.\n\nRead more: [Read more](url)"
RIGHT: "You can find the [full pipe systems overview here](url)."

The link text should describe the destination, not just say "Læs mere" or "Read more". Never write a label like "Læs mere her:" before the link. Never output the link on its own line. Never strip or alter the URL part.

Rule 4: If NO URL appears in [RELEVANT VIDEN] for the topic being discussed, do NOT provide one. Say you don't have a direct link, or direct the user to jkfuniverse.com to find it themselves.

Rule 5: Documents ("Kilde: dokument 'filename'") have no URL. Mention the document name. Do not invent a URL for it.

Rule 6: Never repeat the same URL more than once in a single response.

VIOLATION EXAMPLE (never do this): User asks about contact — you write "visit https://jkfuniverse.com/da/contact/" — this is a violation because you invented that URL from your training knowledge, not from [RELEVANT VIDEN]. Even if the URL would normally work, you must not produce it unless it is literally in [RELEVANT VIDEN].

## Knowledge sources and when to use them

### 1. Knowledge base [RELEVANT VIDEN]
Injected automatically when relevant. Contains four types of content:
- **Website content**: JKF's website pages, company info, solutions and industries
- **Q&A entries**: Curated answers to common customer questions
- **Documents**: Uploaded product sheets, guides and technical documentation
- **Product catalogues**: Full product catalogue content including categories, specifications and descriptions

Use [RELEVANT VIDEN] as your primary source for general questions about JKF, products, solutions and company information. Stick strictly to what is in it. Do not invent information, and do not supplement with your general knowledge about the world — especially not with external websites, URLs, or facts not present in the knowledge base.

### 2. Live stock and item tools (available to all users)
All stock data is cached locally and updated regularly from Business Central. These tools are fast and do not require a live BC connection for basic lookups.

Use these when a user asks about a specific item, stock level, price or availability:
- **search_items**: Use when the user describes a product in natural language and the item number is unknown. Searches the local item cache by description. Pass the user's description directly.
- **get_item_inventory**: Returns current stock level and price for a known item number. Uses the local cache first; supplements with live data for price if needed.
- **get_item_details**: Returns full details for a known item number including price, lead time, stock and availability status.

When to use these vs [RELEVANT VIDEN]:
- General "what products do you have for X application?" -> [RELEVANT VIDEN] (catalogues)
- "Do you have item 1234 in stock?" or "How many of X do you have?" -> item tools (live stock data)
- Both sources may be useful together for a complete answer.

### 3. Order tools (require identity verification for anonymous users)
Use these when a user asks about a specific order or shipment status:
- **verify_customer_identity**: MUST be called first for non-logged-in users. Call immediately once you have the order number AND a customer number — do NOT ask for confirmation first.
  - **Always ask for kundenummer** — it works for both open and posted/shipped orders.
  - Order references accepted: BC ordrenr (e.g. 0435897), forsendelsesnr (F-237780), or eksternt bilagsnr/kundens eget ordrenr (e.g. 4500041080).
  - If the user says "jeg er 1788", "kundenr. 1788", "kunde 2680" — that is their kundenummer, use it immediately.
  - Only ask for email if the customer explicitly cannot provide their kundenummer.
- **get_order_status**: Returns order status, and a 'lines' list with each order line's item_no, quantity, outstanding_quantity, and planned_shipment_date. Always report the full line detail to the customer — items ordered, quantities, and per-line planned shipment dates. Only call after identity is verified.
- **get_order_shipment**: Returns tracking number, carrier and tracking link. Only call after identity is verified. Always call this for posted/shipped orders to get tracking info.
- For logged-in users (JKF Universe), identity is pre-confirmed — call get_order_status or get_order_shipment immediately without any verification step. If get_order_status returns found=false, automatically call get_order_shipment (order is likely posted/shipped).

### Decision guide
1. General question about JKF, products or solutions -> [RELEVANT VIDEN]
2. "What products do you have for dust extraction?" -> [RELEVANT VIDEN] (product catalogues)
3. "Do you have item 1234 in stock?" or "What does it cost?" -> item tools
4. "Where is my order S-000123?" -> verify identity if needed, then order tools
5. No answer found anywhere -> say so honestly and offer to help the user reach JKF

## Metadata requirement
Append after every response without exception:
[METADATA]
{"language": "[Language used, in Danish]", "category": "[Topic/Category in Danish]", "knowledge_gaps": "[0 or 1]"}
[/METADATA]"""


LANGUAGE_MAP = {
    'danish': 'Dansk', 'dansk': 'Dansk', 'da': 'Dansk',
    'english': 'Engelsk', 'engelsk': 'Engelsk', 'en': 'Engelsk',
    'german': 'Tysk', 'tysk': 'Tysk', 'de': 'Tysk',
    'swedish': 'Svensk', 'svensk': 'Svensk', 'sv': 'Svensk',
    'norwegian': 'Norsk', 'norsk': 'Norsk', 'no': 'Norsk', 'nb': 'Norsk',
    'french': 'Fransk', 'fransk': 'Fransk', 'fr': 'Fransk',
    'polish': 'Polsk', 'polsk': 'Polsk', 'pl': 'Polsk',
    'dutch': 'Hollandsk', 'hollandsk': 'Hollandsk', 'nl': 'Hollandsk',
    'spanish': 'Spansk', 'spansk': 'Spansk', 'es': 'Spansk',
    'unknown': 'Ukendt',
}

CATEGORY_MAP = {
    'general inquiry': 'Generel henvendelse',
    'generel henvendelse': 'Generel henvendelse',
    'pricing': 'Priser', 'priser': 'Priser',
    'support': 'Support',
    'contact': 'Kontakt', 'kontakt': 'Kontakt',
    'product': 'Produkt', 'produkt': 'Produkt',
    'other': 'Andet', 'andet': 'Andet',
    'unknown': 'Ukendt',
}


def normalize_metadata(metadata):
    if not metadata:
        return metadata
    n = dict(metadata)
    if 'language' in n and n['language']:
        n['language'] = LANGUAGE_MAP.get(n['language'].lower(), n['language'])
    if 'category' in n and n['category']:
        n['category'] = CATEGORY_MAP.get(n['category'].lower(), n['category'])
    return n


def needs_classification_fallback(metadata):
    if not metadata:
        return True
    return not metadata.get('language') or not metadata.get('category')


def fallback_classify(user_msg, assistant_response):
    try:
        resp = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[{"role": "user", "content": (
                "Klassificer denne samtale.\n"
                'Returner KUN JSON: {"language": "...", "category": "..."}\n'
                "language = sproget på dansk (f.eks. Dansk, Engelsk)\n"
                "category = kort emne på dansk (f.eks. Leveringstid, Priser)\n\n"
                f"Bruger: {user_msg[:400]}\nAssistent: {assistant_response[:400]}"
            )}],
            max_tokens=80, temperature=0
        )
        return json.loads(resp.choices[0].message.content.strip())
    except Exception as e:
        logger.warning(f"Fallback classification failed: {e}")
        return {}


_DANISH_STOPWORDS = {
    'og', 'i', 'er', 'det', 'at', 'en', 'den', 'til', 'de', 'et', 'der',
    'som', 'på', 'med', 'af', 'for', 'ikke', 'var', 'om', 'men', 'vi',
    'han', 'hun', 'du', 'jeg', 'har', 'da', 'fra', 'men', 'sig', 'ham',
    'kan', 'skal', 'vil', 'være', 'når', 'ud', 'op', 'se', 'dem', 'os',
    'eller', 'hvis', 'alle', 'man', 'sin', 'sit', 'denne', 'dette', 'disse',
    'så', 'nu', 'her', 'hen', 'hvad', 'hvem', 'hvor', 'hvordan', 'hvorfor',
    'lige', 'selv', 'efter', 'over', 'under', 'ind', 'ned', 'ved', 'også',
    'the', 'and', 'or', 'in', 'on', 'at', 'to', 'of', 'is', 'are', 'was',
    'for', 'with', 'this', 'that', 'be', 'by', 'an', 'it', 'as', 'from',
}

def _compute_sparse_vector(text: str):
    """BM25-weighted sparse vector with Danish stopword filtering.
    Uses hash-bucketed indices (100k buckets) compatible with Qdrant sparse vectors.
    NOTE: must match the implementation in scrape_website.py exactly.
    """
    import hashlib
    from collections import Counter
    words = re.sub(r'[^\w\s]', ' ', text.lower()).split()
    words = [w for w in words if len(w) > 1 and w not in _DANISH_STOPWORDS]
    if not words:
        return [], []
    counts = Counter(words)
    doc_len = len(words)
    # BM25 parameters
    k1, b, avg_len = 1.5, 0.75, 100.0
    seen: dict = {}
    indices, values = [], []
    for word, tf in counts.items():
        bm25 = (tf * (k1 + 1)) / (tf + k1 * (1 - b + b * (doc_len / avg_len)))
        idx = int(hashlib.sha1(word.encode()).hexdigest(), 16) % 100000
        if idx in seen:
            values[seen[idx]] += bm25
        else:
            seen[idx] = len(indices)
            indices.append(idx)
            values.append(bm25)
    return indices, values


def _rerank_chunks(query: str, chunks: list, top_k: int = 5) -> list:
    if len(chunks) <= top_k:
        return chunks
    api_key = os.environ.get("JINA_API_KEY", "")
    if api_key:
        for attempt in range(3):
            try:
                resp = requests.post(
                    "https://api.jina.ai/v1/rerank",
                    headers={"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"},
                    json={
                        "model": "jina-reranker-v2-base-multilingual",
                        "query": query,
                        "documents": [c.get("text", "")[:2000] for c in chunks],
                        "top_n": top_k,
                    },
                    timeout=8,
                )
                resp.raise_for_status()
                results = resp.json().get("results", [])
                if results:
                    return [chunks[r["index"]] for r in results]
                break
            except Exception as e:
                wait = 2 ** attempt
                logger.warning(f"Jina reranking attempt {attempt+1} failed ({e}), retrying in {wait}s")
                time.sleep(wait)
    # Fallback: LLM reranking with scoring rubric
    try:
        passages = "\n\n".join(
            f"[{i}] {c.get('text', '')[:600]}" for i, c in enumerate(chunks)
        )
        resp = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[{"role": "user", "content": (
                f"Query: {query}\n\nPassages:\n{passages}\n\n"
                "Score each passage 1-5 for relevance to the query "
                "(5=directly answers it, 1=unrelated). "
                "Return JSON with key 'order' listing passage indices from most to least relevant. "
                'Example: {"order": [2, 0, 3, 1, 4]}'
            )}],
            response_format={"type": "json_object"},
            max_tokens=150, temperature=0,
        )
        data = json.loads(resp.choices[0].message.content)
        order = data.get("order", list(range(len(chunks))))
        reranked = [chunks[i] for i in order[:top_k] if isinstance(i, int) and i < len(chunks)]
        if reranked:
            return reranked
    except Exception as e:
        logger.warning(f"LLM reranking failed: {e}")
    return chunks[:top_k]


def _generate_hyde_query(user_message: str, prior_user_turns: list = None) -> str:
    """Generate a hypothetical KB passage for HyDE (used ONLY for the dense search vector).

    Higher temperature gives more lexical diversity, which improves recall in the
    vector index. The generated text is never shown to users.
    """
    history_hint = ""
    if prior_user_turns:
        recent = " | ".join(prior_user_turns[-5:])
        history_hint = f"Samtalekontekst: {recent}\n\n"
    prompt = (
        f"{history_hint}"
        f'En bruger spørger: "{user_message}"\n\n'
        "Skriv ét sammenhængende afsnit (4-6 sætninger) der KUNNE stå på en "
        "industrivirksomheds hjemmeside eller i en FAQ og som ville besvare dette "
        "spørgsmål præcist. Brug brancherelevante termer, produktnavne og varenumre "
        "der ville forekomme på siden. Skriv som faktaindhold, ikke som svar til "
        "brugeren. Brug samme sprog som spørgsmålet."
    )
    try:
        resp = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[{"role": "user", "content": prompt}],
            max_tokens=250, temperature=0.4,
        )
        return resp.choices[0].message.content.strip() or user_message
    except Exception as e:
        logger.warning(f"HyDE generation failed: {e}")
        return user_message


def get_context_from_qdrant(user_message: str, top_k: int = 7,
                             prior_user_turns: list = None) -> str:
    """Hybrid RAG search: HyDE dense + BM25 sparse, RRF fusion, Jina rerank.

    Retrieval flow:
    1. HyDE: generate a hypothetical passage for dense embedding
    2. BM25 sparse vector from original query (keyword signal)
    3. Qdrant hybrid search with RRF fusion (falls back to dense-only on error)
    4. Jina multilingual reranker (falls back to LLM reranker)
    5. Deduplicated context assembly (one Kilde per unique source URL)
    """
    qc = get_qdrant_client()
    if not qc:
        return ""
    try:
        query_text = user_message
        if prior_user_turns:
            query_text = (" ".join(prior_user_turns) + " " + user_message)[-2000:]

        hyde_text = _generate_hyde_query(user_message, prior_user_turns)
        embedding_resp = client.embeddings.create(model="text-embedding-3-large", input=hyde_text)
        dense_vector = embedding_resp.data[0].embedding

        sp_indices, sp_values = _compute_sparse_vector(query_text)

        CANDIDATE_K = 20
        results = None
        try:
            results = qc.query_points(
                collection_name=QDRANT_COLLECTION,
                prefetch=[
                    Prefetch(query=dense_vector, using="dense", limit=CANDIDATE_K),
                    Prefetch(query=SparseVector(indices=sp_indices, values=sp_values),
                             using="sparse", limit=CANDIDATE_K),
                ],
                query=FusionQuery(fusion=Fusion.RRF),
                limit=CANDIDATE_K,
            ).points
        except Exception as hybrid_err:
            logger.warning(f"Hybrid search failed ({hybrid_err}), falling back to dense-only")
            results = qc.query_points(
                collection_name=QDRANT_COLLECTION,
                query=dense_vector,
                using="dense",
                limit=CANDIDATE_K,
            ).points

        if not results:
            return ""

        chunks = [
            {"text": r.payload.get("text", ""),
             "section_title": r.payload.get("section_title", ""),
             "source_url": r.payload.get("source_url", ""),
             "filename": r.payload.get("filename", ""),
             "source_type": r.payload.get("source_type", "")}
            for r in results
        ]
        chunks = _rerank_chunks(user_message, chunks, top_k=top_k)

        # Assemble context — URL is always taken from metadata (source_url payload),
        # never from embedded text. This matches the ConvoTech approach and ensures
        # the correct page URL is always used, even after chunk splitting.
        seen_urls: set = set()
        parts = []
        for c in chunks:
            entry = (f"[{c['section_title']}]\n" if c['section_title'] else "") + c['text']
            url = c.get('source_url', '')
            filename = c.get('filename', '')
            source_type = c.get('source_type', '')
            if url and url not in seen_urls:
                entry += f"\n[Læs mere]({url})"
                seen_urls.add(url)
            elif source_type == 'document' and filename:
                entry += f"\nKilde: dokument '{filename}'"
            parts.append(entry)
        return "\n\n---\n\n".join(parts)
    except Exception as e:
        logger.error(f"Qdrant search failed: {e}")
        return ""


def load_thread_history(thread_id: str, limit: int = 20) -> list:
    rows = (Message.query
            .filter_by(thread_id=thread_id)
            .order_by(Message.timestamp)
            .limit(limit)
            .all())
    history = []
    for msg in rows:
        content = re.sub(r'<[^>]+>', ' ', msg.content or "")
        content = re.sub(r'\s+', ' ', content).strip()
        content = re.sub(r'\[METADATA\].*?\[/METADATA\]', '', content, flags=re.DOTALL).strip()
        if content:
            history.append({"role": msg.role, "content": content})
    return history


def format_message(text: str) -> str:
    """Convert markdown-ish assistant text to HTML for the chat widget."""
    text = re.sub(r'【[^】]*】', '', text)
    text = re.sub(r'\[([^\]]+)\]\((https?://[^)]+)\)', r'<a href="\2" target="_blank">\1</a>', text)
    text = re.sub(r'\[([^\]]+)\]\((mailto:[^)]+)\)', r'<a href="\2">\1</a>', text)
    text = re.sub(r'\[([^\]]+)\]\((/[^)]*)\)', r'<a href="\2" target="_blank">\1</a>', text)
    text = re.sub(r'(?<!["\'>])(https?://[^\s<>")\]]+)', r'<a href="\1" target="_blank">\1</a>', text)
    text = re.sub(r'\*\*(.*?)\*\*', r'<strong>\1</strong>', text)
    text = re.sub(r'^###\s*(.*?)$', r'<strong>\1</strong>', text, flags=re.MULTILINE)
    lines = text.split('\n')
    formatted, in_list = [], False
    for line in lines:
        line = line.strip()
        if not line:
            continue
        if line.startswith('-'):
            if not in_list:
                formatted.append('<ul>')
                in_list = True
            formatted.append(f'<li>{line[1:].strip()}</li>')
        else:
            if in_list:
                formatted.append('</ul>')
                in_list = False
            formatted.append(f'<p>{line}</p>')
    if in_list:
        formatted.append('</ul>')
    return ''.join(formatted)


def process_stream_chunk(chunk_text: str, buf: dict):
    buf['raw_text'] += chunk_text
    text = buf['raw_text']

    m = re.search(r'\[METADATA\]\s*(\{.*?\})\s*\[/METADATA\]', text, re.DOTALL)
    if m:
        try:
            buf['metadata'] = json.loads(m.group(1))
        except json.JSONDecodeError:
            pass
        text = re.sub(r'\[METADATA\].*?\[/METADATA\]', '', text, flags=re.DOTALL)
        buf['raw_text'] = text

    # Don't send while partial METADATA tag is accumulating
    meta_start = '[METADATA]'
    for i in range(1, len(meta_start)):
        if text.endswith(meta_start[:i]):
            return None, buf
    if '[METADATA]' in text and '[/METADATA]' not in text:
        return None, buf

    text = re.sub(r'【[^】]*】', '', text)
    buf['processed_text'] = format_message(text)
    return buf['processed_text'], buf


def get_jkf_settings() -> JKFSettings:
    s = JKFSettings.query.first()
    if not s:
        s = JKFSettings()
        db.session.add(s)
        db.session.commit()
    return s


def _get_ai_client(model_name: str):
    if "/" in model_name and openrouter_client:
        return openrouter_client
    return client


# ─────────────────────────────────────────────────────────────────────────────
# Business Central integration helpers
# ─────────────────────────────────────────────────────────────────────────────
_bc_token_cache: dict = {}  # {tenant_id: {token, expires_at}}


def get_bc_integration() -> "Integration | None":
    """Return the enabled Business Central integration row, or None."""
    return Integration.query.filter_by(
        integration_type='business_central', enabled=True
    ).first()


def get_bc_config(integration: "Integration") -> dict:
    try:
        return json.loads(integration.config or '{}')
    except Exception:
        return {}


def get_bc_token(config: dict) -> str:
    """Obtain (and cache) an Azure AD OAuth2 access token for BC."""
    tenant_id = config.get('tenant_id', '')
    client_id = config.get('client_id', '')
    client_secret = config.get('client_secret', '')

    cached = _bc_token_cache.get(tenant_id)
    if cached and cached['expires_at'] > time.time() + 60:
        return cached['token']

    url = f"https://login.microsoftonline.com/{tenant_id}/oauth2/v2.0/token"
    resp = requests.post(url, data={
        'grant_type': 'client_credentials',
        'client_id': client_id,
        'client_secret': client_secret,
        'scope': 'https://api.businesscentral.dynamics.com/.default',
    }, timeout=15)
    resp.raise_for_status()
    data = resp.json()
    token = data['access_token']
    _bc_token_cache[tenant_id] = {
        'token': token,
        'expires_at': time.time() + int(data.get('expires_in', 3600)),
    }
    return token


def bc_request(config: dict, path: str, custom: bool = False) -> dict:
    """Make an authenticated GET request to the BC OData API.

    custom=True uses the custom API base (/api/{publisher}/{group}/{version}/...)
    instead of the standard v2.0 base.
    """
    tenant_id  = config.get('tenant_id', '')
    env        = config.get('environment', 'production')
    company_id = config.get('company_id', '')
    token      = get_bc_token(config)

    if custom:
        pub = config.get('custom_api_publisher', '')
        grp = config.get('custom_api_group', '')
        ver = config.get('custom_api_version', 'v1.0')
        base = (
            f"https://api.businesscentral.dynamics.com/v2.0/"
            f"{tenant_id}/{env}/api/{pub}/{grp}/{ver}/companies({company_id})"
        )
    else:
        base = (
            f"https://api.businesscentral.dynamics.com/v2.0/"
            f"{tenant_id}/{env}/api/v2.0/companies({company_id})"
        )

    resp = requests.get(
        f"{base}{path}",
        headers={'Authorization': f'Bearer {token}', 'Accept': 'application/json'},
        timeout=15,
    )
    resp.raise_for_status()
    return resp.json()


def _has_custom_api(config: dict) -> bool:
    """True when all three custom API identifiers are configured."""
    return bool(
        config.get('custom_api_publisher', '').strip() and
        config.get('custom_api_group', '').strip() and
        config.get('custom_api_version', '').strip()
    )


def _bc_date(val) -> str | None:
    """Return date string or None — treats BC's min-date sentinel (0001-...) as absent."""
    if not val or str(val).startswith('0001'):
        return None
    return str(val)[:10]  # trim time portion if present


# Carrier code (BC shippingAgentCode) → tracking URL template.
# {t} = tracking number. URLs without {t} link to the carrier's tracking page
# (customer pastes the number manually).
_BC_CARRIER_URLS: dict = {
    # Danish / Nordic carriers
    'GLS':       'https://gls-group.com/track/{t}',
    'POSTNORD':  'https://tracking.postnord.com/tracking?id={t}',
    'POST':      'https://tracking.postnord.com/tracking?id={t}',
    'DAO':       'https://dao.as/tools/tracking/?shipmentid={t}',
    'BRING':     'https://tracking.bring.com/tracking/{t}',
    'FREJA':     'https://en.freja.com/online-services/track-and-trace-road/',  # form-based, no deep-link
    'ALPI':      'https://www.alpi.dk/e-services/',  # form-based, no deep-link
    # BACH = regional Danish carrier, no public tracking portal found
    # International carriers
    'DHL':       'https://www.dhl.com/en/express/tracking.html?AWB={t}',
    'DHLFREIGHT':'https://freight.dhl.com/en/tracking.html?id={t}',
    'UPS':       'https://www.ups.com/track?tracknum={t}',
    'TNT':       'https://www.tnt.com/express/en_gb/site/tracking.html?searchType=CON&cons={t}',
    'FEDEX':     'https://www.fedex.com/apps/fedextrack/?tracknumbers={t}',
    'SCHENKER':  'https://www.dbschenker.com/app/tracking-public/?refNumber={t}',
    'DSV':       'https://www.dsv.com/en/our-solutions/modes-of-transport/road-transport/online-services-and-document-handling',
}

# In-memory store: thread_id → verified customer email (normalised lowercase)
_bc_verified_sessions: dict = {}

# In-memory store: thread_id → BC customer account number from JKF Universe login
_bc_thread_customer_no: dict = {}

# Shared secret for verifying HMAC-signed company tokens from JKF Universe
CHATBOT_HMAC_SECRET = os.environ.get('CHATBOT_HMAC_SECRET', '')


def _verify_and_extract_customer_no(token: str) -> "str | None":
    """Verify an HMAC-signed company token and return the customer account number.

    Token format: "{customer_no}:{sha256_hmac_hex}"
    If CHATBOT_HMAC_SECRET is not configured, the token is accepted as-is (plain
    customer number) so the feature degrades gracefully during initial rollout.
    """
    if not token or not isinstance(token, str):
        return None
    token = token.strip()
    if not CHATBOT_HMAC_SECRET:
        logger.warning("CHATBOT_HMAC_SECRET not set — accepting company_id without HMAC verification")
        return token or None
    if ':' not in token:
        return None
    customer_no, provided_sig = token.rsplit(':', 1)
    expected_sig = hmac.new(
        CHATBOT_HMAC_SECRET.encode(),
        customer_no.encode(),
        'sha256',
    ).hexdigest()
    if not hmac.compare_digest(expected_sig, provided_sig):
        logger.warning("Invalid HMAC for company_token — rejecting")
        return None
    return customer_no.strip() or None


def _bc_tracking_url(shipper: str, tracking_no: str):
    """Return a tracking URL for the given carrier and tracking number.

    If the carrier URL contains {t}, the tracking number is embedded directly.
    If not (form-based carriers like FREJA), the page URL is returned as-is so
    the customer at least gets a clickable link to the carrier's tracking page.
    Returns None if the carrier is unknown or tracking_no is empty.
    """
    if not tracking_no:
        return None
    pattern = _BC_CARRIER_URLS.get((shipper or '').upper().strip())
    if not pattern:
        return None
    return pattern.replace('{t}', tracking_no)


def bc_get_order(config: dict, order_no: str,
                 verified_email: str = None,
                 customer_no: str = None) -> dict:
    """Fetch a sales order by number.

    Tries the custom API page first (salesOrdersEx) when configured — this
    returns both standard and any custom fields your BC specialist has exposed.
    Falls back to the standard v2.0 salesOrders endpoint automatically.

    If customer_no is given (JKF Universe logged-in user), the OData filter
    includes sellToCustomerNumber so BC itself enforces data scoping.
    If verified_email is given (anonymous flow), the email must match the order.
    """
    o = None

    # Try to find the order by BC number or external document number (customer's own PO ref).
    for order_filter in [f"number eq '{order_no}'", f"externalDocumentNumber eq '{order_no}'"]:
        if o is not None:
            break
        if _has_custom_api(config):
            try:
                data = bc_request(config, f"/salesOrders?$filter={order_filter}&$top=1", custom=True)
                items = data.get('value', [])
                if items:
                    o = items[0]
                    break
            except Exception as e:
                logger.debug(f"Custom BC salesOrders not available for {order_no}: {e}")

        if o is None:
            try:
                data = bc_request(config, f"/salesOrders?$filter={order_filter}&$top=1")
                items = data.get('value', [])
                if items:
                    o = items[0]
            except Exception:
                pass

    # Fallback: BC v2.0 may not support filtering salesOrders by externalDocumentNumber.
    # If we have a customer number, fetch that customer's orders and match in Python.
    if o is None and customer_no:
        cust_norm = customer_no.strip().lstrip('0')
        for cust_fmt in [cust_norm, customer_no.strip()]:
            if not cust_fmt:
                continue
            try:
                data = bc_request(config, f"/salesOrders?$filter=customerNumber eq '{cust_fmt}'&$top=100")
                all_orders = data.get('value', [])
                matched = [x for x in all_orders
                           if x.get('externalDocumentNumber', '').strip() == order_no
                           or x.get('number', '').strip() == order_no]
                if matched:
                    o = matched[0]
                    break
            except Exception:
                pass

    if o is None:
        return {'found': False, 'order_number': order_no,
                'hint': 'Not found in open orders. Order may be posted/shipped — call get_order_shipment.'}

    # Verify customer ownership in Python — normalise both sides (BC may include leading zeros)
    if customer_no is not None:
        bc_customer = (o.get('customerNumber') or o.get('sellToCustomerNumber') or '').strip().lstrip('0')
        if bc_customer != customer_no.strip().lstrip('0'):
            return {
                'found': False,
                'error': 'unauthorized',
                'message': 'Ordren tilhører ikke din konto.',
            }

    if verified_email is not None:
        order_email = (o.get('email') or o.get('sellToEmail') or o.get('billToEmail') or '').lower().strip()
        if not order_email or verified_email != order_email:
            return {
                'found': False,
                'error': 'unauthorized',
                'message': 'Ordren tilhører ikke den verificerede e-mailadresse.',
            }

    result = {
        'found':          True,
        'order_number':   o.get('number'),
        'status':         o.get('status'),          # Open / Released / Pending Approval / etc.
        'fully_shipped':  o.get('fullyShipped'),
        'partial_shipping': o.get('partialShipping'),
        # planned_shipment_date = "Planlagt afsendelsesdato" in BC UI (shipmentDate field)
        'planned_shipment_date': _bc_date(o.get('shipmentDate')),
        # confirmed_delivery_date — may come from custom field or requestedDeliveryDate
        'confirmed_delivery_date': _bc_date(
            o.get('confirmedShipmentDate')
            or o.get('estimatedDeliveryDate')
            or o.get('requestedDeliveryDate')
        ),
        'customer_name':  o.get('customerName'),
        'ship_to_name':   o.get('shipToName') or None,
        'ship_to_country': o.get('shipToCountry') or None,
        'total_amount':   o.get('totalAmountIncludingTax'),
        'currency':       o.get('currencyCode', 'DKK'),
        'salesperson':    o.get('salesperson') or None,
        'last_modified':  _bc_date(o.get('lastModifiedDateTime', '')[:10] if o.get('lastModifiedDateTime') else None),
    }
    # Remove None values so the AI doesn't get confused by explicit nulls
    result = {k: v for k, v in result.items() if v is not None}
    result['found'] = True  # always keep

    # Attach custom fields only when present
    for custom_field in ('trackingNumber', 'carrierCode', 'deliveryNote', 'internalComment'):
        if o.get(custom_field) is not None:
            result[custom_field] = o[custom_field]

    # Build tracking URL from custom carrier + tracking fields on the order
    carrier = o.get('carrierCode')
    tracking = o.get('trackingNumber')
    if carrier and tracking:
        result['tracking_url'] = _bc_tracking_url(carrier, tracking)

    # Fetch order lines from Patrick's custom API (quantities + planned dates per line)
    bc_order_no = o.get('number', '')
    if bc_order_no and _has_custom_api(config):
        # Pass customer_no so we can filter by Sell_to_Customer_No_ (document key not filterable)
        order_cust = customer_no or o.get('customerNumber') or o.get('sellToCustomerNumber') or ''
        lines = bc_get_order_lines(config, bc_order_no, customer_no=order_cust)
        if lines:
            result['lines'] = lines

    return result


def bc_get_order_lines(config: dict, document_no: str, customer_no: str = None) -> list:
    """Fetch sales order lines from Patrick's custom Salesorderlines endpoint.

    Returns a list of line dicts with item_no, quantity, outstanding_quantity,
    and planned_shipment_date. Lines without an item number (comment/text lines)
    are excluded.
    """
    try:
        # Document_No_ is the middle field of Sales Line's composite key —
        # BC Query APIs can't filter on it. Filter by Sell_to_Customer_No_ instead
        # (an indexed regular field), then match by document number in Python.
        raw_lines = []

        # Primary strategy: filter by customer number, match document in Python
        if customer_no:
            cust_stripped = customer_no.strip().lstrip('0')
            for cust_fmt in [customer_no.strip(), f"0{cust_stripped}", cust_stripped]:
                if not cust_fmt:
                    continue
                try:
                    data = bc_request(
                        config,
                        f"/Salesorderlines?$filter=Sell_to_Customer_No_ eq '{cust_fmt}'",
                        custom=True,
                    )
                    raw_lines = data.get('value', [])
                    if raw_lines is not None:
                        break
                except Exception:
                    pass

        # Fallback: try document number filters directly (works if Patrick changes the query)
        if not raw_lines:
            for attempt_filter in [
                f"Document_No_ eq '{document_no}'",
                f"documentNo eq '{document_no}'",
            ]:
                try:
                    data = bc_request(
                        config,
                        f"/Salesorderlines?$filter={attempt_filter}",
                        custom=True,
                    )
                    raw_lines = data.get('value', [])
                    if raw_lines is not None:
                        break
                except Exception:
                    pass

        lines = []
        for item in raw_lines:
            doc = (item.get('Document_No_') or item.get('documentNo') or '').strip()
            if doc and doc != document_no:
                continue  # Python-side filter when we fetched by customer
            item_no = (item.get('No_') or item.get('no') or '').strip()
            if not item_no:
                continue  # skip comment / text lines
            line = {
                'item_no':             item_no,
                'quantity':            item.get('Quantity') or item.get('quantity'),
                'outstanding_quantity': item.get('Outstanding_Quantity') or item.get('outstandingQuantity'),
                'planned_shipment_date': _bc_date(item.get('Planned_Shipment_Date') or item.get('plannedShipmentDate')),
            }
            line = {k: v for k, v in line.items() if v is not None}
            lines.append(line)
        return lines
    except Exception as e:
        logger.warning(f"BC order lines fetch failed for {document_no}: {e}")
        return []


def _bc_find_std_shipments(config: dict, ref: str, top: int = 5) -> list:
    """Search salesShipments by orderNumber, shipment number, or external document number.

    Tries three filters in order so customers can use any reference they have:
    1. orderNumber eq '...'  — BC internal order no (e.g. 0435897)
    2. number eq '...'       — shipment no (e.g. F-237780)
    3. externalDocumentNumber eq '...' — customer's own PO / reference (e.g. 4500041080)
    """
    for flt in [
        f"orderNumber eq '{ref}'",
        f"number eq '{ref}'",
        f"externalDocumentNumber eq '{ref}'",
    ]:
        try:
            data = bc_request(config, f"/salesShipments?$filter={flt}&$top={top}")
            items = data.get('value', [])
            if items:
                return items
        except Exception:
            continue
    return []


def _bc_find_custom_shipments(config: dict, ref: str, top: int = 5) -> list:
    """Search salesShipmentsEx (Patrick's custom endpoint) by order no or shipment no."""
    for flt in [
        f"orderNo eq '{ref}'",
        f"no eq '{ref}'",
    ]:
        try:
            data = bc_request(config, f"/salesShipmentsEx?$filter={flt}&$top={top}", custom=True)
            items = data.get('value', [])
            if items:
                return items
        except Exception:
            continue
    return []


def bc_get_shipment(config: dict, order_no: str,
                    verified_email: str = None,
                    customer_no: str = None) -> dict:
    """Fetch posted shipment / tracking info for an order.

    Tries the custom API page first (salesShipmentsEx) when configured.
    Falls back to the standard v2.0 salesShipments endpoint automatically.
    Searches by orderNumber, shipment number (F-XXXXXX), or external document number.

    For customer_no: ownership is verified on the shipment record itself,
    NOT via bc_get_order — because posted orders no longer appear in salesOrders.
    For verified_email: pre-checks via bc_get_order (anonymous flow, open orders only).
    """
    if verified_email is not None:
        check = bc_get_order(config, order_no, verified_email=verified_email)
        if not check.get('found'):
            return check

    raw_items = None

    # Step 1: standard salesShipments first — resolves any reference type to a real record.
    # Searches by orderNumber, shipment number (F-XXXXXX), and external document number.
    std_items = _bc_find_std_shipments(config, order_no)

    # Step 2: custom salesShipmentsEx (Patrick's endpoint) — has packageTrackingNo.
    # Try original ref first; if not found (e.g. user gave an external doc number),
    # retry with the real orderNumber or shipment number resolved from std result.
    custom_items = []
    if _has_custom_api(config):
        custom_items = _bc_find_custom_shipments(config, order_no)
        if not custom_items and std_items:
            for resolved_ref in filter(None, [
                std_items[0].get('orderNumber', ''),
                std_items[0].get('number', ''),
            ]):
                custom_items = _bc_find_custom_shipments(config, resolved_ref)
                if custom_items:
                    break

    # Prefer custom endpoint (has packageTrackingNo); fall back to standard
    raw_items = custom_items or std_items

    if not raw_items:
        return {'found': False, 'order_number': order_no}

    # Verify customer ownership using standard endpoint (has customerNumber field).
    # salesShipmentsEx does NOT include customerNo yet — use std_items for the check.
    if customer_no is not None:
        check_records = std_items or raw_items
        bc_customer = (
            check_records[0].get('customerNumber')
            or check_records[0].get('sellToCustomerNumber')
            or check_records[0].get('sellToCustomerNo')
            or ''
        ).strip().lstrip('0')
        if bc_customer and bc_customer != customer_no.strip().lstrip('0'):
            return {
                'found': False,
                'error': 'unauthorized',
                'message': 'Forsendelsen tilhører ikke din konto.',
            }

    # Merge: if we used custom_items, supplement with shipmentDate from std_items
    std_by_no = {}
    if std_items:
        for s in std_items:
            std_by_no[s.get('no') or s.get('number', '')] = s

    shipments = []
    for s in raw_items:
        shipment_no = s.get('no') or s.get('number')
        std = std_by_no.get(shipment_no, {})

        tracking_no = s.get('packageTrackingNo') or std.get('packageTrackingNo') or s.get('trackingNumber')
        carrier     = s.get('shippingAgentCode') or std.get('shippingAgentCode') or s.get('carrierCode')
        ship_date   = s.get('shipmentDate') or std.get('shipmentDate')
        tracking_url = _bc_tracking_url(carrier, tracking_no)

        entry = {
            'shipment_number': shipment_no,
            'carrier':         carrier,
        }
        if ship_date and not str(ship_date).startswith('0001'):
            entry['shipment_date'] = str(ship_date)
        if tracking_no:
            entry['tracking_number'] = tracking_no
        if tracking_url:
            entry['tracking_url'] = tracking_url

        shipments.append(entry)

    return {'found': True, 'order_number': order_no, 'shipments': shipments}


def bc_verify_customer(config: dict, order_no: str, email: str, thread_id: str,
                       customer_number: str = None) -> dict:
    """Verify customer identity against a BC order or posted shipment.

    Strategy:
    1. Try salesOrders (open orders) → verify by customer number OR email
    2. If not found there, try salesShipments (posted orders) → verify by customer number
    Both steps accept customer_number as the primary identifier; email is a fallback.
    """
    email_norm = (email or '').lower().strip()
    cust_no_norm = (customer_number or '').strip().lstrip('0')  # normalise: "01788" → "1788"

    try:
        # 1. Open orders — verify by customer number (preferred) or email
        items = []
        for flt in [f"number eq '{order_no}'", f"externalDocumentNumber eq '{order_no}'"]:
            try:
                data = bc_request(config, f"/salesOrders?$filter={flt}&$top=1")
                items = data.get('value', [])
                if items:
                    break
            except Exception:
                pass

        # Fallback: BC v2.0 may not support filtering by externalDocumentNumber.
        # If we have a customer number, fetch that customer's open orders and match in Python.
        if not items and cust_no_norm:
            for cust_fmt in [cust_no_norm, customer_number or '']:
                if not cust_fmt:
                    continue
                try:
                    data = bc_request(config, f"/salesOrders?$filter=customerNumber eq '{cust_fmt}'&$top=100")
                    all_orders = data.get('value', [])
                    matched = [o for o in all_orders
                               if o.get('externalDocumentNumber', '').strip() == order_no
                               or o.get('number', '').strip() == order_no]
                    if matched:
                        items = matched[:1]
                        break
                except Exception:
                    pass

        if items:
            o = items[0]
            order_cust_raw = (o.get('customerNumber') or o.get('sellToCustomerNumber') or '').strip()
            order_cust = order_cust_raw.lstrip('0')
            order_email = (o.get('email') or o.get('sellToEmail') or o.get('billToEmail') or '').lower().strip()

            # Customer number check (preferred — customers always know this)
            if cust_no_norm and order_cust and cust_no_norm == order_cust:
                # Store raw (un-normalised) customer number so BC OData queries work with it
                _bc_verified_sessions[thread_id] = f'__cust__{order_cust_raw}'
                return {'verified': True, 'message': 'Identity verified. You can now see your order details.'}

            # Email fallback
            if email_norm and order_email and email_norm == order_email:
                _bc_verified_sessions[thread_id] = email_norm
                return {'verified': True, 'message': 'Identity verified. You can now see your order details.'}

            if cust_no_norm:
                return {'verified': False, 'reason': 'customer_number_mismatch',
                        'message': 'Customer number does not match the order. Please try again.'}
            if email_norm:
                return {'verified': False, 'reason': 'email_mismatch',
                        'message': 'Email address does not match the order. Please try again.'}
            # Nothing provided
            return {'verified': False, 'reason': 'identity_required',
                    'message': f'Please provide your customer number (or the email address associated with order {order_no}).'}

        # 2. Not in open orders — try posted shipments (verify by customer number).
        # Searches by orderNumber, shipment number (F-XXXXXX), or external document number
        # so customers can use whatever reference they have.
        shipments = _bc_find_std_shipments(config, order_no, top=1)

        if shipments:
            s = shipments[0]
            bc_cust_raw = (s.get('customerNumber') or s.get('sellToCustomerNumber') or '').strip()
            bc_cust = bc_cust_raw.lstrip('0')
            if cust_no_norm and bc_cust and cust_no_norm == bc_cust:
                # Store raw customer number so BC OData queries work with it
                _bc_verified_sessions[thread_id] = f'__cust__{bc_cust_raw}'
                return {'verified': True,
                        'order_is_posted': True,
                        'message': 'Identity verified. Order is posted/shipped — fetching shipment details.'}
            if cust_no_norm:
                return {'verified': False, 'reason': 'customer_number_mismatch',
                        'message': 'Customer number does not match the order.'}
            return {'verified': False, 'reason': 'customer_number_required',
                    'message': f'Order {order_no} is posted. Please provide your customer number to verify.'}

        return {'verified': False, 'reason': 'order_not_found',
                'message': f'Order {order_no} was not found. Please check the order number.'}

    except Exception as e:
        logger.warning(f"BC verify_customer failed for order {order_no}: {e}")
        return {'verified': False, 'reason': 'error', 'message': str(e)}


def _bc_fetch_item(config: dict, item_no: str) -> dict:
    """Internal helper: fetch a single item row.

    Priority:
    1. Local SQLite cache (fast, no BC round-trip, populated by sync_bc_items)
    2. Custom API /items endpoint (Patrick's chatbotJKF API)
    3. Standard BC v2.0 items endpoint
    """
    # 1. Local cache — covers inventory, description, description2, lead_time
    cached = BCItem.query.filter_by(item_no=item_no).first()
    if cached:
        row = {
            'number':               cached.item_no,
            'displayName':          cached.description,
            'description2':         cached.description2,
            'inventory':            cached.inventory,
            'baseUnitOfMeasureCode': 'stk',
        }
        if cached.lead_time:
            row['salesLeadTimeEVM'] = cached.lead_time
        return row

    # 2. Custom API (Patrick's /items endpoint — NOT /itemsEx)
    if _has_custom_api(config):
        try:
            data = bc_request(
                config,
                f"/items?$filter=no eq '{item_no}'&$top=1",
                custom=True,
            )
            items = data.get('value', [])
            if items:
                i = items[0]
                # Normalise field names to match standard API shape
                if 'no' in i and 'number' not in i:
                    i['number'] = i['no']
                if 'description' in i and 'displayName' not in i:
                    i['displayName'] = i['description']
                return i
        except Exception as e:
            logger.warning(f"Custom BC items failed for {item_no}, falling back: {e}")

    # 3. Standard BC v2.0 items — filter by displayName is unreliable so try
    #    both common filter fields gracefully.
    for filter_expr in (f"number eq '{item_no}'", f"no eq '{item_no}'"):
        try:
            data = bc_request(
                config,
                f"/items?$filter={filter_expr}&$top=1"
                "&$select=number,displayName,inventory,unitPrice,"
                "leadTimeCalculation,baseUnitOfMeasureCode,blocked",
            )
            items = data.get('value', [])
            if items:
                return items[0]
        except Exception:
            pass

    return {}


def sync_bc_items() -> dict:
    """Fetch ALL items from BC custom items API (following @odata.nextLink pages)
    and upsert them into the local bc_items cache table.

    Requires custom API to be configured (publisher/group/version).
    Patrick's endpoint: /api/JKFIndustri/chatbotJKF/v1.0/companies(...)/items
    Fields expected: no (or number), description, description2, inventory
    """
    integration = get_bc_integration()
    if not integration:
        return {'success': False, 'message': 'Ingen BC integration konfigureret'}
    config = get_bc_config(integration)
    if not _has_custom_api(config):
        return {'success': False, 'message': 'Custom API ikke konfigureret (publisher/group/version mangler)'}

    tenant_id = config['tenant_id']
    env       = config['environment']
    company_id = config['company_id']
    pub = config['custom_api_publisher']
    grp = config['custom_api_group']
    ver = config['custom_api_version']
    token = get_bc_token(config)

    start_url = (
        f"https://api.businesscentral.dynamics.com/v2.0/"
        f"{tenant_id}/{env}/api/{pub}/{grp}/{ver}/companies({company_id})/items"
    )

    all_items = []
    next_link: "str | None" = start_url
    while next_link:
        resp = requests.get(
            next_link,
            headers={'Authorization': f'Bearer {token}', 'Accept': 'application/json'},
            timeout=30,
        )
        resp.raise_for_status()
        data = resp.json()
        batch = data.get('value', [])
        all_items.extend(batch)
        next_link = data.get('@odata.nextLink')
        logger.info(f"BC item sync: {len(all_items)} items fetched so far…")

    # Upsert into local cache
    count = 0
    for raw in all_items:
        # BC custom API may use 'no' (from No_) or 'number'
        item_no = (raw.get('no') or raw.get('number') or '').strip()
        if not item_no:
            continue
        desc      = raw.get('description') or raw.get('Description') or ''
        desc2     = raw.get('description2') or raw.get('Description_2') or raw.get('Description2') or ''
        inv       = float(raw.get('inventory') or raw.get('Inventory') or 0)
        lead_time = raw.get('salesLeadTimeEVM') or raw.get('leadTimeCalculation') or ''

        existing = BCItem.query.filter_by(item_no=item_no).first()
        if existing:
            existing.description  = desc
            existing.description2 = desc2
            existing.inventory    = inv
            existing.lead_time    = lead_time
            existing.synced_at    = datetime.utcnow()
        else:
            db.session.add(BCItem(
                item_no=item_no, description=desc,
                description2=desc2, inventory=inv, lead_time=lead_time,
            ))
        count += 1

    db.session.commit()
    logger.info(f"BC item sync complete: {count} items cached")
    return {'success': True, 'count': count}


def search_bc_items(query: str, limit: int = 5) -> list:
    """Search local bc_items cache by description using word matching.

    Strategy (most-to-least strict):
    1. AND on longest 3 words (most distinctive)
    2. AND on longest 2 words
    3. OR on all significant words (any match)
    Returns a list of BCItem objects.
    """
    # Danish dimension words that BC encodes differently (°, mm, etc.) — exclude
    # from match requirements so "90 grader" still finds "90°" items.
    _stopwords = {'grader', 'meter', 'styk', 'stk', 'antal', 'lager',
                  'har', 'mange', 'hvad', 'hvilke', 'type', 'typer'}

    raw_words = [w.strip('.,?!') for w in query.split()]
    words = [w for w in raw_words if len(w) > 2 and w.lower() not in _stopwords]
    if not words:
        return []

    # Sort longest-first — longer words are more distinctive product terms
    words_by_length = sorted(set(words), key=len, reverse=True)

    def word_condition(w):
        # Build search patterns — also try without trailing 'r' to handle Danish
        # plurals (e.g. "T-stykker" → also search "%T-stykke%" to match singular form).
        patterns = [f'%{w}%']
        if w.endswith('r') and len(w) > 4:
            patterns.append(f'%{w[:-1]}%')
        return db.or_(
            *[BCItem.description.ilike(p) for p in patterns],
            *[BCItem.description2.ilike(p) for p in patterns],
        )

    # Attempt 1: AND on top 3 most distinctive words
    anchor_words = words_by_length[:3]
    results = (BCItem.query
               .filter(db.and_(*[word_condition(w) for w in anchor_words]))
               .limit(limit)
               .all())

    # Attempt 2: AND on top 2 most distinctive words
    if not results and len(words_by_length) >= 2:
        anchor_words = words_by_length[:2]
        results = (BCItem.query
                   .filter(db.and_(*[word_condition(w) for w in anchor_words]))
                   .limit(limit)
                   .all())

    # Attempt 3: OR on all words (any match)
    if not results:
        results = (BCItem.query
                   .filter(db.or_(*[word_condition(w) for w in words]))
                   .limit(limit)
                   .all())

    return results


def _bc_fetch_item_live(config: dict, item_no: str) -> dict:
    """Fetch a single item directly from BC live API (custom /items endpoint).
    Used to supplement cache data with price and lead time fields.
    Returns empty dict on failure."""
    if not _has_custom_api(config):
        return {}
    try:
        data = bc_request(config, f"/items?$filter=no eq '{item_no}'&$top=1", custom=True)
        items = data.get('value', [])
        if items:
            i = items[0]
            if 'no' in i and 'number' not in i:
                i['number'] = i['no']
            if 'description' in i and 'displayName' not in i:
                i['displayName'] = i['description']
            return i
    except Exception as e:
        logger.debug(f"Live item fetch failed for {item_no}: {e}")
    return {}


def bc_get_inventory(config: dict, item_no: str) -> dict:
    """Fetch inventory level and price for an item.

    Checks local cache first (fast). Supplements with live API for price.
    """
    i = _bc_fetch_item(config, item_no)
    if not i:
        return {'found': False, 'item_number': item_no}

    # If price is missing from cache, try a live fetch to supplement
    if i.get('unitPrice') is None:
        live = _bc_fetch_item_live(config, item_no)
        if live.get('unitPrice') is not None:
            i['unitPrice'] = live['unitPrice']

    result = {
        'found':       True,
        'item_number': i.get('number'),
        'item_name':   i.get('displayName'),
        'inventory':   i.get('inventory', 0),
        'unit':        i.get('baseUnitOfMeasureCode', 'stk'),
        'currency':    'DKK',
    }
    if i.get('unitPrice') is not None:
        result['unit_price'] = i['unitPrice']
    # Custom fields
    for f in ('minimumOrderQuantity', 'isDiscontinued', 'supplierItemNumber'):
        if i.get(f) is not None:
            result[f] = i[f]
    return result


def bc_get_item_details(config: dict, item_no: str) -> dict:
    """Fetch full item details including lead time and price.

    Checks local cache, supplements with live API for price and lead time.
    """
    i = _bc_fetch_item(config, item_no)
    if not i:
        return {'found': False, 'item_number': item_no}

    # Supplement from live API when price or lead time is missing
    if i.get('unitPrice') is None or i.get('leadTimeCalculation') is None:
        live = _bc_fetch_item_live(config, item_no)
        if live.get('unitPrice') is not None and i.get('unitPrice') is None:
            i['unitPrice'] = live['unitPrice']
        if i.get('leadTimeCalculation') is None:
            i['leadTimeCalculation'] = (live.get('salesLeadTimeEVM')
                                        or live.get('leadTimeCalculation')
                                        or live.get('supplierLeadTimeDays'))

    result = {
        'found':       True,
        'item_number': i.get('number'),
        'item_name':   i.get('displayName'),
        'unit_price':  i.get('unitPrice'),
        'currency':    'DKK',
        'inventory':   i.get('inventory', 0),
        'unit':        i.get('baseUnitOfMeasureCode', 'stk'),
        # salesLeadTimeEVM is Patrick's field (e.g. "5D"), fall back to standard
        'lead_time':   (i.get('salesLeadTimeEVM')
                        or i.get('supplierLeadTimeDays')
                        or i.get('leadTimeCalculation')),
        'blocked':     i.get('blocked', False),
    }
    # Custom fields
    for f in ('minimumOrderQuantity', 'isDiscontinued', 'supplierItemNumber',
              'webDescription', 'alternativeItemNumber'):
        if i.get(f) is not None:
            result[f] = i[f]
    return result


# Tool schemas passed to OpenAI function calling when BC is enabled
BC_TOOLS = [
    {
        "type": "function",
        "function": {
            "name": "verify_customer_identity",
            "description": (
                "MUST be called before get_order_status or get_order_shipment. "
                "Verifies the customer by checking their kundenummer (preferred) or email against the order in Business Central. "
                "ALWAYS ask for kundenummer first — it works for both open and posted/shipped orders. "
                "Only fall back to email if the customer explicitly cannot provide their kundenummer. "
                "If the user mentions their customer number in any form (e.g. 'jeg er 1788', 'kundenr 1788', 'kunde 2680'), use customer_number. "
                "Only call get_order_status or get_order_shipment after this returns verified=true."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "order_number":    {"type": "string", "description": "The sales order number, shipment number (F-XXXXXX), or customer's own PO reference (eksternt bilagsnr)"},
                    "email":           {"type": "string", "description": "The customer's email address — only use if customer cannot provide their kundenummer"},
                    "customer_number": {"type": "string", "description": "The customer's BC account number / kundenummer (e.g. '1788', '2680') — preferred verification method"},
                },
                "required": ["order_number"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "get_order_status",
            "description": (
                "Look up a sales order in Business Central. "
                "IMPORTANT: verify_customer_identity MUST be called and return verified=true before calling this. "
                "Returns order status and expected shipment/delivery date."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "order_number": {"type": "string", "description": "The sales order number, e.g. S-000123"}
                },
                "required": ["order_number"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "get_order_shipment",
            "description": (
                "Retrieve posted shipment and tracking information for a sales order. "
                "Returns tracking number(s), carrier name, and a direct tracking URL. "
                "IMPORTANT: verify_customer_identity MUST be called and return verified=true before calling this."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "order_number": {"type": "string", "description": "The sales order number"}
                },
                "required": ["order_number"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "get_item_inventory",
            "description": (
                "Check the current stock / inventory level and unit price for an item by item number. "
                "No identity verification required — this is public product data."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "item_number": {"type": "string", "description": "The item/product number, e.g. 1000"}
                },
                "required": ["item_number"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "get_item_details",
            "description": (
                "Get full details for an item: price, lead time, inventory and availability status. "
                "No identity verification required — this is public product data."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "item_number": {"type": "string", "description": "The item/product number"}
                },
                "required": ["item_number"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "search_items",
            "description": (
                "Search for items/products by description when the item number is unknown. "
                "Use this when a customer describes a product in natural language "
                "(e.g. 'Bøjning glasblæst ø 80 90 grader'). "
                "Returns matching items with item numbers and current stock levels. "
                "No identity verification required — this is public product data."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "description": {
                        "type": "string",
                        "description": "Natural language description of the item to search for",
                    }
                },
                "required": ["description"],
            },
        },
    },
]


_TOOL_LABELS: dict = {
    'da': {
        'verify_customer_identity': 'Verificerer identitet…',
        'get_order_status':         'Henter ordrestatus…',
        'get_order_shipment':       'Henter sporingsoplysninger…',
        'get_item_inventory':       'Henter lagerinfo…',
        'get_item_details':         'Henter vareoplysninger…',
        'search_items':             'Søger i varekatalog…',
        '_default':                 'Arbejder…',
    },
    'en': {
        'verify_customer_identity': 'Verifying identity…',
        'get_order_status':         'Fetching order status…',
        'get_order_shipment':       'Fetching shipment info…',
        'get_item_inventory':       'Checking stock…',
        'get_item_details':         'Fetching product details…',
        'search_items':             'Searching catalogue…',
        '_default':                 'Working…',
    },
    'de': {
        'verify_customer_identity': 'Identität wird geprüft…',
        'get_order_status':         'Bestellstatus wird abgerufen…',
        'get_order_shipment':       'Versanddaten werden abgerufen…',
        'get_item_inventory':       'Lagerbestand wird geprüft…',
        'get_item_details':         'Produktdetails werden abgerufen…',
        'search_items':             'Katalog wird durchsucht…',
        '_default':                 'Wird bearbeitet…',
    },
    'fr': {
        'verify_customer_identity': 'Vérification de l\'identité…',
        'get_order_status':         'Récupération du statut de commande…',
        'get_order_shipment':       'Récupération des infos d\'expédition…',
        'get_item_inventory':       'Vérification du stock…',
        'get_item_details':         'Récupération des détails produit…',
        'search_items':             'Recherche dans le catalogue…',
        '_default':                 'Traitement en cours…',
    },
    'pl': {
        'verify_customer_identity': 'Weryfikacja tożsamości…',
        'get_order_status':         'Pobieranie statusu zamówienia…',
        'get_order_shipment':       'Pobieranie informacji o wysyłce…',
        'get_item_inventory':       'Sprawdzanie stanu magazynowego…',
        'get_item_details':         'Pobieranie szczegółów produktu…',
        'search_items':             'Przeszukiwanie katalogu…',
        '_default':                 'Przetwarzanie…',
    },
}


def _detect_lang(text: str) -> str:
    """Lightweight language detection based on common function words."""
    t = text.lower()
    if any(w in t for w in ['quand', ' ma ', ' mon ', ' le ', ' la ', 'comment', 'bonjour', 'numéro', 'expédié', 'commande']):
        return 'fr'
    if any(w in t for w in ['wann', 'wird', 'meine', ' ich ', ' ist ', ' sind ', ' der ', ' die ', ' das ', 'bestellung']):
        return 'de'
    if any(w in t for w in ['kiedy', 'moje', ' jest ', 'zamówienie', 'numer', 'proszę', 'wysyłka']):
        return 'pl'
    if any(w in t for w in ['the ', ' is ', ' are ', ' my ', 'when', 'what', ' i ', 'order', 'dispatch', 'shipment']):
        return 'en'
    return 'da'

_UNVERIFIED_MSG = {
    'requires_verification': True,
    'message': (
        'Kald verify_customer_identity med ordrenummer og kundens kundenummer '
        'for at bekræfte identiteten, inden ordredata returneres.'
    ),
}

# Tool list used when the customer is already identified via JKF Universe login.
# verify_customer_identity is omitted — identity is pre-confirmed.
BC_TOOLS_LOGGED_IN = [
    {
        "type": "function",
        "function": {
            "name": "get_order_status",
            "description": (
                "Look up an open sales order in Business Central. "
                "The customer is already authenticated via JKF Universe — call this immediately, no verification needed. "
                "Accepts BC order number, shipment number (F-XXXXXX), or customer's own PO reference. "
                "If the result is found=false, the order is likely posted/shipped — immediately call "
                "get_order_shipment with the same order number to find tracking info."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "order_number": {"type": "string", "description": "Order number, shipment number (F-XXXXXX), or external PO reference"}
                },
                "required": ["order_number"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "get_order_shipment",
            "description": (
                "Retrieve posted/shipped order tracking information from Business Central. "
                "The customer is already authenticated via JKF Universe — call this immediately, no verification needed. "
                "Accepts BC order number, shipment number (F-XXXXXX), or customer's own PO reference. "
                "Always call this when the user asks about tracking, or when get_order_status returns found=false."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "order_number": {"type": "string", "description": "Order number, shipment number (F-XXXXXX), or external PO reference"}
                },
                "required": ["order_number"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "get_item_inventory",
            "description": (
                "Check the current stock / inventory level and unit price for an item by item number. "
                "No identity verification required — this is public product data."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "item_number": {"type": "string", "description": "The item/product number, e.g. 1000"}
                },
                "required": ["item_number"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "get_item_details",
            "description": (
                "Get full details for an item: price, lead time, inventory and availability status. "
                "No identity verification required — this is public product data."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "item_number": {"type": "string", "description": "The item/product number"}
                },
                "required": ["item_number"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "search_items",
            "description": (
                "Search for items/products by description when the item number is unknown. "
                "Use this when a customer describes a product in natural language "
                "(e.g. 'Bøjning glasblæst ø 80 90 grader'). "
                "Returns matching items with item numbers and current stock levels. "
                "No identity verification required — this is public product data."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "description": {
                        "type": "string",
                        "description": "Natural language description of the item to search for",
                    }
                },
                "required": ["description"],
            },
        },
    },
]

# Tool list for anonymous (non-logged-in) users.
# Same as the full BC_TOOLS list — includes verify_customer_identity so the
# email-verification flow works. Order data is gated behind that verification.
# BC_TOOLS_LOGGED_IN is the subset that skips verification (identity pre-confirmed
# via JKF Universe HMAC token).
BC_TOOLS_ANONYMOUS = BC_TOOLS


def _dispatch_bc_tool(config: dict, tool_name: str, arguments: dict,
                      thread_id: str = '',
                      customer_no: str = None) -> str:
    """Call the appropriate BC helper and return a JSON string result.

    customer_no: BC customer account number from JKF Universe login.
    When set, identity is pre-confirmed and queries are scoped to that customer.
    When absent, the anonymous email-verification flow is used instead.
    """
    try:
        if tool_name == 'verify_customer_identity':
            if customer_no:
                # Identity already confirmed via JKF Universe login
                result = {'verified': True, 'message': 'Identity verified via JKF Universe login.'}
            else:
                result = bc_verify_customer(
                    config,
                    arguments.get('order_number', ''),
                    arguments.get('email', ''),
                    thread_id,
                    customer_number=arguments.get('customer_number'),
                )
        elif tool_name == 'get_order_status':
            if customer_no:
                result = bc_get_order(config, arguments.get('order_number', ''),
                                      customer_no=customer_no)
            else:
                session = _bc_verified_sessions.get(thread_id, '')
                if not session:
                    result = _UNVERIFIED_MSG
                elif session.startswith('__cust__'):
                    # Verified via customer number — scope query to that customer
                    verified_cust = session[len('__cust__'):]
                    result = bc_get_order(config, arguments.get('order_number', ''),
                                          customer_no=verified_cust)
                else:
                    result = bc_get_order(config, arguments.get('order_number', ''),
                                          verified_email=session)
        elif tool_name == 'get_order_shipment':
            if customer_no:
                result = bc_get_shipment(config, arguments.get('order_number', ''),
                                         customer_no=customer_no)
            else:
                session = _bc_verified_sessions.get(thread_id, '')
                if not session:
                    result = _UNVERIFIED_MSG
                elif session.startswith('__cust__'):
                    # Verified via customer number — pass it for ownership check on shipment
                    verified_cust = session[len('__cust__'):]
                    result = bc_get_shipment(config, arguments.get('order_number', ''),
                                             customer_no=verified_cust)
                else:
                    result = bc_get_shipment(config, arguments.get('order_number', ''),
                                             verified_email=session)
        elif tool_name == 'get_item_inventory':
            result = bc_get_inventory(config, arguments.get('item_number', ''))
        elif tool_name == 'get_item_details':
            result = bc_get_item_details(config, arguments.get('item_number', ''))
        elif tool_name == 'search_items':
            matches = search_bc_items(arguments.get('description', ''))
            if matches:
                result = {
                    'found': True,
                    'matches': [
                        {
                            'item_number':  m.item_no,
                            'description':  m.description,
                            'description2': m.description2,
                            'inventory':    m.inventory,
                        }
                        for m in matches
                    ],
                }
            else:
                result = {
                    'found': False,
                    'message': 'Ingen varer fundet med den beskrivelse. Prøv med andre søgeord.',
                }
        else:
            result = {'error': f'Unknown tool: {tool_name}'}
    except Exception as e:
        logger.warning(f"BC tool {tool_name} failed: {e}")
        result = {'error': str(e)}
    return json.dumps(result, ensure_ascii=False)


# ─────────────────────────────────────────────────────────────────────────────
# Public chatbot API routes (no auth required)
# ─────────────────────────────────────────────────────────────────────────────
@app.route('/init_thread', methods=['POST'])
@csrf.exempt
def init_thread():
    return jsonify({"thread_id": str(uuid.uuid4())})


@app.route('/clear_thread', methods=['POST'])
@csrf.exempt
def clear_thread():
    data = request.get_json() or {}
    old_thread_id = data.get('thread_id')
    if old_thread_id:
        _bc_verified_sessions.pop(old_thread_id, None)
        _bc_thread_customer_no.pop(old_thread_id, None)
    return jsonify({"status": "success", "thread_id": str(uuid.uuid4())})


@app.route('/chat', methods=['POST'])
@csrf.exempt
def chat():
    """Non-streaming RAG chat endpoint."""
    data = request.get_json() or {}
    user_message = data.get('message', '').strip()
    thread_id = data.get('thread_id') or str(uuid.uuid4())
    training_mode = data.get('training_mode', False)
    company_token = data.get('company_id')

    if not user_message:
        return jsonify({"error": "Ingen besked"}), 400

    # Register customer context for this thread (HMAC-verified)
    customer_no = _verify_and_extract_customer_no(company_token)
    if customer_no:
        _bc_thread_customer_no[thread_id] = customer_no
    else:
        _bc_thread_customer_no.pop(thread_id, None)

    try:
        settings = get_jkf_settings()

        if training_mode:
            history = list(_training_history.get(thread_id, []))
        else:
            history = load_thread_history(thread_id)
        prior_user_turns = [m["content"] for m in history if m["role"] == "user"]
        context = get_context_from_qdrant(user_message, prior_user_turns=prior_user_turns)

        system_prompt = (settings.system_prompt or "").strip() or DEFAULT_SYSTEM_PROMPT
        link_rule = "\n\n## Link-formatering\nNår vidensdatabasen indeholder links som [tekst](url), gengiv dem PRÆCIS som [tekst](url)."
        system_content = (f"{system_prompt}{link_rule}\n\n[RELEVANT VIDEN]\n{context}"
                          if context else f"{system_prompt}{link_rule}")

        if customer_no:
            system_content += (
                f"\n\n[KUNDEKONTEKST] Brugeren er logget ind på JKF Universe som kunde {customer_no}. "
                "Alle ordreforespørgsler er automatisk begrænset til denne kundes data."
            )

        messages = [
            {"role": "system", "content": system_content},
            *history,
            {"role": "user", "content": get_datetime_context() + user_message},
        ]

        model_name = (settings.model or "gpt-5.4-mini").strip()
        ai_client = _get_ai_client(model_name)

        # ── Business Central function calling ─────────────────────────────────
        bc_integration = get_bc_integration()
        bc_config = get_bc_config(bc_integration) if bc_integration else None
        bc_available = bool(bc_config)
        # Logged-in customers skip email verification (identity pre-confirmed via HMAC).
        # Anonymous users get the full tool set including verify_customer_identity.
        active_bc_tools = (BC_TOOLS_LOGGED_IN if customer_no else BC_TOOLS_ANONYMOUS) if bc_available else []
        call_kwargs: dict = {"model": model_name, "messages": messages}
        if bc_available:
            call_kwargs["tools"] = active_bc_tools
            call_kwargs["tool_choice"] = "auto"

        completion = ai_client.chat.completions.create(**call_kwargs)
        choice = completion.choices[0]

        # Handle tool calls — up to 3 rounds to support multi-step flows
        # (e.g. verify_customer_identity → get_order_shipment for posted orders)
        did_tool_calls = False
        for _round in range(3):
            if not (bc_available and choice.finish_reason == "tool_calls" and choice.message.tool_calls):
                break
            did_tool_calls = True
            messages.append(choice.message)
            for tc in choice.message.tool_calls:
                fn_name = tc.function.name
                try:
                    fn_args = json.loads(tc.function.arguments or '{}')
                except Exception:
                    fn_args = {}
                tool_result = _dispatch_bc_tool(bc_config, fn_name, fn_args,
                                                thread_id=thread_id,
                                                customer_no=customer_no)
                messages.append({
                    "role": "tool",
                    "tool_call_id": tc.id,
                    "content": tool_result,
                })
            # Next call — pass tools so the model can make further calls if needed
            completion = ai_client.chat.completions.create(
                model=model_name, messages=messages,
                tools=active_bc_tools, tool_choice="auto",
            )
            choice = completion.choices[0]

        # After tool calls, remind the model not to leak raw JSON or internal reasoning.
        if did_tool_calls:
            messages.append({
                "role": "system",
                "content": (
                    f"Now write your final response to the user. "
                    f"The user's message was: \"{user_message}\"\n"
                    f"CRITICAL: Detect the language of that message and reply in THAT language. "
                    f"If the message is in English, reply in English. "
                    f"If German, reply in German. If French, reply in French. If Polish, reply in Polish. "
                    f"Only reply in Danish if the user wrote in Danish. Do NOT default to Danish.\n"
                    f"Rules:\n"
                    f"- Present order line information in a clear, readable format (item number, quantities, planned shipment dates).\n"
                    f"- Do NOT output raw JSON, field names (like 'item_no', 'outstanding_quantity'), or data structures.\n"
                    f"- Do NOT describe what tools you called or what you need to do next — just give the answer.\n"
                    f"- Do NOT ask the user to confirm anything you already have data for."
                ),
            })
            completion = ai_client.chat.completions.create(model=model_name, messages=messages)
            choice = completion.choices[0]

        raw_text = choice.message.content or ""

        assistant_clean = re.sub(r'\[METADATA\].*?\[/METADATA\]', '', raw_text, flags=re.DOTALL).strip()
        m = re.search(r'\[METADATA\]\s*(\{.*?\})\s*\[/METADATA\]', raw_text, re.DOTALL)
        metadata = normalize_metadata(json.loads(m.group(1))) if m else {}
        if needs_classification_fallback(metadata):
            metadata = fallback_classify(user_message, assistant_clean)

        formatted = format_message(assistant_clean)

        if training_mode:
            turns = _training_history.setdefault(thread_id, [])
            turns.append({"role": "user", "content": user_message})
            turns.append({"role": "assistant", "content": assistant_clean})
        else:
            _store_messages(thread_id, user_message, formatted, metadata)

        return jsonify({"response": formatted, "thread_id": thread_id, "metadata": metadata})
    except Exception as e:
        logger.error(f"Chat error: {e}")
        return jsonify({"error": str(e), "thread_id": thread_id}), 500


@app.route('/chat_stream', methods=['POST'])
@csrf.exempt
def chat_stream():
    """Streaming RAG chat endpoint (SSE).

    training_mode=true uses in-memory history and does NOT write to the database.
    Use this for testing the chatbot without polluting conversation analytics.
    """
    data = request.get_json() or {}
    user_message = data.get('message', '').strip()
    thread_id = data.get('thread_id')
    training_mode = data.get('training_mode', False)
    company_token = data.get('company_id')

    def generate():
        nonlocal thread_id
        try:
            settings = get_jkf_settings()
            if not thread_id:
                thread_id = str(uuid.uuid4())
            yield f"data: {json.dumps({'type': 'thread_id', 'content': thread_id})}\n\n"

            # Register customer context for this thread (HMAC-verified)
            customer_no = _verify_and_extract_customer_no(company_token)
            if customer_no:
                _bc_thread_customer_no[thread_id] = customer_no
            else:
                _bc_thread_customer_no.pop(thread_id, None)

            if training_mode:
                history = list(_training_history.get(thread_id, []))
            else:
                history = load_thread_history(thread_id)
            prior_user_turns = [m["content"] for m in history if m["role"] == "user"]
            context = get_context_from_qdrant(user_message, prior_user_turns=prior_user_turns)

            system_prompt = (settings.system_prompt or "").strip() or DEFAULT_SYSTEM_PROMPT
            link_rule = "\n\n## Link-formatering\nNår vidensdatabasen indeholder links som [tekst](url), gengiv dem PRÆCIS som [tekst](url)."
            system_content = (f"{system_prompt}{link_rule}\n\n[RELEVANT VIDEN]\n{context}"
                              if context else f"{system_prompt}{link_rule}")

            if customer_no:
                system_content += (
                    f"\n\n[KUNDEKONTEKST] Brugeren er logget ind på JKF Universe som kunde {customer_no}. "
                    "Alle ordreforespørgsler er automatisk begrænset til denne kundes data."
                )

            messages = [
                {"role": "system", "content": system_content},
                *history,
                {"role": "user", "content": get_datetime_context() + user_message},
            ]

            model_name = (settings.model or "gpt-5.4-mini").strip()
            ai_client = _get_ai_client(model_name)

            # ── Business Central function calling (non-streaming first pass) ──
            bc_integration = get_bc_integration()
            bc_config = get_bc_config(bc_integration) if bc_integration else None
            bc_available = bool(bc_config)
            # Logged-in customers get order/shipment/inventory tools; anonymous users
            # get only the public catalogue search tool.
            active_bc_tools = (BC_TOOLS_LOGGED_IN if customer_no else BC_TOOLS_ANONYMOUS) if bc_available else []

            if bc_available:
                # Non-streaming probe calls — up to 3 rounds for multi-step flows
                # (e.g. verify_customer_identity → get_order_shipment for posted orders)
                did_tool_calls = False
                for _round in range(3):
                    probe = ai_client.chat.completions.create(
                        model=model_name,
                        messages=messages,
                        tools=active_bc_tools,
                        tool_choice="auto",
                    )
                    probe_choice = probe.choices[0]
                    if probe_choice.finish_reason == "tool_calls" and probe_choice.message.tool_calls:
                        did_tool_calls = True
                        messages.append(probe_choice.message)
                        for tc in probe_choice.message.tool_calls:
                            fn_name = tc.function.name
                            lang = _detect_lang(user_message)
                            label = _TOOL_LABELS.get(lang, _TOOL_LABELS['da']).get(fn_name) \
                                    or _TOOL_LABELS.get(lang, _TOOL_LABELS['da'])['_default']
                            # Emit tool_call event so the frontend can show a status label
                            yield f"data: {json.dumps({'type': 'tool_call', 'label': label})}\n\n"
                            try:
                                fn_args = json.loads(tc.function.arguments or '{}')
                            except Exception:
                                fn_args = {}
                            tool_result = _dispatch_bc_tool(bc_config, fn_name, fn_args,
                                                            thread_id=thread_id,
                                                            customer_no=customer_no)
                            messages.append({
                                "role": "tool",
                                "tool_call_id": tc.id,
                                "content": tool_result,
                            })
                        # Continue to next round
                    else:
                        if not did_tool_calls:
                            # No tool calls at all — stream the text from the probe response
                            raw_probe = probe_choice.message.content or ""
                            buf = {'raw_text': '', 'metadata': {}, 'processed_text': ''}
                            full_assistant_text = ""
                            for char in raw_probe:
                                full_assistant_text += char
                                formatted, buf = process_stream_chunk(char, buf)
                                if formatted is not None:
                                    yield f"data: {json.dumps({'type': 'content', 'content': formatted})}\n\n"
                            metadata = normalize_metadata(buf.get('metadata', {}))
                            if needs_classification_fallback(metadata):
                                metadata = fallback_classify(user_message, full_assistant_text)
                            if training_mode:
                                turns = _training_history.setdefault(thread_id, [])
                                turns.append({"role": "user", "content": user_message})
                                turns.append({"role": "assistant", "content": full_assistant_text})
                            else:
                                _store_messages(thread_id, user_message, buf.get('processed_text', ''), metadata)
                            yield f"data: {json.dumps({'type': 'done', 'thread_id': thread_id, 'metadata': metadata})}\n\n"
                            return
                        # Tool calls done, model ready to give final answer — fall through to streaming
                        break

            buf = {'raw_text': '', 'metadata': {}, 'processed_text': ''}
            full_assistant_text = ""

            # After tool calls, remind the model not to leak raw JSON or internal reasoning.
            if did_tool_calls if bc_available else False:
                messages.append({
                    "role": "system",
                    "content": (
                        f"Now write your final response to the user. "
                        f"The user's message was: \"{user_message}\"\n"
                        f"CRITICAL: Detect the language of that message and reply in THAT language. "
                        f"If the message is in English, reply in English. "
                        f"If German, reply in German. If French, reply in French. If Polish, reply in Polish. "
                        f"Only reply in Danish if the user wrote in Danish. Do NOT default to Danish.\n"
                        f"Rules:\n"
                        f"- Present order line information in a clear, readable format (item number, quantities, planned shipment dates).\n"
                        f"- Do NOT output raw JSON, field names (like 'item_no', 'outstanding_quantity'), or data structures.\n"
                        f"- Do NOT describe what tools you called or what you need to do next — just give the answer.\n"
                        f"- Do NOT ask the user to confirm anything you already have data for."
                    ),
                })

            with ai_client.chat.completions.create(
                model=model_name, messages=messages, stream=True
            ) as stream:
                for chunk in stream:
                    delta = chunk.choices[0].delta.content if chunk.choices else None
                    if delta:
                        full_assistant_text += delta
                        formatted, buf = process_stream_chunk(delta, buf)
                        if formatted is not None:
                            yield f"data: {json.dumps({'type': 'content', 'content': formatted})}\n\n"

            metadata = normalize_metadata(buf.get('metadata', {}))
            if needs_classification_fallback(metadata):
                metadata = fallback_classify(user_message, full_assistant_text)

            if training_mode:
                turns = _training_history.setdefault(thread_id, [])
                turns.append({"role": "user", "content": user_message})
                turns.append({"role": "assistant", "content": full_assistant_text})
            else:
                _store_messages(thread_id, user_message, buf.get('processed_text', ''), metadata)

            yield f"data: {json.dumps({'type': 'done', 'thread_id': thread_id, 'metadata': metadata})}\n\n"

        except Exception as e:
            logger.error(f"Stream error: {e}")
            yield f"data: {json.dumps({'type': 'error', 'content': str(e)})}\n\n"

    return Response(stream_with_context(generate()), mimetype='text/event-stream',
                    headers={'Cache-Control': 'no-cache', 'X-Accel-Buffering': 'no'})


@app.route('/feedback', methods=['POST'])
@app.route('/submit_feedback', methods=['POST'])
@csrf.exempt
def feedback():
    data = request.get_json() or {}
    thread_id = data.get('thread_id')
    rating = data.get('rating')
    if thread_id and rating:
        try:
            last_assistant = (Message.query
                              .filter_by(thread_id=thread_id, role='assistant')
                              .order_by(desc(Message.timestamp))
                              .first())
            if last_assistant:
                last_assistant.feedback = int(rating)
                db.session.commit()
        except Exception as e:
            logger.error(f"Feedback error: {e}")
    return jsonify({"status": "success", "ok": True})


@app.route('/training/reset', methods=['POST'])
@csrf.exempt
def training_reset():
    """Clear in-memory training history for a thread (or all threads)."""
    data = request.get_json() or {}
    tid = data.get('thread_id')
    if tid and tid in _training_history:
        del _training_history[tid]
    elif not tid:
        _training_history.clear()
    # Also clear BC session state for this thread
    if tid:
        _bc_verified_sessions.pop(tid, None)
        _bc_thread_customer_no.pop(tid, None)
    else:
        _bc_verified_sessions.clear()
        _bc_thread_customer_no.clear()
    return jsonify({"ok": True})


@app.route('/track_click', methods=['POST'])
@csrf.exempt
def track_click():
    data = request.get_json() or {}
    thread_id = data.get('thread_id')
    url = data.get('url', '')
    if thread_id:
        try:
            last_assistant = (Message.query
                              .filter_by(thread_id=thread_id, role='assistant')
                              .order_by(desc(Message.timestamp))
                              .first())
            if last_assistant:
                last_assistant.clicked_link = url[:500]
                db.session.commit()
        except Exception as e:
            logger.error(f"Track click error: {e}")
    return jsonify({"status": "success"})


@app.route('/chatbot_config', methods=['GET'])
@csrf.exempt
def chatbot_config():
    """Return chatbot appearance config for the embed script."""
    s = get_jkf_settings()
    return jsonify({
        "chatbot_name": s.chatbot_name,
        "welcome_message": s.welcome_message,
        "disclaimer_text": s.disclaimer_text,
        "primary_color": s.primary_color,
        "toggle_color": s.toggle_color,
        "chatbot_position": s.chatbot_position,
        "logo_url": s.logo_url or "",

        "quick_questions": json.loads(s.quick_questions) if s.quick_questions else [],
        "speech_to_text_enabled": s.speech_to_text_enabled,
    })


# ─────────────────────────────────────────────────────────────────────────────
# Auth routes
# ─────────────────────────────────────────────────────────────────────────────
@app.route('/login', methods=['GET', 'POST'])
@limiter.limit("10 per minute")
def login():
    if 'user_id' in session:
        return redirect(url_for('dashboard'))
    form = LoginForm()
    if form.validate_on_submit():
        user = User.query.filter_by(username=form.username.data).first()
        if user and user.check_password(form.password.data):
            db.session.commit()  # persist any in-place password hash upgrade
            session['user_id'] = user.id
            flash(f'Velkommen, {user.username}!', 'success')
            next_url = request.args.get('next', '')
            # Reject absolute URLs to prevent open redirect
            parsed_next = urlparse(next_url)
            if parsed_next.netloc or parsed_next.scheme:
                next_url = ''
            if not user.is_admin:
                if user.budget_only:
                    return redirect(url_for('budget_agent'))
                if user.sales_only:
                    return redirect(url_for('sales_chatbot'))
                if user.agents_only:
                    if user.can_access_master_agent:
                        return redirect(url_for('master_agent'))
                    if user.can_access_sales_chatbot:
                        return redirect(url_for('sales_chatbot'))
                    if user.can_access_budget_agent:
                        return redirect(url_for('budget_agent'))
            return redirect(next_url or url_for('dashboard'))
        flash('Forkert brugernavn eller adgangskode.', 'danger')
    return render_template('login.html', form=form, hide_header=True)


@app.route('/logout')
def logout():
    session.clear()
    return redirect(url_for('login'))


# ─────────────────────────────────────────────────────────────────────────────
# Dashboard routes (auth required)
# ─────────────────────────────────────────────────────────────────────────────
@app.route('/')
@login_required
def index():
    return redirect(url_for('dashboard'))


@app.route('/dashboard')
@login_required
def dashboard():
    today = datetime.utcnow().date()
    week_start = today - timedelta(days=today.weekday())
    month_start = today.replace(day=1)

    conversations_today = _count_unique_threads(today, today)
    conversations_week = _count_unique_threads(week_start, today)
    conversations_month = _count_unique_threads(month_start, today)

    # Build global chronological conversation numbers
    all_threads_ordered = (db.session.query(Message.thread_id, func.min(Message.timestamp).label('first_ts'))
                           .filter(Message.role == 'user')
                           .group_by(Message.thread_id)
                           .order_by('first_ts').all())
    thread_numbers = {t.thread_id: i + 1 for i, t in enumerate(all_threads_ordered)}

    # Recent conversations (last 10 unique threads)
    recent_threads = (db.session.query(
        Message.thread_id,
        func.max(Message.timestamp).label('last_msg'),
    )
    .filter(Message.role == 'user')
    .group_by(Message.thread_id)
    .order_by(desc('last_msg'))
    .limit(5)
    .all())

    # Fetch first user message + message count per recent thread
    recent_thread_ids = [t.thread_id for t in recent_threads]
    first_messages = {}
    msg_counts = {}
    if recent_thread_ids:
        for tid in recent_thread_ids:
            first_msg = (Message.query
                         .filter_by(thread_id=tid, role='user')
                         .order_by(Message.timestamp)
                         .first())
            first_messages[tid] = first_msg.content if first_msg else ''
            msg_counts[tid] = (Message.query
                               .filter_by(thread_id=tid, role='user')
                               .count())

    recent_conversations = []
    for tid, ts in recent_threads:
        preview = first_messages.get(tid, '')
        recent_conversations.append({
            'thread_id': tid,
            'date': ts,
            'number': thread_numbers.get(tid, '?'),
            'preview': preview[:70] + '…' if len(preview) > 70 else preview,
            'msg_count': msg_counts.get(tid, 0),
        })

    return render_template('dashboard.html',
                           conversations_today=conversations_today,
                           conversations_week=conversations_week,
                           conversations_month=conversations_month,
                           recent_conversations=recent_conversations)


@app.route('/conversations')
@login_required
def all_conversations():
    page = request.args.get('page', 1, type=int)
    search = request.args.get('q', '').strip()

    query = (db.session.query(
        Message.thread_id,
        func.min(Message.timestamp).label('started_at'),
        func.max(Message.timestamp).label('last_msg'),
        func.sum(case((Message.role == 'user', 1), else_=0)).label('msg_count'),
        func.max(Message.category).label('category'),
        func.max(Message.language).label('language'),
        func.max(Message.feedback).label('feedback'),
        func.sum(Message.knowledge_gaps).label('knowledge_gaps'),
    )
    .group_by(Message.thread_id))

    if search:
        query = query.filter(Message.thread_id.in_(
            db.session.query(Message.thread_id)
            .filter(Message.content.contains(search))
            .filter(Message.role == 'user')
        ))

    total = query.count()
    per_page = 25
    threads = query.order_by(desc('last_msg')).offset((page - 1) * per_page).limit(per_page).all()

    conversations = []
    for i, t in enumerate(threads):
        conversations.append({
            'thread_id': t.thread_id,
            'started_at': t.started_at,
            'last_msg': t.last_msg,
            'msg_count': t.msg_count,
            'category': t.category,
            'language': LANGUAGE_MAP.get((t.language or '').lower(), t.language or ''),
            'feedback': t.feedback,
            'knowledge_gaps': t.knowledge_gaps,
            'number': (page - 1) * per_page + i + 1,
        })

    return render_template('all_conversations.html',
                           conversations=conversations,
                           page=page, per_page=per_page, total=total,
                           search=search)


@app.route('/conversation/<thread_id>')
@login_required
def conversation(thread_id):
    messages = (Message.query
                .filter_by(thread_id=thread_id)
                .order_by(Message.timestamp)
                .all())
    if not messages:
        flash('Samtale ikke fundet.', 'warning')
        return redirect(url_for('all_conversations'))

    # Build global chronological conversation numbers
    all_threads_ordered = (db.session.query(Message.thread_id, func.min(Message.timestamp).label('first_ts'))
                           .filter(Message.role == 'user')
                           .group_by(Message.thread_id)
                           .order_by('first_ts').all())
    thread_numbers = {t.thread_id: i + 1 for i, t in enumerate(all_threads_ordered)}

    # All threads for sidebar (ordered most recent first)
    sidebar_threads = (db.session.query(
        Message.thread_id,
        func.max(Message.timestamp).label('last_ts'),
        func.max(Message.feedback).label('feedback'),
    )
    .filter(Message.role == 'user')
    .group_by(Message.thread_id)
    .order_by(desc('last_ts')).all())

    # Group by Danish date for sidebar
    grouped_conversations = {}
    for t in sidebar_threads:
        ts = t.last_ts
        if ts is None:
            continue
        local_ts = ts.replace(tzinfo=pytz.utc).astimezone(danish_tz) if ts.tzinfo is None else ts.astimezone(danish_tz)
        date_str = local_ts.strftime('%d. %b %Y')
        if date_str not in grouped_conversations:
            grouped_conversations[date_str] = []
        grouped_conversations[date_str].append({
            'thread_id': t.thread_id,
            'conversation_number': thread_numbers.get(t.thread_id, '?'),
            'timestamp': local_ts.strftime('%H:%M'),
            'feedback': t.feedback,
        })

    conv_number = thread_numbers.get(thread_id, '?')
    feedback = next((m.feedback for m in reversed(messages) if m.role == 'assistant' and m.feedback), None)
    has_knowledge_gap = any(m.knowledge_gaps for m in messages if m.role == 'assistant')

    # Load comments
    comments = (Comment.query.filter_by(thread_id=thread_id)
                .order_by(Comment.timestamp).all())
    comment_data = [{'user': c.user.username if c.user else 'Ukendt',
                     'content': c.content,
                     'timestamp': c.timestamp} for c in comments]

    return render_template('conversation.html',
                           messages=messages,
                           thread_id=thread_id,
                           conversation_number=conv_number,
                           grouped_conversations=grouped_conversations,
                           current_thread_id=thread_id,
                           feedback=feedback,
                           has_knowledge_gap=has_knowledge_gap,
                           comments=comment_data)


def _compute_analytics_data(start_dt: datetime, end_dt: datetime, interval_label: str = '30') -> dict:
    """Compute all analytics data for the given UTC time window."""
    # ── Conversation count ────────────────────────────────────────────────────
    total_convs = (db.session.query(func.count(Message.thread_id.distinct()))
                   .filter(Message.role == 'user',
                           Message.timestamp.between(start_dt, end_dt))
                   .scalar() or 0)

    # ── Hourly + weekday/weekend breakdown (process in Python for SQLite compat)
    ts_rows = (db.session.query(Message.timestamp)
               .filter(Message.role == 'user',
                       Message.timestamp.between(start_dt, end_dt))
               .all())
    hourly = defaultdict(int)
    weekday_count = weekend_count = 0
    for (ts,) in ts_rows:
        if ts:
            local_ts = ts.replace(tzinfo=pytz.utc).astimezone(danish_tz) if ts.tzinfo is None else ts.astimezone(danish_tz)
            hourly[local_ts.hour] += 1
            if local_ts.weekday() < 5:
                weekday_count += 1
            else:
                weekend_count += 1
    hourly_counts = [{'hour': h, 'count': hourly.get(h, 0)} for h in range(24)]
    weekend_pct = round(weekend_count / total_convs * 100, 1) if total_convs else 0.0

    # ── Feedback ─────────────────────────────────────────────────────────────
    fb_rows = (db.session.query(Message.feedback, func.count(Message.id))
               .filter(Message.role == 'assistant', Message.feedback.isnot(None),
                       Message.timestamp.between(start_dt, end_dt))
               .group_by(Message.feedback).all())
    feedback_counts = {'1': 0, '2': 0, '3': 0, '4': 0, '5': 0}
    feedback_dict = {}
    for rating, cnt in fb_rows:
        feedback_counts[str(rating)] = cnt
        feedback_dict[rating] = cnt
    csat = _calculate_csat(feedback_dict)

    # ── Knowledge gaps ────────────────────────────────────────────────────────
    gaps = (db.session.query(func.count(Message.thread_id.distinct()))
            .filter(Message.role == 'assistant', Message.knowledge_gaps == 1,
                    Message.timestamp.between(start_dt, end_dt))
            .scalar() or 0)

    # ── FCR / CTR ─────────────────────────────────────────────────────────────
    fcr = round((1 - gaps / total_convs) * 100, 1) if total_convs else 0.0
    clicks_count = (db.session.query(func.count(Message.thread_id.distinct()))
                    .filter(Message.clicked_link.isnot(None),
                            Message.timestamp.between(start_dt, end_dt))
                    .scalar() or 0)
    ctr = round(clicks_count / total_convs * 100, 1) if total_convs else 0.0

    # ── Language distribution ─────────────────────────────────────────────────
    lang_rows = (db.session.query(Message.language, func.count(Message.thread_id.distinct()))
                 .filter(Message.role == 'assistant', Message.language.isnot(None),
                         Message.timestamp.between(start_dt, end_dt))
                 .group_by(Message.language).all())
    lang_merged: dict = {}
    for l, c in lang_rows:
        label = LANGUAGE_MAP.get((l or '').lower(), l or 'Ukendt')
        lang_merged[label] = lang_merged.get(label, 0) + c
    lang_dist = [{'language': label, 'count': c} for label, c in lang_merged.items()]

    # ── Top topics ────────────────────────────────────────────────────────────
    topic_rows = (db.session.query(Message.category, func.count(Message.thread_id.distinct()))
                  .filter(Message.role == 'assistant', Message.category.isnot(None),
                          Message.timestamp.between(start_dt, end_dt))
                  .group_by(Message.category)
                  .order_by(desc(func.count(Message.thread_id.distinct())))
                  .limit(10).all())
    top_topics = [{'category': c, 'count': n} for c, n in topic_rows]

    # ── Clicked links ─────────────────────────────────────────────────────────
    link_rows = (db.session.query(Message.clicked_link, func.count(Message.id))
                 .filter(Message.clicked_link.isnot(None),
                         Message.timestamp.between(start_dt, end_dt))
                 .group_by(Message.clicked_link)
                 .order_by(desc(func.count(Message.id)))
                 .limit(10).all())
    clicked_links = [{'clicked_link': l, 'count': c} for l, c in link_rows]

    # ── Knowledge gap conversations ───────────────────────────────────────────
    gap_threads = (db.session.query(Message.thread_id, func.max(Message.timestamp).label('last_ts'))
                   .filter(Message.role == 'assistant', Message.knowledge_gaps == 1,
                           Message.timestamp.between(start_dt, end_dt))
                   .group_by(Message.thread_id)
                   .order_by(desc('last_ts')).all())

    all_threads = (db.session.query(Message.thread_id, func.min(Message.timestamp).label('first_ts'))
                   .filter(Message.role == 'user')
                   .group_by(Message.thread_id)
                   .order_by('first_ts').all())
    thread_numbers = {t.thread_id: i + 1 for i, t in enumerate(all_threads)}

    def _fmt_ts(ts):
        if ts is None:
            return ''
        local = ts.replace(tzinfo=pytz.utc).astimezone(danish_tz) if ts.tzinfo is None else ts.astimezone(danish_tz)
        return local.strftime('%d-%m-%Y %H:%M')

    gap_convs = [{'thread_id': t.thread_id, 'timestamp': _fmt_ts(t.last_ts),
                  'conversation_number': thread_numbers.get(t.thread_id, '?')}
                 for t in gap_threads]

    # ── Previous period for trends ────────────────────────────────────────────
    period_days = max(int((end_dt - start_dt).total_seconds() / 86400), 1)
    prev_end = start_dt
    prev_start = start_dt - timedelta(days=period_days)

    prev_convs = (db.session.query(func.count(Message.thread_id.distinct()))
                  .filter(Message.role == 'user',
                          Message.timestamp.between(prev_start, prev_end))
                  .scalar() or 0)
    prev_gaps = (db.session.query(func.count(Message.thread_id.distinct()))
                 .filter(Message.role == 'assistant', Message.knowledge_gaps == 1,
                         Message.timestamp.between(prev_start, prev_end))
                 .scalar() or 0)
    prev_fb = (db.session.query(Message.feedback, func.count(Message.id))
               .filter(Message.role == 'assistant', Message.feedback.isnot(None),
                       Message.timestamp.between(prev_start, prev_end))
               .group_by(Message.feedback).all())
    prev_fb_dict = {r: c for r, c in prev_fb}
    prev_clicks = (db.session.query(func.count(Message.thread_id.distinct()))
                   .filter(Message.clicked_link.isnot(None),
                           Message.timestamp.between(prev_start, prev_end))
                   .scalar() or 0)

    prev_csat = _calculate_csat(prev_fb_dict)
    prev_fcr = round((1 - prev_gaps / prev_convs) * 100, 1) if prev_convs else 0.0
    prev_ctr = round(prev_clicks / prev_convs * 100, 1) if prev_convs else 0.0

    def _trend_abs(cur, prev):
        d = cur - prev
        return {'direction': 'up' if d > 0 else 'down' if d < 0 else 'flat', 'change': abs(d)}

    def _trend_pp(cur, prev):
        d = round(cur - prev, 1)
        return {'direction': 'up' if d > 0 else 'down' if d < 0 else 'flat', 'change': abs(d)}

    return {
        'total_conversations': total_convs,
        'hourly_counts': hourly_counts,
        'feedback_counts': feedback_counts,
        'language_distribution': lang_dist,
        'knowledge_gaps': gaps,
        'knowledge_gap_conversations': gap_convs,
        'top_topics': top_topics,
        'clicked_links': clicked_links,
        'csat_score': csat,
        'fcr_rate': fcr,
        'ctr': ctr,
        'weekend_conversations': weekend_count,
        'weekday_conversations': weekday_count,
        'weekend_percentage': weekend_pct,
        'conversations_trend': _trend_abs(total_convs, prev_convs),
        'csat_trend': _trend_pp(csat, prev_csat),
        'fcr_trend': _trend_pp(fcr, prev_fcr),
        'ctr_trend': _trend_pp(ctr, prev_ctr),
    }


def _parse_date_range(interval: str, start_date_str: str, end_date_str: str):
    end_dt = datetime.utcnow()
    if start_date_str and end_date_str:
        try:
            start_dt = datetime.strptime(start_date_str, '%Y-%m-%d')
            end_dt = datetime.strptime(end_date_str, '%Y-%m-%d').replace(hour=23, minute=59, second=59)
            return start_dt, end_dt
        except ValueError:
            pass
    if interval == 'all':
        return datetime(2000, 1, 1), end_dt
    days = int(interval) if str(interval).isdigit() else 30
    return end_dt - timedelta(days=days), end_dt


@app.route('/analytics')
@login_required
def analytics():
    interval = request.args.get('period', '30')
    start_dt, end_dt = _parse_date_range(interval, '', '')
    data = _compute_analytics_data(start_dt, end_dt, interval)
    return render_template('analytics.html', interval=interval, **data)


# ─────────────────────────────────────────────────────────────────────────────
# Knowledge Base routes
# ─────────────────────────────────────────────────────────────────────────────
@app.route('/knowledge-base')
@login_required
def knowledge_base():
    entries = KnowledgeBase.query.filter_by(is_active=True).order_by(KnowledgeBase.created_at.desc()).all()
    return render_template('knowledge_base.html', entries=entries)


@app.route('/knowledge-base/add', methods=['POST'])
@login_required
def kb_add():
    question = request.form.get('question', '').strip()
    answer = request.form.get('answer', '').strip()
    category = request.form.get('category', '').strip()
    source_url = request.form.get('source_url', '').strip() or None
    if not question or not answer:
        flash('Spørgsmål og svar er påkrævet.', 'danger')
        return redirect(url_for('knowledge_base'))
    entry = KnowledgeBase(
        question=question, answer=answer, category=category or None,
        source_url=source_url, created_by=session.get('user_id'), qdrant_synced=False,
    )
    db.session.add(entry)
    db.session.commit()
    flash('Videnspost tilføjet.', 'success')
    return redirect(url_for('knowledge_base'))


@app.route('/knowledge-base/<int:kb_id>/edit', methods=['POST'])
@login_required
def kb_edit(kb_id):
    entry = KnowledgeBase.query.get_or_404(kb_id)
    entry.question = request.form.get('question', entry.question).strip()
    entry.answer = request.form.get('answer', entry.answer).strip()
    entry.category = request.form.get('category', '').strip() or None
    entry.source_url = request.form.get('source_url', '').strip() or None
    entry.updated_at = datetime.utcnow()
    entry.qdrant_synced = False
    db.session.commit()
    flash('Videnspost opdateret.', 'success')
    return redirect(url_for('knowledge_base'))


@app.route('/knowledge-base/<int:kb_id>/delete', methods=['POST'])
@login_required
def kb_delete(kb_id):
    entry = KnowledgeBase.query.get_or_404(kb_id)
    entry.is_active = False
    db.session.commit()
    flash('Videnspost slettet.', 'success')
    return redirect(url_for('knowledge_base'))


@app.route('/knowledge-base/sync-qdrant', methods=['POST'])
@login_required
def kb_sync_qdrant():
    """Sync all unsynced KB entries to Qdrant."""
    qc = get_qdrant_client()
    if not qc:
        flash('Qdrant er ikke forbundet.', 'danger')
        return redirect(url_for('knowledge_base'))

    from qdrant_client.models import PointStruct, SparseVector as QSparseVector

    entries = KnowledgeBase.query.filter_by(is_active=True, qdrant_synced=False).all()
    if not entries:
        flash('Alle poster er allerede synkroniseret.', 'info')
        return redirect(url_for('knowledge_base'))

    synced = 0
    for entry in entries:
        try:
            text = f"Spørgsmål: {entry.question}\nSvar: {entry.answer}"
            emb = client.embeddings.create(model="text-embedding-3-large", input=text)
            dense = emb.data[0].embedding
            sp_idx, sp_val = _compute_sparse_vector(text)

            qc.upsert(
                collection_name=QDRANT_COLLECTION,
                points=[PointStruct(
                    id=str(uuid.uuid5(uuid.NAMESPACE_DNS, f"jkf-kb-{entry.id}")),
                    vector={"dense": dense, "sparse": QSparseVector(indices=sp_idx, values=sp_val)},
                    payload={
                        "text": text, "type": "qa",
                        "question": entry.question, "answer": entry.answer,
                        "source_type": "qa",
                        "source_url": entry.source_url or "",
                        "created_at": utc_iso(entry.created_at),
                    }
                )]
            )
            entry.qdrant_synced = True
            synced += 1
        except Exception as e:
            logger.error(f"Qdrant sync error for KB entry {entry.id}: {e}")

    db.session.commit()
    flash(f'{synced} poster synkroniseret til Qdrant.', 'success')
    return redirect(url_for('knowledge_base'))


# ─────────────────────────────────────────────────────────────────────────────
# Analytics JSON API
# ─────────────────────────────────────────────────────────────────────────────
@app.route('/api/analytics')
@login_required
def api_analytics():
    interval = request.args.get('interval', '30')
    start_date_str = request.args.get('start_date', '')
    end_date_str = request.args.get('end_date', '')
    start_dt, end_dt = _parse_date_range(interval, start_date_str, end_date_str)
    return jsonify(_compute_analytics_data(start_dt, end_dt, interval))


@app.route('/api/conversation/<thread_id>')
@login_required
def api_conversation(thread_id):
    msgs = (Message.query.filter_by(thread_id=thread_id)
            .order_by(Message.timestamp).all())
    result = []
    for m in msgs:
        local_ts = m.timestamp.replace(tzinfo=pytz.utc).astimezone(danish_tz) if m.timestamp.tzinfo is None else m.timestamp.astimezone(danish_tz)
        result.append({'role': m.role, 'content': m.content, 'timestamp': local_ts.strftime('%H:%M')})
    return jsonify({'messages': result})


@app.route('/api/knowledge_gap_conversations')
@login_required
def api_knowledge_gap_conversations():
    interval = request.args.get('interval', '30')
    start_dt, end_dt = _parse_date_range(interval, '', '')
    gap_threads = (db.session.query(Message.thread_id, func.max(Message.timestamp).label('last_ts'))
                   .filter(Message.role == 'assistant', Message.knowledge_gaps == 1,
                           Message.timestamp.between(start_dt, end_dt))
                   .group_by(Message.thread_id)
                   .order_by(desc('last_ts')).all())
    all_threads = (db.session.query(Message.thread_id, func.min(Message.timestamp).label('first_ts'))
                   .filter(Message.role == 'user')
                   .group_by(Message.thread_id).order_by('first_ts').all())
    numbers = {t.thread_id: i + 1 for i, t in enumerate(all_threads)}

    def _fmt(ts):
        if ts is None:
            return ''
        local = ts.replace(tzinfo=pytz.utc).astimezone(danish_tz) if ts.tzinfo is None else ts.astimezone(danish_tz)
        return local.strftime('%d-%m-%Y %H:%M')

    return jsonify({'conversations': [
        {'thread_id': t.thread_id, 'timestamp': _fmt(t.last_ts),
         'conversation_number': numbers.get(t.thread_id, '?')}
        for t in gap_threads
    ]})


@app.route('/api/feedback_conversations')
@login_required
def api_feedback_conversations():
    interval = request.args.get('interval', '30')
    start_dt, end_dt = _parse_date_range(interval, '', '')
    fb_threads = (db.session.query(
        Message.thread_id,
        func.max(Message.timestamp).label('last_ts'),
        func.max(Message.feedback).label('feedback'))
        .filter(Message.feedback.isnot(None),
                Message.timestamp.between(start_dt, end_dt))
        .group_by(Message.thread_id)
        .order_by(desc('last_ts')).all())
    all_threads = (db.session.query(Message.thread_id, func.min(Message.timestamp).label('first_ts'))
                   .filter(Message.role == 'user')
                   .group_by(Message.thread_id).order_by('first_ts').all())
    numbers = {t.thread_id: i + 1 for i, t in enumerate(all_threads)}

    def _fmt(ts):
        if ts is None:
            return ''
        local = ts.replace(tzinfo=pytz.utc).astimezone(danish_tz) if ts.tzinfo is None else ts.astimezone(danish_tz)
        return local.strftime('%d-%m-%Y %H:%M')

    return jsonify({'conversations': [
        {'thread_id': t.thread_id, 'conversation_number': numbers.get(t.thread_id, '?'),
         'last_message_time': _fmt(t.last_ts), 'feedback': t.feedback}
        for t in fb_threads
    ]})


@app.route('/api/analyze_knowledge_gaps', methods=['POST'])
@login_required
def api_analyze_knowledge_gaps():
    interval = request.args.get('interval', '30')
    start_dt, end_dt = _parse_date_range(interval, '', '')
    try:
        gap_threads = (db.session.query(Message.thread_id)
                       .filter(Message.role == 'assistant', Message.knowledge_gaps == 1,
                               Message.timestamp.between(start_dt, end_dt))
                       .distinct().all())
        if not gap_threads:
            return jsonify({'recommendations': [], 'conversations_analyzed': 0})

        thread_ids = [t.thread_id for t in gap_threads]
        total = len(thread_ids)
        MAX_CHARS = 350000
        excerpts = []
        total_chars = 0
        included = 0
        for tid in thread_ids:
            msgs = (Message.query.filter_by(thread_id=tid)
                    .order_by(Message.timestamp).all())
            lines = [('Bruger' if m.role == 'user' else 'Bot') + ': ' + m.content.strip() for m in msgs]
            excerpt = '\n'.join(lines)[:600]
            entry = f'--- Samtale {included + 1} ---\n{excerpt}'
            if total_chars + len(entry) > MAX_CHARS:
                break
            excerpts.append(entry)
            total_chars += len(entry)
            included += 1

        combined = '\n\n'.join(excerpts)
        truncation_note = f' (analyserede {included} ud af {total} pga. datamængde)' if included < total else ''

        interval_label = f'{interval} dage' if str(interval).isdigit() else 'alle'
        prompt = (
            f'Du er en AI-assistent der hjælper med at forbedre en chatbots vidensbase.\n\n'
            f'Nedenfor er {included} samtaler hvor chatbotten ikke kunne besvare brugerens spørgsmål '
            f'tilfredsstillende (videnshuller). Analyser samtalerne og giv præcis 5 korte, konkrete '
            f'anbefalinger til hvilken viden der bør tilføjes til chatbottens vidensbase.\n\n'
            f'Krav:\n- Fokuser på de mest tilbagevendende emner\n'
            f'- Vær specifik og handlingsorienteret\n- Skriv på dansk\n'
            f'- Returner KUN et JSON-objekt: {{"recommendations": ["...", ...]}}\n\n'
            f'Samtaler:\n{combined}'
        )
        resp = client.chat.completions.create(
            model='gpt-4o-mini',
            messages=[{'role': 'user', 'content': prompt}],
            response_format={'type': 'json_object'},
            timeout=60,
        )
        data = json.loads(resp.choices[0].message.content)
        recs = data.get('recommendations', [])

        analysis = KnowledgeGapAnalysis(
            interval=interval_label,
            conversations_analyzed=included,
            total_conversations=total,
            recommendations=json.dumps(recs, ensure_ascii=False),
        )
        db.session.add(analysis)
        db.session.commit()

        return jsonify({
            'recommendations': recs,
            'conversations_analyzed': included,
            'total_conversations': total,
            'truncation_note': truncation_note,
        })
    except Exception as e:
        logger.error(f'Gap analysis error: {e}')
        return jsonify({'error': 'Analysen mislykkedes. Prøv igen.'}), 500


@app.route('/api/knowledge_gap_analysis_history')
@login_required
def api_knowledge_gap_analysis_history():
    records = (KnowledgeGapAnalysis.query
               .order_by(desc(KnowledgeGapAnalysis.created_at))
               .limit(20).all())
    history = []
    for r in records:
        local = r.created_at.replace(tzinfo=pytz.utc).astimezone(danish_tz) if r.created_at.tzinfo is None else r.created_at.astimezone(danish_tz)
        history.append({
            'id': r.id,
            'created_at': local.strftime('%d-%m-%Y %H:%M'),
            'interval_label': r.interval or '—',
            'conversations_analyzed': r.conversations_analyzed,
            'total_conversations': r.total_conversations,
            'recommendations': json.loads(r.recommendations) if r.recommendations else [],
        })
    return jsonify({'history': history})


@app.route('/api/comments/<thread_id>')
@login_required
def api_get_comments(thread_id):
    comments = (Comment.query.filter_by(thread_id=thread_id)
                .order_by(Comment.timestamp).all())
    return jsonify({'comments': [
        {'user': c.user.username if c.user else 'Ukendt',
         'content': c.content,
         'timestamp': c.timestamp.strftime('%d-%m-%Y %H:%M')}
        for c in comments
    ]})


@app.route('/api/add_comment/<thread_id>', methods=['POST'])
@login_required
def api_add_comment(thread_id):
    data = request.get_json() or {}
    content = data.get('comment', '').strip()
    if not content:
        return jsonify({'success': False, 'message': 'Tom kommentar'}), 400
    comment = Comment(thread_id=thread_id, user_id=session.get('user_id'), content=content)
    db.session.add(comment)
    db.session.commit()
    return jsonify({'success': True})


# ─────────────────────────────────────────────────────────────────────────────
# Settings route
# ─────────────────────────────────────────────────────────────────────────────
@app.route('/settings', methods=['GET', 'POST'])
@login_required
def settings():
    s = get_jkf_settings()
    if request.method == 'POST':
        s.chatbot_name = request.form.get('chatbot_name', s.chatbot_name).strip()
        s.welcome_message = request.form.get('welcome_message', s.welcome_message).strip()
        s.disclaimer_text = request.form.get('disclaimer_text', s.disclaimer_text).strip()
        s.primary_color = request.form.get('primary_color', s.primary_color).strip()
        s.toggle_color = request.form.get('toggle_color', s.toggle_color).strip()
        s.logo_url = request.form.get('logo_url', '').strip() or None

        s.speech_to_text_enabled = 'speech_to_text_enabled' in request.form
        s.model = request.form.get('model', s.model).strip()
        s.system_prompt = request.form.get('system_prompt', '').strip() or None
        s.website_url = request.form.get('website_url', '').strip() or None

        # Quick questions – convert textarea (one per line) to JSON array
        raw_qq = request.form.get('quick_questions', '')
        questions = [q.strip() for q in raw_qq.split('\n') if q.strip()]
        s.quick_questions = json.dumps(questions, ensure_ascii=False) if questions else None

        db.session.commit()
        flash('Indstillinger gemt.', 'success')
        return redirect(url_for('settings'))
    return render_template('settings.html', s=s, DEFAULT_SYSTEM_PROMPT=DEFAULT_SYSTEM_PROMPT)


# ─────────────────────────────────────────────────────────────────────────────
# Integrations
# ─────────────────────────────────────────────────────────────────────────────
@app.route('/integrations')
@login_required
def integrations():
    bc = Integration.query.filter_by(integration_type='business_central').first()
    dw = Integration.query.filter_by(integration_type='datawarehouse_sql').first()
    return render_template('integrations.html', bc=bc, dw=dw)


@app.route('/integrations/business-central', methods=['GET', 'POST'])
@login_required
def integration_bc():
    bc = Integration.query.filter_by(integration_type='business_central').first()
    if not bc:
        bc = Integration(integration_type='business_central', name='Business Central')
        db.session.add(bc)
        db.session.commit()

    if request.method == 'POST':
        bc.enabled = 'enabled' in request.form
        existing_secret = json.loads(bc.config or '{}').get('client_secret', '')
        submitted_secret = request.form.get('client_secret', '').strip()
        # Keep existing secret when the form submits the masked placeholder
        resolved_secret = (submitted_secret
                           if submitted_secret and submitted_secret != '••••••••'
                           else existing_secret)
        config = {
            'tenant_id':     request.form.get('tenant_id', '').strip(),
            'client_id':     request.form.get('client_id', '').strip(),
            'client_secret': resolved_secret,
            'environment':   request.form.get('environment', 'production').strip() or 'production',
            'company_id':    request.form.get('company_id', '').strip(),
            # Custom API fields
            'custom_api_publisher': request.form.get('custom_api_publisher', '').strip(),
            'custom_api_group':     request.form.get('custom_api_group', '').strip(),
            'custom_api_version':   request.form.get('custom_api_version', '').strip(),
        }
        bc.config = json.dumps(config)
        bc.updated_at = datetime.utcnow()
        db.session.commit()
        # Invalidate cached token when credentials change
        tenant = config.get('tenant_id', '')
        _bc_token_cache.pop(tenant, None)
        flash('Business Central indstillinger gemt.', 'success')
        return redirect(url_for('integration_bc'))

    cfg = get_bc_config(bc)
    return render_template('integration_bc.html', bc=bc, cfg=cfg)


@app.route('/api/integrations/business-central/test', methods=['POST'])
@login_required
def api_bc_test():
    """Test Business Central credentials by fetching the company list."""
    data = request.get_json() or {}
    config = {
        'tenant_id':     data.get('tenant_id', '').strip(),
        'client_id':     data.get('client_id', '').strip(),
        'client_secret': data.get('client_secret', '').strip(),
        'environment':   data.get('environment', 'production').strip() or 'production',
        'company_id':    data.get('company_id', '').strip(),
    }
    # If secret is the placeholder, use the saved one
    if config['client_secret'] == '••••••••':
        bc = Integration.query.filter_by(integration_type='business_central').first()
        if bc:
            config['client_secret'] = get_bc_config(bc).get('client_secret', '')

    try:
        token = get_bc_token(config)
        tenant_id = config['tenant_id']
        env = config['environment']
        resp = requests.get(
            f"https://api.businesscentral.dynamics.com/v2.0/{tenant_id}/{env}/api/v2.0/companies",
            headers={'Authorization': f'Bearer {token}', 'Accept': 'application/json'},
            timeout=15,
        )
        resp.raise_for_status()
        companies = resp.json().get('value', [])
        names = [c.get('name', c.get('id', '')) for c in companies]
        return jsonify({
            'success': True,
            'message': f'Forbundet! Fandt {len(companies)} virksomhed(er): {", ".join(names)}',
            'companies': [{'id': c.get('id'), 'name': c.get('name')} for c in companies],
        })
    except Exception as e:
        return jsonify({'success': False, 'message': f'Forbindelsesfejl: {e}'}), 400


@app.route('/api/integrations/business-central/sync-items', methods=['POST'])
@login_required
def api_bc_sync_items():
    """Fetch all items from BC custom API and cache them locally for description search."""
    try:
        result = sync_bc_items()
        return jsonify(result)
    except Exception as e:
        logger.error(f"BC item sync failed: {e}")
        return jsonify({'success': False, 'message': str(e)}), 500


@app.route('/api/integrations/business-central/debug-custom-apis', methods=['GET'])
@login_required
def api_bc_debug_custom_apis():
    """Test all three custom API endpoints and report which ones are reachable."""
    bc = get_bc_integration()
    if not bc:
        return jsonify({'error': 'Business Central integration not configured'}), 400
    config = get_bc_config(bc)
    if not _has_custom_api(config):
        return jsonify({'error': 'Custom API not configured (missing publisher/group/version)'}), 400

    results = {}

    # 1. salesShipmentsEx
    try:
        data = bc_request(config, '/salesShipmentsEx?$top=1', custom=True)
        results['salesShipmentsEx'] = {'status': 'ok', 'sample_fields': list(data.get('value', [{}])[0].keys()) if data.get('value') else [], 'count': len(data.get('value', []))}
    except Exception as e:
        results['salesShipmentsEx'] = {'status': 'error', 'error': str(e)}

    # 2. Salesorderlines
    try:
        data = bc_request(config, '/Salesorderlines?$top=1', custom=True)
        results['Salesorderlines'] = {'status': 'ok', 'sample_fields': list(data.get('value', [{}])[0].keys()) if data.get('value') else [], 'count': len(data.get('value', []))}
    except Exception as e:
        results['Salesorderlines'] = {'status': 'error', 'error': str(e)}

    # 3. items (custom)
    try:
        data = bc_request(config, '/items?$top=1', custom=True)
        results['items'] = {'status': 'ok', 'sample_fields': list(data.get('value', [{}])[0].keys()) if data.get('value') else [], 'count': len(data.get('value', []))}
    except Exception as e:
        results['items'] = {'status': 'error', 'error': str(e)}

    return jsonify(results)


@app.route('/api/integrations/business-central/debug-shipment', methods=['GET'])
@login_required
def api_bc_debug_shipment():
    """Return raw BC API response for a shipment — used to inspect available field names."""
    order_ref = request.args.get('order', '').strip()
    if not order_ref:
        return jsonify({'error': 'Provide ?order=<ordrenr or forsendelsesnr>'}), 400
    bc = get_bc_integration()
    if not bc:
        return jsonify({'error': 'Business Central integration not configured'}), 400
    config = get_bc_config(bc)
    std_items = _bc_find_std_shipments(config, order_ref)

    custom_items = []
    custom_errors = []
    if _has_custom_api(config):
        refs_to_try = [order_ref]
        if std_items:
            for r in [std_items[0].get('orderNumber'), std_items[0].get('number')]:
                if r and r not in refs_to_try:
                    refs_to_try.append(r)
        for ref in refs_to_try:
            for flt in [f"orderNo eq '{ref}'", f"no eq '{ref}'"]:
                url = f"/salesShipmentsEx?$filter={flt}&$top=5"
                try:
                    data = bc_request(config, url, custom=True)
                    items = data.get('value', [])
                    custom_errors.append({'url': url, 'status': 'ok', 'count': len(items)})
                    if items:
                        custom_items = items
                except Exception as e:
                    custom_errors.append({'url': url, 'status': 'error', 'error': str(e)})

    return jsonify({
        'custom_api_configured':   _has_custom_api(config),
        'standard_salesShipments': std_items,
        'custom_salesShipmentsEx': custom_items,
        'custom_api_attempts':     custom_errors,
    })


@app.route('/api/integrations/business-central/item-cache-status', methods=['GET'])
@login_required
def api_bc_item_cache_status():
    """Return stats about the local item cache."""
    count = BCItem.query.count()
    last = BCItem.query.order_by(BCItem.synced_at.desc()).first()
    return jsonify({
        'cached_items': count,
        'last_synced': utc_iso(last.synced_at) if last else None,
    })


# ─────────────────────────────────────────────────────────────────────────────
# Admin / user management
# ─────────────────────────────────────────────────────────────────────────────
@app.route('/admin')
@admin_required
def admin():
    users = User.query.order_by(User.created_at).all()
    form = UserForm()
    return render_template('admin.html', users=users, form=form)


@app.route('/admin/users/add', methods=['POST'])
@admin_required
def admin_add_user():
    form = UserForm()
    if form.validate_on_submit():
        if User.query.filter_by(username=form.username.data).first():
            flash('Brugernavnet er allerede i brug.', 'danger')
        elif form.agents_only.data and not (
            form.can_access_master_agent.data or
            form.can_access_sales_chatbot.data or
            form.can_access_budget_agent.data
        ):
            flash('En "Kun agenter"-bruger skal have adgang til mindst én agent.', 'danger')
        else:
            sales_only  = form.sales_only.data
            budget_only = form.budget_only.data
            user = User(
                username=form.username.data,
                is_admin=form.is_admin.data,
                can_access_sales_chatbot=form.can_access_sales_chatbot.data or sales_only,
                sales_only=sales_only,
                can_access_budget_agent=form.can_access_budget_agent.data or budget_only,
                budget_only=budget_only,
                can_access_master_agent=form.can_access_master_agent.data,
                agents_only=form.agents_only.data,
            )
            user.set_password(form.password.data)
            db.session.add(user)
            db.session.commit()
            flash(f'Bruger {form.username.data} oprettet.', 'success')
    else:
        flash('Ugyldigt input.', 'danger')
    return redirect(url_for('admin'))


@app.route('/admin/users/<int:user_id>/delete', methods=['POST'])
@admin_required
def admin_delete_user(user_id):
    user = User.query.get_or_404(user_id)
    if user.id == session.get('user_id'):
        flash('Du kan ikke slette din egen bruger.', 'danger')
    else:
        db.session.delete(user)
        db.session.commit()
        flash(f'Bruger {user.username} slettet.', 'success')
    return redirect(url_for('admin'))


@app.route('/admin/users/<int:user_id>/reset-password', methods=['POST'])
@admin_required
def admin_reset_password(user_id):
    user = User.query.get_or_404(user_id)
    new_pw = request.form.get('new_password', '').strip()
    if not new_pw:
        flash('Adgangskode må ikke være tom.', 'danger')
    else:
        user.set_password(new_pw)
        db.session.commit()
        flash(f'Adgangskode for {user.username} nulstillet.', 'success')
    return redirect(url_for('admin'))


@app.route('/admin/users/<int:user_id>/toggle-sales-chatbot', methods=['POST'])
@admin_required
def admin_toggle_sales_chatbot(user_id):
    user = User.query.get_or_404(user_id)
    # Cycle: Ingen → Fuld adgang → Kun salgs-assistent → Ingen
    if not user.can_access_sales_chatbot and not user.sales_only:
        user.can_access_sales_chatbot = True
        user.sales_only = False
        label = 'Fuld adgang'
    elif user.can_access_sales_chatbot and not user.sales_only:
        user.can_access_sales_chatbot = True
        user.sales_only = True
        label = 'Kun salgs-assistent'
    else:
        user.can_access_sales_chatbot = False
        user.sales_only = False
        label = 'Ingen adgang'
    db.session.commit()
    flash(f'Salgs-adgang for {user.username} sat til: {label}.', 'success')
    return redirect(url_for('admin'))


@app.route('/admin/users/<int:user_id>/toggle-budget-agent', methods=['POST'])
@admin_required
def admin_toggle_budget_agent(user_id):
    user = User.query.get_or_404(user_id)
    # Cycle: Ingen → Fuld adgang → Kun økonomi → Ingen
    if not user.can_access_budget_agent and not user.budget_only:
        user.can_access_budget_agent = True
        user.budget_only = False
        label = 'Fuld adgang'
    elif user.can_access_budget_agent and not user.budget_only:
        user.can_access_budget_agent = True
        user.budget_only = True
        label = 'Kun økonomi'
    else:
        user.can_access_budget_agent = False
        user.budget_only = False
        label = 'Ingen adgang'
    db.session.commit()
    flash(f'Budget-adgang for {user.username} sat til: {label}.', 'success')
    return redirect(url_for('admin'))


@app.route('/admin/users/<int:user_id>/toggle-master-agent', methods=['POST'])
@admin_required
def admin_toggle_master_agent(user_id):
    user = User.query.get_or_404(user_id)
    user.can_access_master_agent = not user.can_access_master_agent
    label = 'Adgang' if user.can_access_master_agent else 'Ingen adgang'
    db.session.commit()
    flash(f'Master Agent-adgang for {user.username} sat til: {label}.', 'success')
    return redirect(url_for('admin'))


# ─────────────────────────────────────────────────────────────────────────────
# Standalone training chatbot page (embedded via iframe in knowledge base)
# ─────────────────────────────────────────────────────────────────────────────
@app.route('/training-chatbot')
def training_chatbot_page():
    """Standalone training chatbot – only accessible in iframe from dashboard."""
    return render_template('training_chatbot.html')


# ─────────────────────────────────────────────────────────────────────────────
# Knowledge Base JSON API (used by AJAX on knowledge_base.html)
# ─────────────────────────────────────────────────────────────────────────────
@app.route('/api/kb', methods=['GET'])
@login_required
def api_kb_list():
    entries = KnowledgeBase.query.filter_by(is_active=True).order_by(KnowledgeBase.updated_at.desc()).all()
    return jsonify({
        "success": True,
        "entries": [
            {
                "id": e.id,
                "question": e.question,
                "answer": e.answer,
                "category": e.category or "",
                "qdrant_synced": e.qdrant_synced,
                "updated_at": e.updated_at.strftime('%d-%m-%Y') if e.updated_at else "",
            }
            for e in entries
        ]
    })


@app.route('/api/kb', methods=['POST'])
@login_required
def api_kb_add():
    data = request.get_json() or {}
    question = data.get('question', '').strip()
    answer = data.get('answer', '').strip()
    category = data.get('category', '').strip()
    source_url = data.get('source_url', '').strip() or None
    if not question or not answer:
        return jsonify({"success": False, "message": "Spørgsmål og svar er påkrævet"}), 400
    entry = KnowledgeBase(
        question=question, answer=answer, category=category or None,
        source_url=source_url, created_by=session.get('user_id'), qdrant_synced=False,
    )
    db.session.add(entry)
    db.session.commit()
    return jsonify({"success": True, "id": entry.id})


@app.route('/api/kb/<int:kb_id>', methods=['PUT'])
@login_required
def api_kb_edit(kb_id):
    entry = KnowledgeBase.query.get_or_404(kb_id)
    data = request.get_json() or {}
    question = data.get('question', entry.question).strip()
    answer = data.get('answer', entry.answer).strip()
    category = data.get('category', '').strip()
    source_url = data.get('source_url', entry.source_url or '').strip() or None
    if not question or not answer:
        return jsonify({"success": False, "message": "Spørgsmål og svar er påkrævet"}), 400
    entry.question = question
    entry.answer = answer
    entry.category = category or None
    entry.source_url = source_url
    entry.updated_at = datetime.utcnow()
    entry.qdrant_synced = False
    db.session.commit()
    return jsonify({"success": True})


@app.route('/api/kb/<int:kb_id>', methods=['DELETE'])
@login_required
def api_kb_delete(kb_id):
    entry = KnowledgeBase.query.get_or_404(kb_id)
    # Delete from Qdrant if it was synced
    if entry.qdrant_synced:
        try:
            qc = get_qdrant_client()
            if qc:
                from qdrant_client.models import PointIdsList
                point_id = str(uuid.uuid5(uuid.NAMESPACE_DNS, f"jkf-kb-{entry.id}"))
                qc.delete(
                    collection_name=QDRANT_COLLECTION,
                    points_selector=PointIdsList(points=[point_id])
                )
        except Exception as e:
            logger.error(f"Qdrant delete error for KB entry {entry.id}: {e}")
    entry.is_active = False
    db.session.commit()
    return jsonify({"success": True})


@app.route('/api/kb/suggest', methods=['POST'])
@csrf.exempt
def api_kb_suggest():
    """Public endpoint for employee test page — saves a KB suggestion without requiring login."""
    data = request.get_json() or {}
    question = data.get('question', '').strip()
    answer = data.get('answer', '').strip()
    if not question or not answer:
        return jsonify({"success": False, "message": "Spørgsmål og svar er påkrævet"}), 400
    entry = KnowledgeBase(
        question=question, answer=answer, category=None,
        source_url=None, created_by=None, qdrant_synced=False,
    )
    db.session.add(entry)
    db.session.commit()
    return jsonify({"success": True, "id": entry.id})


@app.route('/api/kb/sync', methods=['POST'])
@login_required
def api_kb_sync():
    """Sync all unsynced KB entries to Qdrant (AJAX version)."""
    qc = get_qdrant_client()
    if not qc:
        return jsonify({"success": False, "message": "Qdrant er ikke forbundet"}), 503
    from qdrant_client.models import PointStruct, SparseVector as QSparseVector
    entries = KnowledgeBase.query.filter_by(is_active=True, qdrant_synced=False).all()
    if not entries:
        return jsonify({"success": True, "message": "Alt er allerede synkroniseret", "synced": 0})
    synced = 0
    for entry in entries:
        try:
            text = f"Spørgsmål: {entry.question}\nSvar: {entry.answer}"
            emb = client.embeddings.create(model="text-embedding-3-large", input=text)
            dense = emb.data[0].embedding
            sp_idx, sp_val = _compute_sparse_vector(text)
            qc.upsert(
                collection_name=QDRANT_COLLECTION,
                points=[PointStruct(
                    id=str(uuid.uuid5(uuid.NAMESPACE_DNS, f"jkf-kb-{entry.id}")),
                    vector={"dense": dense, "sparse": QSparseVector(indices=sp_idx, values=sp_val)},
                    payload={"text": text, "type": "qa", "question": entry.question,
                             "answer": entry.answer, "source_type": "qa",
                             "source_url": entry.source_url or "",
                             "created_at": utc_iso(entry.created_at)}
                )]
            )
            entry.qdrant_synced = True
            synced += 1
        except Exception as e:
            logger.error(f"Qdrant sync error for KB entry {entry.id}: {e}")
    db.session.commit()
    return jsonify({"success": True, "message": f"{synced} poster synkroniseret", "synced": synced})


# ─────────────────────────────────────────────────────────────────────────────
# Website scraper — background job + API routes
# ─────────────────────────────────────────────────────────────────────────────
WEBSITE_START_URL   = "https://jkfuniverse.com/da/"
WEBSITE_DOMAIN      = "jkfuniverse.com"
WEBSITE_PATH_PREFIX = "/da/"          # Only index Danish-language pages
WEBSITE_REQUEST_DELAY = 1.2           # Seconds between page fetches

_SCRAPER_HEADERS = {
    "User-Agent": "JKF-Chatbot-Crawler/1.0 (internal knowledge base indexer)"
}
_BOILERPLATE_CLASSES = re.compile(
    r'cookie|banner|popup|modal|breadcrumb|sidebar|social|share|'
    r'menu|navbar|nav-|footer|header|topbar|ribbon|announcement|'
    r'widget|advertisement|ads|overlay|consent',
    re.IGNORECASE,
)
_WEB_NOISE_TEXT_PATTERNS = (
    "read all about",
    "data sheet",
    "download datablade",
    "download datablad",
    "contact us",
    "kontakt os",
    "learn more",
    "show more",
    "read more",
)
_WEB_EXCLUDED_PATH_PATTERNS = (
    re.compile(r"/privacy-policy/?$"),
    re.compile(r"/password-change/"),
    re.compile(r"/myjkf/", re.I),
    re.compile(r"/pqt", re.I),
    re.compile(r"/find_", re.I),
)
_WEB_BOILERPLATE_LINE_PATTERNS = (
    "read all about",
    "data sheet",
    "download datablad",
    "downloadable files",
    "2d cad files",
    "3d cad files",
    "download selected",
    "select all",
    "files selected",
    "senest besøgte projekter",
    "see product",
)
_WEB_DISCOVERY_LIMIT = 600


def _web_strip_line_prefix(text: str) -> str:
    return re.sub(r"^(##+|[-|])\s*", "", text).strip()


def _web_is_excluded_url(url: str) -> bool:
    path = urlparse(url).path.lower()
    return any(pattern.search(path) for pattern in _WEB_EXCLUDED_PATH_PATTERNS)


def _web_extract_links(html: str, url: str) -> list[str]:
    try:
        from bs4 import BeautifulSoup
    except ImportError:
        return []

    soup = BeautifulSoup(html, "html.parser")
    links = []
    seen = set()
    for a in soup.find_all("a", href=True):
        href = (a.get("href") or "").strip()
        if not href or href.startswith("#"):
            continue
        abs_url = _web_normalise_url(urljoin(url, href))
        if not abs_url.startswith("http"):
            continue
        if not _web_is_allowed(abs_url) or not _web_is_crawlable(abs_url):
            continue
        if abs_url in seen:
            continue
        seen.add(abs_url)
        links.append(abs_url)
    return links


def _web_clean_extracted_text(text: str, title: str) -> str:
    cleaned_lines = []
    seen = set()
    title_key = re.sub(r"[\W_]+", "", re.sub(r"\s+", " ", (title or "")).strip().lower())
    for raw_line in text.splitlines():
        line = re.sub(r"\s+", " ", raw_line).strip()
        lowered = line.lower()
        bare = _web_strip_line_prefix(line)
        bare_lower = bare.lower()
        key = re.sub(r"[\W_]+", "", lowered)
        bare_key = re.sub(r"[\W_]+", "", bare_lower)
        if not key:
            continue
        if bare_lower in _WEB_BOILERPLATE_LINE_PATTERNS:
            continue
        if any(marker in bare_lower for marker in _WEB_BOILERPLATE_LINE_PATTERNS) and len(bare) < 80:
            continue
        if bare_key == title_key and cleaned_lines:
            continue
        if key in seen or bare_key in seen:
            continue
        seen.add(key)
        seen.add(bare_key)
        cleaned_lines.append(line)
    return "\n".join(cleaned_lines).strip()


def _web_has_semantic_signal(text: str) -> bool:
    meaningful_lines = []
    spec_like_lines = 0
    for line in text.splitlines():
        bare = _web_strip_line_prefix(line)
        lowered = bare.lower()
        if not bare or lowered in _WEB_BOILERPLATE_LINE_PATTERNS:
            continue
        if bare.startswith("http"):
            continue
        word_count = len(bare.split())
        if bare.startswith("| ") or ":" in bare:
            spec_like_lines += 1
        if word_count >= 8 and len(bare) >= 60 and re.search(r"[.:;)]", bare):
            meaningful_lines.append(bare)
    return len(meaningful_lines) >= 1 or spec_like_lines >= 3


def _web_is_quality_page(text: str, title: str, url: str) -> bool:
    if _web_is_excluded_url(url):
        return False
    if not text:
        return False
    lowered = text.lower()
    generic_title = re.sub(r"[\W_]+", "", (title or "").lower()) in {"jkfuniverse", "jkf"}
    if generic_title and len(text) < 600:
        return False
    boilerplate_hits = sum(1 for marker in _WEB_BOILERPLATE_LINE_PATTERNS if marker in lowered)
    if len(text) < 180 and not _web_has_semantic_signal(text):
        return False
    if boilerplate_hits >= 2 and not _web_has_semantic_signal(text):
        return False
    return True


def _web_extract_fallback_container_text(soup, title: str) -> str:
    """Fallback for pages without a clean main/article container."""
    title_hint = re.sub(r"\s+", " ", (title or "")).strip().lower()
    candidates = []
    for node in soup.find_all(["section", "div"]):
        text = re.sub(r"\s+", " ", node.get_text(" ", strip=True)).strip()
        if len(text) < 200:
            continue
        if title_hint and title_hint not in text.lower():
            continue
        lowered = text.lower()
        score = sum(
            1 for marker in (
                "specifikationer",
                "downloadable files",
                "downloads",
                "download datablad",
                "2d cad files",
                ".dwg",
                "dimensioner",
            )
            if marker in lowered
        )
        candidates.append((score, len(text), node))

    if not candidates:
        return ""

    _, _, best_node = max(candidates, key=lambda item: (item[0], item[1]))
    lines = []
    seen = set()
    for part in best_node.get_text("\n", strip=True).splitlines():
        text = re.sub(r"\s+", " ", part).strip()
        lowered = text.lower()
        key = re.sub(r"[\W_]+", "", lowered)
        if not key or key in seen:
            continue
        if lowered in {"read all about", "data sheet", "see product"}:
            continue
        seen.add(key)
        lines.append(text)
    return "\n".join(lines).strip()

# In-memory job state (reset on server restart — that's fine for a long-running task)
_scrape_job: dict = {
    "status":      "idle",   # idle | running | done | error
    "pages_found": 0,
    "pages_done":  0,
    "chunks_total": 0,
    "current_url": None,
    "phase":       "idle",
    "discovered_count": 0,
    "started_at":  None,
    "finished_at": None,
    "error":       None,
}
_scrape_lock = threading.Lock()


def _web_is_allowed(url: str) -> bool:
    p = urlparse(url)
    netloc = p.netloc.lower()
    on_domain = netloc == WEBSITE_DOMAIN or netloc == f"www.{WEBSITE_DOMAIN}"
    return on_domain and p.path.startswith(WEBSITE_PATH_PREFIX) and not _web_is_excluded_url(url)


def _web_is_crawlable(url: str) -> bool:
    skip_exts = ('.pdf', '.jpg', '.jpeg', '.png', '.gif', '.svg', '.webp',
                 '.mp4', '.mp3', '.zip', '.docx', '.xlsx', '.pptx')
    path = urlparse(url).path.lower()
    return not any(path.endswith(e) for e in skip_exts) and not url.startswith(('mailto:', 'tel:'))


def _web_is_manual_include_url(url: str) -> bool:
    p = urlparse(url)
    return p.scheme in {"http", "https"} and bool(p.netloc) and _web_is_crawlable(url)


def _web_normalise_url(url: str) -> str:
    url, _ = urldefrag(url)
    p = urlparse(url)
    path = p.path or "/"
    if path != "/":
        path = path.rstrip("/")
    netloc = p.netloc.lower()
    if (p.scheme.lower() == "https" and netloc.endswith(":443")) or (p.scheme.lower() == "http" and netloc.endswith(":80")):
        netloc = netloc.rsplit(":", 1)[0]
    return p._replace(scheme=p.scheme.lower(), netloc=netloc, path=path).geturl()


def _web_get_sitemap_urls() -> list:
    ns = {"sm": "http://www.sitemaps.org/schemas/sitemap/0.9"}
    all_urls = []

    def _fetch(sitemap_url):
        try:
            r = requests.get(sitemap_url, headers=_SCRAPER_HEADERS, timeout=15)
            if r.status_code != 200:
                return []
            root = ET.fromstring(r.content)
            # sitemap index
            for child in root.findall(".//sm:sitemap/sm:loc", ns):
                all_urls.extend(_fetch(child.text.strip()))
            # regular sitemap
            return [loc.text.strip() for loc in root.findall(".//sm:url/sm:loc", ns) if loc.text]
        except Exception as e:
            logger.warning(f"Sitemap fetch {sitemap_url} failed: {e}")
            return []

    for candidate in [
        urljoin(WEBSITE_START_URL, "/sitemap.xml"),
        urljoin(WEBSITE_START_URL, "/sitemap_index.xml"),
    ]:
        found = _fetch(candidate)
        if found:
            all_urls.extend(found)
            break

    include_urls, exclude_urls = _get_website_rules()

    urls = {
        _web_normalise_url(u)
        for u in all_urls
        if _web_is_allowed(u) and _web_is_crawlable(u)
    }
    urls.update(u for u in include_urls if _web_is_manual_include_url(u))
    urls.difference_update(exclude_urls)

    return sorted(urls)


def _web_discover_urls() -> list:
    sitemap_urls = set(_web_get_sitemap_urls())
    discovered = set(sitemap_urls)
    queue = [_web_normalise_url(WEBSITE_START_URL)]
    visited = set()
    include_urls, exclude_urls = _get_website_rules()

    for url in include_urls:
        if _web_is_manual_include_url(url):
            discovered.add(url)
    discovered.difference_update(exclude_urls)

    with _scrape_lock:
        _scrape_job["phase"] = "discovering"
        _scrape_job["discovered_count"] = len(discovered)
        _scrape_job["current_url"] = WEBSITE_START_URL

    while queue and len(discovered) < _WEB_DISCOVERY_LIMIT:
        url = queue.pop(0)
        if url in visited:
            continue
        visited.add(url)
        with _scrape_lock:
            _scrape_job["current_url"] = url
        try:
            resp = requests.get(url, headers=_SCRAPER_HEADERS, timeout=15)
            if resp.status_code != 200 or "html" not in resp.headers.get("content-type", ""):
                continue
            for link in _web_extract_links(resp.text, url):
                if link not in discovered:
                    discovered.add(link)
                    with _scrape_lock:
                        _scrape_job["discovered_count"] = len(discovered)
                if link not in visited and link not in queue and len(discovered) < _WEB_DISCOVERY_LIMIT:
                    queue.append(link)
        except Exception as e:
            logger.debug(f"Recursive discovery failed for {url}: {e}")
        time.sleep(min(WEBSITE_REQUEST_DELAY, 0.4))

    logger.info(f"Website discovery: {len(sitemap_urls)} sitemap URLs, {len(discovered)} total after crawling")
    return sorted(discovered)


def _web_extract_content(html: str, url: str) -> tuple:
    """Return (title, clean_text). Strips nav/header/footer/boilerplate."""
    try:
        from bs4 import BeautifulSoup
    except ImportError:
        logger.error("beautifulsoup4 not installed — run: pip install beautifulsoup4")
        return "", ""

    soup = BeautifulSoup(html, "html.parser")

    # Collect tags to remove FIRST, then decompose — modifying the tree during
    # find_all iteration orphans child tags still in the list, corrupting their state.
    noise = list(soup(["script", "style", "noscript", "iframe",
                       "nav", "header", "footer", "aside", "form", "button", "svg"]))
    for tag in soup.find_all(True):
        try:
            attrs = tag.attrs or {}
            cls = " ".join(attrs.get("class", []) or [])
            tid = attrs.get("id", "") or ""
            if _BOILERPLATE_CLASSES.search(cls) or _BOILERPLATE_CLASSES.search(tid):
                noise.append(tag)
        except Exception:
            pass
    for tag in noise:
        try:
            tag.decompose()
        except Exception:
            pass

    # Title
    title = ""
    if soup.title and soup.title.string:
        title = soup.title.string.strip().split("|")[0].strip()
    if not title:
        h1 = soup.find("h1")
        title = h1.get_text(" ", strip=True) if h1 else url

    content_root = (soup.find("main") or soup.find("article")
                    or soup.find(attrs={"role": "main"})
                    or soup.find("div", {"id": re.compile(r"content|main|product", re.I)})
                    or soup.find("div", {"class": re.compile(r"content|main|product|detail", re.I)})
                    or soup.body)
    if not content_root:
        return title, ""

    lines = []
    seen: set = set()
    for el in content_root.find_all(("h1", "h2", "h3", "h4", "h5", "h6", "p", "li", "td", "th")):
        try:
            attrs = el.attrs or {}
            cls = " ".join(attrs.get("class", []) or [])
            tid = attrs.get("id", "") or ""
            if attrs.get("hidden") is not None or _BOILERPLATE_CLASSES.search(cls) or _BOILERPLATE_CLASSES.search(tid):
                continue
        except Exception:
            pass

        t = re.sub(r"\s+", " ", el.get_text(" ", strip=True)).strip()
        if not t or len(t) < 3:
            continue
        lowered = t.lower()
        if lowered in {"read all about", "data sheet"}:
            continue
        if any(pattern in lowered for pattern in _WEB_NOISE_TEXT_PATTERNS) and len(t) < 80:
            continue

        key = re.sub(r"[\W_]+", "", lowered)
        if not key or key in seen:
            continue
        seen.add(key)

        if el.name in ("h1", "h2", "h3"):
            lines.append(f"## {t}")
        elif el.name in ("h4", "h5", "h6"):
            lines.append(f"### {t}")
        elif el.name == "li":
            lines.append(f"- {t}")
        elif el.name in ("td", "th"):
            lines.append(f"| {t}")
        elif len(t) >= 30:
            lines.append(t)

    clean = re.sub(r'\n{3,}', '\n\n', "\n\n".join(lines)).strip()
    if len(clean) < 120:
        clean = _web_extract_fallback_container_text(soup, title) or clean
    clean = _web_clean_extracted_text(clean, title)
    return title, clean


def _run_website_scrape(clear_first: bool = False):
    """Background thread: crawl jkfuniverse.com/da/ and upsert all pages to Qdrant."""
    global _scrape_job
    with _scrape_lock:
        _scrape_job.update({
            "status": "running", "pages_found": 0, "pages_done": 0,
            "chunks_total": 0, "current_url": None, "phase": "discovering",
            "discovered_count": 0, "error": None,
            "started_at": datetime.now(danish_tz).isoformat(), "finished_at": None,
        })

    try:
        with app.app_context():
            qc = get_qdrant_client()
            if not qc:
                raise RuntimeError("Qdrant er ikke forbundet")

            WebsiteIndexedPage.query.delete()
            db.session.commit()

            # Always clear existing website chunks so re-runs never leave stale content
            # from removed or renamed pages. (clear_first flag is now a no-op but kept
            # for backwards compatibility with the API.)
            from qdrant_client.models import Filter as QFilter, FieldCondition as QFC, MatchValue as QMV
            logger.info("Website scraper: clearing existing website chunks")
            qc.delete(
                collection_name=QDRANT_COLLECTION,
                points_selector=QFilter(must=[QFC(key="source_type", match=QMV(value="website"))]),
            )

            # Discover pages
            logger.info("Website scraper: discovering website URLs")
            urls = _web_discover_urls()
            if not urls:
                urls = [_web_normalise_url(WEBSITE_START_URL)]
            logger.info(f"Website scraper: {len(urls)} pages to crawl")
            with _scrape_lock:
                _scrape_job["phase"] = "indexing"
                _scrape_job["pages_found"] = len(urls)

            from qdrant_client.models import PointStruct, SparseVector as QSV

            for url in urls:
                with _scrape_lock:
                    _scrape_job["current_url"] = url

                try:
                    resp = requests.get(url, headers=_SCRAPER_HEADERS, timeout=15)
                    if resp.status_code != 200 or "html" not in resp.headers.get("content-type", ""):
                        continue

                    title, text = _web_extract_content(resp.text, url)
                    if not text or len(text) < 80 or not _web_is_quality_page(text, title, url):
                        continue

                    chunks = _chunk_text(text)
                    for i, chunk in enumerate(chunks):
                        emb  = client.embeddings.create(model="text-embedding-3-large", input=chunk)
                        dense = emb.data[0].embedding
                        sp_idx, sp_val = _compute_sparse_vector(chunk)
                        point_id = str(uuid.uuid5(uuid.NAMESPACE_URL, f"jkf-web-{url}-chunk-{i}"))
                        qc.upsert(
                            collection_name=QDRANT_COLLECTION,
                            points=[PointStruct(
                                id=point_id,
                                vector={"dense": dense, "sparse": QSV(indices=sp_idx, values=sp_val)},
                                payload={
                                    "text":          chunk,
                                    "source_type":   "website",
                                    "source_url":    url,
                                    "section_title": title,
                                    "chunk_index":   i,
                                },
                            )],
                        )
                        with _scrape_lock:
                            _scrape_job["chunks_total"] += 1

                    _record_indexed_page(url, title, len(chunks))
                    db.session.commit()

                except Exception as e:
                    logger.error(f"Website scraper error on {url}: {e}")

                with _scrape_lock:
                    _scrape_job["pages_done"] += 1

                time.sleep(WEBSITE_REQUEST_DELAY)

            with _scrape_lock:
                _scrape_job.update({"status": "done",
                                    "phase": "done",
                                    "finished_at": datetime.now(danish_tz).isoformat(),
                                    "current_url": None})
            settings = JKFSettings.query.first()
            if settings:
                settings.last_scraped_date = datetime.utcnow()
                db.session.commit()
            logger.info(f"Website scraper done: {_scrape_job['pages_done']} pages, "
                        f"{_scrape_job['chunks_total']} chunks")

    except Exception as e:
        logger.error(f"Website scraper fatal error: {e}")
        with _scrape_lock:
            _scrape_job.update({"status": "error", "error": str(e),
                                "phase": "error",
                                "finished_at": datetime.now(danish_tz).isoformat(),
                                "current_url": None})


@app.route('/api/website/status', methods=['GET'])
@login_required
def api_website_status():
    """Return current scrape job state + Qdrant chunk count for website source."""
    with _scrape_lock:
        job = dict(_scrape_job)

    # Count website chunks currently in Qdrant
    chunk_count = 0
    try:
        qc = get_qdrant_client()
        if qc:
            from qdrant_client.models import Filter as QFilter, FieldCondition as QFC, MatchValue as QMV
            result = qc.count(
                collection_name=QDRANT_COLLECTION,
                count_filter=QFilter(must=[QFC(key="source_type", match=QMV(value="website"))]),
                exact=False,
            )
            chunk_count = result.count
    except Exception as e:
        logger.warning(f"api_website_status: qdrant count failed: {e}")

    indexed_page_count = WebsiteIndexedPage.query.count()
    stale_trailing_count = (WebsiteIndexedPage.query
                            .filter(WebsiteIndexedPage.url.like('%/'))
                            .count())
    with _scrape_lock:
        is_running = _scrape_job["status"] == "running"
    if not is_running and chunk_count and (not indexed_page_count or stale_trailing_count):
        indexed_page_count, _ = _sync_website_indexed_pages_from_qdrant()

    job["qdrant_chunk_count"] = chunk_count

    # Include the persisted last_scraped_date so the frontend can show the correct
    # "Sidst kørt" time even after a server restart (when _scrape_job is reset to idle).
    last_scraped_iso = None
    try:
        settings = JKFSettings.query.first()
        if settings and settings.last_scraped_date:
            last_scraped_iso = utc_iso(settings.last_scraped_date)
    except Exception:
        pass

    return jsonify({
        "success": True,
        "job": job,
        "qdrant_chunk_count": chunk_count,
        "indexed_page_count": indexed_page_count,
        "last_scraped_date": last_scraped_iso,
        "include_rule_count": WebsiteScrapeRule.query.filter_by(rule_type='include').count(),
        "exclude_rule_count": WebsiteScrapeRule.query.filter_by(rule_type='exclude').count(),
    })


@app.route('/api/website/scrape', methods=['POST'])
@login_required
def api_website_scrape():
    """Start a website scrape job in the background."""
    with _scrape_lock:
        if _scrape_job["status"] == "running":
            return jsonify({"success": False, "message": "Indeksering er allerede i gang"}), 409

    data = request.get_json(silent=True) or {}
    clear_first = bool(data.get("clear", False))

    t = threading.Thread(target=_run_website_scrape, args=(clear_first,), daemon=True)
    t.start()
    return jsonify({"success": True, "message": "Indeksering startet"})


@app.route('/api/website/clear', methods=['DELETE'])
@login_required
def api_website_clear():
    """Delete all website chunks from Qdrant without re-indexing."""
    with _scrape_lock:
        if _scrape_job["status"] == "running":
            return jsonify({"success": False, "message": "Kan ikke rydde mens indeksering kører"}), 409
    try:
        qc = get_qdrant_client()
        if not qc:
            return jsonify({"success": False, "message": "Qdrant er ikke forbundet"}), 503
        from qdrant_client.models import Filter as QFilter, FieldCondition as QFC, MatchValue as QMV
        qc.delete(
            collection_name=QDRANT_COLLECTION,
            points_selector=QFilter(must=[QFC(key="source_type", match=QMV(value="website"))]),
        )
        WebsiteIndexedPage.query.delete()
        db.session.commit()
        with _scrape_lock:
            _scrape_job.update({"status": "idle", "pages_done": 0,
                                "pages_found": 0, "chunks_total": 0,
                                "current_url": None, "finished_at": None, "error": None})
        return jsonify({"success": True, "message": "Hjemmeside-data er ryddet fra vidensbasen"})
    except Exception as e:
        return jsonify({"success": False, "message": str(e)}), 500


@app.route('/api/website/pages', methods=['GET'])
@login_required
def api_website_pages():
    with _scrape_lock:
        is_running = _scrape_job["status"] == "running"
    if not is_running:
        _sync_website_indexed_pages_from_qdrant()

    indexed_pages = WebsiteIndexedPage.query.order_by(WebsiteIndexedPage.indexed_at.desc(), WebsiteIndexedPage.url.asc()).all()
    include_rules = WebsiteScrapeRule.query.filter_by(rule_type='include').order_by(WebsiteScrapeRule.created_at.desc()).all()
    exclude_rules = WebsiteScrapeRule.query.filter_by(rule_type='exclude').order_by(WebsiteScrapeRule.created_at.desc()).all()
    return jsonify({
        "success": True,
        "indexed_pages": [
            {
                "id": p.id,
                "url": p.url,
                "title": p.title,
                "chunk_count": p.chunk_count,
                "indexed_at": utc_iso(p.indexed_at),
            }
            for p in indexed_pages
        ],
        "include_rules": [
            {"id": r.id, "url": r.url, "created_at": utc_iso(r.created_at)}
            for r in include_rules
        ],
        "exclude_rules": [
            {"id": r.id, "url": r.url, "created_at": utc_iso(r.created_at)}
            for r in exclude_rules
        ],
    })


@app.route('/api/website/include', methods=['POST'])
@login_required
def api_website_include():
    data = request.get_json(silent=True) or {}
    raw_url = (data.get('url') or '').strip()
    if not raw_url:
        return jsonify({"success": False, "message": "URL mangler"}), 400
    url = _normalise_rule_url(raw_url)
    if not _web_is_manual_include_url(url):
        return jsonify({"success": False, "message": "URL'en skal være en gyldig http(s)-side"}), 400

    rule = WebsiteScrapeRule.query.filter_by(url=url).first()
    if not rule:
        rule = WebsiteScrapeRule(url=url)
        db.session.add(rule)
    rule.rule_type = 'include'
    rule.created_at = datetime.utcnow()
    rule.created_by = session.get('user_id')
    db.session.commit()
    return jsonify({"success": True, "message": "Siden er tilføjet til scrape-listen"})


@app.route('/api/website/exclude', methods=['POST'])
@login_required
def api_website_exclude():
    data = request.get_json(silent=True) or {}
    raw_url = (data.get('url') or '').strip()
    if not raw_url:
        return jsonify({"success": False, "message": "URL mangler"}), 400
    url = _normalise_rule_url(raw_url)

    rule = WebsiteScrapeRule.query.filter_by(url=url).first()
    if not rule:
        rule = WebsiteScrapeRule(url=url)
        db.session.add(rule)
    rule.rule_type = 'exclude'
    rule.created_at = datetime.utcnow()
    rule.created_by = session.get('user_id')

    page = WebsiteIndexedPage.query.filter_by(url=url).first()
    if page:
        db.session.delete(page)
    db.session.commit()
    _delete_website_page_from_qdrant(url)
    return jsonify({"success": True, "message": "Siden er ekskluderet fra scraping"})


@app.route('/api/website/rules/<int:rule_id>', methods=['DELETE'])
@login_required
def api_website_delete_rule(rule_id):
    rule = WebsiteScrapeRule.query.get_or_404(rule_id)
    db.session.delete(rule)
    db.session.commit()
    return jsonify({"success": True, "message": "Reglen er fjernet"})


# ─────────────────────────────────────────────────────────────────────────────
# Chatbot preview / demo page
# ─────────────────────────────────────────────────────────────────────────────
@app.route('/chatbot-demo')
@login_required
def chatbot_demo():
    """Standalone blank demo page showing the chatbot as it appears on the website."""
    return render_template('chatbot_preview.html')


@app.route('/test')
def employee_test():
    """Standalone test/training page for JKF employees — no login required."""
    return render_template('employee_test.html')


# ─────────────────────────────────────────────────────────────────────────────
# Health check
# ─────────────────────────────────────────────────────────────────────────────
@app.route('/health')
def health():
    try:
        db.session.execute(text("SELECT 1"))
        return jsonify({"status": "ok", "database": "connected"}), 200
    except Exception as e:
        return jsonify({"status": "error", "database": str(e)}), 500


# ─────────────────────────────────────────────────────────────────────────────
# Utility helpers
# ─────────────────────────────────────────────────────────────────────────────
def _count_unique_threads(start, end) -> int:
    if isinstance(start, datetime):
        start = start.date()
    if isinstance(end, datetime):
        end = end.date()
    return (db.session.query(func.count(Message.thread_id.distinct()))
            .filter(Message.role == 'user',
                    func.date(Message.timestamp) >= start,
                    func.date(Message.timestamp) <= end)
            .scalar() or 0)


def _calculate_csat(feedback_data: dict) -> float:
    if not feedback_data:
        return 0.0
    total = sum(feedback_data.values())
    if not total:
        return 0.0
    max_r = max(feedback_data.keys())
    satisfied = feedback_data.get(4, 0) + feedback_data.get(5, 0) if max_r >= 4 else feedback_data.get(3, 0)
    return round(satisfied / total * 100, 1)


@retry(stop=stop_after_attempt(3), wait=wait_exponential(multiplier=1, min=2, max=8))
def _store_messages(thread_id: str, user_msg: str, assistant_msg: str, metadata: dict):
    try:
        lang = metadata.get('language') if metadata else None
        cat = metadata.get('category') if metadata else None
        raw_gaps = metadata.get('knowledge_gaps') if metadata else None
        try:
            gaps_int = int(raw_gaps) if raw_gaps is not None else None
        except (ValueError, TypeError):
            gaps_int = None

        db.session.add(Message(thread_id=thread_id, role='user', content=user_msg, language=lang))
        db.session.add(Message(thread_id=thread_id, role='assistant', content=assistant_msg,
                               language=lang, category=cat, knowledge_gaps=gaps_int))
        db.session.commit()
    except Exception as e:
        db.session.rollback()
        logger.error(f"Store messages error: {e}")
        raise


# ─────────────────────────────────────────────────────────────────────────────
# Document (Vidensbase) routes
# ─────────────────────────────────────────────────────────────────────────────

def _allowed_file(filename):
    allowed = app.config.get('ALLOWED_EXTENSIONS', {'pdf', 'txt'})
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in allowed


def _chunk_text(text, chunk_size=1500, overlap=200):
    """Split text into heading-aware chunks using paragraph overlap."""
    text = text.strip()
    if not text:
        return []
    if len(text) <= chunk_size:
        return [text]

    def _split_large_block(block):
        block = block.strip()
        if len(block) <= chunk_size:
            return [block] if block else []

        parts = re.split(r"(?<=[.!?])\s+", block)
        if len(parts) == 1:
            parts = re.split(r"(?<=,)\s+", block)

        chunks = []
        current = ""
        for part in parts:
            part = part.strip()
            if not part:
                continue
            candidate = f"{current} {part}".strip() if current else part
            if current and len(candidate) > chunk_size:
                chunks.append(current)
                current = part
            else:
                current = candidate

        if current:
            chunks.append(current)
        return chunks or [block]

    raw_blocks = [block.strip() for block in re.split(r"\n\s*\n", text) if block.strip()]
    blocks = []
    for block in raw_blocks:
        if len(block) > chunk_size:
            blocks.extend(_split_large_block(block))
        else:
            blocks.append(block)

    chunks = []
    current_blocks = []
    current_len = 0
    last_heading = ""

    def flush():
        nonlocal current_blocks, current_len
        if not current_blocks:
            return

        chunk = "\n\n".join(current_blocks).strip()
        if chunk:
            chunks.append(chunk)

        overlap_blocks = []
        overlap_len = 0
        for block in reversed(current_blocks):
            if block.startswith("## "):
                continue
            block_len = len(block)
            if overlap_blocks and overlap_len + block_len > overlap:
                break
            overlap_blocks.insert(0, block)
            overlap_len += block_len

        current_blocks = []
        if last_heading:
            current_blocks.append(last_heading)
        current_blocks.extend(overlap_blocks)
        current_len = sum(len(block) for block in current_blocks)

    for block in blocks:
        is_heading = block.startswith("## ")
        if is_heading:
            last_heading = block
            if current_blocks and current_blocks != [block]:
                flush()
            current_blocks = [block]
            current_len = len(block)
            continue

        proposed_len = current_len + len(block) + (4 if current_blocks else 0)
        if current_blocks and proposed_len > chunk_size:
            flush()

        if not current_blocks and last_heading:
            current_blocks.append(last_heading)
            current_len = len(last_heading)

        current_blocks.append(block)
        current_len += len(block) + (4 if len(current_blocks) > 1 else 0)

    flush()

    deduped = []
    seen = set()
    for chunk in chunks:
        key = re.sub(r"[\W_]+", "", chunk.lower())
        if key and key not in seen:
            seen.add(key)
            deduped.append(chunk)
    return deduped


def _parse_pdf_worker(filepath, output_file):
    """Stand-alone worker that parses a PDF and writes extracted text to a file.

    Run in a separate subprocess so pdfplumber's CPU-intensive table detection
    does not compete with Flask's GIL and freeze the server.
    """
    import pdfplumber
    page_blocks = []
    with pdfplumber.open(filepath) as pdf:
        for page_num, page in enumerate(pdf.pages, start=1):
            parts = []

            tables = page.extract_tables()
            table_bboxes = []
            for table_obj in page.find_tables():
                table_bboxes.append(table_obj.bbox)

            for table in tables:
                if not table:
                    continue
                all_cells = [cell for row in table for cell in row]
                non_empty = [c for c in all_cells if c and c.strip()]
                if len(all_cells) == 0 or len(non_empty) / len(all_cells) < 0.20:
                    continue
                rows = [[cell or '' for cell in row] for row in table if any(cell for cell in row)]
                if not rows:
                    continue
                header = rows[0]
                formatted_rows = [' | '.join(header)]
                formatted_rows.append('-' * len(' | '.join(header)))
                for row in rows[1:]:
                    formatted_rows.append(' | '.join(row))
                parts.append('\n'.join(formatted_rows))

            remaining_page = page
            for bbox in table_bboxes:
                try:
                    remaining_page = remaining_page.outside_bbox(bbox)
                except Exception:
                    pass
            flowing_text = (remaining_page.extract_text(x_tolerance=3, y_tolerance=3) or '').strip()
            if flowing_text:
                parts.insert(0, flowing_text)

            if parts:
                block = f'[Side {page_num}]\n' + '\n\n'.join(parts)
                page_blocks.append(block)

    with open(output_file, 'w', encoding='utf-8') as fh:
        fh.write('\n\n'.join(page_blocks))


def _parse_pdf(filepath):
    """Return full text of a PDF by running pdfplumber in a subprocess.

    Running in a subprocess gives pdfplumber its own GIL so the heavy
    table-detection work does not stall Flask's request-handling threads.
    """
    import subprocess, tempfile, sys

    # Write the worker script to a temporary .py file so we can execute it
    # cleanly without __file__ / -c mode limitations.
    worker_script = r"""
import sys, pdfplumber

filepath  = sys.argv[1]
out_path  = sys.argv[2]

page_blocks = []
with pdfplumber.open(filepath) as pdf:
    for page_num, page in enumerate(pdf.pages, start=1):
        parts = []
        tables = page.extract_tables()
        table_bboxes = [t.bbox for t in page.find_tables()]
        for table in tables:
            if not table:
                continue
            all_cells = [cell for row in table for cell in row]
            non_empty = [c for c in all_cells if c and c.strip()]
            if not all_cells or len(non_empty) / len(all_cells) < 0.20:
                continue
            rows = [[cell or '' for cell in row] for row in table if any(cell for cell in row)]
            if not rows:
                continue
            header = rows[0]
            formatted = [' | '.join(header), '-' * len(' | '.join(header))]
            for row in rows[1:]:
                formatted.append(' | '.join(row))
            parts.append('\n'.join(formatted))
        remaining = page
        for bbox in table_bboxes:
            try:
                remaining = remaining.outside_bbox(bbox)
            except Exception:
                pass
        ft = (remaining.extract_text(x_tolerance=3, y_tolerance=3) or '').strip()
        if ft:
            parts.insert(0, ft)
        if parts:
            page_blocks.append(f'[Side {page_num}]\n' + '\n\n'.join(parts))

with open(out_path, 'w', encoding='utf-8') as fh:
    fh.write('\n\n'.join(page_blocks))
"""

    with tempfile.NamedTemporaryFile(suffix='.py', delete=False, mode='w') as ws:
        ws.write(worker_script)
        worker_path = ws.name

    with tempfile.NamedTemporaryFile(suffix='.txt', delete=False) as tmp:
        out_path = tmp.name

    try:
        result = subprocess.run(
            [sys.executable, worker_path, filepath, out_path],
            timeout=300,
            capture_output=True,
            text=True,
        )
        if result.returncode != 0:
            logger.error(f'PDF worker stderr: {result.stderr[:500]}')
            raise RuntimeError(f'PDF subprocess failed: {result.stderr[:200]}')
        with open(out_path, 'r', encoding='utf-8') as fh:
            return fh.read()
    finally:
        for p in (worker_path, out_path):
            try:
                os.unlink(p)
            except OSError:
                pass


def _parse_txt(filepath):
    for enc in ('utf-8', 'latin-1', 'cp1252'):
        try:
            with open(filepath, 'r', encoding=enc) as f:
                return f.read()
        except UnicodeDecodeError:
            continue
    return ''


def _process_document(doc_id):
    """Parse, chunk, embed and upsert a document to Qdrant. Runs synchronously."""
    logger.info(f'_process_document starting for doc_id={doc_id}')
    with app.app_context():
        doc = Document.query.get(doc_id)
        if not doc:
            logger.warning(f'_process_document: doc {doc_id} not found')
            return
        logger.info(f'_process_document: parsing {doc.original_name}')
        upload_folder = app.config.get('UPLOAD_FOLDER', 'uploads')
        filepath = os.path.join(upload_folder, doc.stored_name)
        try:
            if doc.file_type == 'pdf':
                raw_text = _parse_pdf(filepath)
            else:
                raw_text = _parse_txt(filepath)

            logger.info(f'_process_document: parsed {len(raw_text)} chars from {doc.original_name}')

            if not raw_text.strip():
                doc.status = 'error'
                doc.error_message = 'Ingen tekst fundet i dokumentet'
                db.session.commit()
                return

            chunks = _chunk_text(raw_text)
            # Drop micro-chunks (PDF headers/footers, lone URLs, stray lines)
            before_filter = len(chunks)
            chunks = [c for c in chunks if len(c.split()) >= 30]
            if len(chunks) < before_filter:
                logger.info(f'_process_document: dropped {before_filter - len(chunks)} micro-chunks (< 20 words)')
            logger.info(f'_process_document: {len(chunks)} chunks for {doc.original_name}')
            if not chunks:
                doc.status = 'error'
                doc.error_message = 'Kunne ikke opdele dokumentet i bidder'
                db.session.commit()
                return

            qc = get_qdrant_client()
            logger.info(f'_process_document: qdrant client = {qc is not None}')
            if not qc:
                doc.status = 'error'
                doc.error_message = 'Qdrant er ikke forbundet'
                db.session.commit()
                return

            from qdrant_client.models import PointStruct, SparseVector as QSparseVector
            logger.info(f'_process_document: starting embedding loop for {doc.original_name}')
            synced = 0
            for i, chunk in enumerate(chunks):
                if i == 0:
                    logger.info(f'_process_document: calling OpenAI for chunk 0')
                try:
                    emb = client.embeddings.create(model='text-embedding-3-large', input=chunk)
                    if i == 0:
                        logger.info(f'_process_document: chunk 0 embedded OK')
                    dense = emb.data[0].embedding
                    sp_idx, sp_val = _compute_sparse_vector(chunk)
                    point_id = str(uuid.uuid5(uuid.NAMESPACE_DNS, f'jkf-doc-{doc.id}-chunk-{i}'))
                    qc.upsert(
                        collection_name=QDRANT_COLLECTION,
                        points=[PointStruct(
                            id=point_id,
                            vector={'dense': dense, 'sparse': QSparseVector(indices=sp_idx, values=sp_val)},
                            payload={
                                'text': chunk,
                                'source_type': 'document',
                                'document_id': doc.id,
                                'filename': doc.original_name,
                                'chunk_index': i,
                                'created_at': utc_iso(doc.created_at),
                            }
                        )]
                    )
                    synced += 1
                except Exception as e:
                    logger.error(f'Chunk {i} embed/upsert error for doc {doc.id}: {e}')

            doc.num_chunks = len(chunks)
            doc.qdrant_synced = synced > 0
            doc.status = 'ready' if synced > 0 else 'error'
            if synced == 0:
                doc.error_message = 'Ingen bidder kunne synkroniseres til Qdrant'
            db.session.commit()

        except Exception as e:
            logger.error(f'Document processing error for doc {doc.id}: {e}')
            doc.status = 'error'
            doc.error_message = str(e)[:500]
            db.session.commit()


@app.route('/vidensbase')
@login_required
def documents_page():
    docs = Document.query.filter_by(is_active=True).order_by(Document.created_at.desc()).all()
    return render_template('documents.html', documents=docs)


@app.route('/api/documents', methods=['GET'])
@login_required
def api_documents_list():
    docs = Document.query.filter_by(is_active=True).order_by(Document.created_at.desc()).all()
    return jsonify({'success': True, 'documents': [_doc_to_dict(d) for d in docs]})


def _doc_to_dict(d):
    return {
        'id': d.id,
        'original_name': d.original_name,
        'file_type': d.file_type,
        'file_size': d.file_size,
        'num_chunks': d.num_chunks,
        'status': d.status,
        'error_message': d.error_message,
        'qdrant_synced': d.qdrant_synced,
        'created_at': d.created_at.replace(tzinfo=pytz.utc).astimezone(danish_tz).strftime('%d/%m/%Y %H:%M') if d.created_at else '',
    }


@app.route('/api/documents/upload', methods=['POST'])
@login_required
def api_upload_document():
    """Accept a single file per request (called sequentially from the frontend)."""
    if 'file' not in request.files:
        return jsonify({'success': False, 'message': 'Ingen fil valgt'}), 400

    file = request.files['file']
    if not file or file.filename == '':
        return jsonify({'success': False, 'message': 'Ingen fil valgt'}), 400

    safe_name = secure_filename(file.filename)
    if not safe_name:
        return jsonify({'success': False, 'message': 'Ugyldigt filnavn'}), 400

    if not _allowed_file(safe_name):
        return jsonify({'success': False, 'message': 'Filtype ikke understøttet (kun PDF og TXT)',
                        'name': safe_name}), 400

    existing = Document.query.filter_by(original_name=safe_name, is_active=True).first()
    if existing:
        return jsonify({'success': False,
                        'message': f'En fil med navnet "{safe_name}" findes allerede i vidensbasen.',
                        'name': safe_name}), 409

    upload_folder = app.config.get('UPLOAD_FOLDER', 'uploads')
    os.makedirs(upload_folder, exist_ok=True)

    ext = safe_name.rsplit('.', 1)[1].lower()
    stored = f"{uuid.uuid4().hex}.{ext}"
    filepath = os.path.join(upload_folder, stored)
    file.save(filepath)
    size = os.path.getsize(filepath)

    doc = Document(
        original_name=safe_name,
        stored_name=stored,
        file_type=ext,
        file_size=size,
        status='processing',
        created_by=session.get('user_id'),
    )
    db.session.add(doc)
    db.session.flush()
    doc_id = doc.id
    db.session.commit()

    t = threading.Thread(target=_process_document, args=(doc_id,), daemon=True)
    t.start()

    return jsonify({'success': True, 'document': _doc_to_dict(doc)})


@app.route('/api/documents/<int:doc_id>', methods=['DELETE'])
@login_required
def api_delete_document(doc_id):
    doc = Document.query.get_or_404(doc_id)
    _delete_document_from_qdrant(doc)
    doc.is_active = False
    db.session.commit()
    return jsonify({'success': True})


@app.route('/api/documents/<int:doc_id>', methods=['PATCH'])
@login_required
def api_rename_document(doc_id):
    doc = Document.query.get_or_404(doc_id)
    data = request.get_json() or {}
    new_name = data.get('name', '').strip()
    if not new_name:
        return jsonify({'success': False, 'message': 'Navn må ikke være tomt'}), 400
    doc.original_name = new_name
    db.session.commit()
    return jsonify({'success': True, 'document': _doc_to_dict(doc)})


@app.route('/api/documents/batch-delete', methods=['POST'])
@login_required
def api_batch_delete_documents():
    data = request.get_json() or {}
    ids = data.get('ids', [])
    if not ids:
        return jsonify({'success': False, 'message': 'Ingen dokumenter valgt'}), 400
    deleted = 0
    for doc_id in ids:
        doc = Document.query.get(doc_id)
        if doc and doc.is_active:
            _delete_document_from_qdrant(doc)
            doc.is_active = False
            deleted += 1
    db.session.commit()
    return jsonify({'success': True, 'deleted': deleted})


def _delete_document_from_qdrant(doc):
    """Remove all Qdrant points belonging to this document."""
    if not doc.qdrant_synced or doc.num_chunks == 0:
        return
    try:
        qc = get_qdrant_client()
        if not qc:
            return
        from qdrant_client.models import PointIdsList
        point_ids = [
            str(uuid.uuid5(uuid.NAMESPACE_DNS, f'jkf-doc-{doc.id}-chunk-{i}'))
            for i in range(doc.num_chunks)
        ]
        qc.delete(collection_name=QDRANT_COLLECTION,
                  points_selector=PointIdsList(points=point_ids))
    except Exception as e:
        logger.error(f'Qdrant delete error for document {doc.id}: {e}')


# ─────────────────────────────────────────────────────────────────────────────
# Sales Datawarehouse Chatbot
# ─────────────────────────────────────────────────────────────────────────────
_dw_schema_cache = None


def get_dw_config() -> dict:
    """Return DW connection config from DB integration record, falling back to env vars."""
    dw = Integration.query.filter_by(integration_type='datawarehouse_sql').first()
    if dw and dw.config:
        cfg = json.loads(dw.config)
        if cfg.get('host'):
            return cfg
    return {
        'host':     os.environ.get('DW_HOST', ''),
        'port':     int(os.environ.get('DW_PORT', 1433)),
        'database': os.environ.get('DW_DATABASE', ''),
        'username': os.environ.get('DW_USERNAME', ''),
        'password': os.environ.get('DW_PASSWORD', ''),
    }


def get_dw_connection():
    import pymssql
    cfg = get_dw_config()
    return pymssql.connect(
        server=cfg['host'],
        port=int(cfg.get('port', 1433)),
        database=cfg['database'],
        user=cfg['username'],
        password=cfg['password'],
        login_timeout=10,
        charset='UTF-8',
    )


# ─────────────────────────────────────────────────────────────────────────────
# Datawarehouse integration settings
# ─────────────────────────────────────────────────────────────────────────────
@app.route('/integrations/datawarehouse', methods=['GET', 'POST'])
@login_required
def integration_dw():
    global _dw_schema_cache
    dw = Integration.query.filter_by(integration_type='datawarehouse_sql').first()
    if not dw:
        dw = Integration(integration_type='datawarehouse_sql', name='Salgs-datawarehouse')
        db.session.add(dw)
        db.session.commit()

    if request.method == 'POST':
        # Fall back to env var if DB config has no password yet (first save)
        existing_password = (
            json.loads(dw.config or '{}').get('password', '')
            or os.environ.get('DW_PASSWORD', '')
        )
        submitted_password = request.form.get('password', '').strip()
        resolved_password = (
            submitted_password
            if submitted_password and submitted_password != '••••••••'
            else existing_password
        )
        config = {
            'host':     request.form.get('host', '').strip(),
            'port':     int(request.form.get('port', '1433').strip() or 1433),
            'database': request.form.get('database', '').strip(),
            'username': request.form.get('username', '').strip(),
            'password': resolved_password,
        }
        dw.config = json.dumps(config)
        dw.updated_at = datetime.utcnow()
        db.session.commit()
        _dw_schema_cache = None     # invalidate cached schemas on credential change
        _budget_schema_cache = None
        _budget_view_map.clear()
        flash('Datawarehouse indstillinger gemt.', 'success')
        return redirect(url_for('integration_dw'))

    cfg = json.loads(dw.config or '{}')
    if not cfg.get('host'):
        cfg = {
            'host':     os.environ.get('DW_HOST', ''),
            'port':     os.environ.get('DW_PORT', '1433'),
            'database': os.environ.get('DW_DATABASE', ''),
            'username': os.environ.get('DW_USERNAME', ''),
            'password': os.environ.get('DW_PASSWORD', ''),
        }
    return render_template('integration_dw.html', dw=dw, cfg=cfg)


@app.route('/api/integrations/datawarehouse/test', methods=['POST'])
@login_required
def api_dw_test():
    import pymssql
    data = request.get_json() or {}
    host     = data.get('host', '').strip()
    port     = int(data.get('port', 1433))
    database = data.get('database', '').strip()
    username = data.get('username', '').strip()
    password = data.get('password', '').strip()
    if not all([host, database, username, password]):
        return jsonify({'success': False, 'message': 'Udfyld alle felter.'})
    try:
        conn = pymssql.connect(
            server=host, port=port, database=database,
            user=username, password=password, login_timeout=10,
            charset='UTF-8',
        )
        cursor = conn.cursor()
        cursor.execute('SELECT COUNT(*) FROM INFORMATION_SCHEMA.TABLES')
        count = cursor.fetchone()[0]
        conn.close()
        return jsonify({'success': True, 'message': f'Forbindelse OK – {count} tabeller fundet i {database}.'})
    except Exception as e:
        return jsonify({'success': False, 'message': f'Forbindelsesfejl: {str(e)}'})


def get_dw_schema() -> str:
    global _dw_schema_cache
    if _dw_schema_cache:
        return _dw_schema_cache
    conn = get_dw_connection()
    cursor = conn.cursor()
    cursor.execute("""
        SELECT TABLE_NAME, COLUMN_NAME, DATA_TYPE
        FROM INFORMATION_SCHEMA.COLUMNS
        ORDER BY TABLE_NAME, ORDINAL_POSITION
    """)
    rows = cursor.fetchall()
    conn.close()
    tables: dict = {}
    for table, col, dtype in rows:
        tables.setdefault(table, []).append(f"{col} {dtype}")
    lines = [f"{t}({', '.join(cols)})" for t, cols in tables.items()]
    _dw_schema_cache = "\n".join(lines)
    return _dw_schema_cache


_DW_DANGEROUS = re.compile(
    r'\b(INSERT|UPDATE|DELETE|DROP|TRUNCATE|ALTER|CREATE|EXEC|EXECUTE|MERGE|GRANT|REVOKE)\b',
    re.IGNORECASE,
)
_DW_UNSUPPORTED = re.compile(r'\bQUALIFY\b', re.IGNORECASE)


_ALLOWED_SALES_TABLES = {
    'salg',
    'sales invoice header',
    'sales invoice line',
    'customer',
    'item',
}


def _validate_sales_tables(sql: str) -> tuple:
    """Return (ok, offending_name). Only allows known sales tables and CTE names."""
    cte_names = {m.lower() for m in re.findall(
        r'(?:WITH|,)\s+(\w+)\s*(?:\([^)]*\))?\s+AS\s*\(', sql, re.IGNORECASE
    )}
    allowed = cte_names | _ALLOWED_SALES_TABLES
    # Match both bracketed [Multi Word Name] and plain identifiers
    for m in re.finditer(r'\b(?:FROM|JOIN)\s+(?:\[([^\]]+)\]|(\w+))', sql, re.IGNORECASE):
        ref = (m.group(1) or m.group(2) or '').lower()
        if ref and ref not in allowed:
            return False, m.group(1) or m.group(2)
    return True, ''


def _extract_sql(text: str) -> str:
    """Strip markdown fences and return the raw SQL string."""
    text = text.strip()
    fenced = re.search(r'```(?:sql)?\s*([\s\S]+?)```', text, re.IGNORECASE)
    if fenced:
        return fenced.group(1).strip()
    return text


@app.route('/sales-chatbot')
@sales_chatbot_required
def sales_chatbot():
    user_id = session.get('user_id')
    conversations = (SalesConversation.query
                     .filter_by(user_id=user_id)
                     .order_by(SalesConversation.updated_at.desc())
                     .all())
    return render_template('sales_chatbot.html', conversations=conversations)


@app.route('/api/sales-conversations', methods=['GET'])
@sales_chatbot_required
def api_sales_conversations():
    user_id = session.get('user_id')
    convs = (SalesConversation.query
             .filter_by(user_id=user_id)
             .order_by(SalesConversation.updated_at.desc())
             .all())
    return jsonify([{
        'id': c.id,
        'title': c.title,
        'updated_at': c.updated_at.isoformat(),
    } for c in convs])


@app.route('/api/sales-conversations/<int:conv_id>', methods=['GET'])
@sales_chatbot_required
def api_sales_conversation_detail(conv_id):
    user_id = session.get('user_id')
    conv = SalesConversation.query.filter_by(id=conv_id, user_id=user_id).first_or_404()
    return jsonify({
        'id': conv.id,
        'title': conv.title,
        'messages': [{
            'id': m.id,
            'role': m.role,
            'content': m.content,
            'sql': m.sql_query,
            'row_count': m.row_count,
        } for m in conv.messages],
    })


@app.route('/api/sales-conversations/<int:conv_id>', methods=['DELETE'])
@sales_chatbot_required
def api_sales_conversation_delete(conv_id):
    user_id = session.get('user_id')
    conv = SalesConversation.query.filter_by(id=conv_id, user_id=user_id).first_or_404()
    db.session.delete(conv)
    db.session.commit()
    return jsonify({'ok': True})


def _pdf_html(title, agent_name, exported_at, incl_user, messages_html):
    incl_label = 'inkluderet' if incl_user else 'ikke vist'
    return f'''<!DOCTYPE html>
<html lang="da">
<head>
<meta charset="UTF-8">
<title>{title} – JKF</title>
<style>
  @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700&display=swap');
  *, *::before, *::after {{ box-sizing: border-box; margin: 0; padding: 0; }}
  body {{ font-family: "Inter", "Segoe UI", Arial, sans-serif; font-size: 12.5px; line-height: 1.65; color: #1e293b; background: #fff; padding: 0 0 40px; }}
  .page-header {{ background: linear-gradient(135deg, #1a3a72 0%, #214786 60%, #2d5fa8 100%); color: #fff; padding: 28px 44px 22px; display: flex; align-items: flex-start; justify-content: space-between; gap: 24px; }}
  .ph-left {{ display: flex; flex-direction: column; gap: 6px; }}
  .ph-brand {{ font-size: 11px; font-weight: 600; letter-spacing: .12em; text-transform: uppercase; opacity: .75; }}
  .ph-title {{ font-size: 20px; font-weight: 700; line-height: 1.25; max-width: 560px; }}
  .ph-right {{ text-align: right; font-size: 11px; opacity: .85; line-height: 1.9; white-space: nowrap; }}
  .ph-right strong {{ font-weight: 600; opacity: 1; }}
  .sub-bar {{ background: #f0f4fb; border-bottom: 1px solid #d4dff5; padding: 9px 44px; font-size: 11px; color: #4a5f8a; display: flex; gap: 24px; }}
  .sub-bar span {{ display: flex; align-items: center; gap: 5px; }}
  .content {{ padding: 20px 32px; }}
  .turn {{ margin-bottom: 26px; }}
  .turn-label {{ margin-bottom: 6px; }}
  .label-chip {{ display: inline-block; padding: 2px 9px; border-radius: 20px; font-size: 10px; font-weight: 600; letter-spacing: .06em; text-transform: uppercase; }}
  .user-chip  {{ background: #dbeafe; color: #1d4ed8; }}
  .asst-chip  {{ background: #dcfce7; color: #15803d; }}
  .bubble {{ border-radius: 10px; padding: 14px 18px; }}
  .user-bubble {{ background: #eff6ff; border-left: 3px solid #3b82f6; color: #1e3a5f; }}
  .asst-bubble {{ background: #f8fafc; color: #1e293b; }}
  .asst-bubble p  {{ margin-bottom: 6px; }}
  .asst-bubble p:last-child {{ margin-bottom: 0; }}
  .asst-bubble h2 {{ font-size: 13px; font-weight: 700; margin: 10px 0 4px; color: #214786; }}
  .asst-bubble h3 {{ font-size: 12px; font-weight: 600; margin: 8px 0 3px; color: #334155; }}
  .asst-bubble ul {{ padding-left: 18px; margin: 4px 0 6px; }}
  .asst-bubble li {{ margin-bottom: 3px; }}
  .spacer {{ height: 4px; }}
  .tbl-wrap {{ overflow-x: auto; margin: 10px 0; border-radius: 6px; border: 1px solid #e2e8f0; width: 100%; }}
  .md-table {{ border-collapse: collapse; width: 100%; font-size: 10px; table-layout: auto; }}
  .md-table th {{ background: #214786; color: #fff; padding: 5px 7px; text-align: left; font-weight: 600; font-size: 9.5px; letter-spacing: .02em; }}
  .md-table td {{ padding: 5px 7px; border-bottom: 1px solid #e8eef6; font-size: 10px; }}
  .md-table tbody tr:last-child td {{ border-bottom: none; }}
  .md-table tbody tr:nth-child(even) td {{ background: #f5f8ff; }}
  strong {{ font-weight: 600; }}
  em      {{ font-style: italic; }}
  @page {{ margin: 0; size: A4 portrait; }}
  @media print {{
    body {{ padding: 0; }}
    .content {{ padding: 16px 30px; }}
    .page-header {{ padding: 20px 30px 16px; print-color-adjust: exact; -webkit-print-color-adjust: exact; }}
    .sub-bar {{ padding: 7px 30px; print-color-adjust: exact; -webkit-print-color-adjust: exact; }}
    .md-table th {{ print-color-adjust: exact; -webkit-print-color-adjust: exact; }}
    .tbl-wrap {{ overflow-x: visible; }}
    .md-table tr {{ page-break-inside: avoid; }}
  }}
</style>
</head>
<body>
<div class="page-header">
  <div class="ph-left">
    <div class="ph-brand">JKF AI Platform · {agent_name}</div>
    <div class="ph-title">{title}</div>
  </div>
  <div class="ph-right">
    <strong>Eksporteret</strong><br>{exported_at}
  </div>
</div>
<div class="sub-bar">
  <span>Bruger-spørgsmål: {incl_label}</span>
</div>
<div class="content">
{messages_html}
</div>
</body>
</html>'''


@app.route('/api/sales-conversations/<int:conv_id>/export/pdf')
@sales_chatbot_required
def api_sales_conversation_export_pdf(conv_id):
    import re as _re

    user_id = session.get('user_id')
    conv = SalesConversation.query.filter_by(id=conv_id, user_id=user_id).first_or_404()

    # Query-param filtering
    raw_ids   = request.args.get('msg_ids', '')
    incl_user = request.args.get('include_user', '1') == '1'
    selected_ids = set(int(x) for x in raw_ids.split(',') if x.strip().isdigit()) if raw_ids else None

    def md_to_html(text):
        text = _re.sub(r'\*\*(.+?)\*\*', r'<strong>\1</strong>', text)
        text = _re.sub(r'\*(.+?)\*',     r'<em>\1</em>',         text)
        lines = text.split('\n')
        out, in_table, in_list = [], False, False
        for line in lines:
            if line.startswith('|'):
                cells = [c.strip() for c in line.strip('|').split('|')]
                if all(set(c) <= set('-:| ') for c in cells):
                    continue
                if not in_table:
                    if in_list: out.append('</ul>'); in_list = False
                    out.append('<div class="tbl-wrap"><table class="md-table"><thead><tr>')
                    out.extend(f'<th>{c}</th>' for c in cells)
                    out.append('</tr></thead><tbody>')
                    in_table = True
                else:
                    out.append('<tr>')
                    out.extend(f'<td>{c}</td>' for c in cells)
                    out.append('</tr>')
            else:
                if in_table: out.append('</tbody></table></div>'); in_table = False
                stripped = line.strip()
                if stripped.startswith('- ') or stripped.startswith('* '):
                    if not in_list: out.append('<ul>'); in_list = True
                    out.append(f'<li>{stripped[2:]}</li>')
                elif stripped.startswith('### '):
                    if in_list: out.append('</ul>'); in_list = False
                    out.append(f'<h3>{stripped[4:]}</h3>')
                elif stripped.startswith('## '):
                    if in_list: out.append('</ul>'); in_list = False
                    out.append(f'<h2>{stripped[3:]}</h2>')
                elif stripped == '':
                    if in_list: out.append('</ul>'); in_list = False
                    out.append('<div class="spacer"></div>')
                else:
                    if in_list: out.append('</ul>'); in_list = False
                    out.append(f'<p>{line}</p>')
        if in_table: out.append('</tbody></table></div>')
        if in_list:  out.append('</ul>')
        return '\n'.join(out)

    messages_html = ''
    turn_num = 0
    all_messages = list(conv.messages)
    for i, msg in enumerate(all_messages):
        if msg.role == 'user':
            if not incl_user:
                continue
            # Only show this question if its paired answer is selected
            next_asst = next((m for m in all_messages[i+1:] if m.role == 'assistant'), None)
            if selected_ids is not None and (next_asst is None or next_asst.id not in selected_ids):
                continue
            messages_html += f'''
<div class="turn user-turn">
  <div class="turn-label"><span class="label-chip user-chip">Spørgsmål</span></div>
  <div class="bubble user-bubble"><p>{msg.content}</p></div>
</div>'''
        else:
            if selected_ids is not None and msg.id not in selected_ids:
                continue
            turn_num += 1
            content_html = md_to_html(msg.content or '')
            messages_html += f'''
<div class="turn assistant-turn">
  <div class="turn-label"><span class="label-chip asst-chip">Svar {turn_num}</span></div>
  <div class="bubble asst-bubble">{content_html}</div>
</div>'''

    _da_months = ['januar','februar','marts','april','maj','juni',
                  'juli','august','september','oktober','november','december']
    _now = datetime.utcnow().replace(tzinfo=pytz.utc).astimezone(danish_tz)
    exported_at = f"{_now.day}. {_da_months[_now.month - 1]} {_now.year} kl. {_now.strftime('%H:%M')}"

    html = _pdf_html(conv.title, 'Salgs-assistent', exported_at, incl_user, messages_html)
    return html, 200, {'Content-Type': 'text/html; charset=utf-8', 'Content-Disposition': 'inline'}


@app.route('/api/sales-chat', methods=['POST'])
@sales_chatbot_required
def api_sales_chat():
    data = request.get_json(silent=True) or {}
    message = (data.get('message') or '').strip()
    history = data.get('history') or []
    conversation_id = data.get('conversation_id')

    if not message:
        return jsonify({'error': 'Ingen besked modtaget.'}), 400

    user_id = session.get('user_id')

    # Resolve or create the SalesConversation record
    conv = None
    if conversation_id:
        conv = SalesConversation.query.filter_by(id=conversation_id, user_id=user_id).first()
    if conv is None:
        # Use first ~60 chars of message as auto-title
        title = message[:60] + ('…' if len(message) > 60 else '')
        conv = SalesConversation(user_id=user_id, title=title)
        db.session.add(conv)
        db.session.flush()   # get conv.id without full commit

    # Persist the user turn immediately
    user_turn = SalesMessage(
        conversation_id=conv.id,
        role='user',
        content=message,
    )
    db.session.add(user_turn)

    try:
        full_schema = get_dw_schema()
    except Exception as e:
        logger.error(f'DW schema fetch failed: {e}')
        return jsonify({'error': 'Kunne ikke oprette forbindelse til datalageret.'}), 500

    # Restrict schema to only the allowed sales tables
    sales_lines = [l for l in full_schema.split('\n')
                   if l.split('(')[0].strip().lower() in _ALLOWED_SALES_TABLES]
    schema = '\n'.join(sales_lines) if sales_lines else full_schema

    # ── Customer name resolution ──────────────────────────────────────────────
    # Extract any customer names the user mentioned, then look up exact matches
    # in the DB so the SQL generator uses precise names instead of LIKE guesses.
    # We scan BOTH the current message AND recent user messages from history so
    # follow-up questions ("vis det for dem") resolve the same customer.
    customer_context = ''
    try:
        recent_user_msgs = [h['content'] for h in history[-8:] if h.get('role') == 'user']
        extraction_text = '\n'.join(recent_user_msgs + [message])
        extraction_resp = client.chat.completions.create(
            model='gpt-5.4-mini',
            messages=[
                {
                    'role': 'system',
                    'content': (
                        'Extract company/customer names mentioned across these messages. '
                        'Return ONLY a JSON array of strings, e.g. ["technor", "camfil"]. '
                        'Return [] if no customer names are mentioned. '
                        'Do not include general words like "kunde", "kunden", "top", "largest", "dem", "de".'
                    ),
                },
                {'role': 'user', 'content': extraction_text},
            ],
            temperature=0,
            max_completion_tokens=80,
        )
        raw = extraction_resp.choices[0].message.content or '[]'
        # Strip markdown fences if model wraps in ```json
        raw = raw.strip().strip('`').removeprefix('json').strip()
        mentioned = json.loads(raw)
    except Exception:
        mentioned = []

    if mentioned:
        resolved = {}
        try:
            conn = get_dw_connection()
            cursor = conn.cursor()
            for name in mentioned[:5]:   # cap at 5 names
                cursor.execute(
                    "SELECT DISTINCT TOP 10 [KundeNavn] FROM [Salg] "
                    "WHERE [KundeNavn] LIKE %s ORDER BY [KundeNavn]",
                    (f'%{name}%',),
                )
                rows = cursor.fetchall()
                if rows:
                    resolved[name] = [r[0] for r in rows]
            conn.close()
        except Exception as e:
            logger.warning(f'Customer name resolution failed: {e}')

        if resolved:
            lines = []
            for mention, matches in resolved.items():
                exact = ', '.join(f'"{m}"' for m in matches)
                lines.append(f'  Bruger nævnte "{mention}" → eksakte KundeNavn i databasen: {exact}')
            customer_context = (
                '\nKUNDENAVNE-OPSLAG (brug disse eksakte værdier i WHERE-klausulen, IKKE LIKE):\n'
                + '\n'.join(lines) + '\n'
            )

    sql_system = (
        "Du er en T-SQL ekspert for JKF's SQL Server datawarehouse (SQL Server 2019).\n\n"
        "VIGTIGE TABELLER:\n"
        "- [Salg]: Primær salgsoversigt – Varenummer, Beskrivelse, Varegruppe, Dato (date), KundeNavn, Land, Salgsbeløb (decimal), Rabatbeløb, dækningsbidrag. "
        "Brug kolonnen [Varegruppe] direkte fra [Salg] til alle spørgsmål om varegrupper – join ALDRIG [Item] for dette formål. "
        "Brug denne til de fleste salgs- og omsætningsspørgsmål.\n"
        "- [Sales Invoice Header]: Fakturahoveder – [Posting Date], [Sell-to Customer No_], [Bill-to Name], CompanyCode.\n"
        "- [Sales Invoice Line]: Fakturalinjer – Amount, Quantity, [No_] (varenummer), CompanyCode.\n"
        "- [Customer]: Kunder – [No_], Name, [Country_Region Code], [Salesperson Code].\n"
        "- [Item]: Varer – [No_], Description.\n\n"
        "ADGANGSBEGRÆNSNING: Du må KUN forespørge på de ovenstående tabeller. Brug ALDRIG andre tabeller eller views.\n\n"
        "FULDT SKEMA:\n"
        f"{schema}\n"
        f"{customer_context}\n"
        "OBLIGATORISKE REGLER – følg dem præcist:\n"
        "0. Du har ALTID fuld adgang til databasen med alle data. Svar ALDRIG med forklaringer om manglende data eller hvad du har brug for. "
        "Generér ALTID et SELECT-statement – uanset om spørgsmålet er et opfølgningsspørgsmål eller en ny forespørgsel.\n"
        "1. Returner KUN rå SQL, ingen forklaring, ingen markdown, ingen ```.\n"
        "2. Brug firkantede parenteser om tabelnavne og kolonner med mellemrum eller specialtegn, fx [Sales Invoice Header], [Posting Date].\n"
        "3. Giv ALTID subqueries og CTEs et tabel-alias UDEN AS-nøgleordet, fx: FROM (SELECT ...) sub – IKKE FROM (SELECT ...) AS sub. SQL Server 2019 kræver dette i visse kontekster.\n"
        "4. Kolonne-aliaser bruger AS normalt, fx: SUM(Salgsbeløb) AS Total.\n"
        "5. Brug TOP n (ikke LIMIT) for at begrænse resultater, fx SELECT TOP 20 ...\n"
        "6. CTEs: skriv WITH ctnavn (kolonne) AS (SELECT ...) SELECT ... – uden semikolon foran WITH.\n"
        "7. Brug aldrig: DROP, INSERT, UPDATE, DELETE, TRUNCATE, ALTER, CREATE, EXEC, QUALIFY.\n"
        "8. For ranking: brug ROW_NUMBER() OVER (...) i en subquery – ALDRIG QUALIFY.\n"
        "9. Sammenligning på tværs af år: brug to separate SUM med CASE eller to subqueries joinet på gruppering.\n"
        "10. YEAR(Dato) og MONTH(Dato) virker til dato-filtrering på [Salg]-tabellen.\n"
        "11. Når du finder minimum, maximum, laveste eller højeste af en beregnet værdi (margin, ratio, vækst osv.): "
        "tilføj ALWAYS en HAVING-klausul der udelukker nul-salg og NULL-resultater, "
        "fx HAVING SUM(Salgsbeløb) > 0. Dette forhindrer meningsløse NULL-resultater.\n"
        "12. SQL Server forbyder ORDER BY inde i enhver subquery eller CTE medmindre den pågældende SELECT selv har TOP/OFFSET/FOR XML. "
        "Den mest almindelige fejl er at pakke en ORDER BY i en ekstra subquery: "
        "FORKERT – giver fejl 1033:\n"
        "  WITH cte AS (SELECT TOP 10 x FROM (SELECT x FROM t ORDER BY y) inner_t)\n"
        "RIGTIGT – skriv ORDER BY direkte i det niveau der har TOP:\n"
        "  WITH cte AS (SELECT TOP 10 x FROM t GROUP BY x ORDER BY SUM(y) DESC)\n"
        "Tilføj ALDRIG et ekstra subquery-lag blot for at sortere – skriv ORDER BY i det samme SELECT der har TOP.\n"
        "13. Bevar tidsperioden fra samtalehistorikken. Hvis et tidligere spørgsmål filtrerede på fx 2025, "
        "brug samme årsfilter i opfølgningsspørgsmål med mindre brugeren eksplicit angiver et andet år.\n"
        "14. CROSS JOIN andels-beregning – dette er en hyppig fejlkilde:\n"
        "Når du beregner hver kundes/vares andel af en total skal du bruge dette mønster:\n"
        "  WITH top5 AS (\n"
        "      SELECT TOP 5 [KundeNavn], SUM([Salgsbeløb]) AS Omsætning\n"
        "      FROM [Salg] WHERE YEAR([Dato])=2025\n"
        "      GROUP BY [KundeNavn] HAVING SUM([Salgsbeløb])>0\n"
        "      ORDER BY SUM([Salgsbeløb]) DESC\n"
        "  ), total AS (\n"
        "      SELECT SUM([Salgsbeløb]) AS TotalOmsætning FROM [Salg] WHERE YEAR([Dato])=2025\n"
        "  )\n"
        "  SELECT k.[KundeNavn], k.Omsætning,\n"
        "         CAST(100.0 * k.Omsætning / NULLIF(t.TotalOmsætning,0) AS decimal(10,2)) AS AndelPct\n"
        "  FROM top5 k CROSS JOIN total t ORDER BY k.Omsætning DESC\n"
        "KRITISK REGEL: i den ydre SELECT må du ALDRIG skrive SUM(k.Omsætning) eller SUM(k.noget). "
        "CTE/subquery-kolonner er allerede aggregerede enkeltværdier – brug dem direkte (k.Omsætning, ikke SUM(k.Omsætning)). "
        "SUM() på en allerede-aggregeret kolonne giver SQL Server fejl 8120."
    )

    sql_messages = [{'role': 'system', 'content': sql_system}]
    for h in history[-10:]:
        if h.get('role') in ('user', 'assistant') and h.get('content'):
            sql_messages.append({'role': h['role'], 'content': h['content']})
    sql_messages.append({'role': 'user', 'content': message})

    try:
        sql_resp = client.chat.completions.create(
            model='gpt-5.4-mini',
            messages=sql_messages,
            temperature=0,
            max_completion_tokens=800,
        )
        raw_sql = _extract_sql(sql_resp.choices[0].message.content or '')
    except Exception as e:
        logger.error(f'SQL generation failed: {e}')
        return jsonify({'error': 'Kunne ikke generere SQL-forespørgsel.'}), 500

    # If the model returned prose instead of SQL (non-data question), return it directly
    _SQL_START = re.compile(r'^\s*(SELECT|WITH|;WITH)\b', re.IGNORECASE)
    if not _SQL_START.match(raw_sql):
        logger.info('Non-SQL response from model – cleaning up via answer model')
        try:
            cleanup_resp = client.chat.completions.create(
                model='gpt-5.4-mini',
                messages=[
                    {'role': 'system', 'content': (
                        'Du er en hjælpsom salgsassistent hos JKF. '
                        'Omskriv følgende besked til et pænt, kort dansk svar. '
                        'Fjern alle pladsholdere i kantede parenteser som [mangler] eller [ukendt]. '
                        'Forklar venligt at du ikke har nok information i konteksten til at svare præcist, '
                        'og opfordr brugeren til at stille spørgsmålet som en ny selvstændig forespørgsel.'
                    )},
                    {'role': 'user', 'content': raw_sql},
                ],
                max_completion_tokens=300,
                temperature=0.3,
            )
            clean_answer = cleanup_resp.choices[0].message.content or raw_sql
        except Exception:
            clean_answer = 'Jeg har ikke nok information i den nuværende samtale til at svare. Prøv at stille spørgsmålet på ny som en selvstændig forespørgsel.'
        return jsonify({'answer': clean_answer, 'sql': None, 'row_count': None})

    if _DW_DANGEROUS.search(raw_sql):
        return jsonify({'error': 'Sikkerhedsfejl: kun læse-forespørgsler er tilladt.'}), 400

    if _DW_UNSUPPORTED.search(raw_sql):
        return jsonify({'error': 'Den genererede SQL bruger QUALIFY, som ikke understøttes af denne SQL Server. Prøv at omformulere spørgsmålet.'}), 400

    salg_ok, bad_ref = _validate_sales_tables(raw_sql)
    if not salg_ok:
        logger.warning(f'SQL validation blocked reference to table: {bad_ref}')
        return jsonify({'error': f'Adgangsfejl: forespørgslen forsøgte at tilgå "{bad_ref}", som ikke er tilladt.'}), 400

    logger.info(f'DW SQL generated:\n{raw_sql}')
    try:
        conn = get_dw_connection()
        cursor = conn.cursor(as_dict=True)
        cursor.execute(raw_sql)
        rows = cursor.fetchmany(500)
        conn.close()
        row_count = len(rows)
        # Build a TSV-style result so the model sees unambiguous raw numbers
        # (avoids JSON float representation confusing the formatting model)
        if rows:
            headers = list(rows[0].keys())
            lines = ['\t'.join(headers)]
            for r in rows:
                lines.append('\t'.join(str(r[h]) for h in headers))
            result_text = '\n'.join(lines)
        else:
            result_text = '(ingen rækker)'
    except Exception as e:
        err_str = str(e)
        logger.error(f'DW query failed: {e}\nSQL was:\n{raw_sql}')
        if '1033' in err_str:
            return jsonify({'error': 'Den genererede SQL indeholder ORDER BY i en subquery uden TOP, som SQL Server ikke tillader. Prøv at omformulere spørgsmålet.'}), 500
        return jsonify({'error': 'Databasefejl: forespørgslen kunne ikke udføres. Prøv at omformulere dit spørgsmål.'}), 500

    answer_system = (
        "Du er en hjælpsom salgsanalytiker hos JKF. Svar altid på dansk.\n\n"
        "FORMATERINGSREGLER – følg dem præcist:\n"
        "- Når svaret indeholder tabeldata med flere rækker: brug ALTID en markdown-tabel med pipe-syntaks (|).\n"
        "- TABELRETNING – vælg den retning der giver den korteste og bredeste tabel:\n"
        "  * HORISONTAL (foretrukket): én række per enhed (kunde, varegruppe, land osv.), "
        "med tidsperioder eller målinger som kolonner. "
        "Brug dette ved sammenligninger på tværs af år/perioder/metrics, fx:\n"
        "    | Kunde         | Oms. 2023 | Oms. 2024 | Oms. 2025 | Margin% 2025 |\n"
        "    |---------------|-----------|-----------|-----------|---------------|\n"
        "    | Camfil Norge  | 1.200.000 | 980.000   | 1.450.000 | 33,8%        |\n"
        "  * VERTIKAL: kun når data naturligt har én kolonne med værdier, "
        "fx en simpel top-10 liste med én metric.\n"
        "- Aldrig lav en tabel med en 'Periode' eller 'År'-kolonne og én værdi-kolonne – "
        "det er altid bedre som en horisontal tabel med år som kolonneoverskrifter.\n"
        "- Brug punktopstilling (- eller 1.) til lister og opsummeringer.\n"
        "- Brug **fed** til vigtige tal og nøgleord.\n"
        "- TALLFORMATERINGSKRAV – dette er kritisk:\n"
        "  * Beløb og mængder: rund til nærmeste hele tal og brug dansk tusindtalsseparator (.). "
        "53980.45 vises som 53.981. 1234567.89 vises som 1.234.568. INGEN decimaler på beløb.\n"
        "  * Procenter: afrund til 1 decimal med komma som decimaltegn. 30.2934 vises som 30,3%. 5.0 vises som 5,0%.\n"
        "  * ALDRIG afkort tal til færre cifre – 53980 må IKKE vises som 53,98 eller 53.98.\n"
        "  * Behandl databaseværdien som autoritativ kilde – omformuler ikke tallet baseret på kontekst.\n"
        "- Vær præcis og konkret – brug tal direkte fra resultatet.\n"
        "- Afslut gerne med en kort konklusion eller et opfølgningsforslag."
    )
    answer_messages = [
        {'role': 'system', 'content': answer_system},
        {
            'role': 'user',
            'content': (
                f"Brugerens spørgsmål: {message}\n\n"
                f"SQL der blev kørt:\n{raw_sql}\n\n"
                f"Resultat ({row_count} rækker):\n{result_text}"
            ),
        },
    ]

    try:
        ans_resp = client.chat.completions.create(
            model='gpt-5.4-mini',
            messages=answer_messages,
            temperature=0.3,
            max_completion_tokens=3000,
        )
        answer = ans_resp.choices[0].message.content or ''
    except Exception as e:
        logger.error(f'Answer generation failed: {e}')
        return jsonify({'error': 'Kunne ikke formulere svar.'}), 500

    # Persist the assistant turn
    assistant_turn = SalesMessage(
        conversation_id=conv.id,
        role='assistant',
        content=answer,
        sql_query=raw_sql,
        row_count=row_count,
    )
    db.session.add(assistant_turn)
    conv.updated_at = datetime.utcnow()
    db.session.commit()

    return jsonify({
        'answer': answer,
        'sql': raw_sql,
        'row_count': row_count,
        'conversation_id': conv.id,
        'msg_id': assistant_turn.id,
    })


# ─────────────────────────────────────────────────────────────────────────────
# Budget agent — database helpers
# ─────────────────────────────────────────────────────────────────────────────

def get_budget_connection():
    """Connect to BC2SQL_Data (same DB as sales DW) for budget views."""
    return get_dw_connection()


_budget_schema_cache = None
_budget_account_names_cache = None
_budget_view_map = {}   # kept for cache-invalidation compatibility

_BUDGET_VIEW_SHORT_NAMES = {'vw_gl_actuals_vs_budget', 'vw_gl_entry_detailed'}


def get_budget_schema() -> str:
    global _budget_schema_cache
    if _budget_schema_cache:
        return _budget_schema_cache
    conn = get_budget_connection()
    cursor = conn.cursor()
    cursor.execute("""
        SELECT TABLE_NAME, COLUMN_NAME, DATA_TYPE
        FROM INFORMATION_SCHEMA.COLUMNS
        WHERE TABLE_NAME IN ('vw_GL_actuals_vs_budget', 'vw_GL_entry_detailed')
        ORDER BY TABLE_NAME, ORDINAL_POSITION
    """)
    rows = cursor.fetchall()
    conn.close()
    tables: dict = {}
    for table, col, dtype in rows:
        tables.setdefault(table, []).append(f"{col} {dtype}")
    lines = [f"[{t}]({', '.join(cols)})" for t, cols in tables.items()]
    _budget_schema_cache = "\n".join(lines)
    return _budget_schema_cache


def get_budget_account_names() -> str:
    """Return a cached newline-separated list of all distinct Account Name values."""
    global _budget_account_names_cache
    if _budget_account_names_cache:
        return _budget_account_names_cache
    try:
        conn = get_budget_connection()
        cursor = conn.cursor()
        cursor.execute("""
            SELECT DISTINCT [Account Name]
            FROM [vw_GL_actuals_vs_budget]
            WHERE [Account Name] IS NOT NULL
            ORDER BY [Account Name]
        """)
        names = [row[0] for row in cursor.fetchall()]
        conn.close()
        _budget_account_names_cache = "\n".join(names)
    except Exception as e:
        logger.warning(f'Could not fetch budget account names: {e}')
        _budget_account_names_cache = ''
    return _budget_account_names_cache


def _validate_budget_views(sql: str) -> tuple:
    """Return (ok, offending_name). Allows budget views (any schema) and CTE names."""
    cte_names = {m.lower() for m in re.findall(
        r'(?:WITH|,)\s+(\w+)\s*(?:\([^)]*\))?\s+AS\s*\(', sql, re.IGNORECASE
    )}

    # Match full object references after FROM/JOIN, e.g.:
    #   [dbo].[vw_GL_actuals_vs_budget]  |  [vw_GL_actuals_vs_budget]  |  plain_name
    ref_pattern = re.compile(
        r'\b(?:FROM|JOIN)\s+'
        r'((?:\[[^\]]+\]|\w+)(?:\.(?:\[[^\]]+\]|\w+))*)',
        re.IGNORECASE
    )
    for m in ref_pattern.finditer(sql):
        full_ref = m.group(1)
        # Extract all identifier parts (strip brackets)
        parts = re.findall(r'\[([^\]]+)\]|(\w+)', full_ref)
        tokens = [(a or b).lower() for a, b in parts]
        # The last token is the actual object name; earlier tokens are schema/db
        obj_name = tokens[-1] if tokens else ''
        if obj_name and obj_name not in cte_names and obj_name not in _BUDGET_VIEW_SHORT_NAMES:
            return False, obj_name
    return True, ''


# ─────────────────────────────────────────────────────────────────────────────
# Budget agent — routes
# ─────────────────────────────────────────────────────────────────────────────

@app.route('/budget-agent')
@budget_agent_required
def budget_agent():
    user_id = session.get('user_id')
    conversations = (BudgetConversation.query
                     .filter_by(user_id=user_id)
                     .order_by(BudgetConversation.updated_at.desc())
                     .all())
    return render_template('budget_agent.html', conversations=conversations)


@app.route('/api/budget-db-views', methods=['GET'])
@admin_required
def api_budget_db_views():
    """Diagnostic: list all objects visible to powerbi in JKF-Public-Data."""
    try:
        conn = get_budget_connection()
        cursor = conn.cursor()

        # Which database did we actually land in?
        cursor.execute("SELECT DB_NAME() AS db, USER_NAME() AS usr, SCHEMA_NAME() AS sch")
        ctx = cursor.fetchone()

        # All views (INFORMATION_SCHEMA)
        cursor.execute("""
            SELECT TABLE_TYPE, TABLE_SCHEMA, TABLE_NAME
            FROM INFORMATION_SCHEMA.TABLES
            ORDER BY TABLE_TYPE, TABLE_SCHEMA, TABLE_NAME
        """)
        tables = cursor.fetchall()

        # sys.objects for everything including synonyms
        cursor.execute("""
            SELECT o.type_desc, s.name AS schema_name, o.name
            FROM sys.objects o
            JOIN sys.schemas s ON s.schema_id = o.schema_id
            ORDER BY o.type_desc, s.name, o.name
        """)
        sys_objs = cursor.fetchall()

        # Synonym definitions
        cursor.execute("""
            SELECT name, base_object_name
            FROM sys.synonyms
            ORDER BY name
        """)
        synonyms = cursor.fetchall()

        conn.close()
        return jsonify({
            'context': {'db': ctx[0], 'user': ctx[1], 'schema': ctx[2]},
            'information_schema': [{'type': r[0], 'schema': r[1], 'name': r[2]} for r in tables],
            'sys_objects': [{'type': r[0], 'schema': r[1], 'name': r[2]} for r in sys_objs],
            'synonyms': [{'name': r[0], 'points_to': r[1]} for r in synonyms],
        })
    except Exception as e:
        return jsonify({'error': str(e)}), 500


@app.route('/api/budget-conversations', methods=['GET'])
@budget_agent_required
def api_budget_conversations():
    user_id = session.get('user_id')
    convs = (BudgetConversation.query
             .filter_by(user_id=user_id)
             .order_by(BudgetConversation.updated_at.desc())
             .all())
    return jsonify([{
        'id': c.id,
        'title': c.title,
        'updated_at': c.updated_at.isoformat(),
    } for c in convs])


@app.route('/api/budget-conversations/<int:conv_id>', methods=['GET'])
@budget_agent_required
def api_budget_conversation_detail(conv_id):
    user_id = session.get('user_id')
    conv = BudgetConversation.query.filter_by(id=conv_id, user_id=user_id).first_or_404()
    return jsonify({
        'id': conv.id,
        'title': conv.title,
        'messages': [{
            'id': m.id,
            'role': m.role,
            'content': m.content,
            'sql': m.sql_query,
            'row_count': m.row_count,
        } for m in conv.messages],
    })


@app.route('/api/budget-conversations/<int:conv_id>', methods=['DELETE'])
@budget_agent_required
def api_budget_conversation_delete(conv_id):
    user_id = session.get('user_id')
    conv = BudgetConversation.query.filter_by(id=conv_id, user_id=user_id).first_or_404()
    db.session.delete(conv)
    db.session.commit()
    return jsonify({'ok': True})


@app.route('/api/budget-chat', methods=['POST'])
@budget_agent_required
def api_budget_chat():
    data = request.get_json(silent=True) or {}
    message = (data.get('message') or '').strip()
    history = data.get('history') or []
    conversation_id = data.get('conversation_id')

    if not message:
        return jsonify({'error': 'Ingen besked modtaget.'}), 400

    user_id = session.get('user_id')

    # Resolve or create BudgetConversation
    conv = None
    if conversation_id:
        conv = BudgetConversation.query.filter_by(id=conversation_id, user_id=user_id).first()
    if conv is None:
        title = message[:60] + ('…' if len(message) > 60 else '')
        conv = BudgetConversation(user_id=user_id, title=title)
        db.session.add(conv)
        db.session.flush()

    user_turn = BudgetMessage(
        conversation_id=conv.id,
        role='user',
        content=message,
    )
    db.session.add(user_turn)

    try:
        schema = get_budget_schema()
    except Exception as e:
        logger.error(f'Budget schema fetch failed: {e}')
        return jsonify({'error': 'Kunne ikke oprette forbindelse til budget-datalageret.'}), 500

    account_names = get_budget_account_names()
    account_names_section = (
        "PRÆCISE KONTONAVNE (Account Name) — brug disse EKSAKT ved filtrering, inkl. store/små bogstaver og bindestreger:\n"
        f"{account_names}\n\n"
        "Når brugeren nævner en konto (f.eks. 'IT konsulent' eller 'lønninger'), find det nærmeste navn på listen ovenfor "
        "og brug det præcist i WHERE-klausulen. Brug ALTID = eller IN med præcise navne — brug kun LIKE som absolut sidste udvej.\n\n"
    ) if account_names else ''

    sql_system = (
        "Du er en T-SQL ekspert for JKF's budget-datawarehouse (SQL Server, database: BC2SQL_Data).\n\n"
        "Du har adgang til præcis to views — brug navnene PRÆCIS som angivet (ingen schema-prefix):\n\n"
        "1. [vw_GL_actuals_vs_budget] — Budgetafvigelser pr. konto/afdeling/måned.\n"
        "   Kolonner: G_L Account No_ (kontonummer), Account Name (kontonavn), Year (int), Month (int), "
        "Global Dimension 1 Code (afdelingskode), Department Name (afdelingsnavn), "
        "Actual (aktuel beløb for måneden), Budget (budgetteret beløb), Variance (afvigelse = Actual - Budget).\n"
        "   Brug dette view til: afvigelsesanalyse, oversigt over over/underforbrug, budget vs. aktuel pr. konto eller afdeling.\n\n"
        "2. [vw_GL_entry_detailed] — Rå posteringer fra kontoplanen.\n"
        "   Kolonner: Account Name (kontonavn), Posting Date (bogføringsdato), Document No_ (bilagsnummer), "
        "Description (beskrivelse), Amount (beløb), Department Code (afdelingskode), "
        "Department Name (afdelingsnavn), Source No_ (leverandør-/kildenummer), Source Name (leverandørnavn).\n"
        "   Brug dette view til: forklaring af specifikke posteringer, leverandøroverblik, sammenligning med tidligere år, "
        "analyse af hvad der udgør en kontos forbrug.\n\n"
        "STRATEGI – vælg det rette view (eller begge):\n"
        "- 'Hvad er afvigelsen?' / 'Hvilke konti er over budget?' → brug [vw_GL_actuals_vs_budget]\n"
        "- 'Hvorfor er konto X over budget?' / 'Hvad er der posteret?' / 'Sammenlign med sidste år' → brug [vw_GL_entry_detailed]\n"
        "- Kombiner begge views via CTE eller JOIN, når du vil konstatere afvigelsen OG forklare den med posteringer.\n\n"
        "FULDT SKEMA:\n"
        f"{schema}\n\n"
        f"{account_names_section}"
        "OBLIGATORISKE REGLER:\n"
        "0. Generér ALTID et SELECT-statement. Du har fuld adgang til dataene.\n"
        "1. Returner KUN rå SQL, ingen forklaring, ingen markdown, ingen ```.\n"
        "2. Brug firkantede parenteser om view-navne og kolonner med mellemrum eller specialtegn.\n"
        "2b. Inkluder ALTID [G_L Account No_] i SELECT når [Account Name] er med — kontonummer skal altid fremgå ved siden af kontonavnet.\n"
        "2c. FILTRERING: brug ALTID WHERE til at filtrere rækker — brug ALDRIG COALESCE i GROUP BY som erstatning for filtrering. "
        "COALESCE i GROUP BY returnerer alle rækker og maskerer blot NULL-værdier, men filtrerer intet ud. "
        "Vil du kun have rækker uden afdeling: brug WHERE [Department Name] IS NULL. "
        "COALESCE må kun bruges i SELECT til at formatere output, f.eks. COALESCE([Department Name], 'Ingen afdeling') AS [Department Name].\n"
        "3. Subquery-aliaser UDEN AS: FROM (SELECT ...) sub — IKKE FROM (SELECT ...) AS sub.\n"
        "4. Kolonne-aliaser bruger AS normalt: SUM(Actual) AS AktuelTotal.\n"
        "5. Brug TOP n (ikke LIMIT) for at begrænse resultater.\n"
        "6. CTEs: WITH ctename AS (SELECT ...) SELECT ... — uden semikolon foran WITH.\n"
        "7. Brug aldrig: DROP, INSERT, UPDATE, DELETE, TRUNCATE, ALTER, CREATE, EXEC, QUALIFY.\n"
        "8. ORDER BY er IKKE tilladt inde i subqueries eller CTEs uden TOP.\n"
        "9. Du må KUN forespørge på de to views nævnt ovenfor — ingen andre tabeller eller views.\n"
        "10. Negative Variance-værdier betyder overforbrug (Actual > Budget); positive betyder underforbrug.\n"
        "11. Bevar årsfilter fra samtalehistorikken medmindre brugeren eksplicit angiver et andet år.\n"
        "12. VIGTIGT — kolonner til dato varierer mellem views:\n"
        "    [vw_GL_actuals_vs_budget]: har kolonnerne [Year] og [Month] direkte — brug dem i WHERE, GROUP BY og SELECT.\n"
        "    [vw_GL_entry_detailed]: har INGEN [Year] eller [Month] kolonne — brug KUN:\n"
        "      WHERE: YEAR([Posting Date]) IN (...) AND MONTH([Posting Date]) <= ...\n"
        "      SELECT: YEAR([Posting Date]) AS [Year], MONTH([Posting Date]) AS [Month]\n"
        "      GROUP BY: YEAR([Posting Date]), MONTH([Posting Date])\n"
        "    Brug ALDRIG [Year] eller [Month] direkte i [vw_GL_entry_detailed] — det giver fejl 207.\n"
        f"13. DATO-KONTEKST: I dag er {datetime.now().strftime('%Y-%m-%d')}. Indeværende år = {datetime.now().year}. Indeværende måned = {datetime.now().month}.\n"
        "    - Inkluder ALTID Year og Month i SELECT, så brugeren kan se hvilken periode tallene tilhører.\n"
        "    - Hent ALTID det efterspurgte år OG året før i samme forespørgsel, så svaret kan sammenligne med forrige år.\n"
        f"    - SAMME PERIODE-REGEL (kritisk): sammenlign kun de måneder der er gået i det nyeste år.\n"
        f"      Vi er i måned {datetime.now().month} ({datetime.now().year}), så filtrer begge år til Month <= {datetime.now().month} medmindre andet angives.\n"
        "      Eksempler:\n"
        f"      Ingen årsangivelse (standard) → WHERE Year IN ({datetime.now().year - 1}, {datetime.now().year}) AND Month <= {datetime.now().month}\n"
        f"      'I 2025' → WHERE Year IN (2024, 2025) AND Month <= 12  -- fuldt år, ingen månedsbegrænsning\n"
        "      'I 2023' → WHERE Year IN (2022, 2023) AND Month <= 12\n"
        f"      'Denne måned' → WHERE Year IN ({datetime.now().year - 1}, {datetime.now().year}) AND Month = {datetime.now().month}\n"
        "      'Hele 2025' / 'hele året' → Month <= 12 (ingen månedsbegrænsning)\n"
        "    - Undtagelse: hvis brugeren spørger om et historisk år (ikke indeværende), brug Month <= 12 (hele året).\n"
        "    - Undtagelse: hvis brugeren EKSPLICIT siger 'kun i år' eller 'kun [årstal]' uden sammenligning, hent kun det ene år.\n"
        "    - Sig ALDRIG 'ingen data for forrige år' — forrige år er altid inkluderet i forespørgslen.\n"
    )

    sql_messages = [{'role': 'system', 'content': sql_system}]
    for h in history[-10:]:
        if h.get('role') in ('user', 'assistant') and h.get('content'):
            sql_messages.append({'role': h['role'], 'content': h['content']})
    sql_messages.append({'role': 'user', 'content': message})

    try:
        sql_resp = client.chat.completions.create(
            model='gpt-5.4-mini',
            messages=sql_messages,
            temperature=0,
            max_completion_tokens=1500,
        )
        raw_sql = _extract_sql(sql_resp.choices[0].message.content or '')
    except Exception as e:
        logger.error(f'Budget SQL generation failed: {e}')
        return jsonify({'error': 'Kunne ikke generere SQL-forespørgsel.'}), 500

    # Handle prose/non-SQL response
    _SQL_START = re.compile(r'^\s*(SELECT|WITH|;WITH)\b', re.IGNORECASE)
    if not _SQL_START.match(raw_sql):
        try:
            cleanup_resp = client.chat.completions.create(
                model='gpt-5.4-mini',
                messages=[
                    {'role': 'system', 'content': (
                        'Du er en hjælpsom budget-assistent hos JKF. '
                        'Omskriv følgende besked til et pænt, kort dansk svar. '
                        'Forklar venligt at du ikke har nok information til at svare præcist, '
                        'og opfordr brugeren til at stille spørgsmålet på ny.'
                    )},
                    {'role': 'user', 'content': raw_sql},
                ],
                max_completion_tokens=300,
                temperature=0.3,
            )
            clean_answer = cleanup_resp.choices[0].message.content or raw_sql
        except Exception:
            clean_answer = 'Jeg har ikke nok information til at svare. Prøv at stille spørgsmålet på ny.'
        return jsonify({'answer': clean_answer, 'sql': None, 'row_count': None})

    if _DW_DANGEROUS.search(raw_sql):
        return jsonify({'error': 'Sikkerhedsfejl: kun læse-forespørgsler er tilladt.'}), 400

    if _DW_UNSUPPORTED.search(raw_sql):
        return jsonify({'error': 'Den genererede SQL bruger QUALIFY, som ikke understøttes. Prøv at omformulere spørgsmålet.'}), 400

    budget_ok, bad_ref = _validate_budget_views(raw_sql)
    if not budget_ok:
        logger.warning(f'Budget SQL validation blocked reference: {bad_ref}')
        return jsonify({'error': f'Adgangsfejl: forespørgslen forsøgte at tilgå "{bad_ref}", som ikke er tilladt.'}), 400

    logger.info(f'Budget SQL generated:\n{raw_sql}')
    try:
        conn = get_budget_connection()
        cursor = conn.cursor(as_dict=True)
        cursor.execute(raw_sql)
        rows = cursor.fetchmany(500)
        conn.close()
        row_count = len(rows)
        if rows:
            headers = list(rows[0].keys())
            lines = ['\t'.join(headers)]
            for r in rows:
                lines.append('\t'.join(str(r[h]) for h in headers))
            result_text = '\n'.join(lines)
        else:
            result_text = '(ingen rækker)'
    except Exception as e:
        err_str = str(e)
        logger.error(f'Budget query failed: {e}\nSQL was:\n{raw_sql}')
        if '1033' in err_str:
            return jsonify({'error': 'Den genererede SQL indeholder ORDER BY i en subquery uden TOP. Prøv at omformulere spørgsmålet.'}), 500
        return jsonify({'error': 'Databasefejl: forespørgslen kunne ikke udføres. Prøv at omformulere dit spørgsmål.'}), 500

    answer_system = (
        "Du er en hjælpsom budget-analytiker hos JKF. Svar altid på dansk.\n\n"
        "Din opgave er at forklare budgetafvigelser klart og præcist — angiv ALTID:\n"
        "1. Konstateringen: hvad er afvigelsen (beløb og procent af budget)?\n"
        "2. Forklaringen: hvad udgør forbruget (posteringer, leverandører, perioder)?\n"
        "3. Evt. sammenligning: er dette anderledes end samme periode sidste år?\n\n"
        "KONTONAVNE OG KONTONUMRE:\n"
        "- Når du omtaler en konto, skriv ALTID kontonummer og navn sammen, f.eks.: **IT-konsulent (6320)**.\n"
        "- I tabeller: inkluder kolonnen Kontonummer når(G_L Account No_) anvendes sammen med Kontonavn.\n"
        "- Nævn aldrig et kontonavn uden kontonummeret i parentes.\n\n"
        "FORMATERINGSREGLER:\n"
        "- Brug markdown-tabel (|) til tabeldata med flere rækker.\n"
        "- TABELRETNING – vælg den korteste og bredeste form:\n"
        "  * HORISONTAL (foretrukket): én række per enhed (konto, afdeling), "
        "perioder/metrics som kolonner. Fx:\n"
        "    | Konto                | Budget | Actual | Afvigelse |\n"
        "    |----------------------|--------|--------|-----------|\n"
        "    | IT-konsulent (6320)  | 50.000 | 62.000 | -12.000   |\n"
        "  * VERTIKAL: kun ved simple lister med én metric.\n"
        "- Aldrig én-kolonne tabeller med 'År' og én værdi – brug i stedet år som kolonneoverskrifter.\n"
        "- Brug punktopstilling til lister og opsummeringer.\n"
        "- Brug **fed** til vigtige tal og nøgleord.\n"
        "- TALFORMAT (kritisk): dansk tusindtalsseparator (.), komma som decimaltegn.\n"
        "  Eksempel: 53.980 DKK, -12,3%.\n"
        "  Beløb: ingen decimaler. Procenter: 1 decimal.\n"
        "  ALDRIG afkort tal — 53980 skrives som 53.980, ikke 53,98.\n"
        "- Negative afvigelser = overforbrug (fremhæv med ⚠️ eller **fed**).\n"
        "- Positive afvigelser = underforbrug.\n"
        "- Afslut med en kort konklusion eller et opfølgningsforslag."
    )

    answer_messages = [
        {'role': 'system', 'content': answer_system},
        {
            'role': 'user',
            'content': (
                f"Brugerens spørgsmål: {message}\n\n"
                f"SQL der blev kørt:\n{raw_sql}\n\n"
                f"Resultat ({row_count} rækker):\n{result_text}"
            ),
        },
    ]

    try:
        ans_resp = client.chat.completions.create(
            model='gpt-5.4-mini',
            messages=answer_messages,
            temperature=0.3,
            max_completion_tokens=6000,
        )
        answer = ans_resp.choices[0].message.content or ''
    except Exception as e:
        logger.error(f'Budget answer generation failed: {e}')
        return jsonify({'error': 'Kunne ikke formulere svar.'}), 500

    assistant_turn = BudgetMessage(
        conversation_id=conv.id,
        role='assistant',
        content=answer,
        sql_query=raw_sql,
        row_count=row_count,
    )
    db.session.add(assistant_turn)
    conv.updated_at = datetime.utcnow()
    db.session.commit()

    return jsonify({
        'answer': answer,
        'sql': raw_sql,
        'row_count': row_count,
        'conversation_id': conv.id,
        'msg_id': assistant_turn.id,
    })


@app.route('/api/budget-conversations/<int:conv_id>/export/pdf')
@budget_agent_required
def api_budget_conversation_export_pdf(conv_id):
    import re as _re
    user_id = session.get('user_id')
    conv = BudgetConversation.query.filter_by(id=conv_id, user_id=user_id).first_or_404()

    raw_ids   = request.args.get('msg_ids', '')
    incl_user = request.args.get('include_user', '1') == '1'
    selected_ids = set(int(x) for x in raw_ids.split(',') if x.strip().isdigit()) if raw_ids else None

    def md_to_html(text):
        text = _re.sub(r'\*\*(.+?)\*\*', r'<strong>\1</strong>', text)
        text = _re.sub(r'\*(.+?)\*',     r'<em>\1</em>',         text)
        lines = text.split('\n')
        out, in_table, in_list = [], False, False
        for line in lines:
            if line.startswith('|'):
                cells = [c.strip() for c in line.strip('|').split('|')]
                if all(set(c) <= set('-:| ') for c in cells):
                    continue
                if not in_table:
                    if in_list: out.append('</ul>'); in_list = False
                    out.append('<div class="tbl-wrap"><table class="md-table"><thead><tr>')
                    out.extend(f'<th>{c}</th>' for c in cells)
                    out.append('</tr></thead><tbody>')
                    in_table = True
                else:
                    out.append('<tr>')
                    out.extend(f'<td>{c}</td>' for c in cells)
                    out.append('</tr>')
            else:
                if in_table: out.append('</tbody></table></div>'); in_table = False
                stripped = line.strip()
                if stripped.startswith('- ') or stripped.startswith('* '):
                    if not in_list: out.append('<ul>'); in_list = True
                    out.append(f'<li>{stripped[2:]}</li>')
                elif stripped.startswith('### '):
                    if in_list: out.append('</ul>'); in_list = False
                    out.append(f'<h3>{stripped[4:]}</h3>')
                elif stripped.startswith('## '):
                    if in_list: out.append('</ul>'); in_list = False
                    out.append(f'<h2>{stripped[3:]}</h2>')
                elif stripped == '':
                    if in_list: out.append('</ul>'); in_list = False
                    out.append('<div class="spacer"></div>')
                else:
                    if in_list: out.append('</ul>'); in_list = False
                    out.append(f'<p>{line}</p>')
        if in_table: out.append('</tbody></table></div>')
        if in_list:  out.append('</ul>')
        return '\n'.join(out)

    messages_html = ''
    turn_num = 0
    all_messages = list(conv.messages)
    for i, msg in enumerate(all_messages):
        if msg.role == 'user':
            if not incl_user:
                continue
            next_asst = next((m for m in all_messages[i+1:] if m.role == 'assistant'), None)
            if selected_ids is not None and (next_asst is None or next_asst.id not in selected_ids):
                continue
            messages_html += f'''
<div class="turn user-turn">
  <div class="turn-label"><span class="label-chip user-chip">Spørgsmål</span></div>
  <div class="bubble user-bubble"><p>{msg.content}</p></div>
</div>'''
        else:
            if selected_ids is not None and msg.id not in selected_ids:
                continue
            turn_num += 1
            content_html = md_to_html(msg.content or '')
            messages_html += f'''
<div class="turn assistant-turn">
  <div class="turn-label"><span class="label-chip asst-chip">Svar {turn_num}</span></div>
  <div class="bubble asst-bubble">{content_html}</div>
</div>'''

    _da_months = ['januar','februar','marts','april','maj','juni',
                  'juli','august','september','oktober','november','december']
    _now = datetime.utcnow().replace(tzinfo=pytz.utc).astimezone(danish_tz)
    exported_at = f"{_now.day}. {_da_months[_now.month - 1]} {_now.year} kl. {_now.strftime('%H:%M')}"

    html = _pdf_html(conv.title, 'Budget Agent', exported_at, incl_user, messages_html)
    return html, 200, {'Content-Type': 'text/html; charset=utf-8', 'Content-Disposition': 'inline'}


@app.route('/api/master-conversations/<int:conv_id>/export/pdf')
@master_agent_required
def api_master_conversation_export_pdf(conv_id):
    import re as _re
    user_id = session.get('user_id')
    conv = MasterConversation.query.filter_by(id=conv_id, user_id=user_id).first_or_404()

    raw_ids   = request.args.get('msg_ids', '')
    incl_user = request.args.get('include_user', '1') == '1'
    selected_ids = set(int(x) for x in raw_ids.split(',') if x.strip().isdigit()) if raw_ids else None

    def md_to_html(text):
        import re as _re2
        text = _re2.sub(r'\*\*(.+?)\*\*', r'<strong>\1</strong>', text)
        text = _re2.sub(r'\*(.+?)\*',     r'<em>\1</em>',         text)
        lines = text.split('\n')
        out, in_table, in_list = [], False, False
        for line in lines:
            if line.startswith('|'):
                cells = [c.strip() for c in line.strip('|').split('|')]
                if all(set(c) <= set('-:| ') for c in cells):
                    continue
                if not in_table:
                    if in_list: out.append('</ul>'); in_list = False
                    out.append('<div class="tbl-wrap"><table class="md-table"><thead><tr>')
                    out.extend(f'<th>{c}</th>' for c in cells)
                    out.append('</tr></thead><tbody>')
                    in_table = True
                else:
                    out.append('<tr>')
                    out.extend(f'<td>{c}</td>' for c in cells)
                    out.append('</tr>')
            else:
                if in_table: out.append('</tbody></table></div>'); in_table = False
                stripped = line.strip()
                if stripped.startswith('- ') or stripped.startswith('* '):
                    if not in_list: out.append('<ul>'); in_list = True
                    out.append(f'<li>{stripped[2:]}</li>')
                elif stripped.startswith('### '):
                    if in_list: out.append('</ul>'); in_list = False
                    out.append(f'<h3>{stripped[4:]}</h3>')
                elif stripped.startswith('## '):
                    if in_list: out.append('</ul>'); in_list = False
                    out.append(f'<h2>{stripped[3:]}</h2>')
                elif stripped == '':
                    if in_list: out.append('</ul>'); in_list = False
                    out.append('<div class="spacer"></div>')
                else:
                    if in_list: out.append('</ul>'); in_list = False
                    out.append(f'<p>{line}</p>')
        if in_table: out.append('</tbody></table></div>')
        if in_list:  out.append('</ul>')
        return '\n'.join(out)

    messages_html = ''
    turn_num = 0
    all_messages = list(conv.messages)
    for i, msg in enumerate(all_messages):
        if msg.role == 'user':
            if not incl_user:
                continue
            next_asst = next((m for m in all_messages[i+1:] if m.role == 'assistant'), None)
            if selected_ids is not None and (next_asst is None or next_asst.id not in selected_ids):
                continue
            messages_html += f'''
<div class="turn user-turn">
  <div class="turn-label"><span class="label-chip user-chip">Spørgsmål</span></div>
  <div class="bubble user-bubble"><p>{msg.content}</p></div>
</div>'''
        else:
            if selected_ids is not None and msg.id not in selected_ids:
                continue
            turn_num += 1
            content_html = md_to_html(msg.content or '')
            messages_html += f'''
<div class="turn assistant-turn">
  <div class="turn-label"><span class="label-chip asst-chip">Svar {turn_num}</span></div>
  <div class="bubble asst-bubble">{content_html}</div>
</div>'''

    _da_months = ['januar','februar','marts','april','maj','juni',
                  'juli','august','september','oktober','november','december']
    _now = datetime.utcnow().replace(tzinfo=pytz.utc).astimezone(danish_tz)
    exported_at = f"{_now.day}. {_da_months[_now.month - 1]} {_now.year} kl. {_now.strftime('%H:%M')}"

    html = _pdf_html(conv.title, 'Master Agent', exported_at, incl_user, messages_html)
    return html, 200, {'Content-Type': 'text/html; charset=utf-8', 'Content-Disposition': 'inline'}


# ─────────────────────────────────────────────────────────────────────────────
# Master Agent
# ─────────────────────────────────────────────────────────────────────────────

MASTER_AGENT_SYSTEM_PROMPT = (
    "Du er JKF's interne Master Agent – et samlet AI-system med adgang til alle JKF's datakilder.\n\n"
    "Du har fire værktøjer:\n"
    "- search_knowledge_base: Søg i videnbase, uploadede dokumenter og hjemmesideindhold\n"
    "- query_sales_data: Hent salgsdata (omsætning, kunder, varer, marginer) fra data warehouse\n"
    "- query_budget_data: Hent budgetdata (budget vs. forbrug, afvigelser, GL-konti) fra ERP\n"
    "- search_bc_products: Søg produkter og lagerstatus i Business Central\n\n"
    "Regler:\n"
    "1. Kald ALTID det relevante værktøj – gæt aldrig på svar uden at hente data\n"
    "2. Du kan kalde flere værktøjer i samme svar, hvis spørgsmålet kræver data fra flere kilder\n"
    "3. Svar på det sprog brugeren skriver på (dansk, engelsk, tysk osv.)\n"
    "4. Brug markdown-tabeller til tabeldata og **fed** til vigtige tal\n"
    "5. Tal formateres med dansk konvention: tusindtalsseparator (.), decimal med komma (,)\n"
    "6. Vær konkret og præcis – brug tal direkte fra de hentede data"
)

MASTER_TOOLS = [
    {
        "type": "function",
        "function": {
            "name": "search_knowledge_base",
            "description": (
                "Søg i JKF's videnbase: Q&A-artikler, uploadede dokumenter og hjemmesideindhold. "
                "Brug dette til generelle spørgsmål om JKF, produkter, processer, politikker, "
                "leveringsbetingelser, returret og alt andet der ikke er salgs- eller budgetdata."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "query": {"type": "string", "description": "Søgeforespørgsel på naturligt sprog"},
                },
                "required": ["query"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "query_sales_data",
            "description": (
                "Forespørg salgsdata fra JKF's data warehouse. "
                "Brug dette til spørgsmål om omsætning, salg pr. kunde/vare/land/periode, "
                "salgstendenser, marginer, top-kunder og top-varer."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "question": {
                        "type": "string",
                        "description": "Spørgsmål om salgsdata på naturligt sprog",
                    },
                },
                "required": ["question"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "query_budget_data",
            "description": (
                "Forespørg budgetdata og regnskabsoplysninger fra JKF's ERP-system. "
                "Brug dette til spørgsmål om budget vs. forbrug, afvigelser, GL-konti, "
                "afdelingsudgifter og posteringer."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "question": {
                        "type": "string",
                        "description": "Spørgsmål om budget- og regnskabsdata på naturligt sprog",
                    },
                },
                "required": ["question"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "search_bc_products",
            "description": (
                "Søg produkter og lagerstatus i Business Central. "
                "Brug dette til spørgsmål om varenumre, lagerbeholdning, priser og leveringstider."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "description": {
                        "type": "string",
                        "description": "Produktbeskrivelse eller varenummer at søge efter",
                    },
                },
                "required": ["description"],
            },
        },
    },
]


def _master_search_kb(query: str, history: list) -> str:
    prior_user_turns = [m['content'] for m in history if m.get('role') == 'user'][-5:]
    ctx = get_context_from_qdrant(query, top_k=7, prior_user_turns=prior_user_turns)
    return ctx if ctx else "Ingen relevant information fundet i videnbasen."


def _master_query_sales(question: str, history: list) -> str:
    try:
        full_schema = get_dw_schema()
    except Exception as e:
        return f"Fejl: kunne ikke oprette forbindelse til salgs-datalageret ({e})."

    sales_lines = [l for l in full_schema.split('\n')
                   if l.split('(')[0].strip().lower() in _ALLOWED_SALES_TABLES]
    schema = '\n'.join(sales_lines) if sales_lines else full_schema

    sql_system = (
        "Du er en T-SQL ekspert for JKF's SQL Server datawarehouse (SQL Server 2019).\n\n"
        "VIGTIGE TABELLER:\n"
        "- [Salg]: Primær salgsoversigt – Varenummer, Beskrivelse, Varegruppe, Dato (date), KundeNavn, Land, Salgsbeløb (decimal), Rabatbeløb, dækningsbidrag. "
        "Brug kolonnen [Varegruppe] direkte fra [Salg] til alle spørgsmål om varegrupper – join ALDRIG [Item] for dette formål.\n"
        "- [Sales Invoice Header]: Fakturahoveder – [Posting Date], [Sell-to Customer No_], [Bill-to Name], CompanyCode.\n"
        "- [Sales Invoice Line]: Fakturalinjer – Amount, Quantity, [No_] (varenummer), CompanyCode.\n"
        "- [Customer]: Kunder – [No_], Name, [Country_Region Code], [Salesperson Code].\n"
        "- [Item]: Varer – [No_], Description.\n\n"
        "ADGANGSBEGRÆNSNING: Brug KUN ovenstående tabeller.\n\n"
        "FULDT SKEMA:\n"
        f"{schema}\n\n"
        "OBLIGATORISKE REGLER:\n"
        "0. Generér ALTID et SELECT-statement.\n"
        "1. Returner KUN rå SQL, ingen forklaring, ingen markdown, ingen ```.\n"
        "2. Brug firkantede parenteser om tabelnavne og kolonner med mellemrum.\n"
        "3. Subquery-aliaser UDEN AS: FROM (SELECT ...) sub.\n"
        "4. Brug TOP n (ikke LIMIT).\n"
        "5. Brug aldrig DROP, INSERT, UPDATE, DELETE, TRUNCATE, ALTER, CREATE, EXEC, QUALIFY.\n"
        "6. YEAR(Dato) og MONTH(Dato) til dato-filtrering på [Salg].\n"
        "7. Ingen ORDER BY i subqueries uden TOP.\n"
        "8. HAVING SUM(Salgsbeløb) > 0 ved min/max af beregnet værdi.\n"
    )

    sql_messages = [{'role': 'system', 'content': sql_system}]
    for h in history[-10:]:
        if h.get('role') in ('user', 'assistant') and h.get('content'):
            sql_messages.append({'role': h['role'], 'content': h['content']})
    sql_messages.append({'role': 'user', 'content': question})

    try:
        sql_resp = client.chat.completions.create(
            model='gpt-5.4-mini', messages=sql_messages, temperature=0, max_completion_tokens=3000,
        )
        raw_sql = _extract_sql(sql_resp.choices[0].message.content or '')
    except Exception as e:
        return f"SQL-generering fejlede: {e}"

    _SQL_START_RE = re.compile(r'^\s*(SELECT|WITH|;WITH)\b', re.IGNORECASE)
    if not _SQL_START_RE.match(raw_sql):
        return raw_sql

    if _DW_DANGEROUS.search(raw_sql) or _DW_UNSUPPORTED.search(raw_sql):
        return "Sikkerhedsfejl: forespørgslen er ikke tilladt."

    salg_ok, bad_ref = _validate_sales_tables(raw_sql)
    if not salg_ok:
        return f"Adgangsfejl: forespørgslen forsøgte at tilgå '{bad_ref}'."

    try:
        conn = get_dw_connection()
        cursor = conn.cursor(as_dict=True)
        cursor.execute(raw_sql)
        rows = cursor.fetchmany(500)
        conn.close()
        row_count = len(rows)
        if rows:
            headers = list(rows[0].keys())
            lines = ['\t'.join(headers)]
            for r in rows:
                lines.append('\t'.join(str(r[h]) for h in headers))
            result_text = '\n'.join(lines)
        else:
            result_text = '(ingen rækker)'
    except Exception as e:
        logger.error(f'Master sales query failed: {e}\nSQL: {raw_sql}')
        return "Databasefejl: forespørgslen kunne ikke udføres. Prøv at omformulere dit spørgsmål."

    return f"SQL kørt:\n{raw_sql}\n\nResultat ({row_count} rækker):\n{result_text}"


def _master_query_budget(question: str, history: list) -> str:
    try:
        schema = get_budget_schema()
    except Exception as e:
        return f"Fejl: kunne ikke oprette forbindelse til budget-datalageret ({e})."

    account_names = get_budget_account_names()
    account_names_section = (
        "PRÆCISE KONTONAVNE — brug disse eksakt:\n"
        f"{account_names}\n\n"
    ) if account_names else ''

    sql_system = (
        "Du er en T-SQL ekspert for JKF's budget-datawarehouse (SQL Server, database: BC2SQL_Data).\n\n"
        "Du har adgang til to views:\n"
        "1. [vw_GL_actuals_vs_budget] — Budgetafvigelser: G_L Account No_, Account Name, Year, Month, "
        "Global Dimension 1 Code, Department Name, Actual, Budget, Variance.\n"
        "2. [vw_GL_entry_detailed] — Rå posteringer: Account Name, Posting Date, Document No_, "
        "Description, Amount, Department Code, Department Name, Source No_, Source Name.\n\n"
        "FULDT SKEMA:\n"
        f"{schema}\n\n"
        f"{account_names_section}"
        "OBLIGATORISKE REGLER:\n"
        "0. Generér ALTID et SELECT-statement.\n"
        "1. Returner KUN rå SQL, ingen forklaring, ingen markdown, ingen ```.\n"
        "2. Brug firkantede parenteser.\n"
        "3. Subquery-aliaser UDEN AS.\n"
        "4. Brug TOP n (ikke LIMIT).\n"
        "5. Brug aldrig DROP, INSERT, UPDATE, DELETE, TRUNCATE, ALTER, CREATE, EXEC, QUALIFY.\n"
        "6. Ingen ORDER BY i subqueries uden TOP.\n"
        "7. [vw_GL_actuals_vs_budget]: brug [Year] og [Month] direkte.\n"
        "8. [vw_GL_entry_detailed]: brug YEAR([Posting Date]) og MONTH([Posting Date]) – ALDRIG [Year]/[Month].\n"
        f"9. DATO-KONTEKST: I dag er {datetime.now().strftime('%Y-%m-%d')}. "
        f"Indeværende år = {datetime.now().year}, måned = {datetime.now().month}.\n"
        "10. Inkluder ALTID begge år (indeværende + forrige) og begræns til Month <= indeværende måned.\n"
        "11. Negative Variance = overforbrug. Positive = underforbrug.\n"
    )

    sql_messages = [{'role': 'system', 'content': sql_system}]
    for h in history[-10:]:
        if h.get('role') in ('user', 'assistant') and h.get('content'):
            sql_messages.append({'role': h['role'], 'content': h['content']})
    sql_messages.append({'role': 'user', 'content': question})

    try:
        sql_resp = client.chat.completions.create(
            model='gpt-5.4-mini', messages=sql_messages, temperature=0, max_completion_tokens=3000,
        )
        raw_sql = _extract_sql(sql_resp.choices[0].message.content or '')
    except Exception as e:
        return f"SQL-generering fejlede: {e}"

    _SQL_START_RE = re.compile(r'^\s*(SELECT|WITH|;WITH)\b', re.IGNORECASE)
    if not _SQL_START_RE.match(raw_sql):
        return raw_sql

    if _DW_DANGEROUS.search(raw_sql) or _DW_UNSUPPORTED.search(raw_sql):
        return "Sikkerhedsfejl: forespørgslen er ikke tilladt."

    budget_ok, bad_ref = _validate_budget_views(raw_sql)
    if not budget_ok:
        return f"Adgangsfejl: forespørgslen forsøgte at tilgå '{bad_ref}'."

    try:
        conn = get_budget_connection()
        cursor = conn.cursor(as_dict=True)
        cursor.execute(raw_sql)
        rows = cursor.fetchmany(500)
        conn.close()
        row_count = len(rows)
        if rows:
            headers = list(rows[0].keys())
            lines = ['\t'.join(headers)]
            for r in rows:
                lines.append('\t'.join(str(r[h]) for h in headers))
            result_text = '\n'.join(lines)
        else:
            result_text = '(ingen rækker)'
    except Exception as e:
        logger.error(f'Master budget query failed: {e}\nSQL: {raw_sql}')
        return "Databasefejl: forespørgslen kunne ikke udføres. Prøv at omformulere dit spørgsmål."

    return f"SQL kørt:\n{raw_sql}\n\nResultat ({row_count} rækker):\n{result_text}"


def _master_search_products(description: str) -> str:
    matches = search_bc_items(description)
    if not matches:
        return "Ingen varer fundet med den beskrivelse. Prøv med andre søgeord."
    lines = ["Fundne varer:"]
    for m in matches:
        entry = f"- {m.item_no}: {m.description}"
        if m.description2:
            entry += f" / {m.description2}"
        entry += f" (lager: {m.inventory})"
        lines.append(entry)
    return '\n'.join(lines)


def run_master_agent(messages: list) -> tuple:
    """Tool-calling orchestration loop. Returns (answer_text, tools_used_list)."""
    full_messages = [{"role": "system", "content": MASTER_AGENT_SYSTEM_PROMPT}] + messages
    tools_used = []

    for _ in range(4):
        try:
            resp = client.chat.completions.create(
                model='gpt-5.4-mini',
                messages=full_messages,
                tools=MASTER_TOOLS,
                tool_choice="auto",
                temperature=0.3,
                max_completion_tokens=4000,
            )
        except Exception as e:
            logger.error(f"Master agent LLM call failed: {e}")
            return "Beklager, der opstod en fejl. Prøv igen.", tools_used

        msg = resp.choices[0].message

        if not msg.tool_calls:
            return msg.content or "Ingen svar genereret.", list(dict.fromkeys(tools_used))

        full_messages.append({
            "role": "assistant",
            "content": msg.content,
            "tool_calls": [
                {
                    "id": tc.id,
                    "type": "function",
                    "function": {"name": tc.function.name, "arguments": tc.function.arguments},
                }
                for tc in msg.tool_calls
            ],
        })

        sub_history = [m for m in messages if m.get('role') in ('user', 'assistant')]

        for tc in msg.tool_calls:
            tool_name = tc.function.name
            try:
                args = json.loads(tc.function.arguments)
            except Exception:
                args = {}
            tools_used.append(tool_name)

            if tool_name == "search_knowledge_base":
                result = _master_search_kb(args.get("query", ""), sub_history)
            elif tool_name == "query_sales_data":
                result = _master_query_sales(args.get("question", ""), sub_history)
            elif tool_name == "query_budget_data":
                result = _master_query_budget(args.get("question", ""), sub_history)
            elif tool_name == "search_bc_products":
                result = _master_search_products(args.get("description", ""))
            else:
                result = f"Ukendt værktøj: {tool_name}"

            full_messages.append({
                "role": "tool",
                "tool_call_id": tc.id,
                "content": str(result),
            })

    try:
        final_resp = client.chat.completions.create(
            model='gpt-5.4-mini',
            messages=full_messages,
            temperature=0.3,
            max_completion_tokens=4000,
        )
        return final_resp.choices[0].message.content or "Ingen svar genereret.", list(dict.fromkeys(tools_used))
    except Exception as e:
        logger.error(f"Master agent final call failed: {e}")
        return "Beklager, der opstod en fejl. Prøv igen.", list(dict.fromkeys(tools_used))


@app.route('/master-agent')
@master_agent_required
def master_agent():
    user_id = session.get('user_id')
    conversations = (MasterConversation.query
                     .filter_by(user_id=user_id)
                     .order_by(MasterConversation.updated_at.desc())
                     .all())
    return render_template('master_agent.html', conversations=conversations)


@app.route('/api/master-conversations', methods=['GET'])
@master_agent_required
def api_master_conversations():
    user_id = session.get('user_id')
    convs = (MasterConversation.query
             .filter_by(user_id=user_id)
             .order_by(MasterConversation.updated_at.desc())
             .all())
    return jsonify([{
        'id': c.id,
        'title': c.title,
        'updated_at': c.updated_at.isoformat(),
    } for c in convs])


@app.route('/api/master-conversations/<int:conv_id>', methods=['GET'])
@master_agent_required
def api_master_conversation_detail(conv_id):
    user_id = session.get('user_id')
    conv = MasterConversation.query.filter_by(id=conv_id, user_id=user_id).first_or_404()
    return jsonify({
        'id': conv.id,
        'title': conv.title,
        'messages': [{
            'id': m.id,
            'role': m.role,
            'content': m.content,
            'tools_used': json.loads(m.tools_used or '[]'),
        } for m in conv.messages],
    })


@app.route('/api/master-conversations/<int:conv_id>', methods=['DELETE'])
@master_agent_required
def api_master_conversation_delete(conv_id):
    user_id = session.get('user_id')
    conv = MasterConversation.query.filter_by(id=conv_id, user_id=user_id).first_or_404()
    db.session.delete(conv)
    db.session.commit()
    return jsonify({'ok': True})


@app.route('/api/master-chat', methods=['POST'])
@master_agent_required
def api_master_chat():
    data = request.get_json(silent=True) or {}
    message = (data.get('message') or '').strip()
    conversation_id = data.get('conversation_id')

    if not message:
        return jsonify({'error': 'Ingen besked modtaget.'}), 400

    user_id = session.get('user_id')

    # Phase 1: read DB state, create conversation if new, then commit immediately
    # so the session holds no pending state during the long LLM operation below.
    conv = None
    if conversation_id:
        conv = MasterConversation.query.filter_by(id=conversation_id, user_id=user_id).first()
    if conv is None:
        title = message[:60] + ('…' if len(message) > 60 else '')
        conv = MasterConversation(user_id=user_id, title=title)
        db.session.add(conv)
        db.session.flush()

    existing = (MasterMessage.query.filter_by(conversation_id=conv.id)
                .order_by(MasterMessage.id).all())
    agent_messages = [
        {'role': m.role, 'content': m.content}
        for m in existing
        if m.role in ('user', 'assistant')
    ]
    agent_messages.append({'role': 'user', 'content': message})
    conv_id = conv.id

    # Commit early — closes the transaction before the ~15-30 s agent run so
    # concurrent requests are not blocked by a held SQLite write lock.
    db.session.commit()

    # Phase 2: run the agent (no session open during this long operation)
    answer, tools_used = run_master_agent(agent_messages)

    # Phase 3: persist results in a fresh, short-lived transaction
    conv = MasterConversation.query.get(conv_id)
    db.session.add(MasterMessage(conversation_id=conv_id, role='user', content=message))
    master_asst_turn = MasterMessage(
        conversation_id=conv_id,
        role='assistant',
        content=answer,
        tools_used=json.dumps(tools_used),
    )
    db.session.add(master_asst_turn)
    conv.updated_at = datetime.utcnow()
    db.session.commit()

    return jsonify({
        'answer': answer,
        'tools_used': tools_used,
        'conversation_id': conv_id,
        'msg_id': master_asst_turn.id,
    })


# ─────────────────────────────────────────────────────────────────────────────
# Entry point
# ─────────────────────────────────────────────────────────────────────────────
if __name__ == '__main__':
    import sys
    init_db()
    dev_mode = '--dev' in sys.argv or os.environ.get('FLASK_DEV') == '1'
    if dev_mode:
        # Development: Flask's built-in server with auto-reload on file changes.
        # Note: avoid uploading large files in this mode — the dev server is
        # single-threaded and may drop connections under heavy concurrent load.
        logger.info('Starting dev server with auto-reload on http://localhost:5001')
        app.run(host='0.0.0.0', port=5001, debug=True, use_reloader=True)
    else:
        # Production: Waitress handles concurrent requests reliably.
        from waitress import serve
        logger.info('Starting server on http://localhost:5001')
        serve(app, host='0.0.0.0', port=5001, threads=8)
