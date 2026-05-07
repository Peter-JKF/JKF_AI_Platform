# Server Migration Guide

## Prerequisites

- Ubuntu 22.04+ (or Debian equivalent)
- Python 3.11+
- nginx
- Access to port 443 / a domain pointed at the server (e.g. `chatbot.jkf.dk`)

---

## 1. Upload the project

```bash
# From your local machine
scp -r "JKF chatbot og dashboard platform" user@your-server:/opt/jkf-platform
```

Or clone from git if you have a repo. The key files that must be present:

```
app.py  config.py  extensions.py  requirements.txt  templates/  static/  uploads/  logs/
```

> **Do not** upload your `.env` — you will create a fresh one on the server (step 3).

---

## 2. Set up the Python environment

```bash
cd /opt/jkf-platform
python3 -m venv .venv
source .venv/bin/activate
pip install --upgrade pip
pip install -r requirements.txt
```

---

## 3. Create the production `.env`

```bash
nano /opt/jkf-platform/.env
```

Paste and fill in all values:

```dotenv
# OpenAI
OPENAI_API_KEY=sk-proj-...

# Jina Reranker
JINA_API_KEY=jina_...

# Qdrant Cloud
QDRANT_URL=https://<your-cluster>.europe-west3-0.gcp.cloud.qdrant.io
QDRANT_API_KEY=<your-qdrant-key>

# Flask
FLASK_SECRET_KEY=<generate with: python3 -c "import secrets; print(secrets.token_hex(32))">
SQLALCHEMY_SILENCE_UBER_WARNING=1

# Database (SQLite default — switch to PostgreSQL for production)
# DATABASE_URL=postgresql://user:password@localhost/jkf_db

# Datawarehouse (SQL Server — ensure server firewall allows this server's IP)
DW_HOST=10.45.10.221
DW_PORT=1433
DW_DATABASE=BC2SQL_Data
DW_USERNAME=powerbi
DW_PASSWORD=<password>
```

Lock down the file:

```bash
chmod 600 /opt/jkf-platform/.env
```

---

## 4. Initialize the database

```bash
cd /opt/jkf-platform
source .venv/bin/activate
python - <<'EOF'
from app import app, init_db
with app.app_context():
    init_db()
EOF
```

If migrating an existing `jkf.db` from your Mac, copy it to the server before running this step:

```bash
scp jkf.db user@your-server:/opt/jkf-platform/jkf.db
```

---

## 5. Create a systemd service

```bash
sudo nano /etc/systemd/system/jkf-platform.service
```

```ini
[Unit]
Description=JKF Chatbot & Dashboard Platform
After=network.target

[Service]
User=www-data
WorkingDirectory=/opt/jkf-platform
EnvironmentFile=/opt/jkf-platform/.env
ExecStart=/opt/jkf-platform/.venv/bin/python app.py
Restart=on-failure
RestartSec=5

[Install]
WantedBy=multi-user.target
```

```bash
sudo chown -R www-data:www-data /opt/jkf-platform
sudo systemctl daemon-reload
sudo systemctl enable jkf-platform
sudo systemctl start jkf-platform
sudo systemctl status jkf-platform   # should show "active (running)"
```

---

## 6. Set up nginx as reverse proxy

```bash
sudo nano /etc/nginx/sites-available/jkf-platform
```

```nginx
server {
    listen 80;
    server_name chatbot.jkf.dk;
    return 301 https://$host$request_uri;
}

server {
    listen 443 ssl;
    server_name chatbot.jkf.dk;

    ssl_certificate     /etc/letsencrypt/live/chatbot.jkf.dk/fullchain.pem;
    ssl_certificate_key /etc/letsencrypt/live/chatbot.jkf.dk/privkey.pem;

    client_max_body_size 256M;

    location / {
        proxy_pass         http://127.0.0.1:5001;
        proxy_set_header   Host $host;
        proxy_set_header   X-Real-IP $remote_addr;
        proxy_set_header   X-Forwarded-For $proxy_add_x_forwarded_for;
        proxy_set_header   X-Forwarded-Proto $scheme;
        proxy_read_timeout 120s;
    }
}
```

```bash
sudo ln -s /etc/nginx/sites-available/jkf-platform /etc/nginx/sites-enabled/
sudo nginx -t
sudo systemctl reload nginx
```

---

## 7. SSL certificate (Let's Encrypt)

```bash
sudo apt install certbot python3-certbot-nginx -y
sudo certbot --nginx -d chatbot.jkf.dk
```

---

## 8. Firewall

```bash
sudo ufw allow 22    # SSH
sudo ufw allow 80    # HTTP (redirect only)
sudo ufw allow 443   # HTTPS
sudo ufw enable
```

> The app runs internally on port 5001 — nginx proxies it. No need to expose 5001 publicly.

---

## 9. Verify datawarehouse connectivity

The SQL Server at `10.45.10.221:1433` must be reachable from the new server. Ask JKF IT to whitelist the server's public IP in the SQL Server firewall rules.

```bash
python3 -c "import pymssql; pymssql.connect('10.45.10.221', 'powerbi', '<password>', 'BC2SQL_Data'); print('OK')"
```

---

## 10. Sync Qdrant knowledge base (if needed)

If the Qdrant collection `jkf_kb` is empty on a fresh Qdrant cluster:

```bash
cd /opt/jkf-platform
source .venv/bin/activate
python setup_qdrant.py        # creates the collection
python sync_qa_to_qdrant.py   # uploads Q&A pairs
```

---

## Post-deployment checklist

- [ ] `https://chatbot.jkf.dk` loads the dashboard login page
- [ ] Chatbot responds in the embed widget
- [ ] Order lookups work (SQL Server connection active)
- [ ] File uploads work (check `uploads/` folder permissions)
- [ ] Logs appear in `logs/jkf.log`
- [ ] `CHATBOT_HMAC_SECRET` set on both this server and JKF Universe (see `EMBED_SETUP.md`)
