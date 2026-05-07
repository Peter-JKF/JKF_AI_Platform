# Chatbot Embed Setup

## 1. Basic embed (anonymous users)

Add this to the base HTML template on the JKF website, just before `</body>`:

```html
<script>
  window.CHATBOT_BASE_URL = 'https://chatbot.jkf.dk';
</script>
<script src="https://chatbot.jkf.dk/static/chatbot-embed.js" defer></script>
```

This gives all visitors access to the chatbot for general questions and public product/stock lookups. Order lookups will require the visitor to provide their customer number manually.

---

## 2. Logged-in users (automatic order access)

When a JKF Universe user is logged in, inject their BC customer number as a signed token. The chatbot will then answer order questions automatically without asking for verification.

### Django view / template tag

```python
# views.py or a context processor
import hmac, hashlib
from django.conf import settings

def get_chatbot_token(user):
    if not user.is_authenticated:
        return None
    customer_no = getattr(user.profile, 'bc_customer_no', None)  # adjust field name
    if not customer_no:
        return None
    sig = hmac.new(
        settings.CHATBOT_HMAC_SECRET.encode(),
        str(customer_no).encode(),
        hashlib.sha256,
    ).hexdigest()
    return f"{customer_no}:{sig}"
```

### Base template

```html
<script>
  window.CHATBOT_BASE_URL = 'https://chatbot.jkf.dk';
  {% if request.user.is_authenticated %}
    window.CHATBOT_COMPANY_TOKEN = "{{ chatbot_company_token }}";
  {% else %}
    window.CHATBOT_COMPANY_TOKEN = null;
  {% endif %}
</script>
<script src="https://chatbot.jkf.dk/static/chatbot-embed.js" defer></script>
```

> **Note:** `bc_customer_no` is the BC customer account number on the user's profile (e.g. `1788`, `2680`). Confirm the exact field name with the JKF Universe dev team.

---

## 3. Shared secret

Both sides must use the same secret key.

**Chatbot backend** — add to `.env`:
```
CHATBOT_HMAC_SECRET=your-secret-here
```

**JKF Universe** — add to `settings.py` or `.env`:
```
CHATBOT_HMAC_SECRET=your-secret-here
```

Generate a strong secret (run once):
```bash
python3 -c "import secrets; print(secrets.token_hex(32))"
```

---

## 4. What the chatbot can do per user type

| User | Order lookup | Verification required |
|---|---|---|
| Anonymous | Yes — by order ref + customer number | Yes, asked in chat |
| Logged-in (JKF Universe) | Yes — fully automatic | No |

**Order references accepted** (all three work):
- BC internal order number (e.g. `0435897`)
- Shipment number (e.g. `F-237780`)
- Customer's own PO number / eksternt bilagsnr (e.g. `4500041080`)

---

## 5. Rollout order

1. Deploy chatbot backend changes
2. Set `CHATBOT_HMAC_SECRET` on the chatbot server
3. Add the Django token helper and set `CHATBOT_HMAC_SECRET` in JKF Universe
4. Update the base template to inject `window.CHATBOT_COMPANY_TOKEN`
5. Test with a real logged-in account on staging
