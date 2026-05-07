"""
scrape_website.py — Crawl jkfuniverse.com/da/ and index all pages into Qdrant.

Usage:
    python scrape_website.py              # Crawl and index everything
    python scrape_website.py --dry-run   # Crawl + parse, print stats, no Qdrant writes
    python scrape_website.py --clear     # Delete existing website chunks first, then re-index
    python scrape_website.py --no-js     # Use plain requests instead of Selenium (faster, less JS)

Re-running is safe: point IDs are deterministic (URL + chunk index), so Qdrant
will upsert (update) existing points rather than creating duplicates.
"""
from dotenv import load_dotenv
load_dotenv()

import os
import re
import json
import time
import uuid
import hashlib
import logging
import argparse
from collections import Counter
from urllib.parse import urljoin, urlparse, urldefrag, urlunparse
from xml.etree import ElementTree

import requests
from bs4 import BeautifulSoup
from openai import OpenAI
from qdrant_client import QdrantClient
from qdrant_client.models import PointStruct, SparseVector, Filter, FieldCondition, MatchValue

# ── Configuration ─────────────────────────────────────────────────────────────
START_URL         = "https://jkfuniverse.com/da/"
ALLOWED_DOMAIN    = "jkfuniverse.com"
URL_PATH_PREFIX   = "/da/"          # Only index Danish-language pages
QDRANT_COLLECTION = "jkf_kb"
REQUEST_DELAY     = 1.0   # seconds between HTTP requests
REQUEST_TIMEOUT   = 20    # seconds
DISCOVERY_LIMIT   = 600
CLIENT_NAME       = "jkf"

HEADERS = {
    "User-Agent": "JKF-Chatbot-Crawler/1.0 (internal knowledge base indexer; "
                  "contact: admin@jkf.dk)"
}

# ── Logging ────────────────────────────────────────────────────────────────────
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-7s  %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)

# ── URL exclusion rules ────────────────────────────────────────────────────────
_EXCLUDED_PATH_PATTERNS = (
    re.compile(r"/privacy-policy/?$"),
    re.compile(r"/password-change/"),
    re.compile(r"/myjkf/", re.I),
    re.compile(r"/pqt", re.I),
    re.compile(r"/find_", re.I),
)

# Phrases that indicate a 404 / error page — skip these entirely
_ERROR_TITLE_PHRASES = (
    "ikke fundet", "not found", "siden findes ikke", "page not found",
    "404", "fejl", "error", "adgang nægtet", "access denied",
    "ser ikke ud til at findes", "siden eksisterer ikke",
    "kan ikke finde siden", "page does not exist", "siden er ikke her",
    "oops", "noget gik galt", "something went wrong",
)

_BOILERPLATE_CLASSES = re.compile(
    r'cookie|banner|popup|modal|breadcrumb|sidebar|social|share|'
    r'menu|navbar|nav-|footer|header|topbar|ribbon|announcement|'
    r'widget|advertisement|ads|overlay|consent',
    re.IGNORECASE
)

_NOISE_TEXT_PATTERNS = (
    "read all about", "data sheet", "download datablade",
    "download datablad", "contact us", "kontakt os",
    "learn more", "show more", "read more",
)

_BOILERPLATE_LINE_PATTERNS = (
    "read all about", "data sheet", "download datablad",
    "downloadable files", "2d cad files", "3d cad files",
    "download selected", "select all", "files selected",
    "senest besøgte projekter", "see product",
)

# Minimum words a page must have after cleaning to be worth indexing
_MIN_PAGE_WORDS = 30


# ─────────────────────────────────────────────────────────────────────────────
# Sparse vector helper (BM25-weighted — must mirror app.py exactly)
# ─────────────────────────────────────────────────────────────────────────────

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

def _compute_sparse_vector(text: str) -> tuple[list[int], list[float]]:
    """BM25-weighted sparse vector with Danish stopword filtering.
    Uses SHA1 hash bucketed to 100k indices. Must match app.py exactly.
    """
    words = re.sub(r'[^\w\s]', ' ', text.lower()).split()
    words = [w for w in words if len(w) > 1 and w not in _DANISH_STOPWORDS]
    if not words:
        return [], []
    counts = Counter(words)
    doc_len = len(words)
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


# ─────────────────────────────────────────────────────────────────────────────
# URL utilities
# ─────────────────────────────────────────────────────────────────────────────

def normalise_url(url: str) -> str:
    """Remove fragment and canonicalise host/path to avoid slash duplicates."""
    url, _ = urldefrag(url)
    p = urlparse(url)
    path = p.path or "/"
    if path != "/":
        path = path.rstrip("/")
    netloc = p.netloc.lower()
    if (p.scheme.lower() == "https" and netloc.endswith(":443")) or \
       (p.scheme.lower() == "http"  and netloc.endswith(":80")):
        netloc = netloc.rsplit(":", 1)[0]
    return p._replace(scheme=p.scheme.lower(), netloc=netloc, path=path).geturl()


def is_allowed(url: str) -> bool:
    p = urlparse(url)
    netloc = p.netloc.lower()
    on_domain = netloc == ALLOWED_DOMAIN or netloc == f"www.{ALLOWED_DOMAIN}"
    on_da_path = p.path.startswith(URL_PATH_PREFIX)
    return on_domain and on_da_path


def is_crawlable(url: str) -> bool:
    skip_exts = ('.jpg', '.jpeg', '.png', '.gif', '.svg', '.webp',
                 '.mp4', '.mp3', '.zip', '.docx', '.xlsx', '.pptx',
                 '.pdf', '.css', '.js', '.woff', '.ttf')
    path = urlparse(url).path.lower()
    if any(path.endswith(ext) for ext in skip_exts):
        return False
    if url.startswith(('mailto:', 'tel:', 'javascript:', 'data:')):
        return False
    return True


def _is_excluded_url(url: str) -> bool:
    path = urlparse(url).path.lower()
    return any(pattern.search(path) for pattern in _EXCLUDED_PATH_PATTERNS)


# ─────────────────────────────────────────────────────────────────────────────
# Sitemap + recursive URL discovery
# ─────────────────────────────────────────────────────────────────────────────

def _fetch_single_sitemap(sitemap_url: str) -> list[str]:
    ns = {"sm": "http://www.sitemaps.org/schemas/sitemap/0.9"}
    try:
        resp = requests.get(sitemap_url, headers=HEADERS, timeout=REQUEST_TIMEOUT)
        if resp.status_code != 200:
            return []
        root = ElementTree.fromstring(resp.content)
        return [loc.text.strip() for loc in root.findall(".//sm:url/sm:loc", ns) if loc.text]
    except Exception:
        return []


def get_sitemap_urls(base_url: str) -> list[str]:
    candidates = [
        urljoin(base_url, "/sitemap.xml"),
        urljoin(base_url, "/sitemap_index.xml"),
        urljoin(base_url, "/da/sitemap.xml"),
    ]
    ns = {"sm": "http://www.sitemaps.org/schemas/sitemap/0.9"}
    all_urls: list[str] = []

    for sitemap_url in candidates:
        try:
            resp = requests.get(sitemap_url, headers=HEADERS, timeout=REQUEST_TIMEOUT)
            if resp.status_code != 200:
                continue
            ct = resp.headers.get("content-type", "")
            if "xml" not in ct and not resp.text.strip().startswith("<"):
                continue
            root = ElementTree.fromstring(resp.content)
            for sitemap in root.findall(".//sm:sitemap/sm:loc", ns):
                all_urls.extend(_fetch_single_sitemap(sitemap.text.strip()))
            locs = [loc.text.strip() for loc in root.findall(".//sm:url/sm:loc", ns) if loc.text]
            all_urls.extend(locs)
            if all_urls:
                logger.info(f"Sitemap: found {len(all_urls)} URLs from {sitemap_url}")
                return all_urls
        except Exception as e:
            logger.debug(f"Sitemap {sitemap_url} failed: {e}")
    return []


def extract_links_from_html(html: str, base_url: str) -> list[str]:
    soup = BeautifulSoup(html, "html.parser")
    links: list[str] = []
    seen: set[str] = set()
    for a in soup.find_all("a", href=True):
        href = (a.get("href") or "").strip()
        if not href or href.startswith("#"):
            continue
        abs_url = normalise_url(urljoin(base_url, href))
        if not abs_url.startswith("http"):
            continue
        if not is_allowed(abs_url) or not is_crawlable(abs_url):
            continue
        if abs_url in seen:
            continue
        seen.add(abs_url)
        links.append(abs_url)
    return links


def discover_urls(start_url: str) -> list[str]:
    sitemap_urls = set(get_sitemap_urls(start_url))
    discovered = set(sitemap_urls)
    queue = [normalise_url(start_url)]
    visited: set[str] = set()

    while queue and len(discovered) < DISCOVERY_LIMIT:
        url = queue.pop(0)
        if url in visited:
            continue
        visited.add(url)
        try:
            resp = requests.get(url, headers=HEADERS, timeout=REQUEST_TIMEOUT)
            if resp.status_code != 200 or "html" not in resp.headers.get("content-type", ""):
                continue
            for link in extract_links_from_html(resp.text, url):
                if link not in discovered:
                    discovered.add(link)
                if link not in visited and link not in queue and len(discovered) < DISCOVERY_LIMIT:
                    queue.append(link)
        except Exception as e:
            logger.debug(f"Recursive discovery failed for {url}: {e}")
        time.sleep(min(REQUEST_DELAY, 0.4))

    logger.info(f"Discovery: {len(sitemap_urls)} sitemap URLs, {len(discovered)} total after crawling")
    return sorted(discovered)


# ─────────────────────────────────────────────────────────────────────────────
# Selenium (JS rendering) — used for actual content fetch
# ─────────────────────────────────────────────────────────────────────────────

def setup_driver():
    """Set up headless Chrome with anti-detection measures."""
    from selenium import webdriver
    from selenium.webdriver.chrome.service import Service
    from selenium.webdriver.chrome.options import Options

    options = Options()
    options.add_argument('--headless')
    options.add_argument('--no-sandbox')
    options.add_argument('--disable-dev-shm-usage')
    options.add_argument('--disable-extensions')
    options.add_argument('--disable-gpu')
    options.add_argument('--window-size=1920,1080')
    options.add_argument('--disable-blink-features=AutomationControlled')
    options.add_experimental_option("excludeSwitches", ["enable-automation"])
    options.add_experimental_option('useAutomationExtension', False)

    chromedriver_path = '/usr/local/bin/chromedriver'
    if not os.path.exists(chromedriver_path):
        from webdriver_manager.chrome import ChromeDriverManager
        chromedriver_path = ChromeDriverManager().install()

    service = Service(chromedriver_path)
    driver = webdriver.Chrome(service=service, options=options)
    driver.set_page_load_timeout(30)
    driver.set_script_timeout(10)
    driver.execute_cdp_cmd('Network.setUserAgentOverride', {
        "userAgent": ('Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) '
                      'AppleWebKit/537.36 (KHTML, like Gecko) '
                      'Chrome/120.0.0.0 Safari/537.36')
    })
    return driver


def fetch_with_selenium(url: str, driver) -> str | None:
    """
    Fetch a page using Selenium and wait for JS content to fully render.

    Strategy:
      1. Wait for document.readyState == 'complete'
      2. Fixed 2s pause for JS frameworks (React/Vue component mount)
      3. DOM stability: wait until body text length stops growing
      4. Scroll to trigger lazy-loaded sections, then wait for new content
    """
    from selenium.webdriver.support.ui import WebDriverWait
    from selenium.common.exceptions import TimeoutException

    try:
        driver.get(url)

        # Step 1: readyState == 'complete'
        try:
            WebDriverWait(driver, 20).until(
                lambda d: d.execute_script("return document.readyState") == "complete"
            )
        except TimeoutException:
            logger.warning(f"readyState timeout for {url} — continuing anyway")

        # Step 2: JS framework render window
        time.sleep(2)

        # Step 3: DOM stability check
        try:
            prev_len = 0
            for _ in range(6):
                current_len = len(driver.execute_script("return document.body.innerText"))
                if current_len == prev_len and current_len > 0:
                    break
                prev_len = current_len
                time.sleep(0.5)
        except Exception:
            pass

        # Step 4: Scroll to trigger lazy-loaded sections
        try:
            total_height = driver.execute_script("return document.body.scrollHeight")
            viewport = driver.execute_script("return window.innerHeight")
            position = 0
            while position < total_height:
                position = min(position + viewport, total_height)
                driver.execute_script(f"window.scrollTo(0, {position});")
                time.sleep(0.3)
            time.sleep(0.8)
            driver.execute_script("window.scrollTo(0, 0);")
            time.sleep(0.2)
        except Exception as e:
            logger.debug(f"Scroll error for {url}: {e}")

        return driver.page_source

    except TimeoutException:
        logger.error(f"Page load timeout for {url}")
        return None
    except Exception as e:
        logger.error(f"Selenium error for {url}: {e}")
        return None


# ─────────────────────────────────────────────────────────────────────────────
# Content extraction (ported from cloud_scraper.py)
# ─────────────────────────────────────────────────────────────────────────────

def _extract_table_content(table_element) -> str | None:
    """Extract a <table> as a markdown table, handling colspan, thead/tbody/tfoot."""
    try:
        caption = table_element.find('caption')
        caption_text = caption.get_text(strip=True) if caption else ''

        def cell_texts(cell):
            text = ' '.join(cell.get_text(separator=' ', strip=True).split())
            text = text.replace('|', '\\|')
            try:
                span = min(max(1, int(cell.get('colspan', 1))), 8)
            except (ValueError, TypeError):
                span = 1
            return [text] * span

        def parse_row(tr):
            row = []
            for cell in tr.find_all(['th', 'td'], recursive=False):
                row.extend(cell_texts(cell))
            return row

        def is_header_row(tr):
            cells = tr.find_all(['th', 'td'], recursive=False)
            return bool(cells) and all(c.name == 'th' for c in cells)

        header_rows, body_rows = [], []

        thead = table_element.find('thead', recursive=False)
        if thead:
            for tr in thead.find_all('tr', recursive=False):
                row = parse_row(tr)
                if any(t.strip() for t in row):
                    header_rows.append(row)

        for tbody in table_element.find_all('tbody', recursive=False):
            for tr in tbody.find_all('tr', recursive=False):
                row = parse_row(tr)
                if any(t.strip() for t in row):
                    body_rows.append(row)

        tfoot = table_element.find('tfoot', recursive=False)
        if tfoot:
            for tr in tfoot.find_all('tr', recursive=False):
                row = parse_row(tr)
                if any(t.strip() for t in row):
                    body_rows.append(row)

        direct_trs = table_element.find_all('tr', recursive=False)
        for i, tr in enumerate(direct_trs):
            row = parse_row(tr)
            if not any(t.strip() for t in row):
                continue
            if i == 0 and not header_rows and is_header_row(tr):
                header_rows.append(row)
            else:
                body_rows.append(row)

        if not header_rows and body_rows and not direct_trs:
            tbodies = table_element.find_all('tbody', recursive=False)
            first_tr = (tbodies[0] if tbodies else table_element).find('tr')
            if first_tr and is_header_row(first_tr):
                header_rows = [body_rows.pop(0)]

        all_rows = header_rows + body_rows
        if not all_rows:
            return None

        col_count = max(len(r) for r in all_rows)
        if col_count == 0:
            return None

        def pad(row):
            return row + [''] * (col_count - len(row))

        lines = []
        if header_rows:
            for r in header_rows:
                lines.append('| ' + ' | '.join(pad(r)) + ' |')
            lines.append('|' + '|'.join([' --- '] * col_count) + '|')
        elif body_rows:
            lines.append('| ' + ' | '.join(pad(body_rows[0])) + ' |')
            lines.append('|' + '|'.join([' --- '] * col_count) + '|')
            body_rows = body_rows[1:]

        for r in body_rows:
            lines.append('| ' + ' | '.join(pad(r)) + ' |')

        if not lines:
            return None

        table_md = '\n'.join(lines)
        return (f"\n**{caption_text}**\n{table_md}\n" if caption_text
                else f"\n{table_md}\n")

    except Exception as e:
        logger.warning(f"Error extracting table: {e}")
        return None


def _extract_jsonld(soup: BeautifulSoup) -> str:
    """Extract useful data from JSON-LD schema markup (FAQPage, LocalBusiness, etc.)."""
    blocks = []
    for tag in soup.find_all('script', type='application/ld+json'):
        try:
            data = json.loads(tag.string or '')
            items = data if isinstance(data, list) else [data]
            for item in items:
                schema_type = item.get('@type', '')

                if schema_type == 'FAQPage':
                    for entry in item.get('mainEntity', []):
                        q = entry.get('name', '').strip()
                        a_obj = entry.get('acceptedAnswer', {})
                        a = BeautifulSoup(a_obj.get('text', ''), 'html.parser').get_text(strip=True)
                        if q and a:
                            blocks.append(f"**FAQ: {q}**\n{a}")

                elif schema_type in ('LocalBusiness', 'Organization', 'Store',
                                     'Hotel', 'Restaurant', 'MedicalBusiness'):
                    parts = []
                    if item.get('name'):
                        parts.append(f"Navn: {item['name']}")
                    addr = item.get('address', {})
                    if isinstance(addr, dict):
                        addr_parts = [addr.get('streetAddress', ''), addr.get('postalCode', ''),
                                      addr.get('addressLocality', '')]
                        addr_str = ', '.join(p for p in addr_parts if p)
                        if addr_str:
                            parts.append(f"Adresse: {addr_str}")
                    if item.get('telephone'):
                        parts.append(f"Telefon: {item['telephone']}")
                    if item.get('email'):
                        parts.append(f"Email: {item['email']}")
                    hours = item.get('openingHours') or item.get('openingHoursSpecification', [])
                    if hours:
                        if isinstance(hours, list):
                            parts.append("Åbningstider: " + '; '.join(
                                h if isinstance(h, str) else
                                f"{h.get('dayOfWeek','')}: {h.get('opens','')}-{h.get('closes','')}"
                                for h in hours
                            ))
                        elif isinstance(hours, str):
                            parts.append(f"Åbningstider: {hours}")
                    if parts:
                        blocks.append('\n'.join(parts))
        except Exception:
            continue
    return '\n\n'.join(blocks)


def _extract_footer_contact(soup: BeautifulSoup) -> str:
    """Selectively extract contact info (phone, email, address, hours) from the footer."""
    footer = soup.find('footer')
    if not footer:
        return ''

    contact_blocks = []
    seen = set()

    for el in footer.find_all(string=re.compile(
            r'(\+45[\s\-]?\d{2}[\s\-]?\d{2}[\s\-]?\d{2}[\s\-]?\d{2}|\b\d{2}\s?\d{2}\s?\d{2}\s?\d{2}\b)')):
        t = el.strip()
        if t and t not in seen:
            contact_blocks.append(f"Telefon: {t}")
            seen.add(t)

    for el in footer.find_all(string=re.compile(r'[a-zA-Z0-9._%+\-]+@[a-zA-Z0-9.\-]+\.[a-zA-Z]{2,}')):
        t = el.strip()
        if t and t not in seen:
            contact_blocks.append(f"Email: {t}")
            seen.add(t)

    for el in footer.find_all(['p', 'li', 'address', 'span', 'div']):
        text = el.get_text(strip=True)
        if not text or len(text) < 5 or text in seen:
            continue
        is_contact = bool(re.search(
            r'(åbne?tider?|mandag|tirsdag|onsdag|torsdag|fredag|lørdag|søndag|'
            r'adresse|postnr|telefon|email|tlf|cvr|ejer|kontakt|'
            r'monday|tuesday|wednesday|thursday|friday|saturday|sunday|'
            r'opening hours|address|phone|contact)',
            text, re.IGNORECASE
        ))
        if is_contact:
            contact_blocks.append(text)
            seen.add(text)

    if not contact_blocks:
        return ''
    return '## Kontakt og åbningstider\n' + '\n'.join(contact_blocks)


def extract_content(html: str, url: str) -> tuple[str, str]:
    """
    Parse HTML and return (page_title, clean_text).

    Uses the improved cloud_scraper extraction: tables → markdown, <dl> pairs,
    <details> accordions, JSON-LD structured data, footer contact info,
    image alt text, meta description, and anchors with hrefs preserved.
    """
    try:
        soup = BeautifulSoup(html, 'html.parser')

        # Extract JSON-LD and footer contact BEFORE removing those elements
        jsonld_text = _extract_jsonld(soup)
        footer_text = _extract_footer_contact(soup)

        # Extract meta description
        meta_desc = ''
        for attr in ({'name': 'description'}, {'property': 'og:description'}):
            tag = soup.find('meta', attrs=attr)
            if tag:
                meta_desc = tag.get('content', '').strip()
                if meta_desc:
                    break

        # Remove noise elements
        for el in soup(['script', 'style', 'noscript', 'nav', 'header', 'footer', 'iframe']):
            el.decompose()

        # Also remove boilerplate by class/id
        for tag in soup.find_all(True):
            try:
                attrs = tag.attrs or {}
                classes = " ".join(attrs.get("class", []) or [])
                tag_id = attrs.get("id", "") or ""
                if _BOILERPLATE_CLASSES.search(classes) or _BOILERPLATE_CLASSES.search(tag_id):
                    tag.decompose()
            except Exception:
                pass

        # Page title
        title = ''
        if soup.title and soup.title.string:
            title = soup.title.string.strip().split('|')[0].strip()
        if not title:
            h1 = soup.find('h1')
            title = h1.get_text(' ', strip=True) if h1 else url

        # Main content area
        main_selectors = [
            'article', 'main', '[role="main"]',
            '.content', '#content', '.main-content',
            '.entry-content', '.post-content', '.article-content',
        ]
        main_content = None
        for selector in main_selectors:
            main_content = soup.select_one(selector)
            if main_content:
                break
        if not main_content:
            main_content = soup.body

        content_blocks = []
        extracted_texts: set[str] = set()

        if meta_desc:
            content_blocks.append(f"**Sidebeskrivelse:** {meta_desc}")
            extracted_texts.add(meta_desc)

        if main_content:
            # Pass 1: structured elements
            for element in main_content.find_all(
                    ['h1', 'h2', 'h3', 'h4', 'h5', 'h6',
                     'p', 'li', 'blockquote', 'dd', 'dt',
                     'pre', 'code', 'table', 'a', 'dl', 'details']):

                if element.name in ['h1', 'h2', 'h3', 'h4', 'h5', 'h6']:
                    text = element.get_text(strip=True)
                    if text and len(text) > 1 and text not in extracted_texts:
                        level = int(element.name[1])
                        content_blocks.append('#' * level + ' ' + text)
                        extracted_texts.add(text)

                elif element.name == 'table':
                    table_text = _extract_table_content(element)
                    if table_text and table_text not in extracted_texts:
                        content_blocks.append(table_text)
                        extracted_texts.add(table_text)

                elif element.name == 'dl':
                    pairs = []
                    current_term = None
                    for child in element.children:
                        if not hasattr(child, 'name'):
                            continue
                        if child.name == 'dt':
                            current_term = child.get_text(strip=True)
                        elif child.name == 'dd' and current_term:
                            definition = child.get_text(strip=True)
                            if definition:
                                pairs.append(f"{current_term}: {definition}")
                                extracted_texts.add(current_term)
                                extracted_texts.add(definition)
                            current_term = None
                    if pairs:
                        dl_text = '\n'.join(pairs)
                        if dl_text not in extracted_texts:
                            content_blocks.append(dl_text)
                            extracted_texts.add(dl_text)

                elif element.name == 'details':
                    summary_el = element.find('summary')
                    summary_text = summary_el.get_text(strip=True) if summary_el else ''
                    detail_parts = []
                    for child in element.children:
                        if not hasattr(child, 'name') or child.name == 'summary':
                            continue
                        t = child.get_text(separator=' ', strip=True)
                        t = ' '.join(t.split())
                        if t:
                            detail_parts.append(t)
                    detail_text = ' '.join(detail_parts).strip()
                    if summary_text:
                        extracted_texts.add(summary_text)
                    if detail_text:
                        extracted_texts.add(detail_text)
                    if summary_text or detail_text:
                        block = (f"**{summary_text}**\n{detail_text}"
                                 if summary_text and detail_text
                                 else summary_text or detail_text)
                        if block not in extracted_texts:
                            content_blocks.append(block)
                            extracted_texts.add(block)

                elif element.name in ['pre', 'code']:
                    text = element.get_text(strip=True)
                    if text and len(text) > 1 and text not in extracted_texts:
                        content_blocks.append(f'```\n{text}\n```')
                        extracted_texts.add(text)

                elif element.name == 'a':
                    text = element.get_text(strip=True)
                    href = element.get('href', '').strip()
                    if not text or len(text) < 3:
                        continue
                    # Skip generic CTA / navigation button text — these produce
                    # low-quality inline links that confuse the LLM
                    _cta_patterns = re.compile(
                        r'^(read (more|all)|learn more|show more|see (more|product|all)|'
                        r'download|kontakt os|contact us|get in touch|book|'
                        r'find out|more info|click here|go to|tilbage|tilmeld|'
                        r'log (in|ud|på)|sign (in|up)|submit|send|next|previous|'
                        r'se (mere|produkt|alle)|læs mere|hent|åbn|open)$',
                        re.IGNORECASE
                    )
                    if _cta_patterns.match(text):
                        # Store just the text, not as a link
                        if text not in extracted_texts:
                            content_blocks.append(text)
                            extracted_texts.add(text)
                        continue
                    if href and href not in ('#', '') and not href.startswith(
                            ('javascript:', 'data:', 'tel:', 'mailto:')):
                        if not href.startswith('http'):
                            href = urljoin(url, href)
                        entry = f'[{text}]({href})'
                    else:
                        entry = text
                    if entry not in extracted_texts:
                        content_blocks.append(entry)
                        extracted_texts.add(entry)

                else:
                    text = element.get_text(strip=True)
                    if text and len(text) > 1 and text not in extracted_texts:
                        content_blocks.append(text)
                        extracted_texts.add(text)

            # Pass 2: direct text in divs/spans
            for element in main_content.find_all(['div', 'span']):
                direct_text = ''.join(
                    element.find_all(string=True, recursive=False)
                ).strip()
                if (direct_text and len(direct_text) > 3
                        and direct_text not in extracted_texts
                        and not element.find(['p', 'div', 'h1', 'h2', 'h3',
                                              'h4', 'h5', 'h6', 'ul', 'ol', 'table'])):
                    content_blocks.append(direct_text)
                    extracted_texts.add(direct_text)

            # Pass 3: image alt text
            for img in main_content.find_all('img'):
                alt = img.get('alt', '').strip()
                if alt and len(alt) > 5 and alt not in extracted_texts:
                    content_blocks.append(f'[Image: {alt}]')
                    extracted_texts.add(alt)

        if footer_text:
            content_blocks.append(footer_text)
        if jsonld_text:
            content_blocks.append('## Strukturerede data\n' + jsonld_text)

        full_text = '\n\n'.join(content_blocks)
        return title, full_text

    except Exception as e:
        logger.error(f"Error extracting content from {url}: {e}")
        return url, ''


# ─────────────────────────────────────────────────────────────────────────────
# Semantic chunking (ported from cloud_scraper.py)
# ─────────────────────────────────────────────────────────────────────────────

def _semantic_chunk_document(doc: dict) -> list[dict]:
    """
    Split a document into semantic chunks.

    Strategy:
    - Flush on headings (markdown # or ALL-CAPS/Title-Case short lines)
    - Flush when chunk exceeds MAX_WORDS
    - Split oversized chunks at sentence boundaries with 150-word overlap
    - Each chunk carries: text, section_title, source_url, page_title
    """
    content    = doc.get('content', '').strip()
    source_url = doc.get('url', '')
    page_title = doc.get('title', '')

    if not content:
        return []

    MAX_WORDS = 400  # Smaller chunks = more focused retrieval hits
    MIN_WORDS = 100  # Ensure minimum informativeness per chunk

    # Pre-process: fix hyphenated line-breaks, collapse blank lines, remove lone page numbers
    content = re.sub(r'(\w)-\n(\w)', r'\1\2', content)
    content = re.sub(r'\n{3,}', '\n\n', content)
    content = re.sub(r'^\s*\d{1,4}\s*$', '', content, flags=re.MULTILINE)

    chunks          = []
    current_lines   = []
    current_section = page_title
    overlap_seed    = ''

    def _is_heading(line):
        s = line.strip()
        if not s or len(s) > 120:
            return False
        if re.match(r'^#{1,4}\s+\S', s):
            return True
        if s == s.upper() and len(s.split()) >= 2 and s[-1] not in '.!?,;:)':
            return True
        words = s.split()
        if (len(words) <= 10 and words[0][0].isupper() and s[-1] not in '.!?,;:)'):
            return True
        if re.match(r'^(\d+\.)+\s+\S', s):
            return True
        return False

    def _make_overlap(chunk_text, n_words=150):
        words = chunk_text.split()
        if len(words) <= n_words:
            return chunk_text
        tail = ' '.join(words[-n_words:])
        m = re.search(r'(?<=[.!?])\s+\S', tail)
        return tail[m.start():].strip() if m else tail

    def flush_chunk(lines, section, overlap):
        raw = '\n'.join(lines).strip()
        if not raw:
            return None
        text = (overlap + '\n\n' + raw).strip() if overlap else raw
        # URL is stored ONLY in source_url metadata — never embedded in text.
        # Embedding it pollutes BM25 sparse vectors and breaks chunk splitting.
        return {
            'text':          text,
            'section_title': section,
            'source_url':    source_url,
            'page_title':    page_title,
        }

    def split_oversized(chunk_dict):
        words = chunk_dict['text'].split()
        if len(words) <= MAX_WORDS:
            return [chunk_dict]
        sentences = re.split(r'(?<=[.!?])\s+', chunk_dict['text'])
        parts, cur_sents, cur_words = [], [], 0
        for sent in sentences:
            sw = len(sent.split())
            if cur_words + sw > MAX_WORDS and cur_sents:
                part_text = ' '.join(cur_sents)
                parts.append({**chunk_dict, 'text': part_text})
                overlap = _make_overlap(part_text)
                cur_sents = ([overlap, sent] if overlap else [sent])
                cur_words = len(overlap.split()) + sw if overlap else sw
            else:
                cur_sents.append(sent)
                cur_words += sw
        if cur_sents:
            parts.append({**chunk_dict, 'text': ' '.join(cur_sents)})
        return parts or [chunk_dict]

    for line in content.split('\n'):
        stripped = line.strip()

        if _is_heading(stripped):
            current_words = len(' '.join(current_lines).split()) if current_lines else 0
            if current_words >= MIN_WORDS:
                chunk = flush_chunk(current_lines, current_section, overlap_seed)
                if chunk:
                    for sub in split_oversized(chunk):
                        chunks.append(sub)
                    overlap_seed = _make_overlap(chunks[-1]['text']) if chunks else ''
                current_lines = []
            current_section = re.sub(r'^#{1,4}\s+', '', stripped)
            current_lines.append(stripped)
            continue

        if stripped == '':
            continue

        current_lines.append(stripped)

        if len(' '.join(current_lines).split()) >= MAX_WORDS:
            chunk = flush_chunk(current_lines, current_section, overlap_seed)
            if chunk:
                for sub in split_oversized(chunk):
                    chunks.append(sub)
                overlap_seed = _make_overlap(chunks[-1]['text']) if chunks else ''
            current_lines = []

    # End-of-document flush
    if current_lines:
        chunk = flush_chunk(current_lines, current_section, overlap_seed)
        if chunk:
            for sub in split_oversized(chunk):
                chunks.append(sub)

    return chunks


def _build_embed_text(chunk: dict) -> str:
    """Build a contextualized string for embedding by prepending chunk metadata.

    The prefix signals to the embedding model what domain and section the chunk
    belongs to, improving recall for domain-specific queries.

    SAFETY: The returned string is ONLY used to generate the embedding vector.
    chunk["text"] (the original content) is what gets stored and served to the LLM.
    """
    page_title    = (chunk.get("page_title", "") or "").strip()
    section_title = (chunk.get("section_title", "") or "").strip()
    source_url    = (chunk.get("source_url", "") or "").strip()

    # Derive a human-readable page name from the URL path.
    # Many pages use "JKFuniverse" as their HTML title, which is useless as embedding context.
    # The URL path is always specific: /da/contact/jkf-industri → "contact jkf industri"
    url_label = ""
    if source_url:
        from urllib.parse import urlparse
        path = urlparse(source_url).path.lstrip('/')
        if path.startswith('da/'):
            path = path[3:]  # strip language prefix
        url_label = re.sub(r'[-_/]+', ' ', path).strip()

    parts = ["JKF"]
    # Use URL-derived label if page_title is too generic
    _GENERIC_TITLES = {"jkfuniverse", "jkf", "jkf universe", "home", "forside"}
    if page_title and page_title.lower().strip() not in _GENERIC_TITLES:
        parts.append(page_title[:80])
    elif url_label:
        parts.append(url_label[:80])
    if section_title and section_title != page_title and section_title.lower().strip() not in _GENERIC_TITLES:
        parts.append(section_title[:80])

    prefix = " | ".join(parts)
    return f"[{prefix}]\n\n{chunk['text']}"


# ─────────────────────────────────────────────────────────────────────────────
# Embedding — batched with contextual prefixes
# ─────────────────────────────────────────────────────────────────────────────

def embed_chunks(chunks: list[dict], openai_client: OpenAI) -> list[list[float]]:
    """Embed a list of chunk dicts using contextual prefixes for better precision."""
    texts = [_build_embed_text(c) for c in chunks]
    vectors = []
    batch_size = 100
    for i in range(0, len(texts), batch_size):
        batch = texts[i:i + batch_size]
        resp = openai_client.embeddings.create(
            model="text-embedding-3-large",
            input=batch,
        )
        vectors.extend([item.embedding for item in resp.data])
        logger.info(f"  Embedded {min(i + batch_size, len(texts))}/{len(texts)} chunks")
    return vectors


# ─────────────────────────────────────────────────────────────────────────────
# Qdrant operations
# ─────────────────────────────────────────────────────────────────────────────

def clear_website_chunks(qc: QdrantClient) -> None:
    """Remove all previously indexed website chunks from Qdrant."""
    logger.info("Deleting existing website chunks from Qdrant...")
    qc.delete(
        collection_name=QDRANT_COLLECTION,
        points_selector=Filter(
            must=[FieldCondition(key="source_type", match=MatchValue(value="website"))]
        ),
    )
    logger.info("Done — existing website chunks removed.")


def index_pages(
    pages: dict[str, tuple[str, str]],
    openai_client: OpenAI,
    qc: QdrantClient,
) -> None:
    """
    Semantically chunk all pages, embed them, deduplicate, and upsert into Qdrant.

    Always deletes all existing website chunks first so re-runs never leave stale
    content from removed or renamed pages.

    Payload stored per chunk (compatible with app.py):
      - text          : the chunk text (fed to the model as context)
      - source_type   : "website" (used for filtering / bulk delete)
      - source_url    : exact page URL (chatbot cites this)
      - section_title : section heading (shown as context header)
    """
    # Always clear old website chunks — guarantees no stale content from previous runs
    clear_website_chunks(qc)

    # Build all chunks
    all_chunks = []
    seen_404_hashes: set[str] = set()

    for url, (title, text) in pages.items():
        # Skip 404/error pages by title
        if any(phrase in title.lower() for phrase in _ERROR_TITLE_PHRASES):
            logger.warning(f"Skipping error/404 page: {url} (title: '{title}')")
            content_hash = hashlib.md5(text.strip().encode()).hexdigest()
            seen_404_hashes.add(content_hash)
            continue

        # Skip pages whose content matches a known 404 body
        content_hash = hashlib.md5(text.strip().encode()).hexdigest()
        if content_hash in seen_404_hashes:
            logger.warning(f"Skipping 404-body page: {url}")
            continue

        word_count = len(text.split())
        if word_count < _MIN_PAGE_WORDS:
            logger.warning(f"Skipping low-content page: {url} ({word_count} words)")
            continue

        doc = {'url': url, 'title': title, 'content': text}
        chunks = _semantic_chunk_document(doc)
        all_chunks.extend(chunks)

    # Drop stub chunks (navigation lists, lone headings, CTA-only content)
    MIN_CHUNK_WORDS = 50
    before = len(all_chunks)
    all_chunks = [c for c in all_chunks if len(c.get('text', '').split()) >= MIN_CHUNK_WORDS]
    if before != len(all_chunks):
        logger.info(f"Dropped {before - len(all_chunks)} stub chunks (< {MIN_CHUNK_WORDS} words)")

    # Cross-page deduplication: keep the chunk with the most-specific URL (deepest path)
    chunk_by_hash: dict[str, dict] = {}
    for chunk in all_chunks:
        text_hash = hashlib.md5(chunk.get('text', '').strip().encode()).hexdigest()
        if text_hash not in chunk_by_hash:
            chunk_by_hash[text_hash] = chunk
        else:
            existing_url = chunk_by_hash[text_hash].get('source_url', '')
            new_url      = chunk.get('source_url', '')
            existing_depth = len(existing_url.rstrip('/').split('/'))
            new_depth      = len(new_url.rstrip('/').split('/'))
            if new_depth > existing_depth:
                chunk_by_hash[text_hash] = chunk

    deduped = list(chunk_by_hash.values())
    if len(deduped) < len(all_chunks):
        logger.info(f"Deduped {len(all_chunks) - len(deduped)} cross-page duplicate chunks")
    all_chunks = deduped

    logger.info(f"Total: {len(all_chunks)} semantic chunks from {len(pages)} pages")

    if not all_chunks:
        logger.warning("No chunks to index.")
        return

    # Embed in batches
    logger.info("Embedding chunks with contextual prefixes…")
    vectors = embed_chunks(all_chunks, openai_client)

    # Upsert into Qdrant
    total_chunks = 0
    failed       = 0
    upsert_batch = 100

    # Per-URL chunk counter for deterministic IDs: same URL+chunk_index always → same UUID.
    # Using a global index breaks this (adding/removing pages shifts all subsequent IDs).
    url_chunk_counters: dict[str, int] = {}

    points = []
    for i, chunk in enumerate(all_chunks):
        try:
            sp_idx, sp_val = _compute_sparse_vector(chunk['text'])
            src_url = chunk['source_url']
            per_url_idx = url_chunk_counters.get(src_url, 0)
            url_chunk_counters[src_url] = per_url_idx + 1
            # Deterministic per-URL ID: stable across re-runs as long as page content order is stable
            point_id = str(uuid.uuid5(
                uuid.NAMESPACE_URL,
                f"jkf-web-{src_url}-chunk-{per_url_idx}"
            ))
            points.append(PointStruct(
                id=point_id,
                vector={
                    "dense":  vectors[i],
                    "sparse": SparseVector(indices=sp_idx, values=sp_val),
                },
                payload={
                    "text":          chunk['text'],
                    "source_type":   "website",
                    "source_url":    chunk['source_url'],
                    "section_title": chunk.get('section_title', chunk.get('page_title', '')),
                },
            ))
        except Exception as e:
            failed += 1
            logger.error(f"Failed to prepare chunk {i}: {e}")

    for i in range(0, len(points), upsert_batch):
        batch = points[i:i + upsert_batch]
        try:
            qc.upsert(collection_name=QDRANT_COLLECTION, points=batch)
            total_chunks += len(batch)
            logger.info(f"  Upserted {i + len(batch)}/{len(points)} points")
        except Exception as e:
            failed += len(batch)
            logger.error(f"  Batch upsert failed: {e}")

    logger.info(
        f"\n{'='*60}\n"
        f"Indexing complete!\n"
        f"  Pages indexed   : {len(pages)}\n"
        f"  Chunks upserted : {total_chunks}\n"
        f"  Chunks failed   : {failed}\n"
        f"{'='*60}"
    )


# ─────────────────────────────────────────────────────────────────────────────
# Crawler
# ─────────────────────────────────────────────────────────────────────────────

def crawl(start_url: str, use_js: bool = True) -> dict[str, tuple[str, str]]:
    """
    Discover all pages and fetch + extract their content.

    URL discovery uses fast plain requests (sitemap + recursive link follow).
    Content fetching uses Selenium by default (use_js=False for plain requests).
    """
    discovered = discover_urls(start_url)
    to_visit: set[str] = {
        normalise_url(u) for u in discovered
        if is_allowed(u) and is_crawlable(u) and not _is_excluded_url(u)
    }
    logger.info(f"Crawling {len(to_visit)} pages (JS rendering: {use_js})")

    pages: dict[str, tuple[str, str]] = {}
    driver = None

    if use_js:
        try:
            driver = setup_driver()
            logger.info("Selenium driver ready")
        except Exception as e:
            logger.warning(f"Could not start Selenium ({e}) — falling back to plain requests")
            use_js = False

    visited: set[str] = set()

    try:
        for url in sorted(to_visit):
            if url in visited:
                continue
            visited.add(url)

            try:
                logger.info(f"  GET {url}")

                if use_js and driver:
                    html = fetch_with_selenium(url, driver)
                    if not html:
                        logger.warning(f"      Selenium returned nothing — skipping")
                        continue
                else:
                    resp = requests.get(url, headers=HEADERS, timeout=REQUEST_TIMEOUT)
                    if resp.status_code != 200:
                        logger.warning(f"      HTTP {resp.status_code} — skipping")
                        continue
                    if "html" not in resp.headers.get("content-type", ""):
                        continue
                    html = resp.text
                    time.sleep(REQUEST_DELAY)

                title, clean_text = extract_content(html, url)

                word_count = len(clean_text.split())
                if word_count < _MIN_PAGE_WORDS:
                    logger.info(f"      Too little content ({word_count} words) — skipping")
                    continue

                pages[url] = (title, clean_text)
                logger.info(f"      '{title}' | {word_count} words")

            except Exception as e:
                logger.error(f"      Error on {url}: {e}")

    finally:
        if driver:
            try:
                driver.quit()
            except Exception:
                pass

    logger.info(f"\nCrawl complete: {len(pages)} usable pages found")
    return pages


# ─────────────────────────────────────────────────────────────────────────────
# Entry point
# ─────────────────────────────────────────────────────────────────────────────

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Crawl jkfuniverse.com and index pages into Qdrant for the JKF chatbot."
    )
    parser.add_argument(
        "--dry-run", action="store_true",
        help="Crawl and extract content but do NOT write anything to Qdrant."
    )
    parser.add_argument(
        "--clear", action="store_true",
        help="(No-op — website chunks are now always cleared before re-indexing.)"
    )
    parser.add_argument(
        "--no-js", action="store_true",
        help="Use plain requests instead of Selenium (faster but misses JS-rendered content)."
    )
    args = parser.parse_args()

    use_js = not args.no_js

    # Step 1: Crawl
    logger.info(f"Starting crawl from {START_URL}\n{'='*60}")
    pages = crawl(START_URL, use_js=use_js)

    if not pages:
        logger.error("No pages found. Check internet access and the start URL.")
        return

    if args.dry_run:
        logger.info("\nDRY RUN — Qdrant writes skipped. Summary:")
        total_chunks = 0
        for url, (title, text) in sorted(pages.items()):
            doc = {'url': url, 'title': title, 'content': text}
            n = len(_semantic_chunk_document(doc))
            total_chunks += n
            logger.info(f"  {n:3d} chunks | {title[:60]}")
            logger.info(f"          {url}")
        logger.info(f"\nTotal: {len(pages)} pages, ~{total_chunks} chunks would be indexed.")
        return

    # Step 2: Connect to Qdrant
    qdrant_url = os.environ.get("QDRANT_URL", "")
    qdrant_key = os.environ.get("QDRANT_API_KEY")
    if not qdrant_url:
        logger.error("QDRANT_URL not set in .env — cannot connect.")
        return

    qc = QdrantClient(url=qdrant_url, api_key=qdrant_key or None, prefer_grpc=False)
    openai_client = OpenAI(api_key=os.environ["OPENAI_API_KEY"])

    # Step 3: Embed and upsert (clear is handled automatically inside index_pages)
    index_pages(pages, openai_client, qc)


if __name__ == "__main__":
    main()
