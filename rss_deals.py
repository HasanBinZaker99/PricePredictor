from urllib.parse import parse_qsl, urlencode, urlsplit, urlunsplit

import feedparser
import requests
from bs4 import BeautifulSoup


FEEDS = [
    ("Electronics", "https://www.dealnews.com/c142/Electronics/?rss=1"),
    ("Computers", "https://www.dealnews.com/c39/Computers/?rss=1"),
    ("Smart Home", "https://www.dealnews.com/f1912/Smart-Home/?rss=1"),
]

TIMEOUT = (5, 15)  # Connection timeout, then read timeout, in seconds.
HEADERS = {"User-Agent": "MyProject-Learning-App/0.2"}


def clean_text(html_text):
    """Turn a feed's HTML summary into plain text for the table."""
    soup = BeautifulSoup(html_text or "", "html.parser")
    for tag in soup(["script", "style"]):
        tag.decompose()
    return " ".join(soup.get_text(" ", strip=True).split())


def clean_deal_url(raw_url):
    """Check the DealNews link and remove common tracking parameters."""
    parts = urlsplit(str(raw_url).strip())
    if (
        parts.scheme not in ("http", "https")
        or parts.hostname not in {"dealnews.com", "www.dealnews.com"}
        or parts.username
        or parts.password
    ):
        raise ValueError("The RSS entry needs an ordinary DealNews link.")
    query = [
        (key, value)
        for key, value in parse_qsl(parts.query)
        if key != "iref" and not key.startswith("utm_")
    ]
    return urlunsplit((parts.scheme, parts.netloc.lower(), parts.path, urlencode(query), ""))


def fetch_rss_deals(limit_per_feed=5, session=None):
    """Return a list of entry dictionaries plus any source warnings.

    A dictionary has source, title, summary, and url. It has no numeric price
    yet: selecting clearly priced products is the next learning portion.
    The optional session lets checks supply saved RSS without using the web.
    """
    if not isinstance(limit_per_feed, int) or not 1 <= limit_per_feed <= 10:
        raise ValueError("limit_per_feed must be an integer from 1 to 10.")

    deals = []
    warnings = []
    seen_urls = set()
    readable_feeds = 0
    owns_session = session is None
    session = requests.Session() if owns_session else session

    try:
        for source, feed_url in FEEDS:
            try:
                response = session.get(
                    feed_url, headers=HEADERS, timeout=TIMEOUT, allow_redirects=False
                )
                response.raise_for_status()
                if 300 <= response.status_code < 400:
                    raise ValueError("The feed returned a redirect.")
                feed = feedparser.parse(response.content)
                if not feed.get("version") and not feed.entries:
                    raise ValueError("The source did not return an RSS feed.")
                if feed.get("bozo") and not feed.entries:
                    raise ValueError("The RSS feed could not be read.")
                readable_feeds += 1
            except (requests.RequestException, ValueError):
                warnings.append(f"{source}: the feed could not be read; it was skipped.")
                continue

            for entry in feed.entries[:limit_per_feed]:
                try:
                    title = clean_text(entry.get("title", ""))
                    url = clean_deal_url(entry.get("link", ""))
                    if not title:
                        raise ValueError("The entry has no title.")
                except (ValueError, TypeError):
                    warnings.append(f"{source}: an incomplete entry was skipped.")
                    continue

                if url in seen_urls:
                    continue
                summary = clean_text(entry.get("summary") or entry.get("description", ""))
                deals.append({
                    "source": source,
                    "title": title,
                    "summary": summary[:1500] or "No summary was supplied in this feed entry.",
                    "url": url,
                })
                seen_urls.add(url)

        if readable_feeds == 0:
            raise RuntimeError(
                "No DealNews feed could be read. Check your internet connection and try again. "
                "The source may also be temporarily unavailable."
            )
        return deals, warnings
    finally:
        if owns_session:
            session.close()
