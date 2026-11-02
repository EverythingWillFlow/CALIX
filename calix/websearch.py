"""Shared web-access tools for every agent and every baseline.

CALIX and all baselines receive **the same** tool object, so any measured
difference comes from the coordination method rather than from unequal tool
access.  A per-task call budget is enforced by the toolkit itself; when the
budget is exhausted the tool returns an explicit notice instead of a result.

Backends are tried in order and require no third-party package:

- keyed : Tavily / Serper / Brave  (used when the matching env var is set)
- keyless: DuckDuckGo HTML endpoint, Wikipedia search API (fallback)

Page fetching uses a dependency-free HTML-to-text conversion.
"""
from __future__ import annotations

import html
import json
import os
import re
import threading
import time
import urllib.error
import urllib.parse
import urllib.request

_UA = ("Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 "
       "(KHTML, like Gecko) Chrome/124.0 Safari/537.36")

_TAG_RE = re.compile(r"<[^>]+>")
_WS_RE = re.compile(r"[ \t\r\f\v]+")
_NL_RE = re.compile(r"\n{3,}")
_SCRIPT_RE = re.compile(r"<(script|style|noscript)[^>]*>.*?</\1>",
                        re.IGNORECASE | re.DOTALL)
_DDG_LINK_RE = re.compile(
    r'<a[^>]*class="result__a"[^>]*href="([^"]+)"[^>]*>(.*?)</a>',
    re.IGNORECASE | re.DOTALL)
_DDG_SNIP_RE = re.compile(
    r'class="result__snippet"[^>]*>(.*?)</a>', re.IGNORECASE | re.DOTALL)

TOOL_SPEC = """\
You have access to two web tools. To use one, emit the call alone on a line:

SEARCH[<your search query>]   -> returns up to {k} web results (title, url, snippet)
FETCH[<url>]                  -> returns the main text of that page (truncated)

Rules:
- Emit at most one tool call per turn, then stop.
- The result is returned to you as OBSERVATION[...]. You may then emit another
  tool call or give your final answer.
- Never invent a URL. Only FETCH urls that appeared in a previous observation.
- You have a limited number of tool calls; use them on what you cannot answer
  from your own knowledge.
"""


class WebToolkit:
    """Search + page fetch with caching, budgeting and usage accounting."""

    def __init__(self, max_search_calls: int = 8, max_fetch_calls: int = 6,
                 results_per_search: int = 5, max_doc_chars: int = 4000,
                 timeout: int = 25, cache_dir: str | None = None,
                 enabled: bool = True):
        self.max_search_calls = int(max_search_calls)
        self.max_fetch_calls = int(max_fetch_calls)
        self.results_per_search = int(results_per_search)
        self.max_doc_chars = int(max_doc_chars)
        self.timeout = int(timeout)
        self.enabled = bool(enabled)
        self.cache_dir = cache_dir
        self._cache: dict[str, str] = {}
        if cache_dir:
            os.makedirs(cache_dir, exist_ok=True)
            self._cache_path = os.path.join(cache_dir, "webcache.json")
            if os.path.exists(self._cache_path):
                try:
                    with open(self._cache_path, "r", encoding="utf-8") as f:
                        self._cache = json.load(f)
                except (OSError, json.JSONDecodeError):
                    self._cache = {}
        else:
            self._cache_path = None
        self._lock = threading.Lock()
        self.reset_usage()
        self.reset_budget()

    def clone(self) -> "WebToolkit":
        """Independent counters, shared result cache.

        Used to give every parallel worker its own per-task budget while
        keeping a single HTTP cache (identical results, no duplicate traffic).
        """
        other = WebToolkit.__new__(WebToolkit)
        other.max_search_calls = self.max_search_calls
        other.max_fetch_calls = self.max_fetch_calls
        other.results_per_search = self.results_per_search
        other.max_doc_chars = self.max_doc_chars
        other.timeout = self.timeout
        other.enabled = self.enabled
        other.cache_dir = self.cache_dir
        other._cache = self._cache
        other._cache_path = self._cache_path
        other._lock = self._lock
        other.reset_usage()
        other.reset_budget()
        return other

    # -- accounting ---------------------------------------------------------
    def reset_budget(self) -> None:
        self.searches_used = 0
        self.fetches_used = 0

    def reset_usage(self) -> None:
        self.searches = 0
        self.fetches = 0
        self.cache_hits = 0
        self.errors = 0
        self.search_chars = 0
        self.fetch_chars = 0

    def usage(self) -> dict:
        return {
            "search_calls": self.searches,
            "fetch_calls": self.fetches,
            "cache_hits": self.cache_hits,
            "errors": self.errors,
            "search_chars": self.search_chars,
            "fetch_chars": self.fetch_chars,
        }

    def budget_left(self) -> dict:
        return {"search": self.max_search_calls - self.searches_used,
                "fetch": self.max_fetch_calls - self.fetches_used}

    # -- low level ----------------------------------------------------------
    def _get(self, url: str, headers: dict | None = None,
             data: bytes | None = None) -> tuple[int, bytes]:
        req = urllib.request.Request(url, data=data,
                                     headers={**{"User-Agent": _UA},
                                              **(headers or {})})
        try:
            with urllib.request.urlopen(req, timeout=self.timeout) as resp:
                return resp.status, resp.read()
        except urllib.error.HTTPError as e:
            return e.code, e.read()
        except Exception as e:  # network-level failure
            self.errors += 1
            return -1, f"{type(e).__name__}: {e}".encode("utf-8")

    @staticmethod
    def _html_to_text(raw: str) -> str:
        raw = _SCRIPT_RE.sub(" ", raw)
        # prefer the main content container when the page marks one
        for pat in (r"<main[^>]*>(.*?)</main>",
                    r"<article[^>]*>(.*?)</article>",
                    r"<div[^>]+id=[\"'](?:content|mw-content-text|main|bodyContent)[\"'][^>]*>(.*?)</div>"):
            m = re.search(pat, raw, re.IGNORECASE | re.DOTALL)
            if m and len(m.group(1)) > 500:
                raw = m.group(1)
                break
        raw = re.sub(r"<br\s*/?>", "\n", raw, flags=re.IGNORECASE)
        raw = re.sub(r"</(p|div|li|h[1-6]|tr|td|section)>", "\n", raw,
                     flags=re.IGNORECASE)
        text = _TAG_RE.sub(" ", raw)
        text = html.unescape(text).replace("\xa0", " ")
        lines = [ln.strip() for ln in text.splitlines()]
        lines = [ln for ln in lines if ln]
        return "\n".join(lines)

    def _cached(self, key: str) -> str | None:
        if key in self._cache:
            self.cache_hits += 1
            return self._cache[key]
        return None

    def _store(self, key: str, value: str) -> None:
        with self._lock:
            self._cache[key] = value
            if self._cache_path:
                try:
                    tmp = self._cache_path + ".tmp"
                    with open(tmp, "w", encoding="utf-8") as f:
                        json.dump(self._cache, f, ensure_ascii=False)
                    os.replace(tmp, self._cache_path)
                except OSError:
                    pass

    # -- backends -----------------------------------------------------------
    def _search_tavily(self, q: str) -> list[dict] | None:
        key = os.environ.get("TAVILY_API_KEY")
        if not key:
            return None
        body = json.dumps({"api_key": key, "query": q,
                           "max_results": self.results_per_search,
                           "search_depth": "basic"}).encode()
        code, raw = self._post_json("https://api.tavily.com/search", body)
        if code != 200:
            return None
        try:
            data = json.loads(raw.decode("utf-8", "replace"))
        except json.JSONDecodeError:
            return None
        return [{"title": r.get("title", ""), "url": r.get("url", ""),
                 "snippet": r.get("content", "")} for r in data.get("results", [])]

    def _post_json(self, url: str, body: bytes) -> tuple[int, bytes]:
        req = urllib.request.Request(
            url, data=body,
            headers={"Content-Type": "application/json", "User-Agent": _UA})
        try:
            with urllib.request.urlopen(req, timeout=self.timeout) as resp:
                return resp.status, resp.read()
        except urllib.error.HTTPError as e:
            return e.code, e.read()
        except Exception as e:
            self.errors += 1
            return -1, str(e).encode()

    def _search_serper(self, q: str) -> list[dict] | None:
        key = os.environ.get("SERPER_API_KEY")
        if not key:
            return None
        body = json.dumps({"q": q, "num": self.results_per_search}).encode()
        req = urllib.request.Request(
            "https://google.serper.dev/search", data=body,
            headers={"X-API-KEY": key, "Content-Type": "application/json",
                     "User-Agent": _UA})
        try:
            with urllib.request.urlopen(req, timeout=self.timeout) as resp:
                data = json.loads(resp.read().decode("utf-8", "replace"))
        except Exception:
            return None
        return [{"title": r.get("title", ""), "url": r.get("link", ""),
                 "snippet": r.get("snippet", "")}
                for r in data.get("organic", [])]

    def _search_brave(self, q: str) -> list[dict] | None:
        key = os.environ.get("BRAVE_API_KEY")
        if not key:
            return None
        url = ("https://api.search.brave.com/res/v1/web/search?q="
               + urllib.parse.quote(q) + f"&count={self.results_per_search}")
        code, raw = self._get(url, {"X-Subscription-Token": key,
                                    "Accept": "application/json"})
        if code != 200:
            return None
        try:
            data = json.loads(raw.decode("utf-8", "replace"))
        except json.JSONDecodeError:
            return None
        return [{"title": r.get("title", ""), "url": r.get("url", ""),
                 "snippet": r.get("description", "")}
                for r in data.get("web", {}).get("results", [])]

    @staticmethod
    def _ddg_unwrap(href: str) -> str:
        """DuckDuckGo wraps external links in /l/?uddg=<encoded>."""
        if "uddg=" in href:
            m = re.search(r"uddg=([^&]+)", href)
            if m:
                return urllib.parse.unquote(m.group(1))
        if href.startswith("//"):
            href = "https:" + href
        return html.unescape(href)

    def _search_ddg(self, q: str) -> list[dict]:
        url = "https://html.duckduckgo.com/html/?q=" + urllib.parse.quote(q)
        code, raw = self._get(url)
        if code != 200:
            return []
        page = raw.decode("utf-8", "replace")
        links = _DDG_LINK_RE.findall(page)
        snips = _DDG_SNIP_RE.findall(page)
        out = []
        for i, (href, title) in enumerate(links[:self.results_per_search]):
            snip = self._html_to_text(snips[i]) if i < len(snips) else ""
            out.append({"title": self._html_to_text(title),
                        "url": self._ddg_unwrap(href), "snippet": snip})
        return out

    def _search_wikipedia(self, q: str) -> list[dict]:
        url = ("https://en.wikipedia.org/w/api.php?action=query&list=search"
               "&srsearch=" + urllib.parse.quote(q)
               + f"&srlimit={self.results_per_search}&format=json")
        code, raw = self._get(url)
        if code != 200:
            return []
        try:
            data = json.loads(raw.decode("utf-8", "replace"))
        except json.JSONDecodeError:
            return []
        out = []
        for s in data.get("query", {}).get("search", []):
            title = s.get("title", "")
            out.append({
                "title": title,
                "url": "https://en.wikipedia.org/wiki/" + title.replace(" ", "_"),
                "snippet": self._html_to_text(s.get("snippet", "")),
            })
        return out

    def _run_search(self, q: str) -> list[dict]:
        for backend in (self._search_tavily, self._search_serper,
                        self._search_brave):
            res = backend(q)
            if res:
                return res
        res = self._search_ddg(q)
        if len(res) < 2:
            res += self._search_wikipedia(q)
        return res

    # -- public API ---------------------------------------------------------
    def search(self, query: str) -> str:
        """Return formatted search results, or an explicit failure notice."""
        if not self.enabled:
            return "TOOL ERROR: web tools are disabled."
        query = (query or "").strip()
        if not query:
            return "TOOL ERROR: empty search query."
        if self.searches_used >= self.max_search_calls:
            return ("TOOL NOTICE: the search budget for this task is exhausted "
                    f"({self.max_search_calls} calls). Continue with what you "
                    "already have.")
        key = "s:" + query.lower()
        cached = self._cached(key)
        # the budget is consumed by every invocation, cached or not, so that
        # the number of tool accesses is identical and predictable
        self.searches_used += 1
        if cached is None:
            self.searches += 1
            t0 = time.time()
            results = self._run_search(query)
            if not results:
                cached = (f"SEARCH[{query}] -> no results "
                          f"(backend error or empty response, {time.time()-t0:.1f}s)")
            else:
                lines = [f"SEARCH[{query}] -> {len(results)} results:"]
                for i, r in enumerate(results, 1):
                    lines.append(
                        f"{i}. {r['title']}\n   {r['url']}\n   {r['snippet'][:400]}")
                cached = "\n".join(lines)
            self._store(key, cached)
        self.search_chars += len(cached)
        return cached[: self.max_doc_chars]

    def fetch(self, url: str) -> str:
        """Return the main text of a page, or an explicit failure notice."""
        if not self.enabled:
            return "TOOL ERROR: web tools are disabled."
        url = (url or "").strip()
        if not url.startswith(("http://", "https://")):
            return "TOOL ERROR: url must start with http:// or https://"
        if self.fetches_used >= self.max_fetch_calls:
            return ("TOOL NOTICE: the page-fetch budget for this task is "
                    f"exhausted ({self.max_fetch_calls} calls). Continue with "
                    "what you already have.")
        key = "f:" + url
        cached = self._cached(key)
        self.fetches_used += 1
        if cached is None:
            self.fetches += 1
            raw = None
            code = -1
            wiki = re.match(r"https?://en\.wikipedia\.org/wiki/(.+)$", url)
            if wiki:
                # Wikipedia exposes a clean plaintext extract endpoint
                api = ("https://en.wikipedia.org/w/api.php?action=query"
                       "&prop=extracts&explaintext=1&redirects=1&format=json"
                       "&titles=" + urllib.parse.quote(
                           urllib.parse.unquote(wiki.group(1))))
                code, raw = self._get(api)
                if code == 200:
                    try:
                        data = json.loads(raw.decode("utf-8", "replace"))
                        pages = data.get("query", {}).get("pages", {})
                        body = "\n".join(
                            p.get("extract", "") for p in pages.values())
                        cached = f"FETCH[{url}] ->\n{body[: self.max_doc_chars]}"
                    except json.JSONDecodeError:
                        raw = None
            if cached is None:
                code, raw = self._get(url)
                if code != 200:
                    cached = f"FETCH[{url}] -> failed (HTTP {code})"
                else:
                    text = self._html_to_text(raw.decode("utf-8", "replace"))
                    cached = (f"FETCH[{url}] ->\n{text[: self.max_doc_chars]}"
                              if text.strip() else
                              f"FETCH[{url}] -> page returned no readable text")
            self._store(key, cached)
        self.fetch_chars += len(cached)
        return cached[: self.max_doc_chars + 200]

    # -- tool-call dispatch --------------------------------------------------
    _SEARCH_RE = re.compile(r"^\s*SEARCH\s*\[(.+?)\]\s*$", re.IGNORECASE)
    _FETCH_RE = re.compile(r"^\s*FETCH\s*\[(.+?)\]\s*$", re.IGNORECASE)

    def dispatch(self, text: str) -> str | None:
        """If ``text`` contains a tool call, execute it and return the result.

        Returns ``None`` when no tool call is present.
        """
        for line in text.splitlines():
            line = line.strip()
            m = self._SEARCH_RE.match(line)
            if m:
                return self.search(m.group(1))
            m = self._FETCH_RE.match(line)
            if m:
                return self.fetch(m.group(1))
        return None


def build_toolkit(cfg: dict) -> WebToolkit:
    """Construct the shared toolkit from a configuration mapping."""
    t = dict(cfg.get("tools", {}) or {})
    return WebToolkit(
        max_search_calls=t.get("max_search_calls", 8),
        max_fetch_calls=t.get("max_fetch_calls", 6),
        results_per_search=t.get("results_per_search", 5),
        max_doc_chars=t.get("max_doc_chars", 4000),
        timeout=t.get("timeout", 25),
        cache_dir=t.get("cache_dir"),
        enabled=bool(t.get("enabled", True)),
    )
