#!/usr/bin/env python3
"""Check every page's internal links/assets on the live findings site.

Stdlib only. Crawl the site's root pages, collect href/src targets, HEAD each
unique one, fail (exit 1) if any returns non-200. Used by CI (.github/workflows/ci.yml).
Usage: python tools/check_links.py [base_url]
"""
import re
import sys
import time
import urllib.parse
import urllib.request

BASE = (sys.argv[1] if len(sys.argv) > 1
        else "https://harsh74-2.github.io/N-body-surrogates/").rstrip("/") + "/"
PAGES = ["index.html", "main_interactive.html", "sandbox.html",
         "animations.html", "plots.html", "n10.html", "n25.html",
         "n50.html", "n100.html"]


def fetch(url, tries=3):
    for attempt in range(tries):
        req = urllib.request.Request(url, method="HEAD")
        try:
            return urllib.request.urlopen(req, timeout=20).status
        except urllib.error.HTTPError as e:
            # GitHub Pages throws transient 503s under load; retry those, not 4xx
            if e.code >= 500 and attempt < tries - 1:
                time.sleep(2)
                continue
            return e.code
        except Exception:
            time.sleep(2)
    return 503


def main():
    seen, bad = set(), []
    for page in PAGES:
        try:
            html = urllib.request.urlopen(BASE + page, timeout=20).read().decode("utf8", "ignore")
        except Exception:
            print(f"[FAIL] could not fetch page {page}")
            bad.append(f"{page}: page fetch failed")
            continue
        for link in set(re.findall(r'(?:href|src)=["\']([^"\']+)["\']', html)):
            if link.startswith(("http", "mailto:", "data:", "#")):
                continue
            url = urllib.parse.urljoin(BASE + page, link)
            if url in seen:
                continue
            seen.add(url)
            status = fetch(url)
            if status != 200:
                bad.append(f"{page} -> {status} {link}")
                print(f"[{status}] {page} -> {link}")
    print(f"checked {len(seen)} targets on {len(PAGES)} pages: "
          f"{'ALL OK' if not bad else f'{len(bad)} broken'}")
    return 1 if bad else 0


if __name__ == "__main__":
    sys.exit(main())