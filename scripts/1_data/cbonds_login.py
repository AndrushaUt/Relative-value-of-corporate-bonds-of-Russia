"""
Save Cbonds session cookies for headless scraping.

Opens a browser window — log in manually, then press Enter in the terminal.
Cookies are saved to data/cbonds_cookies.json for use by other scripts.

Usage:
    python scripts/cbonds_login.py
"""

import json
from pathlib import Path
from playwright.sync_api import sync_playwright

COOKIES_FILE = Path(__file__).resolve().parents[2] / "data" / "cbonds_cookies.json"


def main():
    with sync_playwright() as p:
        browser = p.chromium.launch(headless=False)
        context = browser.new_context()
        page = context.new_page()

        page.goto("https://cbonds.ru/")
        print("Browser opened. Log in to Cbonds, then press Enter here...")
        input()

        cookies = context.cookies()
        COOKIES_FILE.parent.mkdir(parents=True, exist_ok=True)
        COOKIES_FILE.write_text(json.dumps(cookies, ensure_ascii=False, indent=2))
        print(f"Saved {len(cookies)} cookies to {COOKIES_FILE}")

        browser.close()


if __name__ == "__main__":
    main()
