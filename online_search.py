# online_search.py
# NAIF Internet Search Module
# Nitish Artificial Intelligence Friend

from ddgs import DDGS
import requests
from bs4 import BeautifulSoup

# ============================================================
# SEARCH WEB (DuckDuckGo)
# ============================================================

def search_web(query, max_results=5):
    """
    Searches DuckDuckGo and returns list of URLs.
    """

    urls = []
    seen = set()

    try:
        with DDGS() as ddgs:

            results = ddgs.text(
                query,
                max_results=max_results
            )

            for r in results:

                url = r.get("href")

                if url and url not in seen:
                    urls.append(url)
                    seen.add(url)

    except Exception as e:
        print("Search error:", e)

    print("DEBUG: URLs found:", urls)

    return urls


# ============================================================
# EXTRACT TEXT FROM WEBPAGE
# ============================================================

def extract_text(url, max_chars=2000):
    """
    Extracts clean readable text from webpage.
    """

    try:

        print("DEBUG: Extracting from:", url)

        headers = {
            "User-Agent": (
                "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
                "AppleWebKit/537.36 (KHTML, like Gecko) "
                "Chrome/120.0 Safari/537.36"
            )
        }

        response = requests.get(
            url,
            headers=headers,
            timeout=10
        )

        print("DEBUG: Status code:", response.status_code)

        if response.status_code != 200:
            return ""

        soup = BeautifulSoup(response.text, "html.parser")

        # Remove junk elements
        for tag in soup([
            "script",
            "style",
            "nav",
            "footer",
            "header",
            "aside",
            "noscript"
        ]):
            tag.decompose()

        paragraphs = soup.find_all("p")

        text_parts = []

        for p in paragraphs:

            text = p.get_text().strip()

            if len(text) > 40:
                text_parts.append(text)

        if not text_parts:
            print("DEBUG: No usable paragraphs")
            return ""

        combined = " ".join(text_parts)

        # Clean whitespace
        combined = " ".join(combined.split())

        print("DEBUG: Extracted length:", len(combined))

        return combined[:max_chars]

    except Exception as e:
        print("Extraction error:", e)
        return ""


# ============================================================
# COMPLETE SEARCH PIPELINE
# ============================================================

def search_and_extract(query, max_results=5):
    """
    Full pipeline:
    1. Search web
    2. Extract text
    3. Combine results
    """

    urls = search_web(query, max_results)

    if not urls:
        return ""

    combined_text = ""

    for url in urls:

        text = extract_text(url)

        if text:

            combined_text += (
                f"\nSource: {url}\n"
                f"{text}\n"
            )

    if combined_text.strip() == "":
        print("DEBUG: No text extracted from any site")
        return ""

    print("DEBUG: Total context length:", len(combined_text))

    return combined_text[:4000]


# ============================================================
# INTERNET CHECK
# ============================================================

def internet_available():
    """
    Checks if internet is available.
    """

    try:
        requests.get("https://duckduckgo.com", timeout=3)
        return True

    except:
        return False


# ============================================================
# TEST BLOCK
# ============================================================

if __name__ == "__main__":

    query = "next Batman movie"

    print("\nTesting internet search...\n")

    if internet_available():

        context = search_and_extract(query)

        print("\nFINAL RESULT:\n")
        print(context)

    else:

        print("No internet connection.")