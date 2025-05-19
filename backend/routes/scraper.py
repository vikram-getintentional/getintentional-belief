from fastapi import APIRouter, Query
import requests
from bs4 import BeautifulSoup
from urllib.parse import urljoin, urlparse
import tldextract
import re

router = APIRouter()

KEYWORDS = [
    "home", "product", "products", "feature", "features",
    "solutions", "platform", "use-case", "use-cases",
    "customers", "case-studies", "pricing", "plans"
]

PLG_CTA_KEYWORDS = ["signup", "sign up", "free trial", "get started"]

def is_internal_link(href: str, base_url: str, domain_root: str) -> bool:
    if not href or href.startswith("#") or href.startswith("mailto:") or "javascript:void" in href:
        return False
    full_url = urljoin(base_url, href)
    parsed = urlparse(full_url)
    extracted = tldextract.extract(parsed.netloc)
    return f"{extracted.domain}.{extracted.suffix}" == domain_root

def extract_footer_features(soup: BeautifulSoup) -> list[str]:
    footer = soup.find("footer")
    features = []

    if not footer:
        return features

    # Look for headers like "Features", "Products", etc.
    target_keywords = ["feature", "product", "solution", "use case", "capabilities"]
    section_headers = footer.find_all(["h2", "h3", "strong"])

    for header in section_headers:
        heading_text = header.get_text(strip=True).lower()
        if any(kw in heading_text for kw in target_keywords):
            next_sibling = header.find_next_sibling()
            if next_sibling:
                if next_sibling.name == "ul":
                    items = [li.get_text(strip=True) for li in next_sibling.find_all("li")]
                    features.extend(items)
                elif next_sibling.name == "div":
                    links = [a.get_text(strip=True) for a in next_sibling.find_all("a")]
                    features.extend(links)

    return [f for f in features if len(f) > 3]

def clean_and_extract_text(html: str) -> tuple[str, bool, list[str]]:
    soup = BeautifulSoup(html, "html.parser")

    full_page_text = soup.get_text(" ", strip=True).lower()
    plg_cta_found = any(kw in full_page_text for kw in PLG_CTA_KEYWORDS)

    tags = soup.find_all(["h1", "h2", "h3", "p", "li"])
    lines = [tag.get_text(strip=True) for tag in tags if tag.get_text(strip=True)]
    lines = [line for line in lines if len(line) > 3]

    cta_keywords = ["book a demo", "contact us", "login", "sign up", "subscribe", "request", "start now"]
    cleaned_lines = [
        line for line in lines
        if not any(kw.lower() in line.lower() for kw in cta_keywords)
    ]

    seen = set()
    deduped = []
    for line in cleaned_lines:
        norm = re.sub(r"\W+", "", line.lower())
        if norm not in seen:
            seen.add(norm)
            deduped.append(line)

    footer_features = extract_footer_features(soup)

    return "\n".join(deduped), plg_cta_found, footer_features

@router.get("/scrape", tags=["Scraper"])
def scrape_website(url: str = Query(..., description="Website to scrape")):
    try:
        base_url = url if url.startswith("http") else f"https://{url}"
        parsed_base = urlparse(base_url)
        domain_root = f"{tldextract.extract(parsed_base.netloc).domain}.{tldextract.extract(parsed_base.netloc).suffix}"

        visited = set()
        collected_text = []
        all_footer_features = []
        plg_cta_flag = False

        # Scrape homepage
        homepage_res = requests.get(base_url, timeout=10, headers={"User-Agent": "Mozilla/5.0"})
        homepage_html = homepage_res.text
        homepage_text, homepage_plg, homepage_footer = clean_and_extract_text(homepage_html)
        plg_cta_flag = homepage_plg
        all_footer_features.extend(homepage_footer)
        collected_text.append(homepage_text)
        visited.add(base_url)

        # Find internal links
        soup = BeautifulSoup(homepage_html, "html.parser")
        all_links = [a.get("href") for a in soup.find_all("a")]
        internal_links = []

        for href in all_links:
            full_url = urljoin(base_url, href)
            if is_internal_link(href, base_url, domain_root) and full_url not in visited:
                if any(keyword in full_url.lower() for keyword in KEYWORDS):
                    internal_links.append(full_url)

        internal_links = list(set(internal_links))[:8]

        for link in internal_links:
            visited.add(link)
            try:
                res = requests.get(link, timeout=10, headers={"User-Agent": "Mozilla/5.0"})
                text, found_plg, footer = clean_and_extract_text(res.text)
                plg_cta_flag = plg_cta_flag or found_plg
                all_footer_features.extend(footer)
                collected_text.append(text)
            except Exception:
                continue

        seen_features = set()
        deduped_features = []
        for feat in all_footer_features:
            norm = re.sub(r"\W+", "", feat.lower())
            if norm not in seen_features:
                seen_features.add(norm)
                deduped_features.append(feat)

        final_text = "\n\n".join(collected_text)
        return {
            "text": final_text[:12000],
            "plg_cta_found": plg_cta_flag,
            "footer_features": deduped_features,
            "pages_scraped": len(collected_text)
        }

    except Exception as e:
        return {"error": str(e)}
