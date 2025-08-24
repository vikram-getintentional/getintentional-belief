import time
from fastapi import APIRouter, Query
import requests
from bs4 import BeautifulSoup
from urllib.parse import urljoin, urlparse
import tldextract
import re

router = APIRouter()

KEYWORDS = [
    "home", "product", "products", "feature", "features", "capabilities", "capability", 
    "solutions", "platform", "overview", "industry", "industries", "use-case", "use-cases",
    "customers", "customer", "case-study", "case-studies", "pricing", "plans"
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

CUSTOMER_PAGE_KEYWORDS = [
    "customer", "customers", "case-study", "case-studies", "usecase", "usecases",
    "customer-story", "customer-stories", "used-by", "trusted-by", "testimonial", "testimonials", "success-story", "success-stories"
]

def is_customer_page_link(href: str, anchor_text: str = "") -> bool:
    href = (href or "").lower()
    anchor_text = (anchor_text or "").lower()
    return any(
        kw in href or kw in anchor_text
        for kw in CUSTOMER_PAGE_KEYWORDS
    )
def collect_internal_links(base_url, homepage_html, domain_root, visited, max_links=100):
    soup = BeautifulSoup(homepage_html, "html.parser")
    all_links = [a.get("href") for a in soup.find_all("a")]
    internal_links = set()
    for href in all_links:
        full_url = urljoin(base_url, href)
        if is_internal_link(href, base_url, domain_root) and full_url not in visited:
            internal_links.add(full_url)
    return list(internal_links)[:max_links]

def extract_customer_stories_from_site(base_url, urls, visited):
    customer_stories = []
    for url in urls:
        if is_customer_page_link(url):  # Now checks URL directly
            try:
                res = requests.get(url, timeout=10, headers={"User-Agent": "Mozilla/5.0"})
                page_soup = BeautifulSoup(res.text, "html.parser")
                for section in page_soup.find_all(["section", "div", "article"]):
                    text = section.get_text(" ", strip=True)
                    if len(text) > 200:
                        customer_stories.append(text)
                visited.add(url)
                time.sleep(0.5)
            except Exception:
                continue
    # Deduplicate and filter
    seen = set()
    deduped = []
    for story in customer_stories:
        norm = re.sub(r"\W+", "", story.lower())
        if norm not in seen:
            seen.add(norm)
            deduped.append(story)
    return deduped[:10]

def find_review_links(soup: BeautifulSoup) -> dict:
    # Look for direct G2/Capterra links
    review_links = {}
    for a in soup.find_all("a", href=True):
        href = a["href"]
        if "g2.com/products" in href:
            review_links["g2"] = href
        elif "capterra.com/p/" in href:
            review_links["capterra"] = href
    return review_links

def guess_product_name(soup, url):
    # Try title tag
    title = soup.title.string if soup.title else ""
    if title:
        # Use the first word or phrase before a separator
        name = title.split("|")[0].split("-")[0].strip().split()[0]
        if name.isalpha():
            return name.lower()
    # Fallback to domain
    parsed = urlparse(url)
    domain = tldextract.extract(parsed.netloc).domain
    return domain.lower()

def search_g2_capterra(product_name, base_url, soup=None):
    # Try to guess G2 product review URL
    g2_url = f"https://www.g2.com/products/{product_name}/reviews"
    return {"g2": g2_url, "capterra": None}

def scrape_g2_capterra(url):
    # Placeholder: In production, use official APIs or scraping with care
    try:
        res = requests.get(url, timeout=10, headers={"User-Agent": "Mozilla/5.0"})
        soup = BeautifulSoup(res.text, "html.parser")
        # Extract review text, categories, and capabilities
        reviews = []
        categories = []
        capabilities = []
        # Example extraction logic (should be customized per site structure)
        for review in soup.find_all("div", class_=re.compile("review|testimonial")):
            text = review.get_text(" ", strip=True)
            if len(text) > 50:
                reviews.append(text)
        for cat in soup.find_all("a", href=True):
            if "/categories/" in cat["href"]:
                categories.append(cat.get_text(strip=True))
        for cap in soup.find_all("li", class_=re.compile("capability|feature")):
            capabilities.append(cap.get_text(strip=True))
        return {
            "reviews": reviews[:10],
            "categories": list(set(categories)),
            "listing_capabilities": list(set(capabilities))
        }
    except Exception:
        return {"reviews": [], "categories": [], "listing_capabilities": []}

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
    print("Starting website scrape for:", url)
    try:
        base_url = url if url.startswith("http") else f"https://{url}"
        parsed_base = urlparse(base_url)
        domain_root = f"{tldextract.extract(parsed_base.netloc).domain}.{tldextract.extract(parsed_base.netloc).suffix}"

        visited = set()
        collected_text = []
        all_footer_features = []
        plg_cta_flag = False
        customer_stories = []

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

        internal_links = list(set(internal_links))

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
        # Also look for customer/case study/testimonial pages
        customer_links = collect_internal_links(base_url, homepage_html, domain_root, visited)
        customer_stories = extract_customer_stories_from_site(base_url, customer_links, visited)

        seen_features = set()
        deduped_features = []
        for feat in all_footer_features:
            norm = re.sub(r"\W+", "", feat.lower())
            if norm not in seen_features:
                seen_features.add(norm)
                deduped_features.append(feat)

        final_text = "\n\n".join(collected_text)
        print("Scraping completed. Total pages scraped:", len(collected_text))
        # --- Review site scraping ---
        review_links = find_review_links(soup)
        product_name = guess_product_name(soup, url)
        if not review_links:
            review_links = search_g2_capterra(product_name, base_url)

        g2_data = scrape_g2_capterra(review_links.get("g2", ""))
        capterra_data = scrape_g2_capterra(review_links.get("capterra", ""))

        # Merge reviews, categories, and capabilities from both sources
        all_reviews = (g2_data.get("reviews", []) + capterra_data.get("reviews", []))[:20]
        all_categories = list(set(g2_data.get("categories", []) + capterra_data.get("categories", [])))
        all_listing_capabilities = list(set(g2_data.get("listing_capabilities", []) + capterra_data.get("listing_capabilities", [])))
        
        return {
            "text": final_text[:12000],
            "plg_cta_found": plg_cta_flag,
            "footer_features": deduped_features,
            "pages_scraped": len(collected_text),
            "customer_stories": customer_stories[:10],
            "reviews": all_reviews,
            "categories": all_categories,
            "listing_capabilities": all_listing_capabilities
        }

    except Exception as e:
        return {"error": str(e)}
