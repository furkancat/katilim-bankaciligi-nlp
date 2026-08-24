"""
Türkiye Finans Kampanya Scraper (GÜNCELLENMİŞ)
TEKNOFEST Türkçe Yapay Zeka Dil Ajanları Yarışması - Veri Toplama Modülü (2. Senaryo)

Kullanım:
    python turkiyefinans_scraper.py

Çıktı:
    data/turkiyefinans_kampanyalar.jsonl
"""

import json
import os
import time
from datetime import datetime
from urllib.parse import urljoin

from playwright.sync_api import sync_playwright, TimeoutError as PlaywrightTimeout

# ─── Konfigürasyon ───────────────────────────────────────────────────────────
BASE_URL = "https://www.turkiyefinans.com.tr"
KATEGORI_URL = f"{BASE_URL}/tr-tr/kampanyalar/Sayfalar/default.aspx"
CIKTI_DIZINI = "data"
CIKTI_DOSYASI = os.path.join(CIKTI_DIZINI, "turkiyefinans_kampanyalar.jsonl")

# ─── Selector'lar ────────────────────────────────────────────────────────────
KATEGORI_BTN_SELECTOR = "div.box .hover a"
KAMPANYA_KART_SELECTOR = "div.campaign"
DETAY_BASLIK_SELECTOR = "h1#content-title"
# Hem class'ı hem de senin belirttiğin ID formatını yakalamak için daha esnek bir seçici:
DETAY_METIN_SELECTOR = "div.ms-rtestate-field, div[id$='_RichHtmlField']" 

# ─── Yardımcı Fonksiyonlar ───────────────────────────────────────────────────

def ensure_dir(path: str) -> None:
    os.makedirs(path, exist_ok=True)

def write_jsonl(filepath: str, record: dict) -> None:
    with open(filepath, "a", encoding="utf-8") as f:
        f.write(json.dumps(record, ensure_ascii=False) + "\n")

def get_category_urls(page) -> list[str]:
    urls = set()
    links = page.query_selector_all(KATEGORI_BTN_SELECTOR)
    for link in links:
        try:
            text = link.inner_text().strip()
            if "Detaylı Bilgi" in text:
                href = link.get_attribute("href")
                if href and "javascript" not in href:
                    urls.add(urljoin(BASE_URL, href))
        except Exception:
            pass
    return list(urls)

def extract_campaign_cards(page) -> list[dict]:
    cards = page.query_selector_all(KAMPANYA_KART_SELECTOR)
    results = []
    
    for card in cards:
        try:
            title_el = card.query_selector("h2 a")
            title = title_el.inner_text().strip() if title_el else None
            href = title_el.get_attribute("href") if title_el else None
            full_url = urljoin(BASE_URL, href) if href else None

            summary_el = card.query_selector("p a:first-child")
            summary = summary_el.inner_text().strip() if summary_el else None

            results.append({
                "liste_baslik": title,
                "liste_ozet": summary,
                "liste_url": full_url,
                "liste_bitis_tarihi": None,
                "liste_etiket": None,
            })
        except Exception as e:
            print(f"  [Kart parse hatası] {e}")
            continue
            
    return results

def scrape_detail_page(page, url: str) -> dict:
    try:
        page.goto(url, wait_until="domcontentloaded", timeout=15000)
        time.sleep(1)

        # Başlık
        title_el = page.query_selector(DETAY_BASLIK_SELECTOR)
        detail_title = title_el.inner_text().strip() if title_el else None

        # Sayfadaki sr-only ve display:none olan divleri uçuralım
        page.evaluate("""
            document.querySelectorAll('.sr-only, [style*="display:none"], [style*="display: none"]').forEach(el => el.remove());
        """)

        # DÜZELTME: query_selector yerine query_selector_all kullanıyoruz
        content_elements = page.query_selector_all(DETAY_METIN_SELECTOR)
        
        if content_elements:
            raw_text_list = []
            html_list = []
            
            # Sayfadaki tüm içerik div'lerini gezip birleştiriyoruz
            for el in content_elements:
                text = el.inner_text().strip()
                html = el.inner_html()
                if text or "<img" in html.lower(): # Boş div'leri atla
                    raw_text_list.append(text)
                    html_list.append(html)
            
            raw_text = "\n\n".join(raw_text_list)
            detail_html = "\n".join(html_list)
            
            clean_text = raw_text.replace('\u200b', '').replace('\xa0', ' ').strip()
            clean_text = ' '.join(clean_text.split())
            
            # Eğer tüm birleştirilmiş metinlerin toplamı 30 karakterden kısaysa ve görsel varsa
            if len(clean_text) < 30 and "<img" in detail_html.lower():
                img_src = ""
                for el in content_elements:
                    img_el = el.query_selector("img")
                    if img_el:
                        img_src = img_el.get_attribute("src") or ""
                        break
                
                if img_src and not img_src.startswith("http"):
                    img_src = urljoin(BASE_URL, img_src)
                    
                detail_text = f"[BU KAMPANYA SADECE GÖRSELDEN OLUŞMAKTADIR - METİN YOK] Afiş Linki: {img_src}"
            else:
                detail_text = raw_text.replace('\u200b', '').strip()

        else:
            detail_text = None
            detail_html = None

        return {
            "detay_baslik": detail_title,
            "detay_metin": detail_text,
            "detay_html": detail_html,
        }
    except Exception as e:
        print(f"  [Detay hatası: {url}] {e}")
        return {
            "detay_baslik": None,
            "detay_metin": None,
            "detay_html": None,
            "detay_hata": str(e),
        }

# ─── Ana Akış ────────────────────────────────────────────────────────────────

def main():
    ensure_dir(CIKTI_DIZINI)

    if os.path.exists(CIKTI_DOSYASI):
        os.remove(CIKTI_DOSYASI)

    print("=" * 60)
    print("Türkiye Finans Kampanya Scraper başlatılıyor...")
    print("=" * 60)

    with sync_playwright() as p:
        browser = p.chromium.launch(headless=True)
        context = browser.new_context(
            viewport={"width": 1280, "height": 800},
            user_agent=(
                "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
                "AppleWebKit/537.36 (KHTML, like Gecko) "
                "Chrome/128.0.0.0 Safari/537.36"
            ),
        )
        page = context.new_page()

        print(f"\n[1/3] Kampanya Kategorileri taranıyor: {KATEGORI_URL}")
        page.goto(KATEGORI_URL, wait_until="domcontentloaded", timeout=30000)
        time.sleep(2)
        
        category_urls = get_category_urls(page)
        print(f"    Toplam {len(category_urls)} farklı kampanya kategorisi bulundu.")

        print("\n[2/3] Kategorilerin içindeki kampanyalar toplanıyor...")
        all_unique_cards = {}
        
        for idx, cat_url in enumerate(category_urls, 1):
            print(f"    -> Kategori [{idx}/{len(category_urls)}]: {cat_url}")
            try:
                page.goto(cat_url, wait_until="domcontentloaded", timeout=15000)
                time.sleep(1.5)
                cards = extract_campaign_cards(page)
                
                for c in cards:
                    url = c.get("liste_url")
                    if url and url not in all_unique_cards:
                        all_unique_cards[url] = c
            except Exception as e:
                print(f"       Kategori yüklenirken hata: {e}")

        unique_cards_list = list(all_unique_cards.values())
        print(f"    Farklı kategorilerden tekil toplam kampanya sayısı: {len(unique_cards_list)}")

        print("\n[3/3] Detay sayfaları çekiliyor...")
        detail_page = context.new_page()

        for idx, card in enumerate(unique_cards_list, 1):
            url = card.get("liste_url")
            print(f"    [{idx}/{len(unique_cards_list)}] İşleniyor: {url}")
            detail_data = scrape_detail_page(detail_page, url)

            record = {
                "kaynak": "turkiye_finans",
                "banka_adi": "Türkiye Finans",
                "liste_url": url,
                "liste_baslik": card.get("liste_baslik"),
                "liste_ozet": card.get("liste_ozet"),
                "liste_bitis_tarihi": card.get("liste_bitis_tarihi"), 
                "liste_etiket": card.get("liste_etiket"),
                "detay_baslik": detail_data.get("detay_baslik"),
                "detay_metin": detail_data.get("detay_metin"),
                "detay_html": detail_data.get("detay_html"),
                "detay_hata": detail_data.get("detay_hata"),
                "scraped_at": datetime.now().isoformat(),
            }
            write_jsonl(CIKTI_DOSYASI, record)

        detail_page.close()
        page.close()
        browser.close()

    print("\n" + "=" * 60)
    print(f"İşlem Tamam! {len(unique_cards_list)} adet kampanya data klasörüne kaydedildi.")
    print("=" * 60)

if __name__ == "__main__":
    main()