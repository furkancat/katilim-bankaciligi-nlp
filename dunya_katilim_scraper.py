"""
Dünya Katılım Bankası Kampanya Scraper
TEKNOFEST Türkçe Yapay Zeka Dil Ajanları Yarışması - Veri Toplama Modülü (2. Senaryo)

Kullanım:
    python dunya_katilim_scraper.py

Çıktı:
    data/dunya_katilim_kampanyalar.jsonl
"""

import json
import os
import time
from datetime import datetime
from urllib.parse import urljoin

from playwright.sync_api import sync_playwright, TimeoutError as PlaywrightTimeout

# ─── Konfigürasyon ───────────────────────────────────────────────────────────
BASE_URL = "https://dunyakatilim.com.tr"
LISTE_URL = f"{BASE_URL}/kampanyalar"
CIKTI_DIZINI = "data"
CIKTI_DOSYASI = os.path.join(CIKTI_DIZINI, "dunya_katilim_kampanyalar.jsonl")

# Selector'lar (Verilen HTML yapısına göre)
KAMPANYA_KART_SELECTOR = "div.page-campaigns-content-item"
DAHA_FAZLA_BTN_SELECTOR = "#moreCampaigns"
DETAY_BASLIK_SELECTOR = "h1.campaign-detail-header-left-title"
DETAY_METIN_SELECTOR = "div.page-content-list .page-left-content"

# ─── Yardımcı Fonksiyonlar ───────────────────────────────────────────────────

def ensure_dir(path: str) -> None:
    os.makedirs(path, exist_ok=True)

def write_jsonl(filepath: str, record: dict) -> None:
    with open(filepath, "a", encoding="utf-8") as f:
        f.write(json.dumps(record, ensure_ascii=False) + "\n")

def extract_campaign_cards(page) -> list[dict]:
    """
    DOM'daki kampanya kartlarını çeker.
    """
    cards = page.query_selector_all(KAMPANYA_KART_SELECTOR)
    results = []
    for card in cards:
        try:
            # Başlık
            title_el = card.query_selector("h3")
            title = title_el.inner_text().strip() if title_el else None

            # Link (Alt kısımdaki kampanya detay linkinden)
            link_el = card.query_selector("a.link")
            href = link_el.get_attribute("href") if link_el else None
            full_url = urljoin(BASE_URL, href) if href else None

            # Özet metin
            summary_el = card.query_selector("div.desc")
            summary = summary_el.inner_text().strip() if summary_el else None

            # Bitiş tarihi
            date_el = card.query_selector("div.date")
            date_text = date_el.inner_text().replace("Bitiş Tarihi:", "").strip() if date_el else None

            # Etiket / Kategori
            badge_el = card.query_selector("div.status-box .status")
            badge = badge_el.inner_text().strip() if badge_el else None

            results.append({
                "liste_baslik": title,
                "liste_ozet": summary,
                "liste_url": full_url,
                "liste_bitis_tarihi": date_text,
                "liste_etiket": badge,
            })
        except Exception as e:
            print(f"  [Kart parse hatası] {e}")
            continue
    return results

def click_load_more(page) -> bool:
    """
    'Daha Fazla' butonuna tıklar.
    display: none olduğunda Playwright'ın is_visible() metodu False döner.
    """
    try:
        btn = page.query_selector(DAHA_FAZLA_BTN_SELECTOR)
        if not btn:
            return False
        
        # Buton HTML'de kalsa bile CSS ile gizlenmişse (display:none) is_visible() False verir.
        if not btn.is_visible():
            return False
            
        btn.click()
        time.sleep(1.5) # AJAX yüklemesi için tolerans
        return True
    except PlaywrightTimeout:
        return False
    except Exception as e:
        print(f"  [Buton tıklama hatası] {e}")
        return False

def scrape_detail_page(page, url: str) -> dict:
    """
    Detay sayfasına girip başlık ve tam metni çeker.
    """
    try:
        page.goto(url, wait_until="domcontentloaded", timeout=15000)
        time.sleep(1)

        # Başlık
        title_el = page.query_selector(DETAY_BASLIK_SELECTOR)
        detail_title = title_el.inner_text().strip() if title_el else None

        # İçerik Metni ve HTML
        content_el = page.query_selector(DETAY_METIN_SELECTOR)
        if content_el:
            detail_text = content_el.inner_text().strip()
            detail_html = content_el.inner_html()
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
    print("Dünya Katılım Kampanya Scraper başlatılıyor...")
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
        print(f"\n[1/3] Liste sayfası yükleniyor: {LISTE_URL}")
        page.goto(LISTE_URL, wait_until="domcontentloaded", timeout=30000)
        time.sleep(2)

        print("[2/3] 'Daha Fazla' butonuna tıklanıyor...")
        click_count = 0
        while True:
            success = click_load_more(page)
            if not success:
                print(f"    Daha fazla içerik kalmadı. Toplam {click_count} tıklama yapıldı.")
                break
            click_count += 1
            print(f"    Tıklama #{click_count} başarılı.")

        print("\n[3/3] Kampanya kartları toplanıyor...")
        all_cards = extract_campaign_cards(page)
        
        seen_urls = set()
        unique_cards = []
        for c in all_cards:
            url = c.get("liste_url")
            if url and url not in seen_urls:
                seen_urls.add(url)
                unique_cards.append(c)

        print(f"    Tekil kampanya sayısı: {len(unique_cards)}")

        print("\nDetay sayfaları çekiliyor...")
        detail_page = context.new_page()

        for idx, card in enumerate(unique_cards, 1):
            url = card.get("liste_url")
            print(f"    [{idx}/{len(unique_cards)}] İşleniyor: {url}")
            detail_data = scrape_detail_page(detail_page, url)

            record = {
                "kaynak": "dunya_katilim",
                "banka_adi": "Dünya Katılım Bankası",
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
    print(f"İşlem Tamam! Toplam {len(unique_cards)} adet kampanya verisi PostgreSQL'e veya LLM'e beslenmek üzere JSONL formatında kaydedildi.")
    print("=" * 60)

if __name__ == "__main__":
    main()