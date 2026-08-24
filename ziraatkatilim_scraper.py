"""
Ziraat Katılım Kampanya Scraper (GÜNCELLENMİŞ - SÜRESİ DOLANLAR HARİÇ)
TEKNOFEST Türkçe Yapay Zeka Dil Ajanları Yarışması - Veri Toplama Modülü (2. Senaryo)

Kullanım:
    python ziraatkatilim_scraper.py

Çıktı:
    data/ziraatkatilim_kampanyalar.jsonl
"""

import json
import os
import time
from datetime import datetime
from urllib.parse import urljoin

from playwright.sync_api import sync_playwright, TimeoutError as PlaywrightTimeout

# ─── Konfigürasyon ───────────────────────────────────────────────────────────
BASE_URL = "https://www.ziraatkatilim.com.tr"
LISTE_URL = f"{BASE_URL}/kart-kampanyalari"
CIKTI_DIZINI = "data"
CIKTI_DOSYASI = os.path.join(CIKTI_DIZINI, "ziraatkatilim_kampanyalar.jsonl")

# ─── Düzeltilmiş Selector'lar ────────────────────────────────────────────────
# HATA DÜZELTİLDİ: "archived-item" (süresi dolmuş/gizli) sınıfına sahip olmayan kapsayıcıları seçiyoruz
KAMPANYA_KART_SELECTOR = "div.campaign-item-wrapper:not(.archived-item) div.campaign-item"

LISTE_BASLIK_SELECTOR = "div.front a.item-title"
LISTE_KATEGORI_SELECTOR = "div.front span.item-category"
LISTE_TARIH_SELECTOR = "div.front p.item-date span.campaign-date"

DETAY_BASLIK_SELECTOR = "h1.node-title"
DETAY_TARIH_SELECTOR = "div.campaign-period span.campaign-date"
DETAY_METIN_SELECTOR = "div.body-content"

# ─── Yardımcı Fonksiyonlar ───────────────────────────────────────────────────

def ensure_dir(path: str) -> None:
    os.makedirs(path, exist_ok=True)

def write_jsonl(filepath: str, record: dict) -> None:
    with open(filepath, "a", encoding="utf-8") as f:
        f.write(json.dumps(record, ensure_ascii=False) + "\n")

def extract_campaign_cards(page) -> list[dict]:
    cards = page.query_selector_all(KAMPANYA_KART_SELECTOR)
    results = []
    
    for card in cards:
        try:
            # Başlık ve Link
            title_el = card.query_selector(LISTE_BASLIK_SELECTOR)
            title = title_el.inner_text().strip() if title_el else None
            href = title_el.get_attribute("href") if title_el else None
            full_url = urljoin(BASE_URL, href) if href else None

            # Kategori / Etiket
            badge_el = card.query_selector(LISTE_KATEGORI_SELECTOR)
            badge = badge_el.inner_text().strip() if badge_el else None

            # Bitiş Tarihi
            date_el = card.query_selector(LISTE_TARIH_SELECTOR)
            date_text = date_el.inner_text().strip() if date_el else None
            
            # Ekstra güvenlik kontrolü: Eğer metin içinde "Sonlanmıştır" geçiyorsa yine de atla
            date_full_text = card.query_selector("div.front p.item-date").inner_text() if card.query_selector("div.front p.item-date") else ""
            if "Sonlanmıştır" in date_full_text:
                continue

            results.append({
                "liste_baslik": title,
                "liste_ozet": None, 
                "liste_url": full_url,
                "liste_bitis_tarihi": date_text,
                "liste_etiket": badge,
            })
        except Exception as e:
            print(f"  [Kart parse hatası] {e}")
            continue
            
    return results

def scrape_detail_page(page, url: str) -> dict:
    try:
        page.goto(url, wait_until="domcontentloaded", timeout=15000)
        time.sleep(1)

        title_el = page.query_selector(DETAY_BASLIK_SELECTOR)
        detail_title = title_el.inner_text().strip() if title_el else None
        
        date_el = page.query_selector(DETAY_TARIH_SELECTOR)
        detail_date = None
        if date_el:
            raw_date = date_el.inner_text().strip()
            detail_date = ' '.join(raw_date.split())

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
            "detay_tarih": detail_date
        }
    except Exception as e:
        print(f"  [Detay hatası: {url}] {e}")
        return {
            "detay_baslik": None,
            "detay_metin": None,
            "detay_html": None,
            "detay_tarih": None,
            "detay_hata": str(e),
        }

# ─── Ana Akış ────────────────────────────────────────────────────────────────

def main():
    ensure_dir(CIKTI_DIZINI)

    if os.path.exists(CIKTI_DOSYASI):
        os.remove(CIKTI_DOSYASI)

    print("=" * 60)
    print("Ziraat Katılım Kampanya Scraper başlatılıyor...")
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
        print(f"\n[1/2] Liste sayfası yükleniyor: {LISTE_URL}")
        page.goto(LISTE_URL, wait_until="domcontentloaded", timeout=30000)
        
        time.sleep(2)

        print("\n[2/2] Kampanya kartları toplanıyor...")
        all_cards = extract_campaign_cards(page)
        
        seen_urls = set()
        unique_cards = []
        for c in all_cards:
            url = c.get("liste_url")
            if url and url not in seen_urls:
                seen_urls.add(url)
                unique_cards.append(c)

        print(f"    Tekil aktif kampanya sayısı: {len(unique_cards)}")

        print("\nDetay sayfaları çekiliyor...")
        detail_page = context.new_page()

        for idx, card in enumerate(unique_cards, 1):
            url = card.get("liste_url")
            print(f"    [{idx}/{len(unique_cards)}] İşleniyor: {url}")
            detail_data = scrape_detail_page(detail_page, url)

            liste_bitis = detail_data.get("detay_tarih") if detail_data.get("detay_tarih") else card.get("liste_bitis_tarihi")

            record = {
                "kaynak": "ziraat_katilim",
                "banka_adi": "Ziraat Katılım",
                "liste_url": url,
                "liste_baslik": card.get("liste_baslik"),
                "liste_ozet": card.get("liste_ozet"),
                "liste_bitis_tarihi": liste_bitis, 
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
    print(f"İşlem Tamam! {len(unique_cards)} adet aktif kampanya data klasörüne kaydedildi.")
    print("=" * 60)

if __name__ == "__main__":
    main()