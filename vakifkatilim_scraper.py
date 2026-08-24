"""
Vakıf Katılım Kampanya Scraper
TEKNOFEST Türkçe Yapay Zeka Dil Ajanları Yarışması - Veri Toplama Modülü (2. Senaryo)

Kullanım:
    python vakifkatilim_scraper.py

Çıktı:
    data/vakifkatilim_kampanyalar.jsonl
"""

import json
import os
import time
from datetime import datetime
from urllib.parse import urljoin

from playwright.sync_api import sync_playwright, TimeoutError as PlaywrightTimeout

# ─── Konfigürasyon ───────────────────────────────────────────────────────────
BASE_URL = "https://www.vakifkatilim.com.tr"
LISTE_URL = f"{BASE_URL}/tr/kendim-icin/kampanyalar"
CIKTI_DIZINI = "data"
CIKTI_DOSYASI = os.path.join(CIKTI_DIZINI, "vakifkatilim_kampanyalar.jsonl")

# ─── Selector'lar ────────────────────────────────────────────────────────────
KAMPANYA_KART_SELECTOR = "a.card.card-md"
DAHA_FAZLA_BTN_SELECTOR = "#load-more-btn"
DETAY_BASLIK_SELECTOR = "div.hero-content h1"
DETAY_TARIH_SELECTOR = "div.hero-content div.text-color-primary b"
# Detay sayfasında hem Kampanya Detayları hem de Kampanya Şartları bölümlerini alacağız:
DETAY_SECTIONS = [
    "#kampanya-detaylari .col-lg-8", 
    "#kampanya-sartlari .col-lg-8 .mask-area"
]

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
            title_el = card.query_selector("h4.card-title")
            title = title_el.inner_text().strip() if title_el else None

            # Link doğrudan a etiketinin kendisinde
            href = card.get_attribute("href")
            full_url = urljoin(BASE_URL, href) if href else None

            results.append({
                "liste_baslik": title,
                "liste_ozet": None,
                "liste_url": full_url,
                "liste_bitis_tarihi": None, # Detay sayfasından alacağız
                "liste_etiket": None,
            })
        except Exception as e:
            print(f"  [Kart parse hatası] {e}")
            continue
            
    return results

def click_load_more(page) -> bool:
    """
    'Daha Fazla Kampanya Gör' butonuna tıklar.
    Buton d-none class'ı aldığında is_visible() False döner.
    """
    try:
        btn = page.query_selector(DAHA_FAZLA_BTN_SELECTOR)
        if not btn: 
            return False
            
        if not btn.is_visible():
            return False
            
        btn.click()
        time.sleep(1.5)
        return True
    except PlaywrightTimeout:
        return False
    except Exception as e:
        print(f"  [Buton tıklama hatası] {e}")
        return False

def scrape_detail_page(page, url: str) -> dict:
    """
    Detay sayfasına girip başlık, tarih ve tüm içeriği (şartlar dahil) çeker.
    """
    try:
        page.goto(url, wait_until="domcontentloaded", timeout=15000)
        time.sleep(1)

        # Başlık
        title_el = page.query_selector(DETAY_BASLIK_SELECTOR)
        detail_title = title_el.inner_text().strip() if title_el else None

        # Tarih
        date_el = page.query_selector(DETAY_TARIH_SELECTOR)
        detail_date = date_el.inner_text().strip() if date_el else None

        # İçerikteki "Tümünü Göster" vb. gereksiz UI butonlarını silelim
        page.evaluate("""
            document.querySelectorAll('button.mask-area-open-btn').forEach(el => el.remove());
        """)

        raw_texts = []
        html_contents = []

        # Hem detaylar hem de şartlar bölümlerini gezip birleştiriyoruz
        for section_selector in DETAY_SECTIONS:
            content_el = page.query_selector(section_selector)
            if content_el:
                text = content_el.inner_text().strip()
                html = content_el.inner_html()
                if text:
                    raw_texts.append(text)
                    html_contents.append(html)

        if raw_texts:
            # Birden fazla bölüm varsa aralarına boşluk bırakarak birleştir
            detail_text = "\n\n".join(raw_texts)
            detail_html = "\n".join(html_contents)
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
    print("Vakıf Katılım Kampanya Scraper başlatılıyor...")
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

        print("[2/3] 'Daha Fazla Kampanya Gör' butonuna tıklanıyor...")
        click_count = 0
        while True:
            success = click_load_more(page)
            if not success:
                print(f"    Daha fazla içerik kalmadı veya buton d-none oldu. Toplam {click_count} tıklama yapıldı.")
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

            # Vakıf Katılım'da tarih bilgisini detay sayfasından çekiyoruz
            liste_bitis = detail_data.get("detay_tarih") if detail_data.get("detay_tarih") else card.get("liste_bitis_tarihi")

            record = {
                "kaynak": "vakif_katilim",
                "banka_adi": "Vakıf Katılım",
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
    print(f"İşlem Tamam! {len(unique_cards)} adet kampanya data klasörüne kaydedildi.")
    print("=" * 60)

if __name__ == "__main__":
    main()