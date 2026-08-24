"""
Albaraka Türk Katılım Bankası Kampanya Scraper
TEKNOFEST Türkçe Yapay Zeka Dil Ajanları Yarışması - Veri Toplama Modülü

Kullanım:
    python albaraka_scraper.py

Çıktı:
    data/albaraka_kampanyalar.jsonl  - Her satır bir kampanya (JSON Lines)
"""

import json
import os
import time
from datetime import datetime
from urllib.parse import urljoin

from playwright.sync_api import sync_playwright, TimeoutError as PlaywrightTimeout

# ─── Konfigürasyon ───────────────────────────────────────────────────────────
BASE_URL = "https://www.albaraka.com.tr"
LISTE_URL = f"{BASE_URL}/tr/kampanyalar"
CIKTI_DIZINI = "data"
CIKTI_DOSYASI = os.path.join(CIKTI_DIZINI, "albaraka_kampanyalar.jsonl")

# Selector'lar
KAMPANYA_KART_SELECTOR = "div.col-lg-4.col-md-6.mb-5"
DAHA_FAZLA_BTN_SELECTOR = "a.btn.btn-outline-kampanyalar-primary"
DETAY_BASLIK_SELECTOR = "h1.searchTitle"
DETAY_METIN_SELECTOR = "div.searchContent.custom-table.custom-ul"

# ─── Yardımcı Fonksiyonlar ───────────────────────────────────────────────────

def ensure_dir(path: str) -> None:
    os.makedirs(path, exist_ok=True)

def write_jsonl(filepath: str, record: dict) -> None:
    with open(filepath, "a", encoding="utf-8") as f:
        f.write(json.dumps(record, ensure_ascii=False) + "\n")

def extract_campaign_cards(page) -> list[dict]:
    """
    Mevcut DOM'daki tüm kampanya kartlarını çeker.
    Sayfa dinamik yüklendiği için her tıklamadan sonra tekrar çağrılır.
    """
    cards = page.query_selector_all(KAMPANYA_KART_SELECTOR)
    results = []
    for card in cards:
        try:
            # Başlık
            title_el = card.query_selector("h2.card-title a.searchContent")
            title = title_el.inner_text().strip() if title_el else None

            # Link (relative)
            link_el = card.query_selector("h2.card-title a")
            href = link_el.get_attribute("href") if link_el else None
            full_url = urljoin(BASE_URL, href) if href else None

            # Özet metin
            summary_el = card.query_selector("p.card-text.searchContent a")
            summary = summary_el.inner_text().strip() if summary_el else None

            # Bitiş tarihi metni
            date_el = card.query_selector("span.kampanyalar-gun-pasive")
            date_text = date_el.inner_text().strip() if date_el else None

            # Etiket (badge)
            badge_el = card.query_selector("span.card-image-over")
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
    'Daha Fazla Kampanya Göster' butonuna tıklar.
    Buton yoksa veya tıklanamazsa False döner.
    """
    try:
        btn = page.query_selector(DAHA_FAZLA_BTN_SELECTOR)
        if not btn:
            return False
        # Buton görünür mü?
        if not btn.is_visible():
            return False
        btn.click()
        # Küçük bekleme: AJAX ile yeni kartlar gelsin
        time.sleep(1.5)
        return True
    except PlaywrightTimeout:
        return False
    except Exception as e:
        print(f"  [Buton tıklama hatası] {e}")
        return False

def scrape_detail_page(page, url: str) -> dict:
    """
    Kampanya detay sayfasına gidip başlık ve tam metni çeker.
    """
    try:
        page.goto(url, wait_until="domcontentloaded", timeout=15000)
        # İçerik yüklenmesi için kısa bekleme
        time.sleep(0.8)

        # Başlık
        title_el = page.query_selector(DETAY_BASLIK_SELECTOR)
        detail_title = title_el.inner_text().strip() if title_el else None

        # Tam metin (HTML + text)
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

    # Eski dosyayı temizle (idempotent çalışma için)
    if os.path.exists(CIKTI_DOSYASI):
        os.remove(CIKTI_DOSYASI)

    print("=" * 60)
    print("Albaraka Türk Kampanya Scraper başlatılıyor...")
    print(f"Liste URL: {LISTE_URL}")
    print(f"Çıktı: {CIKTI_DOSYASI}")
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

        # ─── ADIM 1: Liste Sayfası ─────────────────────────────────────────
        page = context.new_page()
        print(f"\n[1/3] Liste sayfası yükleniyor: {LISTE_URL}")
        page.goto(LISTE_URL, wait_until="domcontentloaded", timeout=30000)
        time.sleep(2)  # İlk yükleme için bekle

        # ─── ADIM 2: "Daha Fazla" butonuna tıkla tıkla ─────────────────────
        print("[2/3] 'Daha Fazla Kampanya Göster' butonuna tıklanıyor...")
        click_count = 0
        while True:
            success = click_load_more(page)
            if not success:
                print(f"    Buton kalmadı veya tıklanamadı. Toplam {click_count} tıklama.")
                break
            click_count += 1
            print(f"    Tıklama #{click_count} yapıldı.")

        # ─── ADIM 3: Tüm kartları topla ────────────────────────────────────
        print("\n[3/3] Kampanya kartları toplanıyor...")
        all_cards = extract_campaign_cards(page)
        total = len(all_cards)
        print(f"    Toplam {total} kampanya kartı bulundu.")

        # Tekrarları önlemek için URL bazlı set
        seen_urls = set()
        unique_cards = []
        for c in all_cards:
            url = c.get("liste_url")
            if url and url not in seen_urls:
                seen_urls.add(url)
                unique_cards.append(c)

        print(f"    Tekil kampanya sayısı: {len(unique_cards)}")

        # ─── ADIM 4: Detay sayfalarını çek ─────────────────────────────────
        print("\n[4/4] Detay sayfaları çekiliyor...")
        detail_page = context.new_page()

        for idx, card in enumerate(unique_cards, 1):
            url = card.get("liste_url")
            if not url:
                continue

            print(f"    [{idx}/{len(unique_cards)}] {url}")
            detail_data = scrape_detail_page(detail_page, url)

            record = {
                "kaynak": "albaraka_turk",
                "banka_adi": "Albaraka Türk Katılım Bankası",
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
    print(f"Tamamlandı! {len(unique_cards)} kampanya kaydedildi.")
    print(f"Dosya: {CIKTI_DOSYASI}")
    print("=" * 60)


if __name__ == "__main__":
    main()