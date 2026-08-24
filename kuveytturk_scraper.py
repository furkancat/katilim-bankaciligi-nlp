"""
Kuveyt Türk Kampanya Scraper Modülü

Playwright otomasyon altyapısı kullanılarak, Kuveyt Türk'ün dinamik web arayüzünden 
kampanya verilerinin toplanması ve yapısal formata (JSONL) dönüştürülmesi işlemini gerçekleştirir.
"""

import json
import os
import time
from datetime import datetime
from urllib.parse import urljoin

from playwright.sync_api import sync_playwright, TimeoutError as PlaywrightTimeout

# Konfigürasyon
BASE_URL = "https://www.kuveytturk.com.tr"
LISTE_URL = f"{BASE_URL}/kampanyalar/kendim-icin"
CIKTI_DIZINI = "data"
CIKTI_DOSYASI = os.path.join(CIKTI_DIZINI, "kuveytturk_kampanyalar.jsonl")

# DOM Selectors (Frontend yapısındaki değişikliklere karşı merkezi yönetim)
KAMPANYA_KART_SELECTOR = "div.campaign-item"
DAHA_FAZLA_BTN_SELECTOR = "a.load-more-btn"
DETAY_BASLIK_SELECTOR = "h1#pageTitle"
# Detay sayfasında sol taraftaki paylaşım/tarih gibi gereksiz DOM bloklarını almamak için doğrudan asıl metin hedeflenir:
DETAY_METIN_SELECTOR = "div.col-12.col-lg.search-content" 

# Yardımcı Fonksiyonlar

def ensure_dir(path: str) -> None:
    # İlgili dizin mevcut değilse oluşturur (Dosya sistemi hatasını önler)
    os.makedirs(path, exist_ok=True)

def write_jsonl(filepath: str, record: dict) -> None:
    # Streaming yazma metodu: Verileri bellekte (RAM) biriktirmeden diske aktararak bellek şişmesini (OOM) önler.
    with open(filepath, "a", encoding="utf-8") as f:
        f.write(json.dumps(record, ensure_ascii=False) + "\n")

def extract_campaign_cards(page) -> list[dict]:
    """
    DOM'daki kampanya kartlarını çeker.
    Eksik veri girilmesi durumunda pipeline'ın çökmesini önlemek için None fallback stratejisi uygulanır.
    """
    cards = page.query_selector_all(KAMPANYA_KART_SELECTOR)
    results = []
    
    for card in cards:
        try:
            # Başlık
            title_el = card.query_selector("h3 a")
            title = title_el.inner_text().strip() if title_el else None

            # Link
            href = title_el.get_attribute("href") if title_el else None
            full_url = urljoin(BASE_URL, href) if href else None

            # Özet metin
            summary_el = card.query_selector("div.description")
            summary = summary_el.inner_text().strip() if summary_el else None

            # Bitiş tarihi (Veri kalitesini artırmak için HTML'deki "Kampanya Aralığı: " etiketleri temizlenir)
            date_el = card.query_selector("div.date")
            date_text = date_el.inner_text().replace("Kampanya Aralığı:", "").strip() if date_el else None

            # Etiket / Kategori (Kuveyt Türk kartlarında belirgin bir badge (etiket) olmadığı için None geçilir)
            badge = None

            results.append({
                "liste_baslik": title,
                "liste_ozet": summary,
                "liste_url": full_url,
                "liste_bitis_tarihi": date_text,
                "liste_etiket": badge,
            })
        except Exception as e:
            # Hatalı DOM elemanını yoksayarak sürecin devam etmesini sağla
            print(f"  [Kart parse hatası] {e}")
            continue
            
    return results

def click_load_more(page) -> bool:
    """
    'Daha Fazla Yükle' butonuna tıklar.
    Sayfa sonu tespiti (Pagination Control): Buton DOM'da kalsa bile CSS ile 
    (display: none) gizlendiğinde sonsuz döngüyü önlemek için is_visible() kullanılır.
    """
    try:
        btn = page.query_selector(DAHA_FAZLA_BTN_SELECTOR)
        if not btn:
            return False
        
        if not btn.is_visible():
            return False
            
        btn.click()
        # AJAX isteğinin sunucuya gidip yanıt dönmesi (Network Latency) için bekleme payı
        time.sleep(1.5) 
        return True
    except PlaywrightTimeout:
        return False
    except Exception as e:
        print(f"  [Buton tıklama hatası] {e}")
        return False

def scrape_detail_page(page, url: str) -> dict:
    """
    Detay sayfasına girip başlık ve tam metni çeker.
    Network hatalarında veya kırık linklerde (404) ana akışın bozulmaması için
    bağımsız hata yakalama bloklarıyla korunmuştur.
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

# Ana Akış

def main():
    ensure_dir(CIKTI_DIZINI)

    # Idempotent yapı: Script tekrar çalıştığında verilerin üst üste binmesini (duplicate) önler
    if os.path.exists(CIKTI_DOSYASI):
        os.remove(CIKTI_DOSYASI)

    print("=" * 60)
    print("Kuveyt Türk Kampanya Scraper başlatılıyor...")
    print("=" * 60)

    with sync_playwright() as p:
        browser = p.chromium.launch(headless=True)
        # Anti-bot WAF sistemlerini atlatmak için Standart Kullanıcı User-Agent profili taklit edilir
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

        print("[2/3] 'Daha Fazla Yükle' butonuna tıklanıyor...")
        click_count = 0
        while True:
            success = click_load_more(page)
            if not success:
                print(f"    Daha fazla içerik kalmadı veya buton gizlendi. Toplam {click_count} tıklama yapıldı.")
                break
            click_count += 1
            print(f"    Tıklama #{click_count} başarılı.")

        print("\n[3/3] Kampanya kartları toplanıyor...")
        all_cards = extract_campaign_cards(page)
        
        # Veritabanı tutarlılığı için URL tabanlı tekrarlayan (duplicate) kayıt engelleme
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
                "kaynak": "kuveyt_turk",
                "banka_adi": "Kuveyt Türk",
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
    print(f"İşlem Tamam! {len(unique_cards)} adet kampanya data klasörüne kaydedildi.")
    print("=" * 60)

if __name__ == "__main__":
    main()
