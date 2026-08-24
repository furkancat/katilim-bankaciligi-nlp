"""
Emlak Katılım Kampanya Scraper Modülü

Playwright otomasyon altyapısı kullanılarak, Emlak Katılım Bankası'nın 
web arayüzünden kampanya verilerinin toplanması ve Retrieval-Augmented Generation (RAG) 
sistemine uygun yapısal formata (JSONL) dönüştürülmesi işlemini gerçekleştirir.
"""

import json
import os
import time
from datetime import datetime
from urllib.parse import urljoin

from playwright.sync_api import sync_playwright, TimeoutError as PlaywrightTimeout

# Konfigürasyon
BASE_URL = "https://www.emlakkatilim.com.tr"
LISTE_URL = f"{BASE_URL}/tr/bireysel/kampanyalar"
CIKTI_DIZINI = "data"
CIKTI_DOSYASI = os.path.join(CIKTI_DIZINI, "emlakkatilim_kampanyalar.jsonl")

# DOM Selectors (Frontend yapısındaki değişikliklere karşı merkezi yönetim)
KAMPANYA_KART_SELECTOR = "article.campaign-card"
# Hedef sitede sayfalama (pagination) verisi tek seferde yüklendiği için None atanmıştır
DAHA_FAZLA_BTN_SELECTOR = None 
# Detay sayfasındaki başlık h1 yapısında değilse, karttaki class fallback (yedek) olarak kullanılır
DETAY_BASLIK_SELECTOR = "h1, h3.campaign-card__title"
DETAY_METIN_SELECTOR = "div.searchContent"

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
    DOM'daki kampanya kartlarını gezer.
    Sayfa mimarisinde eksik veri girilmesi durumunda pipeline'ın çökmesini 
    önlemek için None fallback stratejisi uygulanır.
    """
    cards = page.query_selector_all(KAMPANYA_KART_SELECTOR)
    results = []
    
    for card in cards:
        try:
            # Başlık
            title_el = card.query_selector("h3.campaign-card__title")
            title = title_el.inner_text().strip() if title_el else None

            # Link
            link_el = card.query_selector("a.campaign-card__button")
            href = link_el.get_attribute("href") if link_el else None
            full_url = urljoin(BASE_URL, href) if href else None

            # Emlak Katılım liste kartında özet, tarih veya etiket HTML'i bulunmadığı için None atanarak geçilir
            results.append({
                "liste_baslik": title,
                "liste_ozet": None,
                "liste_url": full_url,
                "liste_bitis_tarihi": None, 
                "liste_etiket": None,
            })
        except Exception as e:
            # Hatalı DOM elemanını yoksayarak sürecin devam etmesini sağla
            print(f"  [Kart parse hatası] {e}")
            continue
            
    return results

def scrape_detail_page(page, url: str) -> dict:
    """
    Hedef sayfadan RAG bağlamı (context) üretimi için içerik çeker.
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
    print("Emlak Katılım Kampanya Scraper başlatılıyor...")
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

        # Eğer sayfada sonradan fark ettiğin bir load-more butonu olursa aşağıdaki bloğu aktifleştirebilirsin:
        # print("[2/3] 'Daha fazla göster' butonuna tıklanıyor...")
        # while True:
        #     if not click_load_more(page): break

        print("\n[2/2] Kampanya kartları toplanıyor...")
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
                "kaynak": "emlak_katilim",
                "banka_adi": "Emlak Katılım",
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
