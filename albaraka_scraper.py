"""
Albaraka Türk Veri Toplama (Web Scraping) Modülü

Bu modül, Albaraka Türk Katılım Bankası'nın dinamik (AJAX tabanlı) kampanya sayfasından
veri toplamak için Playwright otomasyon altyapısını kullanır.

Tasarım Kararları (Architecture Decisions):
1. Dinamik DOM Yönetimi: Sayfa tamamen yüklenene kadar beklenir ve pagination yerine kullanılan 
   'Daha Fazla' butonu tükenene kadar DOM manipüle edilerek tüm kampanyalar açığa çıkarılır.
2. RAM Optimizasyonu: Veriler bellekte devasa bir liste olarak tutulmak yerine, JSON Lines (JSONL) 
   formatında satır satır diske yazılarak (streaming) bellek şişmeleri (OOM) engellenmiştir.
3. Hata Toleransı (Resilience): Banka arayüzündeki olası HTML değişikliklerinin veya eksik alanların 
   tüm veri hattını (pipeline) çökertmemesi için bağımsız try-except blokları kurgulanmıştır.
"""

import json
import os
import time
from datetime import datetime
from urllib.parse import urljoin

from playwright.sync_api import sync_playwright, TimeoutError as PlaywrightTimeout

# Konfigürasyon

BASE_URL = "https://www.albaraka.com.tr"
LISTE_URL = f"{BASE_URL}/tr/kampanyalar"
CIKTI_DIZINI = "data"
CIKTI_DOSYASI = os.path.join(CIKTI_DIZINI, "albaraka_kampanyalar.jsonl")

# DOM Selectors (Bankanın frontend framework'üne uygun CSS seçicileri)
KAMPANYA_KART_SELECTOR = "div.col-lg-4.col-md-6.mb-5"
DAHA_FAZLA_BTN_SELECTOR = "a.btn.btn-outline-kampanyalar-primary"
DETAY_BASLIK_SELECTOR = "h1.searchTitle"
DETAY_METIN_SELECTOR = "div.searchContent.custom-table.custom-ul"

# Yardımcı Fonksiyonlar

def ensure_dir(path: str) -> None:
    os.makedirs(path, exist_ok=True)

def write_jsonl(filepath: str, record: dict) -> None:
    """
    Büyük veri setlerinde belleği (RAM) optimize etmek için verileri
    toplu halde değil, tek tek append moduyla diske yazar.
    """
    with open(filepath, "a", encoding="utf-8") as f:
        f.write(json.dumps(record, ensure_ascii=False) + "\n")

def extract_campaign_cards(page) -> list[dict]:
    """
    Mevcut DOM ağacındaki kampanya kartlarını ayrıştırır (parse).
    Bankanın web arayüzünde yaşanabilecek minik tasarım değişikliklerinin
    (örneğin etiketin veya bitiş tarihinin girilmemesi) sistemi durdurmaması 
    için her bir alan bağımsız olarak None fallback ile kontrol edilir.
    """
    cards = page.query_selector_all(KAMPANYA_KART_SELECTOR)
    results = []
    for card in cards:
        try:
            title_el = card.query_selector("h2.card-title a.searchContent")
            title = title_el.inner_text().strip() if title_el else None

            link_el = card.query_selector("h2.card-title a")
            href = link_el.get_attribute("href") if link_el else None
            full_url = urljoin(BASE_URL, href) if href else None

            summary_el = card.query_selector("p.card-text.searchContent a")
            summary = summary_el.inner_text().strip() if summary_el else None

            date_el = card.query_selector("span.kampanyalar-gun-pasive")
            date_text = date_el.inner_text().strip() if date_el else None

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
            # Sessiz hata yakalama: Hatalı bir kart yüzünden diğer 40 kartı kaybetmiyoruz
            print(f"  [UYARI] Kart parse edilirken anormallik tespit edildi: {e}")
            continue
    return results

def click_load_more(page) -> bool:
    """
    Gizli kampanyaları açığa çıkarmak için AJAX tetikleyici butona tıklar.
    PlaywrightTimeout yakalanarak sayfa sonuna gelindiği (butonun kaybolduğu)
    dinamik olarak tespit edilir.
    """
    try:
        btn = page.query_selector(DAHA_FAZLA_BTN_SELECTOR)
        if not btn or not btn.is_visible():
            return False
            
        btn.click()
        # AJAX isteğinin sunucuya gidip DOM'u güncellemesi için ağ gecikmesi payı (Network Latency)
        time.sleep(1.5)
        return True
    except PlaywrightTimeout:
        return False
    except Exception as e:
        print(f"  [UYARI] Sayfalama butonu ile etkileşim kurulamadı: {e}")
        return False

def scrape_detail_page(page, url: str) -> dict:
    """
    Kampanyanın detay sayfasına giderek LLM'in RAG pipeline'ında bağlam (context) çıkarımı
    yapabilmesi için hem doğal dil (text) hem de yapısal (HTML) metni toplar.
    """
    try:
        page.goto(url, wait_until="domcontentloaded", timeout=15000)
        time.sleep(0.8)

        title_el = page.query_selector(DETAY_BASLIK_SELECTOR)
        detail_title = title_el.inner_text().strip() if title_el else None

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
        print(f"  [HATA] Detay sayfası çekilemedi ({url}): {e}")
        return {
            "detay_baslik": None,
            "detay_metin": None,
            "detay_html": None,
            "detay_hata": str(e),
        }

# Ana Akış

def main():
    ensure_dir(CIKTI_DIZINI)

    # Idempotent Çalışma: Scraper birden fazla kez çalıştırılırsa verilerin 
    # üst üste binmesini (duplicate) önlemek için önceki çalışma durumunu sıfırlarız.
    if os.path.exists(CIKTI_DOSYASI):
        os.remove(CIKTI_DOSYASI)

    print("=" * 60)
    print("[SİSTEM] Albaraka Türk Kampanya Veri Toplayıcısı (Scraper) başlatılıyor...")
    print(f"Hedef: {LISTE_URL}")
    print(f"Çıktı: {CIKTI_DOSYASI}")
    print("=" * 60)

    with sync_playwright() as p:
        browser = p.chromium.launch(headless=True)
        
        # Anti-bot mekanizmalarına (WAF/Cloudflare) takılmamak için gerçek bir 
        # son kullanıcı (End-User) tarayıcı kimliği (User-Agent) taklit edilir.
        context = browser.new_context(
            viewport={"width": 1280, "height": 800},
            user_agent=(
                "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
                "AppleWebKit/537.36 (KHTML, like Gecko) "
                "Chrome/128.0.0.0 Safari/537.36"
            ),
        )

        page = context.new_page()
        print(f"\n[1/3] Ana liste sayfası yükleniyor: {LISTE_URL}")
        page.goto(LISTE_URL, wait_until="domcontentloaded", timeout=30000)
        time.sleep(2)

        print("[2/3] Dinamik içerikler (AJAX) tetikleniyor...")
        click_count = 0
        while True:
            success = click_load_more(page)
            if not success:
                print(f"    DOM tamamen genişletildi. Toplam {click_count} sayfalama isteği gönderildi.")
                break
            click_count += 1

        print("\n[3/3] Kampanya kartları DOM üzerinden parse ediliyor...")
        all_cards = extract_campaign_cards(page)
        
        # Aynı kampanya farklı kategorilerde (örn: Hem 'Yeni' hem 'Kart') listelenmişse,
        # vektör veritabanında (ChromaDB) kirlilik yaratmaması için URL bazlı tekilleştirme yapılır.
        seen_urls = set()
        unique_cards = []
        for c in all_cards:
            url = c.get("liste_url")
            if url and url not in seen_urls:
                seen_urls.add(url)
                unique_cards.append(c)

        print(f"    Saptanan toplam kart: {len(all_cards)} | Tekilleştirilmiş kart: {len(unique_cards)}")

        print("\n[İŞLEM] Detay sayfaları ziyaret edilerek ham metinler toplanıyor...")
        detail_page = context.new_page()

        for idx, card in enumerate(unique_cards, 1):
            url = card.get("liste_url")
            if not url:
                continue

            print(f"    [{idx}/{len(unique_cards)}] İşleniyor: {url}")
            detail_data = scrape_detail_page(detail_page, url)

            # İleride eklenebilecek diğer bankalarla veritabanı şema (schema) 
            # uyumluluğunu sağlamak için standartlaştırılmış JSON belgesi oluşturulur.
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
    print(f"[BAŞARILI] {len(unique_cards)} adet kampanya verisi çıkarıldı ve diske yazıldı.")
    print(f"Hedef Dosya: {CIKTI_DOSYASI}")
    print("=" * 60)


if __name__ == "__main__":
    main()
