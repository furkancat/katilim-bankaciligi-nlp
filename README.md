# Türkçe Finansal NLP Asistanı

> **Katılım Bankacılığı Ürün Metinlerinden Otomatik Bilgi Çıkarımı ve Karşılaştırmalı Analiz Platformu**

## 📋 Proje Özeti

Bu proje, katılım bankalarının resmi web sitelerinde yer alan finansman, kart ve yatırım kampanya metinlerini otomatik olarak toplayan, doğal dil işleme (NLP) teknikleriyle yapılandırılmış veriye dönüştüren ve kullanıcılara chatbot ile dashboard arayüzleri üzerinden erişilebilir kılan uçtan uca bir NLP çözümüdür.

Proje; metin madenciliği, bilgi çıkarımı, metin sınıflandırma ve Retrieval-Augmented Generation (RAG) mimarilerini birleştirerek, farklı bankalara ait ürünlerin karşılaştırılabilir hale getirilmesini sağlar.

---

## 🚀 Temel Özellikler

### 1. Çoklu Kaynaklı Veri Toplama
- **9 farklı katılım bankası** için özel olarak tasarlanmış Playwright tabanlı scraper'lar
- Dinamik sayfa yapılarına uyumlu (lazy loading, AJAX pagination, infinite scroll)
- Kampanya listeleme ve detay sayfalarının tam metin çekimi
- JSON Lines formatında yapılandırılmış ham veri çıktısı

### 2. Hibrit Bilgi Çıkarımı (Regex + LLM)
- **İki aşamalı pipeline**: Regex ön işleme + LLM doğrulama ve zenginleştirme
- Kampanya türü otomatik sınıflandırma (Konut, İhtiyaç, Taşıt, Kart, Alışveriş Puanı, Yeni Müşteri, Yatırım)
- Finansal parametre çıkarımı:
  - Kâr payı oranı
  - Finansman tutarı ve vade süresi
  - Tahsis ücreti
  - İndirim oranı
  - Alışveriş puanı / mil
  - Masraf bilgisi ve kampanya koşulları
- Katılım bankacılığı terminolojisine özel kural tabanlı validasyon

### 3. RAG Tabanlı Akıllı Chatbot
- **ChromaDB** vektör veritabanı ile semantik arama
- **HuggingFace** çok dilli embedding modelleri (paraphrase-multilingual-MiniLM)
- **Max Marginal Relevance (MMR)** ile sonuç çeşitliliği (farklı bankaların kampanyalarını zorla getirme)
- Lokal LLM entegrasyonu (Ollama) ile tam on-premise çalışabilirlik
- Banka kimlik koruması: Kampanya-banka eşleştirmesinde sıfır hata toleransı

### 4. Karşılaştırmalı Analiz
- Farklı bankaların benzer ürünlerinin yan yana karşılaştırılması
- En düşük kâr payı, en uzun vade, en avantajlı kampanya gibi kriterlere göre sıralama
- Yapılandırılmış tablo çıktıları

### 5. Web Dashboard
- Streamlit tabanlı interaktif kullanıcı arayüzü
- Gerçek zamanlı soru-cevap ve kampanya karşılaştırma

---

## 🏗️ Sistem Mimarisi

```
┌─────────────────────────────────────────────────────────────────┐
│                        VERİ TOPLAMA KATMANI                      │
│  ┌──────────┐ ┌──────────┐ ┌──────────┐ ┌──────────┐ ┌─────────┐ │
│  │ Albaraka │ │ Dünya    │ │ Kuveyt   │ │ Ziraat   │ │ Vakıf   │ │
│  │ Türk     │ │ Katılım  │ │ Türk     │ │ Katılım  │ │ Katılım │ │
│  └────┬─────┘ └────┬─────┘ └────┬─────┘ └────┬─────┘ └────┬────┘ │
│  ┌──────────┐ ┌──────────┐ ┌──────────┐ ┌──────────┐              │
│  │ Türkiye  │ │ Emlak    │ │ Hayat    │ │ TOM      │              │
│  │ Finans   │ │ Katılım  │ │ Finans   │ │ Bank     │              │
│  └────┬─────┘ └────┬─────┘ └────┬─────┘ └────┬─────┘              │
└───────┼────────────┼────────────┼────────────┼────────────────────┘
        │            │            │            │
        └────────────┴──────┬─────┴────────────┘
                            ▼
              ┌─────────────────────────────┐
              │   HAM VERİ (JSON Lines)     │
              │   data/raw/*.jsonl          │
              └──────────────┬──────────────┘
                             │
                             ▼
              ┌─────────────────────────────┐
              │   NLP PIPELINE (LangGraph)  │
              │                             │
              │  ┌───────────────────────┐    │
              │  │ 1. Regex Extractor    │    │
              │  │    (Hızlı ön filtre)  │    │
              │  └───────────┬───────────┘    │
              │              ▼                │
              │  ┌───────────────────────┐    │
              │  │ 2. LLM Extractor    │    │
              │  │    (Qwen 2.5 /       │    │
              │  │     Ollama)          │    │
              │  └───────────┬───────────┘    │
              │              ▼                │
              │  ┌───────────────────────┐    │
              │  │ 3. Validator        │    │
              │  │    (Mantıksal kontrol)│   │
              │  └───────────┬───────────┘    │
              │              ▼                │
              │  ┌───────────────────────┐    │
              │  │ 4. Normalizer         │    │
              │  │    (Standart format)  │    │
              │  └───────────┬───────────┘    │
              └──────────────┼──────────────┘
                             │
                             ▼
              ┌─────────────────────────────┐
              │ YAPIILANDIRILMIŞ VERİ       │
              │ data/structured_*.jsonl     │
              └──────────────┬──────────────┘
                             │
              ┌──────────────┼──────────────┐
              │              │              │
              ▼              ▼              ▼
    ┌─────────────────┐ ┌─────────────┐ ┌──────────────┐
    │  EMBEDDING      │ │   BATCH     │ │   STREAMLIT  │
    │  (HuggingFace)  │ │  PROCESSOR  │ │  DASHBOARD   │
    │                 │ │             │ │              │
    │  ┌───────────┐  │ │  Tüm veriyi │ │  Interaktif  │
    │  │ ChromaDB  │  │ │  pipeline'dan│ │  web arayüzü │
    │  │ Vektör DB │  │ │  geçirme    │ │              │
    │  └─────┬─────┘  │ └─────────────┘ └──────────────┘
    └────────┼────────┘
             │
             ▼
    ┌─────────────────┐
    │   RAG CHATBOT   │
    │                 │
    │  ┌───────────┐  │
    │  │  Ollama   │  │
    │  │  (Lokal)  │  │
    │  │  LLM      │  │
    │  └───────────┘  │
    └─────────────────┘
```

---

## 🛠️ Teknoloji Stack'i

| Katman | Teknoloji |
|--------|-----------|
| **Web Scraping** | Playwright, Python |
| **NLP Pipeline** | LangGraph, LangChain, Ollama |
| **LLM** | Qwen 2.5 (lokal, 4B+) |
| **Embedding** | sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2 |
| **Vektör DB** | ChromaDB |
| **Web Arayüzü** | Streamlit |
| **Veri Formatı** | JSON Lines (JSONL) |

---

## 📦 Kurulum

### Gereksinimler
- Python 3.10+
- Ollama (lokal LLM sunucusu)
- Playwright tarayıcıları

### Adım 1: Depoyu Klonlayın
```bash
git clone <repo-url>
cd katilim-bankasi-nlp
```

### Adım 2: Python Bağımlılıklarını Yükleyin
```bash
pip install -r requirements.txt
```

### Adım 3: Playwright Tarayıcılarını Yükleyin
```bash
playwright install chromium
```

### Adım 4: Ollama'yı Kurun ve Modeli İndirin
```bash
# Ollama kurulumu: https://ollama.com
ollama pull qwen2.5
```

### Adım 5: Vektör Veritabanını Oluşturun
```bash
python chatbot.py --rebuild
```

---

## 🚀 Kullanım

### 1. Veri Toplama (Scraping)

Her banka için ayrı scraper çalıştırılabilir:

```bash
# Tüm bankaları çekmek için
python albaraka_scraper.py
python dunya_katilim_scraper.py
python kuveytturk_scraper.py
python ziraatkatilim_scraper.py
python vakifkatilim_scraper.py
python turkiyefinans_scraper.py
python emlakkatilim_scraper.py
python hayatfinans_scraper.py
python tombankhadi_scraper.py
```

Çıktılar `data/` dizinine `*.jsonl` formatında kaydedilir.

### 2. NLP Pipeline ile Bilgi Çıkarımı

Ham veriyi yapılandırılmış formata dönüştürün:

```bash
python batch_processor.py \
    --input data/raw \
    --output data/structured_kampanyalar.jsonl
```

Pipeline aşamaları:
1. **Regex Extractor**: Hızlı ön filtreleme (kâr payı, vade, tutar vb.)
2. **LLM Extractor**: Derin anlamsal analiz ve doğrulama
3. **Validator**: Finansal mantık kontrolleri (örn: kâr payı < %50)
4. **Normalizer**: Standart veri formatına dönüştürme

### 3. Chatbot (CLI)

```bash
python chatbot.py
```

Örnek etkileşimler:
- *"En düşük kâr payı oranı hangi bankada?"*
- *"Albaraka konut finansmanı oranı nedir?"*
- *"Albaraka mı daha avantajlı, Dünya Katılım mı?"*
- *"120 ay vadeli konut finansmanı olan bankalar?"*

### 4. Web Dashboard

```bash
streamlit run app.py
```

---

## 🧠 NLP Pipeline Detayları

### Regex + LLM Hibrit Yaklaşım

| Aşama | Amaç | Teknoloji |
|-------|------|-----------|
| Regex Extractor | Hızlı ön bilgi çıkarımı | Python `re` modülü |
| LLM Extractor | Anlamsal derinlik ve doğrulama | Qwen 2.5 (Ollama) |
| Validator | Finansal mantık ve tutarlılık kontrolü | Kural tabanlı |
| Normalizer | Standart formata dönüştürme | Python |

### Çıkarılan Finansal Alanlar

| Alan | Açıklama | Örnek |
|------|----------|-------|
| `kar_payi_orani` | Finansman kâr payı | `%1.89` |
| `finansman_tutari` | Maksimum finansman tutarı | `500.000 TL` |
| `vade_suresi_ay` | Vade süresi (ay) | `120` |
| `taksit_sayisi` | Taksit sayısı | `120` |
| `tahsis_ucreti` | Tek seferlik tahsis ücreti | `0 TL` |
| `indirim_orani` | Kampanyaya özel indirim | `%10` |
| `alisveris_puani` | Kazanılan puan/mil | `500` |
| `masraf_bilgisi` | Masraf durumu | `Dosya masrafı alınmıyor` |
| `kampanya_turu` | Sınıflandırılmış kampanya tipi | `KonutFinansmaniKampanyasi` |
| `hedef_kitle` | Müşteri segmenti | `["YeniMusteri", "Bireysel"]` |

### Kampanya Türü Sınıflandırma

- `KonutFinansmaniKampanyasi`
- `IhtiyacFinansmaniKampanyasi`
- `TasitFinansmaniKampanyasi`
- `KartKampanyasi`
- `AlisverisPuaniKampanyasi`
- `YeniMusteriKampanyasi`
- `YatirimUrunuKampanyasi`
- `Diger`

---

## 💬 RAG Chatbot Detayları

### Mimari

```
Kullanıcı Sorusu
       │
       ▼
┌──────────────────┐
│ Semantic Search  │ ← HuggingFace Embeddings
│    (ChromaDB)    │
│     MMR: k=12    │ ← Çeşitlilik garantisi
└────────┬─────────┘
         │
         ▼
┌──────────────────┐
│ Context Builder  │ ← Banka bilgisi ile zenginleştirilmiş
│                  │   doğal dil metinleri
└────────┬─────────┘
         │
         ▼
┌──────────────────┐
│  RAG Prompt      │ ← Sistem talimatları + bağlam
│  + LLM (Ollama)  │
└────────┬─────────┘
         │
         ▼
┌──────────────────┐
│  Post-Processing │ ← "faiz"→"kâr payı", "kredi"→"finansman"
│  (Terminoloji)   │
└────────┬─────────┘
         │
         ▼
    Kullanıcı Yanıtı
```

### Güvenlik ve Doğruluk Önlemleri

- **Kimlik Koruması**: Kampanya-banka eşleştirmesi kesinlikle korunur
- **Ürün Tipi Eşleştirme**: Kullanıcı "konut" sorarsa sadece konut kampanyaları getirilir
- **Terminoloji Filtresi**: "Faiz" ve "kredi" kelimeleri otomatik olarak "kâr payı" ve "finansman" ile değiştirilir
- **Çeşitlilik (MMR)**: Aynı bankanın benzer kampanyalarını tekrarlamak yerine farklı bankaların ürünlerini getirir

---

## 📁 Proje Yapısı

```
.
├── app.py                          # Streamlit dashboard
├── chatbot.py                      # RAG chatbot (CLI + API)
├── langgraph_pipeline.py           # NLP bilgi çıkarım pipeline'ı
├── batch_processor.py              # Toplu veri işleme
├── albaraka_scraper.py             # Albaraka Türk scraper
├── dunya_katilim_scraper.py        # Dünya Katılım scraper
├── kuveytturk_scraper.py           # Kuveyt Türk scraper
├── ziraatkatilim_scraper.py        # Ziraat Katılım scraper
├── vakifkatilim_scraper.py         # Vakıf Katılım scraper
├── turkiyefinans_scraper.py        # Türkiye Finans scraper
├── emlakkatilim_scraper.py         # Emlak Katılım scraper
├── hayatfinans_scraper.py          # Hayat Finans scraper
├── tombankhadi_scraper.py          # TOM Bank scraper
├── data/
│   ├── raw/                        # Ham scraping çıktıları
│   └── structured_kampanyalar.jsonl # Pipeline çıktısı
└── requirements.txt
```

---

## 🔒 On-Premise & Veri Güvenliği

- Tüm model ve vektör işlemleri **tamamen lokal** ortamda çalışır
- Harici API servislerine (OpenAI, Anthropic vb.) **bağımlılık yoktur**
- Müşteri verileri kurum dışına çıkmaz
- Ollama ile çalışan LLM, kurum içi sunucularda barındırılabilir
- Açık kaynak lisanslı (Apache 2.0 uyumlu) tüm bağımlılıklar

---

## 📊 Performans ve Doğruluk

- **Regex Confidence**: Her kayıt için regex ön işleme güven skoru
- **LLM Doğrulama**: Regex sonuçlarının LLM tarafından ikincil kontrolü
- **Validasyon**: Finansal mantık kuralları ile otomatik hata tespiti
- **Retry Mekanizması**: LLM hatalarında otomatik yeniden deneme (max 2)

---

## 📝 Geliştirici Notları

- Pipeline, LangGraph'in durum yönetimi (state management) özelliklerini kullanarak modüler bir akış sunar
- Chatbot, `lambda_mult=0.6` MMR parametresiyle farklı bankaların çeşitliliğini garanti eder
- Tüm scraper'lar benzer arayüzü paylaşır: `liste_url`, `detay_metin`, `detay_html` alanları standarttır

---

## 📄 Lisans

Bu proje açık kaynaklıdır ve Apache License 2.0 ile lisanslanmıştır.

---

> **Not:** Bu proje, katılım bankacılığı sektöründe NLP tabanlı otomasyon ve karşılaştırmalı analiz ihtiyacına yönelik geliştirilmiş, üretime hazır bir proof-of-concept çalışmasıdır.
