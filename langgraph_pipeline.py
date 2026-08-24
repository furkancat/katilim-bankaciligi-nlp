"""
Katılım Bankacılığı NLP Pipeline
LangGraph + Ollama (Qwen 2.5) Hibrit Bilgi Çıkarımı

Düzeltmeler:
  - Şartname tablosundaki TÜM alanlar eklendi:
      • tahsis_ucreti (Finansman Bilgileri)
      • indirim_orani (Kampanya Bilgileri)
      • alisveris_puani (Kampanya Bilgileri)
  - Regex pattern'leri genişletildi
  - LLM prompt'u yeni alanları içerecek şekilde güncellendi
  - Normalizer ve final_output yeni alanları işliyor

Bu modül; deterministik kural tabanlı (Regex) çıkarım ile olasılıksal Büyük Dil Modeli (LLM) 
çıkarımını LangGraph mimarisi üzerinde birleştirerek halüsinasyonu en aza indiren 
bir hibrit RAG (Retrieval-Augmented Generation) veri işleme hattıdır.
"""

import json
import re
import os
from typing import TypedDict, Optional
from datetime import datetime

try:
    from langgraph.graph import StateGraph, END
    from langchain_ollama import OllamaLLM
except ImportError:
    print("[HATA] pip install langgraph langchain-ollama")
    raise

OLLAMA_MODEL = os.getenv("OLLAMA_MODEL", "qwen2.5")
OLLAMA_BASE_URL = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434")

# Pipeline State (Durum Yönetimi):
# İş akışındaki her bir düğümün (node) birbiriyle iletişim kurmasını ve
# verinin (context) kaybolmadan akmasını sağlayan tip güvenli (type-safe) sözlük yapısı.
class PipelineState(TypedDict):
    raw_text: str
    banka_adi: str
    url: str
    liste_etiket: str
    liste_bitis_tarihi: str
    kaynak: str
    regex_results: dict
    regex_confidence: float
    llm_results: dict
    llm_raw_response: str
    validation_errors: list[str]
    final_output: dict
    retry_count: int


# Node 1: Regex Extractor

class RegexExtractor:
    # Deterministik Çıkarım (Kural Tabanlı Ön İşleme):
    # LLM'lerin sayısal verilerde halüsinasyon yapma (uydurma) riskine karşı,
    # finansal veriler ilk olarak kesin kurallı Regex (Düzenli İfadeler) ile taranır.
    PATTERNS = {
        "kar_payi_orani": re.compile(r"%(\d+[,.]?\d*)"),
        "vade_suresi": re.compile(r"(\d+)\s*(?:ay|aya|ayı)\s*(?:kadar|varan)?", re.IGNORECASE),
        "finansman_tutari": re.compile(r"(\d{1,3}(?:[.,]?\d{3})*)\s*(?:TL|₺)\s*(?:'ye|'ya)?\s*(?:kadar|varan)?", re.IGNORECASE),
        "taksit_sayisi": re.compile(r"(\d+)\s*(?:taksit|taksitli)", re.IGNORECASE),
        "tarih_sayisal": re.compile(r"(\d{1,2})[./](\d{1,2})[./](\d{4})"),
        # v6: Yeni pattern'ler
        "tahsis_ucreti": re.compile(r"(?:tahsis\s*ücreti|tahsis\s*ucreti)[^\d]*(\d+[.,]?\d*)\s*(?:TL|₺)?", re.IGNORECASE),
        "indirim_orani": re.compile(r"(\d+[,.]?\d*)\s*%\s*(?:indirim|iskonto)", re.IGNORECASE),
        "alisveris_puani": re.compile(r"(\d+)\s*(?:worldpuan|puan|mile|mil|world)", re.IGNORECASE),
    }

    def extract(self, text: str) -> tuple[dict, float]:
        results = {}
        filled = 0
        total = 7  # v6: Toplam alan sayısı arttı

        m = self.PATTERNS["kar_payi_orani"].search(text)
        if m:
            results["kar_payi_orani"] = f"%{m.group(1)}"
            filled += 1

        m = self.PATTERNS["vade_suresi"].search(text)
        if m:
            results["vade_suresi_ay"] = int(m.group(1))
            filled += 1

        m = self.PATTERNS["finansman_tutari"].search(text)
        if m:
            results["finansman_tutari"] = m.group(0)
            filled += 1

        m = self.PATTERNS["taksit_sayisi"].search(text)
        if m:
            results["taksit_sayisi"] = int(m.group(1))
            filled += 1

        m = self.PATTERNS["tarih_sayisal"].search(text)
        if m:
            d, mo, y = m.groups()
            results["kampanya_suresi"] = f"{d}.{mo}.{y}"

        # v6: Yeni alanlar
        m = self.PATTERNS["tahsis_ucreti"].search(text)
        if m:
            results["tahsis_ucreti"] = m.group(1) + " TL"
            filled += 1

        m = self.PATTERNS["indirim_orani"].search(text)
        if m:
            results["indirim_orani"] = f"%{m.group(1)}"
            filled += 1

        m = self.PATTERNS["alisveris_puani"].search(text)
        if m:
            results["alisveris_puani"] = m.group(1)
            filled += 1

        return results, filled / total


def regex_extractor_node(state: PipelineState) -> PipelineState:
    extractor = RegexExtractor()
    results, confidence = extractor.extract(state["raw_text"])
    state["regex_results"] = results
    state["regex_confidence"] = round(confidence, 2)
    state["retry_count"] = 0
    print(f"[Regex] Confidence: {confidence:.2f} | Bulunan: {list(results.keys())}")
    return state


# Node 2: LLM Extractor (Her Zaman Çalışır + Regex Context)

EXTRACTION_PROMPT = """Sen bir katılım bankacılığı finansal analistisin. 
Aşağıdaki kampanya metnini analiz et ve KESİNLİKLE geçerli JSON çıktısı üret.

ÖNEMLİ: Regex ön işleme şu değerleri buldu (AMA bunlar HATALI olabilir, lütfen doğrula):
{regex_context}

Regex değerlerini KONTROL ET:
- "3 aya varan ödemesiz dönem" gibi ifadeler VADE DEĞİLDİR. Vadeyi metnin kendi içinden dikkatlice bularak çıkar.
- "140.000 TL" gibi tutarlar ödül/indirim, asgari harcama şartı veya hesap bakiyesi olabilir, finansman tutarı olmayabilir.
- Kampanya türünü metnin genel bağlamından çıkar, regex'in bulduğu sayılara göre karar verme.

KESİN KURALLAR:
1. "Kâr payı oranı" SADECE finansman (kredi benzeri) ürünlerde vardır.
   - Kart/ödül kampanyalarında kar_payi_orani KESİNLİKLE null OLMALIDIR.
   - "%8 nakit iade", "%10 indirim" gibi değerler kar_payi_orani DEĞİL, odul_miktari veya indirim_orani'ne yazılır.

2. "Finansman tutarı" SADECE konut/ihtiyaç/taşıt finansmanında vardır.
   - "Asgari harcama", "ödül miktarı", "hesap bakiyesi", "indirim limiti" finansman_tutari DEĞİLDİR. Bu durumlarda null bırak.

3. "Tahsis ücreti" bankanın finansmanı kullandırırken aldığı tek seferlik ücrettir. Eğer metinde "tahsis ücreti ... TL" veya "tahsis ücreti alınmıyor" gibi ifade varsa çıkar.

4. "İndirim oranı" kampanyada belirtilen yüzdelik indirimdir. Örn: "%10 indirim", "%5 iskonto". Bu kar_payi_orani DEĞİLDİR.

5. "Alışveriş puanı" kart kampanyalarında geçen puan/mil/worldpuan değeridir. Örn: "100 worldpuan", "500 mil".

6. Eğer değer net değilse veya "belirsiz"/"dolaylı" ise null bırak.
   - ASLA "belirsiz", "yok" gibi string'ler yazma.

7. Tutar aralığı varsa maximum tutarı al.

KATILIM BANKACILIĞI TERMINOLOJİSİ:
- Faiz YERİNE "kâr payı oranı"
- Kredi YERİNE "finansman"
- "Avantajlı kâr payı", "özel oran", "düşük maliyetli" = düşük kar payı
- "Masrafsız", "dosya masrafı alınmıyor" = maliyet avantajı
- "Tahsis ücreti" = finansmanın kullandırılması için alınan ücret
- "İndirim oranı" = kampanyaya özel yüzdelik indirim
- "Alışveriş puanı" = harcamalardan kazanılan puan/mil

KAMPANYA TÜRLERİ (SADECE bunlardan birini seç):
KonutFinansmaniKampanyasi, IhtiyacFinansmaniKampanyasi (Not: Alışveriş kredileri ve pratik finansman kartları buraya girer), TasitFinansmaniKampanyasi,
KartKampanyasi, AlisverisPuaniKampanyasi, YeniMusteriKampanyasi, YatirimUrunuKampanyasi, Diger

HEDEF KİTLE (birden fazla olabilir):
YeniMusteri, MevcutMusteri, MaasMusteri, DijitalMusteri, Bireysel, Ticari, Genel

ÇIKTI FORMATI (Aşağıdaki yapıyı kopyala, ancak < > içindeki alanları metne göre doldur. SADECE JSON VER):
{{
  "kar_payi_orani": "<Varsa oran, yoksa null>",
  "vade_suresi_ay": <Sadece sayı veya null>,
  "finansman_tutari": "<Varsa tutar, yoksa null>",
  "tahsis_ucreti": "<Varsa ücret, yoksa null>",
  "kampanya_turu": "<Listeden uygun olanı seç>",
  "hedef_kitle": ["<Listeden uygun olanları seç>"],
  "odul_miktari": "<Varsa ödül/indirim miktarı, yoksa null>",
  "indirim_orani": "<Varsa indirim yüzdesi, yoksa null>",
  "alisveris_puani": "<Varsa puan/mil miktarı, yoksa null>",
  "masraf_bilgisi": "<Varsa masraf durumu, yoksa null>",
  "kampanya_suresi": "<Tarih veya null>",
  "kampanya_kosullari": ["koşul 1", "koşul 2"],
  "yorum": "<Kampanyanın kısa özeti>"
}}

KAMPANYA METNİ:
---
{text}
---

KAYNAK: {banka}
ETİKET: {etiket}

SADECE JSON:"""


def _parse_llm_response(response: str) -> dict:
    # Modelin olası markdown (```json) blokları üretmesi senaryosuna 
    # karşı esnek (resilient) JSON ayrıştırma mantığı.
    if not response or not response.strip():
        raise ValueError("LLM boş yanıt döndürdü")

    text = response.strip()

    m = re.search(r"```json\s*(\{.*?\})\s*```", text, re.DOTALL)
    if m:
        return json.loads(m.group(1))

    m = re.search(r"```\s*(\{.*?\})\s*```", text, re.DOTALL)
    if m:
        return json.loads(m.group(1))

    m = re.search(r"(\{.*\})", text, re.DOTALL)
    if m:
        return json.loads(m.group(1))

    lines = text.split("\n")
    for i, line in enumerate(lines):
        if line.strip().startswith("{"):
            candidate = "\n".join(lines[i:])
            m = re.search(r"(\{.*\})", candidate, re.DOTALL)
            if m:
                return json.loads(m.group(1))

    raise ValueError(f"JSON parse edilemedi. Ham yanıt: {text[:200]}")


def _sanitize_llm_output(data: dict) -> dict:
    # LLM Sanatization: Modelin "yok", "belirsiz" gibi string olarak döndürdüğü
    # veritabanına uymayan tipleri standart null değerlere zorlayan güvenlik filtresi.
    null_keywords = {"belirsiz", "dolaylı", "dolaylı ifade", "yok", "bilinmiyor", "null", "none", ""}

    cleaned = {}
    for key, val in data.items():
        if isinstance(val, str) and val.lower().strip() in null_keywords:
            cleaned[key] = None
        else:
            cleaned[key] = val

    kar = cleaned.get("kar_payi_orani")
    if kar and isinstance(kar, str):
        odul_keywords = ["puan", "iade", "indirim", "world", "altın", "çek", "kazan", "hediye", "cashback"]
        if any(kw in kar.lower() for kw in odul_keywords):
            cleaned["kar_payi_orani"] = None
            if not cleaned.get("odul_miktari"):
                cleaned["odul_miktari"] = kar

    tutar = cleaned.get("finansman_tutari")
    if tutar and isinstance(tutar, str):
        odul_keywords = [
            "ödül", "harcama", "bakiye", "limit", "asgari", "üzeri", 
            "çek", "puan", "iade", "indirim", "hediye", "kazan", "worldpuan",
            "altın", "gram", "davet", "kodu", "kupon", "fırsat"
        ]
        if any(kw in tutar.lower() for kw in odul_keywords):
            cleaned["finansman_tutari"] = None
            if not cleaned.get("odul_miktari"):
                cleaned["odul_miktari"] = tutar

    # v6: İndirim oranını ve alışveriş puanını temizle
    indirim = cleaned.get("indirim_orani")
    if indirim and isinstance(indirim, str):
        if "kar payı" in indirim.lower() or "kâr payı" in indirim.lower():
            cleaned["indirim_orani"] = None
            if not cleaned.get("kar_payi_orani"):
                cleaned["kar_payi_orani"] = indirim

    return cleaned


def llm_extractor_node(state: PipelineState) -> PipelineState:
    llm = OllamaLLM(
        model=OLLAMA_MODEL,
        base_url=OLLAMA_BASE_URL,
        temperature=0.05,
        num_predict=1200,
    )

    # Regex sonuçlarını context olarak ver
    regex_context = json.dumps(state["regex_results"], ensure_ascii=False, indent=2)
    if not regex_context or regex_context == "{}":
        regex_context = "Regex hiçbir değer bulamadı."

    prompt = EXTRACTION_PROMPT.format(
        text=state["raw_text"],
        banka=state["banka_adi"],
        etiket=state.get("liste_etiket", ""),
        regex_context=regex_context,
    )

    # Hata Toleransı (Resilience): Yanıtın bozuk JSON gelmesi ihtimaline karşı 
    # otomatik yeniden deneme (retry) mekanizması.
    max_retries = 2
    for attempt in range(max_retries):
        print(f"[LLM] {OLLAMA_MODEL} çağrılıyor... (deneme {attempt + 1}/{max_retries})")
        try:
            response = llm.invoke(prompt)
            state["llm_raw_response"] = response

            parsed = _parse_llm_response(response)
            sanitized = _sanitize_llm_output(parsed)
            state["llm_results"] = sanitized

            print(f"[LLM] Başarılı. Tür: {sanitized.get('kampanya_turu')}, "
                  f"kar_payi: {sanitized.get('kar_payi_orani')}, "
                  f"tutar: {sanitized.get('finansman_tutari')}, "
                  f"vade: {sanitized.get('vade_suresi_ay')}, "
                  f"indirim: {sanitized.get('indirim_orani')}, "
                  f"puan: {sanitized.get('alisveris_puani')}")
            return state

        except Exception as e:
            print(f"[LLM HATA] {e}")
            if attempt < max_retries - 1:
                print("[LLM] Tekrar deneniyor...")
            else:
                state["llm_results"] = {}
                state["validation_errors"] = [f"LLM parse hatası: {e}"]

    return state


# Node 3: Validator

def validator_node(state: PipelineState) -> PipelineState:
    # Veri Kalitesi (Data Quality) Kontrolü: 
    # LLM ve Regex katmanından çıkan verilerin finansal mantığa
    # oturup oturmadığını (örn. %50 üstü kâr payı olamaz) denetler.
    errors = []
    combined = dict(state["regex_results"])
    if state.get("llm_results"):
        combined.update(state["llm_results"])

    kar = combined.get("kar_payi_orani")
    if kar is not None:
        kar_norm = combined.get("kar_payi_orani_normalized")
        if kar_norm is not None:
            try:
                kar_val = float(kar_norm)
                if kar_val > 50:
                    errors.append(f"Kâr payı %50'den yüksek: {kar_val}%")
                if kar_val < 0.01:
                    errors.append(f"Kâr payı çok düşük: {kar_val}%")
            except (ValueError, TypeError):
                pass
        else:
            kar_clean = str(kar).replace("%", "").replace(",", ".").strip()
            try:
                kar_val = float(kar_clean)
                if kar_val > 50:
                    errors.append(f"Kâr payı %50'den yüksek: {kar_val}%")
            except (ValueError, TypeError):
                pass

    vade = combined.get("vade_suresi_ay")
    if vade is not None:
        try:
            vade_int = int(vade)
            if vade_int > 360 or vade_int < 1:
                errors.append(f"Vade süresi mantık dışı: {vade_int} ay")
        except (ValueError, TypeError):
            errors.append(f"Vade süresi sayı değil: {vade}")

    tutar = combined.get("finansman_tutari")
    if tutar is not None and isinstance(tutar, str):
        m = re.search(r"(\d+[.,]?\d*)", tutar.replace(" ", ""))
        if m:
            try:
                val = float(m.group(1).replace(".", "").replace(",", ""))
                if val > 10_000_000:
                    errors.append(f"Tutar çok yüksek: {val}")
                if val < 100 and "finansman" in state["raw_text"].lower():
                    errors.append(f"Finansman tutarı çok düşük: {val}")
            except (ValueError, TypeError):
                pass

    # v6: İndirim oranı validasyonu
    indirim = combined.get("indirim_orani")
    if indirim is not None:
        ind_clean = str(indirim).replace("%", "").replace(",", ".").strip()
        try:
            ind_val = float(ind_clean)
            if ind_val > 100:
                errors.append(f"İndirim oranı %100'den yüksek: {ind_val}%")
        except (ValueError, TypeError):
            pass

    state["validation_errors"] = errors
    if errors:
        print(f"[Validator] Uyarı: {errors}")
    else:
        print("[Validator] Geçerli.")

    return state


# Node 4: Normalizer

def _extract_max_tutar(text: str) -> Optional[str]:
    range_pattern = re.compile(
        r"(\d{1,3}(?:[.,]?\d{3})*)\s*(?:TL|₺).*?(?:ile|ve|\-).*?(\d{1,3}(?:[.,]?\d{3})*)\s*(?:TL|₺)",
        re.IGNORECASE
    )
    m = range_pattern.search(text)
    if m:
        return f"{m.group(2)} TL"
    return None

def _normalize_sayi(deger):
    # Standardizasyon: Farklı kaynaklardan gelen binlik/ondalık ayraç formatlarını
    # (nokta veya virgül) veritabanı ile uyumlu tek bir tip (float) yapıya dönüştürür.
    if not deger:
        return None

    # TL, ₺, %, boşluk temizle
    metin = str(deger).upper().replace("TL", "").replace("₺", "").replace("%", "").replace(" ", "")

    # Sadece rakam, nokta ve virgülü al
    eslesme = re.search(r'[\d\.\,]+', metin)
    if not eslesme:
        return None

    sayi_str = eslesme.group(0)

    # Türkiye formatı mantığı (Nokta ve Virgül senaryoları)
    if '.' in sayi_str and ',' in sayi_str:
        sayi_str = sayi_str.replace('.', '').replace(',', '.')
    elif ',' in sayi_str:
        sayi_str = sayi_str.replace(',', '.')
    elif '.' in sayi_str:
        bolumler = sayi_str.split('.')
        if len(bolumler[-1]) == 3: # Virgülden sonra 3 rakam varsa bu binlik ayıracıdır
            sayi_str = sayi_str.replace('.', '')

    try:
        return float(sayi_str)
    except ValueError:
        return None

def normalizer_node(state: PipelineState) -> PipelineState:
    regex = state.get("regex_results", {})
    llm = state.get("llm_results", {})

    def pick(field: str, regex_key: str = None):
        if field in llm and llm[field] is not None:
            return llm[field]
        if regex_key and regex_key in regex:
            return regex[regex_key]
        if field in regex:
            return regex[field]
        return None

    # Kâr payı
    kar_raw = pick("kar_payi_orani", "kar_payi_orani")
    kar_norm = _normalize_sayi(kar_raw)

    # Tutar
    tutar_raw = pick("finansman_tutari", "finansman_tutari")
    if not tutar_raw:
        tutar_raw = _extract_max_tutar(state["raw_text"])
    tutar_norm = _normalize_sayi(tutar_raw)

    # v6: Yeni alanlar
    tahsis_raw = pick("tahsis_ucreti", "tahsis_ucreti")
    tahsis_norm = _normalize_sayi(tahsis_raw)

    indirim_raw = pick("indirim_orani", "indirim_orani")
    indirim_norm = _normalize_sayi(indirim_raw)

    puan_raw = pick("alisveris_puani", "alisveris_puani")

    # KESİN İŞ KURALI
    kampanya_turu = llm.get("kampanya_turu") if llm else "Diger"
    ham_metin_kucuk = state["raw_text"].lower()

    finansman_olmayan_turler = [
        "KartKampanyasi",
        "AlisverisPuaniKampanyasi",
        "YeniMusteriKampanyasi",
        "YatirimUrunuKampanyasi",
    ]

    if kampanya_turu in finansman_olmayan_turler:
        kontrol_metni = ham_metin_kucuk.replace("kredi kart", "").replace("kredikart", "")

        istisna_kelimeler = [
            "ihtiyaç finansmanı", "ihtiyac finansmani",
            "taşıt finansmanı", "tasit finansmani",
            "konut finansmanı", "konut finansmani",
            "alışveriş finansmanı", "alisveris finansmani",
            "pratik finansman", "ihtiyaç kart", "ihtiyac kart",
            "alışveriş kredisi", "alisveris kredisi", "sağlık kredisi"
        ]

        if not any(kelime in kontrol_metni for kelime in istisna_kelimeler):
            tutar_raw = None
            tutar_norm = None
            kar_raw = None
            kar_norm = None
            tahsis_raw = None
            tahsis_norm = None
  
    # Kampanya süresi
    sure = pick("kampanya_suresi")
    if not sure and state.get("liste_bitis_tarihi"):
        sure = state["liste_bitis_tarihi"]

    final = {
        "banka_bilgisi": state["banka_adi"],
        "kaynak": state.get("kaynak", "bilinmiyor"),
        "url": state["url"],
        # Finansman Bilgileri
        "kar_payi_orani": kar_raw,
        "kar_payi_orani_normalized": kar_norm,
        "finansman_tutari": tutar_raw,
        "finansman_tutari_normalized": tutar_norm,
        "vade_suresi_ay": pick("vade_suresi_ay", "vade_suresi_ay"),
        "taksit_sayisi": pick("taksit_sayisi", "taksit_sayisi"),
        "tahsis_ucreti": tahsis_raw,
        "tahsis_ucreti_normalized": tahsis_norm,
        "masraf_bilgisi": pick("masraf_bilgisi"),
        # Kampanya Bilgileri
        "kampanya_turu": kampanya_turu,
        "odul_miktari": pick("odul_miktari"),
        "indirim_orani": indirim_raw,
        "indirim_orani_normalized": indirim_norm,
        "alisveris_puani": puan_raw,
        "kampanya_suresi": sure,
        "kampanya_kosullari": llm.get("kampanya_kosullari", []) if llm else [],
        # Hedef Kitle
        "hedef_kitle": llm.get("hedef_kitle", ["GenelMusteri"]) if llm else ["GenelMusteri"],
        # Meta
        "validation_errors": state.get("validation_errors", []),
        "regex_confidence": state["regex_confidence"],
        "llm_used": True,
        "extracted_at": datetime.now().isoformat(),
    }

    state["final_output"] = final
    print(f"[Normalizer] Çıktı üretildi. Tür: {kampanya_turu}, "
          f"finansman: {tutar_raw}, kar_payi: {kar_raw}, "
          f"indirim: {indirim_raw}, puan: {puan_raw}")
    return state


# Graph Build

def build_pipeline():
    workflow = StateGraph(PipelineState)

    workflow.add_node("regex", regex_extractor_node)
    workflow.add_node("llm", llm_extractor_node)
    workflow.add_node("validator", validator_node)
    workflow.add_node("normalizer", normalizer_node)

    workflow.set_entry_point("regex")
    workflow.add_edge("regex", "llm")
    workflow.add_edge("llm", "validator")
    workflow.add_edge("validator", "normalizer")
    workflow.add_edge("normalizer", END)

    return workflow.compile()


# CLI / Demo

def main():
    sample = {
        "kaynak": "albaraka_turk",
        "banka_adi": "Albaraka Türk Katılım Bankası",
        "url": "[https://www.albaraka.com.tr/tr/kampanyalar/detay/vade-farksiz-kampanyasi](https://www.albaraka.com.tr/tr/kampanyalar/detay/vade-farksiz-kampanyasi)",
        "liste_etiket": "Yeni Müşterilere Özel",
        "liste_bitis_tarihi": "Son gün 31 Aralık",
        "detay_metin": (
            "Kâr payı yok. Beklemek yok. 140.000 TL’ye kadar vade farksız destek Albaraka’da!\n\n"
            "Şimdi Albaraka Mobil’den müşteri olanlar, %0 kâr payı ile 40.000 TL’ye kadar "
            "Pratik Finansman Kart (İhtiyaç Finansmanı) kullanabiliyor ve 100.000 TL’ye varan "
            "seçili sektörlerde vade farksız taksitli alışveriş fırsatından yararlanabiliyor.\n\n"
            "Hemen siz de şubeye gitmeden Albaraka Türk müşterisi olun;\n"
            "• %0 kâr paylı 40.000 TL’ye kadar, 3 aya varan ödemesiz dönem ve 4 taksitli Pratik Finansman Kart ve,\n"
            "• 100.000 TL’ye kadar vade farksız taksitli alışveriş fırsatını bir arada yakalayın.\n\n"
            "Toplamda 140.000 TL’ye kadar vade farksız finansman desteği için Albaraka Türk sizleri bekliyor."
        ),
    }

    print("=" * 70)
    print("LANGGRAPH + OLLAMA NLP PIPELINE v6")
    print("(Tüm şartname alanları destekleniyor)")
    print("=" * 70)
    print(f"Model: {OLLAMA_MODEL}")
    print("=" * 70)

    app = build_pipeline()

    initial_state: PipelineState = {
        "raw_text": sample["detay_metin"],
        "banka_adi": sample["banka_adi"],
        "url": sample["url"],
        "liste_etiket": sample["liste_etiket"],
        "liste_bitis_tarihi": sample["liste_bitis_tarihi"],
        "kaynak": sample.get("kaynak", "bilinmiyor"),
        "regex_results": {},
        "regex_confidence": 0.0,
        "llm_results": {},
        "llm_raw_response": "",
        "validation_errors": [],
        "final_output": {},
        "retry_count": 0,
    }

    print("\nPipeline çalıştırılıyor...\n")
    result = app.invoke(initial_state)

    print("\n" + "=" * 70)
    print("FİNAL ÇIKTI")
    print("=" * 70)
    print(json.dumps(result["final_output"], ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
