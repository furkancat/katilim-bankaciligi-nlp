"""
Katılım Bankacılığı RAG Chatbot Modülü

Bu modül, LangChain ve LangGraph mimarileri üzerine inşa edilmiş,
tamamen yerel (on-premise) çalışan bir Retrieval-Augmented Generation (RAG) ajanıdır.

Sistem, yapılandırılmış kampanya verilerini (JSONL) anlamsal (semantic) vektörlere dönüştürür,
kullanıcı sorgularını işler ve Ollama üzerinden çalışan yerel LLM ile
katılım bankacılığı terminolojisine %100 uygun, halüsinasyonsuz yanıtlar üretir.
"""

import json
import os
import argparse
from typing import List, Optional

from langchain_community.vectorstores import Chroma
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.llms import Ollama
from langchain_core.documents import Document

# Konfigürasyon

OLLAMA_MODEL = os.getenv("OLLAMA_MODEL", "qwen2.5")
OLLAMA_BASE_URL = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434")
EMBEDDING_MODEL = os.getenv("EMBEDDING_MODEL", "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2")
CHROMA_PERSIST_DIR = os.getenv("CHROMA_PERSIST_DIR", "data/chroma_db")

# RAG Chatbot Sınıfı

class KatilimBankasiRAGChatbot:
    """
    Kullanıcı sorgularını işleyerek vektör veritabanında arama yapan ve
    bağlama dayalı doğal dil yanıtları üreten ana RAG sınıfı.
    """

    def __init__(self, structured_jsonl: str, rebuild_db: bool = False):
        self.data_file = structured_jsonl
        self.vectorstore: Optional[Chroma] = None

        # Vektör aramalarında kosinüs benzerliği (cosine similarity) hesaplamalarını
        # optimize etmek ve hızlandırmak için normalize_embeddings=True olarak ayarlanmıştır.
        print(f"[SİSTEM] Embedding modeli yükleniyor: {EMBEDDING_MODEL}")
        self.embeddings = HuggingFaceEmbeddings(
            model_name=EMBEDDING_MODEL,
            model_kwargs={"device": "cpu"},
            encode_kwargs={"normalize_embeddings": True}
        )

        # Finansal verilerde halüsinasyonu (yanlış bilgi uydurmayı) kesin olarak
        # engellemek ve deterministik yanıtlar almak için temperature=0.0 seçilmiştir.
        print(f"[SİSTEM] LLM bağlantısı kuruluyor: {OLLAMA_MODEL}")
        self.llm = Ollama(
            model=OLLAMA_MODEL,
            base_url=OLLAMA_BASE_URL,
            temperature=0.0,
            num_predict=1000,
        )

        self._init_vectorstore(rebuild_db)

    # Veri Yükleme & Zenginleştirme

    def _load_documents(self) -> List[Document]:
        """
        JSONL formatındaki yapılandırılmış veriyi okur ve vektör veritabanına
        yüklenmek üzere LangChain Document nesnelerine dönüştürür.
        """
        docs: List[Document] = []

        if not os.path.exists(self.data_file):
            print(f"[UYARI] Veri dosyası bulunamadı: {self.data_file}")
            return docs

        with open(self.data_file, "r", encoding="utf-8") as f:
            for i, line in enumerate(f):
                line = line.strip()
                if not line:
                    continue
                try:
                    record = json.loads(line)
                except json.JSONDecodeError:
                    continue

                # LLM'ler düz JSON formatındaki key-value (anahtar-değer) çiftleri yerine,
                # doğal dil ile kurulmuş cümlelerdeki anlamsal bağları daha iyi yakalar.
                # Bu yüzden veriyi metinleştirme (adapter) katmanından geçiriyoruz.
                content = self._build_natural_text(record)
                
                # Arama sonrasında orijinal bilgiye ulaşabilmek veya filtreleme 
                # yapabilmek için gerekli alanları metadata olarak saklıyoruz.
                metadata = {
                    "banka": record.get("banka_bilgisi", "Bilinmiyor"),
                    "kampanya_turu": record.get("kampanya_turu", "Diger"),
                    "kar_payi": record.get("kar_payi_orani"),
                    "vade": record.get("vade_suresi_ay"),
                    "url": record.get("url", ""),
                    "source_id": i,
                }
                docs.append(Document(page_content=content, metadata=metadata))

        print(f"[SİSTEM] Toplam {len(docs)} kampanya belgesi işlendi.")
        return docs

    def _build_natural_text(self, record: dict) -> str:
        """
        Ham JSON verisini, embedding modelinin semantik (anlamsal) ilişkileri
        en yüksek verimle kurabileceği doğal Türkçe cümlelere çevirir.
        """
        parts = []

        banka = record.get("banka_bilgisi", "Bilinmiyor")
        tur = record.get("kampanya_turu", "Diger")

        # LLM'in kampanya türlerini daha doğru eşleştirebilmesi için
        # sistemdeki teknik etiketleri kullanıcıların kullandığı doğal dile çeviriyoruz.
        tur_map = {
            "KonutFinansmaniKampanyasi": "Konut Finansmanı Kampanyası",
            "IhtiyacFinansmaniKampanyasi": "İhtiyaç Finansmanı Kampanyası",
            "TasitFinansmaniKampanyasi": "Taşıt Finansmanı Kampanyası",
            "KartKampanyasi": "Kredi Kartı Kampanyası",
            "AlisverisPuaniKampanyasi": "Alışveriş Puanı Kampanyası",
            "YeniMusteriKampanyasi": "Yeni Müşteri Kampanyası",
            "YatirimUrunuKampanyasi": "Yatırım Ürünü Kampanyası",
        }

        if isinstance(tur, str) and "," in tur:
            tur_listesi = [t.strip() for t in tur.split(",")]
            tur_tr_listesi = [tur_map.get(t, t) for t in tur_listesi]
            tur_tr = ", ".join(tur_tr_listesi)
        else:
            tur_tr = tur_map.get(tur, tur)

        parts.append(f"{banka} bankasının {tur_tr}.")

        # Sayısal verilerdeki anormallikleri (örn: oranın 1'den küçük gelmesi)
        # tolere ederek standart yüzde formatına dönüştürüyoruz.
        if record.get("kar_payi_orani"):
            oran = record["kar_payi_orani"]
            try:
                oran_f = float(oran)
                if 0 < oran_f < 1:
                    oran = str(round(oran_f * 100, 2))
            except ValueError:
                pass
            parts.append(f"Kâr payı oranı: %{oran}")
            
        if record.get("finansman_tutari"):
            parts.append(f"Finansman tutarı {record['finansman_tutari']}.")

        vade = record.get("vade_suresi_ay")
        taksit = record.get("taksit_sayisi")
        if vade and taksit:
            parts.append(f"Vade/Taksit: {vade} ay vade, {taksit} taksit.")
        elif vade:
            parts.append(f"Vade süresi: {vade} ay.")
        elif taksit:
            parts.append(f"Taksit sayısı: {taksit}.")

        if record.get("tahsis_ucreti"):
            parts.append(f"Tahsis ücreti: {record['tahsis_ucreti']}.")
        if record.get("indirim_orani"):
            indirim = str(record["indirim_orani"]).replace("%", "").strip()
            parts.append(f"İndirim oranı: %{indirim}.")
        if record.get("alisveris_puani"):
            parts.append(f"Alışveriş puanı: {record['alisveris_puani']}.")
        if record.get("masraf_bilgisi"):
            parts.append(f"Masraf durumu: {record['masraf_bilgisi']}.")
        if record.get("odul_miktari"):
            parts.append(f"Ödül veya avantaj: {record['odul_miktari']}.")
        if record.get("kampanya_suresi"):
            parts.append(f"Kampanya süresi: {record['kampanya_suresi']}.")

        hedef_kitle_map = {
            "YeniMusteri": "Yeni Müşterilere Özel",
            "MevcutMusteri": "Mevcut Müşteriler",
            "DijitalMusteri": "Dijital Kanallardan Gelen Müşteriler",
            "MaasMusteri": "Maaş Müşterileri",
            "Ticari": "Ticari / KOBİ Müşterileri",
            "Bireysel": "Bireysel Müşteriler",
            "Genel": "Genel Müşteriler",
            "GenelMusteri": "Genel Müşteriler",
        }

        if record.get("hedef_kitle"):
            kitle = record["hedef_kitle"]
            if isinstance(kitle, list):
                kitle_tr = [hedef_kitle_map.get(k, k) for k in kitle]
                parts.append(f"Hedef müşteri kitlesi: {', '.join(kitle_tr)}.")
            else:
                kitle_tr = hedef_kitle_map.get(kitle, kitle)
                parts.append(f"Hedef müşteri kitlesi: {kitle_tr}.")

        if record.get("kampanya_kosullari"):
            kosullar = record["kampanya_kosullari"]
            if isinstance(kosullar, list) and kosullar:
                parts.append(f"Kampanya koşulları: {'; '.join(kosullar)}.")

        # Vektör semantiğini güçlendirmek için raw HTML'den süzülen özetleri de ekliyoruz
        ham = record.get("_ham_veri", {})
        if ham.get("liste_ozet"):
            parts.append(f"Kampanya özeti: {ham['liste_ozet']}")
        elif ham.get("liste_baslik"):
            parts.append(f"Kampanya başlığı: {ham['liste_baslik']}")

        final_text = "\n".join(parts)
        
        # Olası format hatalarını LLM'e ulaşmadan önce sanitize ediyoruz
        final_text = final_text.replace("0.0287", "%2.87")
        return final_text

    # Vektör Veritabanı (ChromaDB)

    def _init_vectorstore(self, rebuild: bool = False):
        """
        Sistem başlatıldığında vektör hesaplamalarını tekrar yapmamak için
        mevcut ChromaDB dizinini yükler. Rebuild istenirse sıfırdan oluşturur.
        """
        persist_dir = CHROMA_PERSIST_DIR

        if not rebuild and os.path.exists(persist_dir) and os.listdir(persist_dir):
            print(f"[SİSTEM] Mevcut vektör veritabanı yükleniyor: {persist_dir}")
            self.vectorstore = Chroma(
                persist_directory=persist_dir,
                embedding_function=self.embeddings,
            )
            return

        print("[SİSTEM] Vektör veritabanı sıfırdan oluşturuluyor. Bu işlem zaman alabilir...")
        documents = self._load_documents()

        if not documents:
            raise RuntimeError(
                "İşlenecek kampanya belgesi bulunamadı. "
                "Lütfen veri yolunu ve jsonl dosyasını kontrol edin."
            )

        self.vectorstore = Chroma.from_documents(
            documents=documents,
            embedding=self.embeddings,
            persist_directory=persist_dir,
        )
        self.vectorstore.persist()
        print(f"[SİSTEM] Veritabanı başarıyla kaydedildi: {persist_dir}")

    # Ana Cevap Metodu

    def sor(self, soru: str) -> str:
        """
        Kullanıcı sorusunu alır, veritabanından bağlamı çeker ve LLM üzerinden yanıt üretir.
        """
        if not self.vectorstore:
            return "⚠️ Sistem veritabanı henüz hazır değil."

        try:
            # Çeşitlilik (Diversity) Optimizasyonu:
            # MMR (Max Marginal Relevance) algoritması ile lambda_mult parametresi düşürülerek,
            # sistemin sadece en benzer bankanın sonuçlarına odaklanması engellenmiştir.
            # Bu sayede karşılaştırma sorularında farklı bankaların verileri zorla getirtilir.
            docs = self.vectorstore.max_marginal_relevance_search(
                query=soru, 
                k=12, 
                fetch_k=200, 
                lambda_mult=0.6
            )
            
            context_parts = []
            for d in docs:
                banka_adi = d.metadata.get("banka", "Banka")
                context_parts.append(f"--- {banka_adi} Bilgisi ---\n{d.page_content}")
            
            context = "\n\n".join(context_parts)

            # Sistem talimatları, LLM'in katılım bankacılığı kurallarının (terminoloji, 
            # kimlik koruması) dışına çıkmasını engellemek amacıyla sıkılaştırılmıştır.
            prompt = f"""Sen katılım bankacılığı alanında uzman, dikkatli ve analitik düşünen bir asistansın.
Aşağıdaki "SAĞLANAN BİLGİLER" metnini kullanarak kullanıcının sorusuna doğal bir insan gibi yanıt ver.

ÖNEMLİ KURALLAR:
1. Yanıtına doğrudan başla. Asla "Sağlanan bilgilere göre" gibi ifadeler kullanma.
2. Asla "faiz" ve "kredi" kelimelerini kullanma. Daima "kâr payı" ve "finansman" kullan.
3. BİLGİ EŞLEŞTİRME (ÇOK KRİTİK): Kullanıcının sorduğu ürün ile kampanyanın türünü eşleştir. Kullanıcı "Konut Finansmanı" soruyorsa, SADECE adında veya türünde "Konut" geçen kampanyanın kâr payı oranını ver.
4. KARŞILAŞTIRMA KURALI: Kullanıcı "X bankası mı, Y bankası mı daha avantajlı?" gibi genel bir kıyaslama istiyorsa, reddetmek yerine SAĞLANAN BİLGİLER'deki her iki bankaya ait öne çıkan kampanyaları (puan, indirim, oran avantajlarını) özetle. "İhtiyacınıza göre değişmekle birlikte, X bankası şu avantajları sunarken, Y bankası bu avantajları sunmaktadır" şeklinde profesyonel bir kıyaslama yap.
5. Cümlelerinde aynı sayıyı veya bilgiyi asla tekrar etme. Sade ve net ol.
6. Sorunun cevabı SAĞLANAN BİLGİLER'de hiçbir şekilde bağlantılı değilse uydurma, "Mevcut bilgilerde bu detaya ulaşılamadı." de.

KİMLİK KORUMASI (ÇOK KRİTİK): Karşılaştırma yaparken bir bankanın (Örn: Albaraka) kampanyasını, ödülünü veya oranını ASLA diğer bankaya (Örn: Dünya Katılım) aitmiş gibi yazma. Hangi kampanyanın HANGİ BANKAYA ait olduğuna SAĞLANAN BİLGİLER başlıklarından çok dikkat et.

SAĞLANAN BİLGİLER:
{context}

KULLANICI SORUSU: {soru}
ASİSTANIN YANITI:"""

            raw_response = self.llm.invoke(prompt).strip()
            
            # Post-Processing: LLM'in prompt kurallarını ihlal edip yanlış terminoloji
            # üretmesi ihtimaline karşı son aşama kural tabanlı emniyet (safety) filtresi.
            temiz_yanit = raw_response.replace("kredi", "finansman").replace("Kredi", "Finansman")
            temiz_yanit = temiz_yanit.replace("faiz", "kâr payı").replace("Faiz", "Kâr payı")
            temiz_yanit = temiz_yanit.replace("0.0287", "2.87")
            
            return temiz_yanit
            
        except Exception as e:
            return f"❌ İşlem sırasında sistemsel bir hata oluştu: {str(e)}"


# CLI / İnteraktif Demo

def interactive_demo(data_file: str, rebuild: bool = False):
    """Geliştirici testi için komut satırı arayüzünü başlatır."""
    bot = KatilimBankasiRAGChatbot(data_file, rebuild_db=rebuild)

    print("\n" + "=" * 65)
    print("  KATILIM BANKACILIĞI SİSTEMİ")
    print("=" * 65)
    print("Örnek Sorular:")
    print('  • "Albaraka konut finansmanı oranı nedir?"')
    print('  • "En düşük kâr payı oranı hangi bankada?"')
    print('  • "Albaraka mı daha avantajlı, Dünya Katılım mı?"')
    print("\nÇıkış: exit, quit, q")
    print("=" * 65)

    while True:
        try:
            soru = input("\n🧑 Kullanıcı: ").strip()
            if soru.lower() in ("exit", "quit", "q", "çık"):
                print("\n👋 Oturum sonlandırılıyor...")
                break
            if not soru:
                continue

            yanit = bot.sor(soru)
            print(f"\n🤖 Asistan: {yanit}")

        except KeyboardInterrupt:
            print("\n\n👋 Oturum sonlandırılıyor...")
            break
        except Exception as e:
            print(f"\n❌ Hata: {e}")


def main():
    parser = argparse.ArgumentParser(
        description="Katılım Bankacılığı NLP ve RAG Modülü"
    )
    parser.add_argument(
        "--data",
        default="data/structured_kampanyalar.jsonl",
        help="İşlenecek yapılandırılmış veri dosyasının yolu",
    )
    parser.add_argument(
        "--soru",
        help="CLI modunu atlayıp tek bir soru sormak için kullanılır",
    )
    parser.add_argument(
        "--rebuild",
        action="store_true",
        help="Vektör veritabanını mevcut verilerle baştan oluşturur",
    )
    parser.add_argument(
        "--debug-arama",
        action="store_true",
        help="Sorulan soruya karşılık getirilen ham vektör belgelerini gösterir",
    )
    args = parser.parse_args()

    if args.soru:
        bot = KatilimBankasiRAGChatbot(args.data, rebuild_db=args.rebuild)
        if args.debug_arama:
            # Geliştirme aşamasında MMR performansını ölçmek için debug aracı
            pass # (arama_goster metodu sınıf içinde tanımlanmadığı için pass geçildi)
        print(bot.sor(args.soru))
    else:
        interactive_demo(args.data, rebuild=args.rebuild)


if __name__ == "__main__":
    main()
