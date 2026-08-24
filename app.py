import streamlit as st
from chatbot import KatilimBankasiRAGChatbot

# 1. Sayfa ve Sekme Ayarları
st.set_page_config(
    page_title="Katılım Bankacılığı Asistanı",
    page_icon="🏦",
    layout="centered"
)

# 2. RAG Motorunu Önbelleğe Alma (Çok Kritik!)
# Bu sayede her mesajda ChromaDB ve Embedding modeli baştan yüklenmez, RAM şişmez.
@st.cache_resource(show_spinner=False)
def init_bot():
    return KatilimBankasiRAGChatbot("data/structured_kampanyalar.jsonl", rebuild_db=False)

# 3. Başlık ve Arayüz Tasarımı
st.title("🏦 Katılım Bankacılığı Asistanı")
st.markdown("Katılım bankalarının güncel finansman, kart ve yatırım kampanyaları hakkında sorular sorabilir, bankaları kıyaslayabilirsiniz.")
st.divider()

# Botu Başlat
with st.spinner("Yapay Zeka Motoru Yükleniyor (Sadece ilk girişte zaman alır)..."):
    bot = init_bot()

# 4. Sohbet Geçmişi Yönetimi (Session State)
if "messages" not in st.session_state:
    st.session_state.messages = [
        {"role": "assistant", "content": "Merhaba! Ben Katılım Bankacılığı uzman asistanınızım. Size nasıl yardımcı olabilirim?"}
    ]

# Geçmiş mesajları ekrana çiz
for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])

# 5. Kullanıcı Girişi ve Yanıt Üretimi
if prompt := st.chat_input("Örn: Konut finansmanı için en düşük kâr payı oranı hangi bankada?"):
    
    # Kullanıcı mesajını ekrana yaz ve geçmişe ekle
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)
    
    # Asistanın yanıtını oluştur (Yükleniyor animasyonu ile)
    with st.chat_message("assistant"):
        with st.spinner("Yanıt üretiliyor..."):
            try:
                # chatbot.py içindeki sor() fonksiyonunu çağırıyoruz
                cevap = bot.sor(prompt)
                st.markdown(cevap)
                
                # Başarılı yanıtı geçmişe ekle
                st.session_state.messages.append({"role": "assistant", "content": cevap})
            except Exception as e:
                hata_mesaji = f"❌ Bir hata oluştu: {str(e)}"
                st.error(hata_mesaji)
                st.session_state.messages.append({"role": "assistant", "content": hata_mesaji})