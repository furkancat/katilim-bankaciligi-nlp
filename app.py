"""
Katılım Bankacılığı Asistanı - Web Arayüzü (UI) Modülü

Bu modül, LangChain RAG altyapısını son kullanıcıya sunmak için Streamlit
kullanılarak geliştirilmiş interaktif sohbet (chatbot) arayüzüdür.
"""
import streamlit as st
from chatbot import KatilimBankasiRAGChatbot

# Kullanıcı deneyimini (UX) artırmak ve uygulamanın tarayıcıdaki görünümünü
# kurumsal bir standarda oturtmak için global sayfa yapılandırması ayarlanır.
st.set_page_config(
    page_title="Katılım Bankacılığı Asistanı",
    page_icon="🏦",
    layout="centered"
)

# RAM Optimizasyonu ve Performans:
# LLM (Qwen) ve Embedding modelleri devasa bellek tüketir. Streamlit'in mimarisi
# gereği her tetiklemede sayfa baştan render edildiği için, modellerin her seferinde
# yeniden RAM'e yüklenmesini engellemek adına önbellekleme (cache) kullanılır.
@st.cache_resource(show_spinner=False)
def init_bot():
    return KatilimBankasiRAGChatbot("data/structured_kampanyalar.jsonl", rebuild_db=False)

st.title("🏦 Katılım Bankacılığı Asistanı")
st.markdown("Katılım bankalarının güncel finansman, kart ve yatırım kampanyaları hakkında sorular sorabilir, bankaları kıyaslayabilirsiniz.")
st.divider()

with st.spinner("Yapay Zeka Motoru Yükleniyor (Sadece ilk girişte zaman alır)..."):
    bot = init_bot()

# Durum Yönetimi (State Management):
# HTTP protokolü state-less (durumsuz) olduğu için, kullanıcının önceki mesajlarını
# hatırlamak adına Streamlit'in session_state (oturum durumu) belleği kullanılır.
if "messages" not in st.session_state:
    st.session_state.messages = [
        {"role": "assistant", "content": "Merhaba! Ben Katılım Bankacılığı uzman asistanınızım. Size nasıl yardımcı olabilirim?"}
    ]

# Ekran her yenilendiğinde sohbet akışının (context) bozulmaması için
# geçmiş mesajlar UI üzerine sırayla yeniden çizilir.
for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])

# Kullanıcıdan asenkron girdi beklenir
if prompt := st.chat_input("Örn: Konut finansmanı için en düşük kâr payı oranı hangi bankada?"):
    
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)
    
    # Asistanın yanıt üretme sürecinde UX'i bozmamak için yükleniyor animasyonu gösterilir
    with st.chat_message("assistant"):
        with st.spinner("Yanıt üretiliyor..."):
            try:
                # RAG altyapısına sorgu atılır ve dönen temizlenmiş yanıt ekrana basılır
                cevap = bot.sor(prompt)
                st.markdown(cevap)
                
                st.session_state.messages.append({"role": "assistant", "content": cevap})
            except Exception as e:
                hata_mesaji = f"❌ Bir hata oluştu: {str(e)}"
                st.error(hata_mesaji)
                st.session_state.messages.append({"role": "assistant", "content": hata_mesaji})
