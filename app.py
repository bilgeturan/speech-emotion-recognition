import streamlit as st
import librosa
import numpy as np
import pickle
from tensorflow.keras.models import load_model
import sounddevice as sd
from scipy.io.wavfile import write
import pandas as pd
import io
import matplotlib.pyplot as plt
import tensorflow as tf

# --- AYARLAR ---
MODEL_YOLU = "duygu_tanima_modeli.h5"
SOZLUK_YOLU = "etiket_sozlugu.pkl"
SCALER_YOLU = "scaler.pkl"
KAYIT_SURESI = 4  
ORNEK_HIZI = 22050

# --- RENK VE EMOJİ PALETİ ---
duygu_temalari = {
    "angry":     {"renk": "#FF0000", "emoji": "😡", "tr": "KIZGIN"},
    "disgust":   {"renk": "#006400", "emoji": "🤢", "tr": "İĞRENMİŞ"},
    "fearful":   {"renk": "#800080", "emoji": "😨", "tr": "KORKMUŞ"},
    "happy":     {"renk": "#FFD700", "emoji": "😄", "tr": "MUTLU"},
    "neutral":   {"renk": "#808080", "emoji": "😐", "tr": "NÖTR"},
    "sad":       {"renk": "#0000FF", "emoji": "😢", "tr": "ÜZGÜN"},
    "surprised": {"renk": "#FF4500", "emoji": "😲", "tr": "ŞAŞIRMIŞ"},
    "calm":      {"renk": "#00FFFF", "emoji": "😌", "tr": "SAKİN"}
}

# --- SAYFA DÜZENİ ---
st.set_page_config(
    page_title="Duygu Analizi Projesi", 
    page_icon="🎙️", 
    layout="wide",
    initial_sidebar_state="expanded"
)

# --- CSS ---
st.markdown("""
    <style>
    .stButton>button { width: 100%; height: 3em; font-size: 20px; background-color: #FF4B4B; color: white; border-radius: 10px; }
    </style>
    """, unsafe_allow_html=True)

# --- HAFIZA (SESSION STATE) ---
if 'gecmis' not in st.session_state:
    st.session_state.gecmis = []
if 'son_sonuc' not in st.session_state: # En son yapılan analizi tutmak için
    st.session_state.son_sonuc = None

# --- YAN MENÜ ---
with st.sidebar:
    st.image("https://cdn-icons-png.flaticon.com/512/4712/4712009.png", width=100)
    st.title("Proje Hakkında")
    
    st.info("""
    Bu proje, ses dalgalarını analiz ederek konuşmacının duygu durumunu tahmin eden bir Yapay Zeka uygulamasıdır.
    
    **Kullanılan Teknolojiler:**
    - CNN (Convolutional Neural Network)
    - MFCC Ses Analizi
    - Python & TensorFlow
    """)
    
    st.divider()
    
    if st.button("🗑️ Geçmişi Temizle"):
        st.session_state.gecmis = []
        st.session_state.son_sonuc = None
        st.rerun()
        
    st.divider()
    st.write("Geliştirici: **Bilge Turan**")

st.title("🎙️ Sesten Duygu Tanıma Sistemi")
st.markdown("### Yapay Zeka Duygularınızı Analiz Ediyor...")
st.divider()

# --- MODEL YÜKLEME ---
@st.cache_resource
def model_yukle():
    model = load_model(MODEL_YOLU)
    with open(SOZLUK_YOLU, "rb") as f:
        encoder = pickle.load(f)
    with open(SCALER_YOLU, "rb") as f:
        scaler = pickle.load(f)
    return model, encoder, scaler

try:
    model, encoder, scaler = model_yukle()
except:
    st.error("⚠️ Model dosyaları eksik! Lütfen model_egitimi.py dosyasını çalıştır.")
    st.stop()

# --- SES İŞLEME ---
def sesi_hazirla(audio_data, sr, return_full_mfcc=False):
    y = audio_data.astype(np.float32)
    if y.max() > 1.0:
        y = y / 32768.0

    if y.ndim > 1:
        y = np.mean(y, axis=1)

    if sr != ORNEK_HIZI:
        y = librosa.resample(y, orig_sr=sr, target_sr=ORNEK_HIZI)

    maks_uzunluk = int(3 * ORNEK_HIZI)
    y = y[:maks_uzunluk]

    mfccs_full = librosa.feature.mfcc(y=y, sr=ORNEK_HIZI, n_mfcc=40)
    mfccs_scaled = np.mean(mfccs_full.T, axis=0)

    mfccs_scaled = mfccs_scaled.reshape(1, -1)
    mfccs_scaled = scaler.transform(mfccs_scaled)
    mfccs_scaled = np.expand_dims(mfccs_scaled, axis=2)

    if return_full_mfcc:
        return mfccs_scaled, mfccs_full
    return mfccs_scaled

def grad_cam_1d(model, input_data, class_idx):
    """1D CNN için Grad-CAM: hangi MFCC katsayıları tahmini etkiledi?"""
    try:
        # Son Conv1D katmanının index'ini bul
        conv_idx = None
        for i, layer in enumerate(model.layers):
            if 'conv' in layer.__class__.__name__.lower():
                conv_idx = i

        if conv_idx is None:
            return np.zeros(40)

        input_tensor = tf.convert_to_tensor(input_data, dtype=tf.float32)

        with tf.GradientTape() as tape:
            # Katmanları manuel ilerlet → conv çıktısını al
            x = input_tensor
            for i in range(conv_idx + 1):
                x = model.layers[i](x, training=False)
            conv_output = x
            tape.watch(conv_output)

            # Devam et → final tahmin
            for i in range(conv_idx + 1, len(model.layers)):
                x = model.layers[i](x, training=False)

            predictions = x
            loss = predictions[:, class_idx]

        grads = tape.gradient(loss, conv_output)
        if grads is None:
            print("Grad-CAM: gradyan None döndü")
            return np.zeros(40)

        pooled_grads = tf.reduce_mean(grads, axis=(0, 1))
        conv_output = conv_output[0]
        heatmap = conv_output @ pooled_grads[..., tf.newaxis]
        heatmap = tf.squeeze(heatmap).numpy()

        #heatmap = np.maximum(heatmap, 0)
        heatmap = np.abs(heatmap)
        if heatmap.max() > 0:
            heatmap = heatmap / heatmap.max()

        heatmap_resized = np.interp(
            np.linspace(0, len(heatmap) - 1, 40),
            np.arange(len(heatmap)),
            heatmap
        )
        return heatmap_resized
    except Exception as e:
        print(f"Grad-CAM hatası: {e}")
        return np.zeros(40)

# --- ANALİZ FONKSİYONU ---
def analizi_yap(audio_bytes, kaynak_tipi):
    try:
        import soundfile as sf
        y, sr = sf.read(io.BytesIO(audio_bytes))
        
        veri, mfcc_full = sesi_hazirla(y, sr, return_full_mfcc=True)
        tahminler = model.predict(veri)
        en_yuksek_indeks = np.argmax(tahminler)
        ham_tahmin = encoder.inverse_transform([en_yuksek_indeks])[0]

        # Grad-CAM: hangi MFCC katsayıları tahmini etkiledi
        with st.spinner("Modelin karar süreci analiz ediliyor..."):
            gradcam = grad_cam_1d(model, veri, en_yuksek_indeks)
        
        
        # Sonuç Paketi
        sonuc = {
           "ses": audio_bytes,
           "tahmin": ham_tahmin,
           "olasiliklar": tahminler[0],
           "kaynak": kaynak_tipi,
           "waveform": y[::50],
           "mfcc_full": mfcc_full,       # ← YENİ
           "grad_cam": gradcam,           # ← YENİ
}
        
        # 1. En son sonuç olarak kaydet (Ekrana büyük basmak için)
        st.session_state.son_sonuc = sonuc
        
        # 2. Geçmiş listesine ekle
        st.session_state.gecmis.insert(0, sonuc)
        
    except Exception as e:
        st.error(f"Hata: {e}")

# --- ARAYÜZ (GİRİŞ) ---
tab1, tab2 = st.tabs(["🎙️ Mikrofon", "📂 Dosya Yükle"])

st.subheader("📂 Ses Dosyası Yükle")
yuklenen_dosya = st.file_uploader("WAV Dosyası Seç", type=["wav"])
if yuklenen_dosya is not None:
    if st.button("🔍 Analiz Et"):
        analizi_yap(yuklenen_dosya.getvalue(), "Dosya Yükleme")

st.divider()

st.subheader("🎙️ Canlı Kayıt")
if st.button("KAYDI BAŞLAT"):
    with st.spinner("Dinliyorum..."):
        kayit = sd.rec(int(KAYIT_SURESI * ORNEK_HIZI), samplerate=ORNEK_HIZI, channels=1)
        sd.wait()
        sanal_dosya = io.BytesIO()
        write(sanal_dosya, ORNEK_HIZI, kayit)
        analizi_yap(sanal_dosya.getvalue(), "Canlı Kayıt")

# --- BÖLÜM 1: AKTİF SONUÇ (DEV EKRAN) ---
if st.session_state.son_sonuc:
    sonuc = st.session_state.son_sonuc
    tahmin = sonuc["tahmin"]
    tema = duygu_temalari.get(tahmin, {"renk": "gray", "emoji": "🤔", "tr": tahmin})
    
    st.divider()
    st.markdown("## 🎯 Analiz Sonucu")
    
    # 1. Ses ve Dalga Grafiği (BÜYÜK)
    col_audio, col_wave = st.columns([1, 2])
    with col_audio:
        st.write("**🎧 Kaydı Dinle:**")
        st.audio(sonuc["ses"])
    with col_wave:
        st.write("**📈 Ses Sinyali:**")
        st.line_chart(sonuc["waveform"], height=150)

    # 2. Sonuç Kutusu ve Grafik (BÜYÜK)
    col_res_1, col_res_2 = st.columns([1, 1])
    
    with col_res_1:
        st.markdown(f"""
        <div style="background-color: {tema['renk']}20; padding: 40px; border-radius: 20px; border-left: 15px solid {tema['renk']}; text-align: center; margin-top: 20px;">
            <h1 style="color: {tema['renk']}; margin:0; font-size: 50px;">{tema['emoji']}</h1>
            <h1 style="color: {tema['renk']}; margin:0; font-size: 40px;">{tema['tr']}</h1>
            <p style="font-size: 20px;">Yapay Zeka Kararı: <b>{tahmin.upper()}</b></p>
        </div>
        """, unsafe_allow_html=True)
        
    with col_res_2:
        st.write("**📊 Detaylı Olasılıklar:**")
        grafik_verisi = pd.DataFrame({
            "Duygu": encoder.classes_,
            "Olasılık": sonuc["olasiliklar"]
        }).set_index("Duygu")
        st.bar_chart(grafik_verisi, height=300) # Yüksekliği artırdık




# --- AÇIKLANABİLİRLİK ---
    if "mfcc_full" in sonuc and "grad_cam" in sonuc:
        st.divider()
        st.markdown("### 🔍 Modelin Karar Süreci")
        st.caption("Modelin tahmininde **neye baktığını** ve **hangi özelliklerin etkili olduğunu** gösteren analiz.")

        fig, (ax1, ax2) = plt.subplots(
            2, 1, figsize=(12, 6),
            gridspec_kw={'height_ratios': [3, 1]}
        )
        fig.patch.set_facecolor('#0E1117')

        # MFCC Heatmap (modelin gördüğü)
        im = ax1.imshow(sonuc["mfcc_full"], aspect='auto', origin='lower',
                        cmap='magma', interpolation='nearest')
        ax1.set_title("🎵 Modelin Gördüğü: MFCC Spektral İmza",
                      fontsize=13, color='white', pad=10)
        ax1.set_ylabel("MFCC Katsayısı", color='white')
        ax1.set_xlabel("Zaman (frame)", color='white')
        ax1.tick_params(colors='white')
        cbar = plt.colorbar(im, ax=ax1, fraction=0.025)
        cbar.ax.yaxis.set_tick_params(color='white')
        plt.setp(plt.getp(cbar.ax.axes, 'yticklabels'), color='white')

        # Grad-CAM Importance Bar
        importance = sonuc["grad_cam"]
        colors = plt.cm.Reds(0.35 + 0.65 * importance)
        ax2.bar(range(40), importance, color=colors, edgecolor='none')
        ax2.set_title("🎯 Modelin Odaklandığı Özellikler (Grad-CAM)",
                      fontsize=13, color='white', pad=10)
        ax2.set_xlabel("MFCC Katsayısı", color='white')
        ax2.set_ylabel("Önem", color='white')
        ax2.set_xlim(-0.5, 39.5)
        ax2.set_ylim(0, 1.1)
        ax2.set_facecolor('#0E1117')
        ax2.tick_params(colors='white')
        for spine in ax2.spines.values():
            spine.set_color('#444')
        ax1.set_facecolor('#0E1117')
        for spine in ax1.spines.values():
            spine.set_color('#444')

        plt.tight_layout()
        st.pyplot(fig)
        plt.close(fig)

        st.caption("Üstte: ses kaydının modele giden sayısal temsili (parlak alanlar = yüksek enerji). "
                   "Altta: tahmini en çok etkileyen MFCC katsayıları (kırmızı = yüksek etki).")

# --- BÖLÜM 2: GEÇMİŞ LİSTESİ (ALTTA) ---
if st.session_state.gecmis:
    st.divider()
    st.subheader("🗂️ Geçmiş Analizler")
    
    for i, kayit in enumerate(st.session_state.gecmis):
        # İlk kaydı (zaten yukarıda kocaman gösterdiğimiz için) atlayabiliriz veya küçük gösterebiliriz.
        # Karışıklık olmasın diye hepsini küçük liste olarak gösterelim.
        
        tahmin = kayit["tahmin"]
        tema = duygu_temalari.get(tahmin, {"renk": "gray", "emoji": "🤔", "tr": tahmin})
        
        with st.expander(f"#{len(st.session_state.gecmis)-i} - {tema['tr']} ({kayit['kaynak']})"):
            c1, c2 = st.columns([1, 3])
            with c1:
                st.audio(kayit["ses"])
                st.caption(f"Tahmin: {tahmin.upper()}")
            with c2:
                # Küçük grafik
                df_mini = pd.DataFrame({"Duygu": encoder.classes_, "Olasılık": kayit["olasiliklar"]}).set_index("Duygu")
                st.bar_chart(df_mini, height=100)