import librosa
import numpy as np
import pandas as pd
import os

# --- AYARLAR ---
ravdess_yolu = "./archive/"
tess_yolu = "./TESS/"
crema_yolu = "./CREMA/" 


ravdess_sozlugu = {
    '01': 'neutral', '02': 'calm', '03': 'happy', '04': 'sad',
    '05': 'angry', '06': 'fearful', '07': 'disgust', '08': 'surprised'
}

tess_cevirici = {
    'angry': 'angry', 'disgust': 'disgust', 'fear': 'fearful',
    'happy': 'happy', 'neutral': 'neutral', 'ps': 'surprised', 'sad': 'sad'
}

crema_cevirici = {
    'ANG': 'angry', 'DIS': 'disgust', 'FEA': 'fearful',
    'HAP': 'happy', 'NEU': 'neutral', 'SAD': 'sad'
}
def duyguyu_normalize_et(duygu):
    if duygu == 'calm':
        return 'neutral'
    return duygu
data = []

def veri_ekle(dosya_yolu, duygu):
    try:
        duygu = duyguyu_normalize_et(duygu)   # ← YENİ SATIR
        audio, sr = librosa.load(dosya_yolu, res_type='kaiser_fast', duration=3)
        mfccs = librosa.feature.mfcc(y=audio, sr=sr, n_mfcc=40)
        mfccs_scaled = np.mean(mfccs.T, axis=0)
        data.append([duygu] + mfccs_scaled.tolist())
    except: pass

print("Veri işleme başladı... RAVDESS + TESS + CREMA 🚀")

# 1. RAVDESS
if os.path.exists(ravdess_yolu):
    for root, dirs, files in os.walk(ravdess_yolu):
        for file in files:
            if file.endswith('.wav'):
                try:
                    duygu = ravdess_sozlugu.get(file.split('-')[2])
                    if duygu: veri_ekle(os.path.join(root, file), duygu)
                except: continue

# 2. TESS
if os.path.exists(tess_yolu):
    for root, dirs, files in os.walk(tess_yolu):
        for file in files:
            if file.endswith('.wav'):
                try:
                    duygu = tess_cevirici.get(file.split('.')[0].split('_')[-1].lower())
                    if duygu: veri_ekle(os.path.join(root, file), duygu)
                except: continue

# 3. CREMA
if os.path.exists(crema_yolu):
    for root, dirs, files in os.walk(crema_yolu):
        for file in files:
            if file.endswith('.wav'):
                try:
                    duygu = crema_cevirici.get(file.split('_')[2])
                    if duygu: veri_ekle(os.path.join(root, file), duygu)
                except: continue

# 4. MELD
meld_yolu = "./MELD/"

meld_cevirici = {
    'neutral': 'neutral',
    'anger': 'angry',
    'sadness': 'sad',
    'joy': 'happy',
    'fear': 'fearful',
    'disgust': 'disgust',
    'surprise': 'surprised'
}
if os.path.exists(meld_yolu):
    for csv_dosya, klasor in [
        ('train_sent_emo_cleaned_processed.csv', 'train/train_splits'),
        ('dev_sent_emo_cleaned_processed.csv', 'dev/dev_splits_complete'),
        ('test_sent_emo_cleaned_processed.csv', 'test/output_repeated_splits_test')
    ]:
        csv_yolu = os.path.join(meld_yolu, csv_dosya)
        if not os.path.exists(csv_yolu):
            continue

        df_meld = pd.read_csv(csv_yolu)
        ses_klasoru = os.path.join(meld_yolu, klasor)

        for _, satir in df_meld.iterrows():
            duygu = meld_cevirici.get(satir['Emotion'].lower())
            if not duygu:
                continue

            dosya_adi = f"dia{satir['Dialogue_ID']}_utt{satir['Utterance_ID']}.mp4"
            dosya_yolu = os.path.join(ses_klasoru, dosya_adi)

            if os.path.exists(dosya_yolu):
                veri_ekle(dosya_yolu, duygu)

    print(f"✅ MELD eklendi! Toplam veri: {len(data)}")

df = pd.DataFrame(data)
if not df.empty:
    df.rename(columns={0: 'label'}, inplace=True)
    df.to_csv("islenmis_veriler.csv", index=False)
    print("✅ Temiz CSV hazırlandı!")