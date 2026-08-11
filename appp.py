import streamlit as st
import pandas as pd
import numpy as np
from catboost import CatBoostRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score, mean_absolute_error
import re
import os

# =========================================================
# SAYFA AYARLARI VE KURUMSAL TEMA (CSS)
# =========================================================
st.set_page_config(
    page_title="Gayrimenkul Değerleme Platformu",
    layout="wide",
    initial_sidebar_state="expanded"
)

st.markdown("""
<style>
    .stApp { background-color: #0B0F19; color: #E2E8F0; }
    h1, h2, h3, h4 { color: #F8FAFC !important; font-weight: 600 !important; }
    div[data-testid="stMetric"] { background: #111827; border: 1px solid #1F2937; border-radius: 8px; padding: 16px; }
    div[data-testid="stMetric"] label { color: #94A3B8 !important; font-size: 0.85rem; font-weight: 500; }
    div[data-testid="stMetric"] div[data-testid="stMetricValue"] { color: #3B82F6 !important; font-weight: 700 !important; font-size: 1.35rem !important; }
    section[data-testid="stSidebar"] { background-color: #0F172A; border-right: 1px solid #1E293B; }
    .stButton > button { background-color: #2563EB; color: white; border-radius: 6px; border: none; font-weight: 600; width: 100%; }
    .stButton > button:hover { background-color: #1D4ED8; }
    .content-card { background-color: #111827; border: 1px solid #1F2937; border-radius: 8px; padding: 20px; margin-bottom: 20px; }
    .ozellik-etiket { display: inline-block; background: #1E3A5F; color: #93C5FD; padding: 4px 12px; border-radius: 4px; font-size: 0.82rem; margin: 3px 4px; border: 1px solid #2B4C7E; }
    .rapor-kart { background-color: #111827; border: 1px solid #3B82F6; border-radius: 8px; padding: 20px; margin-top: 15px; }
</style>
""", unsafe_allow_html=True)

def basliktan_ozellik_ve_yas_cikar(baslik):
    if pd.isna(baslik):
        return [], 10
    t = str(baslik).upper()
    oz = []
    
    bina_yasi = 10  
    if any(k in t for k in ['SIFIR', 'YENİ BİNA', 'PROJEDEN', 'SIFIR DAİRE']):
        bina_yasi = 0
        oz.append("Sıfır / Yeni")
    else:
        m = re.search(r'(\d+)\s*(YILLIK|YAŞINDA|YAŞ|YASINDA|YAS)', t)
        if m:
            bina_yasi = int(m.group(1))
            oz.append(f"{bina_yasi} Yıllık Bina")

    if any(k in t for k in ['DEPREM YÖNETMELİĞİ', 'RADYE TEMEL', 'PERDE BETON', 'C30', 'C35']):
        oz.append("Deprem Yönetmeliğine Uygun")
    if any(k in t for k in ['KENTSEL DÖNÜŞÜM', 'GÜÇLENDİRİLMİŞ', 'HASARSIZ']):
        oz.append("Kentsel Dönüşüm / Güçlendirilmiş")
    if any(k in t for k in ['METRO', 'METROBUS', 'MARMARAY', 'TRAMVAY']):
        oz.append("Ulaşım")
    if any(k in t for k in ['MARKET', 'ÇARŞI', 'AVM']):
        oz.append("Market / AVM")
    if any(k in t for k in ['MANZARA', 'DENİZ', 'BOĞAZ', 'ORMAN', 'GÖL']):
        oz.append("Manzara")
    if any(k in t for k in ['SİTE', 'SITE']):
        oz.append("Site içinde")
    if any(k in t for k in ['HAVUZ', 'YÜZME HAVUZU']):
        oz.append("Havuz")
    if any(k in t for k in ['KAPALI OTOPARK']):
        oz.append("Kapalı Otopark")
    elif any(k in t for k in ['OTOPARK', 'GARAJ']):
        oz.append("Otopark")
    if any(k in t for k in ['GÜVENLİK', '7/24', 'KAMERA']):
        oz.append("Güvenlik")
    if any(k in t for k in ['FITNESS', 'SPOR', 'SAUNA', 'SPA']):
        oz.append("Spor Salonu / Tesis")
    if any(k in t for k in ['ASANSÖR', 'ASANSOR']):
        oz.append("Asansör")
    if any(k in t for k in ['EBEVEYN', 'EBEVEYN BANYO']):
        oz.append("Ebeveyn Banyolu")
    if any(k in t for k in ['DUBLEKS', 'DUPLEX', 'TRİPLEKS']):
        oz.append("Dubleks")
    if any(k in t for k in ['BAHÇE KATI', 'BAHÇELİ']):
        oz.append("Bahçeli / Bahçe Katı")
    if any(k in t for k in ['ARA KAT']):
        oz.append("Ara Kat")
    if any(k in t for k in ['YERDEN ISITMA', 'KOMBİ', 'MANTOLAMA']):
        oz.append("Isıtma / Yalıtım")
        
    return oz, bina_yasi

@st.cache_resource(show_spinner="Veri seti ve yapay zeka modelleri yükleniyor...")
def load_model_and_data():
    excel_path = "ilanlar_mesafeli.xlsx" if os.path.exists("ilanlar_mesafeli.xlsx") else "ilanlar_tamamlanmis.xlsx"
    df = pd.read_excel(excel_path)
    df = df.dropna(subset=['Fiyat', 'm² (Brüt)'])
    df['oda_sayisi'] = pd.to_numeric(df['oda_sayisi'], errors='coerce').fillna(2).astype(int)
    df['salon_sayisi'] = pd.to_numeric(df['salon_sayisi'], errors='coerce').fillna(1).astype(int)
    df['Oda Düzeni'] = df['oda_sayisi'].astype(str) + '+' + df['salon_sayisi'].astype(str)

    oz_yas = [basliktan_ozellik_ve_yas_cikar(b) for b in df['İlan Başlığı']]
    df['Ozellikler'] = [x[0] for x in oz_yas]
    df['bina_yasi'] = [x[1] for x in oz_yas]

    df['İl'] = 'İstanbul'
    df['İlçe'] = df['İlçe'].fillna('Bilinmiyor').astype(str)
    df['Mahalle'] = df['Mahalle'].fillna('Bilinmiyor').astype(str)
    df['Semt'] = df['İlçe'] + " / " + df['Mahalle']

    df['fiyat_m2'] = df['Fiyat'] / df['m² (Brüt)']
    df['toplam_oda'] = df['oda_sayisi'] + df['salon_sayisi']
    df['m2_per_oda'] = df['m² (Brüt)'] / df['toplam_oda'].clip(lower=1)

    for col in ['ulasim_mesafe_m', 'market_mesafe_m', 'hastane_mesafe_m', 'okul_mesafe_m', 'taksi_mesafe_m', 'metro_mesafe_m']:
        df[col] = pd.to_numeric(df[col], errors='coerce').fillna(9999).astype(int) if col in df.columns else 9999

    feature_cols = ['m² (Brüt)', 'oda_sayisi', 'salon_sayisi', 'm2_per_oda', 'bina_yasi', 'Semt']
    X = df[feature_cols].copy()
    y_log = np.log1p(df['Fiyat'])
    X['Semt'] = X['Semt'].astype(str)

    X_train, X_test, y_train, y_test = train_test_split(X, y_log, test_size=0.2, random_state=42)
    
    model_mid = CatBoostRegressor(iterations=600, learning_rate=0.05, depth=6, cat_features=['Semt'], verbose=0, random_seed=42)
    model_mid.fit(X_train, y_train)

    model_low = CatBoostRegressor(iterations=600, learning_rate=0.05, depth=6, loss_function='Quantile:alpha=0.10', cat_features=['Semt'], verbose=0, random_seed=42)
    model_low.fit(X_train, y_train)

    model_high = CatBoostRegressor(iterations=600, learning_rate=0.05, depth=6, loss_function='Quantile:alpha=0.90', cat_features=['Semt'], verbose=0, random_seed=42)
    model_high.fit(X_train, y_train)

    y_pred = np.expm1(model_mid.predict(X_test))
    r2 = r2_score(np.expm1(y_test), y_pred)
    mae = mean_absolute_error(np.expm1(y_test), y_pred)

    return (model_mid, model_low, model_high), df, r2, mae

try:
    models, df, model_r2, model_mae = load_model_and_data()
    st.session_state['models'] = models
    st.session_state['df'] = df
    st.session_state['r2'] = model_r2
    st.session_state['mae'] = model_mae
except Exception as e:
    st.error(f"Veri yükleme hatası: {e}")
    st.stop()

st.title("Gayrimenkul Değerleme ve Analiz Platformu")
st.caption("Yapay Zeka Destekli Taşınmaz Analiz ve Değerleme Portalı")

st.markdown("""
<div class="content-card">
    <h4>Sisteme Hoş Geldiniz</h4>
    Sol taraftaki menüden yapmak istediğiniz modülü seçerek devam edebilirsiniz.
</div>
""", unsafe_allow_html=True)

col1, col2, col3 = st.columns(3)
col1.metric("Toplam Kayıtlı İlan Havuzu", f"{len(df):,} Adet")
col2.metric("Ortalama İlan Fiyatı", f"{df['Fiyat'].mean():,.0f} TL")
col3.metric("Yapay Zeka Model Başarımı (R²)", f"%{model_r2*100:.1f}")
