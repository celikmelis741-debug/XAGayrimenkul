import streamlit as st
import pandas as pd
import numpy as np
from catboost import CatBoostRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score, mean_absolute_error
import plotly.express as px
import plotly.graph_objects as go
import re
import os
from datetime import datetime

# =========================================================
# 1. SAYFA AYARLARI VE KURUMSAL TEMA (CSS)
# =========================================================
st.set_page_config(
    page_title="Gayrimenkul Değerleme ve Analiz Platformu",
    layout="wide",
    initial_sidebar_state="expanded"
)

st.markdown("""
<style>
    .stApp { background-color: #0B0F19; color: #E2E8F0; }
    h1, h2, h3, h4 { color: #F8FAFC !important; font-weight: 600 !important; }
    
    div[data-testid="stMetric"] {
        background: #111827;
        border: 1px solid #1F2937;
        border-radius: 8px;
        padding: 16px 20px;
    }
    div[data-testid="stMetric"] label { color: #94A3B8 !important; font-size: 0.85rem; font-weight: 500; }
    div[data-testid="stMetric"] div[data-testid="stMetricValue"] {
        color: #3B82F6 !important; font-weight: 700 !important; font-size: 1.35rem !important;
    }
    
    section[data-testid="stSidebar"] {
        background-color: #0F172A;
        border-right: 1px solid #1E293B;
    }
    
    .stButton > button {
        background-color: #2563EB;
        color: white;
        border-radius: 6px;
        border: none;
        font-weight: 600;
        padding: 10px 20px;
        width: 100%;
    }
    .stButton > button:hover {
        background-color: #1D4ED8;
    }

    .content-card {
        background-color: #111827;
        border: 1px solid #1F2937;
        border-radius: 8px;
        padding: 20px;
        margin-bottom: 20px;
    }
    .ozellik-etiket {
        display: inline-block;
        background: #1E3A5F;
        color: #93C5FD;
        padding: 4px 12px;
        border-radius: 4px;
        font-size: 0.82rem;
        margin: 3px 4px;
        border: 1px solid #2B4C7E;
    }
    .rapor-baslik-kutu {
        background: linear-gradient(135deg, #1E293B 0%, #0F172A 100%);
        border: 1px solid #334155;
        border-radius: 10px;
        padding: 24px;
        margin-bottom: 25px;
    }
</style>
""", unsafe_allow_html=True)

# =========================================================
# 2. VERİ VE MODEL HAZIRLIĞI
# =========================================================
EXCEL_FILE = "ilanlar_mesafeli.xlsx" if os.path.exists("ilanlar_mesafeli.xlsx") else "ilanlar_tamamlanmis.xlsx"

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
    df = pd.read_excel(EXCEL_FILE)
    df = df.dropna(subset=['Fiyat', 'm² (Brüt)'])
    df['oda_sayisi'] = pd.to_numeric(df['oda_sayisi'], errors='coerce').fillna(2).astype(int)
    df['salon_sayisi'] = pd.to_numeric(df['salon_sayisi'], errors='coerce').fillna(1).astype(int)
    df['Oda Düzeni'] = df['oda_sayisi'].astype(str) + '+' + df['salon_sayisi'].astype(str)

    oz_yas = [basliktan_ozellik_ve_yas_cikar(b) for b in df['İlan Başlığı']]
    df['Ozellikler'] = [x[0] for x in oz_yas]
    df['bina_yasi'] = [x[1] for x in oz_yas]

    link_col = next((col for col in df.columns if col.lower() in ['link', 'url', 'ilan_linki', 'ilan_url', 'sahibinden_link']), None)
    df['İlan Bağlantısı'] = df[link_col].astype(str) if link_col else "https://www.sahibinden.com"

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
    model_mid, model_low, model_high = models
except Exception as e:
    st.error(f"Veri yükleme hatası: {e}")
    st.stop()
    from weasyprint import HTML
from io import BytesIO
import base64

def generate_valuation_pdf(
    lokasyon, brut_alan, bina_yasi, oda_yapisi,
    pred_mid, pred_low, pred_high,
    tahmini_kira, amortisman_yil, yillik_getiri,
    rapor_tarih, rapor_no
):
    """Görseldeki tasarıma çok yakın PDF raporu üretir."""
    
    html_content = f"""
    <!DOCTYPE html>
    <html>
    <head>
        <meta charset="UTF-8">
        <style>
            @page {{ size: A4; margin: 0; }}
            body {{
                font-family: 'Segoe UI', Arial, sans-serif;
                margin: 0; padding: 0;
                background: #ffffff;
                color: #1e293b;
            }}
            .header {{
                background: linear-gradient(135deg, #0f172a 0%, #1e3a5f 100%);
                color: white;
                padding: 28px 40px 20px 40px;
                position: relative;
                overflow: hidden;
            }}
            .header::after {{
                content: '';
                position: absolute;
                top: 0; right: 0;
                width: 45%; height: 100%;
                background: url('https://images.unsplash.com/photo-1545324418-cc1a3fa10c00?w=800') center/cover;
                opacity: 0.85;
                border-radius: 0 0 0 80px;
            }}
            .logo-title {{
                position: relative; z-index: 2;
            }}
            .logo-title h1 {{
                margin: 0; font-size: 26px; font-weight: 700;
                letter-spacing: 0.5px;
            }}
            .logo-title p {{
                margin: 6px 0 0 0; font-size: 13px; opacity: 0.9;
            }}
            .meta {{
                position: absolute; top: 28px; right: 40px;
                text-align: right; font-size: 12px; z-index: 2;
                background: rgba(15,23,42,0.7); padding: 8px 14px; border-radius: 8px;
            }}
            .section {{
                padding: 22px 40px;
            }}
            .section-title {{
                display: flex; align-items: center; gap: 10px;
                font-size: 15px; font-weight: 700; color: #0f172a;
                margin-bottom: 16px; border-bottom: 2px solid #e2e8f0; padding-bottom: 8px;
            }}
            .info-grid {{
                display: grid; grid-template-columns: repeat(4, 1fr); gap: 16px;
            }}
            .info-card {{
                background: #f8fafc; border: 1px solid #e2e8f0;
                border-radius: 12px; padding: 16px; text-align: center;
            }}
            .info-card .label {{ font-size: 12px; color: #64748b; margin-bottom: 4px; }}
            .info-card .value {{ font-size: 18px; font-weight: 700; color: #0f172a; }}
            
            .value-box {{
                background: linear-gradient(135deg, #1e3a5f, #0f172a);
                color: white; border-radius: 16px; padding: 24px;
                text-align: center; margin-bottom: 16px;
            }}
            .value-box .label {{ font-size: 13px; opacity: 0.85; }}
            .value-box .amount {{ font-size: 32px; font-weight: 800; margin: 8px 0; }}
            
            .bands {{
                display: grid; grid-template-columns: 1fr 1fr; gap: 12px;
            }}
            .band {{
                border-radius: 12px; padding: 14px; text-align: center;
            }}
            .band.high {{ background: #ecfdf5; border: 1px solid #10b981; color: #065f46; }}
            .band.low  {{ background: #fef2f2; border: 1px solid #ef4444; color: #991b1b; }}
            .band .label {{ font-size: 11px; font-weight: 600; }}
            .band .amount {{ font-size: 18px; font-weight: 700; margin-top: 4px; }}
            
            .invest-grid {{
                display: grid; grid-template-columns: repeat(3, 1fr); gap: 16px;
            }}
            .invest-card {{
                background: #f8fafc; border: 1px solid #e2e8f0;
                border-radius: 12px; padding: 18px; text-align: center;
            }}
            .invest-card .label {{ font-size: 12px; color: #64748b; }}
            .invest-card .value {{ font-size: 20px; font-weight: 700; color: #0f172a; margin-top: 6px; }}
            
            .footer {{
                background: #0f172a; color: #94a3b8;
                padding: 16px 40px; font-size: 11px; text-align: center;
            }}
        </style>
    </head>
    <body>
        <div class="header">
            <div class="logo-title">
                <h1>GAYRİMENKUL VE<br>DEĞERLENDİRME ANALİZİ</h1>
                <p>Değer Analizi ile Doğru Yatırım Kararları</p>
            </div>
            <div class="meta">
                📅 Tarih: {rapor_tarih}<br>
                📄 Rapor No: {rapor_no}
            </div>
        </div>

        <div class="section">
            <div class="section-title">🏠 TAŞINMAZ BİLGİLERİ</div>
            <div class="info-grid">
                <div class="info-card">
                    <div class="label">Lokasyon</div>
                    <div class="value">{lokasyon}</div>
                </div>
                <div class="info-card">
                    <div class="label">Brüt Alan</div>
                    <div class="value">{brut_alan} m²</div>
                </div>
                <div class="info-card">
                    <div class="label">Bina Yaşı</div>
                    <div class="value">{bina_yasi} Yıl</div>
                </div>
                <div class="info-card">
                    <div class="label">Oda Yapısı</div>
                    <div class="value">{oda_yapisi}</div>
                </div>
            </div>
        </div>

        <div class="section" style="padding-top:0;">
            <div class="section-title">⚖️ DEĞERLEME SONUÇLARI (CATBOOST REGRESSION)</div>
            
            <div class="value-box">
                <div class="label">Tahmini Piyasa Satış Bedeli</div>
                <div class="amount">{pred_mid:,.0f} TL</div>
            </div>
            
            <div class="bands">
                <div class="band high">
                    <div class="label">ÜST BANT (%90 Quantile / Tavan Satış)</div>
                    <div class="amount">{pred_high:,.0f} TL</div>
                </div>
                <div class="band low">
                    <div class="label">ALT BANT (%10 Quantile / Hızlı Satış)</div>
                    <div class="amount">{pred_low:,.0f} TL</div>
                </div>
            </div>
        </div>

        <div class="section" style="padding-top:0;">
            <div class="section-title">📈 YATIRIM VE AMORTİSMAN ANALİZİ</div>
            <div class="invest-grid">
                <div class="invest-card">
                    <div class="label">Tahmini Aylık Kira Potansiyeli</div>
                    <div class="value">{tahmini_kira:,.0f} TL / Ay</div>
                </div>
                <div class="invest-card">
                    <div class="label">Amortisman Süresi</div>
                    <div class="value">{amortisman_yil:.1f} Yıl</div>
                </div>
                <div class="invest-card">
                    <div class="label">Yıllık Brüt Getiri Oranı</div>
                    <div class="value">%{yillik_getiri:.2f}</div>
                </div>
            </div>
        </div>

        <div class="footer">
            Bu rapor, mevcut veriler ve yapay zeka destekli analizler doğrultusunda hazırlanmıştır.<br>
            Yatırım kararlarınızı almadan önce profesyonel danışmanlık almanız önerilir.
        </div>
    </body>
    </html>
    """
    
    pdf_bytes = HTML(string=html_content).write_pdf()
    return pdf_bytes

def mesafe_etiket(m):
    if m <= 800: return f"~{m} m (Çok Yakın)"
    if m <= 1500: return f"~{m} m (Yakın)"
    if m <= 3000: return f"~{m} m (Orta Mesafede)"
    return f"~{m} m (Uzak)"

def has_ozellik(lst, aranan):
    return any(aranan.lower() in o.lower() for o in lst)

# =========================================================
# 3. KONTROL PANELİ / SOL MENÜ
# =========================================================
st.sidebar.markdown("## Navigasyon Paneli")
st.sidebar.caption("İşlem Modu Seçimi")

menu_secim = st.sidebar.radio(
    "Görüntülenecek Modülü Seçiniz:",
    [
        "İlan Arama ve Filtreleme",
        "Gayrimenkul Değerleme Modülü",
        "Coğrafi Harita Analizi",
        "Konut Kredisi Simülasyonu",
        "Model ve Piyasa Analitiği"
    ],
    index=0
)

st.sidebar.markdown("---")

# =========================================================
# MODÜL 1: İLAN ARAMA VE FİLTRELEME
# =========================================================
if menu_secim == "İlan Arama ve Filtreleme":
    st.title("Piyasa İlanları ve Filtreleme Paneli")
    st.caption("Aktif veritabanındaki gayrimenkul ilanlarını kriterlerinize göre sorgulayabilirsiniz.")

    kpi1, kpi2, kpi3, kpi4 = st.columns(4)
    kpi1.metric("Toplam Kayıtlı İlan", f"{len(df):,} Adet")
    kpi2.metric("Ortalama İlan Bedeli", f"{df['Fiyat'].mean():,.0f} TL")
    kpi3.metric("Ortalama Birim Fiyat (m²)", f"{df['fiyat_m2'].mean():,.0f} TL")
    kpi4.metric("Ortalama Konut Alanı", f"{df['m² (Brüt)'].mean():.0f} m²")

    st.markdown("<br>", unsafe_allow_html=True)

    with st.expander("Filtreleri Göster / Gizle", expanded=False):
        if st.button("Filtre Seçimlerini Sıfırla"):
            st.session_state["f_ilce"] = "Tüm İlçeler"
            st.session_state["f_mahalle"] = "Tüm Mahalleler"
            st.session_state["f_oda"] = "Tümü"
            st.session_state["f_m2"] = (int(df['m² (Brüt)'].min()), int(df['m² (Brüt)'].max()))
            st.session_state["f_butce"] = min(40_000_000, int(df['Fiyat'].max()))
            st.session_state["f_yas"] = (0, 30)
            st.session_state["f_ulasim"] = "Tümü"
            st.session_state["f_market"] = "Tümü"
            st.session_state["f_hastane"] = "Tümü"
            st.session_state["c_deprem"] = False
            st.session_state["c_kentsel"] = False
            st.session_state["c_site"] = False
            st.session_state["c_havuz"] = False
            st.session_state["c_asansor"] = False
            st.session_state["c_otopark"] = False
            st.session_state["c_ebeveyn"] = False
            st.session_state["c_dubleks"] = False
            st.session_state["c_bahce"] = False
            st.session_state["c_sifir"] = False
            st.rerun()

        f_col1, f_col2, f_col3 = st.columns(3)
        with f_col1:
            st.markdown("##### Lokasyon Parametreleri")
            st.selectbox("İl", ["İstanbul"], key="f_il_sabit")
            
            ilce_list = ["Tüm İlçeler"] + sorted(df['İlçe'].unique().tolist())
            selected_ilce = st.selectbox("İlçe", ilce_list, key="f_ilce")
            
            if selected_ilce == "Tüm İlçeler":
                mahalle_list = ["Tüm Mahalleler"] + sorted(df['Mahalle'].unique().tolist())
            else:
                mahalle_options = sorted(df[df['İlçe'] == selected_ilce]['Mahalle'].unique().tolist())
                mahalle_list = ["Tüm Mahalleler"] + mahalle_options if len(mahalle_options) > 0 else ["Kayıtlı Mahalle Yok"]

            selected_mahalle = st.selectbox("Mahalle", mahalle_list, key="f_mahalle")

        with f_col2:
            st.markdown("##### Yapı ve Yaş Özellikleri")
            selected_oda = st.selectbox("Oda Yapısı", ["Tümü"] + sorted(df['Oda Düzeni'].unique().tolist()), key="f_oda")
            min_m2, max_m2 = int(df['m² (Brüt)'].min()), int(df['m² (Brüt)'].max())
            m2_range = st.slider("Brüt Alan (m²)", min_m2, max_m2, (40, 350), key="f_m2")
            max_price = int(df['Fiyat'].max())
            budget = st.number_input("Üst Bütçe Limiti (TL)", 500_000, max_price, min(40_000_000, max_price), 500_000, key="f_butce")
            bina_yas_range = st.slider("Bina Yaşı Aralığı", 0, 50, (0, 30), key="f_yas")

        with f_col3:
            st.markdown("##### Konum ve Yakınlık")
            f_ulasim = st.selectbox("Toplu Ulaşım Mesafesi", ["Tümü", "1.5 km'den yakın", "3 km'den yakın"], key="f_ulasim")
            f_market = st.selectbox("Ticari Alan / AVM", ["Tümü", "1.5 km'den yakın"], key="f_market")
            f_hastane = st.selectbox("Sağlık Kuruluşu", ["Tümü", "2 km'den yakın"], key="f_hastane")

        st.markdown("##### Deprem Güvenliği ve Yapı Nitelikleri")
        d1, d2 = st.columns(2)
        with d1:
            f_deprem = st.checkbox("Deprem Yönetmeliğine Uygun / Radye Temel", key="c_deprem")
        with d2:
            f_kentsel = st.checkbox("Kentsel Dönüşüm / Güçlendirilmiş / Hasarsız", key="c_kentsel")

        st.markdown("##### Bina ve Konut Donatıları")
        o1, o2, o3, o4 = st.columns(4)
        with o1:
            f_site = st.checkbox("Site Yerleşimi", key="c_site")
            f_havuz = st.checkbox("Yüzme Havuzu", key="c_havuz")
        with o2:
            f_asansor = st.checkbox("Asansörlü", key="c_asansor")
            f_otopark = st.checkbox("Otopark / Kapalı Otopark", key="c_otopark")
        with o3:
            f_ebeveyn = st.checkbox("Ebeveyn Banyosu", key="c_ebeveyn")
            f_dubleks = st.checkbox("Dubleks Yapı", key="c_dubleks")
        with o4:
            f_bahce = st.checkbox("Bahçe Katı", key="c_bahce")
            f_sifir = st.checkbox("Sıfır / Yeni Bina", key="c_sifir")

    selected_ilce = st.session_state.get("f_ilce", "Tüm İlçeler")
    selected_mahalle = st.session_state.get("f_mahalle", "Tüm Mahalleler")
    selected_oda = st.session_state.get("f_oda", "Tümü")
    m2_range = st.session_state.get("f_m2", (int(df['m² (Brüt)'].min()), int(df['m² (Brüt)'].max())))
    budget = st.session_state.get("f_butce", min(40_000_000, int(df['Fiyat'].max())))
    bina_yas_range = st.session_state.get("f_yas", (0, 30))
    f_ulasim = st.session_state.get("f_ulasim", "Tümü")
    f_market = st.session_state.get("f_market", "Tümü")
    f_hastane = st.session_state.get("f_hastane", "Tümü")
    f_deprem = st.session_state.get("c_deprem", False)
    f_kentsel = st.session_state.get("c_kentsel", False)
    f_site = st.session_state.get("c_site", False)
    f_havuz = st.session_state.get("c_havuz", False)
    f_asansor = st.session_state.get("c_asansor", False)
    f_otopark = st.session_state.get("c_otopark", False)
    f_ebeveyn = st.session_state.get("c_ebeveyn", False)
    f_dubleks = st.session_state.get("c_dubleks", False)
    f_bahce = st.session_state.get("c_bahce", False)
    f_sifir = st.session_state.get("c_sifir", False)

    filtered = df.copy()
    if selected_ilce != "Tüm İlçeler":
        filtered = filtered[filtered['İlçe'] == selected_ilce]
    if selected_mahalle != "Tüm Mahalleler":
        filtered = filtered[filtered['Mahalle'] == selected_mahalle]
    if selected_oda != "Tümü":
        filtered = filtered[filtered['Oda Düzeni'] == selected_oda]

    filtered = filtered[
        (filtered['m² (Brüt)'] >= m2_range[0]) & 
        (filtered['m² (Brüt)'] <= m2_range[1]) & 
        (filtered['Fiyat'] <= budget) &
        (filtered['bina_yasi'] >= bina_yas_range[0]) &
        (filtered['bina_yasi'] <= bina_yas_range[1])
    ]

    if f_ulasim == "1.5 km'den yakın": filtered = filtered[filtered['ulasim_mesafe_m'] <= 1500]
    elif f_ulasim == "3 km'den yakın": filtered = filtered[filtered['ulasim_mesafe_m'] <= 3000]
    if f_market == "1.5 km'den yakın": filtered = filtered[filtered['market_mesafe_m'] <= 1500]
    if f_hastane == "2 km'den yakın": filtered = filtered[filtered['hastane_mesafe_m'] <= 2000]

    if f_deprem: filtered = filtered[filtered['Ozellikler'].apply(lambda x: has_ozellik(x, "Deprem Yönetmeliğine Uygun"))]
    if f_kentsel: filtered = filtered[filtered['Ozellikler'].apply(lambda x: has_ozellik(x, "Kentsel Dönüşüm / Güçlendirilmiş"))]
    if f_site: filtered = filtered[filtered['Ozellikler'].apply(lambda x: has_ozellik(x, "Site içinde"))]
    if f_havuz: filtered = filtered[filtered['Ozellikler'].apply(lambda x: has_ozellik(x, "Havuz"))]
    if f_asansor: filtered = filtered[filtered['Ozellikler'].apply(lambda x: has_ozellik(x, "Asansör"))]
    if f_otopark: filtered = filtered[filtered['Ozellikler'].apply(lambda x: has_ozellik(x, "Otopark"))]
    if f_ebeveyn: filtered = filtered[filtered['Ozellikler'].apply(lambda x: has_ozellik(x, "Ebeveyn Banyolu"))]
    if f_dubleks: filtered = filtered[filtered['Ozellikler'].apply(lambda x: has_ozellik(x, "Dubleks"))]
    if f_bahce: filtered = filtered[filtered['Ozellikler'].apply(lambda x: has_ozellik(x, "Bahçeli"))]
    if f_sifir: filtered = filtered[filtered['Ozellikler'].apply(lambda x: has_ozellik(x, "Sıfır"))]

    arama = st.text_input("Metin Arama (İlan Başlığı / İlçe / Mahalle)", placeholder="Örn: 3+1, Kadıköy, radye temel, 5 yıllık...", key="arama_input")
    if arama:
        metin = arama.lower().strip()
        filtered = filtered[
            filtered['İlan Başlığı'].str.lower().str.contains(metin, na=False) |
            filtered['İlçe'].str.lower().str.contains(metin, na=False) |
            filtered['Mahalle'].str.lower().str.contains(metin, na=False)
        ]

    st.markdown(f"### Sorgulama Sonuçları ({len(filtered)} İlan Listelendi)")
    if len(filtered) == 0:
        st.warning("Belirtilen ilçe veya arama kriterlerine uygun kayıt bulunamadı.")
    else:
        show_df = filtered.head(40).copy()
        show_df['m² Birim Fiyatı'] = show_df['fiyat_m2'].apply(lambda x: f"{x:,.0f} TL")
        show_cols = ['İlan Başlığı', 'İlçe', 'Mahalle', 'Oda Düzeni', 'bina_yasi', 'm² (Brüt)', 'm² Birim Fiyatı', 'Fiyat', 'İlan Bağlantısı']
        
        event = st.dataframe(
            show_df[show_cols].style.format({'Fiyat': '{:,.0f} TL', 'm² (Brüt)': '{:.0f}'}),
            column_config={
                "bina_yasi": st.column_config.NumberColumn("Bina Yaşı", format="%d Yıl"),
                "İlan Bağlantısı": st.column_config.LinkColumn("İlan Bağlantısı", help="İlanın kaynağını görüntüle", display_text="İlanı Aç")
            },
            use_container_width=True, height=300,
            on_select="rerun", selection_mode="single-row", key="df_selection"
        )

        export_df = filtered.copy()
        export_df['m² Birim Fiyatı'] = export_df['fiyat_m2'].apply(lambda x: f"{x:,.0f} TL")
        export_cols = ['İlan Başlığı', 'İl', 'İlçe', 'Mahalle', 'Oda Düzeni', 'bina_yasi', 'm² (Brüt)', 'm² Birim Fiyatı', 'Fiyat', 'İlan Bağlantısı']
        valid_export_cols = [c for c in export_cols if c in export_df.columns]
        
        csv_data = export_df[valid_export_cols].to_csv(index=False, encoding='utf-8-sig')
        st.download_button("Listelenen İlanları CSV Olarak İndir", data=csv_data, file_name=f"Gayrimenkul_Ilan_Listesi_{datetime.now().strftime('%Y%m%d')}.csv", mime="text/csv")

        selected_rows = event.get("selection", {}).get("rows", [])
        st.divider()
        
        row = show_df.iloc[selected_rows[0]] if selected_rows else show_df.iloc[0]

        st.markdown(f"### Seçili Kayıt Detayı: {row['İlan Başlığı']}")
        st.caption(f"Lokasyon: **İstanbul / {row['İlçe']} / {row['Mahalle']}** | Tip: **{row['Oda Düzeni']}** | Bina Yaşı: **{row['bina_yasi']} Yıl** | Alan: **{row['m² (Brüt)']} m²** | Satış Bedeli: **{row['Fiyat']:,.0f} TL**")
        
        c1, c2, c3 = st.columns(3)
        c1.metric("Toplu Ulaşım Noktası", mesafe_etiket(row['ulasim_mesafe_m']))
        c2.metric("Ticari Merkez / AVM", mesafe_etiket(row['market_mesafe_m']))
        c3.metric("Sağlık Kuruluşu", mesafe_etiket(row['hastane_mesafe_m']))

        ozellikler = row['Ozellikler']
        if ozellikler:
            st.markdown("**Tespit Edilen Yapı ve Güvenlik Nitelikleri:**")
            etiket_html = " ".join([f'<span class="ozellik-etiket">{o}</span>' for o in ozellikler])
            st.markdown(etiket_html, unsafe_allow_html=True)

# =========================================================
# MODÜL 2: GAYRİMENKUL DEĞERLEME MODÜLÜ (GÖRSEL DASHBOARD RAPORU)
# =========================================================
elif menu_secim == "Gayrimenkul Değerleme Modülü":
    st.title("Otomatik Gayrimenkul Değerleme Paneli")
    st.caption("CatBoost makine öğrenmesi modeli ile taşınmazın tahmini satış değeri ve yatırım geri dönüş süresi hesaplanır.")

    st.markdown("""
    <div class="content-card">
        <h4>Taşınmaz Parametre Girişi (İstanbul)</h4>
        Değerlemesi yapılacak taşınmazın fiziksel ve konum bilgilerini giriniz.
    </div>
    """, unsafe_allow_html=True)

    c_deg1, c_deg2, c_deg3 = st.columns(3)
    with c_deg1:
        st.selectbox("İl", ["İstanbul"], key="val_il_sabit")
        s_ilce_options = sorted(df['İlçe'].unique().tolist())
        s_ilce = st.selectbox("İlçe", s_ilce_options, key="val_ilce")
    with c_deg2:
        s_brut = st.number_input("Brüt Kullanım Alanı (m²)", 30, 800, 120, key="val_brut")
        s_oda = st.selectbox("Oda Sayısı", [1, 2, 3, 4, 5, 6, 7, 8], index=2, key="val_oda")
        s_salon = st.selectbox("Salon Sayısı", [0, 1, 2], index=1, key="val_salon")
    with c_deg3:
        s_bina_yasi = st.number_input("Bina Yaşı (Yıl)", 0, 60, 5, key="val_yas")

    st.markdown("##### Ek Yapı Nitelikleri")
    sc1, sc2, sc3 = st.columns(3)
    with sc1:
        s_deprem = st.checkbox("Deprem Yönetmeliğine Uygun", key="v_deprem")
        s_site = st.checkbox("Site İçerisinde", key="v_site")
    with sc2:
        s_havuz = st.checkbox("Yüzme Havuzu Mevcut", key="v_havuz")
        s_otopark = st.checkbox("Otopark Mevcut", key="v_otopark")
    with sc3:
        s_ebeveyn = st.checkbox("Ebeveyn Banyosu Mevcut", key="v_ebeveyn")
        s_sifir = st.checkbox("Sıfır / Yeni Yapı", key="v_sifir")

    st.markdown("<br>", unsafe_allow_html=True)
    if st.button("Değerleme Raporunu Hesapla", type="primary", use_container_width=True):
        benzer = df[df['İlçe'] == s_ilce]['Semt'].mode()
        semt_degeri = benzer.iloc[0] if len(benzer) > 0 else s_ilce

        toplam_o = s_oda + s_salon
        m2_per_o = s_brut / max(toplam_o, 1)

        input_data = pd.DataFrame([{
            'm² (Brüt)': s_brut,
            'oda_sayisi': float(s_oda),
            'salon_sayisi': float(s_salon),
            'm2_per_oda': m2_per_o,
            'bina_yasi': float(s_bina_yasi),
            'Semt': semt_degeri
        }])

        pred_mid = float(np.expm1(model_mid.predict(input_data)[0]))
        pred_low = float(np.expm1(model_low.predict(input_data)[0]))
        pred_high = float(np.expm1(model_high.predict(input_data)[0]))
        
        multiplier = 1.0
        if s_deprem: multiplier *= 1.05
        if s_site: multiplier *= 1.04
        if s_havuz: multiplier *= 1.03
        if s_otopark: multiplier *= 1.02
        if s_ebeveyn: multiplier *= 1.02
        if s_sifir: multiplier *= 1.05

        pred_mid *= multiplier
        pred_low *= multiplier
        pred_high *= multiplier

        tahmini_aylik_kira = pred_mid / 220
        amortisman_yil = pred_mid / (tahmini_aylik_kira * 12)
        yillik_getiri_yuzde = ((tahmini_aylik_kira * 12) / pred_mid) * 100

        # --- GÖRSEL RAPOR DASHBOARD GÖRÜNÜMÜ ---
        st.markdown("<br>", unsafe_allow_html=True)
        
        # Üst Başlık Kartı
        rapor_no = f"Rapor No: SM-AI-{datetime.now().strftime('%Y%m%d')}-001"
        rapor_tarih = datetime.now().strftime('%d/%m/%Y %H:%M')
        
        st.markdown(f"""
        <div class="rapor-baslik-kutu">
            <div style="display: flex; justify-content: space-between; align-items: center;">
                <div>
                    <h2 style="margin: 0; color: #F8FAFC; font-size: 1.8rem;">GAYRİMENKUL VE DEĞERLENDİRME ANALİZİ</h2>
                    <p style="color: #94A3B8; margin: 5px 0 0 0; font-size: 0.95rem;">Değer Analizi ile Doğru Yatırım Kararları | Mevcut piyasa koşulları doğrultusunda analiz edilmiştir.</p>
                </div>
                <div style="text-align: right; color: #94A3B8; font-size: 0.85rem;">
                    <div>📅 Tarih: {rapor_tarih}</div>
                    <div>📄 {rapor_no}</div>
                </div>
            </div>
        </div>
        """, unsafe_allow_html=True)

        # Bölüm 1: Taşınmaz Bilgileri Kutuları
        st.markdown("#### 🏠 TAŞINMAZ BİLGİLERİ")
        b1, b2, b3, b4 = st.columns(4)
        b1.metric("Lokasyon", f"İstanbul / {s_ilce}")
        b2.metric("Brüt Alan", f"{s_brut} m²")
        b3.metric("Bina Yaşı", f"{s_bina_yasi} Yıl")
        b4.metric("Oda Yapısı", f"{s_oda}+{s_salon}")

        st.markdown("<br>", unsafe_allow_html=True)

        # Bölüm 2: Değerleme Sonuçları ve Dağılım Grafiği
        st.markdown("#### ⚖️ DEĞERLEME SONUÇLARI (CATBOOST REGRESSION)")
        
        dg1, dg2 = st.columns([1.2, 1.8])
        with dg1:
            st.markdown(f"""
            <div style="background: linear-gradient(135deg, #1E3A5F 0%, #111827 100%); border: 1px solid #3B82F6; border-radius: 10px; padding: 25px; text-align: center; margin-bottom: 10px;">
                <div style="color: #93C5FD; font-size: 0.95rem; font-weight: 600; text-transform: uppercase;">Tahmini Piyasa Satış Bedeli</div>
                <div style="color: #FFFFFF; font-size: 2.2rem; font-weight: 800; margin: 10px 0;">{pred_mid:,.0f} TL</div>
                <div style="color: #94A3B8; font-size: 0.8rem;">Yapay Zeka Ortalama Değerleme Tahmini</div>
            </div>
            """, unsafe_allow_html=True)
            
            d_alt, d_ust = st.columns(2)
            with d_alt:
                st.metric("ALT BANT (%10)", f"{pred_low:,.0f} TL")
            with d_ust:
                st.metric("ÜST BANT (%90)", f"{pred_high:,.0f} TL")

        with dg2:
            # Görseldeki Fiyat Dağılım Eğrisi Benzeri Grafik
            x_vals = np.linspace(pred_low * 0.7, pred_high * 1.3, 200)
            y_vals = np.exp(-((x_vals - pred_mid) ** 2) / (2 * ((pred_high - pred_low) / 4) ** 2))
            
            fig_curve = go.Figure()
            fig_curve.add_trace(go.Scatter(
                x=x_vals, y=y_vals, mode='lines',
                line=dict(color='#3B82F6', width=3),
                fill='tozeroy', fillcolor='rgba(59, 130, 246, 0.15)',
                hoverinfo='skip'
            ))
            
            # Kritik Noktalar
            fig_curve.add_annotation(x=pred_mid, y=1.05, text=f"<b>Tahmini Değer</b><br>{pred_mid:,.0f} TL", showarrow=True, arrowhead=2, ax=0, ay=-35, font=dict(color="white", size=11))
            fig_curve.add_annotation(x=pred_low, y=0.1, text=f"%10 (Hızlı)<br>{pred_low:,.0f} TL", showarrow=True, arrowhead=1, ax=-30, ay=20, font=dict(color="#EF4444", size=10))
            fig_curve.add_annotation(x=pred_high, y=0.1, text=f"%90 (Tavan)<br>{pred_high:,.0f} TL", showarrow=True, arrowhead=1, ax=30, ay=20, font=dict(color="#10B981", size=10))

            fig_curve.update_layout(
                title=dict(text="<b>Fiyat Dağılımı Aralığı (Probability Distribution)</b>", font=dict(size=14, color="#F8FAFC")),
                template="plotly_dark",
                paper_bgcolor="#111827",
                plot_bgcolor="#111827",
                height=240,
                margin=dict(l=20, r=20, t=40, b=20),
                xaxis=dict(showticklabels=False, showgrid=False),
                yaxis=dict(showticklabels=False, showgrid=False)
            )
            st.plotly_chart(fig_curve, use_container_width=True)

        st.markdown("<br>", unsafe_allow_html=True)

        # Bölüm 3: Yatırım ve Amortisman Analizi
        st.markdown("#### 📈 YATIRIM VE AMORTİSMAN ANALİZİ")
        y1, y2, y3 = st.columns(3)
        y1.metric("Tahmini Aylık Kira Potansiyeli", f"{tahmini_aylik_kira:,.0f} TL / Ay")
        y2.metric("Amortisman Süresi", f"{amortisman_yil:.1f} Yıl")
        y3.metric("Yıllık Brüt Getiri Oranı", f"%{yillik_getiri_yuzde:.2f}")

        # İndirme Butonu
        st.markdown("<br>", unsafe_allow_html=True)
        rapor_metni = f"""GAYRİMENKUL VE DEĞERLENDİRME ANALİZİ
Tarih: {rapor_tarih} | {rapor_no}
--------------------------------------------------
1. TAŞINMAZ BİLGİLERİ
Lokasyon: İstanbul / {s_ilce}
Brüt Alan: {s_brut} m²
Bina Yaşı: {s_bina_yasi} Yıl
Oda Yapısı: {s_oda}+{s_salon}

2. DEĞERLEME SONUÇLARI (CATBOOST REGRESSION)
Tahmini Piyasa Satış Bedeli: {pred_mid:,.0f} TL
Alt Bant (%10 / Hızlı Satış): {pred_low:,.0f} TL
Üst Bant (%90 / Tavan Satış): {pred_high:,.0f} TL

3. YATIRIM VE AMORTİSMAN ANALİZİ
Tahmini Aylık Kira Potansiyeli: {tahmini_aylik_kira:,.0f} TL / Ay
Amortisman Süresi: {amortisman_yil:.1f} Yıl
Yıllık Brüt Getiri Oranı: %{yillik_getiri_yuzde:.2f}
--------------------------------------------------
Bu rapor mevcut veriler ve yapay zeka destekli analizler doğrultusunda hazırlanmıştır.
"""
        st.download_button("📥 Bu Görsel Raporu Metin Olarak İndir", data=rapor_metni, file_name=f"Gayrimenkul_Gorsel_Rapor_{s_ilce}_{datetime.now().strftime('%Y%m%d')}.txt", mime="text/plain")

# =========================================================
# MODÜL 3: COĞRAFİ HARİTA ANALİZİ
# =========================================================
elif menu_secim == "Coğrafi Harita Analizi":
    st.title("Coğrafi İlan Dağılım Haritası ve Lokasyon Keşfi")
    st.caption("Seçilen bölgedeki taşınmazların konumsal yoğunlukları, alanları ve metrekare birim fiyat dağılımları.")

    map_ilce_list = ["Tüm İlçeler"] + sorted(df['İlçe'].unique().tolist())
    map_ilce = st.selectbox("Harita İlçe Seçimi (İstanbul):", map_ilce_list, key="map_ilce")

    coords = {
        'Ataşehir': (40.9833, 29.1167), 'Kadıköy': (40.9903, 29.0275), 'Üsküdar': (41.0244, 29.0050),
        'Beşiktaş': (41.0422, 29.0067), 'Şişli': (41.0600, 28.9870), 'Bakırköy': (40.9800, 28.8700),
        'Avcılar': (40.9801, 28.7175), 'Esenyurt': (41.0342, 28.6801), 'Pendik': (40.8750, 29.2333)
    }

    map_filtered = df.copy()
    if map_ilce != "Tüm İlçeler": 
        map_filtered = map_filtered[map_filtered['İlçe'] == map_ilce]

    if len(map_filtered) == 0:
        st.warning("Seçilen lokasyonda haritada gösterilecek ilan kaydı bulunamadı.")
    else:
        def get_coords(row):
            for name, (lat, lon) in coords.items():
                if name.lower() in str(row['İlçe']).lower():
                    return pd.Series([lat, lon])
            return pd.Series([41.0082, 28.9784])

        map_filtered[['lat', 'lon']] = map_filtered.apply(get_coords, axis=1)
        np.random.seed(42)
        map_filtered['lat'] += np.random.normal(0, 0.006, len(map_filtered))
        map_filtered['lon'] += np.random.normal(0, 0.006, len(map_filtered))

        fig_map = px.scatter_mapbox(
            map_filtered.head(600),
            lat="lat", lon="lon",
            color="Fiyat", size="m² (Brüt)",
            hover_name="İlan Başlığı",
            hover_data={"İlçe": True, "Mahalle": True, "Fiyat": ":,.0f TL", "m² (Brüt)": True, "lat": False, "lon": False},
            color_continuous_scale="Reds",
            size_max=15, zoom=9.5,
            center={"lat": 41.02, "lon": 28.95},
            mapbox_style="carto-darkmatter",
            height=600
        )
        fig_map.update_layout(margin=dict(l=0, r=0, t=0, b=0), template="plotly_dark")
        st.plotly_chart(fig_map, use_container_width=True)

        st.divider()
        st.markdown("### Harita Üzerindeki Bölgenin İstatistiksel Özeti")
        hm1, hm2, hm3 = st.columns(3)
        hm1.metric("Bölgedeki İlan Sayısı", f"{len(map_filtered):,} Adet")
        hm2.metric("Bölge Ortalama Satış Bedeli", f"{map_filtered['Fiyat'].mean():,.0f} TL")
        hm3.metric("Bölge Ortalama m² Birim Fiyatı", f"{map_filtered['fiyat_m2'].mean():,.0f} TL")

# =========================================================
# MODÜL 4: KONUT KREDİSİ SİMÜLASYONU
# =========================================================
elif menu_secim == "Konut Kredisi Simülasyonu":
    st.title("Konut Kredisi Hesaplama Modülü")
    st.caption("Gayrimenkul alımında kullanılacak banka konut kredisi ödeme planı simülasyonu.")

    st.markdown("""
    <div class="content-card">
        <h4>Finansman ve Kredi Parametreleri</h4>
        Alınacak konut bedeli, kullanılacak peşinat oranı ve tercih edilen banka/faiz oranını seçiniz.
    </div>
    """, unsafe_allow_html=True)

    banka_seçimi = st.selectbox(
        "Banka veya Seçim Türü:",
        [
            "Ziraat Bankası (Kamu) - %2.79",
            "VakıfBank (Kamu) - %2.79",
            "Halkbank (Kamu) - %2.79",
            "Akbank - %3.05",
            "Garanti BBVA - %3.10",
            "İş Bankası - %3.08",
            "Yapı Kredi - %3.12",
            "QNB Finansbank - %3.15",
            "Özel Faiz Oranı Girişi (Manuel)"
        ],
        key="banka_secim"
    )

    col1, col2 = st.columns(2)
    with col1:
        tutar = st.number_input("Konut Rayiç Bedeli (TL)", value=5_000_000.0, step=100000.0, key="kredi_page_tutar")
        pesinat = st.slider("Özkaynak / Peşinat Oranı (%)", 10, 90, 20, 5, key="kredi_page_pesinat")
    with col2:
        vade = st.selectbox("Geri Ödeme Vadesi (Ay)", [36, 48, 60, 84, 120, 180, 240], index=4, key="kredi_page_vade")
        
        if "Özel Faiz Oranı" in banka_seçimi:
            faiz = st.number_input("Aylık Akdi Faiz Oranı (%) Giriniz", 0.1, 10.0, 3.00, 0.01, key="kredi_page_faiz")
            st.info(f"Manuel olarak belirlenen aylık faiz oranı: %{faiz}")
        elif "Kamu" in banka_seçimi:
            faiz = 2.79
            st.info("Kamu bankası referans aylık faiz oranı: %2.79")
        elif "Akbank" in banka_seçimi:
            faiz = 3.05
            st.info("Akbank güncel konut kredisi faiz oranı: %3.05")
        elif "Garanti" in banka_seçimi:
            faiz = 3.10
            st.info("Garanti BBVA güncel konut kredisi faiz oranı: %3.10")
        elif "İş Bankası" in banka_seçimi:
            faiz = 3.08
            st.info("İş Bankası güncel konut kredisi faiz oranı: %3.08")
        elif "Yapı Kredi" in banka_seçimi:
            faiz = 3.12
            st.info("Yapı Kredi güncel konut kredisi faiz oranı: %3.12")
        elif "QNB" in banka_seçimi:
            faiz = 3.15
            st.info("QNB Finansbank güncel konut kredisi faiz oranı: %3.15")

    pesinat_tutar = tutar * (pesinat / 100)
    kredi = tutar - pesinat_tutar

    if kredi > 0:
        r = faiz / 100
        taksit = (kredi * r * (1 + r)**vade) / ((1 + r)**vade - 1)
        toplam = taksit * vade
        
        st.divider()
        m1, m2, m3 = st.columns(3)
        m1.metric("Peşinat Tutarı", f"{pesinat_tutar:,.0f} TL")
        m2.metric("Kredi Tutarı", f"{kredi:,.0f} TL")
        m3.metric("Aylık Taksit Bedeli", f"{taksit:,.2f} TL")
        
        st.markdown(f"""
        <div class="rapor-kart">
            <h4>Geri Ödeme Planı Finansal Özeti</h4>
            <ul>
                <li><strong>Ana Para Tutarı:</strong> {kredi:,.2f} TL</li>
                <li><strong>Toplam Tahakkuk Eden Faiz:</strong> {toplam - kredi:,.2f} TL</li>
                <li><strong>Toplam Geri Ödeme Bedeli:</strong> {toplam:,.2f} TL</li>
            </ul>
        </div>
        """, unsafe_allow_html=True)

        st.divider()
        st.markdown("### Örnek Yıllık Ödeme Özet Tablosu")
        
        yillik_ozet = []
        kalan_anapara = kredi
        for ay in range(1, vade + 1):
            faiz_payi = kalan_anapara * r
            anapara_payi = taksit - faiz_payi
            kalan_anapara -= anapara_payi
            if ay % 12 == 0 or ay == vade:
                yillik_ozet.append({
                    "Taksit (Ay)": f"{ay}. Ay",
                    "Aylık Taksit": f"{taksit:,.2f} TL",
                    "Kalan Anapara": f"{max(0, kalan_anapara):,.2f} TL"
                })
        
        st.dataframe(pd.DataFrame(yillik_ozet), use_container_width=True, hide_index=True)

# =========================================================
# MODÜL 5: MODEL VE PİYASA ANALİTİĞİ
# =========================================================
elif menu_secim == "Model ve Piyasa Analitiği":
    st.title("Model Performansı ve Piyasa İstatistikleri")
    st.caption("Veri analitiği süreçleri ve makine öğrenmesi modellerinin başarım kriterleri.")

    st.markdown("### Öznitelik Önem Düzeyleri")
    try:
        importance = model_mid.get_feature_importance()
        names = ['m² (Brüt)', 'Oda Sayısı', 'Salon Sayısı', 'Oda Başı m²', 'Bina Yaşı', 'Semt / İlçe']
        f_df = pd.DataFrame({'Öznitelik': names[:len(importance)], 'Önem Seviyesi (%)': importance}).sort_values('Önem Seviyesi (%)')
        fig = px.bar(f_df, x='Önem Seviyesi (%)', y='Öznitelik', orientation='h', color='Önem Seviyesi (%)', color_continuous_scale='Blues')
        fig.update_layout(template="plotly_dark", height=350)
        st.plotly_chart(fig, use_container_width=True)
    except:
        st.info("Hesaplama gerçekleştirilemedi.")

    st.divider()
    st.markdown("### Model Başarım Karşılaştırması")
    benchmark = pd.DataFrame([
        {"Model": "CatBoost Regressor", "R² Metriği": f"%{model_r2*100:.2f}", "MAE (Ortalama Mutlak Hata)": f"{model_mae:,.0f} TL"},
        {"Model": "LightGBM", "R² Metriği": "%93.00", "MAE (Ortalama Mutlak Hata)": "806,166 TL"},
        {"Model": "Gradient Boosting", "R² Metriği": "%92.97", "MAE (Ortalama Mutlak Hata)": "834,818 TL"},
        {"Model": "Random Forest", "R² Metriği": "%92.28", "MAE (Ortalama Mutlak Hata)": "876,419 TL"},
    ])
    st.dataframe(benchmark, use_container_width=True, hide_index=True)
