import streamlit as st
import pandas as pd
import numpy as np
from catboost import CatBoostRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error
import plotly.express as px
import plotly.graph_objects as go
import re
from datetime import datetime

# =========================================================
# 1. SAYFA AYARLARI + TEMA (MOBİL UYUMLU CSS)
# =========================================================
st.set_page_config(
    page_title="XAI Gayrimenkul Değerleme",
    page_icon="🏆",
    layout="wide",
    initial_sidebar_state="collapsed" # Mobilde sol menüyü kapalı başlatıyoruz
)

st.markdown("""
<style>
    .stApp { background-color: #0B0F19; color: #E2E8F0; }
    h1, h2, h3 { color: #F8FAFC !important; font-weight: 600 !important; }
    div[data-testid="stMetric"] {
        background-color: #111827;
        border: 1px solid #1F2937;
        border-radius: 12px;
        padding: 12px 16px;
        margin-bottom: 8px;
    }
    div[data-testid="stMetric"] label { color: #94A3B8 !important; font-size: 0.85rem; }
    div[data-testid="stMetric"] div[data-testid="stMetricValue"] {
        color: #F1F5F9 !important; font-weight: 600 !important; font-size: 1.25rem !important;
    }
    .stButton > button {
        background-color: #3B82F6;
        color: white;
        border-radius: 8px;
        border: none;
        font-weight: 500;
        width: 100%;
    }
    .stButton > button:hover { background-color: #2563EB; }
    .stTabs [data-baseweb="tab"] {
        background-color: #1F2937;
        border-radius: 8px;
        color: #94A3B8;
        padding: 8px 12px;
    }
    .stTabs [aria-selected="true"] {
        background-color: #3B82F6 !important;
        color: white !important;
    }
    .ozellik-etiket {
        display: inline-block;
        background: #1E3A5F;
        color: #93C5FD;
        padding: 4px 10px;
        border-radius: 20px;
        font-size: 0.8rem;
        margin: 2px 3px;
    }
    .rapor-kart {
        background-color: #111827;
        border: 1px solid #374151;
        border-radius: 12px;
        padding: 16px;
        margin-top: 12px;
    }
</style>
""", unsafe_allow_html=True)

st.title("XAI Gayrimenkul Değerleme")
st.caption("CatBoost + Mobil Uyumlu Arayüz & Quantile Regression")

# TÜRKİYE 81 İL LİSTESİ
TURKIYE_ILLERI = [
    "Adana", "Adıyaman", "Afyonkarahisar", "Ağrı", "Aksaray", "Amasya", "Ankara", "Antalya", "Ardahan", "Artvin",
    "Aydın", "Balıkesir", "Bartın", "Batman", "Bayburt", "Bilecik", "Bingöl", "Bitlis", "Bolu", "Burdur",
    "Bursa", "Çanakkale", "Çankırı", "Çorum", "Denizli", "Diyarbakır", "Düzce", "Edirne", "Elazığ", "Erzincan",
    "Erzurum", "Eskişehir", "Gaziantep", "Giresun", "Gümüşhane", "Hakkari", "Hatay", "Iğdır", "Isparta", "İstanbul",
    "İzmir", "Kahramanmaraş", "Karabük", "Karaman", "Kars", "Kastamonu", "Kayseri", "Kırıkkale", "Kırklareli", "Kırşehir",
    "Kilis", "Kocaeli", "Konya", "Kütahya", "Malatya", "Manisa", "Mardin", "Mersin", "Muğla", "Muş",
    "Nevşehir", "Niğde", "Ordu", "Osmaniye", "Rize", "Sakarya", "Samsun", "Siirt", "Sinop", "Sivas",
    "Şanlıurfa", "Şırnak", "Tekirdağ", "Tokat", "Trabzon", "Tunceli", "Uşak", "Van", "Yalova", "Yozgat", "Zonguldak"
]

# =========================================================
# 2. VERİ VE MODEL
# =========================================================
EXCEL_FILE = "ilanlar_mesafeli.xlsx"

def basliktan_ozellik_cikar(baslik):
    if pd.isna(baslik):
        return []
    t = str(baslik).upper()
    oz = []
    if any(k in t for k in ['METRO', 'METROBUS', 'METROBÜS', 'MARMARAY']):
        oz.append("Ulaşım")
    if any(k in t for k in ['MARKET', 'ÇARŞI', 'AVM']):
        oz.append("Market / AVM")
    if any(k in t for k in ['MANZARA', 'DENİZ', 'BOĞAZ', 'ORMAN', 'GÖL']):
        oz.append("Manzara")
    if any(k in t for k in ['SİTE', 'SITE']):
        oz.append("Site içinde")
    if any(k in t for k in ['HAVUZ', 'YÜZME HAVUZU']):
        oz.append("Havuz")
    if any(k in t for k in ['OTOPARK', 'GARAJ', 'KAPALI OTOPARK']):
        oz.append("Otopark")
    if any(k in t for k in ['GÜVENLİK', '7/24', 'KAMERA']):
        oz.append("Güvenlik")
    if any(k in t for k in ['FITNESS', 'SPOR', 'SAUNA', 'SPA', 'SOSYAL TESİS']):
        oz.append("Spor Salonu / Tesis")
    if any(k in t for k in ['ASANSÖR', 'ASANSOR']):
        oz.append("Asansör")
    if any(k in t for k in ['SIFIR', 'YENİ BİNA', 'PROJEDEN']):
        oz.append("Sıfır / Yeni")
    if any(k in t for k in ['EBEVEYN', 'EBEVEYN BANYO']):
        oz.append("Ebeveyn Banyolu")
    if any(k in t for k in ['BALKON', 'TERAS', 'VERANDA']):
        oz.append("Balkon / Teras")
    if any(k in t for k in ['DUBLEKS', 'DUPLEX', 'TRİPLEKS']):
        oz.append("Dubleks")
    if any(k in t for k in ['BAHÇE KATI', 'BAHÇELİ', 'BAHCE']):
        oz.append("Bahçeli / Bahçe Katı")
    if any(k in t for k in ['YERDEN ISITMA', 'KOMBİ', 'MERKEZİ ISITMA']):
        oz.append("Isıtma Sistemi")
    return oz

@st.cache_resource(show_spinner="Modeller eğitiliyor ve mobil arayüz hazırlanıyor...")
def load_model_and_data():
    df = pd.read_excel(EXCEL_FILE)
    df = df.dropna(subset=['Fiyat', 'm² (Brüt)'])
    df['oda_sayisi'] = pd.to_numeric(df['oda_sayisi'], errors='coerce').fillna(2).astype(int)
    df['salon_sayisi'] = pd.to_numeric(df['salon_sayisi'], errors='coerce').fillna(1).astype(int)
    df['Oda Düzeni'] = df['oda_sayisi'].astype(str) + '+' + df['salon_sayisi'].astype(str)

    if 'İl' in df.columns:
        df['İl'] = df['İl'].fillna('İstanbul').astype(str)
    else:
        def il_tespit(row):
            metin = str(row.get('Semt / Mahalle', '')) + " " + str(row.get('İlçe', ''))
            for il in TURKIYE_ILLERI:
                if il.lower() in metin.lower():
                    return il
            return "İstanbul"
        df['İl'] = df.apply(il_tespit, axis=1)

    df['İlçe'] = df['İlçe'].fillna('Bilinmiyor').astype(str)
    df['Mahalle'] = df['Mahalle'].fillna('Bilinmiyor').astype(str)
    
    if 'Semt / Mahalle' in df.columns:
        df['Semt'] = df['Semt / Mahalle'].astype(str)
    else:
        df['Semt'] = df['İlçe'] + " / " + df['Mahalle']

    df['fiyat_m2'] = df['Fiyat'] / df['m² (Brüt)']
    df['toplam_oda'] = df['oda_sayisi'] + df['salon_sayisi']
    df['m2_per_oda'] = df['m² (Brüt)'] / df['toplam_oda'].clip(lower=1)

    for col in ['ulasim_mesafe_m', 'market_mesafe_m', 'hastane_mesafe_m', 'okul_mesafe_m', 'taksi_mesafe_m', 'metro_mesafe_m']:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors='coerce').fillna(9999).astype(int)
        else:
            df[col] = 9999

    df['Ozellikler'] = df['İlan Başlığı'].apply(basliktan_ozellik_cikar)

    feature_cols = ['m² (Brüt)', 'oda_sayisi', 'salon_sayisi', 'm2_per_oda', 'Semt']
    X = df[feature_cols].copy()
    y_log = np.log1p(df['Fiyat'])
    X['Semt'] = X['Semt'].astype(str)

    X_train, X_test, y_train, y_test = train_test_split(X, y_log, test_size=0.2, random_state=42)
    
    model_mid = CatBoostRegressor(
        iterations=600, learning_rate=0.05, depth=6,
        cat_features=['Semt'], verbose=0, random_seed=42
    )
    model_mid.fit(X_train, y_train)

    model_low = CatBoostRegressor(
        iterations=600, learning_rate=0.05, depth=6,
        loss_function='Quantile:alpha=0.10',
        cat_features=['Semt'], verbose=0, random_seed=42
    )
    model_low.fit(X_train, y_train)

    model_high = CatBoostRegressor(
        iterations=600, learning_rate=0.05, depth=6,
        loss_function='Quantile:alpha=0.90',
        cat_features=['Semt'], verbose=0, random_seed=42
    )
    model_high.fit(X_train, y_train)

    y_pred = np.expm1(model_mid.predict(X_test))
    y_true = np.expm1(y_test)
    r2 = r2_score(y_true, y_pred)
    mae = mean_absolute_error(y_true, y_pred)
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))

    return (model_mid, model_low, model_high), df, r2, mae, rmse

try:
    models, df, model_r2, model_mae, model_rmse = load_model_and_data()
    model_mid, model_low, model_high = models
except Exception as e:
    st.error(f"Veri yüklenirken hata: {e}")
    st.stop()

def mesafe_etiket(m):
    if m <= 800: return f"~{m} m (çok yakın)"
    if m <= 1500: return f"~{m} m (yakın)"
    if m <= 3000: return f"~{m} m (orta)"
    return f"~{m} m (uzak)"

def has_ozellik(lst, aranan):
    return any(aranan.lower() in o.lower() for o in lst)

# =========================================================
# 3. ÜST FİLTRE PANELİ (MOBİL UYUMLU - EXPANDER PANELİ)
# =========================================================
with st.expander("🔍 Filtreler ve Lokasyon Seçimi (Tıklayın)", expanded=True):
    f_col1, f_col2, f_col3 = st.columns(3)
    
    with f_col1:
        st.markdown("**Lokasyon**")
        il_options = ["Tüm İller"] + sorted(list(set(TURKIYE_ILLERI + df['İl'].unique().tolist())))
        selected_il = st.selectbox("İl", il_options, index=0, key="il_sec")

        if selected_il == "Tüm İller":
            ilce_options = sorted(df['İlçe'].unique().tolist())
        else:
            ilce_options = sorted(df[df['İl'] == selected_il]['İlçe'].unique().tolist())

        ilce_list = ["Tüm İlçeler"] + ilce_options
        selected_ilce = st.selectbox("İlçe", ilce_list, key="ilce_sec")

        if selected_ilce == "Tüm İlçeler":
            if selected_il == "Tüm İller":
                mahalle_options = sorted(df['Mahalle'].unique().tolist())
            else:
                mahalle_options = sorted(df[df['İl'] == selected_il]['Mahalle'].unique().tolist())
        else:
            mahalle_options = sorted(df[df['İlçe'] == selected_ilce]['Mahalle'].unique().tolist())

        mahalle_list = ["Tüm Mahalleler"] + mahalle_options
        selected_mahalle = st.selectbox("Mahalle", mahalle_list, key="mahalle_sec")

    with f_col2:
        st.markdown("**Konut & Bütçe**")
        selected_oda = st.selectbox("Oda Düzeni", ["Tümü"] + sorted(df['Oda Düzeni'].unique().tolist()), key="oda_sec")
        min_m2, max_m2 = int(df['m² (Brüt)'].min()), int(df['m² (Brüt)'].max())
        m2_range = st.slider("Brüt m²", min_m2, max_m2, (40, 350), key="m2_sec")
        max_price = int(df['Fiyat'].max())
        budget = st.number_input("Maksimum Bütçe (TL)", 500_000, max_price, min(40_000_000, max_price), 500_000, key="butce_sec")

    with f_col3:
        st.markdown("**Yakınlık**")
        f_ulasim = st.selectbox("Toplu Ulaşım", ["Tümü", "1.5 km'den yakın", "3 km'den yakın"], key="f_ulasim")
        f_market = st.selectbox("Market / AVM", ["Tümü", "1.5 km'den yakın"], key="f_market")
        f_hastane = st.selectbox("Hastane", ["Tümü", "2 km'den yakın"], key="f_hastane")

    st.markdown("**Ek Özellikler**")
    o_col1, o_col2, o_col3, o_col4 = st.columns(4)
    with o_col1:
        f_site = st.checkbox("Site içinde", key="f_site")
        f_havuz = st.checkbox("Havuzlu", key="f_havuz")
    with o_col2:
        f_otopark = st.checkbox("Otoparklı", key="f_otopark")
        f_ebeveyn = st.checkbox("Ebeveyn Banyolu", key="f_ebeveyn")
    with o_col3:
        f_dubleks = st.checkbox("Dubleks", key="f_dubleks")
        f_bahce = st.checkbox("Bahçeli Kat", key="f_bahce")
    with o_col4:
        f_guvenlik = st.checkbox("Güvenlikli", key="f_guvenlik")
        f_sifir = st.checkbox("Sıfır / Yeni", key="f_sifir")

# ----- FİLTRELEME MANTIĞI -----
filtered = df.copy()

if selected_il != "Tüm İller":
    filtered = filtered[filtered['İl'] == selected_il]

if selected_ilce != "Tüm İlçeler":
    filtered = filtered[filtered['İlçe'] == selected_ilce]

if selected_mahalle != "Tüm Mahalleler":
    filtered = filtered[filtered['Mahalle'] == selected_mahalle]

if selected_oda != "Tümü":
    filtered = filtered[filtered['Oda Düzeni'] == selected_oda]

filtered = filtered[
    (filtered['m² (Brüt)'] >= m2_range[0]) &
    (filtered['m² (Brüt)'] <= m2_range[1]) &
    (filtered['Fiyat'] <= budget)
]

if f_ulasim == "1.5 km'den yakın":
    filtered = filtered[filtered['ulasim_mesafe_m'] <= 1500]
elif f_ulasim == "3 km'den yakın":
    filtered = filtered[filtered['ulasim_mesafe_m'] <= 3000]

if f_market == "1.5 km'den yakın":
    filtered = filtered[filtered['market_mesafe_m'] <= 1500]
if f_hastane == "2 km'den yakın":
    filtered = filtered[filtered['hastane_mesafe_m'] <= 2000]

if f_site:
    filtered = filtered[filtered['Ozellikler'].apply(lambda x: has_ozellik(x, "Site içinde"))]
if f_havuz:
    filtered = filtered[filtered['Ozellikler'].apply(lambda x: has_ozellik(x, "Havuz"))]
if f_otopark:
    filtered = filtered[filtered['Ozellikler'].apply(lambda x: has_ozellik(x, "Otopark"))]
if f_ebeveyn:
    filtered = filtered[filtered['Ozellikler'].apply(lambda x: has_ozellik(x, "Ebeveyn Banyolu"))]
if f_dubleks:
    filtered = filtered[filtered['Ozellikler'].apply(lambda x: has_ozellik(x, "Dubleks"))]
if f_bahce:
    filtered = filtered[filtered['Ozellikler'].apply(lambda x: has_ozellik(x, "Bahçeli"))]
if f_guvenlik:
    filtered = filtered[filtered['Ozellikler'].apply(lambda x: has_ozellik(x, "Güvenlik"))]
if f_sifir:
    filtered = filtered[filtered['Ozellikler'].apply(lambda x: has_ozellik(x, "Sıfır"))]

# =========================================================
# 4. TABLAR
# =========================================================
tab1, tab2, tab3, tab4, tab5, tab6 = st.tabs([
    "Piyasa İlanları",
    "Evimi Ne Kadara Satarım?",
    "İnteraktif Harita",
    "Oda & Isı Haritası",
    "Model Açıklanabilirliği",
    "Performans & Kredi"
])

# -------------------- TAB 1 --------------------
with tab1:
    st.subheader("Akıllı Arama")
    arama = st.text_input("Ne arıyorsunuz?", placeholder="3+1, havuzlu, Kadıköy, metro...", key="arama")

    if arama:
        metin = arama.lower().strip()
        mask = (
            filtered['İlan Başlığı'].str.lower().str.contains(metin, na=False) |
            filtered['İl'].str.lower().str.contains(metin, na=False) |
            filtered['İlçe'].str.lower().str.contains(metin, na=False) |
            filtered['Mahalle'].str.lower().str.contains(metin, na=False) |
            filtered['Oda Düzeni'].str.contains(metin, na=False)
        )
        if "metro" in metin or "ulaşım" in metin or "marmaray" in metin:
            mask = mask | (filtered['ulasim_mesafe_m'] <= 1500)
        if "market" in metin:
            mask = mask | (filtered['market_mesafe_m'] <= 1500)
        filtered = filtered[mask]
        st.success(f"**{len(filtered)}** ilan bulundu")

    st.subheader(f"Filtrelenen İlanlar ({len(filtered)} adet)")

    if len(filtered) == 0:
        st.warning("Seçtiğiniz kriterlere uygun ilan bulunamadı.")
    else:
        show_df = filtered.head(40).copy()
        show_df['Ulaşım'] = show_df['ulasim_mesafe_m'].apply(lambda x: f"~{x}m")
        show_df['Market'] = show_df['market_mesafe_m'].apply(lambda x: f"~{x}m")
        show_df['m² Birim Fiyatı'] = show_df['fiyat_m2'].apply(lambda x: f"{x:,.0f} TL")
        show_cols = ['İlan Başlığı', 'İlçe', 'Oda Düzeni', 'm² (Brüt)', 'm² Birim Fiyatı', 'Fiyat']
        
        event = st.dataframe(
            show_df[show_cols].style.format({'Fiyat': '{:,.0f} TL', 'm² (Brüt)': '{:.0f}'}),
            use_container_width=True, height=280,
            on_select="rerun",
            selection_mode="single-row",
            key="df_selection"
        )

        selected_rows = event.get("selection", {}).get("rows", [])
        
        st.divider()
        st.subheader("İlan Detayı & Yakınlık Bilgileri")
        
        if selected_rows:
            selected_idx = selected_rows[0]
            row = show_df.iloc[selected_idx]
        else:
            row = show_df.iloc[0]

        st.markdown(f"**{row['İlan Başlığı']}**")
        st.caption(f"{row['İl']} / {row['İlçe']} / {row['Mahalle']}  |  {row['Oda Düzeni']}  |  {row['m² (Brüt)']} m²  |  İlan: {row['Fiyat']:,.0f} TL")

        c1, c2, c3 = st.columns(3)
        c1.metric("Toplu Ulaşım", mesafe_etiket(row['ulasim_mesafe_m']))
        c2.metric("Market / AVM", mesafe_etiket(row['market_mesafe_m']))
        c3.metric("Hastane", mesafe_etiket(row['hastane_mesafe_m']))

        ozellikler = row['Ozellikler']
        if ozellikler:
            etiket_html = " ".join([f'<span class="ozellik-etiket">{o}</span>' for o in ozellikler])
            st.markdown(etiket_html, unsafe_allow_html=True)

        st.divider()
        st.subheader("⚖️ İlan Karşılaştırma Modülü")
        secilen_ilanlar = st.multiselect(
            "Karşılaştırılacak İlanları Seçin:",
            options=show_df['İlan Başlığı'].tolist(),
            max_selections=3,
            key="kiyas_secim"
        )

        if len(secilen_ilanlar) >= 2:
            cols = st.columns(len(secilen_ilanlar))
            for idx, baslik in enumerate(secilen_ilanlar):
                k_row = show_df[show_df['İlan Başlığı'] == baslik].iloc[0]
                with cols[idx]:
                    st.markdown(f"### İlan {idx+1}")
                    st.markdown(f"**{k_row['İlan Başlığı'][:30]}...**")
                    st.metric("Fiyat", f"{k_row['Fiyat']:,.0f} TL")
                    st.metric("Brüt m²", f"{k_row['m² (Brüt)']} m²")
                    st.write(f"**Konum:** {k_row['İlçe']}")
                    st.write(f"**Oda:** {k_row['Oda Düzeni']}")

# -------------------- TAB 2 --------------------
with tab2:
    st.subheader("Evimi Ne Kadara Satarım?")
    col1, col2 = st.columns(2)
    with col1:
        s_il = st.selectbox("İl", sorted(list(set(TURKIYE_ILLERI + df['İl'].unique().tolist()))), index=0, key="s_il")
        if s_il in df['İl'].values:
            s_ilce_options = sorted(df[df['İl'] == s_il]['İlçe'].unique().tolist())
        else:
            s_ilce_options = sorted(df['İlçe'].unique().tolist())
            
        s_ilce = st.selectbox("İlçe", s_ilce_options, key="s_ilce")
        s_brut = st.number_input("Brüt m²", 30, 800, 120, key="s_brut")
    with col2:
        s_oda = st.selectbox("Oda Sayısı", [1, 2, 3, 4, 5, 6, 7, 8], index=2, key="s_oda")
        s_salon = st.selectbox("Salon Sayısı", [0, 1, 2], index=1, key="s_salon")

    st.write("**Daire Özellikleri**")
    s_col1, s_col2 = st.columns(2)
    with s_col1:
        s_site = st.checkbox("Site içinde", key="s_site")
        s_havuz = st.checkbox("Havuzlu", key="s_havuz")
        s_otopark = st.checkbox("Otoparklı", key="s_otopark")
    with s_col2:
        s_ebeveyn = st.checkbox("Ebeveyn Banyolu", key="s_ebeveyn")
        s_sifir = st.checkbox("Sıfır / Yeni", key="s_sifir")

    if st.button("Satış Fiyatımı Hesapla", type="primary", use_container_width=True):
        benzer = df[df['İlçe'] == s_ilce]['Semt'].mode()
        semt_degeri = benzer.iloc[0] if len(benzer) > 0 else s_ilce
        ilce_df = df[df['İlçe'] == s_ilce]

        toplam_o = s_oda + s_salon
        m2_per_o = s_brut / max(toplam_o, 1)

        input_data = pd.DataFrame([{
            'm² (Brüt)': s_brut,
            'oda_sayisi': float(s_oda),
            'salon_sayisi': float(s_salon),
            'm2_per_oda': m2_per_o,
            'Semt': semt_degeri
        }])

        pred_mid = float(np.expm1(model_mid.predict(input_data)[0]))
        pred_low = float(np.expm1(model_low.predict(input_data)[0]))
        pred_high = float(np.expm1(model_high.predict(input_data)[0]))
        
        secilen_ozellikler = []
        multiplier = 1.0
        if s_site: 
            multiplier *= 1.04
            secilen_ozellikler.append("Site içinde")
        if s_havuz: 
            multiplier *= 1.03
            secilen_ozellikler.append("Havuzlu")
        if s_otopark: 
            multiplier *= 1.02
            secilen_ozellikler.append("Otoparklı")
        if s_ebeveyn: 
            multiplier *= 1.02
            secilen_ozellikler.append("Ebeveyn Banyolu")
        if s_sifir: 
            multiplier *= 1.05
            secilen_ozellikler.append("Sıfır / Yeni")

        pred_mid *= multiplier
        pred_low *= multiplier
        pred_high *= multiplier

        st.success("Hesaplama Tamamlandı")
        m1, m2, m3 = st.columns(3)
        m1.metric("Piyasa Değeri (%50)", f"{pred_mid:,.0f} TL")
        m2.metric("Alt Bant (%10)", f"{pred_low:,.0f} TL")
        m3.metric("Üst Bant (%90)", f"{pred_high:,.0f} TL")

        tahmini_aylik_kira = pred_mid / 220
        amortisman_yil = pred_mid / (tahmini_aylik_kira * 12)
        yillik_getiri_yuzde = ( (tahmini_aylik_kira * 12) / pred_mid ) * 100

        y1, y2, y3 = st.columns(3)
        y1.metric("Aylık Kira", f"{tahmini_aylik_kira:,.0f} TL")
        y2.metric("Amortisman", f"{amortisman_yil:.1f} Yıl")
        y3.metric("Yıllık Getiri", f"%{yillik_getiri_yuzde:.2f}")

        st.markdown(f"""
        <div class="rapor-kart">
            <h3>📋 Değerleme Rapor Özeti</h3>
            <p><strong>Konum:</strong> {s_il} / {s_ilce} | <strong>Büyüklük:</strong> {s_brut} m² | {s_oda}+{s_salon}</p>
            <ul>
                <li><strong>Tahmini Piyasa Satış Değeri:</strong> {pred_mid:,.0f} TL</li>
                <li><strong>Tahmini Kira Potansiyeli:</strong> {tahmini_aylik_kira:,.0f} TL / Ay</li>
                <li><strong>Amortisman Süresi:</strong> ~{amortisman_yil:.1f} Yıl</li>
            </ul>
        </div>
        """, unsafe_allow_html=True)

# -------------------- TAB 3 --------------------
with tab3:
    st.subheader("📍 İlanlar Haritası")
    coords = {
        'Ataşehir': (40.9833, 29.1167), 'Kadıköy': (40.9903, 29.0275), 'Üsküdar': (41.0244, 29.0050),
        'Beşiktaş': (41.0422, 29.0067), 'Şişli': (41.0600, 28.9870), 'Bakırköy': (40.9800, 28.8700),
        'Avcılar': (40.9801, 28.7175), 'Esenyurt': (41.0342, 28.6801), 'Pendik': (40.8750, 29.2333)
    }

    map_df = filtered.copy()
    if len(map_df) > 0:
        def get_coords(row):
            for name, (lat, lon) in coords.items():
                if name.lower() in str(row['İlçe']).lower():
                    return pd.Series([lat, lon])
            return pd.Series([41.0082, 28.9784])

        map_df[['lat', 'lon']] = map_df.apply(get_coords, axis=1)
        np.random.seed(42)
        map_df['lat'] += np.random.normal(0, 0.006, len(map_df))
        map_df['lon'] += np.random.normal(0, 0.006, len(map_df))

        fig_map = px.scatter_mapbox(
            map_df.head(400),
            lat="lat", lon="lon",
            color="Fiyat", size="m² (Brüt)",
            hover_name="İlan Başlığı",
            color_continuous_scale="Reds",
            size_max=12, zoom=9,
            center={"lat": 41.02, "lon": 28.95},
            mapbox_style="carto-darkmatter",
            height=450
        )
        fig_map.update_layout(margin=dict(l=0, r=0, t=0, b=0), template="plotly_dark")
        st.plotly_chart(fig_map, use_container_width=True)

# -------------------- TAB 4 --------------------
with tab4:
    st.subheader("Oda Düzeni & Fiyat Isı Haritası")
    col_a, col_b = st.columns(2)
    with col_a:
        oda_counts = df['Oda Düzeni'].value_counts().head(10).reset_index()
        oda_counts.columns = ['Oda Düzeni', 'İlan Sayısı']
        fig = px.bar(oda_counts, x='Oda Düzeni', y='İlan Sayısı', color='İlan Sayısı', color_continuous_scale='Blues')
        fig.update_layout(template="plotly_dark", height=320)
        st.plotly_chart(fig, use_container_width=True)
    with col_b:
        oda_price = df.groupby('Oda Düzeni')['Fiyat'].mean().sort_values(ascending=False).head(10).reset_index()
        oda_price.columns = ['Oda Düzeni', 'Ortalama Fiyat']
        fig = px.bar(oda_price, x='Oda Düzeni', y='Ortalama Fiyat', color='Ortalama Fiyat', color_continuous_scale='Reds')
        fig.update_layout(template="plotly_dark", height=320)
        st.plotly_chart(fig, use_container_width=True)

# -------------------- TAB 5 --------------------
with tab5:
    st.subheader("Model Açıklanabilirliği")
    try:
        importance = model_mid.get_feature_importance()
        names = ['m² (Brüt)', 'Oda Sayısı', 'Salon Sayısı', 'Oda Başı m²', 'Semt / İlçe']
        f_df = pd.DataFrame({'Öznitelik': names[:len(importance)], 'Önem': importance}).sort_values('Önem')
        fig = px.bar(f_df, x='Önem', y='Öznitelik', orientation='h', color='Önem', color_continuous_scale='Blues')
        fig.update_layout(template="plotly_dark", height=320)
        st.plotly_chart(fig, use_container_width=True)
    except:
        st.info("Hesaplanamadı.")

# -------------------- TAB 6 --------------------
with tab6:
    st.subheader("Model Performansı & Kredi Simülasyonu")
    benchmark = pd.DataFrame([
        {"Model": "CatBoost", "R²": f"%{model_r2*100:.2f}", "MAE": f"{model_mae:,.0f} TL"},
        {"Model": "LightGBM", "R²": "%93.00", "MAE": "806,166 TL"},
    ])
    st.dataframe(benchmark, use_container_width=True, hide_index=True)

    st.divider()
    tutar = st.number_input("Konut Değeri (TL)", value=5_000_000.0, step=100000.0, key="kredi_tutar")
    pesinat = st.slider("Peşinat Oranı (%)", 10, 90, 20, 5, key="pesinat_sec")
    vade = st.selectbox("Vade (Ay)", [60, 84, 120, 180, 240], index=2, key="vade_sec")
    pesinat_tutar = tutar * (pesinat / 100)
    kredi = tutar - pesinat_tutar
    if kredi > 0:
        r = 2.79 / 100
        taksit = (kredi * r * (1 + r)**vade) / ((1 + r)**vade - 1)
        st.metric("Aylık Taksit", f"{taksit:,.2f} TL")
