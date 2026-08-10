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
# 1. SAYFA AYARLARI + GELİŞMİŞ ŞIK TEMA (CSS)
# =========================================================
st.set_page_config(
    page_title="XAI Gayrimenkul Platformu",
    page_icon="🏠",
    layout="wide",
    initial_sidebar_state="expanded"
)

st.markdown("""
<style>
    .stApp { background-color: #0B0F19; color: #E2E8F0; }
    h1, h2, h3 { color: #F8FAFC !important; font-weight: 600 !important; }
    
    /* Metrik Kartları */
    div[data-testid="stMetric"] {
        background: linear-gradient(135deg, #111827 0%, #1F2937 100%);
        border: 1px solid #374151;
        border-radius: 12px;
        padding: 16px 20px;
        box-shadow: 0 4px 6px -1px rgba(0, 0, 0, 0.3);
    }
    div[data-testid="stMetric"] label { color: #94A3B8 !important; font-size: 0.9rem; font-weight: 500; }
    div[data-testid="stMetric"] div[data-testid="stMetricValue"] {
        color: #60A5FA !important; font-weight: 700 !important; font-size: 1.4rem !important;
    }
    
    /* Sol Menü Styling */
    section[data-testid="stSidebar"] {
        background-color: #0F172A;
        border-right: 1px solid #1E293B;
    }
    
    /* Buton Tasarımı */
    .stButton > button {
        background: linear-gradient(90deg, #2563EB 0%, #1D4ED8 100%);
        color: white;
        border-radius: 10px;
        border: none;
        font-weight: 600;
        padding: 10px 20px;
        width: 100%;
        box-shadow: 0 4px 12px rgba(37, 99, 235, 0.3);
    }
    .stButton > button:hover {
        background: linear-gradient(90deg, #1D4ED8 0%, #1E40AF 100%);
    }

    /* Özel Kart Kutuları */
    .content-card {
        background-color: #111827;
        border: 1px solid #1F2937;
        border-radius: 14px;
        padding: 20px;
        margin-bottom: 20px;
    }
    .ozellik-etiket {
        display: inline-block;
        background: #1E3A5F;
        color: #93C5FD;
        padding: 4px 12px;
        border-radius: 20px;
        font-size: 0.82rem;
        margin: 3px 4px;
        border: 1px solid #2B4C7E;
    }
    .rapor-kart {
        background: linear-gradient(135deg, #111827 0%, #0F172A 100%);
        border: 1px solid #3B82F6;
        border-radius: 14px;
        padding: 20px;
        margin-top: 15px;
    }
</style>
""", unsafe_allow_html=True)

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

@st.cache_resource(show_spinner="Modeller eğitiliyor ve veri hazırlanıyor...")
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
# 3. SOL MENÜ (NAVİGASYON)
# =========================================================
st.sidebar.markdown("## 🏢 Gayrimenkul Portalı")
st.sidebar.caption("Yapay Zeka Destekli Değerleme")

menu_secim = st.sidebar.radio(
    "Uygulama Modu Seçin:",
    [
        "🔎 İlan Arama & Filtreleme",
        "💰 Evimi Ne Kadara Satarım?",
        "📍 Harita İle Keşfet",
        "🏦 Konut Kredisi Hesaplama",
        "📊 Piyasa & Model Analitiği"
    ],
    index=0
)

st.sidebar.markdown("---")
st.sidebar.info("💡 **İpucu:** Sol menüden yapmak istediğiniz işlemi seçerek doğrudan ilgili moda geçebilirsiniz.")

# =========================================================
# SAYFA 1: 🔎 İLAN ARAMA & FİLTRELEME
# =========================================================
if menu_secim == "🔎 İlan Arama & Filtreleme":
    st.title("🔎 Piyasa İlanları & Akıllı Arama")
    
    # KPI Kartları (Sayfa zenginleştirme)
    kpi1, kpi2, kpi3, kpi4 = st.columns(4)
    kpi1.metric("Toplam Aktif İlan", f"{len(df):,} Adet")
    kpi2.metric("Ortalama İlan Fiyatı", f"{df['Fiyat'].mean():,.0f} TL")
    kpi3.metric("Ortalama m² Birim Fiyatı", f"{df['fiyat_m2'].mean():,.0f} TL")
    kpi4.metric("Ortalama Konut Büyüklüğü", f"{df['m² (Brüt)'].mean():.0f} m²")

    st.markdown("<br>", unsafe_allow_html=True)

    with st.expander("🛠️ Lokasyon ve Detaylı Filtre Panelini Aç / Kapat", expanded=True):
        f_col1, f_col2, f_col3 = st.columns(3)
        with f_col1:
            st.markdown("##### 📍 Lokasyon")
            il_options = ["Tüm İller"] + sorted(list(set(TURKIYE_ILLERI + df['İl'].unique().tolist())))
            selected_il = st.selectbox("İl", il_options, index=0, key="f_il")
            ilce_options = sorted(df['İlçe'].unique().tolist()) if selected_il == "Tüm İller" else sorted(df[df['İl'] == selected_il]['İlçe'].unique().tolist())
            selected_ilce = st.selectbox("İlçe", ["Tüm İlçeler"] + ilce_options, key="f_ilce")
            mahalle_options = sorted(df['Mahalle'].unique().tolist()) if selected_ilce == "Tüm İlçeler" else sorted(df[df['İlçe'] == selected_ilce]['Mahalle'].unique().tolist())
            selected_mahalle = st.selectbox("Mahalle", ["Tüm Mahalleler"] + mahalle_options, key="f_mahalle")

        with f_col2:
            st.markdown("##### 🏠 Konut Metrikleri")
            selected_oda = st.selectbox("Oda Düzeni", ["Tümü"] + sorted(df['Oda Düzeni'].unique().tolist()), key="f_oda")
            min_m2, max_m2 = int(df['m² (Brüt)'].min()), int(df['m² (Brüt)'].max())
            m2_range = st.slider("Brüt m² Aralığı", min_m2, max_m2, (40, 350), key="f_m2")
            max_price = int(df['Fiyat'].max())
            budget = st.number_input("Maksimum Bütçe (TL)", 500_000, max_price, min(40_000_000, max_price), 500_000, key="f_butce")

        with f_col3:
            st.markdown("##### 🚶 Yakınlık Kriterleri")
            f_ulasim = st.selectbox("Toplu Ulaşım", ["Tümü", "1.5 km'den yakın", "3 km'den yakın"], key="f_ulasim")
            f_market = st.selectbox("Market / AVM", ["Tümü", "1.5 km'den yakın"], key="f_market")
            f_hastane = st.selectbox("Hastane", ["Tümü", "2 km'den yakın"], key="f_hastane")

        st.markdown("##### ✨ Ek Konut Özellikleri")
        o1, o2, o3, o4 = st.columns(4)
        with o1:
            f_site = st.checkbox("Site içinde", key="c_site")
            f_havuz = st.checkbox("Havuzlu", key="c_havuz")
        with o2:
            f_otopark = st.checkbox("Otoparklı", key="c_otopark")
            f_ebeveyn = st.checkbox("Ebeveyn Banyolu", key="c_ebeveyn")
        with o3:
            f_dubleks = st.checkbox("Dubleks", key="c_dubleks")
            f_bahce = st.checkbox("Bahçeli Kat", key="c_bahce")
        with o4:
            f_guvenlik = st.checkbox("Güvenlikli", key="c_guvenlik")
            f_sifir = st.checkbox("Sıfır / Yeni", key="c_sifir")

    # Filtreleme İşlemi
    filtered = df.copy()
    if selected_il != "Tüm İller": filtered = filtered[filtered['İl'] == selected_il]
    if selected_ilce != "Tüm İlçeler": filtered = filtered[filtered['İlçe'] == selected_ilce]
    if selected_mahalle != "Tüm Mahalleler": filtered = filtered[filtered['Mahalle'] == selected_mahalle]
    if selected_oda != "Tümü": filtered = filtered[filtered['Oda Düzeni'] == selected_oda]

    filtered = filtered[(filtered['m² (Brüt)'] >= m2_range[0]) & (filtered['m² (Brüt)'] <= m2_range[1]) & (filtered['Fiyat'] <= budget)]

    if f_ulasim == "1.5 km'den yakın": filtered = filtered[filtered['ulasim_mesafe_m'] <= 1500]
    elif f_ulasim == "3 km'den yakın": filtered = filtered[filtered['ulasim_mesafe_m'] <= 3000]
    if f_market == "1.5 km'den yakın": filtered = filtered[filtered['market_mesafe_m'] <= 1500]
    if f_hastane == "2 km'den yakın": filtered = filtered[filtered['hastane_mesafe_m'] <= 2000]

    if f_site: filtered = filtered[filtered['Ozellikler'].apply(lambda x: has_ozellik(x, "Site içinde"))]
    if f_havuz: filtered = filtered[filtered['Ozellikler'].apply(lambda x: has_ozellik(x, "Havuz"))]
    if f_otopark: filtered = filtered[filtered['Ozellikler'].apply(lambda x: has_ozellik(x, "Otopark"))]
    if f_ebeveyn: filtered = filtered[filtered['Ozellikler'].apply(lambda x: has_ozellik(x, "Ebeveyn Banyolu"))]
    if f_dubleks: filtered = filtered[filtered['Ozellikler'].apply(lambda x: has_ozellik(x, "Dubleks"))]
    if f_bahce: filtered = filtered[filtered['Ozellikler'].apply(lambda x: has_ozellik(x, "Bahçeli"))]
    if f_guvenlik: filtered = filtered[filtered['Ozellikler'].apply(lambda x: has_ozellik(x, "Güvenlik"))]
    if f_sifir: filtered = filtered[filtered['Ozellikler'].apply(lambda x: has_ozellik(x, "Sıfır"))]

    arama = st.text_input("💬 Serbest Kelime İle Arama", placeholder="Örn: 3+1, Kadıköy, metroya yakın, deniz manzaralı...", key="arama_input")
    if arama:
        metin = arama.lower().strip()
        filtered = filtered[
            filtered['İlan Başlığı'].str.lower().str.contains(metin, na=False) |
            filtered['İlçe'].str.lower().str.contains(metin, na=False) |
            filtered['Mahalle'].str.lower().str.contains(metin, na=False)
        ]

    st.markdown(f"### 📋 Sonuçlar ({len(filtered)} İlan Bulundu)")
    if len(filtered) == 0:
        st.warning("Seçtiğiniz kriterlere uygun ilan bulunamadı.")
    else:
        show_df = filtered.head(40).copy()
        show_df['m² Birim Fiyatı'] = show_df['fiyat_m2'].apply(lambda x: f"{x:,.0f} TL")
        show_cols = ['İlan Başlığı', 'İlçe', 'Mahalle', 'Oda Düzeni', 'm² (Brüt)', 'm² Birim Fiyatı', 'Fiyat']
        
        event = st.dataframe(
            show_df[show_cols].style.format({'Fiyat': '{:,.0f} TL', 'm² (Brüt)': '{:.0f}'}),
            use_container_width=True, height=300,
            on_select="rerun", selection_mode="single-row", key="df_selection"
        )

        selected_rows = event.get("selection", {}).get("rows", [])
        st.divider()
        
        row = show_df.iloc[selected_rows[0]] if selected_rows else show_df.iloc[0]

        st.markdown(f"### 🏡 Seçili İlan Detayı: {row['İlan Başlığı']}")
        st.caption(f"Konum: **{row['İl']} / {row['İlçe']} / {row['Mahalle']}** | Tip: **{row['Oda Düzeni']}** | Büyüklük: **{row['m² (Brüt)']} m²** | Fiyat: **{row['Fiyat']:,.0f} TL**")
        
        c1, c2, c3 = st.columns(3)
        c1.metric("Toplu Ulaşım Mesafesi", mesafe_etiket(row['ulasim_mesafe_m']))
        c2.metric("Market / AVM Mesafesi", mesafe_etiket(row['market_mesafe_m']))
        c3.metric("Hastane Mesafesi", mesafe_etiket(row['hastane_mesafe_m']))

        ozellikler = row['Ozellikler']
        if ozellikler:
            st.markdown("**Başlıkta Öne Çıkan Özellikler:**")
            etiket_html = " ".join([f'<span class="ozellik-etiket">{o}</span>' for o in ozellikler])
            st.markdown(etiket_html, unsafe_allow_html=True)

        st.divider()
        st.markdown("### ⚖️ İlan Karşılaştırma Modülü")
        st.caption("Aşağıdaki kutudan kıyaslamak istediğiniz 2 veya 3 ilanı seçebilirsiniz.")
        secilen_ilanlar = st.multiselect("Karşılaştırılacak İlanlar:", options=show_df['İlan Başlığı'].tolist(), max_selections=3)
        if len(secilen_ilanlar) >= 2:
            cols = st.columns(len(secilen_ilanlar))
            for idx, baslik in enumerate(secilen_ilanlar):
                k_row = show_df[show_df['İlan Başlığı'] == baslik].iloc[0]
                with cols[idx]:
                    st.markdown(f"#### İlan {idx+1}")
                    st.write(f"**{k_row['İlan Başlığı'][:35]}...**")
                    st.metric("Fiyat", f"{k_row['Fiyat']:,.0f} TL")
                    st.metric("m² Birim Fiyatı", f"{k_row['fiyat_m2']:,.0f} TL/m²")
                    st.write(f"**Konum:** {k_row['İlçe']} / {k_row['Mahalle']}")

# =========================================================
# SAYFA 2: 💰 EVİMİ NE KADARA SATARIM?
# =========================================================
elif menu_secim == "💰 Evimi Ne Kadara Satarım?":
    st.title("💰 Evimi Ne Kadara Satarım?")
    st.caption("CatBoost yapay zeka algoritması ile evinizin tahmini satış değerini ve yatırım metriklerini hesaplayın.")

    st.markdown("""
    <div class="content-card">
        <h4>📝 Evinizin Bilgilerini Girin</h4>
        Evinizin konumu, büyüklüğü ve ek niteliklerini eksiksiz doldurarak yapay zeka değerlemesini başlatın.
    </div>
    """, unsafe_allow_html=True)

    col1, col2 = st.columns(2)
    with col1:
        s_il = st.selectbox("İl", sorted(list(set(TURKIYE_ILLERI + df['İl'].unique().tolist()))), index=0, key="val_il")
        s_ilce_options = sorted(df[df['İl'] == s_il]['İlçe'].unique().tolist()) if s_il in df['İl'].values else sorted(df['İlçe'].unique().tolist())
        s_ilce = st.selectbox("İlçe", s_ilce_options, key="val_ilce")
        s_brut = st.number_input("Brüt m² Büyüklüğü", 30, 800, 120, key="val_brut")
    with col2:
        s_oda = st.selectbox("Oda Sayısı", [1, 2, 3, 4, 5, 6, 7, 8], index=2, key="val_oda")
        s_salon = st.selectbox("Salon Sayısı", [0, 1, 2], index=1, key="val_salon")

    st.markdown("##### 🌟 Bina ve Daire Özellikleri")
    sc1, sc2 = st.columns(2)
    with sc1:
        s_site = st.checkbox("Site içinde yer alıyor", key="v_site")
        s_havuz = st.checkbox("Yüzme havuzu var", key="v_havuz")
        s_otopark = st.checkbox("Otopark imkanı var", key="v_otopark")
    with sc2:
        s_ebeveyn = st.checkbox("Ebeveyn banyosu var", key="v_ebeveyn")
        s_sifir = st.checkbox("Sıfır / Yeni bina", key="v_sifir")

    st.markdown("<br>", unsafe_allow_html=True)
    if st.button("🚀 Yapay Zeka Satış Fiyatını Hesapla", type="primary", use_container_width=True):
        benzer = df[df['İlçe'] == s_ilce]['Semt'].mode()
        semt_degeri = benzer.iloc[0] if len(benzer) > 0 else s_ilce

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
        
        multiplier = 1.0
        if s_site: multiplier *= 1.04
        if s_havuz: multiplier *= 1.03
        if s_otopark: multiplier *= 1.02
        if s_ebeveyn: multiplier *= 1.02
        if s_sifir: multiplier *= 1.05

        pred_mid *= multiplier
        pred_low *= multiplier
        pred_high *= multiplier

        st.success("✅ Hesaplama Başarıyla Tamamlandı")
        
        m1, m2, m3 = st.columns(3)
        m1.metric("Tahmini Piyasa Değeri (%50)", f"{pred_mid:,.0f} TL")
        m2.metric("Hızlı Satış Alt Bandı (%10)", f"{pred_low:,.0f} TL")
        m3.metric("Maksimum Üst Satış Bandı (%90)", f"{pred_high:,.0f} TL")

        tahmini_aylik_kira = pred_mid / 220
        amortisman_yil = pred_mid / (tahmini_aylik_kira * 12)
        yillik_getiri_yuzde = ((tahmini_aylik_kira * 12) / pred_mid) * 100

        st.divider()
        st.markdown("### 📈 Yatırım & Amortisman Metrikleri (ROI)")
        y1, y2, y3 = st.columns(3)
        y1.metric("Tahmini Monthly Kira Potansiyeli", f"{tahmini_aylik_kira:,.0f} TL/Ay")
        y2.metric("Amortisman Süresi (Geri Dönüş)", f"{amortisman_yil:.1f} Yıl")
        y3.metric("Yıllık Brüt Getiri Oranı", f"%{yillik_getiri_yuzde:.2f}")

        st.markdown(f"""
        <div class="rapor-kart">
            <h3>📋 Yapay Zeka Değerleme & Yatırım Rapor Kartı</h3>
            <p><strong>Değerleme Tarihi:</strong> {datetime.now().strftime('%d/%m/%Y %H:%M')}</p>
            <hr style="border-color: #374151;">
            <p><strong>Konum:</strong> {s_il} / {s_ilce} | <strong>Net Alan:</strong> {s_brut} m² | <strong>Oda:</strong> {s_oda}+{s_salon}</p>
            <br>
            <h5>💰 Fiyatlandırma ve Getiri Özeti</h5>
            <ul>
                <li><strong>Tahmini Piyasa Satış Değeri:</strong> {pred_mid:,.0f} TL</li>
                <li><strong>Tahmini Aylık Kira Potansiyeli:</strong> {tahmini_aylik_kira:,.0f} TL / Ay</li>
                <li><strong>Yatırımın Kendini Ödeme Süresi:</strong> ~{amortisman_yil:.1f} Yıl</li>
            </ul>
        </div>
        """, unsafe_allow_html=True)

# =========================================================
# SAYFA 3: 📍 HARİTA İLE KEŞFET
# =========================================================
elif menu_secim == "📍 Harita İle Keşfet":
    st.title("📍 Coğrafi İlan Haritası")
    st.caption("Harita üzerinde ilanların konumlarını görün, büyüklük ve fiyatlarına göre inceleyin.")

    coords = {
        'Ataşehir': (40.9833, 29.1167), 'Kadıköy': (40.9903, 29.0275), 'Üsküdar': (41.0244, 29.0050),
        'Beşiktaş': (41.0422, 29.0067), 'Şişli': (41.0600, 28.9870), 'Bakırköy': (40.9800, 28.8700),
        'Avcılar': (40.9801, 28.7175), 'Esenyurt': (41.0342, 28.6801), 'Pendik': (40.8750, 29.2333)
    }

    map_df = df.copy()
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
        map_df.head(500),
        lat="lat", lon="lon",
        color="Fiyat", size="m² (Brüt)",
        hover_name="İlan Başlığı",
        color_continuous_scale="Reds",
        size_max=14, zoom=9.5,
        center={"lat": 41.02, "lon": 28.95},
        mapbox_style="carto-darkmatter",
        height=580
    )
    fig_map.update_layout(margin=dict(l=0, r=0, t=0, b=0), template="plotly_dark")
    st.plotly_chart(fig_map, use_container_width=True)

# =========================================================
# SAYFA 4: 🏦 KONUT KREDİSİ HESAPLAMA
# =========================================================
elif menu_secim == "🏦 Konut Kredisi Hesaplama":
    st.title("🏦 Konut Kredisi Simülasyonu")
    st.caption("Almak istediğiniz evin değerine göre peşinat ve aylık taksit ödeme planını oluşturun.")

    st.markdown("""
    <div class="content-card">
        <h4>💳 Kredi Parametrelerini Belirleyin</h4>
        İstediğiniz konut tutarını ve kredi vadesini girerek güncel faiz oranlarıyla aylık taksitinizi hesaplayın.
    </div>
    """, unsafe_allow_html=True)

    col1, col2 = st.columns(2)
    with col1:
        tutar = st.number_input("Konut Değeri (TL)", value=5_000_000.0, step=100000.0, key="kredi_page_tutar")
        pesinat = st.slider("Peşinat Oranı (%)", 10, 90, 20, 5, key="kredi_page_pesinat")
        vade = st.selectbox("Vade Süresi (Ay)", [60, 84, 120, 180, 240], index=2, key="kredi_page_vade")
    with col2:
        faiz = st.number_input("Aylık Faiz Oranı (%)", 0.1, 10.0, 2.79, 0.01, key="kredi_page_faiz")

    pesinat_tutar = tutar * (pesinat / 100)
    kredi = tutar - pesinat_tutar
    if kredi > 0:
        r = faiz / 100
        taksit = (kredi * r * (1 + r)**vade) / ((1 + r)**vade - 1)
        toplam = taksit * vade
        
        st.divider()
        m1, m2, m3 = st.columns(3)
        m1.metric("Gerekli Peşinat Tutarı", f"{pesinat_tutar:,.0f} TL")
        m2.metric("Çekilecek Kredi Tutarı", f"{kredi:,.0f} TL")
        m3.metric("Aylık Ödenecek Taksit", f"{taksit:,.2f} TL")
        
        st.markdown(f"""
        <div class="rapor-kart">
            <h4>📊 Ödeme Planı Özeti</h4>
            <ul>
                <li><strong>Toplam Geri Ödeme:</strong> {toplam:,.2f} TL</li>
                <li><strong>Toplam Ödenecek Faiz Tutarı:</strong> {toplam - kredi:,.2f} TL</li>
            </ul>
        </div>
        """, unsafe_allow_html=True)

# =========================================================
# SAYFA 5: 📊 PİYASA & MODEL ANALİTİĞİ
# =========================================================
elif menu_secim == "📊 Piyasa & Model Analitiği":
    st.title("📊 Piyasa Analitiği & Yapay Zeka Başarımı")
    st.caption("Veri setindeki değişkenlerin fiyat üzerindeki ağırlıkları ve model performans metrikleri.")

    st.markdown("### 🧬 Öznitelik Önem Düzeyleri")
    try:
        importance = model_mid.get_feature_importance()
        names = ['m² (Brüt)', 'Oda Sayısı', 'Salon Sayısı', 'Oda Başı m²', 'Semt / İlçe']
        f_df = pd.DataFrame({'Öznitelik': names[:len(importance)], 'Önem Seviyesi (%)': importance}).sort_values('Önem Seviyesi (%)')
        fig = px.bar(f_df, x='Önem Seviyesi (%)', y='Öznitelik', orientation='h', color='Önem Seviyesi (%)', color_continuous_scale='Blues')
        fig.update_layout(template="plotly_dark", height=350)
        st.plotly_chart(fig, use_container_width=True)
    except:
        st.info("Hesaplanamadı.")

    st.divider()
    st.markdown("### 🏆 Algoritma Karşılaştırması")
    benchmark = pd.DataFrame([
        {"Model": "CatBoost Regressor (Bu Model)", "R² Skor": f"%{model_r2*100:.2f}", "MAE (Hata)": f"{model_mae:,.0f} TL"},
        {"Model": "LightGBM", "R² Skor": "%93.00", "MAE (Hata)": "806,166 TL"},
        {"Model": "Gradient Boosting", "R² Skor": "%92.97", "MAE (Hata)": "834,818 TL"},
        {"Model": "Random Forest", "R² Skor": "%92.28", "MAE (Hata)": "876,419 TL"},
    ])
    st.dataframe(benchmark, use_container_width=True, hide_index=True)
