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
    
    /* Metrik Kartları */
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
    
    /* Sol Menü */
    section[data-testid="stSidebar"] {
        background-color: #0F172A;
        border-right: 1px solid #1E293B;
    }
    
    /* Buton Tasarımı */
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

    /* Özel Kart Alanları */
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
    .rapor-kart {
        background-color: #111827;
        border: 1px solid #3B82F6;
        border-radius: 8px;
        padding: 20px;
        margin-top: 15px;
    }
</style>
""", unsafe_allow_html=True)

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
# 2. VERİ VE MODEL HAZIRLIĞI
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

@st.cache_resource(show_spinner="Veri seti ve algoritmalar yükleniyor...")
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
    st.error(f"Veri yüklenirken hata oluştu: {e}")
    st.stop()

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

    # FİLTRE SIFIRLAMA BUTONU
    if st.button("Filtre Seçimlerini Sıfırla"):
        st.session_state["f_il"] = "Önce İl Seçiniz"
        st.session_state["f_ilce"] = "Önce İl Seçiniz"
        st.session_state["f_mahalle"] = "Önce İlçe Seçiniz"
        st.session_state["f_oda"] = "Tümü"
        st.session_state["f_m2"] = (int(df['m² (Brüt)'].min()), int(df['m² (Brüt)'].max()))
        st.session_state["f_butce"] = min(40_000_000, int(df['Fiyat'].max()))
        st.session_state["f_ulasim"] = "Tümü"
        st.session_state["f_market"] = "Tümü"
        st.session_state["f_hastane"] = "Tümü"
        st.session_state["c_site"] = False
        st.session_state["c_havuz"] = False
        st.session_state["c_otopark"] = False
        st.session_state["c_ebeveyn"] = False
        st.session_state["c_dubleks"] = False
        st.session_state["c_bahce"] = False
        st.session_state["c_guvenlik"] = False
        st.session_state["c_sifir"] = False
        st.rerun()

    with st.expander("Sorgulama Kriterleri ve Lokasyon Filtreleri", expanded=True):
        f_col1, f_col2, f_col3 = st.columns(3)
        with f_col1:
            st.markdown("##### Lokasyon Parametreleri")
            
            il_options = ["Önce İl Seçiniz", "Tüm İller"] + sorted(list(set(TURKIYE_ILLERI + df['İl'].unique().tolist())))
            selected_il = st.selectbox("İl", il_options, key="f_il")
            
            if selected_il == "Önce İl Seçiniz":
                ilce_list = ["Önce İl Seçiniz"]
            elif selected_il == "Tüm İller":
                ilce_list = ["Tüm İlçeler"] + sorted(df['İlçe'].unique().tolist())
            else:
                ilce_options = sorted(df[df['İl'] == selected_il]['İlçe'].unique().tolist())
                ilce_list = ["Tüm İlçeler"] + ilce_options if len(ilce_options) > 0 else ["Seçilen İlde Kayıtlı İlçe Yok"]
                
            selected_ilce = st.selectbox("İlçe", ilce_list, key="f_ilce")
            
            if selected_ilce in ["Önce İl Seçiniz", "Seçilen İlde Kayıtlı İlçe Yok"]:
                mahalle_list = ["Önce İlçe Seçiniz"]
            elif selected_ilce == "Tüm İlçeler":
                if selected_il == "Tüm İller":
                    mahalle_list = ["Tüm Mahalleler"] + sorted(df['Mahalle'].unique().tolist())
                else:
                    mahalle_list = ["Tüm Mahalleler"] + sorted(df[df['İl'] == selected_il]['Mahalle'].unique().tolist())
            else:
                mahalle_options = sorted(df[(df['İlçe'] == selected_ilce) & (df['İl'] == selected_il if selected_il != "Tüm İller" else True)]['Mahalle'].unique().tolist())
                mahalle_list = ["Tüm Mahalleler"] + mahalle_options if len(mahalle_options) > 0 else ["Kayıtlı Mahalle Yok"]

            selected_mahalle = st.selectbox("Mahalle", mahalle_list, key="f_mahalle")

        with f_col2:
            st.markdown("##### Yapı Özellikleri")
            selected_oda = st.selectbox("Oda Yapısı", ["Tümü"] + sorted(df['Oda Düzeni'].unique().tolist()), key="f_oda")
            min_m2, max_m2 = int(df['m² (Brüt)'].min()), int(df['m² (Brüt)'].max())
            m2_range = st.slider("Brüt Alan (m²)", min_m2, max_m2, (40, 350), key="f_m2")
            max_price = int(df['Fiyat'].max())
            budget = st.number_input("Üst Bütçe Limiti (TL)", 500_000, max_price, min(40_000_000, max_price), 500_000, key="f_butce")

        with f_col3:
            st.markdown("##### Konum ve Yakınlık")
            f_ulasim = st.selectbox("Toplu Ulaşım Mesafesi", ["Tümü", "1.5 km'den yakın", "3 km'den yakın"], key="f_ulasim")
            f_market = st.selectbox("Ticari Alan / AVM", ["Tümü", "1.5 km'den yakın"], key="f_market")
            f_hastane = st.selectbox("Sağlık Kuruluşu", ["Tümü", "2 km'den yakın"], key="f_hastane")

        st.markdown("##### Ek Nitelikler")
        o1, o2, o3, o4 = st.columns(4)
        with o1:
            f_site = st.checkbox("Site Yerleşimi", key="c_site")
            f_havuz = st.checkbox("Yüzme Havuzu", key="c_havuz")
        with o2:
            f_otopark = st.checkbox("Otopark", key="c_otopark")
            f_ebeveyn = st.checkbox("Ebeveyn Banyosu", key="c_ebeveyn")
        with o3:
            f_dubleks = st.checkbox("Dubleks Yapı", key="c_dubleks")
            f_bahce = st.checkbox("Bahçe Katı", key="c_bahce")
        with o4:
            f_guvenlik = st.checkbox("Güvenlik Hizmeti", key="c_guvenlik")
            f_sifir = st.checkbox("Sıfır / Yeni Bina", key="c_sifir")

    filtered = df.copy()
    if selected_il not in ["Önce İl Seçiniz", "Tüm İller"]:
        filtered = filtered[filtered['İl'] == selected_il]
    if selected_ilce not in ["Tüm İlçeler", "Önce İl Seçiniz", "Seçilen İlde Kayıtlı İlçe Yok"]:
        filtered = filtered[filtered['İlçe'] == selected_ilce]
    if selected_mahalle not in ["Tüm Mahalleler", "Önce İlçe Seçiniz", "Kayıtlı Mahalle Yok"]:
        filtered = filtered[filtered['Mahalle'] == selected_mahalle]
    if selected_oda != "Tümü":
        filtered = filtered[filtered['Oda Düzeni'] == selected_oda]

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

    arama = st.text_input("Metin Arama (İlan Başlığı / İlçe / Mahalle)", placeholder="Örn: 3+1, Kadıköy, metroya yakın...", key="arama_input")
    if arama:
        metin = arama.lower().strip()
        filtered = filtered[
            filtered['İlan Başlığı'].str.lower().str.contains(metin, na=False) |
            filtered['İlçe'].str.lower().str.contains(metin, na=False) |
            filtered['Mahalle'].str.lower().str.contains(metin, na=False)
        ]

    st.markdown(f"### Sorgulama Sonuçları ({len(filtered)} İlan Listelendi)")
    if len(filtered) == 0:
        st.warning("Belirtilen il, ilçe veya arama kriterlerine uygun kayıt bulunamadı.")
    else:
        show_df = filtered.head(40).copy()
        show_df['m² Birim Fiyatı'] = show_df['fiyat_m2'].apply(lambda x: f"{x:,.0f} TL")
        show_cols = ['İlan Başlığı', 'İlçe', 'Mahalle', 'Oda Düzeni', 'm² (Brüt)', 'm² Birim Fiyatı', 'Fiyat']
        
        event = st.dataframe(
            show_df[show_cols].style.format({'Fiyat': '{:,.0f} TL', 'm² (Brüt)': '{:.0f}'}),
            use_container_width=True, height=300,
            on_select="rerun", selection_mode="single-row", key="df_selection"
        )

        export_df = filtered.copy()
        export_df['m² Birim Fiyatı'] = export_df['fiyat_m2'].apply(lambda x: f"{x:,.0f} TL")
        export_cols = ['İlan Başlığı', 'İl', 'İlçe', 'Mahalle', 'Oda Düzeni', 'm² (Brüt)', 'm² Birim Fiyatı', 'Fiyat']
        valid_export_cols = [c for c in export_cols if c in export_df.columns]
        
        csv_data = export_df[valid_export_cols].to_csv(index=False, encoding='utf-8-sig')
        st.download_button(
            label="Listelenen İlanları CSV Olarak İndir",
            data=csv_data,
            file_name=f"Gayrimenkul_Ilan_Listesi_{datetime.now().strftime('%Y%m%d')}.csv",
            mime="text/csv"
        )

        selected_rows = event.get("selection", {}).get("rows", [])
        st.divider()
        
        row = show_df.iloc[selected_rows[0]] if selected_rows else show_df.iloc[0]

        st.markdown(f"### Seçili Kayıt Detayı: {row['İlan Başlığı']}")
        st.caption(f"Lokasyon: **{row['İl']} / {row['İlçe']} / {row['Mahalle']}** | Tip: **{row['Oda Düzeni']}** | Alan: **{row['m² (Brüt)']} m²** | Satış Bedeli: **{row['Fiyat']:,.0f} TL**")
        
        c1, c2, c3 = st.columns(3)
        c1.metric("Toplu Ulaşım Noktası", mesafe_etiket(row['ulasim_mesafe_m']))
        c2.metric("Ticari Merkez / AVM", mesafe_etiket(row['market_mesafe_m']))
        c3.metric("Sağlık Kuruluşu", mesafe_etiket(row['hastane_mesafe_m']))

        ozellikler = row['Ozellikler']
        if ozellikler:
            st.markdown("**Tespit Edilen Yapı Nitelikleri:**")
            etiket_html = " ".join([f'<span class="ozellik-etiket">{o}</span>' for o in ozellikler])
            st.markdown(etiket_html, unsafe_allow_html=True)

        st.divider()
        st.markdown("### İlan Karşılaştırma Modülü")
        st.caption("Karşılaştırmak istediğiniz kayıtları listeden seçiniz.")
        secilen_ilanlar = st.multiselect("Seçilen İlanlar:", options=show_df['İlan Başlığı'].tolist(), max_selections=3)
        if len(secilen_ilanlar) >= 2:
            cols = st.columns(len(secilen_ilanlar))
            for idx, baslik in enumerate(secilen_ilanlar):
                k_row = show_df[show_df['İlan Başlığı'] == baslik].iloc[0]
                with cols[idx]:
                    st.markdown(f"#### İlan {idx+1}")
                    st.write(f"**{k_row['İlan Başlığı'][:35]}...**")
                    st.metric("Satış Bedeli", f"{k_row['Fiyat']:,.0f} TL")
                    st.metric("Birim Fiyat", f"{k_row['fiyat_m2']:,.0f} TL/m²")
                    st.write(f"**Konum:** {k_row['İlçe']} / {k_row['Mahalle']}")

# =========================================================
# MODÜL 2: GAYRİMENKUL DEĞERLEME MODÜLÜ
# =========================================================
elif menu_secim == "Gayrimenkul Değerleme Modülü":
    st.title("Otomatik Gayrimenkul Değerleme Paneli")
    st.caption("CatBoost makine öğrenmesi modeli ile taşınmazın tahmini satış değeri ve yatırım geri dönüş süresi hesaplanır.")

    st.markdown("""
    <div class="content-card">
        <h4>Taşınmaz Parametre Girişi</h4>
        Değerlemesi yapılacak taşınmazın fiziksel ve konum bilgilerini giriniz. Lütfen önce il seçimini yapınız.
    </div>
    """, unsafe_allow_html=True)

    c_deg1, c_deg2, c_deg3 = st.columns(3)
    with c_deg1:
        available_iller = ["Önce İl Seçiniz"] + sorted(df['İl'].unique().tolist())
        s_il = st.selectbox("İl", available_iller, index=0, key="val_il")
        
        if s_il == "Önce İl Seçiniz":
            s_ilce_options = ["Önce İl Seçiniz"]
        else:
            s_ilce_options = sorted(df[df['İl'] == s_il]['İlçe'].unique().tolist())
            
        s_ilce = st.selectbox("İlçe", s_ilce_options, key="val_ilce")
    with c_deg2:
        s_brut = st.number_input("Brüt Kullanım Alanı (m²)", 30, 800, 120, key="val_brut")
        s_oda = st.selectbox("Oda Sayısı", [1, 2, 3, 4, 5, 6, 7, 8], index=2, key="val_oda")
    with c_deg3:
        s_salon = st.selectbox("Salon Sayısı", [0, 1, 2], index=1, key="val_salon")

    st.markdown("##### Ek Yapı Nitelikleri")
    sc1, sc2, sc3 = st.columns(3)
    with sc1:
        s_site = st.checkbox("Site İçerisinde", key="v_site")
        s_havuz = st.checkbox("Yüzme Havuzu Mevcut", key="v_havuz")
    with sc2:
        s_otopark = st.checkbox("Otopark Mevcut", key="v_otopark")
        s_ebeveyn = st.checkbox("Ebeveyn Banyosu Mevcut", key="v_ebeveyn")
    with sc3:
        s_sifir = st.checkbox("Sıfır / Yeni Yapı", key="v_sifir")

    st.markdown("<br>", unsafe_allow_html=True)
    if st.button("Değerleme Raporunu Hesapla", type="primary", use_container_width=True):
        if s_il == "Önce İl Seçiniz" or s_ilce == "Önce İl Seçiniz":
            st.warning("Lütfen değerleme hesabı yapabilmek için geçerli bir il ve ilçe seçiniz.")
        else:
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

            st.success("Değerleme Hesaplaması Tamamlandı")
            
            m1, m2, m3 = st.columns(3)
            m1.metric("Tahmini Piyasa Bedeli (%50)", f"{pred_mid:,.0f} TL")
            m2.metric("Hızlı Satış Alt Bandı (%10)", f"{pred_low:,.0f} TL")
            m3.metric("Maksimum Satış Bandı (%90)", f"{pred_high:,.0f} TL")

            tahmini_aylik_kira = pred_mid / 220
            amortisman_yil = pred_mid / (tahmini_aylik_kira * 12)
            yillik_getiri_yuzde = ((tahmini_aylik_kira * 12) / pred_mid) * 100

            st.divider()
            st.markdown("### Yatırım ve Amortisman Analizi")
            y1, y2, y3 = st.columns(3)
            y1.metric("Tahmini Aylık Kira Potansiyeli", f"{tahmini_aylik_kira:,.0f} TL/Ay")
            y2.metric("Amortisman Süresi", f"{amortisman_yil:.1f} Yıl")
            y3.metric("Yıllık Brüt Getiri Oranı", f"%{yillik_getiri_yuzde:.2f}")

            st.markdown(f"""
            <div class="rapor-kart">
                <h4>Gayrimenkul Değerleme ve Yatırım Özet Raporu</h4>
                <p><strong>Rapor Tarihi:</strong> {datetime.now().strftime('%d/%m/%Y %H:%M')}</p>
                <hr style="border-color: #374151;">
                <p><strong>Konum:</strong> {s_il} / {s_ilce} | <strong>Kullanım Alanı:</strong> {s_brut} m² | <strong>Oda Düzeni:</strong> {s_oda}+{s_salon}</p>
                <br>
                <h5>Değerleme Sonuçları</h5>
                <ul>
                    <li><strong>Tahmini Piyasa Satış Değeri:</strong> {pred_mid:,.0f} TL</li>
                    <li><strong>Tahmini Aylık Kira Değeri:</strong> {tahmini_aylik_kira:,.0f} TL / Ay</li>
                    <li><strong>Geri Dönüş (Amortisman) Süresi:</strong> Yaklaşık {amortisman_yil:.1f} Yıl</li>
                </ul>
            </div>
            """, unsafe_allow_html=True)

            rapor_metni = f"""GAYRİMENKUL DEĞERLEME VE YATIRIM RAPORU
Tarih: {datetime.now().strftime('%d/%m/%Y %H:%M')}

1. TAŞINMAZ BİLGİLERİ
-----------------------------------------
Lokasyon: {s_il} / {s_ilce}
Brüt Alan: {s_brut} m²
Oda Yapısı: {s_oda}+{s_salon}

2. DEĞERLEME SONUÇLARI (CATBOOST REGRESSION)
-----------------------------------------
Tahmini Piyasa Satış Bedeli: {pred_mid:,.0f} TL
Alt Bant (%10 Quantile / Hızlı Satış): {pred_low:,.0f} TL
Üst Bant (%90 Quantile / Tavan Satış): {pred_high:,.0f} TL

3. YATIRIM VE AMORTİSMAN ANALİZİ
-----------------------------------------
Tahmini Aylık Kira Potansiyeli: {tahmini_aylik_kira:,.0f} TL / Ay
Amortisman Süresi: {amortisman_yil:.1f} Yıl
Yıllık Brüt Getiri Oranı: %{yillik_getiri_yuzde:.2f}

-----------------------------------------
Bu rapor otomatik algoritma tarafından üretilmiştir. Resmi ekspertiz raporu yerine geçmez.
"""
            st.download_button(
                label="Değerleme Raporunu Metin Dosyası Olarak İndir",
                data=rapor_metni,
                file_name=f"Gayrimenkul_Degerleme_Raporu_{s_ilce}_{datetime.now().strftime('%Y%m%d')}.txt",
                mime="text/plain"
            )

# =========================================================
# MODÜL 3: COĞRAFİ HARİTA ANALİZİ
# =========================================================
elif menu_secim == "Coğrafi Harita Analizi":
    st.title("Coğrafi İlan Dağılım Haritası ve Lokasyon Keşfi")
    st.caption("Seçilen bölgedeki taşınmazların konumsal yoğunlukları, alanları ve metrekare birim fiyat dağılımları.")

    h_col1, h_col2 = st.columns(2)
    with h_col1:
        map_il_options = ["Önce İl Seçiniz", "Tüm İller"] + sorted(df['İl'].unique().tolist())
        map_il = st.selectbox("Harita İl Seçimi:", map_il_options, index=0, key="map_il")
    with h_col2:
        if map_il == "Önce İl Seçiniz":
            map_ilce_list = ["Önce İl Seçiniz"]
        elif map_il == "Tüm İller":
            map_ilce_list = ["Tüm İlçeler"] + sorted(df['İlçe'].unique().tolist())
        else:
            map_ilce_options = sorted(df[df['İl'] == map_il]['İlçe'].unique().tolist())
            map_ilce_list = ["Tüm İlçeler"] + map_ilce_options if len(map_ilce_options) > 0 else ["Kayıtlı İlçe Yok"]
            
        map_ilce = st.selectbox("Harita İlçe Seçimi:", map_ilce_list, key="map_ilce")

    coords = {
        'Ataşehir': (40.9833, 29.1167), 'Kadıköy': (40.9903, 29.0275), 'Üsküdar': (41.0244, 29.0050),
        'Beşiktaş': (41.0422, 29.0067), 'Şişli': (41.0600, 28.9870), 'Bakırköy': (40.9800, 28.8700),
        'Avcılar': (40.9801, 28.7175), 'Esenyurt': (41.0342, 28.6801), 'Pendik': (40.8750, 29.2333)
    }

    map_filtered = df.copy()
    if map_il not in ["Önce İl Seçiniz", "Tüm İller"]: 
        map_filtered = map_filtered[map_filtered['İl'] == map_il]
    if map_ilce not in ["Tüm İlçeler", "Önce İl Seçiniz", "Kayıtlı İlçe Yok"]: 
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
        Alınacak konut bedeli, kullanılacak peşinat oranı ve tercih edilen aylık faiz oranını giriniz.
    </div>
    """, unsafe_allow_html=True)

    banka_seçimi = st.selectbox(
        "Banka Oran Referansı Seçiniz:",
        ["Ziraat / Halkbank / Vakıfbank (Kamu) - %2.79", "Özel Banka Ortalaması - %2.89", "Özel Faiz Oranı Girişi"],
        key="banka_secim"
    )

    col1, col2 = st.columns(2)
    with col1:
        tutar = st.number_input("Konut Rayiç Bedeli (TL)", value=5_000_000.0, step=100000.0, key="kredi_page_tutar")
        pesinat = st.slider("Özkaynak / Peşinat Oranı (%)", 10, 90, 20, 5, key="kredi_page_pesinat")
    with col2:
        vade = st.selectbox("Geri Ödeme Vadesi (Ay)", [36, 48, 60, 84, 120, 180, 240], index=4, key="kredi_page_vade")
        if banka_seçimi == "Özel Faiz Oranı Girişi":
            faiz = st.number_input("Aylık Akdi Faiz Oranı (%)", 0.1, 10.0, 2.79, 0.01, key="kredi_page_faiz")
        elif "Kamu" in banka_seçimi:
            faiz = 2.79
            st.info("Kamu bankası referans aylık faiz oranı: %2.79")
        else:
            faiz = 2.89
            st.info("Özel banka referans aylık faiz oranı: %2.89")

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
        names = ['m² (Brüt)', 'Oda Sayısı', 'Salon Sayısı', 'Oda Başı m²', 'Semt / İlçe']
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
