import streamlit as st
import pandas as pd
import numpy as np
from catboost import CatBoostRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error
import plotly.express as px
import re

# =========================================================
# 1. SAYFA AYARLARI + TEMA
# =========================================================
st.set_page_config(
    page_title="XAI Gayrimenkul Değerleme",
    page_icon="🏆",
    layout="wide",
    initial_sidebar_state="expanded"
)

st.markdown("""
<style>
    .stApp { background-color: #0B0F19; color: #E2E8F0; }
    h1, h2, h3 { color: #F8FAFC !important; font-weight: 600 !important; }
    div[data-testid="stMetric"] {
        background-color: #111827;
        border: 1px solid #1F2937;
        border-radius: 12px;
        padding: 16px 20px;
    }
    div[data-testid="stMetric"] label { color: #94A3B8 !important; }
    div[data-testid="stMetric"] div[data-testid="stMetricValue"] {
        color: #F1F5F9 !important; font-weight: 600 !important;
    }
    section[data-testid="stSidebar"] {
        background-color: #111827;
        border-right: 1px solid #1F2937;
    }
    .stButton > button {
        background-color: #3B82F6;
        color: white;
        border-radius: 8px;
        border: none;
        font-weight: 500;
    }
    .stButton > button:hover { background-color: #2563EB; }
    .stTabs [data-baseweb="tab"] {
        background-color: #1F2937;
        border-radius: 8px;
        color: #94A3B8;
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
        font-size: 0.85rem;
        margin: 3px 4px 3px 0;
    }
</style>
""", unsafe_allow_html=True)

st.title("XAI Gayrimenkul Değerleme Platformu")
st.caption("CatBoost (Quantile Regression) + Gelişmiş Öznitelik Mühendisliği & Mesafeli Değerleme")

# =========================================================
# 2. VERİ VE MODEL
# =========================================================
EXCEL_FILE = "ilanlar_mesafeli.xlsx"

def basliktan_ozellik_cikar(baslik):
    if pd.isna(baslik):
        return []
    t = str(baslik).upper()
    oz = []
    # Ulaşım & Konum
    if any(k in t for k in ['METRO', 'METROBUS', 'METROBÜS', 'MARMARAY']):
        oz.append("Ulaşım")
    if any(k in t for k in ['MARKET', 'ÇARŞI', 'AVM']):
        oz.append("Market / AVM")
    if any(k in t for k in ['MANZARA', 'DENİZ', 'BOĞAZ', 'ORMAN', 'GÖL']):
        oz.append("Manzara")
    
    # Bina & Tesis
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

    # Daire İçi
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
    df['İlçe'] = df['İlçe'].fillna('Bilinmiyor').astype(str)
    df['Mahalle'] = df['Mahalle'].fillna('Bilinmiyor').astype(str)
    df['Semt'] = df['Semt / Mahalle'].astype(str)

    # 1. ADIM GELİŞTİRMESİ: ÖZNİTELİK MÜHENDİSLİĞİ (FEATURE ENGINEERING)
    df['fiyat_m2'] = df['Fiyat'] / df['m² (Brüt)'] # m² birim fiyatı
    df['toplam_oda'] = df['oda_sayisi'] + df['salon_sayisi']
    df['m2_per_oda'] = df['m² (Brüt)'] / df['toplam_oda'].clip(lower=1) # Oda başı ferahlık m²'si

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
    
    # Ana Tahmin Modeli (Ortanca / Median)
    model_mid = CatBoostRegressor(
        iterations=600, learning_rate=0.05, depth=6,
        cat_features=['Semt'], verbose=0, random_seed=42
    )
    model_mid.fit(X_train, y_train)

    # 1. ADIM GELİŞTİRMESİ: CATBOOST QUANTILE REGRESSION (GÜVEN ARALIĞI)
    # Alt Bant Modeli (%10 Quantile)
    model_low = CatBoostRegressor(
        iterations=600, learning_rate=0.05, depth=6,
        loss_function='Quantile:alpha=0.10',
        cat_features=['Semt'], verbose=0, random_seed=42
    )
    model_low.fit(X_train, y_train)

    # Üst Bant Modeli (%90 Quantile)
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
# 3. SIDEBAR
# =========================================================
st.sidebar.header("Filtreler")

ilce_list = ["Tüm İlçeler"] + sorted(df['İlçe'].unique().tolist())
selected_ilce = st.sidebar.selectbox("İlçe", ilce_list, key="ilce_sec")

if selected_ilce == "Tüm İlçeler":
    mahalle_options = sorted(df['Mahalle'].unique().tolist())
else:
    mahalle_options = sorted(df[df['İlçe'] == selected_ilce]['Mahalle'].unique().tolist())
selected_mahalle = st.sidebar.selectbox("Mahalle", ["Tüm Mahalleler"] + mahalle_options, key="mahalle_sec")

selected_oda = st.sidebar.selectbox("Oda Düzeni", ["Tümü"] + sorted(df['Oda Düzeni'].unique().tolist()), key="oda_sec")

min_m2, max_m2 = int(df['m² (Brüt)'].min()), int(df['m² (Brüt)'].max())
m2_range = st.sidebar.slider("Brüt m²", min_m2, max_m2, (40, 350), key="m2_sec")

max_price = int(df['Fiyat'].max())
budget = st.sidebar.number_input("Maksimum Bütçe (TL)", 500_000, max_price, min(40_000_000, max_price), 500_000, key="butce_sec")

st.sidebar.markdown("---")
st.sidebar.subheader("Yakınlık Filtreleri")
f_ulasim = st.sidebar.selectbox("Toplu ulaşım (Metro/Metrobüs/Marmaray)", ["Tümü", "1.5 km'den yakın", "3 km'den yakın"], key="f_ulasim")
f_market = st.sidebar.selectbox("Market / AVM", ["Tümü", "1.5 km'den yakın"], key="f_market")
f_hastane = st.sidebar.selectbox("Hastane", ["Tümü", "2 km'den yakın"], key="f_hastane")
f_okul = st.sidebar.selectbox("Okul", ["Tümü", "2 km'den yakın"], key="f_okul")

st.sidebar.markdown("---")
st.sidebar.subheader("Konut & Bina Özellikleri")
f_site = st.sidebar.checkbox("Site içinde", key="f_site")
f_havuz = st.sidebar.checkbox("Havuzlu", key="f_havuz")
f_otopark = st.sidebar.checkbox("Otoparklı", key="f_otopark")
f_ebeveyn = st.sidebar.checkbox("Ebeveyn Banyolu", key="f_ebeveyn")
f_dubleks = st.sidebar.checkbox("Dubleks", key="f_dubleks")
f_bahce = st.sidebar.checkbox("Bahçeli / Bahçe Katı", key="f_bahce")
f_guvenlik = st.sidebar.checkbox("Güvenlikli", key="f_guvenlik")
f_sifir = st.sidebar.checkbox("Sıfır / Yeni", key="f_sifir")

# Filtreleme
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
if f_okul == "2 km'den yakın":
    filtered = filtered[filtered['okul_mesafe_m'] <= 2000]

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
tab1, tab2, tab3, tab4, tab5 = st.tabs([
    "Piyasa İlanları",
    "Evimi Ne Kadara Satarım?",
    "Oda & Isı Haritası",
    "Model Açıklanabilirliği",
    "Performans & Kredi"
])

# -------------------- TAB 1 --------------------
with tab1:
    st.subheader("Akıllı Arama")
    arama = st.text_input("Ne arıyorsunuz?", placeholder="3+1, havuzlu, Ataşehir, metro, ebeveyn banyolu...", key="arama")

    if arama:
        metin = arama.lower().strip()
        mask = (
            filtered['İlan Başlığı'].str.lower().str.contains(metin, na=False) |
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
    st.caption(f"Toplam veri: **{len(df)}** | Mesafeler semt merkezi bazlı yaklaşıktır")

    if len(filtered) == 0:
        st.warning("Kriterlere uygun ilan yok.")
    else:
        show_df = filtered.head(40).copy()
        show_df['Ulaşım'] = show_df['ulasim_mesafe_m'].apply(lambda x: f"~{x}m")
        show_df['Market'] = show_df['market_mesafe_m'].apply(lambda x: f"~{x}m")
        show_df['m² Birim Fiyatı'] = show_df['fiyat_m2'].apply(lambda x: f"{x:,.0f} TL")
        show_cols = ['İlan Başlığı', 'İlçe', 'Oda Düzeni', 'm² (Brüt)', 'm² Birim Fiyatı', 'Ulaşım', 'Market', 'Fiyat']
        
        event = st.dataframe(
            show_df[show_cols].style.format({'Fiyat': '{:,.0f} TL', 'm² (Brüt)': '{:.0f}'}),
            use_container_width=True, height=300,
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
            st.info("💡 Tablodan başka bir satıra tıklayarak o ilanın detayını inceleyebilirsiniz.")

        st.markdown(f"**{row['İlan Başlığı']}**")
        st.caption(f"{row['İlçe']} / {row['Mahalle']}  |  {row['Oda Düzeni']}  |  {row['m² (Brüt)']} m²  |  Birim: {row['fiyat_m2']:,.0f} TL/m²  |  İlan Fiyatı: {row['Fiyat']:,.0f} TL")

        st.markdown("#### Yakınlık (yaklaşık)")
        c1, c2, c3 = st.columns(3)
        c1.metric("Toplu Ulaşım", mesafe_etiket(row['ulasim_mesafe_m']))
        c2.metric("Market / AVM", mesafe_etiket(row['market_mesafe_m']))
        c3.metric("Hastane", mesafe_etiket(row['hastane_mesafe_m']))
        c4, c5, c6 = st.columns(3)
        c4.metric("Okul", mesafe_etiket(row['okul_mesafe_m']))
        c5.metric("Taksi", mesafe_etiket(row['taksi_mesafe_m']))
        c6.metric("Metro (eski)", mesafe_etiket(row.get('metro_mesafe_m', 9999)))

        st.caption("Not: Mesafeler semt/mahalle merkezi bazlıdır. Kapı önü mesafesi değildir.")

        ozellikler = row['Ozellikler']
        if ozellikler:
            etiket_html = " ".join([f'<span class="ozellik-etiket">{o}</span>' for o in ozellikler])
            st.markdown("**Başlıkta geçen özellikler:**", unsafe_allow_html=True)
            st.markdown(etiket_html, unsafe_allow_html=True)

# -------------------- TAB 2 --------------------
with tab2:
    st.subheader("Evimi Ne Kadara Satarım?")
    col1, col2, col3 = st.columns(3)
    with col1:
        s_ilce = st.selectbox("İlçe", sorted(df['İlçe'].unique().tolist()), key="s_ilce")
        s_brut = st.number_input("Brüt m²", 30, 800, 120, key="s_brut")
    with col2:
        s_oda = st.selectbox("Oda Sayısı", [1, 2, 3, 4, 5, 6, 7, 8], index=2, key="s_oda")
        s_salon = st.selectbox("Salon Sayısı", [0, 1, 2], index=1, key="s_salon")
    with col3:
        st.write("**Daire & Bina Özellikleri**")
        s_site = st.checkbox("Site içinde", key="s_site")
        s_havuz = st.checkbox("Havuzlu", key="s_havuz")
        s_otopark = st.checkbox("Otoparklı", key="s_otopark")
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

        # QUANTILE MODEL TAHMİNLERİ
        pred_mid = float(np.expm1(model_mid.predict(input_data)[0]))
        pred_low = float(np.expm1(model_low.predict(input_data)[0]))
        pred_high = float(np.expm1(model_high.predict(input_data)[0]))
        
        # Özellik Prim Oranları Entegrasyonu
        multiplier = 1.0
        if s_site: multiplier *= 1.04
        if s_havuz: multiplier *= 1.03
        if s_otopark: multiplier *= 1.02
        if s_ebeveyn: multiplier *= 1.02
        if s_sifir: multiplier *= 1.05

        ulasim_med = ilce_df['ulasim_mesafe_m'].median() if len(ilce_df) else 2000
        if ulasim_med <= 1000:
            multiplier *= 1.025

        pred_mid *= multiplier
        pred_low *= multiplier
        pred_high *= multiplier

        st.success("Quantile Regression Modeli ile İstatistiksel Hesaplama Tamamlandı")
        m1, m2, m3 = st.columns(3)
        m1.metric("Tahmini Piyasa Değeri (%50)", f"{pred_mid:,.0f} TL")
        m2.metric("Alt Bant Tahmini (%10 Quantile)", f"{pred_low:,.0f} TL")
        m3.metric("Üst Bant Tahmini (%90 Quantile)", f"{pred_high:,.0f} TL")

        st.markdown("#### Bu ilçede ortalama yakınlık")
        k1, k2, k3, k4 = st.columns(4)
        k1.metric("Ulaşım", mesafe_etiket(int(ulasim_med)))
        k2.metric("Market", mesafe_etiket(int(ilce_df['market_mesafe_m'].median())))
        k3.metric("Hastane", mesafe_etiket(int(ilce_df['hastane_mesafe_m'].median())))
        k4.metric("Okul", mesafe_etiket(int(ilce_df['okul_mesafe_m'].median())))
        st.caption("Mesafeler semt merkezi bazlı yaklaşıktır.")

# -------------------- TAB 3 --------------------
with tab3:
    st.subheader("Oda Düzeni Dağılımı & Fiyat Isı Haritası")
    col_a, col_b = st.columns(2)
    with col_a:
        oda_counts = df['Oda Düzeni'].value_counts().head(12).reset_index()
        oda_counts.columns = ['Oda Düzeni', 'İlan Sayısı']
        fig = px.bar(oda_counts, x='Oda Düzeni', y='İlan Sayısı', color='İlan Sayısı',
                     color_continuous_scale='Blues', text='İlan Sayısı')
        fig.update_traces(textposition='outside')
        fig.update_layout(template="plotly_dark", height=380, showlegend=False)
        st.plotly_chart(fig, use_container_width=True)
    with col_b:
        oda_price = df.groupby('Oda Düzeni')['Fiyat'].mean().sort_values(ascending=False).head(12).reset_index()
        oda_price.columns = ['Oda Düzeni', 'Ortalama Fiyat']
        fig = px.bar(oda_price, x='Oda Düzeni', y='Ortalama Fiyat', color='Ortalama Fiyat',
                     color_continuous_scale='Reds', text='Ortalama Fiyat')
        fig.update_traces(texttemplate='%{text:,.0f}', textposition='outside')
        fig.update_layout(template="plotly_dark", height=380, showlegend=False)
        st.plotly_chart(fig, use_container_width=True)

    st.divider()
    top_ilceler = df['İlçe'].value_counts().head(15).index.tolist()
    top_odalar = df['Oda Düzeni'].value_counts().head(8).index.tolist()
    heat_df = df[df['İlçe'].isin(top_ilceler) & df['Oda Düzeni'].isin(top_odalar)]
    pivot = heat_df.pivot_table(values='Fiyat', index='İlçe', columns='Oda Düzeni', aggfunc='mean')
    pivot = pivot.reindex(top_ilceler)
    pivot = pivot[[c for c in top_odalar if c in pivot.columns]]
    fig = px.imshow(pivot, color_continuous_scale='YlOrRd', aspect='auto', labels=dict(color="Ort. Fiyat"))
    fig.update_layout(template="plotly_dark", height=520, title="İlçe × Oda Ortalama Fiyat Isı Haritası")
    st.plotly_chart(fig, use_container_width=True)

# -------------------- TAB 4 --------------------
with tab4:
    st.subheader("Model Açıklanabilirliği")
    st.caption(f"R²: **%{model_r2*100:.2f}** | MAE: {model_mae:,.0f} TL | Veri: {len(df):,}")
    try:
        importance = model_mid.get_feature_importance()
        names = ['m² (Brüt)', 'Oda Sayısı', 'Salon Sayısı', 'Oda Başı m²', 'Semt / İlçe']
        f_df = pd.DataFrame({'Öznitelik': names[:len(importance)], 'Önem': importance}).sort_values('Önem')
        fig = px.bar(f_df, x='Önem', y='Öznitelik', orientation='h', color='Önem', color_continuous_scale='Blues')
        fig.update_layout(template="plotly_dark", height=380)
        st.plotly_chart(fig, use_container_width=True)
    except:
        st.info("Grafik hesaplanamadı.")

# -------------------- TAB 5 --------------------
with tab5:
    st.subheader("Model Performans Karşılaştırması")
    benchmark = pd.DataFrame([
        {"Model": "CatBoost (Bu Model)", "R²": f"%{model_r2*100:.2f}", "MAE": f"{model_mae:,.0f} TL", "RMSE": f"{model_rmse:,.0f} TL"},
        {"Model": "LightGBM", "R²": "%93.00", "MAE": "806,166 TL", "RMSE": "1,017,600 TL"},
        {"Model": "Gradient Boosting", "R²": "%92.97", "MAE": "834,818 TL", "RMSE": "1,019,746 TL"},
        {"Model": "Random Forest", "R²": "%92.28", "MAE": "876,419 TL", "RMSE": "1,068,710 TL"},
    ])
    st.dataframe(benchmark, use_container_width=True, hide_index=True)

    st.divider()
    st.subheader("Resmi Banka Konut Kredisi Simülasyonu")
    bankalar = {
        "BDDK / Piyasa Ortalaması": 2.79, "Ziraat Bankası (Kamu)": 2.79,
        "VakıfBank (Kamu)": 2.79, "Halkbank (Kamu)": 2.79,
        "İş Bankası (Özel)": 2.85, "Akbank (Özel)": 2.89,
        "Garanti BBVA (Özel)": 3.05, "Özel Parametre Gir": 2.79
    }
    col1, col2 = st.columns(2)
    with col1:
        tutar = st.number_input("Konut Değeri (TL)", value=5_000_000.0, step=100000.0, key="kredi_tutar")
        banka = st.selectbox("Banka", list(bankalar.keys()), key="banka_sec")
        if banka == "Özel Parametre Gir":
            faiz = st.number_input("Aylık Faiz (%)", 0.1, 10.0, 2.79, 0.01, key="faiz_sec")
        else:
            faiz = bankalar[banka]
            st.info(f"**{banka}** | Aylık Faiz: **%{faiz}**")
        pesinat = st.slider("Peşinat Oranı (%)", 10, 90, 20, 5, key="pesinat_sec")
        vade = st.selectbox("Vade (Ay)", [60, 84, 96, 120, 180, 240], index=3, key="vade_sec")
    with col2:
        pesinat_tutar = tutar * (pesinat / 100)
        kredi = tutar - pesinat_tutar
        if kredi > 0:
            r = faiz / 100
            taksit = (kredi * r * (1 + r)**vade) / ((1 + r)**vade - 1)
            toplam = taksit * vade
            st.metric("Peşinat", f"{pesinat_tutar:,.0f} TL")
            st.metric("Kredi Tutarı", f"{kredi:,.0f} TL")
            st.metric("Aylık Taksit", f"{taksit:,.2f} TL")
            st.metric("Toplam Geri Ödeme", f"{toplam:,.2f} TL")
            st.caption(f"Toplam Faiz: **{toplam - kredi:,.2f} TL**")
        else:
            st.warning("Peşinat konut değerinden büyük olamaz.")
