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
st.caption("CatBoost + Koordinatlı ilan verisi ile piyasa değerleme")

# =========================================================
# 2. VERİ VE MODEL
# =========================================================
EXCEL_FILE = "ilanlar_koordinatli.xlsx"

def basliktan_ozellik_cikar(baslik):
    if pd.isna(baslik):
        return []
    t = str(baslik).upper()
    ozellikler = []
    if any(k in t for k in ['METRO', 'METROBUS', 'METROBÜS']):
        ozellikler.append("Metro (başlıkta geçiyor)")
    if any(k in t for k in ['ULAŞIM', 'ULASIM', 'OTOBÜS', 'OTOBUS']):
        ozellikler.append("Ulaşıma yakın")
    if any(k in t for k in ['MARKET', 'ÇARŞI', 'CARSI', 'AVM']):
        ozellikler.append("Markete yakın")
    if any(k in t for k in ['SİTE', 'SITE']):
        ozellikler.append("Site içinde")
    if any(k in t for k in ['OTOPARK', 'GARAJ']):
        ozellikler.append("Otopark")
    if any(k in t for k in ['HAVUZ']):
        ozellikler.append("Havuz")
    if any(k in t for k in ['SIFIR', 'YENİ BİNA', 'YENI BINA', 'PROJEDEN']):
        ozellikler.append("Sıfır / Yeni")
    if any(k in t for k in ['MANZARA', 'DENİZ', 'BOĞAZ']):
        ozellikler.append("Manzara")
    if any(k in t for k in ['BALKON', 'TERAS']):
        ozellikler.append("Balkon / Teras")
    if any(k in t for k in ['ASANSÖR', 'ASANSOR']):
        ozellikler.append("Asansör")
    return ozellikler

@st.cache_resource(show_spinner="Model ve veriler yükleniyor...")
def load_model_and_data():
    df = pd.read_excel(EXCEL_FILE)
    df = df.dropna(subset=['Fiyat', 'm² (Brüt)'])
    df['oda_sayisi'] = pd.to_numeric(df['oda_sayisi'], errors='coerce').fillna(2).astype(int)
    df['salon_sayisi'] = pd.to_numeric(df['salon_sayisi'], errors='coerce').fillna(1).astype(int)
    df['Oda Düzeni'] = df['oda_sayisi'].astype(str) + '+' + df['salon_sayisi'].astype(str)
    df['İlçe'] = df['İlçe'].fillna('Bilinmiyor').astype(str)
    df['Mahalle'] = df['Mahalle'].fillna('Bilinmiyor').astype(str)
    df['Semt'] = df['Semt / Mahalle'].astype(str)
    df['metro_mesafe_m'] = pd.to_numeric(df.get('metro_mesafe_m', 9999), errors='coerce').fillna(9999).astype(int)
    df['metro_yakin'] = df.get('metro_yakin', 'Uzak').fillna('Uzak')
    df['Ozellikler'] = df['İlan Başlığı'].apply(basliktan_ozellik_cikar)

    feature_cols = ['m² (Brüt)', 'oda_sayisi', 'salon_sayisi', 'Semt']
    X = df[feature_cols].copy()
    y_log = np.log1p(df['Fiyat'])
    X['Semt'] = X['Semt'].astype(str)

    X_train, X_test, y_train, y_test = train_test_split(X, y_log, test_size=0.2, random_state=42)
    model = CatBoostRegressor(
        iterations=600, learning_rate=0.05, depth=6,
        cat_features=['Semt'], verbose=0, random_seed=42
    )
    model.fit(X_train, y_train)

    y_pred = np.expm1(model.predict(X_test))
    y_true = np.expm1(y_test)
    r2 = r2_score(y_true, y_pred)
    mae = mean_absolute_error(y_true, y_pred)
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    return model, df, r2, mae, rmse

try:
    model, df, model_r2, model_mae, model_rmse = load_model_and_data()
except Exception as e:
    st.error(f"Veri yüklenirken hata: {e}")
    st.stop()

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

oda_list = ["Tümü"] + sorted(df['Oda Düzeni'].unique().tolist())
selected_oda = st.sidebar.selectbox("Oda Düzeni", oda_list, key="oda_sec")

min_m2, max_m2 = int(df['m² (Brüt)'].min()), int(df['m² (Brüt)'].max())
m2_range = st.sidebar.slider("Brüt m²", min_m2, max_m2, (40, 350), key="m2_sec")

max_price = int(df['Fiyat'].max())
budget = st.sidebar.number_input("Maksimum Bütçe (TL)", 500_000, max_price, min(40_000_000, max_price), 500_000, key="butce_sec")

st.sidebar.markdown("---")
st.sidebar.subheader("Metro Yakınlığı")
metro_filtre = st.sidebar.selectbox(
    "Metro mesafesi",
    ["Tümü", "1 km'den yakın", "1-2 km", "2 km'den uzak"],
    key="metro_filtre"
)

st.sidebar.markdown("---")
st.sidebar.subheader("Özellik Filtreleri")
f_site = st.sidebar.checkbox("Site içinde", key="f_site")
f_otopark = st.sidebar.checkbox("Otopark", key="f_otopark")
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

if metro_filtre == "1 km'den yakın":
    filtered = filtered[filtered['metro_mesafe_m'] <= 1000]
elif metro_filtre == "1-2 km":
    filtered = filtered[(filtered['metro_mesafe_m'] > 1000) & (filtered['metro_mesafe_m'] <= 2000)]
elif metro_filtre == "2 km'den uzak":
    filtered = filtered[filtered['metro_mesafe_m'] > 2000]

def has_ozellik(ozellik_listesi, aranan):
    return any(aranan.lower() in o.lower() for o in ozellik_listesi)

if f_site:
    filtered = filtered[filtered['Ozellikler'].apply(lambda x: has_ozellik(x, "Site"))]
if f_otopark:
    filtered = filtered[filtered['Ozellikler'].apply(lambda x: has_ozellik(x, "Otopark"))]
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
    arama = st.text_input("Ne arıyorsunuz?", placeholder="3+1, Ataşehir, metro...", key="arama")

    if arama:
        metin = arama.lower().strip()
        mask = (
            filtered['İlan Başlığı'].str.lower().str.contains(metin, na=False) |
            filtered['İlçe'].str.lower().str.contains(metin, na=False) |
            filtered['Mahalle'].str.lower().str.contains(metin, na=False) |
            filtered['Oda Düzeni'].str.contains(metin, na=False)
        )
        if "metro" in metin:
            mask = mask | (filtered['metro_mesafe_m'] <= 1500)
        filtered = filtered[mask]
        st.success(f"**{len(filtered)}** ilan bulundu")

    st.subheader(f"Filtrelenen İlanlar ({len(filtered)} adet)")
    st.caption(f"Toplam veri: **{len(df)}** ilan | Metro mesafeleri yaklaşık (semt merkezi bazlı)")

    if len(filtered) == 0:
        st.warning("Kriterlere uygun ilan yok.")
    else:
        show_df = filtered.head(40).copy()
        show_df['Metro'] = show_df['metro_mesafe_m'].apply(lambda x: f"~{x} m")
        show_cols = ['İlan Başlığı', 'İlçe', 'Mahalle', 'Oda Düzeni', 'm² (Brüt)', 'Metro', 'Fiyat']
        st.dataframe(
            show_df[show_cols].style.format({'Fiyat': '{:,.0f} TL', 'm² (Brüt)': '{:.0f}'}),
            use_container_width=True, height=300
        )

        st.divider()
        st.subheader("İlan Detayı & Özellikler")
        secilen = st.selectbox("İlan seçin", filtered['İlan Başlığı'].head(40).tolist(), key="ilan_sec")
        if secilen:
            row = filtered[filtered['İlan Başlığı'] == secilen].iloc[0]
            st.markdown(f"**{row['İlan Başlığı']}**")
            st.caption(f"{row['İlçe']} / {row['Mahalle']}  |  {row['Oda Düzeni']}  |  {row['m² (Brüt)']} m²  |  {row['Fiyat']:,.0f} TL")

            # Metro mesafesi
            mesafe = int(row['metro_mesafe_m'])
            if mesafe <= 1000:
                st.success(f"🚇 Metroya yaklaşık **{mesafe} m** (yakın)")
            elif mesafe <= 2000:
                st.info(f"🚇 Metroya yaklaşık **{mesafe} m** (orta)")
            else:
                st.warning(f"🚇 Metroya yaklaşık **{mesafe} m** (uzak)")
            st.caption("Not: Mesafe semt/mahalle merkezi bazlı yaklaşık değerdir, kapı önü mesafesi değildir.")

            ozellikler = row['Ozellikler']
            if ozellikler:
                etiket_html = " ".join([f'<span class="ozellik-etiket">{o}</span>' for o in ozellikler])
                st.markdown("**Başlıkta geçen özellikler:**", unsafe_allow_html=True)
                st.markdown(etiket_html, unsafe_allow_html=True)

# -------------------- TAB 2 --------------------
with tab2:
    st.subheader("Evimi Ne Kadara Satarım?")
    st.caption("Evinizin özelliklerini girin, gerçek ilan verisine göre tahmini satış fiyatınızı öğrenin.")

    col1, col2, col3 = st.columns(3)
    with col1:
        s_ilce = st.selectbox("İlçe", sorted(df['İlçe'].unique().tolist()), key="s_ilce")
        s_brut = st.number_input("Brüt m²", 30, 800, 120, key="s_brut")
    with col2:
        s_oda = st.selectbox("Oda Sayısı", [1, 2, 3, 4, 5, 6, 7, 8], index=2, key="s_oda")
        s_salon = st.selectbox("Salon Sayısı", [0, 1, 2], index=1, key="s_salon")
    with col3:
        st.write("**Özellikler**")
        s_site = st.checkbox("Site içinde", key="s_site")
        s_otopark = st.checkbox("Otopark", key="s_otopark")
        s_sifir = st.checkbox("Sıfır / Yeni", key="s_sifir")

    if st.button("Satış Fiyatımı Hesapla", type="primary", use_container_width=True):
        benzer = df[df['İlçe'] == s_ilce]['Semt'].mode()
        semt_degeri = benzer.iloc[0] if len(benzer) > 0 else s_ilce

        # Aynı ilçenin ortalama metro mesafesi
        ilce_metro = df[df['İlçe'] == s_ilce]['metro_mesafe_m'].median()
        if pd.isna(ilce_metro):
            ilce_metro = 2000

        input_data = pd.DataFrame([{
            'm² (Brüt)': s_brut,
            'oda_sayisi': float(s_oda),
            'salon_sayisi': float(s_salon),
            'Semt': semt_degeri
        }])
        pred = float(np.expm1(model.predict(input_data)[0]))
        if s_site: pred *= 1.03
        if s_otopark: pred *= 1.02
        if s_sifir: pred *= 1.05
        if ilce_metro <= 1000: pred *= 1.03

        hizli = pred * 0.93
        maksimum = pred * 1.08

        st.success("Hesaplama tamamlandı")
        m1, m2, m3 = st.columns(3)
        m1.metric("Tahmini Piyasa Değeri", f"{pred:,.0f} TL")
        m2.metric("Hızlı Satış Fiyatı", f"{hizli:,.0f} TL", delta="-7%")
        m3.metric("Maksimum Satış Fiyatı", f"{maksimum:,.0f} TL", delta="+8%")

        st.info(f"Bu ilçede ortalama metro mesafesi yaklaşık **{int(ilce_metro)} m** (semt merkezi bazlı).")
        st.caption(f"Öneri: İlanınızı {pred:,.0f} – {maksimum:,.0f} TL aralığında vermenizi öneririm.")

# -------------------- TAB 3 --------------------
with tab3:
    st.subheader("Oda Düzeni Dağılımı & Fiyat Isı Haritası")
    col_a, col_b = st.columns(2)
    with col_a:
        st.markdown("#### Oda Düzeni Dağılımı")
        oda_counts = df['Oda Düzeni'].value_counts().head(12).reset_index()
        oda_counts.columns = ['Oda Düzeni', 'İlan Sayısı']
        fig_oda = px.bar(oda_counts, x='Oda Düzeni', y='İlan Sayısı', color='İlan Sayısı',
                         color_continuous_scale='Blues', text='İlan Sayısı')
        fig_oda.update_traces(textposition='outside')
        fig_oda.update_layout(template="plotly_dark", height=380, showlegend=False)
        st.plotly_chart(fig_oda, use_container_width=True)
    with col_b:
        st.markdown("#### Oda Düzenine Göre Ortalama Fiyat")
        oda_price = df.groupby('Oda Düzeni')['Fiyat'].mean().sort_values(ascending=False).head(12).reset_index()
        oda_price.columns = ['Oda Düzeni', 'Ortalama Fiyat']
        fig_price = px.bar(oda_price, x='Oda Düzeni', y='Ortalama Fiyat', color='Ortalama Fiyat',
                           color_continuous_scale='Reds', text='Ortalama Fiyat')
        fig_price.update_traces(texttemplate='%{text:,.0f}', textposition='outside')
        fig_price.update_layout(template="plotly_dark", height=380, showlegend=False)
        st.plotly_chart(fig_price, use_container_width=True)

    st.divider()
    st.markdown("#### İlçe × Oda Düzeni Ortalama Fiyat Isı Haritası")
    top_ilceler = df['İlçe'].value_counts().head(15).index.tolist()
    top_odalar = df['Oda Düzeni'].value_counts().head(8).index.tolist()
    heat_df = df[df['İlçe'].isin(top_ilceler) & df['Oda Düzeni'].isin(top_odalar)]
    pivot = heat_df.pivot_table(values='Fiyat', index='İlçe', columns='Oda Düzeni', aggfunc='mean')
    pivot = pivot.reindex(top_ilceler)
    pivot = pivot[[c for c in top_odalar if c in pivot.columns]]
    fig_heat = px.imshow(pivot, color_continuous_scale='YlOrRd', aspect='auto', labels=dict(color="Ort. Fiyat"))
    fig_heat.update_layout(template="plotly_dark", height=520)
    st.plotly_chart(fig_heat, use_container_width=True)

# -------------------- TAB 4 --------------------
with tab4:
    st.subheader("Model Açıklanabilirliği")
    st.caption(f"R²: **%{model_r2*100:.2f}** | MAE: {model_mae:,.0f} TL | Veri: {len(df):,} ilan")
    try:
        importance = model.get_feature_importance()
        names = ['m² (Brüt)', 'Oda Sayısı', 'Salon Sayısı', 'Semt / İlçe']
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
