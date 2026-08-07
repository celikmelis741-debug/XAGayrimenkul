import streamlit as st
import pandas as pd
import numpy as np
from catboost import CatBoostRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error
import plotly.express as px

# =========================================================
# 1. SAYFA AYARLARI + TEMA
# =========================================================
st.set_page_config(
    page_title="XAI Gayrimenkul Değerleme",
    page_icon="🏆",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Profesyonel koyu tema
st.markdown("""
<style>
    .stApp {
        background-color: #0B0F19;
        color: #E2E8F0;
    }
    h1, h2, h3 {
        color: #F8FAFC !important;
        font-weight: 600 !important;
    }
    div[data-testid="stMetric"] {
        background-color: #111827;
        border: 1px solid #1F2937;
        border-radius: 12px;
        padding: 16px 20px;
    }
    div[data-testid="stMetric"] label {
        color: #94A3B8 !important;
    }
    div[data-testid="stMetric"] div[data-testid="stMetricValue"] {
        color: #F1F5F9 !important;
        font-weight: 600 !important;
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
    .stButton > button:hover {
        background-color: #2563EB;
    }
    .stTabs [data-baseweb="tab"] {
        background-color: #1F2937;
        border-radius: 8px;
        color: #94A3B8;
    }
    .stTabs [aria-selected="true"] {
        background-color: #3B82F6 !important;
        color: white !important;
    }
</style>
""", unsafe_allow_html=True)

st.title("XAI Gayrimenkul Değerleme Platformu")
st.caption("CatBoost tabanlı açıklanabilir yapay zekâ destekli değerleme sistemi")

# =========================================================
# 2. VERİ VE MODEL YÜKLEME
# =========================================================
EXCEL_FILE = "sahibinden_ilanlar_temiz_cloud.xlsx"

@st.cache_resource(show_spinner="Model ve veriler yükleniyor...")
def load_model_and_data():
    df_raw = pd.read_excel(EXCEL_FILE)

    max_price = df_raw['Fiyat'].quantile(0.97)
    df = df_raw[
        (df_raw['m² (Brüt)'] <= 500) &
        (df_raw['m² (Brüt)'] >= 20) &
        (df_raw['Fiyat'] <= max_price)
    ].copy()

    def parse_features(row):
        title = str(row.get('İlan Başlığı', '')).upper()
        brut = row['m² (Brüt)']
        net = int(brut * 0.82)

        if any(x in title for x in ['SIFIR', 'PROJEDEN', '0 YAŞ', 'YENİ BİNA', 'SIFIR DAİRE']):
            yasi, yasi_num = '0 (Sıfır)', 0
        elif any(x in title for x in ['1 YAŞ', '2 YAŞ', '3 YAŞ']):
            yasi, yasi_num = '1-3 Yaş', 2
        elif any(x in title for x in ['4 YAŞ', '5 YAŞ']):
            yasi, yasi_num = '3-5 Yaş', 4
        elif any(x in title for x in ['6 YAŞ', '7 YAŞ', '8 YAŞ', '9 YAŞ', '10 YAŞ']):
            yasi, yasi_num = '5-10 Yaş', 8
        else:
            yasi, yasi_num = 'Belirtilmemiş / 5+ Yaş', 12

        if any(x in title for x in ['DUBLEKS', 'DUBLEX']):
            kat = 'Dubleks'
        elif any(x in title for x in ['BAHÇE', 'GİRİŞ', 'KOT']):
            kat = 'Giriş / Bahçe Katı'
        elif any(x in title for x in ['TERAS', 'EN ÜST', 'ÇATI']):
            kat = 'En Üst / Teras Kat'
        else:
            kat = 'Ara Kat'

        return pd.Series([net, yasi, yasi_num, kat])

    df[['m² (Net)', 'Bina Yaşı', 'Bina Yaşı Num', 'Bulunduğu Kat']] = df.apply(parse_features, axis=1)
    df['Oda Sayısı Num'] = df['Oda Sayısı'].astype(str).str.extract(r'(\d+)').astype(float).fillna(2)
    df['Semt'] = df['Semt / Mahalle'].astype(str)

    feature_cols = ['m² (Brüt)', 'm² (Net)', 'Oda Sayısı Num', 'Bina Yaşı Num', 'Semt']
    X = df[feature_cols].copy()
    y_log = np.log1p(df['Fiyat'])
    X['Semt'] = X['Semt'].astype(str)

    X_train, X_test, y_train, y_test = train_test_split(X, y_log, test_size=0.2, random_state=42)

    model = CatBoostRegressor(
        iterations=600,
        learning_rate=0.05,
        depth=6,
        cat_features=['Semt'],
        verbose=0,
        random_seed=42
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
    st.error(f"Veri yüklenirken hata oluştu: {e}")
    st.stop()

# Session state
if "ai_price" not in st.session_state:
    st.session_state.ai_price = 5_000_000.0
if "prediction_done" not in st.session_state:
    st.session_state.prediction_done = False
if "actual_price" not in st.session_state:
    st.session_state.actual_price = 0

# =========================================================
# 3. SIDEBAR FİLTRELEME
# =========================================================
st.sidebar.header("Ev Arama ve Filtreleme")

semt_options = ["Tüm Bölgeler"] + sorted(df['Semt / Mahalle'].astype(str).unique().tolist())
selected_semt = st.sidebar.selectbox("Semt / Mahalle", semt_options)

oda_options = ["Tüm Oda Tipleri"] + sorted(df['Oda Sayısı'].astype(str).unique().tolist())
selected_oda = st.sidebar.selectbox("Oda Sayısı", oda_options)

yas_options = ["Tüm Bina Yaşları", "0 (Sıfır)", "1-3 Yaş", "3-5 Yaş", "5-10 Yaş", "Belirtilmemiş / 5+ Yaş"]
selected_yas = st.sidebar.selectbox("Bina Yaşı", yas_options)

kat_options = ["Tüm Katlar", "Ara Kat", "Giriş / Bahçe Katı", "En Üst / Teras Kat", "Dubleks"]
selected_kat = st.sidebar.selectbox("Bulunduğu Kat", kat_options)

min_m2 = int(df['m² (Brüt)'].min())
max_m2 = int(df['m² (Brüt)'].max())
selected_m2 = st.sidebar.slider("Brüt m² Aralığı", min_m2, max_m2, (50, 200), step=5)

max_price = int(df['Fiyat'].max())
selected_budget = st.sidebar.number_input(
    "Maksimum Bütçe (TL)",
    min_value=1_000_000,
    max_value=max_price,
    value=min(15_000_000, max_price),
    step=500_000
)

# Filtre uygula
filtered = df.copy()
if selected_semt != "Tüm Bölgeler":
    filtered = filtered[filtered['Semt / Mahalle'] == selected_semt]
if selected_oda != "Tüm Oda Tipleri":
    filtered = filtered[filtered['Oda Sayısı'] == selected_oda]
if selected_yas != "Tüm Bina Yaşları":
    filtered = filtered[filtered['Bina Yaşı'] == selected_yas]
if selected_kat != "Tüm Katlar":
    filtered = filtered[filtered['Bulunduğu Kat'] == selected_kat]

filtered = filtered[
    (filtered['m² (Brüt)'] >= selected_m2[0]) &
    (filtered['m² (Brüt)'] <= selected_m2[1]) &
    (filtered['Fiyat'] <= selected_budget)
]

# =========================================================
# 4. TABLAR
# =========================================================
tab1, tab2, tab3, tab4 = st.tabs([
    "Yapay Zeka Tahmini & Fırsat Analizi",
    "Lokasyon Dağılımı",
    "Model Açıklanabilirliği",
    "Performans & Kredi Simülasyonu"
])

# -------------------- TAB 1 --------------------
with tab1:
    st.subheader(f"Filtreye Uygun İlanlar ({len(filtered)} adet)")

    if len(filtered) == 0:
        st.warning("Seçtiğiniz kriterlere uygun ilan bulunamadı.")
    else:
        show_cols = ['İlan Başlığı', 'Semt / Mahalle', 'Oda Sayısı', 'm² (Brüt)', 'Bina Yaşı', 'Bulunduğu Kat', 'Fiyat']
        st.dataframe(
            filtered[show_cols].head(25).style.format({'Fiyat': '{:,.0f} TL'}),
            use_container_width=True,
            height=230
        )

        st.divider()
        st.subheader("Listeden Ev Seçerek Analiz Et")

        selected_title = st.selectbox("İlan seçin:", filtered['İlan Başlığı'].tolist())

        if st.button("Seçili İlanı Analiz Et", type="primary"):
            house = filtered[filtered['İlan Başlığı'] == selected_title].iloc[0]

            input_data = pd.DataFrame([{
                'm² (Brüt)': house['m² (Brüt)'],
                'm² (Net)': house['m² (Net)'],
                'Oda Sayısı Num': house['Oda Sayısı Num'],
                'Bina Yaşı Num': house['Bina Yaşı Num'],
                'Semt': str(house['Semt / Mahalle'])
            }])

            pred = float(np.expm1(model.predict(input_data)[0]))
            st.session_state.ai_price = pred
            st.session_state.actual_price = house['Fiyat']
            st.session_state.prediction_done = True

        # Manuel giriş
        with st.expander("Manuel Parametre ile Değerleme"):
            c1, c2, c3 = st.columns(3)
            with c1:
                m_ilce = st.selectbox("İlçe", sorted(df['Semt / Mahalle'].astype(str).unique())[:40], key="m_ilce")
                m_net = st.number_input("Net m²", 30, 400, 95, key="m_net")
                m_brut = st.number_input("Brüt m²", 40, 500, 115, key="m_brut")
            with c2:
                m_oda = st.selectbox("Oda", ["1", "2", "3", "4", "5"], key="m_oda")
                m_yas = st.number_input("Bina Yaşı", 0, 40, 5, key="m_yas")
            with c3:
                m_site = st.checkbox("Site İçinde", True, key="m_site")
                m_asansor = st.checkbox("Asansör", True, key="m_asansor")
                m_balkon = st.checkbox("Balkon", True, key="m_balkon")

            if st.button("Manuel Değerleme Yap"):
                input_data = pd.DataFrame([{
                    'm² (Brüt)': m_brut,
                    'm² (Net)': m_net,
                    'Oda Sayısı Num': float(m_oda),
                    'Bina Yaşı Num': float(m_yas),
                    'Semt': m_ilce
                }])
                pred = float(np.expm1(model.predict(input_data)[0]))
                if m_site: pred *= 1.04
                if m_asansor: pred *= 1.02
                if m_balkon: pred *= 1.015

                st.session_state.ai_price = pred
                st.session_state.actual_price = 0
                st.session_state.prediction_done = True

    # Sonuç gösterimi
    if st.session_state.prediction_done:
        ai = st.session_state.ai_price
        alt = ai * 0.93
        ust = ai * 1.07

        st.success("Analiz tamamlandı")
        m1, m2, m3 = st.columns(3)
        m1.metric("Yapay Zeka Değeri", f"{ai:,.0f} TL")
        m2.metric("Hızlı Satış Bandı", f"{alt:,.0f} TL")
        m3.metric("Üst Bant", f"{ust:,.0f} TL")

        if st.session_state.actual_price > 0:
            actual = st.session_state.actual_price
            fark = ai - actual
            yuzde = (fark / ai) * 100
            if fark > 0:
                st.info(f"Bu ilan tahmini değere göre **%{yuzde:.1f} kelepir** görünüyor. (İlan Fiyatı: {actual:,.0f} TL)")
            else:
                st.warning(f"Bu ilan tahmini değere göre **%{abs(yuzde):.1f} pahalı** görünüyor. (İlan Fiyatı: {actual:,.0f} TL)")

# -------------------- TAB 2 --------------------
with tab2:
    st.subheader("İstanbul İlçeleri Lokasyon Dağılımı")

    coords = {
        'Ataşehir': (40.9833, 29.1167), 'Kadıköy': (40.9903, 29.0275), 'Üsküdar': (41.0244, 29.0050),
        'Beşiktaş': (41.0422, 29.0067), 'Şişli': (41.0600, 28.9870), 'Bakırköy': (40.9800, 28.8700),
        'Bağcılar': (41.0339, 28.8579), 'Bahçelievler': (41.0003, 28.8638), 'Küçükçekmece': (40.9917, 28.7719),
        'Avcılar': (40.9801, 28.7175), 'Esenyurt': (41.0342, 28.6801), 'Başakşehir': (41.0975, 28.8064),
        'Maltepe': (40.9333, 29.1333), 'Kartal': (40.8886, 29.1856), 'Pendik': (40.8750, 29.2333),
        'Ümraniye': (41.0256, 29.1244), 'Sancaktepe': (40.9900, 29.2300), 'Beylikdüzü': (40.9900, 28.6400)
    }

    map_df = df.copy()
    def get_lat_lon(row):
        for name, (lat, lon) in coords.items():
            if name in str(row['Semt / Mahalle']):
                return pd.Series([lat, lon])
        return pd.Series([41.0082, 28.9784])

    map_df[['lat', 'lon']] = map_df.apply(get_lat_lon, axis=1)
    np.random.seed(42)
    map_df['lat'] += np.random.normal(0, 0.006, len(map_df))
    map_df['lon'] += np.random.normal(0, 0.006, len(map_df))

    fig = px.scatter_mapbox(
        map_df.sample(min(700, len(map_df))),
        lat="lat", lon="lon",
        color="Fiyat", size="m² (Brüt)",
        hover_name="Semt / Mahalle",
        color_continuous_scale="Reds",
        size_max=12, zoom=9.5,
        center={"lat": 41.01, "lon": 28.98},
        mapbox_style="carto-darkmatter",
        height=520
    )
    fig.update_layout(margin=dict(l=0, r=0, t=0, b=0))
    st.plotly_chart(fig, use_container_width=True)

# -------------------- TAB 3 --------------------
with tab3:
    st.subheader("Model Açıklanabilirliği")
    st.caption(f"R² Skoru: **%{model_r2*100:.2f}**  |  MAE: {model_mae:,.0f} TL")

    try:
        importance = model.get_feature_importance()
        names = ['m² (Brüt)', 'm² (Net)', 'Oda Sayısı', 'Bina Yaşı', 'Semt / İlçe']
        f_df = pd.DataFrame({'Öznitelik': names, 'Önem': importance}).sort_values('Önem')

        fig = px.bar(
            f_df, x='Önem', y='Öznitelik', orientation='h',
            title="Konut Fiyatını En Çok Etkileyen Faktörler",
            color='Önem', color_continuous_scale='Blues'
        )
        fig.update_layout(template="plotly_dark", height=380)
        st.plotly_chart(fig, use_container_width=True)
    except Exception:
        st.info("Öznitelik önem grafiği şu an hesaplanamadı.")

# -------------------- TAB 4 --------------------
with tab4:
    st.subheader("Akademik Model Performans Karşılaştırması")

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
        "BDDK / Piyasa Ortalaması": 2.79,
        "Ziraat Bankası (Kamu)": 2.79,
        "VakıfBank (Kamu)": 2.79,
        "Halkbank (Kamu)": 2.79,
        "İş Bankası (Özel)": 2.85,
        "Akbank (Özel)": 2.89,
        "Garanti BBVA (Özel)": 3.05,
        "Özel Parametre Gir": 2.79
    }

    col1, col2 = st.columns(2)

    with col1:
        tutar = st.number_input("Konut Ekspertiz / Satış Tutarı (TL)", value=float(st.session_state.ai_price), step=100_000.0)
        banka = st.selectbox("Finansman Sağlayacak Kurum", list(bankalar.keys()))

        if banka == "Özel Parametre Gir":
            faiz = st.number_input("Aylık Akdi Faiz Oranı (%)", 0.1, 10.0, 2.79, 0.01)
        else:
            faiz = bankalar[banka]
            st.info(f"**{banka}**  |  Aylık Faiz: **%{faiz}**")

        pesinat = st.slider("Peşinat Oranı (%)", 10, 90, 20, 5)
        vade = st.selectbox("Vade Süresi (Ay)", [12, 24, 36, 48, 60, 84, 96, 120, 180, 240], index=7)

    with col2:
        pesinat_tutar = tutar * (pesinat / 100)
        kredi = tutar - pesinat_tutar

        if kredi > 0:
            r = faiz / 100
            taksit = (kredi * r * (1 + r)**vade) / ((1 + r)**vade - 1)
            toplam = taksit * vade
            faiz_yuku = toplam - kredi

            st.subheader("Finansman Detay Raporu")
            st.metric("Ödenecek Peşinat", f"{pesinat_tutar:,.0f} TL")
            st.metric("Kullanılacak Kredi", f"{kredi:,.0f} TL")
            st.metric("Aylık Taksit", f"{taksit:,.2f} TL")
            st.metric("Toplam Geri Ödeme", f"{toplam:,.2f} TL")
            st.caption(f"Toplam Faiz Yükü: **{faiz_yuku:,.2f} TL**")
        else:
            st.warning("Peşinat tutarı konut değerinden büyük olamaz.")
