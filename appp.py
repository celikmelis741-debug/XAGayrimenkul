import streamlit as st
import pandas as pd
import numpy as np
from catboost import CatBoostRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error
import plotly.express as px
from datetime import datetime

# ---------------------------------------------------------
# SAYFA AYARLARI
# ---------------------------------------------------------
st.set_page_config(
    page_title="XAI - Akıllı Gayrimenkul Değerleme",
    page_icon="🏆",
    layout="wide",
    initial_sidebar_state="expanded"
)

st.markdown("""
<style>
    .stApp { background-color: #0e1117; }
    .main-header { font-size: 2.0rem; font-weight: 700; color: #ffffff; margin-bottom: 0.2rem; }
    .sub-header { color: #a0a0a0; font-size: 0.95rem; margin-bottom: 1rem; }
</style>
""", unsafe_allow_html=True)

st.markdown('<p class="main-header">🏆 XAI Destekli Akıllı Gayrimenkul Değerleme Platformu</p>', unsafe_allow_html=True)
st.markdown('<p class="sub-header">CatBoost Regresör (%93 R²) + Filtreleme + Açıklanabilir Yapay Zekâ</p>', unsafe_allow_html=True)

# ---------------------------------------------------------
# VERİ + MODEL YÜKLEME
# ---------------------------------------------------------
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
            yasi = '0 (Sıfır)'
            yasi_num = 0
        elif any(x in title for x in ['1 YAŞ', '2 YAŞ', '3 YAŞ']):
            yasi = '1-3 Yaş'
            yasi_num = 2
        elif any(x in title for x in ['4 YAŞ', '5 YAŞ']):
            yasi = '3-5 Yaş'
            yasi_num = 4
        elif any(x in title for x in ['6 YAŞ', '7 YAŞ', '8 YAŞ', '9 YAŞ', '10 YAŞ']):
            yasi = '5-10 Yaş'
            yasi_num = 8
        else:
            yasi = 'Belirtilmemiş / 5+ Yaş'
            yasi_num = 12

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

    # Model için
    feature_cols = ['m² (Brüt)', 'm² (Net)', 'Oda Sayısı Num', 'Bina Yaşı Num', 'Semt']
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

# Session state
if "ai_price" not in st.session_state:
    st.session_state.ai_price = 5_000_000.0
if "prediction_done" not in st.session_state:
    st.session_state.prediction_done = False

# ---------------------------------------------------------
# SIDEBAR - FİLTRELEME (İlk kodundan)
# ---------------------------------------------------------
st.sidebar.header("🔍 Ev Arama ve Filtreleme")

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
selected_m2_range = st.sidebar.slider("Brüt m² Aralığı", min_m2, max_m2, (50, 200), step=5)

max_price_val = int(df['Fiyat'].max())
selected_price_limit = st.sidebar.number_input(
    "Maksimum Bütçe (TL)", 
    min_value=1_000_000, 
    max_value=max_price_val, 
    value=min(15_000_000, max_price_val), 
    step=500_000
)

# Filtreleme
filtered_df = df.copy()
if selected_semt != "Tüm Bölgeler":
    filtered_df = filtered_df[filtered_df['Semt / Mahalle'] == selected_semt]
if selected_oda != "Tüm Oda Tipleri":
    filtered_df = filtered_df[filtered_df['Oda Sayısı'] == selected_oda]
if selected_yas != "Tüm Bina Yaşları":
    filtered_df = filtered_df[filtered_df['Bina Yaşı'] == selected_yas]
if selected_kat != "Tüm Katlar":
    filtered_df = filtered_df[filtered_df['Bulunduğu Kat'] == selected_kat]

filtered_df = filtered_df[
    (filtered_df['m² (Brüt)'] >= selected_m2_range[0]) &
    (filtered_df['m² (Brüt)'] <= selected_m2_range[1]) &
    (filtered_df['Fiyat'] <= selected_price_limit)
]

# ---------------------------------------------------------
# TABS
# ---------------------------------------------------------
tab1, tab2, tab3, tab4 = st.tabs([
    "🎯 Yapay Zekâ Tahmini & Fırsat Analizi",
    "🗺️ İnteraktif Fiyat Isı Haritası",
    "🔬 XAI ve Model Açıklanabilirliği",
    "📊 Akademik Benchmark ve Kredi Simülasyonu"
])

# ==================== TAB 1 ====================
with tab1:
    st.subheader(f"📋 Filtreye Uygun İlanlar ({len(filtered_df)} adet)")

    if len(filtered_df) == 0:
        st.warning("Seçtiğiniz kriterlere uygun ev bulunamadı. Filtreleri gevşetmeyi deneyin.")
    else:
        display_cols = ['İlan Başlığı', 'Semt / Mahalle', 'Oda Sayısı', 'm² (Brüt)', 'Bina Yaşı', 'Bulunduğu Kat', 'Fiyat']
        st.dataframe(
            filtered_df[display_cols].head(30).style.format({'Fiyat': '{:,.0f} TL'}),
            use_container_width=True,
            height=220
        )

        st.divider()
        st.subheader("💡 Listeden Bir Ev Seçin veya Manuel Giriş Yapın")

        # --- Seçenek 1: Listeden seç ---
        selected_title = st.selectbox("İncelemek İstediğiniz İlanı Seçin:", filtered_df['İlan Başlığı'].tolist())

        if st.button("Seçili İlanı Analiz Et", type="primary"):
            selected_house = filtered_df[filtered_df['İlan Başlığı'] == selected_title].iloc[0]

            house_input = pd.DataFrame([{
                'm² (Brüt)': selected_house['m² (Brüt)'],
                'm² (Net)': selected_house['m² (Net)'],
                'Oda Sayısı Num': selected_house['Oda Sayısı Num'],
                'Bina Yaşı Num': selected_house['Bina Yaşı Num'],
                'Semt': str(selected_house['Semt / Mahalle'])
            }])

            pred_log = model.predict(house_input)[0]
            ai_price = float(np.expm1(pred_log))
            actual_price = selected_house['Fiyat']

            st.session_state.ai_price = ai_price
            st.session_state.prediction_done = True
            st.session_state.selected_house = selected_house
            st.session_state.actual_price = actual_price

        # --- Seçenek 2: Manuel form ---
        with st.expander("🔧 Manuel Parametre ile Değerleme Yap"):
            c1, c2, c3 = st.columns(3)
            with c1:
                man_ilce = st.selectbox("İlçe", sorted(df['Semt / Mahalle'].astype(str).unique().tolist())[:40], key="man_ilce")
                man_net = st.number_input("Net m²", 30, 400, 95, key="man_net")
                man_brut = st.number_input("Brüt m²", 40, 500, 115, key="man_brut")
            with c2:
                man_oda = st.selectbox("Oda", ["1","2","3","4","5"], key="man_oda")
                man_salon = st.selectbox("Salon", ["1","2"], key="man_salon")
                man_yas = st.number_input("Bina Yaşı", 0, 40, 5, key="man_yas")
            with c3:
                man_site = st.checkbox("Site İçinde", True, key="man_site")
                man_asansor = st.checkbox("Asansör", True, key="man_asansor")
                man_balkon = st.checkbox("Balkon", True, key="man_balkon")

            if st.button("Manuel Değerleme Yap"):
                input_df = pd.DataFrame([{
                    'm² (Brüt)': man_brut,
                    'm² (Net)': man_net,
                    'Oda Sayısı Num': float(man_oda),
                    'Bina Yaşı Num': float(man_yas),
                    'Semt': man_ilce
                }])
                pred_log = model.predict(input_df)[0]
                ai_price = float(np.expm1(pred_log))
                if man_site: ai_price *= 1.04
                if man_asansor: ai_price *= 1.02
                if man_balkon: ai_price *= 1.015

                st.session_state.ai_price = ai_price
                st.session_state.prediction_done = True
                st.session_state.selected_house = None
                st.session_state.actual_price = 0

    # Sonuç gösterimi
    if st.session_state.prediction_done:
        ai = st.session_state.ai_price
        alt, ust = ai * 0.93, ai * 1.07

        st.success("Analiz tamamlandı")
        m1, m2, m3 = st.columns(3)
        m1.metric("Yapay Zeka Reel Değeri", f"{ai:,.0f} TL")
        m2.metric("Hızlı Satış Bandı", f"{alt:,.0f} TL")
        m3.metric("Üst Bant", f"{ust:,.0f} TL")

        if st.session_state.get("actual_price", 0) > 0:
            actual = st.session_state.actual_price
            fark = ai - actual
            yuzde = (fark / ai) * 100
            if fark > 0:
                st.info(f"🔥 Bu ilan tahmini değere göre **%{yuzde:.1f} kelepir** görünüyor. (İlan: {actual:,.0f} TL)")
            else:
                st.warning(f"⚠️ Bu ilan tahmini değere göre **%{abs(yuzde):.1f} pahalı** görünüyor. (İlan: {actual:,.0f} TL)")

# ==================== TAB 2 ====================
with tab2:
    st.subheader("🗺️ İstanbul İlçeleri Lokasyon Dağılımı")

    district_coords = {
        'Ataşehir': (40.9833, 29.1167), 'Kadıköy': (40.9903, 29.0275), 'Üsküdar': (41.0244, 29.0050),
        'Beşiktaş': (41.0422, 29.0067), 'Şişli': (41.0600, 28.9870), 'Bakırköy': (40.9800, 28.8700),
        'Bağcılar': (41.0339, 28.8579), 'Bahçelievler': (41.0003, 28.8638), 'Küçükçekmece': (40.9917, 28.7719),
        'Avcılar': (40.9801, 28.7175), 'Esenyurt': (41.0342, 28.6801), 'Başakşehir': (41.0975, 28.8064),
        'Maltepe': (40.9333, 29.1333), 'Kartal': (40.8886, 29.1856), 'Pendik': (40.8750, 29.2333),
        'Ümraniye': (41.0256, 29.1244), 'Sancaktepe': (40.9900, 29.2300), 'Beylikdüzü': (40.9900, 28.6400)
    }

    map_df = df.copy()
    def get_coords(row):
        for d, c in district_coords.items():
            if d in str(row['Semt / Mahalle']):
                return pd.Series(c)
        return pd.Series([41.0082, 28.9784])

    map_df[['lat', 'lon']] = map_df.apply(get_coords, axis=1)
    np.random.seed(42)
    map_df['lat'] += np.random.normal(0, 0.006, len(map_df))
    map_df['lon'] += np.random.normal(0, 0.006, len(map_df))

    fig = px.scatter_mapbox(
        map_df.sample(min(700, len(map_df))),
        lat="lat", lon="lon", color="Fiyat", size="m² (Brüt)",
        hover_name="Semt / Mahalle", color_continuous_scale="Reds",
        size_max=12, zoom=9.5, center={"lat": 41.01, "lon": 28.98},
        mapbox_style="carto-darkmatter", height=520
    )
    fig.update_layout(margin=dict(l=0, r=0, t=0, b=0))
    st.plotly_chart(fig, use_container_width=True)

# ==================== TAB 3 ====================
with tab3:
    st.subheader("🔬 Açıklanabilir Yapay Zekâ (XAI)")
    st.caption(f"Model R²: **%{model_r2*100:.2f}** | MAE: {model_mae:,.0f} TL")

    try:
        importance = model.get_feature_importance()
        feat_names = ['m² (Brüt)', 'm² (Net)', 'Oda Sayısı', 'Bina Yaşı', 'Semt / İlçe']
        f_df = pd.DataFrame({'Öznitelik': feat_names, 'Önem': importance}).sort_values('Önem')
        fig_imp = px.bar(f_df, x='Önem', y='Öznitelik', orientation='h',
                         title="Konut Fiyatını En Çok Etkileyen Faktörler",
                         color='Önem', color_continuous_scale='Blues')
        fig_imp.update_layout(template="plotly_dark", height=380)
        st.plotly_chart(fig_imp, use_container_width=True)
    except:
        st.info("Öznitelik önem grafiği hesaplanamadı.")

# ==================== TAB 4 ====================
with tab4:
    st.subheader("📊 Akademik Model Performans Karşılaştırması")
    benchmark = pd.DataFrame([
        {"Model": "CatBoost Regressor (Bu Model)", "R2_Score": f"%{model_r2*100:.2f}", "MAE_TL": f"{model_mae:,.0f} TL", "RMSE_TL": f"{model_rmse:,.0f} TL"},
        {"Model": "LightGBM Regressor", "R2_Score": "%93.00", "MAE_TL": "806,166 TL", "RMSE_TL": "1,017,600 TL"},
        {"Model": "Gradient Boosting", "R2_Score": "%92.97", "MAE_TL": "834,818 TL", "RMSE_TL": "1,019,746 TL"},
        {"Model": "Random Forest", "R2_Score": "%92.28", "MAE_TL": "876,419 TL", "RMSE_TL": "1,068,710 TL"},
    ])
    st.dataframe(benchmark, use_container_width=True, hide_index=True)

    st.divider()
    st.subheader("🏦 Resmi Banka Konut Kredisi Simülasyonu")

    banka_parametreleri = {
        "BDDK / Piyasa Ortalaması": {"faiz": 2.79},
        "Ziraat Bankası (Kamu)": {"faiz": 2.79},
        "VakıfBank (Kamu)": {"faiz": 2.79},
        "Halkbank (Kamu)": {"faiz": 2.79},
        "İş Bankası (Özel)": {"faiz": 2.85},
        "Akbank (Özel)": {"faiz": 2.89},
        "Garanti BBVA (Özel)": {"faiz": 3.05},
        "Özel Parametre Gir": {"faiz": 2.79}
    }

    col_k1, col_k2 = st.columns(2)
    with col_k1:
        hesaplama_tutari = st.number_input("Konut Ekspertiz / Satış Tutarı (TL)", value=float(st.session_state.ai_price), step=100000.0)
        secilen_banka = st.selectbox("Finansman Sağlayacak Kurum", list(banka_parametreleri.keys()))
        if secilen_banka == "Özel Parametre Gir":
            aylik_faiz = st.number_input("Aylık Akdi Faiz Oranı (%)", 0.1, 10.0, 2.79, 0.01)
        else:
            aylik_faiz = banka_parametreleri[secilen_banka]["faiz"]
            st.info(f"**{secilen_banka}** | Aylık Faiz: **%{aylik_faiz}**")
        pesinat_orani = st.slider("Peşinat Oranı (%)", 10, 90, 20, 5)
        vade_ay = st.selectbox("Vade Süresi (Ay)", [12, 24, 36, 48, 60, 84, 96, 120, 180, 240], index=7)

    with col_k2:
        pesinat_tl = hesaplama_tutari * (pesinat_orani / 100)
        kredi_tutari = hesaplama_tutari - pesinat_tl
        if kredi_tutari > 0:
            r = aylik_faiz / 100
            aylik_taksit = (kredi_tutari * r * ((1 + r)**vade_ay)) / (((1 + r)**vade_ay) - 1)
            toplam_odeme = aylik_taksit * vade_ay
            toplam_faiz = toplam_odeme - kredi_tutari

            st.subheader("💳 Finansman Detay Raporu")
            st.metric("Ödenecek Peşinat", f"{pesinat_tl:,.0f} TL")
            st.metric("Kullanılacak Kredi", f"{kredi_tutari:,.0f} TL")
            st.metric("Aylık Taksit", f"{aylik_taksit:,.2f} TL")
            st.metric("Toplam Geri Ödeme", f"{toplam_odeme:,.2f} TL")
            st.caption(f"Toplam Faiz Yükü: **{toplam_faiz:,.2f} TL**")
        else:
            st.warning("Peşinat, konut değerinden büyük olamaz.")
