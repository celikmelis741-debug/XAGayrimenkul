import streamlit as st
import pandas as pd
import numpy as np
from catboost import CatBoostRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error
import shap
import plotly.express as px
from datetime import datetime

# ---------------------------------------------------------
# SAYFA AYARLARI
# ---------------------------------------------------------
st.set_page_config(
    page_title="XAI - Akıllı Gayrimenkul Değerleme",
    page_icon="🏆",
    layout="wide",
    initial_sidebar_state="collapsed"
)

st.markdown("""
<style>
    .stApp { background-color: #0e1117; }
    .main-header { font-size: 2.1rem; font-weight: 700; color: #ffffff; margin-bottom: 0.2rem; }
    .sub-header { color: #a0a0a0; font-size: 0.95rem; margin-bottom: 1.2rem; }
</style>
""", unsafe_allow_html=True)

st.markdown('<p class="main-header">🏆 XAI Destekli Akıllı Gayrimenkul Değerleme Platformu</p>', unsafe_allow_html=True)
st.markdown('<p class="sub-header">CatBoost Regresör Mimarisi (%93.01 R² Başarımı) & Açıklanabilir Yapay Zekâ Karar Destek Sistemi</p>', unsafe_allow_html=True)

# ---------------------------------------------------------
# VERİ + MODEL
# ---------------------------------------------------------
EXCEL_FILE = "sahibinden_ilanlar_temiz_cloud.xlsx"

@st.cache_resource(show_spinner="Model yükleniyor...")
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
        if any(x in title for x in ['SIFIR', 'PROJEDEN', '0 YAŞ', 'YENİ BİNA']):
            yasi = 0
        elif any(x in title for x in ['1 YAŞ', '2 YAŞ', '3 YAŞ']):
            yasi = 2
        elif any(x in title for x in ['4 YAŞ', '5 YAŞ']):
            yasi = 4
        elif any(x in title for x in ['6 YAŞ', '7 YAŞ', '8 YAŞ', '9 YAŞ', '10 YAŞ']):
            yasi = 8
        else:
            yasi = 12
        return pd.Series([net, yasi])

    df[['m² (Net)', 'Bina Yaşı Num']] = df.apply(parse_features, axis=1)
    df['Oda Sayısı Num'] = df['Oda Sayısı'].astype(str).str.extract(r'(\d+)').astype(float).fillna(2)
    df['Semt'] = df['Semt / Mahalle'].astype(str)

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
    st.error(f"Hata: {e}")
    st.stop()

if "ai_price" not in st.session_state:
    st.session_state.ai_price = 5_000_000.0
if "prediction_done" not in st.session_state:
    st.session_state.prediction_done = False

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
    st.subheader("📋 Taşınmaz Parametre Girişi")

    col1, col2, col3 = st.columns([1.2, 1.2, 0.9])

    with col1:
        ilce = st.selectbox("İlçe Seçiniz", sorted(df['Semt / Mahalle'].astype(str).unique().tolist())[:50])
        net_m2 = st.number_input("Net m²", min_value=30, max_value=400, value=95, step=5)
        brut_m2 = st.number_input("Brüt m²", min_value=40, max_value=500, value=115, step=5)

    with col2:
        oda = st.selectbox("Oda Sayısı", ["1", "2", "3", "4", "5"])
        salon = st.selectbox("Salon Sayısı", ["1", "2"])
        bina_yasi = st.number_input("Bina Yaşı", min_value=0, max_value=40, value=5, step=1)

    with col3:
        st.write("")
        st.write("")
        site = st.checkbox("Site İçerisinde", value=True)
        asansor = st.checkbox("Asansör Var", value=True)
        balkon = st.checkbox("Balkon Var", value=True)

    ilan_fiyati = st.number_input("İlan / Satış Fiyatı (TL - Opsiyonel)", min_value=0, value=0, step=50000)

    if st.button("🚀 Piyasa Değerini Hesapla", type="primary", use_container_width=True):
        input_df = pd.DataFrame([{
            'm² (Brüt)': brut_m2,
            'm² (Net)': net_m2,
            'Oda Sayısı Num': float(oda),
            'Bina Yaşı Num': float(bina_yasi),
            'Semt': ilce
        }])
        pred_log = model.predict(input_df)[0]
        ai_price = float(np.expm1(pred_log))

        if site: ai_price *= 1.04
        if asansor: ai_price *= 1.02
        if balkon: ai_price *= 1.015

        st.session_state.ai_price = ai_price
        st.session_state.prediction_done = True
        st.session_state.input_data = {
            "ilce": ilce, "net": net_m2, "brut": brut_m2, "oda": oda,
            "salon": salon, "yas": bina_yasi, "site": site,
            "asansor": asansor, "balkon": balkon, "ilan": ilan_fiyati
        }

    if st.session_state.prediction_done:
        ai = st.session_state.ai_price
        alt, ust = ai * 0.93, ai * 1.07

        m1, m2, m3 = st.columns(3)
        m1.metric("Tahmini Piyasa Değeri", f"{ai:,.0f} TL")
        m2.metric("Hızlı Satış (Alt Bant)", f"{alt:,.0f} TL")
        m3.metric("Maksimum (Üst Bant)", f"{ust:,.0f} TL")

        if st.session_state.input_data["ilan"] > 0:
            fark = ai - st.session_state.input_data["ilan"]
            yuzde = (fark / ai) * 100
            if fark > 0:
                st.success(f"🔥 Bu ilan tahmini değere göre **%{yuzde:.1f} kelepir**.")
            else:
                st.warning(f"⚠️ Bu ilan tahmini değere göre **%{abs(yuzde):.1f} pahalı**.")

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

# ==================== TAB 4 - RESMİ KREDİ KISMI ====================
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

    # Resmi banka parametreleri (önceki kodundaki gibi)
    banka_parametreleri = {
        "BDDK / Piyasa Ortalaması": {"faiz": 2.79, "max_oransal": 0.80},
        "Ziraat Bankası (Kamu)": {"faiz": 2.79, "max_oransal": 0.80},
        "VakıfBank (Kamu)": {"faiz": 2.79, "max_oransal": 0.80},
        "Halkbank (Kamu)": {"faiz": 2.79, "max_oransal": 0.80},
        "İş Bankası (Özel)": {"faiz": 2.85, "max_oransal": 0.75},
        "Akbank (Özel)": {"faiz": 2.89, "max_oransal": 0.75},
        "Garanti BBVA (Özel)": {"faiz": 3.05, "max_oransal": 0.70},
        "Özel Parametre Gir": {"faiz": 2.79, "max_oransal": 0.80}
    }

    col_k1, col_k2 = st.columns(2)

    with col_k1:
        hesaplama_tutari = st.number_input(
            "Konut Ekspertiz / Satış Tutarı (TL)",
            value=float(st.session_state.ai_price),
            step=100_000.0
        )
        secilen_banka = st.selectbox("Finansman Sağlayacak Kurum", list(banka_parametreleri.keys()))

        if secilen_banka == "Özel Parametre Gir":
            aylik_faiz = st.number_input("Aylık Akdi Faiz Oranı (%)", min_value=0.1, max_value=10.0, value=2.79, step=0.01)
        else:
            aylik_faiz = banka_parametreleri[secilen_banka]["faiz"]
            st.info(f"Seçilen Kurum: **{secilen_banka}**  \nAylık Faiz: **%{aylik_faiz}**")

        pesinat_orani = st.slider("Peşinat Oranı (%)", min_value=10, max_value=90, value=20, step=5)
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
            st.metric("Ödenecek Peşinat Tutarı", f"{pesinat_tl:,.0f} TL")
            st.metric("Kullanılacak Kredi Tutarı", f"{kredi_tutari:,.0f} TL")
            st.metric("Aylık Taksit Ödemesi", f"{aylik_taksit:,.2f} TL")
            st.metric("Toplam Geri Ödeme Tutarı", f"{toplam_odeme:,.2f} TL")
            st.caption(f"Toplam Ödenecek Faiz Yükü: **{toplam_faiz:,.2f} TL**")
        else:
            st.warning("Peşinat tutarı konut değerine eşit veya büyük olamaz.")
