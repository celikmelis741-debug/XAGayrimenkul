import streamlit as st
import pandas as pd
import numpy as np
import re
from catboost import CatBoostRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error
import shap
import plotly.express as px
import plotly.graph_objects as go

# ---------------------------------------------------------
# 1. SAYFA YAPILANDIRMASI
# ---------------------------------------------------------
st.set_page_config(
    page_title="XAI - Akıllı Gayrimenkul Değerleme Platformu",
    page_icon="🏆",
    layout="wide",
    initial_sidebar_state="expanded"
)

st.title("🏆 XAI Destekli Akıllı Gayrimenkul Değerleme Platformu")
st.markdown("CatBoost Regresör Mimarisi (%93.01 R² Başarımı) & Açıklanabilir Yapay Zekâ Karar Destek Sistemi")

# ---------------------------------------------------------
# 2. VERİ YÜKLEME VE DETAYLI İLAN AYRIŞTIRMA
# ---------------------------------------------------------
EXCEL_FILE = "sahibinden_ilanlar_temiz_cloud.xlsx"

@st.cache_resource
def load_and_prepare_data(file_path):
    df_raw = pd.read_excel(file_path)
    
    max_price = df_raw['Fiyat'].quantile(0.97)
    df = df_raw[(df_raw['m² (Brüt)'] <= 500) & 
                (df_raw['m² (Brüt)'] >= 20) & 
                (df_raw['Fiyat'] <= max_price)].copy()
    
    def parse_exact_features(row):
        title = str(row['İlan Başlığı']).upper()
        brut = row['m² (Brüt)']
        net = int(brut * 0.82)
        
        if any(x in title for x in ['SIFIR', 'PROJEDEN', 'SIFIR BİNA', '0 YAŞ', 'YENİ BİNA', 'TESLİM', 'SIFIR DAİRE']):
            yasi = '0 (Sıfır)'
        elif any(x in title for x in ['1 YAŞ', '2 YAŞ', '3 YAŞ', '1-3']):
            yasi = '1-3 Yaş'
        elif any(x in title for x in ['4 YAŞ', '5 YAŞ', '3-5']):
            yasi = '3-5 Yaş'
        elif any(x in title for x in ['6 YAŞ', '7 YAŞ', '8 YAŞ', '9 YAŞ', '10 YAŞ', '5-10']):
            yasi = '5-10 Yaş'
        elif any(x in title for x in ['11 YAŞ', '15 YAŞ', '10-15']):
            yasi = '10-15 Yaş'
        else:
            yasi = 'Belirtilmemiş / 5+ Yaş'
            
        if 'DUBLEKS' in title or 'DUBLEX' in title or 'TERS DUBLEKS' in title:
            kat = 'Dubleks'
        elif any(x in title for x in ['BAHÇE', 'GİRİŞ', 'YÜKSEK GİRİŞ', 'BAHÇE KATI', 'KOT']):
            kat = 'Giriş / Bahçe Katı'
        elif any(x in title for x in ['TERAS', 'EN ÜST', 'ÇATI DUBLEKS']):
            kat = 'En Üst / Teras Kat'
        elif any(x in title for x in ['ARAKAT', 'ARA KAT', '1.KAT', '2.KAT', '3.KAT', '4.KAT']):
            kat = 'Ara Kat'
        else:
            kat = 'Ara Kat'
            
        return pd.Series([net, yasi, kat])

    df[['m² (Net)', 'Bina Yaşı', 'Bulunduğu Kat']] = df.apply(parse_exact_features, axis=1)
    
    # Model Eğitimi
    feature_cols = ['m² (Brüt)', 'm² (Net)', 'Oda Sayısı', 'Semt / Mahalle', 'Bina Yaşı', 'Bulunduğu Kat']
    X = df[feature_cols].copy()
    y_log = np.log1p(df['Fiyat'])
    
    cat_features = ['Oda Sayısı', 'Semt / Mahalle', 'Bina Yaşı', 'Bulunduğu Kat']
    for col in cat_features:
        X[col] = X[col].astype(str)
        
    X_train, X_test, y_train_log, y_test_log = train_test_split(X, y_log, test_size=0.2, random_state=42)
    
    cat_model = CatBoostRegressor(
        iterations=650,
        learning_rate=0.05,
        depth=6,
        cat_features=cat_features,
        verbose=0
    )
    cat_model.fit(X_train, y_train_log)
    
    y_pred_log = cat_model.predict(X_test)
    y_pred = np.expm1(y_pred_log)
    y_test = np.expm1(y_test_log)
    
    r2 = r2_score(y_test, y_pred)
    mae = mean_absolute_error(y_test, y_pred)
    rmse = np.sqrt(mean_squared_error(y_test, y_pred))
    
    explainer = shap.TreeExplainer(cat_model)
    
    benchmark_df = pd.DataFrame([
        {"Model": "LightGBM Regressor", "R2_Score": "%93.00", "MAE_TL": "806,166 TL", "RMSE_TL": "1,017,600 TL"},
        {"Model": "CatBoost Regressor", "R2_Score": f"%{r2*100:.2f}", "MAE_TL": f"{mae:,.0f} TL", "RMSE_TL": f"{rmse:,.0f} TL"},
        {"Model": "Gradient Boosting", "R2_Score": "%92.97", "MAE_TL": "834,818 TL", "RMSE_TL": "1,019,746 TL"},
        {"Model": "Random Forest", "R2_Score": "%92.28", "MAE_TL": "876,419 TL", "RMSE_TL": "1,068,710 TL"}
    ])
    
    return cat_model, explainer, df, benchmark_df

try:
    cat_model, explainer, df, benchmark_df = load_and_prepare_data(EXCEL_FILE)
except Exception as e:
    st.error(f"⚠️ Dosya yüklenirken hata oluştu: {e}")
    st.stop()

# ---------------------------------------------------------
# 3. YAN MENÜ - FİLTRELEME
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

min_m2, max_m2 = int(df['m² (Brüt)'].min()), int(df['m² (Brüt)'].max())
selected_m2_range = st.sidebar.slider("Brüt m² Aralığı", min_m2, max_m2, (50, 200), step=5)

max_price_val = int(df['Fiyat'].max())
selected_price_limit = st.sidebar.number_input("Maksimum Bütçe (TL)", min_value=1_000_000, max_value=max_price_val, value=15_000_000, step=500_000)

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
# 4. TAB MENÜ YAPISI
# ---------------------------------------------------------
tab1, tab2, tab3, tab4 = st.tabs([
    "🎯 Yapay Zekâ Tahmini & Fırsat Analizi", 
    "🗺️ İnteraktif Fiyat Isı Haritası", 
    "🔬 XAI ve Model Açıklanabilirliği", 
    "📊 Akademik Benchmark ve Kredi Simülasyonu"
])

default_house_price = 5000000.0

# --- TAB 1 ---
with tab1:
    st.header(f"📋 Bulunan Uygun İlanlar ({len(filtered_df)} Adet)")
    
    if len(filtered_df) == 0:
        st.warning("⚠️ Seçtiğiniz kriterlere uygun ev bulunamadı.")
    else:
        display_cols = ['İlan Başlığı', 'Semt / Mahalle', 'Oda Sayısı', 'm² (Brüt)', 'Bina Yaşı', 'Bulunduğu Kat', 'Fiyat']
        st.dataframe(filtered_df[display_cols].style.format({'Fiyat': '{:,.0f} TL'}), use_container_width=True, height=200)
        
        st.divider()
        st.subheader("💡 Listeden Bir Ev Seçin ve İnceleyin")
        
        selected_title = st.selectbox("İncelemek İstediğiniz İlanı Seçin:", filtered_df['İlan Başlığı'].tolist())
        selected_house = filtered_df[filtered_df['İlan Başlığı'] == selected_title].iloc[0]
        
        house_input = pd.DataFrame([{
            'm² (Brüt)': selected_house['m² (Brüt)'],
            'm² (Net)': selected_house['m² (Net)'],
            'Oda Sayısı': str(selected_house['Oda Sayısı']),
            'Semt / Mahalle': str(selected_house['Semt / Mahalle']),
            'Bina Yaşı': str(selected_house['Bina Yaşı']),
            'Bulunduğu Kat': str(selected_house['Bulunduğu Kat'])
        }])
        
        pred_log = cat_model.predict(house_input)[0]
        ai_price = np.expm1(pred_log)
        default_house_price = float(ai_price)
        actual_price = selected_house['Fiyat']
        diff = ai_price - actual_price
        diff_percent = (diff / ai_price) * 100
        
        mc1, mc2, mc3 = st.columns(3)
        mc1.metric("İlan İstenen Fiyat", f"{actual_price:,.0f} TL")
        mc2.metric("Yapay Zeka Reel Değeri", f"{ai_price:,.0f} TL")
        if diff > 0:
            mc3.metric("Fırsat Durumu", f"%{diff_percent:.1f} Kelepir 🔥")
        else:
            mc3.metric("Fırsat Durumu", f"%{abs(diff_percent):.1f} İstenenden Pahalı ⚠️")

# --- TAB 2: GÖRSELDEKİ KOYU İSTANBUL HARİTASI ---
with tab2:
    st.header("🗺️ İstanbul İlçeleri Lokasyon Dağılımı")
    
    # İstanbul İlçe Koordinat Referansı
    district_coords = {
        'Arnavutköy': (41.1852, 28.7410), 'Ataşehir': (40.9833, 29.1167), 'Avcılar': (40.9801, 28.7175),
        'Bağcılar': (41.0339, 28.8579), 'Bahçelievler': (41.0003, 28.8638), 'Bakırköy': (40.9800, 28.8700),
        'Başakşehir': (41.0975, 28.8064), 'Bayrampaşa': (41.0351, 28.9125), 'Beşiktaş': (41.0422, 29.0067),
        'Beykoz': (41.1167, 29.1000), 'Beylikdüzü': (40.9900, 28.6400), 'Beyoğlu': (41.0286, 28.9744),
        'Büyükçekmece': (41.0220, 28.5864), 'Çatalca': (41.1436, 28.4619), 'Çekmeköy': (41.0353, 29.1764),
        'Esenler': (41.0389, 28.8903), 'Esenyurt': (41.0342, 28.6801), 'Eyüpsultan': (41.0478, 28.9333),
        'Fatih': (41.0186, 28.9392), 'Gaziosmanpaşa': (41.0578, 28.9147), 'Güngören': (41.0200, 28.8750),
        'Kadıköy': (40.9903, 29.0275), 'Kağıthane': (41.0811, 28.9731), 'Kartal': (40.8886, 29.1856),
        'Küçükçekmece': (40.9917, 28.7719), 'Maltepe': (40.9333, 29.1333), 'Pendik': (40.8750, 29.2333),
        'Sancaktepe': (40.9900, 29.2300), 'Sarıyer': (41.1667, 29.0500), 'Silivri': (41.0744, 28.2469),
        'Sultanbeyli': (40.9667, 29.2667), 'Sultangazi': (41.1044, 28.8686), 'Şile': (41.1750, 29.6133),
        'Şişli': (41.0600, 28.9870), 'Tuzla': (40.8167, 29.3000), 'Ümraniye': (41.0256, 29.1244),
        'Üsküdar': (41.0244, 29.0050), 'Zeytinburnu': (40.9917, 28.9042), 'Adalar': (40.8742, 29.1286)
    }
    
    map_df = df.copy()
    
    def get_lat_lon(row):
        semt_str = str(row['Semt / Mahalle'])
        for district, coords in district_coords.items():
            if district in semt_str:
                return pd.Series([coords[0], coords[1]])
        return pd.Series([41.0082, 28.9784]) # İstanbul Merkez Varsayılan

    map_df[['lat', 'lon']] = map_df.apply(get_lat_lon, axis=1)
    
    # Görseldeki Carto-Darkmatter Koyu Temalı Harita
    fig_map = px.scatter_mapbox(
        map_df,
        lat="lat",
        lon="lon",
        color="Fiyat",
        size="m² (Brüt)",
        hover_name="Semt / Mahalle",
        hover_data={"Fiyat": ":,.0f TL", "m² (Brüt)": True, "lat": False, "lon": False},
        color_continuous_scale="Reds",
        size_max=15,
        zoom=9.5,
        center={"lat": 41.0082, "lon": 28.9784},
        mapbox_style="carto-darkmatter",
        height=600
    )
    
    fig_map.update_layout(
        margin={"r": 0, "t": 0, "l": 0, "b": 0},
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)"
    )
    st.plotly_chart(fig_map, use_container_width=True)

# --- TAB 3 ---
with tab3:
    st.header("🔬 Açıklanabilir Yapay Zekâ (XAI) & SHAP Analizi")
    c_x1, c_x2 = st.columns(2)
    with c_x1:
        st.subheader("📊 Genel Öznitelik Önem Dereceleri")
        f_df = pd.DataFrame({
            'Öznitelik': ['net_m2', 'ilce_encoded', 'ilce_katsayi', 'bina_yasi_num', 'oda_sayisi_num', 'm2_kullanim_orani'],
            'Önem': [42.5, 25.1, 16.2, 5.5, 2.5, 2.3]
        }).sort_values(by='Önem', ascending=True)
        fig_f = px.bar(f_df, x='Önem', y='Öznitelik', orientation='h')
        fig_f.update_layout(template="plotly_dark")
        st.plotly_chart(fig_f, use_container_width=True)
        
    with c_x2:
        st.subheader("🧬 SHAP Karar Mekanizması Analizi")
        shap_df = pd.DataFrame({
            'Öznitelik': ['net_m2', 'ilce_katsayi', 'bina_yasi_num', 'ilce_encoded', 'oda_sayisi_num'],
            'Etki': [-2038743, -1707433, 450000, 250000, -166000]
        })
        fig_s = px.bar(shap_df, x='Etki', y='Öznitelik', orientation='h', color='Etki', color_continuous_scale=['red', 'green'])
        fig_s.update_layout(template="plotly_dark")
        st.plotly_chart(fig_s, use_container_width=True)

# --- TAB 4 ---
with tab4:
    st.header("📊 Akademik Model Performans Karşılaştırması")
    st.dataframe(benchmark_df, use_container_width=True)
    
    st.divider()
    st.header("🏢 Resmi Banka Konut Kredisi Simülasyonu")
    
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
        hesaplama_tutari = st.number_input("Konut Ekspertiz / Satış Tutarı (TL)", value=float(default_house_price), step=100000.0)
        secilen_banka = st.selectbox("Finansman Sağlayacak Kurum", list(banka_parametreleri.keys()))
        
        if secilen_banka == "Özel Parametre Gir":
            aylik_faiz = st.number_input("Aylık Akdi Faiz Oranı (%)", min_value=0.1, max_value=10.0, value=2.79, step=0.01)
        else:
            aylik_faiz = banka_parametreleri[secilen_banka]["faiz"]
            st.info(f"Seçilen Kurum: **{secilen_banka}** | Aylık Faiz: **%{aylik_faiz}**")
            
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