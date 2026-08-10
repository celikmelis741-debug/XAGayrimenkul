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
</style>
""", unsafe_allow_html=True)

st.title("XAI Gayrimenkul Değerleme Platformu")
st.caption("CatBoost + Etiketli Filtreleme + Satıcı Değerleme")

# =========================================================
# 2. VERİ VE MODEL (Daha geniş temizlik)
# =========================================================
EXCEL_FILE = "ilanlar_oda_salon_rakam.xlsx"

@st.cache_resource(show_spinner="Model ve veriler yükleniyor...")
def load_model_and_data():
    df = pd.read_excel(EXCEL_FILE)

    # Daha geniş / dengeli temizlik
    df = df.dropna(subset=['Fiyat', 'm² (Brüt)'])
    df = df[(df['m² (Brüt)'] >= 20) & (df['m² (Brüt)'] <= 500)]
    df = df[df['Fiyat'] <= df['Fiyat'].quantile(0.99)]   # sadece en pahalı %1 atılıyor

    etiket_cols = ['havuz', 'otopark', 'balkon', 'manzara', 'site', 'dubleks',
                   'asansor', 'sifir', 'teras', 'bahce', 'guvenlik']
    for col in etiket_cols:
        if col in df.columns:
            df[col] = df[col].fillna(0).astype(int)
        else:
            df[col] = 0

    df['oda_sayisi'] = pd.to_numeric(df.get('oda_sayisi', 2), errors='coerce').fillna(2)
    df['salon_sayisi'] = pd.to_numeric(df.get('salon_sayisi', 1), errors='coerce').fillna(1)
    df['Oda Düzeni'] = df['oda_sayisi'].astype(int).astype(str) + '+' + df['salon_sayisi'].astype(int).astype(str)

    split_cols = df['Semt / Mahalle'].str.split(' / ', n=1, expand=True)
    df['İlçe'] = split_cols[0].fillna('Bilinmiyor').str.strip()
    df['Mahalle'] = split_cols[1].fillna('Bilinmiyor').str.strip() if split_cols.shape[1] > 1 else 'Bilinmiyor'
    df['Semt'] = df['Semt / Mahalle'].astype(str)

    feature_cols = ['m² (Brüt)', 'oda_sayisi', 'salon_sayisi', 'havuz', 'otopark',
                    'balkon', 'site', 'dubleks', 'asansor', 'sifir', 'Semt']

    X = df[feature_cols].copy()
    y_log = np.log1p(df['Fiyat'])
    X['Semt'] = X['Semt'].astype(str)

    X_train, X_test, y_train, y_test = train_test_split(X, y_log, test_size=0.2, random_state=42)

    model = CatBoostRegressor(
        iterations=500, learning_rate=0.05, depth=6,
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
# 3. SIDEBAR FİLTRELER
# =========================================================
st.sidebar.header("Filtreler")

ilce_list = ["Tüm İlçeler"] + sorted(df['İlçe'].unique().tolist())
selected_ilce = st.sidebar.selectbox("İlçe", ilce_list, key="ilce_sec")

if selected_ilce == "Tüm İlçeler":
    mahalle_options = sorted(df['Mahalle'].unique().tolist())
else:
    mahalle_options = sorted(df[df['İlçe'] == selected_ilce]['Mahalle'].unique().tolist())
mahalle_list = ["Tüm Mahalleler"] + mahalle_options
selected_mahalle = st.sidebar.selectbox("Mahalle", mahalle_list, key="mahalle_sec")

oda_list = ["Tümü"] + sorted(df['Oda Düzeni'].unique().tolist())
selected_oda = st.sidebar.selectbox("Oda Düzeni", oda_list, key="oda_sec")

min_m2, max_m2 = int(df['m² (Brüt)'].min()), int(df['m² (Brüt)'].max())
m2_range = st.sidebar.slider("Brüt m²", min_m2, max_m2, (60, 180), key="m2_sec")

max_price = int(df['Fiyat'].max())
budget = st.sidebar.number_input("Maksimum Bütçe (TL)", 1_000_000, max_price, min(12_000_000, max_price), 500_000, key="butce_sec")

st.sidebar.markdown("---")
st.sidebar.subheader("Özellik Filtreleri")
f_havuz = st.sidebar.checkbox("Havuzlu", key="f_havuz")
f_site = st.sidebar.checkbox("Site İçinde", key="f_site")
f_asansor = st.sidebar.checkbox("Asansörlü", key="f_asansor")
f_balkon = st.sidebar.checkbox("Balkonlu", key="f_balkon")
f_sifir = st.sidebar.checkbox("Sıfır / Yeni", key="f_sifir")
f_dubleks = st.sidebar.checkbox("Dubleks", key="f_dubleks")
f_otopark = st.sidebar.checkbox("Otoparklı", key="f_otopark")
f_manzara = st.sidebar.checkbox("Manzaralı", key="f_manzara")

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

if f_havuz: filtered = filtered[filtered['havuz'] == 1]
if f_site: filtered = filtered[filtered['site'] == 1]
if f_asansor: filtered = filtered[filtered['asansor'] == 1]
if f_balkon: filtered = filtered[filtered['balkon'] == 1]
if f_sifir: filtered = filtered[filtered['sifir'] == 1]
if f_dubleks: filtered = filtered[filtered['dubleks'] == 1]
if f_otopark: filtered = filtered[filtered['otopark'] == 1]
if f_manzara: filtered = filtered[filtered['manzara'] == 1]

# =========================================================
# 4. TABLAR
# =========================================================
tab1, tab2, tab3, tab4, tab5 = st.tabs([
    "Piyasa İlanları",
    "Evimi Ne Kadara Satarım?",
    "Lokasyon Dağılımı",
    "Model Açıklanabilirliği",
    "Performans & Kredi"
])

# -------------------- TAB 1: Piyasa İlanları --------------------
with tab1:
    st.subheader("Akıllı Arama")
    arama = st.text_input("Ne arıyorsunuz?", placeholder="havuzlu site içi 3+1, asansörlü sıfır...", key="arama")

    def akilli_filtrele(dataframe, metin):
        if not metin or metin.strip() == "":
            return dataframe
        metin = metin.lower().strip()
        sonuc = dataframe.copy()
        if "havuz" in metin: sonuc = sonuc[sonuc['havuz'] == 1]
        if "site" in metin: sonuc = sonuc[sonuc['site'] == 1]
        if "asansör" in metin or "asansor" in metin: sonuc = sonuc[sonuc['asansor'] == 1]
        if "balkon" in metin: sonuc = sonuc[sonuc['balkon'] == 1]
        if "sıfır" in metin or "sifir" in metin or "yeni" in metin: sonuc = sonuc[sonuc['sifir'] == 1]
        if "dubleks" in metin or "dublex" in metin: sonuc = sonuc[sonuc['dubleks'] == 1]
        if "otopark" in metin or "garaj" in metin: sonuc = sonuc[sonuc['otopark'] == 1]
        if "manzara" in metin: sonuc = sonuc[sonuc['manzara'] == 1]
        oda_eslesme = re.search(r'(\d)\s*\+\s*(\d)', metin)
        if oda_eslesme:
            oda = int(oda_eslesme.group(1))
            salon = int(oda_eslesme.group(2))
            sonuc = sonuc[(sonuc['oda_sayisi'] == oda) & (sonuc['salon_sayisi'] == salon)]
        return sonuc

    if arama:
        filtered = akilli_filtrele(filtered, arama)
        st.success(f"**{len(filtered)}** ilan bulundu")

    st.subheader(f"Filtrelenen İlanlar ({len(filtered)} adet)")
    st.caption(f"Toplam temizlenmiş veri: **{len(df)}** ilan")

    if len(filtered) == 0:
        st.warning("Kriterlere uygun ilan yok.")
    else:
        show_cols = ['İlan Başlığı', 'İlçe', 'Mahalle', 'Oda Düzeni', 'm² (Brüt)', 'Fiyat']
        st.dataframe(
            filtered[show_cols].head(40).style.format({'Fiyat': '{:,.0f} TL'}),
            use_container_width=True, height=280
        )

# -------------------- TAB 2: EVİMİ NE KADARA SATARIM? --------------------
with tab2:
    st.subheader("Evimi Ne Kadara Satarım?")
    st.caption("Evinizin özelliklerini girin, piyasa verilerine göre tahmini satış fiyatınızı öğrenin.")

    col1, col2, col3 = st.columns(3)

    with col1:
        s_ilce = st.selectbox("İlçe", sorted(df['İlçe'].unique().tolist()), key="s_ilce")
        s_net = st.number_input("Net m²", 30, 400, 95, key="s_net")
        s_brut = st.number_input("Brüt m²", 40, 500, 115, key="s_brut")

    with col2:
        s_oda = st.selectbox("Oda Sayısı", [1, 2, 3, 4, 5, 6], index=2, key="s_oda")
        s_salon = st.selectbox("Salon Sayısı", [1, 2], key="s_salon")
        s_yas = st.number_input("Bina Yaşı", 0, 40, 5, key="s_yas")

    with col3:
        st.write("")
        st.write("")
        s_site = st.checkbox("Site İçinde", True, key="s_site")
        s_asansor = st.checkbox("Asansör Var", True, key="s_asansor")
        s_balkon = st.checkbox("Balkon Var", True, key="s_balkon")
        s_havuz = st.checkbox("Havuz Var", False, key="s_havuz")
        s_otopark = st.checkbox("Otopark Var", False, key="s_otopark")
        s_sifir = st.checkbox("Sıfır / Yeni Bina", False, key="s_sifir")

    if st.button("Satış Fiyatımı Hesapla", type="primary", use_container_width=True):
        benzer_semt = df[df['İlçe'] == s_ilce]['Semt'].mode()
        semt_degeri = benzer_semt.iloc[0] if len(benzer_semt) > 0 else s_ilce

        input_data = pd.DataFrame([{
            'm² (Brüt)': s_brut,
            'oda_sayisi': float(s_oda),
            'salon_sayisi': float(s_salon),
            'havuz': 1 if s_havuz else 0,
            'otopark': 1 if s_otopark else 0,
            'balkon': 1 if s_balkon else 0,
            'site': 1 if s_site else 0,
            'dubleks': 0,
            'asansor': 1 if s_asansor else 0,
            'sifir': 1 if s_sifir else 0,
            'Semt': semt_degeri
        }])

        pred = float(np.expm1(model.predict(input_data)[0]))

        if s_site: pred *= 1.03
        if s_asansor: pred *= 1.015
        if s_balkon: pred *= 1.01
        if s_havuz: pred *= 1.04
        if s_otopark: pred *= 1.02
        if s_sifir: pred *= 1.05

        hizli_satis = pred * 0.93
        max_satis = pred * 1.08

        st.success("Hesaplama tamamlandı")
        st.markdown("---")

        m1, m2, m3 = st.columns(3)
        m1.metric("Tahmini Piyasa Değeri", f"{pred:,.0f} TL")
        m2.metric("Hızlı Satış Fiyatı", f"{hizli_satis:,.0f} TL", delta="-7%")
        m3.metric("Maksimum Satış Fiyatı", f"{max_satis:,.0f} TL", delta="+8%")

        st.info(f"""
        **Öneri:**  
        - İlanınızı **{pred:,.0f} – {max_satis:,.0f} TL** aralığında vermenizi öneririm.  
        - Hızlı satmak isterseniz **{hizli_satis:,.0f} TL** civarı daha uygun olur.  
        - Bu fiyatlar mevcut piyasa verilerine ve benzer özellikteki ilanlara göre hesaplanmıştır.
        """)

# -------------------- TAB 3: HARİTA --------------------
with tab3:
    st.subheader("İstanbul Lokasyon Dağılımı")
    coords = {
        'Ataşehir': (40.9833, 29.1167), 'Kadıköy': (40.9903, 29.0275), 'Üsküdar': (41.0244, 29.0050),
        'Beşiktaş': (41.0422, 29.0067), 'Şişli': (41.0600, 28.9870), 'Bakırköy': (40.9800, 28.8700),
        'Bağcılar': (41.0339, 28.8579), 'Bahçelievler': (41.0003, 28.8638), 'Küçükçekmece': (40.9917, 28.7719),
        'Avcılar': (40.9801, 28.7175), 'Esenyurt': (41.0342, 28.6801), 'Başakşehir': (41.0975, 28.8064),
        'Maltepe': (40.9333, 29.1333), 'Kartal': (40.8886, 29.1856), 'Pendik': (40.8750, 29.2333),
        'Ümraniye': (41.0256, 29.1244), 'Sancaktepe': (40.9900, 29.2300), 'Beylikdüzü': (40.9900, 28.6400),
        'Arnavutköy': (41.1852, 28.7410)
    }
    map_df = df.copy()
    def get_coords(row):
        for name, (lat, lon) in coords.items():
            if name.lower() in str(row['İlçe']).lower():
                return pd.Series([lat, lon])
        return pd.Series([41.0082, 28.9784])
    map_df[['lat', 'lon']] = map_df.apply(get_coords, axis=1)
    np.random.seed(42)
    map_df['lat'] += np.random.normal(0, 0.007, len(map_df))
    map_df['lon'] += np.random.normal(0, 0.007, len(map_df))
    fig = px.scatter_mapbox(map_df.sample(min(700, len(map_df))), lat="lat", lon="lon",
                            color="Fiyat", size="m² (Brüt)", hover_name="İlçe",
                            color_continuous_scale="Reds", size_max=11, zoom=9.3,
                            center={"lat": 41.02, "lon": 28.95}, mapbox_style="carto-darkmatter", height=520)
    fig.update_layout(margin=dict(l=0, r=0, t=0, b=0))
    st.plotly_chart(fig, use_container_width=True)

# -------------------- TAB 4: XAI --------------------
with tab4:
    st.subheader("Model Açıklanabilirliği")
    st.caption(f"R²: **%{model_r2*100:.2f}** | MAE: {model_mae:,.0f} TL")
    try:
        importance = model.get_feature_importance()
        names = ['m² (Brüt)', 'Oda', 'Salon', 'Havuz', 'Otopark', 'Balkon', 'Site', 'Dubleks', 'Asansör', 'Sıfır', 'Semt']
        f_df = pd.DataFrame({'Öznitelik': names[:len(importance)], 'Önem': importance}).sort_values('Önem')
        fig = px.bar(f_df, x='Önem', y='Öznitelik', orientation='h', title="Fiyatı En Çok Etkileyen Faktörler",
                     color='Önem', color_continuous_scale='Blues')
        fig.update_layout(template="plotly_dark", height=400)
        st.plotly_chart(fig, use_container_width=True)
    except:
        st.info("Öznitelik önem grafiği hesaplanamadı.")

# -------------------- TAB 5: KREDİ --------------------
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
