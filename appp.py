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
st.caption("CatBoost + 8.839 temiz ilan verisi ile piyasa değerleme")

# =========================================================
# 2. VERİ VE MODEL
# =========================================================
EXCEL_FILE = "ilanlar_temiz_hazir_sonn.xlsx"

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

    feature_cols = ['m² (Brüt)', 'oda_sayisi', 'salon_sayisi', 'Semt']

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
m2_range = st.sidebar.slider("Brüt m²", min_m2, max_m2, (40, 350), key="m2_sec")

max_price = int(df['Fiyat'].max())
budget = st.sidebar.number_input(
    "Maksimum Bütçe (TL)",
    min_value=500_000,
    max_value=max_price,
    value=min(40_000_000, max_price),
    step=500_000,
    key="butce_sec"
)

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
    arama = st.text_input("Ne arıyorsunuz?", placeholder="3+1, Ataşehir, 120 m²...", key="arama")

    if arama:
        metin = arama.lower().strip()
        mask = (
            filtered['İlan Başlığı'].str.lower().str.contains(metin, na=False) |
            filtered['İlçe'].str.lower().str.contains(metin, na=False) |
            filtered['Mahalle'].str.lower().str.contains(metin, na=False) |
            filtered['Oda Düzeni'].str.contains(metin, na=False)
        )
        oda_eslesme = re.search(r'(\d)\s*\+\s*(\d)', metin)
        if oda_eslesme:
            oda = int(oda_eslesme.group(1))
            salon = int(oda_eslesme.group(2))
            mask = mask | ((filtered['oda_sayisi'] == oda) & (filtered['salon_sayisi'] == salon))
        filtered = filtered[mask]
        st.success(f"**{len(filtered)}** ilan bulundu")

    st.subheader(f"Filtrelenen İlanlar ({len(filtered)} adet)")
    st.caption(f"Toplam temiz veri: **{len(df)}** ilan")

    if len(filtered) == 0:
        st.warning("Kriterlere uygun ilan yok.")
    else:
        show_cols = ['İlan Başlığı', 'İlçe', 'Mahalle', 'Oda Düzeni', 'm² (Brüt)', 'Fiyat']
        st.dataframe(
            filtered[show_cols].head(50).style.format({'Fiyat': '{:,.0f} TL', 'm² (Brüt)': '{:.0f}'}),
            use_container_width=True,
            height=300
        )

# -------------------- TAB 2: EVİMİ NE KADARA SATARIM? --------------------
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
        st.write("")
        st.write("")
        st.info("Model m², oda, salon ve semt bilgisine göre fiyat tahmin eder.")

    if st.button("Satış Fiyatımı Hesapla", type="primary", use_container_width=True):
        benzer = df[df['İlçe'] == s_ilce]['Semt'].mode()
        semt_degeri = benzer.iloc[0] if len(benzer) > 0 else s_ilce

        input_data = pd.DataFrame([{
            'm² (Brüt)': s_brut,
            'oda_sayisi': float(s_oda),
            'salon_sayisi': float(s_salon),
            'Semt': semt_degeri
        }])

        pred = float(np.expm1(model.predict(input_data)[0]))
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
        - Bu tahmin {len(df):,} gerçek ilan verisine dayanan CatBoost modeli ile yapılmıştır.
        """)

# -------------------- TAB 3: ODA DAĞILIMI + ISI HARİTASI --------------------
with tab3:
    st.subheader("Oda Düzeni Dağılımı & Fiyat Isı Haritası")

    col_a, col_b = st.columns(2)

    with col_a:
        st.markdown("#### Oda Düzeni Dağılımı")
        oda_counts = df['Oda Düzeni'].value_counts().head(12).reset_index()
        oda_counts.columns = ['Oda Düzeni', 'İlan Sayısı']

        fig_oda = px.bar(
            oda_counts,
            x='Oda Düzeni',
            y='İlan Sayısı',
            color='İlan Sayısı',
            color_continuous_scale='Blues',
            text='İlan Sayısı'
        )
        fig_oda.update_traces(textposition='outside')
        fig_oda.update_layout(template="plotly_dark", height=380, showlegend=False, margin=dict(t=20, b=40))
        st.plotly_chart(fig_oda, use_container_width=True)

    with col_b:
        st.markdown("#### Oda Düzenine Göre Ortalama Fiyat")
        oda_price = df.groupby('Oda Düzeni')['Fiyat'].mean().sort_values(ascending=False).head(12).reset_index()
        oda_price.columns = ['Oda Düzeni', 'Ortalama Fiyat']

        fig_price = px.bar(
            oda_price,
            x='Oda Düzeni',
            y='Ortalama Fiyat',
            color='Ortalama Fiyat',
            color_continuous_scale='Reds',
            text='Ortalama Fiyat'
        )
        fig_price.update_traces(texttemplate='%{text:,.0f}', textposition='outside')
        fig_price.update_layout(template="plotly_dark", height=380, showlegend=False, margin=dict(t=20, b=40))
        st.plotly_chart(fig_price, use_container_width=True)

    st.divider()
    st.markdown("#### İlçe × Oda Düzeni Ortalama Fiyat Isı Haritası")
    st.caption("Renk ne kadar koyu/kırmızıysa ortalama fiyat o kadar yüksek")

    top_ilceler = df['İlçe'].value_counts().head(15).index.tolist()
    top_odalar = df['Oda Düzeni'].value_counts().head(8).index.tolist()

    heat_df = df[df['İlçe'].isin(top_ilceler) & df['Oda Düzeni'].isin(top_odalar)]
    pivot = heat_df.pivot_table(values='Fiyat', index='İlçe', columns='Oda Düzeni', aggfunc='mean')
    pivot = pivot.reindex(top_ilceler)
    pivot = pivot[[c for c in top_odalar if c in pivot.columns]]

    fig_heat = px.imshow(
        pivot,
        color_continuous_scale='YlOrRd',
        aspect='auto',
        labels=dict(color="Ort. Fiyat (TL)")
    )
    fig_heat.update_layout(template="plotly_dark", height=520, margin=dict(l=10, r=10, t=30, b=10))
    fig_heat.update_traces(hovertemplate="İlçe: %{y}<br>Oda: %{x}<br>Ort. Fiyat: %{z:,.0f} TL<extra></extra>")
    st.plotly_chart(fig_heat, use_container_width=True)

# -------------------- TAB 4: XAI --------------------
with tab4:
    st.subheader("Model Açıklanabilirliği")
    st.caption(f"R²: **%{model_r2*100:.2f}**  |  MAE: {model_mae:,.0f} TL  |  Veri: {len(df):,} ilan")
    try:
        importance = model.get_feature_importance()
        names = ['m² (Brüt)', 'Oda Sayısı', 'Salon Sayısı', 'Semt / İlçe']
        f_df = pd.DataFrame({'Öznitelik': names[:len(importance)], 'Önem': importance}).sort_values('Önem')
        fig = px.bar(f_df, x='Önem', y='Öznitelik', orientation='h',
                     title="Fiyatı En Çok Etkileyen Faktörler",
                     color='Önem', color_continuous_scale='Blues')
        fig.update_layout(template="plotly_dark", height=380)
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
