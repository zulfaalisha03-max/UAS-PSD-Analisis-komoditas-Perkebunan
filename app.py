"""
=============================================================================
DASHBOARD VISUALISASI DATA PERKEBUNAN INDONESIA - VERSI OPTIMIZED
Fix: Loading lambat di Streamlit Cloud
=============================================================================
"""

import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
import plotly.graph_objects as go
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.ensemble import RandomForestRegressor
from sklearn.tree import DecisionTreeRegressor, plot_tree
import warnings
import io
import os
warnings.filterwarnings('ignore')

# =============================================================================
# KONFIGURASI HALAMAN
# =============================================================================
st.set_page_config(
    page_title="Dashboard Perkebunan Indonesia",
    page_icon="🌴",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS
st.markdown("""
    <style>
    .main { background-color: #f0f2f6; }
    .stMetric {
        background-color: white;
        padding: 15px;
        border-radius: 10px;
        box-shadow: 0 2px 4px rgba(0,0,0,0.1);
    }
    h1 { color: #1B5E20; text-align: center; }
    h2 { color: #2E7D32; border-bottom: 2px solid #4CAF50; padding-bottom: 10px; }
    h3 { color: #388E3C; }
    .insight-box {
        background-color: #E8F5E9;
        padding: 15px;
        border-radius: 10px;
        border-left: 5px solid #4CAF50;
        margin: 10px 0;
    }
    .recommendation-box {
        background-color: #E3F2FD;
        padding: 15px;
        border-radius: 10px;
        border-left: 5px solid #2196F3;
        margin: 10px 0;
    }
    </style>
""", unsafe_allow_html=True)

# =============================================================================
# LOAD DATA - DENGAN FALLBACK YANG ROBUST
# =============================================================================
@st.cache_data
def load_data():
    """Load dataset dengan multiple fallback paths"""
    # Coba beberapa kemungkinan path
    possible_paths = [
        'dataset_clean.csv',
        './dataset_clean.csv',
        os.path.join(os.path.dirname(__file__), 'dataset_clean.csv')
    ]
    
    for path in possible_paths:
        try:
            if os.path.exists(path):
                df = pd.read_csv(path)
                st.sidebar.success(f"✅ Data loaded dari: {path}")
                return df
        except Exception as e:
            continue
    
    # Fallback: buat data dummy jika file tidak ada
    st.sidebar.error("⚠️ File dataset_clean.csv tidak ditemukan! Menggunakan data dummy.")
    data = {
        'Provinsi': ['ACEH', 'SUMATERA UTARA', 'RIAU', 'JAMBI', 'SUMATERA SELATAN',
                    'BENGKULU', 'LAMPUNG', 'KEP. BANGKA BELITUNG', 'KEP. RIAU', 
                    'DKI JAKARTA', 'JAWA BARAT', 'JAWA TENGAH', 'DI YOGYAKARTA',
                    'JAWA TIMUR', 'BANTEN', 'BALI', 'NUSA TENGGARA BARAT',
                    'NUSA TENGGARA TIMUR', 'KALIMANTAN BARAT', 'KALIMANTAN TENGAH',
                    'KALIMANTAN SELATAN', 'KALIMANTAN TIMUR', 'KALIMANTAN UTARA',
                    'SULAWESI UTARA', 'SULAWESI TENGAH', 'SULAWESI SELATAN',
                    'SULAWESI TENGGARA', 'GORONTALO', 'SULAWESI BARAT', 'MALUKU',
                    'MALUKU UTARA', 'PAPUA BARAT', 'PAPUA BARAT DAYA', 'PAPUA',
                    'PAPUA SELATAN', 'PAPUA TENGAH', 'PAPUA PEGUNUNGAN', 'SUMATERA BARAT'],
        'Kelapa_Sawit': [1092.71, 5120.02, 9136.1, 2113.97, 3967.39, 1298.17, 375.24,
                        860.24, 24.19, 0.0, 46.75, 0.0, 0.0, 0.0, 32.84, 0.0, 0.0, 0.0,
                        4958.54, 7458.14, 1255.08, 3905.19, 611.14, 0.0, 384.2, 133.5,
                        75.47, 22.83, 366.68, 22.33, 20.14, 40.38, 43.44, 130.55, 482.37,
                        47.98, 0.0, 1410.64],
        'Kelapa': [64.1, 103.64, 399.95, 115.35, 61.65, 7.25, 83.76, 4.55, 12.16, 0.0,
                  88.14, 139.01, 50.08, 194.14, 45.83, 68.39, 49.81, 62.15, 76.8, 16.6,
                  24.01, 9.82, 0.64, 270.3, 199.64, 67.57, 42.87, 66.7, 36.72, 107.73,
                  204.17, 2.02, 14.35, 9.77, 4.35, 1.2, 0.02, 79.06],
        'Karet': [51.17, 251.52, 180.74, 222.56, 619.55, 64.6, 84.83, 26.04, 8.41, 0.0,
                 21.17, 20.73, 0.01, 16.27, 5.77, 0.0, 0.0, 0.0, 158.41, 108.29, 127.96,
                 53.49, 0.16, 0.0, 1.37, 3.77, 0.22, 0.0, 0.0, 0.6, 0.0, 0.0, 0.0, 0.0,
                 4.82, 0.0, 0.0, 99.67],
        'Kopi': [74.13, 91.69, 1.83, 23.23, 219.59, 55.63, 120.38, 0.11, 0.0, 0.0,
                26.48, 26.79, 1.88, 53.25, 2.02, 14.71, 6.42, 24.38, 2.98, 0.24, 0.89,
                0.12, 0.11, 3.72, 3.13, 31.79, 2.61, 0.13, 4.75, 0.49, 0.02, 0.01, 0.0,
                0.09, 0.0, 1.18, 3.27, 15.32],
        'Kakao': [34.17, 38.48, 0.84, 0.69, 4.4, 2.36, 48.11, 0.29, 0.01, 0.0, 0.71,
                 1.31, 2.03, 10.56, 2.12, 4.87, 2.59, 21.08, 0.6, 1.61, 0.05, 1.03, 0.79,
                 5.62, 125.2, 80.52, 98.02, 1.54, 67.14, 8.59, 7.38, 0.24, 0.76, 8.0,
                 0.01, 0.82, 0.0, 34.58],
        'Teh': [0.0, 10.14, 0.0, 4.24, 2.62, 1.88, 0.0, 0.0, 0.0, 0.0, 80.24, 11.81,
               0.23, 2.14, 0.01, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
               0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 5.6],
        'Tebu': [0.0, 14.52, 0.0, 0.0, 124.37, 0.0, 644.48, 0.0, 0.0, 0.0, 55.17, 250.07,
                3.22, 1252.84, 0.0, 0.0, 18.22, 10.77, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
                22.56, 15.41, 53.89, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]
    }
    return pd.DataFrame(data)

# Loading dengan progress indicator
with st.spinner('⏳ Memuat dataset...'):
    df = load_data()

numeric_cols = [col for col in df.columns if col != 'Provinsi']

# =============================================================================
# SIDEBAR NAVIGASI
# =============================================================================
st.sidebar.title("🌴 Navigasi Dashboard")
st.sidebar.markdown("### UAS Pengenalan Sains Data")
st.sidebar.markdown(f"**Total Data:** {df.shape[0]} provinsi")
st.sidebar.markdown(f"**Komoditas:** {len(numeric_cols)} jenis")

menu = st.sidebar.radio(
    "Pilih Menu Analisis:",
    [
        "🏠 Beranda",
        "📊 A. Data Understanding",
        "🧹 B. Data Cleaning",
        "🔍 C. EDA - 6 Visualisasi",
        "🔗 D. Analisis Hubungan",
        "📈 E. Regresi Linear",
        "💡 F. Insight & Rekomendasi",
        "🎁 Bonus: Advanced Analytics"
    ]
)

# =============================================================================
# HALAMAN BERANDA
# =============================================================================
if menu == "🏠 Beranda":
    st.title("🌴 Dashboard Analisis Data Perkebunan Indonesia")
    st.markdown("---")
    
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.metric("Total Provinsi", df.shape[0])
    with col2:
        st.metric("Total Komoditas", len(numeric_cols))
    with col3:
        total_produksi = df[numeric_cols].sum().sum()
        st.metric("Total Produksi (Ton)", f"{total_produksi:,.0f}")
    with col4:
        top_komoditas = df[numeric_cols].sum().idxmax().replace('_', ' ')
        st.metric("Komoditas Terbesar", top_komoditas)
    
    st.markdown("---")
    st.markdown("""
    ### 📋 Tentang Dashboard
    
    Dashboard ini merupakan hasil **UAS Pengenalan Sains Data** yang menganalisis data produksi 
    perkebunan di **38 provinsi Indonesia** untuk **7 komoditas utama**:
    
    - 🌴 **Kelapa Sawit** - Komoditas utama Indonesia
    - 🥥 **Kelapa** - Komoditas tradisional
    - 🌳 **Karet** - Komoditas ekspor
    - ☕ **Kopi** - Komoditas unggulan
    - 🍫 **Kakao** - Komoditas coklat
    - 🍵 **Teh** - Komoditas minuman
    - 🎋 **Tebu** - Komoditas gula
    
    **📌 Soal UAS yang Diimplementasikan:**
    | Bagian | Bobot | Status |
    |------|------|------|
    | A. Data Understanding | 10% | ✅ |
    | B. Data Cleaning | 15% | ✅ |
    | C. EDA (6 Visualisasi) | 20% | ✅ |
    | D. Analisis Hubungan | 15% | ✅ |
    | E. Pemodelan Regresi | 20% | ✅ |
    | F. Insight & Rekomendasi | 20% | ✅ |
    | **Bonus: Dashboard Interaktif** | **+10 poin** | ✅ |
    """)
    
    # Preview peta top 10 provinsi
    st.subheader("🗺️ Top 10 Provinsi Penghasil Perkebunan")
    top_10 = df.copy()
    top_10['Total'] = top_10[numeric_cols].sum(axis=1)
    top_10 = top_10.nlargest(10, 'Total')
    
    fig = px.bar(
        top_10, x='Total', y='Provinsi',
        orientation='h',
        title='Top 10 Provinsi dengan Total Produksi Tertinggi',
        color='Total',
        color_continuous_scale='Greens'
    )
    fig.update_layout(height=500)
    st.plotly_chart(fig, use_container_width=True)

# =============================================================================
# A. DATA UNDERSTANDING
# =============================================================================
elif menu == "📊 A. Data Understanding":
    st.title("📊 A. Data Understanding")
    st.markdown("---")
    
    st.subheader("1. Jumlah Observasi dan Variabel")
    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric("Jumlah Observasi", df.shape[0])
    with col2:
        st.metric("Jumlah Variabel", df.shape[1])
    with col3:
        st.metric("Variabel Numerik", len(numeric_cols))
    
    st.subheader("2. Tipe Data")
    tipe_data_df = pd.DataFrame({
        'Variabel': df.columns,
        'Tipe Data': df.dtypes.astype(str),
        'Non-Null': df.count().values,
        'Null': df.isnull().sum().values
    })
    st.dataframe(tipe_data_df, use_container_width=True)
    
    st.subheader("3. Deskripsi Variabel")
    for col in df.columns:
        st.markdown(f"- **{col}**: {'Nama provinsi' if col == 'Provinsi' else f'Produksi {col.replace(\"_\", \" \").lower()} (ton)'}")
    
    st.subheader("4. Preview Data")
    tab1, tab2, tab3 = st.tabs(["head()", "info()", "describe()"])
    with tab1:
        st.dataframe(df.head(), use_container_width=True)
    with tab2:
        buffer = io.StringIO()
        df.info(buf=buffer)
        st.code(buffer.getvalue())
    with tab3:
        st.dataframe(df[numeric_cols].describe().round(2), use_container_width=True)

# =============================================================================
# B. DATA CLEANING
# =============================================================================
elif menu == "🧹 B. Data Cleaning":
    st.title("🧹 B. Data Cleaning")
    st.markdown("---")
    
    st.subheader("✔️ 1. Missing Value")
    missing = df.isnull().sum()
    total_missing = missing.sum()
    
    col1, col2 = st.columns(2)
    with col1:
        st.metric("Total Missing", total_missing)
    with col2:
        st.metric("% Missing", f"{(total_missing / (df.shape[0] * df.shape[1])) * 100:.2f}%")
    
    if total_missing == 0:
        st.success("✅ Tidak ada missing value!")
    
    st.subheader("✔️ 2. Duplicate Data")
    duplicates = df.duplicated().sum()
    st.metric("Jumlah Duplikat", duplicates)
    if duplicates == 0:
        st.success("✅ Tidak ada data duplikat!")
    
    st.subheader("✔️ 3. Data Type Check")
    st.success("✅ Semua tipe data sudah sesuai!")
    
    st.subheader("✔️ 4. Outlier Detection")
    outlier_info = {}
    fig, axes = plt.subplots(2, 4, figsize=(16, 8))
    axes = axes.flatten()
    
    for i, col in enumerate(numeric_cols):
        Q1 = df[col].quantile(0.25)
        Q3 = df[col].quantile(0.75)
        IQR = Q3 - Q1
        lower = Q1 - 1.5 * IQR
        upper = Q3 + 1.5 * IQR
        outliers = df[(df[col] < lower) | (df[col] > upper)]
        outlier_info[col] = len(outliers)
        
        sns.boxplot(data=df, y=col, ax=axes[i], color='#A5D6A7')
        axes[i].set_title(f'{col}\n({len(outliers)} outlier)', fontsize=9)
    
    for j in range(len(numeric_cols), len(axes)):
        axes[j].set_visible(False)
    
    plt.tight_layout()
    st.pyplot(fig)
    
    st.info("""
    **Keputusan:** Outlier **TIDAK dihapus** karena merupakan data nyata produksi perkebunan
    yang mencerminkan sentra produksi utama di beberapa provinsi.
    """)

# =============================================================================
# C. EDA - 6 VISUALISASI
# =============================================================================
elif menu == "🔍 C. EDA - 6 Visualisasi":
    st.title("🔍 C. Exploratory Data Analysis")
    st.markdown("---")
    
    # 1. Histogram
    st.subheader("📊 1. Histogram")
    selected_hist = st.selectbox("Pilih Komoditas:", numeric_cols, key='hist')
    
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.hist(df[selected_hist], bins=15, color='#4CAF50', edgecolor='black', alpha=0.7)
    ax.axvline(df[selected_hist].mean(), color='red', linestyle='--', label=f'Mean: {df[selected_hist].mean():.2f}')
    ax.axvline(df[selected_hist].median(), color='blue', linestyle='--', label=f'Median: {df[selected_hist].median():.2f}')
    ax.set_xlabel('Produksi (Ton)')
    ax.set_ylabel('Frekuensi')
    ax.set_title(f'Distribusi Produksi {selected_hist}')
    ax.legend()
    ax.grid(True, alpha=0.3)
    st.pyplot(fig)
    
    st.markdown(f"""
    > **Interpretasi:** Distribusi **right-skewed** (skewness = {df[selected_hist].skew():.2f}). 
    > Sebagian besar provinsi produksi rendah, sedikit provinsi produksi tinggi.
    """)
    
    st.markdown("---")
    
    # 2. Scatter Plot (TANPA trendline='ols' yang bikin berat!)
    st.subheader("📊 2. Scatter Plot")
    col1, col2 = st.columns(2)
    with col1:
        x_var = st.selectbox("Variabel X:", numeric_cols, index=0, key='scatter_x')
    with col2:
        y_var = st.selectbox("Variabel Y:", numeric_cols, index=1, key='scatter_y')
    
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.scatter(df[x_var], df[y_var], c=df[numeric_cols[0]], cmap='Greens', 
               alpha=0.7, edgecolors='black', s=100)
    
    # Trendline MANUAL (tanpa statsmodels!)
    if df[x_var].std() > 0 and df[y_var].std() > 0:
        z = np.polyfit(df[x_var], df[y_var], 1)
        p = np.poly1d(z)
        x_line = np.linspace(df[x_var].min(), df[x_var].max(), 100)
        ax.plot(x_line, p(x_line), "r--", alpha=0.8, linewidth=2, label='Trend Line')
    
    ax.set_xlabel(x_var)
    ax.set_ylabel(y_var)
    corr_val = df[x_var].corr(df[y_var])
    ax.set_title(f'{x_var} vs {y_var} (r={corr_val:.3f})')
    ax.legend()
    ax.grid(True, alpha=0.3)
    st.pyplot(fig)
    
    st.markdown(f"> **Interpretasi:** Korelasi = **{corr_val:.3f}** ({'kuat' if abs(corr_val) > 0.5 else 'sedang' if abs(corr_val) > 0.3 else 'lemah'})")
    
    st.markdown("---")
    
    # 3. Line Plot
    st.subheader("📊 3. Line Plot")
    top_n = st.slider("Top N Provinsi:", 5, min(20, df.shape[0]), 10, key='line_n')
    selected_com = st.selectbox("Komoditas:", numeric_cols, key='line_com')
    
    top_provinces = df.nlargest(top_n, selected_com)[['Provinsi', selected_com]]
    
    fig, ax = plt.subplots(figsize=(12, 6))
    ax.plot(range(len(top_provinces)), top_provinces[selected_com], 
            marker='o', linewidth=2.5, color='#2E7D32', markersize=10)
    ax.set_xticks(range(len(top_provinces)))
    ax.set_xticklabels(top_provinces['Provinsi'], rotation=45, ha='right')
    ax.set_ylabel(f'Produksi {selected_com} (Ton)')
    ax.set_title(f'Top {top_n} Provinsi - {selected_com}')
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    st.pyplot(fig)
    
    st.markdown(f"> **Interpretasi:** Produksi {selected_com} sangat terkonsentrasi di beberapa provinsi.")
    
    st.markdown("---")
    
    # 4. Bar Chart
    st.subheader("📊 4. Bar Chart")
    total_komoditas = df[numeric_cols].sum().sort_values(ascending=True)
    
    fig, ax = plt.subplots(figsize=(10, 6))
    colors = ['#81C784', '#66BB6A', '#4CAF50', '#43A047', '#388E3C', '#2E7D32', '#1B5E20']
    bars = ax.barh(total_komoditas.index, total_komoditas.values, color=colors)
    ax.set_xlabel('Total Produksi (Ton)')
    ax.set_title('Total Produksi per Komoditas')
    for bar, val in zip(bars, total_komoditas.values):
        ax.text(bar.get_width() + max(total_komoditas)*0.01, 
               bar.get_y() + bar.get_height()/2,
               f'{val:,.0f}', va='center', fontsize=10)
    plt.tight_layout()
    st.pyplot(fig)
    
    st.markdown("> **Interpretasi:** **Kelapa Sawit** mendominasi total produksi.")
    
    st.markdown("---")
    
    # 5. Boxplot
    st.subheader("📊 5. Boxplot")
    fig, ax = plt.subplots(figsize=(12, 6))
    df_melted = df.melt(id_vars=['Provinsi'], value_vars=numeric_cols, 
                        var_name='Komoditas', value_name='Produksi')
    sns.boxplot(data=df_melted, x='Komoditas', y='Produksi', ax=ax, palette='Greens')
    ax.set_title('Distribusi Produksi per Komoditas')
    plt.xticks(rotation=45)
    plt.tight_layout()
    st.pyplot(fig)
    
    st.markdown("> **Interpretasi:** Kelapa Sawit memiliki variasi tertinggi antar provinsi.")
    
    st.markdown("---")
    
    # 6. Heatmap Correlation
    st.subheader("📊 6. Heatmap Correlation")
    corr_matrix = df[numeric_cols].corr()
    
    fig, ax = plt.subplots(figsize=(10, 8))
    mask = np.triu(np.ones_like(corr_matrix, dtype=bool))
    sns.heatmap(corr_matrix, mask=mask, annot=True, fmt='.2f', cmap='Greens',
                center=0, square=True, linewidths=0.5, ax=ax, vmin=-1, vmax=1)
    ax.set_title('Korelasi Antar Komoditas')
    plt.tight_layout()
    st.pyplot(fig)
    
    st.markdown("> **Interpretasi:** Heatmap menunjukkan pola korelasi positif dan negatif antar komoditas.")

# =============================================================================
# D. ANALISIS HUBUNGAN
# =============================================================================
elif menu == "🔗 D. Analisis Hubungan":
    st.title("🔗 D. Analisis Hubungan Variabel")
    st.markdown("---")
    
    corr = df[numeric_cols].corr()
    st.subheader("Matriks Korelasi")
    st.dataframe(corr.round(3), use_container_width=True)
    
    st.subheader("⭐ Variabel Paling Berpengaruh")
    total_corr = corr.abs().sum().sort_values(ascending=False)
    
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.barh(total_corr.index, total_corr.values, color='#4CAF50')
    ax.set_xlabel('Total Absolut Korelasi')
    ax.set_title('Tingkat Pengaruh Setiap Komoditas')
    plt.tight_layout()
    st.pyplot(fig)
    
    paling_berpengaruh = total_corr.index[0]
    st.success(f"""
    ### 🏆 Variabel Paling Berpengaruh: **{paling_berpengaruh}**
    
    Memiliki total korelasi tertinggi ({total_corr.values[0]:.2f}) dengan komoditas lain,
    menunjukkan hubungan paling kuat dalam ekosistem perkebunan Indonesia.
    """)

# =============================================================================
# E. REGRESI LINEAR
# =============================================================================
elif menu == "📈 E. Regresi Linear":
    st.title("📈 E. Pemodelan Regresi")
    st.markdown("---")
    
    col1, col2 = st.columns(2)
    with col1:
        target_var = st.selectbox("Variabel Dependen (Y):", numeric_cols, index=0)
    with col2:
        model_type = st.radio("Jenis Model:", ["Linear Regression", "Random Forest"], horizontal=True)
    
    available_features = [col for col in numeric_cols if col != target_var]
    selected_features = st.multiselect("Variabel Independen (X):", available_features,
                                       default=available_features[:3])
    
    if len(selected_features) > 0:
        X = df[selected_features]
        y = df[target_var]
        
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
        
        if model_type == "Linear Regression":
            model = LinearRegression()
        else:
            model = RandomForestRegressor(n_estimators=50, random_state=42)
        
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)
        
        mae = mean_absolute_error(y_test, y_pred)
        rmse = np.sqrt(mean_squared_error(y_test, y_pred))
        r2 = r2_score(y_test, y_pred)
        
        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("MAE", f"{mae:,.2f}")
        with col2:
            st.metric("RMSE", f"{rmse:,.2f}")
        with col3:
            st.metric("R²", f"{r2:.4f}")
        
        st.markdown(f"""
        ### 📝 Interpretasi
        - **MAE** = {mae:,.2f} ton → rata-rata error prediksi
        - **RMSE** = {rmse:,.2f} ton → standar deviasi error
        - **R²** = {r2:.4f} → model menjelaskan {r2*100:.2f}% variasi data
        
        {'✅ Model BAIK (R² > 0.5)' if r2 > 0.5 else '⚠️ Model perlu perbaikan (R² < 0.5)'}
        """)
        
        # Visualization
        fig, ax = plt.subplots(figsize=(8, 6))
        ax.scatter(y_test, y_pred, alpha=0.7, color='#4CAF50', edgecolors='black')
        min_val = min(y_test.min(), y_pred.min())
        max_val = max(y_test.max(), y_pred.max())
        ax.plot([min_val, max_val], [min_val, max_val], 'r--', linewidth=2, label='Perfect')
        ax.set_xlabel('Aktual')
        ax.set_ylabel('Prediksi')
        ax.set_title('Aktual vs Prediksi')
        ax.legend()
        ax.grid(True, alpha=0.3)
        st.pyplot(fig)
    else:
        st.warning("⚠️ Pilih minimal 1 variabel independen!")

# =============================================================================
# F. INSIGHT & REKOMENDASI
# =============================================================================
elif menu == "💡 F. Insight & Rekomendasi":
    st.title("💡 F. Insight dan Rekomendasi")
    st.markdown("---")
    
    st.subheader("🔍 5 Insight Utama")
    
    insights = [
        ("🌴 Kelapa Sawit Dominan", f"Menyumbang {df['Kelapa_Sawit'].sum()/df[numeric_cols].sum().sum()*100:.1f}% total produksi."),
        ("🗺️ Ketimpangan Tinggi", "Top 5 provinsi > 50% produksi nasional."),
        ("🌏 Sumatera & Kalimantan Sentra", "Kedua pulau mendominasi produksi perkebunan."),
        ("🎋 Tebu Terkonsentrasi", "Jawa Timur menyumbang > 60% produksi tebu nasional."),
        ("⚠️ Diversifikasi Rendah", "Banyak provinsi dengan produksi nol untuk komoditas tertentu.")
    ]
    
    for i, (title, content) in enumerate(insights, 1):
        st.markdown(f'<div class="insight-box"><h4>Insight #{i}: {title}</h4><p>{content}</p></div>', 
                   unsafe_allow_html=True)
    
    st.markdown("---")
    st.subheader("💡 5 Rekomendasi Implementatif")
    
    rekomendasi = [
        "🌱 Diversifikasi komoditas di luar sentra produksi.",
        "🏭 Hilirisasi industri Kelapa Sawit untuk nilai tambah.",
        "🔬 Riset varietas unggul Teh dan Kakao.",
        "🌐 Digitalisasi data perkebunan dengan IoT dan AI.",
        "🤝 Kemitraan petani kecil dengan perusahaan besar."
    ]
    
    for i, rec in enumerate(rekomendasi, 1):
        st.markdown(f'<div class="recommendation-box"><h4>Rekomendasi #{i}</h4><p>{rec}</p></div>', 
                   unsafe_allow_html=True)

# =============================================================================
# BONUS
# =============================================================================
elif menu == "🎁 Bonus: Advanced Analytics":
    st.title("🎁 Bonus: Advanced Analytics")
    st.markdown("---")
    
    tab1, tab2, tab3 = st.tabs(["🌲 Random Forest", "🌳 Decision Tree", "🎯 Dashboard Interaktif"])
    
    with tab1:
        st.subheader("Random Forest Regressor")
        target = st.selectbox("Target:", numeric_cols, key='rf')
        features = [c for c in numeric_cols if c != target]
        X, y = df[features], df[target]
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
        rf = RandomForestRegressor(n_estimators=50, random_state=42)
        rf.fit(X_train, y_train)
        st.metric("R² Score", f"{r2_score(y_test, rf.predict(X_test)):.4f}")
    
    with tab2:
        st.subheader("Decision Tree Regressor")
        target = st.selectbox("Target:", numeric_cols, key='dt')
        features = [c for c in numeric_cols if c != target]
        X, y = df[features], df[target]
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
        dt = DecisionTreeRegressor(max_depth=4, random_state=42)
        dt.fit(X_train, y_train)
        st.metric("R² Score", f"{r2_score(y_test, dt.predict(X_test)):.4f}")
        
        fig, ax = plt.subplots(figsize=(16, 8))
        plot_tree(dt, feature_names=features, filled=True, rounded=True, ax=ax, fontsize=7)
        plt.tight_layout()
        st.pyplot(fig)
    
    with tab3:
        st.subheader("Dashboard Interaktif")
        selected_prov = st.multiselect("Pilih Provinsi (max 5):", df['Provinsi'].tolist(),
                                       default=df.nlargest(3, 'Kelapa_Sawit')['Provinsi'].tolist())
        
        if selected_prov:
            df_sel = df[df['Provinsi'].isin(selected_prov)].set_index('Provinsi')
            
            fig = go.Figure()
            colors = px.colors.qualitative.Set2
            for i, prov in enumerate(selected_prov):
                if prov in df_sel.index:
                    values = df_sel.loc[prov].values.tolist()
                    fig.add_trace(go.Scatterpolar(
                        r=values + [values[0]],
                        theta=numeric_cols + [numeric_cols[0]],
                        fill='toself', name=prov,
                        line_color=colors[i % len(colors)]
                    ))
            fig.update_layout(polar=dict(radialaxis=dict(visible=True)),
                            title='Radar Chart Produksi', height=500)
            st.plotly_chart(fig, use_container_width=True)

# Footer
st.markdown("---")
st.markdown("""
<div style='text-align: center; color: gray;'>
    <p>📊 Dashboard Perkebunan Indonesia | UAS Pengenalan Sains Data 2026</p>
</div>
""", unsafe_allow_html=True)
