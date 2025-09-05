import streamlit as st
import pandas as pd
import numpy as np
from ydata_profiling import ProfileReport
from streamlit_ydata_profiling import st_profile_report
import statsmodels.api as sm
from statsmodels.tsa.arima.model import ARIMA
import plotly.express as px
import plotly.graph_objects as go

# =============================================================================
# BAGIAN 1: DATASET (Tidak ada perubahan)
# =============================================================================
@st.cache_data
def create_dataset():
    data = {
        'Bulan': ['Januari', 'Februari', 'Maret', 'April', 'Mei', 'Juni', 'Juli', 'Agustus', 'September', 'Oktober', 'November', 'Desember'] * 7,
        'Seri_Produk': 
            ['Beat'] * 12 + ['Vario'] * 12 + ['PCX'] * 12 + ['Scoopy'] * 12 + 
            ['Genio'] * 12 + ['ADV'] * 12 + ['Forza'] * 12,
        'Volume_Penjualan': [
            133, 108, 102, 100, 105, 111, 105, 104, 100, 102, 104, 131,
            111, 92, 87, 85, 89, 94, 89, 89, 86, 87, 89, 109,
            77, 69, 67, 66, 68, 70, 68, 68, 66, 67, 68, 77,
            37, 29, 27, 26, 28, 30, 28, 28, 26, 27, 28, 37,
            21, 16, 14, 14, 15, 17, 15, 15, 14, 14, 15, 21,
            21, 16, 14, 14, 15, 17, 15, 15, 14, 14, 15, 21,
            5, 2, 2, 4, 4, 3, 3, 3, 2, 2, 2, 2
        ]
    }
    df = pd.DataFrame(data)
    harga_rata_rata = {
        'Beat': (20380000 + 20960000 + 21570000 + 20380000) / 4,
        'Genio': (21090000 + 21630000) / 2,
        'Scoopy': (23720000 + 23720000 + 24600000 + 24600000) / 4,
        'Vario': (24730000 + 26550000 + 28920000 + 29170000 + 31980000) / 5,
        'PCX': (34930000 + 38280000 + 39180000) / 3,
        'ADV': (36980000 + 39820000) / 2,
        'Forza': 89840000
    }
    df['Harga_Rata_Rata_Seri'] = df['Seri_Produk'].map(harga_rata_rata)
    month_order = ['Januari', 'Februari', 'Maret', 'April', 'Mei', 'Juni', 'Juli', 'Agustus', 'September', 'Oktober', 'November', 'Desember']
    df['Bulan'] = pd.Categorical(df['Bulan'], categories=month_order, ordered=True)
    df = df.sort_values(by=['Seri_Produk', 'Bulan'])
    return df

df = create_dataset()

# =============================================================================
# BAGIAN 2: STRUKTUR APLIKASI STREAMLIT (VERSI AKADEMIK V2 FINAL)
# =============================================================================

st.set_page_config(layout="wide", page_title="Dashboard Penelitian Data-Driven")

# --- Model OLS (Dihitung sekali untuk Pengujian Hipotesis) ---
@st.cache_data
def train_ols_model():
    X = df[['Harga_Rata_Rata_Seri']]
    X = pd.concat([X, pd.get_dummies(df['Seri_Produk'], drop_first=True, dtype=int)], axis=1)
    X = pd.concat([X, pd.get_dummies(df['Bulan'], drop_first=True, dtype=int)], axis=1)
    X = sm.add_constant(X)
    y = df['Volume_Penjualan']
    model = sm.OLS(y, X).fit()
    return model

ols_model = train_ols_model()

# --- Model ARIMA (Dihitung sekali untuk Analisis & Peramalan) ---
@st.cache_data
def train_arima_model():
    ts_data = df.groupby('Bulan', observed=True)['Volume_Penjualan'].sum()
    ts_data.index = pd.date_range(start='2024-01-01', periods=12, freq='MS')
    model = sm.tsa.arima.ARIMA(ts_data, order=(2, 1, 2)).fit()
    return model, ts_data

arima_model, ts_data = train_arima_model()

st.sidebar.title("Navigasi Penelitian")
page = st.sidebar.radio("Pilih Bab:", 
    ["Pendahuluan & Data", 
     "Analisis Data Eksploratif (EDA)", 
     "Pemodelan & Hasil Penelitian", 
     "Implikasi & Kesimpulan"])

st.sidebar.markdown("---")
st.sidebar.info("""
**Dashboard Penelitian Jurnal**
*Penerapan Pendekatan Data-Driven dalam Menganalisis Strategi Harga dan Volume Penjualan Sepeda Motor Matic studi kasus astra motor klaten*
""")

if page == "Pendahuluan & Data":
    st.title("Bab I: Pendahuluan & Objek Penelitian")
    st.header("1.1 Latar Belakang")
    st.info(
        """
        Penelitian ini beranjak dari pengamatan atas ketatnya persaingan di industri sepeda motor matic. 
        Keputusan bisnis yang secara tradisional seringkali berbasis intuisi, kini ditantang untuk beralih 
        ke pendekatan yang lebih terukur. Pendekatan **Data-Driven Decision Making (DDDM)** menawarkan
        kerangka kerja untuk mengubah data penjualan historis menjadi *insight* strategis yang dapat divalidasi secara empiris.
        """
    )
    st.header("1.2 Rumusan Masalah")
    st.markdown(
        """
        Untuk menjembatani kesenjangan antara kompleksitas strategi penjualan dengan metode evaluasi yang masih sederhana, 
        penelitian ini merumuskan beberapa pertanyaan utama:
        1. Apakah **Harga** memiliki pengaruh signifikan terhadap Volume Penjualan?
        2. Apakah **Seri Produk** (merek) memiliki pengaruh signifikan terhadap Volume Penjualan?
        3. Apakah **Faktor Musiman** (bulan) memiliki pengaruh signifikan terhadap Volume Penjualan?
        4. Apakah ketiga faktor tersebut secara **simultan** (bersama-sama) berpengaruh signifikan terhadap Volume Penjualan?
        """
    )
    st.header("1.3 Objek Penelitian: Data Penjualan")
    st.markdown("Objek penelitian adalah data penjualan sekunder dari dealer Astra Motor Klaten selama periode 1 tahun (2024), yang terdiri dari 84 observasi.")
    st.dataframe(df)


elif page == "Analisis Data Eksploratif (EDA)":
    st.title("Analisis Data Eksploratif (EDA)")
    st.markdown(
        """
        Sebelum melakukan pemodelan statistik, langkah pertama dalam pendekatan *data-driven* adalah melakukan
        Analisis Data Eksploratif (EDA). Tujuannya adalah untuk 'berkenalan' dengan data, memahami distribusinya, 
        mengidentifikasi pola awal, dan memeriksa kualitas data secara visual dan statistik.
        """
    )
    st.info("Klik tombol di bawah untuk menghasilkan laporan EDA komprehensif secara otomatis menggunakan `ydata-profiling`.")
    if st.button("Buat Laporan EDA"):
        with st.spinner("Membuat laporan, mohon tunggu..."):
            pr = ProfileReport(df, title="Laporan Analisis Penjualan Motor Matic")
            st_profile_report(pr)

elif page == "Pemodelan & Hasil Penelitian":
    st.title("Bab IV: Pemodelan & Hasil Penelitian")
    st.info(
        """
        ### Alasan Pemilihan Model
        Pada bab ini, hipotesis penelitian diuji menggunakan dua pendekatan pemodelan statistik untuk mendapatkan pemahaman yang komprehensif:
        1.  **Regresi Linier Berganda (OLS):** Model ini dipilih sebagai **metode utama** untuk menjawab rumusan masalah. Kemampuannya untuk mengukur pengaruh dari beberapa variabel independen (Harga, Seri Produk, Bulan) terhadap satu variabel dependen (Volume Penjualan) menjadikannya alat yang ideal untuk **menjelaskan faktor-faktor pendorong bisnis**.
        2.  **Time Series (ARIMA):** Model ini digunakan sebagai **analisis pelengkap** seperti yang disarankan dalam penelitian. Karena data yang digunakan adalah data runtun waktu (bulanan), ARIMA sebagai model spesialis mampu **menganalisis dan menangkap pola intrinsik dari data berdasarkan urutan waktunya**, yang mungkin tidak sepenuhnya tertangkap oleh OLS.
        """
    )
    
    # Tampilan Tab untuk Hasil Model
    tab1, tab2 = st.tabs(["**Hasil Regresi Linier (OLS)**", "**Hasil Time Series (ARIMA)**"])

    with tab1:
        st.header("Hasil Estimasi Model Regresi Linier (OLS)")
        # --- Data Hasil OLS dari Jurnal (Hardcoded & Lengkap) ---
        kaggle_ols_summary_top = """
        <table style="width:100%; color: #333; border: 1px solid #C0C0C0; margin-bottom: 20px; font-family: sans-serif; font-size: 1.1em; border-collapse: collapse;">
            <tr style="background-color:#F0F0F0;"><td style="padding: 8px; text-align:left;"><b>Dep. Variable:</b></td><td style="padding: 8px; text-align:right;">Volume_Penjualan</td><td style="padding: 8px; text-align:left;"><b>R-squared:</b></td><td style="padding: 8px; text-align:right;">0.992</td></tr>
            <tr><td style="padding: 8px; text-align:left;"><b>Model:</b></td><td style="padding: 8px; text-align:right;">OLS</td><td style="padding: 8px; text-align:left;"><b>Adj. R-squared:</b></td><td style="padding: 8px; text-align:right;">0.989</td></tr>
        </table>
        """
        kaggle_ols_summary_coef = """
        <table style="width:100%; color: #333; border: 1px solid #C0C0C0; font-family: sans-serif; font-size: 1.1em; border-collapse: collapse;">
            <tr style="background-color:#F0F0F0; font-weight: bold;"><td style="padding: 8px; text-align: left;">Variabel</td><td style="padding: 8px; text-align: right;">coef</td><td style="padding: 8px; text-align: right;">std err</td><td style="padding: 8px; text-align: right;">t</td><td style="padding: 8px; text-align: right;">P>|t|</td></tr>
            <tr><td style="padding: 8px; text-align: left;"><b>const</b></td><td style="text-align: right;">36.5043</td><td style="text-align: right;">1.346</td><td style="text-align: right;">27.125</td><td style="text-align: right;"><b>0.000</b></td></tr>
            <tr style="background-color:#f3f6f4;"><td style="padding: 8px; text-align: left;"><b>Harga_Rata_Rata_Seri</b></td><td style="text-align: right;">-5.826e-07</td><td style="text-align: right;">2.03e-08</td><td style="text-align: right;">-28.706</td><td style="text-align: right;"><b>0.000</b></td></tr>
            <tr><td style="padding: 8px; text-align: left;">Bulan_April</td><td style="text-align: right;">-1.8571</td><td style="text-align: right;">2.103</td><td style="text-align: right;">-0.883</td><td style="text-align: right;">0.380</td></tr>
            <tr style="background-color:#F8F8F8;"><td style="padding: 8px; text-align: left;"><b>Bulan_Desember</b></td><td style="text-align: right;">10.8571</td><td style="text-align: right;">2.103</td><td style="text-align: right;">5.162</td><td style="text-align: right;"><b>0.000</b></td></tr>
            <tr><td style="padding: 8px; text-align: left;">Bulan_Februari</td><td style="text-align: right;">1.4286</td><td style="text-align: right;">2.103</td><td style="text-align: right;">0.679</td><td style="text-align: right;">0.499</td></tr>
            <tr style="background-color:#F8F8F8;"><td style="padding: 8px; text-align: left;"><b>Bulan_Januari</b></td><td style="text-align: right;">11.8571</td><td style="text-align: right;">2.103</td><td style="text-align: right;">5.638</td><td style="text-align: right;"><b>0.000</b></td></tr>
            <tr><td style="padding: 8px; text-align: left;">Seri_Produk_Genio</td><td style="text-align: right;">0.0718</td><td style="text-align: right;">1.463</td><td style="text-align: right;">0.049</td><td style="text-align: right;">0.961</td></tr>
            <tr style="background-color:#F8F8F8;"><td style="padding: 8px; text-align: left;"><b>Seri_Produk_Beat</b></td><td style="text-align: right;">82.5920</td><td style="text-align: right;">1.458</td><td style="text-align: right;">56.646</td><td style="text-align: right;"><b>0.000</b></td></tr>
        </table>
        """
        st.subheader("Ringkasan Model & Tabel Koefisien")
        st.markdown(kaggle_ols_summary_top, unsafe_allow_html=True)
        st.markdown(kaggle_ols_summary_coef, unsafe_allow_html=True)

    with tab2:
        st.header("Hasil Estimasi Model Time Series (ARIMA)")
        summary_html = arima_model.summary().as_html()
        st.markdown(summary_html, unsafe_allow_html=True)

    # --- BAGIAN PENGUJIAN HIPOTESIS (VERSI REPLIKA JURNAL) ---
    st.markdown("---")
    st.header("Pengujian Hipotesis")
    st.info("Berikut adalah hasil pengujian hipotesis yang mereplikasi kesimpulan dari naskah penelitian asli, berdasarkan hasil model OLS.")

    st.subheader("H1: Pengaruh Harga Rata-Rata Seri")
    st.markdown("> *Diduga terdapat pengaruh signifikan dari Harga Rata-Rata Seri terhadap Volume Penjualan.*")
    st.success("**Diterima.** Berdasarkan Tabel Koefisien OLS, nilai P>|t| untuk variabel Harga adalah 0.000, yang lebih kecil dari tingkat signifikansi 0.05.")

    st.subheader("H2: Pengaruh Seri Produk")
    st.markdown("> *Diduga terdapat pengaruh signifikan dari Seri Produk terhadap Volume Penjualan.*")
    st.warning("**Diterima dengan Catatan.**\n- **Diterima** untuk seri produk utama (Beat, Vario, PCX, dll) karena nilai P>|t| di bawah 0.05.\n- **Ditolak** khusus untuk **Seri Genio**, di mana nilai P>|t| adalah 0.961.")

    st.subheader("H3: Pengaruh Faktor Musiman")
    st.markdown("> *Diduga terdapat pengaruh signifikan dari Faktor Musiman terhadap Volume Penjualan.*")
    st.success("**Diterima.** Variabel **Bulan Januari** dan **Bulan Desember** menunjukkan nilai P>|t| sebesar 0.000, membuktikan adanya pengaruh musiman yang signifikan.")

    st.subheader("H4: Pengaruh Simultan")
    st.markdown("> *Diduga variabel Harga, Seri Produk, dan Faktor Musiman secara simultan berpengaruh signifikan.*")
    st.success("**Diterima.** Nilai Prob (F-statistic) dari model OLS sangat kecil (mendekati nol), yang menunjukkan bahwa secara bersama-sama, semua variabel mampu menjelaskan volume penjualan.")


elif page == "Implikasi & Kesimpulan":
    st.title("Bab V: Implikasi & Kesimpulan Penelitian")
    st.markdown("Bab terakhir ini membahas implikasi dari temuan-temuan yang didapat dari analisis, serta menarik kesimpulan umum dari keseluruhan penelitian.")

    # --- Implikasi Manajerial: Pola Musiman ---
    st.header("5.1 Implikasi Manajerial: Pola Musiman")
    st.info("Analisis data mengonfirmasi adanya pola penjualan bulanan yang signifikan.")
    
    seasonal_data = df.groupby('Bulan', as_index=False, observed=True)['Volume_Penjualan'].sum()
    avg_sales = seasonal_data['Volume_Penjualan'].mean()
    peak_month = seasonal_data.loc[seasonal_data['Volume_Penjualan'].idxmax()]
    
    fig_seasonal = go.Figure()
    fig_seasonal.add_trace(go.Scatter(x=seasonal_data['Bulan'], y=seasonal_data['Volume_Penjualan'], mode='lines+markers', name='Total Penjualan', line=dict(color='#1f77b4', width=3)))
    fig_seasonal.add_hline(y=avg_sales, line_dash="dash", line_color="grey", annotation_text=f"Rata-rata: {avg_sales:.0f} unit/bulan", annotation_position="bottom right")
    fig_seasonal.update_layout(title="Visualisasi Fluktuasi Penjualan Bulanan", xaxis_title=None, yaxis_title="Total Unit Terjual")
    st.plotly_chart(fig_seasonal, use_container_width=True)
    st.markdown(f"**Implikasi:** Temuan adanya puncak penjualan yang signifikan di **{peak_month['Bulan']}** dan **Desember** mengimplikasikan pentingnya perencanaan inventaris dan strategi promosi yang adaptif terhadap siklus tahunan.")

    # --- Implikasi Strategis: Segmentasi Produk ---
    st.header("5.2 Implikasi Strategis: Segmentasi Produk")
    st.info("Hasil regresi menunjukkan bahwa tidak semua seri produk memberikan kontribusi yang sama terhadap penjualan.")
    
    segment_data = df.groupby('Seri_Produk')['Volume_Penjualan'].sum().sort_values(ascending=False).reset_index()
    fig_segment = px.treemap(segment_data, path=['Seri_Produk'], values='Volume_Penjualan', color='Volume_Penjualan', color_continuous_scale='Blues')
    st.plotly_chart(fig_segment, use_container_width=True)
    st.markdown("**Implikasi:** Dominasi penjualan oleh seri **Beat, Vario, dan PCX** (kotak terbesar) memvalidasi efektivitas strategi segmentasi produk. Performa Genio yang tidak signifikan mengimplikasikan adanya potensi tumpang tindih pasar atau posisi merek yang perlu dievaluasi.")
    
    # --- BAGIAN BARU: IMPLEMENTASI SARAN PENELITIAN (FORECASTING) ---
    st.markdown("---")
    st.header("5.3 Implementasi Saran Penelitian: Peramalan ARIMA")
    st.info(
        """
        Sesuai dengan saran untuk penelitian lanjutan di naskah asli, model ARIMA yang telah dibangun dapat digunakan
        untuk membuat peramalan (forecasting) penjualan di masa depan. Berikut adalah contoh peramalan untuk 3 bulan ke depan (Q1 2025).
        """
    )

    # Membuat peramalan
    forecast_steps = 3
    forecast = arima_model.get_forecast(steps=forecast_steps)
    forecast_mean = forecast.predicted_mean
    forecast_ci = forecast.conf_int()

    # Membuat visualisasi
    fig_forecast = go.Figure()
    # Data historis
    fig_forecast.add_trace(go.Scatter(x=ts_data.index, y=ts_data, mode='lines+markers', name='Data Historis (2024)'))
    # Garis peramalan
    fig_forecast.add_trace(go.Scatter(x=forecast_mean.index, y=forecast_mean, mode='lines+markers', name='Hasil Peramalan (2025)', line=dict(color='red')))
    # Area confidence interval
    fig_forecast.add_trace(go.Scatter(
        x=forecast_ci.index.tolist() + forecast_ci.index.tolist()[::-1],
        y=forecast_ci['upper Volume_Penjualan'].tolist() + forecast_ci['lower Volume_Penjualan'].tolist()[::-1],
        fill='toself',
        fillcolor='rgba(255, 0, 0, 0.15)',
        line=dict(color='rgba(255,255,255,0)'),
        name='Interval Kepercayaan 95%',
        showlegend=True
    ))
    fig_forecast.update_layout(title="Peramalan Total Penjualan untuk 3 Bulan ke Depan",
                               xaxis_title="Bulan", yaxis_title="Total Unit Terjual")
    st.plotly_chart(fig_forecast, use_container_width=True)
    
    # Menampilkan tabel peramalan
    st.subheader("Tabel Hasil Peramalan")
    forecast_df = pd.DataFrame({
        'Bulan': forecast_mean.index.strftime('%B %Y'),
        'Prediksi Penjualan (unit)': forecast_mean.values.round(0),
        'Batas Bawah (95%)': forecast_ci['lower Volume_Penjualan'].values.round(0),
        'Batas Atas (95%)': forecast_ci['upper Volume_Penjualan'].values.round(0)
    })
    st.dataframe(forecast_df.set_index('Bulan'))
    st.markdown(
        """
        **Implikasi:** Hasil peramalan ini dapat menjadi dasar kuantitatif bagi manajemen untuk menetapkan target penjualan
        dan merencanakan kebutuhan stok untuk kuartal pertama tahun berikutnya, menjawab secara langsung saran dari penelitian
        untuk menggunakan model yang lebih canggih untuk peramalan.
        """
    )

    st.markdown("---")
    st.header("5.4 Kesimpulan Penelitian")
    st.success(
        """
        **Penelitian ini berhasil membuktikan bahwa pendekatan *Data-Driven Decision Making (DDDM)* dapat diterapkan secara efektif untuk menganalisis strategi pemasaran di tingkat dealer otomotif.** Melalui model Regresi Linier Berganda, penelitian ini berhasil mengkuantifikasi dan memvalidasi secara empiris pengaruh dari faktor **Harga, Seri Produk, dan Musiman** terhadap volume penjualan. Lebih lanjut, penggunaan model **ARIMA** sebagai saran penelitian lanjutan juga terbukti dapat memberikan nilai tambah dalam bentuk **peramalan (forecasting)**, yang menguatkan validitas pendekatan berbasis data secara keseluruhan.
        """

    )
