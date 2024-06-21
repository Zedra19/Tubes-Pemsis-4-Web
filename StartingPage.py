import streamlit as st

st.set_page_config(page_title="Starting Page", page_icon="📈")

st.header("Tugas Besar Pemodelan Simulasi CLO 4")
st.subheader("Kelompok H")
st.markdown("""
* Muhammad Farhan Soedjana (1301213565)           
* Gevin Latifan Abduldjalil (1301213541)
***
## Prediksi Harga Saham dengan Kombinasi Machine Learning dan Gerakan Brown Geometrik (GBM)
            
#### Description

Tugas ini dimaksudkan agar mahasiswa dapat mengeksplorasi kombinasi machine learning dan Gerakan Brown Geometrik (GBM) untuk memprediksi harga saham.
Mahasiswa akan menggunakan data historis, teknik feature engineering, dan model machine learning untuk memprediksi nilai drift/ perubahan harga pada simulasi GBM, sehingga menghasilkan perkiraan pergerakan harga yang lebih dinamis.

#### Goal :
            
* Mempelajari konsep Gerakan Brown Geometrik (GBM) untuk simulasi perubahan harga saham.
* Menggunakan teknik khusus untuk membuat fitur-fitur penting bagi model machine learning.
* Melatih model machine learning untuk memprediksi nilai drift dalam simulasi GBM.
* Merancang simulasi berbagai kemungkinan perubahan harga saham di masa depan menggunakan GBM dengan nilai drift yang diprediksi oleh machine learning.
* Mengimplementasikan hasil prediksi dalam web interface untuk aplikasi web.
            
#### Model Geomeric Brownian motion kontinyu untuk drift (μ) harga saham :

    dS(t)= μ * S(t)* Δt + σ * S(t)* dB(t)
    ds(t)= μ * S(t) * Δt + σ * S(t) * ε(t) *√(Δt)
            
#### Model BGM diskrit:
            """)
st.image(r'./Resources/Rumus 2.png')

st.markdown("""
#### Keterangan notasi:
* dS(t)		: Perubahan infinitesimal harga aset pada waktu t
* μ (mu)		: Koefisien drift (tingkat pengembalian jangka panjang yang diharapkan per unit waktu)
* S(t)		: Harga aset pada waktu t
* σ (sigma)	: Koefisien volatilitas (deviasi standar pengembalian per unit waktu)
* dt		: Interval waktu infinitesimal (perubahan bertahap waktu)
* dB(t)		: Penambahan proses Wiener standar (Gerakan Brown) (merepresentasikan goncangan atau fluktuasi acak pada harga)
* S(t_i)		: Harga aset pada langkah waktu i
* S(t_(i+1))		: Harga aset pada langkah waktu berikutnya (i+1)
* Δt		: Ukuran langkah waktu (perbedaan antara langkah waktu)
* ε(t_i)		: Variabel acak normal standar dengan mean nol dan varians 1
            
#### Langkah-langkah yang akan dilakukan :
1. Pengumpulan dataset
2. Pra-prosessing
3. Training Model
4. Simulasi BGM
5. Visualisasi
####
            
### Visit Processing Page on the Left Navigation Bar
""")
