import streamlit as st

st.set_page_config(page_title="Analysis & Conclusion", page_icon="📈")

st.header("Analysis & Conclusion")

st.markdown("""
1. Akurasi Model Machine Learning:
Model Random Forest Regressor menunjukkan kemampuan yang baik dalam memprediksi drift yang digunakan dalam simulasi GBM.
Fitur-fitur teknis seperti SMA, EMA, dan RSI terbukti efektif sebagai input dalam pelatihan model untuk prediksi harga saham.

2. Efektivitas Simulasi GBM:
Simulasi GBM dengan drift yang diprediksi oleh model machine learning mampu menghasilkan jalur harga saham yang mendekati harga aktual.
Meskipun terdapat beberapa deviasi, secara keseluruhan, simulasi ini memberikan prediksi yang cukup akurat.

3. Visualisasi dan Evaluasi:
Visualisasi perbandingan antara harga yang diprediksi dan harga aktual memberikan gambaran yang jelas tentang performa model.
Distribusi error menunjukkan bahwa kesalahan prediksi berada dalam batas yang wajar dan dapat diterima untuk aplikasi di pasar keuangan.

4. Implementasi dan Penggunaan:
Kombinasi antara model machine learning dan simulasi GBM dapat digunakan sebagai alat bantu dalam pengambilan keputusan investasi.
Sistem ini dapat memberikan perkiraan harga saham yang lebih akurat dan membantu investor dalam merencanakan strategi investasi mereka.

#### Kesimpulan:
Secara keseluruhan, proyek ini berhasil mencapai tujuan yang ditetapkan dengan mengembangkan sistem prediksi harga saham yang menggabungkan kekuatan machine learning 
dan simulasi stokastik. Hasil yang diperoleh menunjukkan potensi besar dari pendekatan ini dalam aplikasi nyata
di pasar keuangan.
""")