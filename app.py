import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import io
import time
import math
from gradio_client import Client

# ==========================================
# 1. KONFIGURASI & INISIALISASI SERVER
# ==========================================
st.set_page_config(page_title="Auto-Predict Gender", page_icon="📊", layout="wide")

@st.cache_resource
def get_hf_client():
    # Menginisiasi koneksi ke Hugging Face Space Anda
    API_URL = "marselferrys/indo_name-gender-prediction"
    return Client(API_URL)

client = get_hf_client()

# ==========================================
# 2. FUNGSI PENDUKUNG (MODULAR)
# ==========================================
def find_name_column(columns):
    """
    Mendeteksi secara otomatis kolom yang kemungkinan berisi nama.
    Mengecek variasi: "nama", "name", "NAMA", "Name", dll.
    """
    target_keywords = ["nama", "name"]
    for col in columns:
        if str(col).strip().lower() in target_keywords:
            return col
    return None

def create_excel_download(df):
    """
    Mengonversi DataFrame pandas ke dalam format file Excel (.xlsx) di memory
    agar bisa diunduh melalui Streamlit.
    """
    buffer = io.BytesIO()
    with pd.ExcelWriter(buffer, engine='openpyxl') as writer:
        df.to_excel(writer, index=False, sheet_name='Hasil_Prediksi')
    return buffer.getvalue()

# ==========================================
# 3. ANTARMUKA PENGGUNA (UI)
# ==========================================
st.title("📊 Mass Gender-Indo_Name Prediction Tool")
st.markdown("""
Unggah dataset Excel Anda di bawah ini. Sistem akan otomatis mencari kolom **Nama** dan menggunakan model *Hybrid MLE-BiLSTM*  untuk melabeli Jenis Kelamin secara massal.
""")

st.markdown("---")

# A. UPLOAD FILE
uploaded_file = st.file_uploader("📂 Unggah file Excel (.xlsx)", type=["xlsx"])

if uploaded_file is not None:
    # B. BACA DATA & DETEKSI KOLOM
    try:
        df = pd.read_excel(uploaded_file)
        
        st.subheader(f"1. Preview Data Asli, Jumlah Data: {len(df)} ")
        st.dataframe(df.head(5), use_container_width=True)
        
        # Deteksi kolom nama
        detected_col = find_name_column(df.columns)
        
        st.subheader("2. Konfigurasi Kolom Target")
        if detected_col:
            st.success(f"✅ Sistem otomatis mendeteksi kolom: **'{detected_col}'**")
            target_col = st.selectbox("Pilih kolom yang berisi Nama Lengkap:", df.columns, index=list(df.columns).index(detected_col))
        else:
            st.warning("⚠️ Sistem tidak menemukan kolom dengan nama 'Nama' atau 'Name'. Silakan pilih secara manual.")
            target_col = st.selectbox("Pilih kolom yang berisi Nama Lengkap:", df.columns)
            
        # C. TOMBOL EKSEKUSI PREDIKSI (VERSI CHUNKING AMAN)
        if st.button("🚀 Mulai Prediksi Massal", type="primary"):
            
            # 1. Ambil seluruh data nama menjadi satu List
            list_nama_lengkap = df[target_col].astype(str).tolist()
            total_data = len(list_nama_lengkap)
            
            # 2. Tentukan ukuran per paket (Batas aman: 5.000 data per pengiriman API)
            batch_size = 5000 
            total_batches = math.ceil(total_data / batch_size)
            
            # Setup UI Progress
            progress_bar = st.progress(0)
            status_text = st.empty()
            
            all_pred_genders = []
            all_confidence = []
            
            try:
                # Mualai hitung waktu
                start_time = time.time()
                # 3. Kirim data per potongan (chunk) ke Hugging Face
                for i in range(total_batches):
                    # Potong list dari indeks awal ke akhir untuk batch ini
                    start_idx = i * batch_size
                    end_idx = min((i + 1) * batch_size, total_data)
                    chunk_nama = list_nama_lengkap[start_idx:end_idx]
                    
                    status_text.info(f"⏳ Memproses batch {i+1} dari {total_batches} (Data {start_idx+1} hingga {end_idx})...")
                    
                    # Tembak API Batch di Hugging Face
                    result = client.predict(chunk_nama, api_name="/predict_batch")
                    
                    # Gabungkan hasil dari batch ini ke list utama
                    all_pred_genders.extend(result[0])
                    all_confidence.extend(result[1])
                    
                    # Update progress bar
                    progress_bar.progress((i + 1) / total_batches)

                end_time = time.time()
                total_inference_time = end_time - start_time
                
                status_text.success(f"✅ Pemrosesan {total_data:,} data Selesai dalam waktu **{total_inference_time:.2f} detik**!")
                
                # 4. Masukkan hasil akhir ke dalam DataFrame
                df['pred_gender'] = all_pred_genders
                df['confidence_score'] = all_confidence
                
                st.markdown("---")
                st.subheader("3. Hasil Prediksi")
                
                # Tampilkan 2 kolom (Kiri untuk Tabel, Kanan untuk Chart)
                col1, col2 = st.columns([2, 1])
                
                with col1:
                    st.markdown("**Preview Data Hasil Prediksi:**")
                    st.dataframe(df, use_container_width=True, height=350)
                    
                    # Tombol Download
                    excel_data = create_excel_download(df)
                    st.download_button(
                        label="💾 Unduh File Excel Hasil Prediksi (.xlsx)",
                        data=excel_data,
                        file_name="Dataset_Gender_Predicted.xlsx",
                        mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
                        type="primary"
                    )
                    
                with col2:
                    st.markdown("**Distribusi Gender:**")
                    # Mapping ulang jika API me-return M/F menjadi Laki-laki/Perempuan (Opsional, jaga-jaga)
                    df['pred_gender'] = df['pred_gender'].replace({'M': 'Laki-laki', 'F': 'Perempuan'})
                    
                    # Menghitung jumlah Laki-laki & Perempuan
                    df_valid = df[df['pred_gender'].isin(['Laki-laki', 'Perempuan'])]
                    gender_counts = df_valid['pred_gender'].value_counts()
                    
                    if not gender_counts.empty:
                        # Visualisasi Pie Chart
                        fig, ax = plt.subplots(figsize=(4, 4))
                        # Set warna statis: Laki-laki Hijau, Perempuan Oranye
                        colors = ['#4CAF50' if x == 'Laki-laki' else '#FF9800' for x in gender_counts.index]
                        
                        ax.pie(
                            gender_counts, 
                            labels=gender_counts.index, 
                            autopct='%1.1f%%', 
                            startangle=90, 
                            colors=colors,
                            textprops={'fontsize': 10}
                        )
                        ax.axis('equal') 
                        st.pyplot(fig)
                    else:
                        st.info("Tidak ada data valid untuk ditampilkan pada grafik.")

            except Exception as e:
                status_text.error(f"❌ Terjadi kesalahan saat memanggil API: {e}")

    except Exception as e:
        st.error(f"❌ Terjadi kesalahan saat membaca file Excel: {e}")
