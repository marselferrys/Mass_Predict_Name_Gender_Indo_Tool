import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import io
import time
import math
from gradio_client import Client

# ==========================================
# 1. KONFIGURASI & STATE
# ==========================================
st.set_page_config(page_title="Auto-Predict Gender", page_icon="📊", layout="wide")

# Session State Initialization
if "cancel_process" not in st.session_state:
    st.session_state.cancel_process = False

if "prediction_done" not in st.session_state:
    st.session_state.prediction_done = False

if "result_df" not in st.session_state:
    st.session_state.result_df = None


# ==========================================
# 2. CACHE CLIENT 
# ==========================================
@st.cache_resource
def get_hf_client():
    API_URL = "marselferrys/indo_name-gender-prediction"
    return Client(API_URL)

client = get_hf_client()


# ==========================================
# 3. FUNGSI PENDUKUNG
# ==========================================

def reset_state():
    """Fungsi untuk mereset semua state kembali ke default"""
    st.session_state.cancel_process = False
    st.session_state.prediction_done = False
    st.session_state.result_df = None
    
def find_name_column(columns):
    target_keywords = ["nama", "name"]
    for col in columns:
        if str(col).strip().lower() in target_keywords:
            return col
    return None


def create_excel_download(df):
    buffer = io.BytesIO()
    with pd.ExcelWriter(buffer, engine='openpyxl') as writer:
        df.to_excel(writer, index=False, sheet_name='Hasil_Prediksi')
    return buffer.getvalue()


# ==========================================
# 4. UI
# ==========================================
st.title("📊 Mass Gender-Indo_Name Prediction Tool")

st.markdown("""
Unggah dataset Excel Anda di bawah ini. Sistem akan otomatis mencari kolom **Nama** 
dan menggunakan model *Hybrid MLE-BiLSTM* untuk melabeli Jenis Kelamin secara massal.
""")

st.markdown("---")

uploaded_file = st.file_uploader("📂 Unggah file Excel (.xlsx)", 
                                 type=["xlsx"], 
                                 on_change=reset_state
                                )

if uploaded_file is not None:

    try:
        df = pd.read_excel(uploaded_file)

        st.subheader(f"1. Preview Data Asli (Total: {len(df)})")
        st.dataframe(df.head(5), use_container_width=True)

        detected_col = find_name_column(df.columns)

        st.subheader("2. Konfigurasi Kolom Target")

        if detected_col:
            st.success(f"✅ Sistem otomatis mendeteksi kolom: **'{detected_col}'**")
            target_col = st.selectbox(
                "Pilih kolom yang berisi Nama Lengkap:",
                df.columns,
                index=list(df.columns).index(detected_col)
            )
        else:
            st.warning("⚠️ Kolom Nama tidak terdeteksi otomatis.")
            target_col = st.selectbox(
                "Pilih kolom yang berisi Nama Lengkap:",
                df.columns
            )

        # ==========================================
        # Tombol Start & Cancel
        # ==========================================
        col_btn1, col_btn2 = st.columns([1, 1], gap="small")

        with col_btn1:
            start_prediction = st.button(
                "🚀 Mulai Prediksi",
                type="primary",
                use_container_width=True
            )

        with col_btn2:
            cancel_prediction = st.button(
                "❌ Cancel",
                use_container_width=True
            )

        if cancel_prediction:
            st.session_state.cancel_process = True
            st.session_state.prediction_done = False
            st.session_state.result_df = None
            st.rerun()

        # ==========================================
        # PROSES PREDIKSI
        # ==========================================
        if start_prediction:

            st.session_state.cancel_process = False
            st.session_state.prediction_done = False

            list_nama_lengkap = df[target_col].astype(str).tolist()
            total_data = len(list_nama_lengkap)

            batch_size = 5000
            total_batches = math.ceil(total_data / batch_size)

            progress_bar = st.progress(0)
            status_text = st.empty()

            all_pred_genders = []
            all_confidence = []

            try:
                start_time = time.time()

                for i in range(total_batches):

                    if st.session_state.cancel_process:
                        status_text.warning("⛔ Proses dibatalkan oleh pengguna.")
                        break

                    start_idx = i * batch_size
                    end_idx = min((i + 1) * batch_size, total_data)
                    chunk_nama = list_nama_lengkap[start_idx:end_idx]

                    status_text.info(
                        f"⏳ Memproses batch {i+1}/{total_batches}"
                    )

                    result = client.predict(
                        chunk_nama,
                        api_name="/predict_batch"
                    )

                    all_pred_genders.extend(result[0])
                    all_confidence.extend(result[1])

                    progress_bar.progress((i + 1) / total_batches)

                # Jika tidak dibatalkan
                if not st.session_state.cancel_process:

                    end_time = time.time()
                    total_time = end_time - start_time

                    status_text.success(
                        f"✅ Selesai dalam {total_time:.2f} detik!"
                    )

                    df['pred_gender'] = all_pred_genders
                    df['confidence_score'] = all_confidence
                    df['pred_gender'] = df['pred_gender'].replace(
                        {'M': 'L', 'F': 'P'}
                    )

                    st.session_state.result_df = df.copy()
                    st.session_state.prediction_done = True

            except Exception as e:
                status_text.error(f"❌ Error API: {e}")

        # ==========================================
        # TAMPILKAN HASIL
        # ==========================================
        if st.session_state.prediction_done:

            df = st.session_state.result_df

            st.markdown("---")
            st.subheader("3. Hasil Prediksi")

            col1, col2 = st.columns([2, 1])

            # =========================
            # TABEL
            # =========================
            with col1:
                st.markdown("**Preview Hasil Prediksi:**")
                st.dataframe(df, use_container_width=True, height=350)

                df_tabel = df.copy()
                excel_data = create_excel_download(df_tabel)

                st.download_button(
                    label="💾 Unduh File (.xlsx)",
                    data=excel_data,
                    file_name="Dataset_Gender_Predicted.xlsx",
                    mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
                    type="primary"
                )

            # =========================
            # PIE CHART
            # =========================
            with col2:
                error_rate = 5 # assign nilai error rate
                st.markdown(f"**Distribusi Data (error rate {error_rate}%)**")

                df_pie = df.copy()
                df_pie['pred_gender'] = df_pie['pred_gender'].replace(
                    {'L': 'Laki-laki', 'P': 'Perempuan'}
                )

                df_valid = df_pie[df_pie['pred_gender'].isin(['Laki-laki', 'Perempuan'])]
                gender_counts = df_valid['pred_gender'].value_counts()

                show_adjusted = st.toggle("Terapkan Error Rate", value=True)

                if show_adjusted:
                    gender_counts_used = (
                        gender_counts * (1 - error_rate / 100)
                    ).round().astype(int)
                    st.caption("Mode: Setelah dikurangi error rate")
                else:
                    gender_counts_used = gender_counts
                    st.caption("Mode: Distribusi asli")

                if not gender_counts_used.empty:

                    fig, ax = plt.subplots(figsize=(4, 4))

                    colors = [
                        '#0072B2' if x == 'Laki-laki' else '#E69F00'
                        for x in gender_counts_used.index
                    ]

                    wedges, texts, autotexts = ax.pie(
                        gender_counts_used,
                        labels=gender_counts_used.index,
                        autopct='%1.1f%%',
                        startangle=90,
                        colors=colors
                    )

                    for text in texts:
                        text.set_fontweight('bold')
                        text.set_fontsize(10)

                    for autotext in autotexts:
                        autotext.set_fontsize(9)

                    ax.axis('equal')
                    st.pyplot(fig)

                    if show_adjusted:
                        st.caption(f"Catatan: Distribusi pie chart merupakan hasil prediksi yang telah " 
                                   f"dikurangi dari asumsi error rate sebesar {error_rate}% pada masing-masing gender.")

                else:
                    st.info("Tidak ada data valid.")

    except Exception as e:
        st.error(f"❌ Error membaca file: {e}")
