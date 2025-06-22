import streamlit as st
from PIL import Image, ImageEnhance
import numpy as np
import subprocess
import sys
import os
import shutil
import time

st.set_page_config(page_title="Ekstraksi Jaringan Jalan SAM-Road", layout="wide")
st.title("Ekstraksi Jaringan Jalan SAM-Road")

# Upload gambar
uploaded_file = st.file_uploader("Unggah citra", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    image = Image.open(uploaded_file)
    
    input_image_path = "input.png"
    image.save(input_image_path)

    display_image = image.copy()
    display_image.thumbnail((512, 512))
    st.image(display_image, caption='Citra yang diunggah', use_container_width=False)

    # Radio Button: Gamma Correction
    apply_gamma = st.radio("Terapkan Gamma Correction?", ["Tidak", "Ya"])
    if apply_gamma == "Ya":
        gamma_value = st.text_input("Masukkan nilai gamma (contoh: 1.25)", "1.25")
    else:
        gamma_value = '1'

    # Radio Button: Algoritma A*
    apply_astar = st.radio("Terapkan Algoritma A*?", ["Tidak", "Ya"])
    if apply_astar == "Ya":
        min_graph_dist = st.text_input("Min Graph Distance", "16")
        max_straight_dist = st.text_input("Max Straight Distance", "8")

    # Tombol inferensi
    if st.button("Lakukan Inferensi"):
        with st.spinner("Sedang menjalankan inferensi..."):
            start_time = time.time()
            
            command = [
                sys.executable, "sam_road/inferencer.py",
                "--config", 'sam_road/config/toponet_vitb_512_cityscale.yaml',
                "--checkpoint", 'sam_road/checkpoint/cityscale_vitb_512_e10.ckpt',
                "--gamma", gamma_value,
            ]
            result = subprocess.run(command, capture_output=True, text=True)
            if result.stderr:
                st.error("Error Inference:")
                st.error(result.stderr)
            
            if apply_astar == "Ya":
                command = [
                    sys.executable, "a-star-graph-reconstruction/astar.py",
                    "--min_graph_distance", min_graph_dist,
                    "--max_straight_distance", max_straight_dist,
                ]
                result = subprocess.run(command, capture_output=True, text=True)
                if result.stderr:
                    st.error("Error A*:")
                    st.error(result.stderr)
            
            end_time = time.time()  # Selesai hitung waktu
            waktu = end_time - start_time
            
        st.success(f"⏱️ Inference selesai dalam {waktu:.2f} detik")
        
        if apply_gamma == "Ya":
            st.info(f"Gamma correction diterapkan dengan nilai gamma: {gamma_value}")

        if apply_astar == "Ya":
            st.info(f"A*:\n- Min graph distance: {min_graph_dist}\n- Max straight distance: {max_straight_dist}")
            
        
        # Gambar 1: Mask
        if apply_gamma == "Ya":
            col1, col2 = st.columns(2)

            with col1:
                mask_no_gamma = Image.open("save/mask/result_no_gamma.png")
                st.image(mask_no_gamma, caption="Segmentation Mask Tanpa Gamma Correction", width=600)

            with col2:
                mask_with_gamma = Image.open("save/mask/result_road.png")
                st.image(mask_with_gamma, caption="Segmentation Mask Dengan Gamma Correction", width=600)
        else:
            mask_image = Image.open("save/mask/result_road.png")
            st.image(mask_image, caption="Segmentation Mask", width=600)

        # Gambar 2: Points
        points_image = Image.open("save/viz/result_points.png")
        st.image(points_image, caption="Non Maximum Suppression", width=600)

        if apply_astar == "Ya":
            col1, col2 = st.columns(2)

            with col1:
                before_astar_image = Image.open("save/viz/result.png")
                st.image(before_astar_image, caption="Sebelum Penerapan Algoritma A*", width=600)

            with col2:
                after_astar_image = Image.open("save/viz_astar/result.png")
                st.image(after_astar_image, caption="Setelah Penerapan Algoritma A*", width=600)
        else:
            result_image = Image.open("save/viz/result.png")
            st.image(result_image, caption="Hasil Ekstraksi Jaringan Jalan", width=600)

        # # Gambar 1: Mask
        # mask_image = Image.open("save/mask/result_road.png")
        # st.image(mask_image, caption="Segmentation Mask", use_container_width=True)

        # # Gambar 2: Points
        # points_image = Image.open("save/viz/result_points.png")
        # st.image(points_image, caption="Non Maximum Suppression", use_container_width=True)

        # if apply_astar == "Ya":
        #     result_image_path = "save/viz_astar/result.png"
        # else:
        #     result_image_path = "save/viz/result.png"

        # # Gambar 3: Hasil akhir
        # try:
        #     result_image = Image.open(result_image_path)
        #     st.image(result_image, caption="Hasil Ekstraksi Jaringan Jalan", use_container_width=True)
        # except FileNotFoundError:
        #     st.error("Gagal menemukan hasil inferensi di path 'save/viz/result.png'")
            
            
        # mask_image = Image.open("save/mask/result_road.png")
        # st.image(mask_image, caption="Segmentation Mask", use_container_width=True)
        
        # points_image = Image.open("save/viz/result_points.png")
        # st.image(points_image, caption="Non Maximum Suppresion", use_container_width=True)
            
        # if apply_gamma == "Ya":
        #     st.info(f"Gamma correction diterapkan dengan nilai gamma: {gamma_value}")
        
        # if apply_astar == "Ya":
        #     st.info(f"A*:\n- Min graph distance: {min_graph_dist}\n- Max straight distance: {max_straight_dist}")
        #     result_image_path = "save/viz_astar/result.png"
        # else:
        #     result_image_path = "save/viz/result.png"
            
        # try:
        #     result_image = Image.open(result_image_path)
        #     st.image(result_image, caption="Hasil Ekstraksi Jaringan Jalan", use_container_width=True)

        # except FileNotFoundError:
        #     st.error("Gagal menemukan hasil inferensi di path 'save/viz/result.png'")