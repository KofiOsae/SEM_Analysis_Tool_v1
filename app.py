import streamlit as st
import numpy as np
import pandas as pd
import cv2
import plotly.express as px
import plotly.graph_objects as go
import scipy.signal
import json
from io import StringIO, BytesIO
import base64
import os
from PIL import Image
from scipy.signal import find_peaks

from streamlit_plotly_events import plotly_events
from streamlit_drawable_canvas import st_canvas

st.set_page_config(page_title="SEM & Profilometer Analyzer", layout="wide")

def parse_profilometer(file):
    try:
        if file.name.endswith(".csv"):
            return pd.read_csv(file)
        elif file.name.endswith((".xls", ".xlsx")):
            return pd.read_excel(file)
        else:
            content = file.read().decode()
            return pd.read_csv(StringIO(content), delim_whitespace=True, header=None)
    except Exception as e:
        st.error(f"Error parsing profilometer file: {e}")
        return None

def smooth_profile(y):
    return scipy.signal.savgol_filter(y, window_length=11, polyorder=3)

def roughness(y):
    return np.mean(np.abs(y - np.mean(y)))

def extract_profile_metrics(df):
    x = df.iloc[:,0].values
    y = df.iloc[:,1].values
    y_smooth = smooth_profile(y)
    norm_y = (y_smooth - np.min(y_smooth))/(np.max(y_smooth)-np.min(y_smooth)+1e-6)
    top_idx = np.where(norm_y > 0.85)[0]
    bottom_idx = np.where(norm_y < 0.15)[0]
    if len(top_idx) < 2 or len(bottom_idx) < 2:
        return {}, x, y, y_smooth
    top_width = x[top_idx[-1]] - x[top_idx[0]]
    bottom_width = x[bottom_idx[-1]] - x[bottom_idx[0]]
    height = np.mean(norm_y[top_idx]) - np.mean(norm_y[bottom_idx])
    angle = np.degrees(np.arctan(height/(0.5*abs(top_width-bottom_width)+1e-6)))
    base_rough = roughness(y[bottom_idx])
    top_rough = roughness(y[top_idx])
    return {
        "Top Width (µm)": round(top_width,3),
        "Bottom Width (µm)": round(bottom_width,3),
        "Height (normalized)": round(height,3),
        "Sidewall Angle (deg)": round(angle,1),
        "Ra Bottom (µm)": round(base_rough,4),
        "Ra Top (µm)": round(top_rough,4)
    }, x, y, y_smooth

def decode_image(uploaded_file):
    bytes_data = np.asarray(bytearray(uploaded_file.read()), dtype=np.uint8)
    return cv2.imdecode(bytes_data, cv2.IMREAD_GRAYSCALE)

def apply_roi(img, x, y, w, h):
    return img[y:y+h, x:x+w]

def detect_contours(img):
    blur = cv2.GaussianBlur(img, (5,5), 0)
    _, thresh = cv2.threshold(blur, 0, 255, cv2.THRESH_BINARY+cv2.THRESH_OTSU)
    contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    return contours

def analyze_contour(c, scale, gray_img, grayscale_to_micron=None, analysis_mode="full", feature_type="Arbitrary"):
    area = cv2.contourArea(c) * (scale**2)
    x, y, w, h = cv2.boundingRect(c)
    aspect_ratio = round(w/h,3) if h>0 else 0
    approx = cv2.approxPolyDP(c, 0.01*cv2.arcLength(c,True), True)
    
    ellipse_fit = None
    if feature_type=="Ellipse" and len(c) >= 5:
        try:
            ellipse_fit = cv2.fitEllipse(c)
        except:
            ellipse_fit = None

    if analysis_mode=="full":
        cropped = gray_img[y:y+h, x:x+w]
        profile = np.mean(cropped, axis=1)
        L = len(profile)
        win = min(11, L if L%2==1 else L-1)
        smooth_vals = profile if win<3 else scipy.signal.savgol_filter(profile, win, 2)
        norm = (smooth_vals - np.min(smooth_vals))/(np.max(smooth_vals)-np.min(smooth_vals)+1e-6)
        top_idx = np.where(norm>0.85)[0]
        bottom_idx = np.where(norm<0.15)[0]
        groove_depth = np.max(smooth_vals) - np.min(smooth_vals)
        rough_bottom = np.std(smooth_vals[bottom_idx]) if len(bottom_idx)>0 else 0
        rough_top = np.std(smooth_vals[top_idx]) if len(top_idx)>0 else 0
        sidewall_angle = None
        if len(top_idx)>0 and len(bottom_idx)>0:
            dx = w*scale
            dy = groove_depth
            sidewall_angle = np.degrees(np.arctan2(dy,dx))
        if grayscale_to_micron is not None:
            groove_depth_conv = groove_depth*grayscale_to_micron
            rough_bottom_conv = rough_bottom*grayscale_to_micron
            rough_top_conv = rough_top*grayscale_to_micron
        else:
            groove_depth_conv, rough_bottom_conv, rough_top_conv = groove_depth, rough_bottom, rough_top
        
        result = {
            "Area (µm²)": round(area,3),
            "Width (µm)": round(w*scale,3),
            "Height (µm)": round(h*scale,3),
            "Aspect Ratio": aspect_ratio,
            "Groove Depth ("+("µm" if grayscale_to_micron is not None else "a.u.")+")": round(groove_depth_conv,3),
            "Ra Bottom ("+("µm" if grayscale_to_micron is not None else "")+")": round(rough_bottom_conv,3),
            "Ra Top ("+("µm" if grayscale_to_micron is not None else "")+")": round(rough_top_conv,3),
            "Sidewall Angle (deg)": round(sidewall_angle,1) if sidewall_angle is not None else None
        }
        perimeter = cv2.arcLength(c, True)
        circularity = (4*np.pi*area)/(perimeter**2) if perimeter>0 else 0
        result["Circularity"] = round(circularity,3)
        if ellipse_fit is not None:
            result["Orientation (deg)"] = round(ellipse_fit[-1],1)
        if feature_type=="Gratings":
            cropped = gray_img[y:y+h, x:x+w]
            horiz_profile = np.mean(cropped, axis=0)
            pitch = fft_pitch_detection(horiz_profile, scale)
            result["Grating Pitch (µm)"] = round(pitch,3)
        return result

    elif analysis_mode=="line":
        cropped = gray_img[y:y+h, x:x+w]
        col_idx = cropped.shape[1]//2
        vertical_profile = np.mean(cropped[:, max(0,col_idx-1):min(cropped.shape[1],col_idx+2)], axis=1)
        row_idx = cropped.shape[0]//2
        horizontal_profile = np.mean(cropped[max(0,row_idx-1):min(cropped.shape[0], row_idx+2),:], axis=0)
        L = len(vertical_profile)
        win = min(11, L if L%2==1 else L-1)
        smooth_vert = vertical_profile if win<3 else scipy.signal.savgol_filter(vertical_profile, win, 2)
        groove_depth = np.max(smooth_vert) - np.min(smooth_vert)
        rough_bottom = np.std(smooth_vert[smooth_vert < np.median(smooth_vert)])
        rough_top = np.std(smooth_vert[smooth_vert >= np.median(smooth_vert)])
        norm_horiz = (horizontal_profile - np.min(horizontal_profile))/(np.max(horizontal_profile)-np.min(horizontal_profile)+1e-6)
        inds = np.where(norm_horiz>=0.5)[0]
        linewidth = inds[-1]-inds[0] if len(inds)>0 else 0
        if grayscale_to_micron is not None:
            groove_depth_conv = groove_depth*grayscale_to_micron
            rough_bottom_conv = rough_bottom*grayscale_to_micron
            rough_top_conv = rough_top*grayscale_to_micron
            linewidth_conv = linewidth*grayscale_to_micron
        else:
            groove_depth_conv, rough_bottom_conv, rough_top_conv, linewidth_conv = groove_depth, rough_bottom, rough_top, linewidth
        result = {
            "Area (µm²)": round(area,3),
            "Width (µm)": round(w*scale,3),
            "Height (µm)": round(h*scale,3),
            "Aspect Ratio": aspect_ratio,
            "Groove Depth ("+("µm" if grayscale_to_micron is not None else "a.u.")+")": round(groove_depth_conv,3),
            "Ra Bottom ("+("µm" if grayscale_to_micron is not None else "")+")": round(rough_bottom_conv,3),
            "Ra Top ("+("µm" if grayscale_to_micron is not None else "")+")": round(rough_top_conv,3),
            "Linewidth ("+("µm" if grayscale_to_micron is not None else "px")+")": round(linewidth_conv,3)
        }
        perimeter = cv2.arcLength(c, True)
        circularity = (4*np.pi*area)/(perimeter**2) if perimeter>0 else 0
        result["Circularity"] = round(circularity,3)
        if ellipse_fit is not None:
            result["Orientation (deg)"] = round(ellipse_fit[-1],1)
        return result

def fft_pitch_detection(profile, scale):
    signal = profile - np.mean(profile)
    fft_res = np.fft.fft(signal)
    freqs = np.fft.fftfreq(len(profile), d=scale)
    mag = np.abs(fft_res)
    mag[0] = 0
    idx = np.argmax(mag)
    freq = abs(freqs[idx])
    pitch = 1.0/freq if freq != 0 else 0
    return pitch

def plot_image_with_polygons(img, contours, selected_indices=[]):
    fig = go.Figure()
    _, im_arr = cv2.imencode('.png', img)
    im_bytes = im_arr.tobytes()
    im_b64 = base64.b64encode(im_bytes).decode()
    fig.add_layout_image(
        dict(
            source="data:image/png;base64,"+im_b64,
            xref="x", yref="y",
            x=0, y=img.shape[0],
            sizex=img.shape[1], sizey=img.shape[0],
            sizing="stretch", layer="below"
        )
    )
    for idx, c in enumerate(contours):
        if idx in selected_indices:
            continue
        approx = cv2.approxPolyDP(c, 0.01*cv2.arcLength(c,True), True)
        path = ""
        for pt in approx:
            x_pt, y_pt = pt[0]
            y_pt_plot = img.shape[0] - y_pt
            if path=="":
                path = f"M{x_pt},{y_pt_plot} "
            else:
                path += f"L{x_pt},{y_pt_plot} "
        path += "Z"
        fig.add_shape(
            type="path",
            path=path,
            line=dict(color="red", width=3)
        )
        M = cv2.moments(c)
        if M["m00"] != 0:
            cx = int(M["m10"]/M["m00"])
            cy = int(M["m01"]/M["m00"])
        else:
            cx, cy = 0, 0
        fig.add_annotation(
            x=cx, 
            y=img.shape[0]-cy,
            text=str(idx),
            showarrow=False,
            font=dict(color="white", size=14),
            bgcolor="red",
            opacity=0.8
        )
    fig.update_xaxes(visible=False, range=[0, img.shape[1]])
    fig.update_yaxes(visible=False, range=[0, img.shape[0]], scaleanchor="x")
    fig.update_layout(clickmode="event+select", margin=dict(l=0,r=0,t=0,b=0))
    return fig

def generate_html_report(df):
    html = "<html><head><title>SEM Analysis Report</title></head><body>"
    html += "<h1>SEM Analysis Report</h1>"
    html += df.to_html(index=False)
    html += "</body></html>"
    return html

def export_csv(df):
    return df.to_csv(index=False)

def export_session_config(config_dict):
    return json.dumps(config_dict, indent=4)

if "sem_line_profile" not in st.session_state:
    st.session_state.sem_line_profile = None

tab1, tab2, tab3 = st.tabs(["📷 SEM Image", "📏 Profilometer", "🧠 Tutorial"])

with tab1:
    st.header("SEM Image Analysis")
    
    st.subheader("Scale Measurement Tool")
    st.markdown("If your SEM image contains a scale bar, use the canvas below. (The uploaded scale image will appear as background if available.)")
    scale_img_file = st.file_uploader("Upload Scale Image (optional)", type=["png","jpg","jpeg"], key="scaleimg", help="Upload a close-up showing the scale bar")
    if scale_img_file is not None:
        scale_pil = Image.open(scale_img_file)
    else:
        scale_pil = None
    canvas_result = st_canvas(
        fill_color="rgba(255,165,0,0.3)",
        stroke_width=2,
        stroke_color="#FF0000",
        background_image=scale_pil,
        height=300,
        width=400,
        drawing_mode="line",
        key="scale_canvas"
    )
    known_length_input = st.text_input("Enter known length (µm)", value="10.0")
    computed_scale = None
    if canvas_result.json_data is not None and canvas_result.json_data.get("objects"):
        first_obj = canvas_result.json_data["objects"][0]
        points = first_obj.get("points", [])
        if points:
            start = points[0]
            end = points[-1]
            pix_length = np.sqrt((end["x"]-start["x"])**2 + (end["y"]-start["y"])**2)
            try:
                known_length = float(known_length_input)
                if pix_length > 0:
                    computed_scale = round(known_length / pix_length, 3)
                    st.write(f"**Computed Scale:** {computed_scale:.3f} µm per pixel")
            except:
                st.error("Invalid value for known length.")
    else:
        st.info("Draw a line over the scale bar (if visible) to calibrate the scale.")
    
    master_batch = st.checkbox("Enable Batch Mode (multiple SEM images)", value=False)
    if master_batch:
        files = st.file_uploader("Upload SEM images", type=["png","jpg","jpeg","tif","tiff"], accept_multiple_files=True, key="sem_files")
    else:
        file = st.file_uploader("Upload SEM image", type=["png","jpg","jpeg","tif","tiff"], key="sem_file")
        files = [file] if file is not None else []
    
    if files:
        st.subheader("Preprocessing Options")
        preprocess = st.checkbox("Apply Preprocessing", value=False)
        if preprocess:
            do_clahe = st.checkbox("Apply CLAHE", value=False)
            do_blur = st.slider("Gaussian Blur Kernel Size", min_value=1, max_value=11, value=5, step=2)
            do_inversion = st.checkbox("Invert Image", value=False)
            threshold_option = st.selectbox("Thresholding Method", ["None", "Fixed", "Adaptive", "Otsu"])
            if threshold_option=="Fixed":
                threshold_val = st.slider("Fixed Threshold Value", min_value=0, max_value=255, value=127, step=1)
            elif threshold_option=="Adaptive":
                adaptive_block = st.slider("Adaptive Block Size (odd)", min_value=3, max_value=31, value=11, step=2)
                adaptive_C = st.slider("Adaptive Constant", min_value=0, max_value=20, value=2, step=1)
        else:
            do_clahe = do_blur = do_inversion = False
            threshold_option = "None"
    
        trim_bottom = st.number_input("Smart image trimming: Crop bottom (px)", min_value=0, value=0)
        use_fft = st.checkbox("Use FFT-based pitch detection (for Gratings)", value=False)
    
        if computed_scale is not None:
            scale = computed_scale
        else:
            scale = st.number_input("Scale (µm per pixel)", min_value=0.001, max_value=100.0, value=0.050, step=0.001, format="%.3f")
    
        convert_units = st.checkbox("Convert grayscale values to physical units (µm)")
        if convert_units:
            grayscale_to_micron = st.number_input("Conversion Factor (µm per grayscale unit)", min_value=0.001, max_value=1.0, value=0.020, step=0.001, format="%.3f")
        else:
            grayscale_to_micron = None
    
        analysis_mode = st.radio("Select Analysis Mode", ["Full Image", "Line Profile"]).lower()
        feature_type = st.selectbox("Select Feature Type", ["Dots", "Array", "Ellipse", "Gratings", "Arbitrary"])
    
        if use_fft and feature_type=="Gratings":
            fft_placeholder = st.empty()
    
        st.sidebar.subheader("Feature Filtering Settings")
        apply_filtering = st.sidebar.checkbox("Activate Filtering", value=True)
        use_area_filter = st.sidebar.checkbox("Area Filter", value=True)
        use_width_filter = st.sidebar.checkbox("Width Filter", value=True)
        use_height_filter = st.sidebar.checkbox("Height Filter", value=True)
        use_aspect_filter = st.sidebar.checkbox("Aspect Ratio Filter", value=True)
        use_solidity_filter = st.sidebar.checkbox("Solidity Filter", value=True)
        use_circularity_filter = st.sidebar.checkbox("Circularity Filter", value=True)
        use_orientation_filter = st.sidebar.checkbox("Orientation Filter (for Ellipses)", value=(feature_type=="Ellipse"))
    
        min_area = st.sidebar.number_input("Min Area (µm²)", min_value=0.0, max_value=1000.0, value=0.0, step=1.0) if use_area_filter else 0
        max_area = st.sidebar.number_input("Max Area (µm²)", min_value=0.0, max_value=1000.0, value=1000.0, step=5.0) if use_area_filter else 1e6
        min_width = st.sidebar.number_input("Min Width (µm)", min_value=0.0, max_value=500.0, value=0.0, step=1.0) if use_width_filter else 0
        max_width = st.sidebar.number_input("Max Width (µm)", min_value=0.0, max_value=500.0, value=500.0, step=5.0) if use_width_filter else 1e6
        min_height = st.sidebar.number_input("Min Height (µm)", min_value=0.0, max_value=500.0, value=0.0, step=1.0) if use_height_filter else 0
        max_height = st.sidebar.number_input("Max Height (µm)", min_value=0.0, max_value=500.0, value=500.0, step=5.0) if use_height_filter else 1e6
        min_aspect = st.sidebar.number_input("Min Aspect Ratio", min_value=0.0, max_value=10.0, value=0.0, step=0.1) if use_aspect_filter else 0
        max_aspect = st.sidebar.number_input("Max Aspect Ratio", min_value=0.0, max_value=10.0, value=10.0, step=0.1) if use_aspect_filter else 100
        min_solidity = st.sidebar.number_input("Min Solidity", min_value=0.0, max_value=1.0, value=0.0, step=0.05) if use_solidity_filter else 0
        if feature_type=="Dots":
            min_circularity = st.sidebar.number_input("Min Circularity (Dots)", min_value=0.0, max_value=1.0, value=0.6, step=0.05) if use_circularity_filter else 0
        elif feature_type=="Ellipse":
            min_orientation = st.sidebar.number_input("Min Orientation (deg)", min_value=0.0, max_value=360.0, value=0.0, step=5.0) if use_orientation_filter else 0
            max_orientation = st.sidebar.number_input("Max Orientation (deg)", min_value=0.0, max_value=360.0, value=180.0, step=5.0) if use_orientation_filter else 360
            min_circularity = st.sidebar.number_input("Min Circularity (Ellipse)", min_value=0.0, max_value=1.0, value=0.4, step=0.05) if use_circularity_filter else 0
        else:
            min_circularity = st.sidebar.number_input("Min Circularity", min_value=0.0, max_value=1.0, value=0.3, step=0.05) if use_circularity_filter else 0
    
        all_results = []
        batch_names = []
    
        for uploaded_file in files:
            if uploaded_file is None:
                continue
            img = decode_image(uploaded_file)
            if preprocess:
                if do_clahe:
                    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
                    img = clahe.apply(img)
                if do_blur:
                    ksize = do_blur if do_blur % 2==1 else do_blur+1
                    img = cv2.GaussianBlur(img, (ksize,ksize),0)
                if do_inversion:
                    img = 255 - img
                if threshold_option=="Fixed" and threshold_val is not None:
                    _, img = cv2.threshold(img, threshold_val, 255, cv2.THRESH_BINARY)
                elif threshold_option=="Adaptive":
                    if adaptive_block is not None and adaptive_C is not None:
                        img = cv2.adaptiveThreshold(img,255,cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, adaptive_block, adaptive_C)
                elif threshold_option=="Otsu":
                    _, img = cv2.threshold(img, 0, 255, cv2.THRESH_BINARY+cv2.THRESH_OTSU)
            if trim_bottom > 0:
                img = img[:-trim_bottom, :]
    
            use_roi = st.checkbox("Use ROI?", value=False, key=uploaded_file.name)
            if use_roi:
                roi_x = st.number_input("ROI X", 0, img.shape[1], 0, key=uploaded_file.name+"roi_x")
                roi_y = st.number_input("ROI Y", 0, img.shape[0], 0, key=uploaded_file.name+"roi_y")
                roi_w = st.number_input("ROI Width", 1, img.shape[1], img.shape[1], key=uploaded_file.name+"roi_w")
                roi_h = st.number_input("ROI Height", 1, img.shape[0], img.shape[0], key=uploaded_file.name+"roi_h")
                img_proc = apply_roi(img, int(roi_x), int(roi_y), int(roi_w), int(roi_h))
            else:
                img_proc = img.copy()
    
            st.image(img_proc, caption=f"Processed SEM Image: {uploaded_file.name}", use_container_width=True)
    
            if use_fft and feature_type=="Gratings":
                profile_fft = np.mean(img_proc, axis=0)
                fft_pitch_val = fft_pitch_detection(profile_fft, scale)
                st.write(f"Estimated Grating Pitch: {fft_pitch_val:.3f} µm")
    
            contours = detect_contours(img_proc)
            if not contours:
                st.warning(f"No contours detected in {uploaded_file.name}.")
                continue
            output = []
            boxes = []
            for idx, c in enumerate(contours):
                metrics = analyze_contour(c, scale, img_proc, grayscale_to_micron, analysis_mode, feature_type)
                output.append(metrics)
                x0, y0, w0, h0 = cv2.boundingRect(c)
                boxes.append({"index": idx, "x": x0, "y": y0, "w": w0, "h": h0})
    
            st.markdown("**Click on a contour (number) in the image to remove it from the results.**")
            fig_poly = plot_image_with_polygons(img_proc, contours, selected_indices=[])
            events = plotly_events(fig_poly, click_event=True)
            del_indices = []
            if events:
                for ev in events:
                    if "text" in ev:
                        try:
                            del_idx = int(ev["text"])
                            if del_idx not in del_indices:
                                del_indices.append(del_idx)
                        except:
                            pass
            updated_output = [o for idx, o in enumerate(output) if idx not in del_indices]
    
            if apply_filtering:
                if feature_type=="Ellipse":
                    filtered_output = [ o for o in updated_output if (
                        o is not None and 
                        (o["Area (µm²)"] >= min_area and o["Area (µm²)"] <= max_area) and
                        (o["Width (µm)"] >= min_width and o["Width (µm)"] <= max_width) and
                        (o["Height (µm)"] >= min_height and o["Height (µm)"] <= max_height) and
                        (o.get("Aspect Ratio",1) >= min_aspect and o.get("Aspect Ratio",1) <= max_aspect) and
                        (o.get("Solidity",1) >= min_solidity) and
                        (o.get("Circularity",1) >= min_circularity) and
                        (min_orientation <= o.get("Orientation (deg)",0) <= max_orientation)
                    )]
                else:
                    filtered_output = [ o for o in updated_output if (
                        o is not None and 
                        (o["Area (µm²)"] >= min_area and o["Area (µm²)"] <= max_area) and
                        (o["Width (µm)"] >= min_width and o["Width (µm)"] <= max_width) and
                        (o["Height (µm)"] >= min_height and o["Height (µm)"] <= max_height) and
                        (o.get("Aspect Ratio",1) >= min_aspect and o.get("Aspect Ratio",1) <= max_aspect) and
                        (o.get("Solidity",1) >= min_solidity) and
                        (o.get("Circularity",1) >= min_circularity)
                    )]
            else:
                filtered_output = [o for o in updated_output if o is not None]
    
            if not filtered_output:
                st.warning(f"No features met filtering criteria in {uploaded_file.name}. Showing all detected features.")
                filtered_output = updated_output
    
            df = pd.DataFrame(filtered_output)
            df["Image"] = uploaded_file.name
            indices_to_remove = st.multiselect(f"Manually remove features from {uploaded_file.name} (by index)", df.index.tolist())
            if indices_to_remove:
                df = df.drop(indices_to_remove)
    
            all_results.append(df)
            batch_names.append(uploaded_file.name)
    
            st.write(f"**Feature Table for {uploaded_file.name}:**")
            st.dataframe(df)
            st.subheader("Summary Statistics")
            st.dataframe(df.describe().T.round(3))
    
            plot_metric = st.selectbox("Select Metric to Plot", df.columns, key=uploaded_file.name+"plot")
            fig_line = px.line(x=list(df.index), y=df[plot_metric],
                               labels={"x": "Feature #", "y": plot_metric},
                               title=f"{plot_metric} vs Feature Index")
            st.plotly_chart(fig_line, use_container_width=True)
    
            ra_bottom_col = "Ra Bottom (" + ("µm" if grayscale_to_micron is not None else "") + ")"
            ra_top_col = "Ra Top (" + ("µm" if grayscale_to_micron is not None else "") + ")"
            if ra_bottom_col in df.columns and ra_top_col in df.columns:
                fig_bar = px.bar(df, x=df.index, y=[ra_bottom_col, ra_top_col],
                                 barmode="group",
                                 labels={"value": "Roughness (Ra)", "variable": "Region"},
                                 title="Roughness (Ra) at Top vs Bottom")
                st.plotly_chart(fig_bar, use_container_width=True)
    
            csv_data = export_csv(df)
            st.download_button("Export CSV for This Image", csv_data, uploaded_file.name+"_features.csv", "text/csv")
    
            if analysis_mode=="line":
                if st.button(f"Extract SEM Line Profile from {uploaded_file.name}"):
                    st.session_state.sem_line_profile = np.mean(apply_roi(img, 0, 0, img.shape[1], img.shape[0]), axis=1)
                    st.success("Line profile extracted and stored for Profilometer analysis.")
    
        if master_batch and all_results:
            combined_df = pd.concat(all_results, ignore_index=True)
            st.subheader("Combined Batch Report")
            st.dataframe(combined_df)
            csv_batch = export_csv(combined_df)
            st.download_button("Download Combined CSV", csv_batch, "combined_features.csv", "text/csv")
            html_report = generate_html_report(combined_df)
            st.markdown(get_download_link(html_report, "SEM_Report.html", "text/html"), unsafe_allow_html=True)
    
        session_config = {
            "scale": scale,
            "convert_units": convert_units,
            "grayscale_to_micron": grayscale_to_micron,
            "analysis_mode": analysis_mode,
            "feature_type": feature_type,
            "thresholding": {
                "method": threshold_option,
                "fixed": threshold_val if threshold_option=="Fixed" else None,
                "adaptive_block": adaptive_block if threshold_option=="Adaptive" else None,
                "adaptive_C": adaptive_C if threshold_option=="Adaptive" else None
            },
            "trim_bottom": trim_bottom,
            "use_fft": use_fft,
            "filtering": {
                "activated": apply_filtering,
                "min_area": min_area,
                "max_area": max_area,
                "min_solidity": min_solidity,
                "min_circularity": min_circularity,
                "min_width": min_width,
                "max_width": max_width,
                "min_height": min_height,
                "max_height": max_height,
                "min_aspect": min_aspect,
                "max_aspect": max_aspect,
                "use_area_filter": use_area_filter,
                "use_width_filter": use_width_filter,
                "use_height_filter": use_height_filter,
                "use_aspect_filter": use_aspect_filter,
                "use_solidity_filter": use_solidity_filter,
                "use_circularity_filter": use_circularity_filter,
                "use_orientation_filter": use_orientation_filter
            }
        }
        session_json = export_session_config(session_config)
        st.download_button("Export Session Config (JSON)", session_json, "session_config.json", "application/json")
    else:
        st.info("Please upload at least one SEM image.")
        

with tab2:
    st.header("Stylus Profilometer Analysis")

    uploaded_file = st.file_uploader("Upload profilometer trace (.csv, .xlsx, .txt)", type=["csv", "txt", "xls", "xlsx"], key="prof_file")
    if uploaded_file:
        df = parse_profilometer(uploaded_file)
        if df is not None and df.shape[1] >= 2:
            x = df.iloc[:, 0].values
            y = df.iloc[:, 1].values
            y_smooth = smooth_profile(y)

            st.subheader("Interactive Profilometer Plot")
            col1, col2 = st.columns(2)
            with col1:
                m_pos = st.slider("Marker M Position (µm)", min_value=float(x[0]), max_value=float(x[-1]), value=float(x[0] + (x[-1] - x[0]) * 0.25))
            with col2:
                r_pos = st.slider("Marker R Position (µm)", min_value=float(x[0]), max_value=float(x[-1]), value=float(x[0] + (x[-1] - x[0]) * 0.75))

            m_index = np.searchsorted(x, m_pos)
            r_index = np.searchsorted(x, r_pos)
            if m_index > r_index:
                m_index, r_index = r_index, m_index
                m_pos, r_pos = r_pos, m_pos

            show_raw = st.checkbox("Show Raw Profile", value=True)
            show_smooth = st.checkbox("Show Smoothed Profile", value=True)

            fig = go.Figure()
            if show_raw:
                fig.add_trace(go.Scatter(x=x, y=y, mode='lines', name='Raw Profile', line=dict(color='gray')))
            if show_smooth:
                fig.add_trace(go.Scatter(x=x, y=y_smooth, mode='lines', name='Smoothed Profile', line=dict(color='blue')))
            fig.add_vline(x=m_pos, line=dict(color='red', dash='dash'), name='Marker M')
            fig.add_vline(x=r_pos, line=dict(color='green', dash='dash'), name='Marker R')
            fig.update_layout(title='Profilometer Profile',
                              xaxis_title='Lateral (µm)',
                              yaxis_title='Height (µm)',
                              showlegend=True)
            st.plotly_chart(fig, use_container_width=True)

            region_x = x[m_index:r_index]
            region_y = y_smooth[m_index:r_index]
            
            # Ensure region_y is defined and valid
            try:
                region_y = np.asarray(region_y, dtype=float).flatten()
                region_y = region_y[np.isfinite(region_y)]
                  if region_y.size > 1:
                     peaks, _ = find_peaks(region_y)
                     valleys, _ = find_peaks(-region_y)
                 
                     max_vals = region_y[peaks] if len(peaks) > 0 else np.array([np.max(region_y)])
                     min_vals = region_y[valleys] if len(valleys) > 0 else np.array([np.min(region_y)])
                 else:
                     peaks = valleys = np.array([])
                     max_vals = min_vals = np.array([0.0])
            
             except Exception as e:
                 st.error(f"Error processing region_y: {e}")
             
                peaks = valleys = np.array([])
                max_vals = min_vals = np.array([0.0])
            
             #region_y, peaks, valleys = sanitize_and_find_peaks(region_y)
             max_vals = region_y[peaks] if len(peaks) > 0 else np.array([np.max(region_y)])
             min_vals = region_y[valleys] if len(valleys) > 0 else np.array([np.min(region_y)])
             avg_max = np.mean(max_vals)
             std_max = np.std(max_vals)
             avg_min = np.mean(min_vals)
             std_min = np.std(min_vals)
            
             top_threshold = avg_max - 0.1 * delta_z
             bottom_threshold = avg_min + 0.1 * delta_z
             top_indices = np.where(region_y >= top_threshold)[0]
             bottom_indices = np.where(region_y <= bottom_threshold)[0]
            
             top_width = region_x[top_indices[-1]] - region_x[top_indices[0]] if len(top_indices) > 1 else 0
             top_width_std = np.std(region_x[top_indices]) if len(top_indices) > 1 else 0
             ra_top = np.mean(np.abs(region_y[top_indices] - np.mean(region_y[top_indices]))) if len(top_indices) > 0 else 0
             ra_top_std = np.std(np.abs(region_y[top_indices] - np.mean(region_y[top_indices]))) if len(top_indices) > 0 else 0
            
             bottom_width = region_x[bottom_indices[-1]] - region_x[bottom_indices[0]] if len(bottom_indices) > 1 else 0
             bottom_width_std = np.std(region_x[bottom_indices]) if len(bottom_indices) > 1 else 0
             ra_bottom = np.mean(np.abs(region_y[bottom_indices] - np.mean(region_y[bottom_indices]))) if len(bottom_indices) > 0 else 0
             ra_bottom_std = np.std(np.abs(region_y[bottom_indices] - np.mean(region_y[bottom_indices]))) if len(bottom_indices) > 0 else 0

        else:
            st.warning("Invalid profilometer file format.")
    else:
        st.info("Upload a profilometer trace file for analysis.")

with tab3:
    st.markdown("""

    **SEM Tab:**
    - **Scale Measurement:**  
      If your SEM image includes a scale bar, use the canvas at the top. Draw a line along the scale bar, then enter the known length (µm) into the provided input. The conversion factor (µm per pixel, accurate to three decimals) will be computed.
      
    - **Thresholding Options:**  
      Choose from several thresholding methods:
         - **None:** No thresholding.
         - **Fixed:** Use a fixed threshold value.
         - **Adaptive:** Adjust block size and constant for local thresholding.
         - **Otsu:** Automatically determine threshold using Otsu's method.
      
    - **Analysis Modes:**  
      - *Full Image:* Extracts a complete vertical profile from each detected feature.
      - *Line Profile:* Extracts separate vertical and horizontal profiles for measuring groove depth, roughness, and linewidth.
      
    - **Feature Types:**  
      - **Dots:** Small, nearly circular features.
      - **Array:** Features arranged in a regular grid.
      - **Ellipse:** Elongated features (with ellipse fitting for orientation and aspect ratio).
      - **Gratings:** Periodic structures (with FFT-based pitch detection).
      - **Arbitrary:** Features that don't conform to standard shapes.
      
    - **Filtering:**  
      Use the sidebar to activate/deactivate individual filters (area, width, height, aspect ratio, solidity, circularity, and orientation for ellipses). Adjust the threshold using the provided number inputs.
      You may also click on contour annotations in the image to remove features directly.
      
    - **Exporting:**  
      Each image's results can be exported as CSV; you can also export a combined CSV for batch mode.
      
    - **SEM to Profilometer Transfer:**  
      In Line Profile mode, you can extract a line profile from the SEM image and transfer it to the Profilometer tab for further analysis.
      
    **Profilometer Tab:**
    - Upload a trace file (CSV, XLSX, or TXT) or import a line profile from the SEM tab.
    - The app displays raw and smoothed traces using interactive Plotly charts.
    - Use the vertical reference sliders to define the region for measurement. The tool calculates the height difference, slope, and angle for that region.
    
    Enjoy using this integrated metrology tool!""")
