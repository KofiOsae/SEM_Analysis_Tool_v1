
# === Enhancements Added ===
# 1. Session Save/Load Functionality
import base64

def save_session(data, filename="sem_session.json"):
    with open(filename, "w") as f:
        json.dump(data, f)
    st.success(f"Session saved to {filename}")

def load_session(filename="sem_session.json"):
    if os.path.exists(filename):
        with open(filename, "r") as f:
            return json.load(f)
    else:
        st.warning("Session file not found.")
        return None

# 2. AI Detection Toggle
st.sidebar.markdown("### Detection Mode")
use_ai_detection = st.sidebar.checkbox("Enable AI-based Feature Detection", value=False)

if use_ai_detection:
    st.sidebar.info("AI detection is enabled. Using pretrained model for feature segmentation.")
    # Placeholder for AI detection logic
    # You can integrate YOLOv5 or U-Net here
else:
    st.sidebar.info("Using contour-based detection.")

# 3. Deployment Readiness
# Add Streamlit Cloud compatibility
st.set_page_config(page_title="SEM Analysis Tool", layout="wide")


combine the existing code and comments to develop a well integrated and unique imaging analysis tool. 

You have room to improvement to exceed expectation and functionality. 

ask questions to understand if you something is unclear.

1. Original Vision & Core Functionality
A. Input and File Handling

SEM Images:

Upload one or multiple SEM images (supporting batch mode) in common formats (PNG, JPG, TIF, etc.).

Optionally define a Region of Interest (ROI) manually.

Profilometer Traces:

Upload trace files (CSV, XLS, TXT) for stylus/profile analysis.

B. Basic Processing & Measurement

For SEM Images:

Convert pixel measurements into physical units using a scale factor (µm per pixel).

Apply basic image preprocessing (e.g., blurring, inversion) to enhance contour detection.

Detect contours and compute basic metrics like area, width, height, and aspect ratio.

Calculate additional metrics such as groove depth, Ra (roughness) for top and bottom regions, and sidewall angle.

Display an annotated overlay on the SEM image (e.g. with bounding boxes) and export results as CSV.

For Profilometer Data:

Plot the raw and smoothed profiles.

Compute summary statistics (e.g., top width, bottom width, height, etc.).

Display these results interactively.

C. Tutorial and Help

A basic tutorial tab explains how to upload images, process them, and export results.

2. Requested Enhancements & New Revised Plan
Our revised plan adds interactivity, advanced analysis, and a more robust user interface. It subdivides into several major areas:

A. Enhanced User Controls & Processing Options
Interactive Scale Calibration

Drawable Canvas: • Add a canvas that lets the user draw a line over a visible scale bar (or over a separately uploaded “scale image”).

Known Length Entry: • Provide a text input—supporting three decimals—for the actual (µm) length of the drawn line.

Automatic Scale Computation: • Compute and display the conversion factor (µm per pixel) that can be used automatically, with the option to override manually.

Expanded Preprocessing and Thresholding Options

Preprocessing Controls: • Offer toggles for Gaussian blur (with adjustable kernel size), image inversion, and cropping (smart trimming at the bottom).

Thresholding Methods: • Include a drop‑down with multiple options: “None”, “Fixed” (with a slider for threshold value), “Adaptive” (with adjustable block size and constant), and “Otsu”.

Selectable Analysis Mode (SEM)

Full Image vs. Line Profile Mode: • In Full Image mode, the tool extracts a complete vertical profile from each feature. • In Line Profile mode, the app extracts two separate profiles: a narrow vertical profile for groove-depth/roughness and a horizontal profile for measuring linewidth.

Transfer to Profilometer: • Allow extraction of a SEM line profile that can be later transferred to the Profilometer tab.

B. Advanced Feature Detection and Feature-Type Specific Analysis
Improved Contour & Boundary Extraction

Instead of only drawing rectangular bounding boxes, use contour approximation (cv2.approxPolyDP) to obtain a polygonal outline that more accurately represents the feature shape.

Feature Types & Their Analysis

Supported Feature Types: • Dots: Small, round features measured with high circularity and solidity criteria. • Array: Features arranged in a regular grid, where spacing might be considered later. • Ellipse: For elongated shapes, use ellipse fitting (cv2.fitEllipse) to extract orientation and precise aspect ratio. • Gratings: Apply FFT‑based pitch detection on the horizontal intensity profile to compute grating pitch (spacing). • Arbitrary: Use the default contour‑based approach for features that don’t fit into the above categories.

The user’s selection of feature type will adjust which metrics are computed (e.g., orientation only for Ellipse analysis, pitch for Gratings) and will determine the default values for filtering.

C. Interactive Filtering and Feature Management
Real-Time, Granular Filtering Control

Individual Filters: • For Area, Width, Height, Aspect Ratio, Solidity, and Circularity—and Orientation (for Ellipse features)—provide sliders (with double‑click to edit values) in a sidebar. • Each filter has its own checkbox so users can activate or disable that particular filter.

Dynamic Feedback: • The filtering criteria are applied in real time. If no features remain after filtering, the app warns the user and falls back to showing all detected features.

Manual Removal: • In addition to slider-based filtering, offer a multiselect widget for the user to manually remove specific features from the DataFrame.

Interactive Deletion via Image Overlays

Clickable Overlays: • Use an interactive Plotly figure to display the SEM image with overlaid polygon outlines (from the contour approximation) and feature index annotations. • The user can click on the annotations (or outlines) to “delete” unwanted features directly, with the corresponding rows being removed from the results DataFrame immediately.

Proper Boundary Representation: • The overlay will now draw the boundaries as polygons rather than simple rectangles so that the feature shape is more accurately represented.

D. Enhanced Profilometer Analysis
Interactive Profile Visualization

Display raw and smoothed line profiles using Plotly charts for their built‑in zoom, pan, and hover capabilities.

Draggable Vertical Reference Markers

Provide interactive vertical reference lines (implemented via sliders and/or Plotly annotations) on the profilometer trace.

As the user moves these markers, automatically recompute region‑based metrics: • The height difference (delta z) between the highest and lowest points in the selected region. • The slope between the nearest local maximum and minimum. • The corresponding angle (via an arctan calculation), mimicking a dektak profilometer’s behavior.

SEMs as Input for Profilometer

Allow the user to import an extracted SEM line profile (from the SEM tab’s “Line Profile” mode) to be further analyzed in the Profilometer section.

E. Export and Data Handling
CSV Export:

Export each image’s results (or the combined batch results) as CSV files.

Optional Session Export:

Export the session configuration (thresholding choices, filtering settings, scale factor, etc.) as JSON for reproducibility if desired.

F. Comprehensive Tutorial and Documentation
Detailed Workflow Documentation:

The Tutorial tab will explain each step in the process:

How to calibrate scale by drawing over the scale bar and entering the known length.

How to select preprocessing options and thresholding methods.

How to choose between Full Image and Line Profile modes and what that means.

What each feature type represents: Dots, Array, Ellipse, Gratings, Arbitrary.

How to adjust filtering settings and remove features interactively.

How to set vertical reference markers on the profilometer trace and measure parameters such as delta height, slope, and angle.

Usage Examples:

The Tutorial includes examples (or references to example images) that show expected outcomes—for instance, how to interpret groove and grating line analyses—and provides troubleshooting guidance (e.g., what to do if “No features met filtering criteria” appears).

3. Overall Final Deliverable
The final system will be a deploy‑ready, integrated metrology tool with:

Robust File Handling: Seamless upload of SEM images and profilometer traces.

Interactive Preprocessing & Scale Calibration: Drawing on a canvas, specifying a known length, and applying multiple thresholding methods.

Advanced, Feature‑Type Specific Analysis: Tailored computation for Dots, Array, Ellipse (with orientation), Gratings (with FFT pitch detection), and Arbitrary features.

Real-Time, Interactive Filtering & Feature Management: With sliders, individual checkboxes for filters, and clickable image overlays to remove unwanted features.

Enhanced Profilometer Analysis: Interactive Plotly traces, draggable vertical markers for defining measurement regions, and computed metrics such as height difference, slope, and angle.

Flexible Data Export: CSV outputs for feature data and optionally exporting session settings.

In‑Depth Tutorial: A comprehensive help section guiding the user through the entire workflow and explaining every control and computed metric.



================================
Comments
================================
I. Original Vision & Core Functionality
File Upload & Basic Processing

SEM Images: • Upload one or multiple SEM images (with options for batch mode). • Options to choose a full image or manually select an ROI.

Profilometer Traces: • Upload trace files (CSV, XLS, TXT) for stylus profilometer analysis.

Basic Measurements: • For each detected feature (via contour detection), compute metrics such as Area, Width, Height, Aspect Ratio, and (initially) Groove Depth, Ra (roughness from top & bottom), and Sidewall Angle.

Overlay and Export

Display processed images with overlaid bounding boxes (or contours) and annotated feature indices.

Provide an export of results as CSV (and a basic tutorial/help section).

II. Enhancements & Requested Updates (New Revised Plan)
A. Interactive User Controls and Processing Options
Scale Measurement Improvements:

Interactive Calibration: • Use a drawable canvas (via a library such as st-drawable-canvas) so the user can draw a line directly over the visible (embedded) scale bar or an optionally uploaded “scale image.” • Provide a text input (supporting three decimal places) for the known real‑world length (in µm). • Compute and display the conversion factor (µm per pixel), with the option to override manually.

Preprocessing & Thresholding:

Multiple Thresholding Methods: • Offer a drop‑down with choices: “None”, “Fixed”, “Adaptive”, and “Otsu”. • If “Fixed” is selected, let the user adjust a fixed threshold value with a slider; if “Adaptive” is selected, let them adjust block size and constant.

Additional Preprocessing: • Options for Gaussian blur (with adjustable kernel size) and inversion.

Selectable Analysis Modes:

Full Image vs. Line Profile: • Allow the user to choose between extracting the full vertical profile of each detected feature or a “Line Profile” that separately captures a narrow vertical profile (for groove depth/roughness) and a horizontal profile (for linewidth measurement). • Offer an option (a button) to extract a SEM line profile that can then be transferred to the Profilometer tab for further analysis.

B. Advanced Feature Detection and Analysis
Feature-Type Specific Analysis:

Multiple Feature Types: • Support “Dots,” “Array,” “Ellipse,” “Gratings,” and “Arbitrary.” • Dots, Array, and Arbitrary use standard contour‐based metrics. • For Ellipse features, apply ellipse fitting (cv2.fitEllipse) to extract orientation and more precise aspect ratio. • For Gratings, incorporate FFT‑based pitch detection to compute the spacing (pitch) from the horizontal intensity profile.

Accurate Boundary Extraction: • Use contour approximation (e.g., cv2.approxPolyDP) so that features are outlined by polygons rather than simple rectangular bounding boxes.

Advanced Measurement Computations:

Ensure that “Ra Top” and “Ra Bottom” are computed from the correct portions of the vertical profile (for example, in a well‑processed image, one area corresponds to the groove and one to the grating line).

Different metrics (e.g., measured widths, heights) are computed based on the analysis mode.

C. Interactive Filtering and Feature Management
Filtering Controls

Individual Criteria with Activation Toggles: • In the sidebar, provide sliders (which are double‑click editable) and number inputs for filtering by Area, Width, Height, Aspect Ratio, Solidity, and Circularity. • Additionally, for Elliptical features include Orientation filters. • Each filter comes with its own checkbox so that the user can activate or deactivate it.

Dynamic Feedback: • Filtering adjustments are applied in real time; if no features pass the filters, a warning is issued and detected features are shown as a fallback.

Interactive Deletion:

Clickable Overlays: • Over the SEM image, display the detected features using an interactive Plotly figure that draws polygons (based on the contour approximation) along with clearly visible index annotations. • Allow the user to click on a feature’s annotation (or its outline) to remove that feature from the final DataFrame automatically.

Manual Removal: • Also provide a multiselect widget to manually remove features by index.

D. Profilometer Tab Enhancements
Interactive Trace Visualization:

Display raw and smoothed profilometer traces as Plotly charts with built‑in zoom and pan.

Draggable Reference Markers:

Add interactive vertical reference lines (implemented via sliders or Plotly annotations) so that the user can define a region for detailed analysis.

When the reference markers are adjusted, automatically calculate and display metrics for that region, including the vertical height difference (delta z), the slope, and the corresponding angle (mimicking a dektak profilometer).

SEMs to Profilometer Transfer:

Allow extracting a line profile from an SEM image (in the “Line Profile” mode) that can be seamlessly transferred to the Profilometer tab for further measurement.

E. Export & Session Handling
CSV Export:

Export individual image’s feature data as CSV files and allow batch combined CSV export.

Session Configuration:

Optionally export the session configuration as JSON for reproducibility.

F. Expanded Tutorial & Documentation
Detailed Instructions:

The Tutorial tab will include step‑by‑step instructions covering the entire workflow—from scale calibration and image preprocessing to feature detection, filtering, interactive deletion, and profilometer analysis.

Clear definitions and examples of each feature type (Dots, Array, Ellipse, Gratings, Arbitrary) will be provided.

Usage Examples:

Describe how to calibrate scale, choose preprocessing options, adjust thresholds, delete unwanted features, and transfer SEM line profiles to the Profilometer section.

Explain how the vertical reference lines in the profilometer work to compute height differences, slopes, and angles.

III. Summary
The new revised plan builds on the original core functionality and expands it with:

Fully interactive scale calibration using a drawable canvas and known-length input.

Expanded thresholding options (Fixed, Adaptive, Otsu, None) along with full preprocessing controls.

Selectable analysis modes for either a full-image profile or a dedicated line profile extraction.

Comprehensive feature detection that adapts based on the chosen feature type (Dots, Array, Ellipse, Gratings, and Arbitrary) with specialized processing (ellipse fitting and FFT for gratings).

Interactive filtering with individual activation checkboxes and editable slider/numerical inputs.

Interactive feature deletion from image overlays via clickable annotations.

Enhanced Profilometer tab featuring interactive Plotly charts with draggable vertical reference markers that compute additional metrics (height difference, slope, angle).

Data export functionality (CSV for results, JSON for session configuration) and an updated comprehensive Tutorial tab.

This plan incorporates all feedback provided since yesterday and earlier, ensuring that the end product is a seamless, user‑friendly, lab‑grade metrology tool ready for deployment.


==============================================================
Code below
===========================================================
# pip install streamlit plotly opencv-python-headless numpy pandas scipy streamlit-drawable-canvas streamlit-plotly-events pillow


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

# For capturing Plotly click events
from streamlit_plotly_events import plotly_events
# For drawable canvas (scale calibration)
from streamlit_drawable_canvas import st_canvas

#############################################
# SETTINGS & PAGE CONFIGURATION
#############################################
st.set_page_config(page_title="SEM & Profilometer Analyzer", layout="wide")

#############################################
# UTILITY FUNCTIONS: PROFILOMETER ANALYSIS
#############################################
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

#############################################
# UTILITY FUNCTIONS: SEM IMAGE ANALYSIS
#############################################
def decode_image(uploaded_file):
    # Read raw, then decode via OpenCV
    bytes_data = np.asarray(bytearray(uploaded_file.read()), dtype=np.uint8)
    return cv2.imdecode(bytes_data, cv2.IMREAD_GRAYSCALE)

def apply_roi(img, x, y, w, h):
    return img[y:y+h, x:x+w]

def detect_contours(img):
    # Preprocessing for contour detection
    blur = cv2.GaussianBlur(img, (5,5), 0)
    _, thresh = cv2.threshold(blur, 0, 255, cv2.THRESH_BINARY+cv2.THRESH_OTSU)
    contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    return contours

#############################################
# ANALYZE CONTOUR (with full and line modes, and feature-type switches)
#############################################
def analyze_contour(c, scale, gray_img, grayscale_to_micron=None, analysis_mode="full", feature_type="Arbitrary"):
    # Compute area in physical units:
    area = cv2.contourArea(c) * (scale**2)
    x, y, w, h = cv2.boundingRect(c)
    aspect_ratio = round(w/h,3) if h>0 else 0
    # Instead of simple rectangle, compute contour polygon approximation:
    approx = cv2.approxPolyDP(c, 0.01*cv2.arcLength(c,True), True)
    
    # For ellipse features, try fitting an ellipse:
    ellipse_fit = None
    if feature_type=="Ellipse" and len(c) >= 5:
        try:
            ellipse_fit = cv2.fitEllipse(c)
        except:
            ellipse_fit = None

    if analysis_mode=="full":
        # Use vertical profile from the cropped region:
        cropped = gray_img[y:y+h, x:x+w]
        profile = np.mean(cropped, axis=1)
        L = len(profile)
        win = min(11, L if L%2==1 else L-1)
        smooth_vals = profile if win<3 else scipy.signal.savgol_filter(profile, win, 2)
        norm = (smooth_vals - np.min(smooth_vals))/(np.max(smooth_vals)-np.min(smooth_vals)+1e-6)
        # Define top rough region as where norm >0.85 and bottom as <0.15
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
            # For gratings, also extract horizontal profile and compute FFT pitch
            cropped = gray_img[y:y+h, x:x+w]
            horiz_profile = np.mean(cropped, axis=0)
            pitch = fft_pitch_detection(horiz_profile, scale)
            result["Grating Pitch (µm)"] = round(pitch,3)
        return result

    elif analysis_mode=="line":
        # In line mode, extract a narrow vertical profile and horizontal profile:
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

#############################################
# FFT-BASED PITCH DETECTION (for Gratings)
#############################################
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

#############################################
# PLOTTING FUNCTION: Display image with contour polygons
#############################################
def plot_image_with_polygons(img, contours, selected_indices=[]):
    # We'll draw the approximated polygon for each contour.
    fig = go.Figure()
    # Convert img to base64 and add as background:
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
    # For each contour, compute polygon string
    for idx, c in enumerate(contours):
        if idx in selected_indices:
            continue
        approx = cv2.approxPolyDP(c, 0.01*cv2.arcLength(c,True), True)
        # Build path string: M x,y L x,y ... Z
        path = ""
        for pt in approx:
            x_pt, y_pt = pt[0]
            # Flip y coordinate (Plotly coordinate origin bottom-left)
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
        # Add an annotation with the index
        # Using the centroid of the contour:
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

#############################################
# REPORT & EXPORT FUNCTIONS
#############################################
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

#############################################
# GLOBAL: To hold extracted SEM line profile for transport to Profilometer
#############################################
if "sem_line_profile" not in st.session_state:
    st.session_state.sem_line_profile = None

#############################################
# MAIN UI: TABS
#############################################
tab1, tab2, tab3 = st.tabs(["📷 SEM Image", "📏 Profilometer", "🧠 Tutorial"])

############################
# SEM IMAGE TAB
############################
with tab1:
    st.header("SEM Image Analysis")
    
    # --- Scale Measurement Tool ---
    st.subheader("Scale Measurement Tool")
    st.markdown("If your SEM image contains a scale bar, use the canvas below. (The uploaded scale image will appear as background if available.)")
    # Optionally, user can upload a separate scale image:
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
    
    # --- File Upload for SEM Images ---
    master_batch = st.checkbox("Enable Batch Mode (multiple SEM images)", value=False)
    if master_batch:
        files = st.file_uploader("Upload SEM images", type=["png","jpg","jpeg","tif","tiff"], accept_multiple_files=True, key="sem_files")
    else:
        file = st.file_uploader("Upload SEM image", type=["png","jpg","jpeg","tif","tiff"], key="sem_file")
        files = [file] if file is not None else []
    
    # Proceed if files are uploaded:
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
            # Otsu thresholding will be done automatically.
        else:
            do_clahe = do_blur = do_inversion = False
            threshold_option = "None"
    
        trim_bottom = st.number_input("Smart image trimming: Crop bottom (px)", min_value=0, value=0)
        use_fft = st.checkbox("Use FFT-based pitch detection (for Gratings)", value=False)
    
        # Scale: use computed scale to 3 decimals or manual input
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
    
        # For Gratings, if selected:
        if use_fft and feature_type=="Gratings":
            fft_placeholder = st.empty()
    
        # FILTERING SETTINGS with individual toggles and editable numbers
        st.sidebar.subheader("Feature Filtering Settings")
        apply_filtering = st.sidebar.checkbox("Activate Filtering", value=True)
        use_area_filter = st.sidebar.checkbox("Area Filter", value=True)
        use_width_filter = st.sidebar.checkbox("Width Filter", value=True)
        use_height_filter = st.sidebar.checkbox("Height Filter", value=True)
        use_aspect_filter = st.sidebar.checkbox("Aspect Ratio Filter", value=True)
        use_solidity_filter = st.sidebar.checkbox("Solidity Filter", value=True)
        use_circularity_filter = st.sidebar.checkbox("Circularity Filter", value=True)
        use_orientation_filter = st.sidebar.checkbox("Orientation Filter (for Ellipses)", value=(feature_type=="Ellipse"))
    
        # Using number input (editable) for each filter threshold:
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
    
        # Process each uploaded image:
        for uploaded_file in files:
            if uploaded_file is None:
                continue
            img = decode_image(uploaded_file)
            # Preprocessing
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
    
            # ROI Selection:
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
    
            # Detect contours:
            contours = detect_contours(img_proc)
            if not contours:
                st.warning(f"No contours detected in {uploaded_file.name}.")
                continue
            output = []
            boxes = []
            for idx, c in enumerate(contours):
                metrics = analyze_contour(c, scale, img_proc, grayscale_to_micron, analysis_mode, feature_type)
                output.append(metrics)
                # Use the approximated polygon bounding box (fallback to rectangle if needed)
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
    
            # Filtering: If filtering yields empty list, fallback to updated_output.
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
            # Allow manual deletion via multiselect:
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
    
            # Interactive Ra visualization:
            ra_bottom_col = "Ra Bottom (" + ("µm" if grayscale_to_micron is not None else "") + ")"
            ra_top_col = "Ra Top (" + ("µm" if grayscale_to_micron is not None else "") + ")"
            if ra_bottom_col in df.columns and ra_top_col in df.columns:
                fig_bar = px.bar(df, x=df.index, y=[ra_bottom_col, ra_top_col],
                                 barmode="group",
                                 labels={"value": "Roughness (Ra)", "variable": "Region"},
                                 title="Roughness (Ra) at Top vs Bottom")
                st.plotly_chart(fig_bar, use_container_width=True)
    
            # Button to export CSV of this image's features:
            csv_data = export_csv(df)
            st.download_button("Export CSV for This Image", csv_data, uploaded_file.name+"_features.csv", "text/csv")
    
            # Optionally, allow extracting a line profile from SEM to transfer to the Profilometer tab:
            if analysis_mode=="line":
                if st.button(f"Extract SEM Line Profile from {uploaded_file.name}"):
                    # Save the average vertical profile for this image to session_state:
                    st.session_state.sem_line_profile = np.mean(apply_roi(img, 0, 0, img.shape[1], img.shape[0]), axis=1)
                    st.success("Line profile extracted and stored for Profilometer analysis.")
    
        # Combined batch export:
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
        
###############################
# STYLUS PROFILOMETER TAB
###############################
with tab2:
    st.header("Stylus Profilometer Analysis")
    up = st.file_uploader("Upload profilometer trace (.csv, .xlsx, .txt)", type=["csv", "txt", "xls", "xlsx"], key="prof_file")
    if up:
        dfp = parse_profilometer(up)
        if dfp is not None and dfp.shape[1]>=2:
            result, x, y, y_smooth = extract_profile_metrics(dfp)
            st.subheader("Profilometer Metrics from File")
            st.json(result)
            fig_prof = px.line(x=x, y=y, labels={"x": "Lateral (µm)", "y": "Height (µm)"}, title="Raw Profilometer Trace")
            st.plotly_chart(fig_prof, use_container_width=True)
            fig_prof2 = px.line(x=x, y=y_smooth, labels={"x": "Lateral (µm)", "y": "Height (µm)"}, title="Smoothed Profilometer Trace")
            st.plotly_chart(fig_prof2, use_container_width=True)
    
            st.markdown("**Define vertical reference lines for measurement:**")
            ref_left = st.slider("Left Reference (µm)", min_value=float(min(x)), max_value=float(max(x)), value=float(min(x)))
            ref_right = st.slider("Right Reference (µm)", min_value=float(min(x)), max_value=float(max(x)), value=float(max(x)))
            st.write(f"Analysis Region: [{ref_left:.2f}, {ref_right:.2f}] µm")
            # Extract region of interest from the smoothed profile:
            ind_left = np.searchsorted(x, ref_left)
            ind_right = np.searchsorted(x, ref_right)
            region_x = x[ind_left:ind_right] if ind_right>ind_left else x
            region_y = y_smooth[ind_left:ind_right] if ind_right>ind_left else y_smooth
            if len(region_y)>0:
                # Compute measurement metrics: delta z, slope, angle (simple linear fit)
                delta_z = np.max(region_y) - np.min(region_y)
                slope = np.polyfit(region_x, region_y, 1)[0]
                angle = np.degrees(np.arctan(slope))
                st.write(f"Delta Height: {delta_z:.3f} µm, Slope: {slope:.3f} µm/µm, Angle: {angle:.1f}°")
            # Optionally, if an SEM line profile was extracted, show it here:
            if st.session_state.sem_line_profile is not None:
                st.subheader("Imported SEM Line Profile")
                region_prof = st.session_state.sem_line_profile
                fig_sem_line = px.line(y=region_prof, labels={"y": "Intensity or Height"}, title="Imported SEM Line Profile")
                st.plotly_chart(fig_sem_line, use_container_width=True)
        else:
            st.warning("Invalid profilometer file format.")
    else:
        st.info("Upload a profilometer trace file for analysis.")
        
#######################
# TUTORIAL TAB
#######################
with tab3:
    st.markdown("""
    ### SEM + Profilometer Analyzer Tutorial

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
    
    Enjoy using this integrated metrology tool!
    """)
