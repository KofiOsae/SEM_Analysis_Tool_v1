import streamlit as st
import numpy as np
import plotly.graph_objs as go
from scipy.signal import find_peaks
from scipy.ndimage import gaussian_filter1d
from PIL import Image
import tifffile as tiff
import os

st.set_page_config(layout="wide")
st.title("Profilometer Analysis")

def load_tiff_image(uploaded_file):
    image = tiff.imread(uploaded_file)
    return image

def calculate_metrics(x, z, m_index, r_index):
    region_x = x[m_index:r_index]
    region_z = z[m_index:r_index]

    delta_z = np.max(region_z) - np.min(region_z)
    delta_z_std = np.std(region_z)

    dz = np.diff(region_z)
    dx = np.diff(region_x)
    slopes = dz / dx
    max_slope = np.max(np.abs(slopes))
    theta = np.degrees(np.arctan(max_slope))
    theta_std = np.std(np.degrees(np.arctan(slopes)))

    peaks, _ = find_peaks(region_z)
    valleys, _ = find_peaks(-region_z)

    max_vals = region_z[peaks] if len(peaks) > 0 else np.array([np.max(region_z)])
    min_vals = region_z[valleys] if len(valleys) > 0 else np.array([np.min(region_z)])

    avg_max = np.mean(max_vals)
    std_max = np.std(max_vals)
    avg_min = np.mean(min_vals)
    std_min = np.std(min_vals)

    top_threshold = avg_max - 0.1 * delta_z
    bottom_threshold = avg_min + 0.1 * delta_z

    top_indices = np.where(region_z >= top_threshold)[0]
    bottom_indices = np.where(region_z <= bottom_threshold)[0]

    if len(top_indices) > 1:
        top_width = region_x[top_indices[-1]] - region_x[top_indices[0]]
        top_width_std = np.std(region_x[top_indices])
        ra_top = np.mean(np.abs(region_z[top_indices] - np.mean(region_z[top_indices])))
        ra_top_std = np.std(np.abs(region_z[top_indices] - np.mean(region_z[top_indices])))
    else:
        top_width = top_width_std = ra_top = ra_top_std = 0

    if len(bottom_indices) > 1:
        bottom_width = region_x[bottom_indices[-1]] - region_x[bottom_indices[0]]
        bottom_width_std = np.std(region_x[bottom_indices])
        ra_bottom = np.mean(np.abs(region_z[bottom_indices] - np.mean(region_z[bottom_indices])))
        ra_bottom_std = np.std(np.abs(region_z[bottom_indices] - np.mean(region_z[bottom_indices])))
    else:
        bottom_width = bottom_width_std = ra_bottom = ra_bottom_std = 0

    return {
        "Δz (avg)": delta_z,
        "Δz (std)": delta_z_std,
        "θ (avg, deg)": theta,
        "θ (std, deg)": theta_std,
        "Avg Max Z": avg_max,
        "Std Max Z": std_max,
        "Avg Min Z": avg_min,
        "Std Min Z": std_min,
        "Top Width (avg)": top_width,
        "Top Width (std)": top_width_std,
        "Bottom Width (avg)": bottom_width,
        "Bottom Width (std)": bottom_width_std,
        "Ra Top (avg)": ra_top,
        "Ra Top (std)": ra_top_std,
        "Ra Bottom (avg)": ra_bottom,
        "Ra Bottom (std)": ra_bottom_std
    }

def plot_profile(x, raw_z, smooth_z, m_pos, r_pos, show_raw, show_smooth):
    fig = go.Figure()
    if show_raw:
        fig.add_trace(go.Scatter(x=x, y=raw_z, mode='lines', name='Raw Profile', line=dict(color='gray')))
    if show_smooth:
        fig.add_trace(go.Scatter(x=x, y=smooth_z, mode='lines', name='Smoothed Profile', line=dict(color='blue')))
    fig.add_vline(x=m_pos, line=dict(color='red', dash='dash'), name='Marker M')
    fig.add_vline(x=r_pos, line=dict(color='green', dash='dash'), name='Marker R')
    fig.update_layout(title='Profilometer Profile',
                      xaxis_title='Lateral (µm)',
                      yaxis_title='Height (µm)',
                      showlegend=True)
    return fig

uploaded_file = st.file_uploader("Upload a .tif or .tiff file", type=["tif", "tiff"])

if uploaded_file:
    image = load_tiff_image(uploaded_file)
    if image.ndim > 2:
        image = image[0]

    profile = np.mean(image, axis=0)
    x = np.arange(len(profile))
    raw_z = profile
    smooth_z = gaussian_filter1d(profile, sigma=2)

    st.subheader("Interactive Profilometer Analysis")

    col1, col2 = st.columns(2)
    with col1:
        m_pos = st.slider("Marker M Position", min_value=int(x[0]), max_value=int(x[-1]), value=int(x[0] + len(x) // 4))
    with col2:
        r_pos = st.slider("Marker R Position", min_value=int(x[0]), max_value=int(x[-1]), value=int(x[0] + 3 * len(x) // 4))

    m_index = np.searchsorted(x, m_pos)
    r_index = np.searchsorted(x, r_pos)
    if m_index > r_index:
        m_index, r_index = r_index, m_index
        m_pos, r_pos = r_pos, m_pos

    show_raw = st.checkbox("Show Raw Profile", value=True)
    show_smooth = st.checkbox("Show Smoothed Profile", value=True)

    metrics = calculate_metrics(x, smooth_z, m_index, r_index)
    fig = plot_profile(x, raw_z, smooth_z, m_pos, r_pos, show_raw, show_smooth)
    st.plotly_chart(fig, use_container_width=True)

    st.subheader("Calculated Metrics")
    for key, value in metrics.items():
        st.write(f"**{key}**: {value:.4f}")
