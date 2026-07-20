#!/usr/bin/env python3

import numpy as np
import napari
import matplotlib.pyplot as plt
from pathlib import Path
import os
import torch
from tqdm import tqdm
from vispy.color import Colormap, Color
from qtpy.QtWidgets import QApplication # Import QApplication to process events
import time


# --- Configuration ---
# Define a dummy volume directory and video directory for demonstration.
# In your actual use, these would point to your data.

# --- Main Processing Loop ---
model_dir = "/home/casap/mlibra/output/DIFFICULT-MANIFOLD-RSAMPLE-5-4-reference-500-1000-anatomical_atlas-1-20.0-0.01-1.010"
image_output_dir = Path(model_dir) / "output_images"
video_dir = image_output_dir
image_output_dir.mkdir(exist_ok=True, parents=True)
#lipids = ['HexCer 42:2;O2', 'PC 38:6', 'SM 36:1;O2', 'PE 40:6', 'PC 36:1', 'PC 32:0', 'PC 34:1 PC 36:4 PE 37:1 PE 39:4', 'PC 34:1', 'PA 34:1', 'PA 36:2']
lipids = ['Hex2Cer 40:1;O2', 'PA 36:1 PA 38:4', 'PC 35:1 PE 38:1']
files = [model_dir + "/{0}_volume.npy".format(s) for s in lipids]
selected_lipid_names = list(np.load("/home/casap/mlibra/mlibra_data/maindata_minimal_available_lipids.npy").astype(list))
BATCH_SIZE = 1000

def create_intensity_colormap():
    """
    Creates a custom colormap for napari with intensity-based transparency.
    Less intense voxels (lower values) will be more transparent.
    More intense voxels (higher values) will be more opaque and colored.
    """
    # Define colors and their corresponding alpha (opacity) values
    # Format: (value_position, R, G, B, A)
    # value_position: 0.0 to 1.0, mapping to data range
    # R, G, B, A: 0.0 to 1.0
    colors = [
        (0.0, [0.0, 0.0, 0.0, 0.0]),  # Fully transparent black at min intensity
        (0.1, [0.0, 0.0, 0.0, 0.05]), # Slightly visible at low intensity
        (0.3, [0.1, 0.2, 0.8, 0.2]),  # Faint blue at medium-low intensity
        (0.6, [0.8, 0.4, 0.1, 0.6]),  # Orange at medium-high intensity
        (1.0, [1.0, 1.0, 0.0, 1.0])   # Opaque yellow at max intensity
    ]
    return Colormap(colors)

for lipid, file in tqdm(zip(lipids, files), desc="Processing lipids"):
    volume = np.load(file)

    # Normalize volume data to 0-1 for consistent colormap application
    volume_min = np.nanmin(volume)
    volume_max = np.nanmax(volume)
    if volume_max - volume_min > 0:
        volume = (volume - volume_min) / (volume_max - volume_min)
    else:
        volume = np.zeros_like(volume) # Handle flat volumes

    # --- 1. Create the napari viewer for the current volume ---
    viewer = napari.Viewer()

    # Add the main volume layer with volume rendering and custom transparency
    main_volume_layer = viewer.add_image(
        volume,
        name=f'{lipid}_intensity',
        colormap='inferno',
        rendering='mip', # Crucial for true volume rendering with transparency
        blending='translucent', # 'additive' or 'translucent' often work well with transparency
        attenuation=0.05, # Adjust this value (e.g., 0.01 to 0.1) for desired transparency
                          # Lower value = more transparent
    )

    # --- 3. Prepare for Screenshots of 3D View Rotation ---
    viewer.dims.ndisplay = 3
    viewer.camera.zoom = 1.8  # Further increased zoom for better visibility
    
    # Capture 10 frames of rotation for the last row
    rotation_frames = []
    for i in range(10):
        # Rotate around y-axis (36 degree increments for a full 360 rotation)
        angle = i * 36
        viewer.camera.angles = (45, angle, 0)  # (elevation, azimuth, roll)
        QApplication.processEvents()
        time.sleep(0.5)
        rotation_screenshot = viewer.screenshot(path=None)
        rotation_frames.append(rotation_screenshot)

    # --- 4. Combine Screenshots into a Multi-Panel Matplotlib Figure ---
    # Create a larger figure with 4 rows (3 for cuts, 1 for rotation) and 10 columns
    # Reduced the overall height to compress the figure vertically
    fig, axs = plt.subplots(nrows=4, ncols=10, figsize=(30, 11))
    fig.suptitle(f"Mice Brain Visualization: {lipid}", fontsize=16)
    
    # Switch to 2D display for slice views
    viewer.dims.ndisplay = 2
    
    # Calculate slice positions for 10 evenly distributed positions
    # For each axis, from 5% to 95% with even spacing
    z_positions = [
        int(volume.shape[0] * 0.05),  # 5%
        int(volume.shape[0] * 0.15),  # 15%
        int(volume.shape[0] * 0.25),  # 25%
        int(volume.shape[0] * 0.35),  # 35%
        int(volume.shape[0] * 0.45),  # 45%
        int(volume.shape[0] * 0.55),  # 55%
        int(volume.shape[0] * 0.65),  # 65%
        int(volume.shape[0] * 0.75),  # 75%
        int(volume.shape[0] * 0.85),  # 85%
        int(volume.shape[0] * 0.95),  # 95%
    ]
    
    y_positions = [
        int(volume.shape[1] * 0.05),  # 5%
        int(volume.shape[1] * 0.15),  # 15%
        int(volume.shape[1] * 0.25),  # 25%
        int(volume.shape[1] * 0.35),  # 35%
        int(volume.shape[1] * 0.45),  # 45%
        int(volume.shape[1] * 0.55),  # 55%
        int(volume.shape[1] * 0.65),  # 65%
        int(volume.shape[1] * 0.75),  # 75%
        int(volume.shape[1] * 0.85),  # 85%
        int(volume.shape[1] * 0.95),  # 95%
    ]
    
    x_positions = [
        int(volume.shape[2] * 0.05),  # 5%
        int(volume.shape[2] * 0.15),  # 15%
        int(volume.shape[2] * 0.25),  # 25%
        int(volume.shape[2] * 0.35),  # 35%
        int(volume.shape[2] * 0.45),  # 45%
        int(volume.shape[2] * 0.55),  # 55%
        int(volume.shape[2] * 0.65),  # 65%
        int(volume.shape[2] * 0.75),  # 75%
        int(volume.shape[2] * 0.85),  # 85%
        int(volume.shape[2] * 0.95),  # 95%
    ]
    
    # Position labels for titles
    position_labels = ["5%", "15%", "25%", "35%", "45%", "55%", "65%", "75%", "85%", "95%"]
    
    # Coronal Cuts (X-Z plane, viewing from front, changing Y)
    for i, y_pos in enumerate(y_positions):
        coronal_data = volume[:, y_pos, :]
        coronal_layer = viewer.add_image(
            coronal_data, 
            name=f'coronal_slice_{i}',
            colormap='inferno',
            visible=True
        )
        main_volume_layer.visible = False  # Hide the volume temporarily
        QApplication.processEvents()
        time.sleep(0.5)
        coronal_screenshot = viewer.screenshot(path=None)
        axs[0, i].imshow(coronal_screenshot)
        axs[0, i].set_title(f"Axial (X-Y) - {position_labels[i]}", fontsize=6)  # Smaller font
        axs[0, i].axis('off')
        viewer.layers.remove(coronal_layer)  # Remove the layer after screenshot
    
    # Axial Cuts (X-Y plane, viewing from top, changing Z)
    for i, z_pos in enumerate(z_positions):
        axial_data = volume[z_pos, :, :]
        axial_layer = viewer.add_image(
            axial_data,
            name=f'axial_slice_{i}',
            colormap='inferno',
            visible=True
        )
        QApplication.processEvents()
        time.sleep(0.5)
        axial_screenshot = viewer.screenshot(path=None)
        axs[1, i].imshow(axial_screenshot)
        axs[1, i].set_title(f"Coronal (X-Z) - {position_labels[i]}", fontsize=6)  # Smaller font
        axs[1, i].axis('off')
        viewer.layers.remove(axial_layer)  # Remove the layer after screenshot
    
    # Sagittal Cuts (Y-Z plane, viewing from side, changing X)
    for i, x_pos in enumerate(x_positions):
        # Get the sagittal slice and flip it horizontally
        sagittal_data = volume[:, :, x_pos]

        sagittal_layer = viewer.add_image(
            sagittal_data,
            name=f'sagittal_slice_{i}',
            colormap='inferno',
            visible=True
        )
        QApplication.processEvents()
        time.sleep(0.5)
        sagittal_screenshot = viewer.screenshot(path=None)
        axs[2, i].imshow(sagittal_screenshot)
        axs[2, i].set_title(f"Sagittal (Y-Z) - {position_labels[i]}", fontsize=6)  # Smaller font
        axs[2, i].axis('off')
        viewer.layers.remove(sagittal_layer)  # Remove the layer after screenshot
    
    # Restore the main volume visibility
    main_volume_layer.visible = True
    
    # Add the rotation frames to the last row
    for i in range(10):
        axs[3, i].imshow(rotation_frames[i])
        angle = i * 36
        axs[3, i].set_title(f"3D Rotation - {angle}°", fontsize=6)  # Smaller font
        axs[3, i].axis('off')

    # Even more aggressive whitespace reduction
    plt.subplots_adjust(wspace=0.01, hspace=0.01)  # Minimal spacing
    
    # We skip tight_layout since it can sometimes add padding
    # Instead, directly set the figure margins
    plt.subplots_adjust(left=0.01, right=0.99, top=0.95, bottom=0.01)
    
    plt.savefig(image_output_dir / f"{lipid}_multi_panel.png", dpi=300, bbox_inches='tight')
    plt.close(fig) # Close the matplotlib figure to free memory
    viewer.close() # Close the napari viewer after taking all screenshots for this volume
def void():

    print(f"Generated multi-panel image for {lipid}")
    # end script for the moment we do not need to save the images

    # --- Original Animation Code (kept separate as requested) ---
    # Re-open viewer for animation if desired, as the previous one was closed.
    # This part can be commented out if you only need the static images.
    viewer_animation = napari.Viewer()
    layer_animation = viewer_animation.add_image(
        volume,
        name=f'{lipid}_animation',
        colormap='inferno', # Use the same custom colormap
        rendering='mip',
        blending='translucent',
        attenuation=0.05
    )
    viewer_animation.dims.ndisplay = 3
    viewer_animation.camera.zoom = 0.7

    # Ensure napari-animation is installed: pip install napari-animation
    try:
        from napari_animation import Animation
        animation = Animation(viewer_animation)

        total_frames = 60
        for i in range(total_frames):
            angle = i * (360 / total_frames)
            # Rotate around the Z-axis (azimuth)
            viewer_animation.camera.angles = (30, angle, 30)
            animation.capture_keyframe()

        animation.animate(video_dir / f"{str(lipid)}_brain_rotation.mp4", canvas_only=True, fps=30)
        print(f"Generated rotation video for {lipid}")
    except ImportError:
        print("napari-animation not found. Skipping video generation.")
        print("Install with: pip install napari-animation")
    finally:
        viewer_animation.close()

print("\nProcessing complete. Check 'output_images' and 'output_videos' directories.")
