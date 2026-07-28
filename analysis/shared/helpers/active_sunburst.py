# dash_app_package.py
"""
Creates and returns a interactable sunburst plot and image display for tree structured data and brain images.

Parameters:
    - data_by_region: DataFrame which contains linking sunburst nodes to Z-slices. 
        This should contain atlas ontology information. Essential columns are 'id','parent_id','acronym','parent_acronym'. 
        The dataframe should be able to construct one hiearchical tree. 
        The dataframe can have data variables. This can be cell density, results of statistical analysis etc...
        The data variables can be specified by data_variable later.
        
    - data_by_img: 3D numpy array with float values. This can be effect size, normalized density etc.
    - annotated_atlas_img: 3D numpy array of the atlas with annotations.
    - outputpath: Directory to save exported figures.
    - atlas_df: DataFrame containing region information (with 'id' and 'name' columns). # not used anymore
    - colormap: Colormap used for image processing (default: plt.cm.coolwarm).
    - tree_node_names: Column name for the sunburst slice labels (default: 'acronym').
    - tree_node_parents: Column name for the sunburst parent relationships (default: 'parent_acronym').
    - data_variable: Column name for coloring the sunburst (default: 'rejected').
    - sunburst_continuous_scale: Colorscale list for continuous color. If None, defaults to viridis.
    - sunburst_range_color: Two-element list [vmin, vmax] to set the range for continuous color.
    
Returns:
    - app: A Dash app instance.
"""
import os
import math
import dash
from dash import dcc, html
import plotly.express as px
import plotly.graph_objects as go
from dash.dependencies import Input, Output, State
import plotly.io as pio  # For exporting figures as HTML
import numpy as np
import pandas as pd
import cv2  # Ensure OpenCV is installed
import matplotlib.pyplot as plt
from matplotlib.colors import Normalize
import matplotlib.colors as mcolors
import atlas_support_function  # custom module with get_subregions

# ------------------------------
# Helper Function: Convert Matplotlib cmap to Plotly colorscale
# ------------------------------

def mpl_to_plotly(cmap, n_colors=256):
    """
    Convert a Matplotlib colormap to a Plotly colorscale.
    
    Parameters:
      - cmap: a Matplotlib colormap.
      - n_colors: number of color steps.
      
    Returns:
      - A list of hex color strings.
    """
    return [mcolors.to_hex(cmap(i/n_colors)) for i in range(n_colors)]

# ------------------------------
# Image Processing Functions
# ------------------------------

def adjust_intensity(image):
    """Normalize image intensity to the [0,1] range."""
    return (image - np.min(image)) / (np.max(image) - np.min(image) + 1e-8)

def convert_to_cmap(image, cmin=-2, cmax=2, colormap=plt.cm.coolwarm):
    """
    Convert a grayscale image to an RGB image using a colormap.
    Returns an RGB image (without the alpha channel) with values in [0,1].
    """
    norm = Normalize(vmin=cmin, vmax=cmax)
    return colormap(norm(image))[:, :, :3]

def set_transparency(rgba_img, mask):
    """
    Applies a transparency mask to an existing RGBA image.

    Parameters:
    - rgba_img: np.ndarray of shape (H, W, 4), dtype uint8
        The input RGBA image.
    - mask: np.ndarray of shape (H, W), dtype bool
        Boolean mask where True means the pixel should be transparent.

    Returns:
    - np.ndarray of shape (H, W, 4), modified RGBA image.
    """
    if rgba_img.shape[-1] != 4:
        raise ValueError("Input image must be RGBA (shape must be H x W x 4).")
    if rgba_img.shape[:2] != mask.shape:
        raise ValueError("Mask shape must match image height and width.")

    # Copy to avoid modifying the original
    result = rgba_img.copy()
    result[mask, 3] = 0  # Set alpha to 0 (transparent) where mask is True
    return result

def generate_black_contours(id_slice):
    """
    For every unique region (except background=0), find its contours and
    draw them in black (opaque) on a transparent RGBA overlay.
    """
    h, w = id_slice.shape
    overlay = np.zeros((h, w, 4), dtype=np.uint8)
    unique_ids = np.unique(id_slice)
    for mask_id in unique_ids:
        if mask_id == 0:
            continue
        binary_mask = np.uint8(id_slice == mask_id) * 255
        contours, _ = cv2.findContours(binary_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        cv2.drawContours(overlay, contours, -1, (0, 0, 0, 255), 1)
    return overlay

def is_close_to_yellow(rgba_color, threshold=10):
    """
    Checks if an RGBA color is close to yellow.
    
    Args:
      - rgba_color: A tuple (r, g, b, a) with values in [0, 255].
      - threshold: Tolerance threshold.
      
    Returns:
      - True if the color is close to yellow, False otherwise.
    """
    r, g, b, _ = rgba_color
    yellow = (255, 255, 0, 255)
    y_r, y_g, y_b, _ = yellow
    distance = math.sqrt((r - y_r)**2 + (g - y_g)**2 + (b - y_b)**2)
    return distance <= threshold

def generate_highlight_contours(id_slice, IDs, colormap):
    """
    For each region in IDs, create a binary mask, find its contours, and
    draw them in yellow (RGB: 255,255,0) with full opacity. If the maximum color 
    in the colormap is close to yellow, draw in white instead.
    """
    if is_close_to_yellow(colormap(1.0, bytes=True)):
        highlight_color = (255, 255, 0, 255)
    else:
        highlight_color = (255, 255, 0, 255)
    h, w = id_slice.shape
    overlay = np.zeros((h, w, 4), dtype=np.uint8)
    for mask_id in IDs:
        binary_mask = np.uint8(id_slice == mask_id) * 255
        contours, _ = cv2.findContours(binary_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        cv2.drawContours(overlay, contours, -1, highlight_color, 2)
    return overlay

def color_code_image(image_slice, id_slice=None, IDs=None, cmin=-2, cmax=2, colormap=plt.cm.coolwarm):
    """
    Create an RGBA image from image_slice using the specified colormap.
    Overlays black contours for all regions and highlighted contours for the selected IDs.
    """
    if id_slice is None:
        id_slice = image_slice
    base_color = convert_to_cmap(image_slice, cmin=cmin, cmax=cmax, colormap=colormap)
    h, w, _ = base_color.shape
    base_uint8 = (base_color * 255).astype(np.uint8)
    base_rgba = np.dstack([base_uint8, np.full((h, w), 255, dtype=np.uint8)])
    black_overlay = generate_black_contours(id_slice)
    yellow_overlay = generate_highlight_contours(id_slice, IDs, colormap)
    final_image = base_rgba.copy()
    mask_black = black_overlay[..., 3] > 0
    final_image[mask_black] = black_overlay[mask_black]
    mask_yellow = yellow_overlay[..., 3] > 0
    final_image[mask_yellow] = yellow_overlay[mask_yellow]

    # Set transparency where annotated_atlas_img == 0
    if id_slice is not None:    
        transparency_mask = id_slice == 0
        final_image = set_transparency(final_image, transparency_mask)

    return final_image

# ------------------------------
# Utility: Convert Boolean/Object to Categorical
# ------------------------------

def convert_data_by_region_boolean_to_integer(data_by_region, column='rejected'):
    """
    If the target column is boolean or object, convert its values to integers
    and then to a categorical type with categories [-1, 0, 1].
    """
    if data_by_region[column].dtype == bool:
        data_by_region[column] = data_by_region[column].astype(int)
        data_by_region[column] = pd.Categorical(data_by_region[column], categories=[-1, 0, 1])
    elif data_by_region[column].dtype == object:
        data_by_region[column] = data_by_region[column].replace(False, 0).replace(True, 1).replace(np.nan, -1)
        data_by_region[column] = pd.Categorical(data_by_region[column], categories=[-1, 0, 1])
    return data_by_region 

# ------------------------------
# Dash App Creation Function
# ------------------------------

def create_dash_app(data_by_region, data_by_img, annotated_atlas_img, outputpath,
                    colormap=plt.cm.coolwarm,
                    tree_node_names='acronym',
                    tree_node_parents='parent_acronym',
                    data_variable='rejected',
                    sunburst_continuous_scale=None,
                    sunburst_range_color=None,**kwargs):
    """
    Creates and returns a Dash app.
    
    Parameters:
      - data_by_region: DataFrame linking sunburst nodes to Z-slices.
      - data: 3D numpy array of effect sizes.
      - annotated_atlas_img: 3D numpy array of the atlas with annotations.
      - outputpath: Directory to save exported figures.
      - atlas_df: DataFrame containing region information (with 'id' and 'name' columns).
      - colormap: Colormap used for image processing (default: plt.cm.coolwarm).
      
      - tree_node_names: Column name for the sunburst slice labels (default: 'acronym').
      - tree_node_parents: Column name for the sunburst parent relationships (default: 'parent_acronym').
      - data_variable: Column name for coloring the sunburst (default: 'rejected').
      - sunburst_continuous_scale: Colorscale list for continuous color. If None, defaults to viridis.
      - sunburst_range_color: Two-element list [vmin, vmax] to set the range for continuous color.
      
    Returns:
      - app: A Dash app instance.
    """
    if not os.path.exists(outputpath):
        os.makedirs(outputpath)
    if sunburst_continuous_scale is None:
        sunburst_continuous_scale = mpl_to_plotly(plt.cm.viridis)
    data_by_region = convert_data_by_region_boolean_to_integer(data_by_region, column=data_variable)
    if data_by_region[data_variable].dtype != 'category':
        fig_sunburst = px.sunburst(
            data_by_region,
            names=tree_node_names,
            parents=tree_node_parents,
            color=data_variable,
            color_continuous_scale=sunburst_continuous_scale,
            range_color=sunburst_range_color,
            hover_data={tree_node_names: True, 'name': True},
        )
        fig_sunburst.update_layout(
            coloraxis_colorbar=dict(
                thickness=10,
                len=0.5
            )
        )
    else:
        fig_sunburst = px.sunburst(
            data_by_region,
            names=tree_node_names,
            parents=tree_node_parents,
            color=data_variable,
            color_discrete_map={-1: 'white', 1: 'magenta', 0: 'grey'},
            hover_data={tree_node_names: True, 'name': True},
        )
    app = dash.Dash(__name__)

    # Layout:
    # We place the Z-index control buttons in an absolute overlay inside the image container,
    # positioned near the top (below the title). The Save buttons remain as an absolute overlay at the bottom-right.
    app.layout = html.Div([
        html.Div([
            dcc.Graph(
                id='sunburst-plot', 
                figure=fig_sunburst, 
                style={'width': '50%', 'display': 'inline-block'}
            ),
            html.Div([
                html.Div(
                    children=[dcc.Graph(id='image-plot', style={'width': '100%'})],
                    style={'position': 'relative', 'width': '100%'}
                ),
                # Z-index buttons overlay: placed inside the image container, below the title.
                html.Div(
                    children=[
                        html.Button("Z -10", id="z-minus-10", n_clicks=0),
                        html.Button("Z -1", id="z-minus-1", n_clicks=0),
                        html.Button("Z +1", id="z-plus-1", n_clicks=0),
                        html.Button("Z +10", id="z-plus-10", n_clicks=0)
                    ],
                    style={
                        'position': 'absolute',
                        'top': '50px',  # Adjust this value as needed to appear just below the title
                        'left': '150px',
                        'backgroundColor': 'rgba(0,0,0,0.5)',
                        'padding': '5px',
                        'borderRadius': '5px'
                    }
                ),
                # Save buttons overlay at bottom-right.
                html.Div(
                    children=[
                        html.Button("Save as PNG", id="export-png-button", n_clicks=0),
                        html.Button("Save as PDF", id="export-pdf-button", n_clicks=0),
                        html.Button("Export as HTML", id="export-html-button", n_clicks=0)
                    ],
                    style={
                        'position': 'absolute',
                        'bottom': '20px',
                        'right': '90px',
                        'backgroundColor': 'rgba(0,0,0,0.5)',
                        'padding': '5px',
                        'borderRadius': '5px'
                    }
                )
            ], style={'width': '50%', 'display': 'inline-block', 'verticalAlign': 'top', 'position': 'relative'})
        ]),
        html.Div(id='save-status',
                 style={'textAlign': 'center', 'fontSize': '14px', 'color': 'green',
                        'padding': '5px', 'marginTop': '5px'}),
        dcc.Store(id='region_info_store')
    ])

    # -------------------------------------------------------
    # Callback: Update the Image Plot and Region Info (including z-index)
    # -------------------------------------------------------
    @app.callback(
        [Output('image-plot', 'figure'),
         Output('region_info_store', 'data')],
        [Input('sunburst-plot', 'clickData'),
         Input('z-minus-10', 'n_clicks'),
         Input('z-minus-1', 'n_clicks'),
         Input('z-plus-1', 'n_clicks'),
         Input('z-plus-10', 'n_clicks')],
        State('region_info_store', 'data')
    )
    def update_image(clickData, n_zm10, n_zm1, n_zp1, n_zp10, region_info,cmin = -7.5,cmax = 7.5):
        ctx = dash.callback_context
        triggered_id = ctx.triggered[0]['prop_id'].split('.')[0] if ctx.triggered else None

        if triggered_id == 'sunburst-plot' and clickData is not None:
            selected_acronym = clickData['points'][0].get('label')
            selected_row = data_by_region.loc[data_by_region[tree_node_names] == selected_acronym]
            if selected_row.empty:
                return go.Figure(), None
            selected_id = selected_row['id'].values[0]
            selected_name = selected_row['name'].values[0]
            if selected_id not in data_by_region['id'].values:
                return go.Figure(), None
            tatlas_df = atlas_support_function.get_subregions(data_by_region, selected_id, return_original=True)
            IDs = tatlas_df.loc[:, 'id'].values
            locations = np.where(np.isin(annotated_atlas_img, IDs))
            if locations[0].size == 0:
                z_index = 0
            else:
                z_index = int(np.bincount(locations[0]).argmax())
            region_info = {
                'selected_acronym': selected_acronym,
                'selected_id': selected_id,
                'selected_name': selected_name,
                'IDs': IDs.tolist(),
                'current_z_index': z_index
            }
        elif triggered_id in ['z-minus-10', 'z-minus-1', 'z-plus-1', 'z-plus-10'] and region_info is not None:
            delta = 0
            if triggered_id == 'z-minus-10':
                delta = -10
            elif triggered_id == 'z-minus-1':
                delta = -1
            elif triggered_id == 'z-plus-1':
                delta = 1
            elif triggered_id == 'z-plus-10':
                delta = 10
            region_info['current_z_index'] = int(region_info.get('current_z_index', 0)) + delta

        if region_info is None:
            return go.Figure(), None
        z_index = region_info.get('current_z_index', 0)
        max_z = data_by_img.shape[0] - 1
        z_index = max(0, min(z_index, max_z))
        region_info['current_z_index'] = z_index

        data_slice = data_by_img[z_index, :, :]
        try:
            image_slice = annotated_atlas_img[z_index, :, :]
        except Exception:
            image_slice = data_slice

        rgb_image_slice = color_code_image(data_slice, image_slice, IDs=np.array(region_info['IDs'])\
            , colormap=colormap,cmin = cmin,cmax= cmax)
        region_info['rgb_image_slice'] = rgb_image_slice  # Store rgb_image_slice in region_info
        fig_image = px.imshow(rgb_image_slice)
        id_to_acronym = {row['id']: row['name'] for idx, row in data_by_region.iterrows()}
        acronym_data = np.vectorize(id_to_acronym.get)(image_slice)
        acronym_data[acronym_data == None] = 'background'
        customdata = np.stack([data_slice, acronym_data], axis=-1)
        fig_image.update_traces(
            customdata=customdata,
            hovertemplate=(
                "X: %{x} <br>"
                "Y: %{y} <br>"
                "Normalized density: %{customdata[0]:.3f} <br>"
                "Acronym: %{customdata[1]}<extra></extra>"
            )
        )
        fig_image.update_layout(
            title=f"{region_info['selected_name']}, Z-Slice: {z_index}",
            coloraxis_showscale=False,
            hovermode='x unified',
            hoverlabel=dict(
                bgcolor="rgba(255,255,255,0.75)",
                font_color="grey",
                font_size=12,
                namelength=0
            )
        )
        return fig_image, region_info

    # -------------------------------------------------------
    # Callback: Export Figures as PNG, PDF, or HTML
    # -------------------------------------------------------
    @app.callback(
        Output("save-status", "children"),
        [Input("export-png-button", "n_clicks"),
         Input("export-pdf-button", "n_clicks"),
         Input("export-html-button", "n_clicks")],
        [State("sunburst-plot", "figure"),
         State("image-plot", "figure"),
         State("region_info_store", "data")]
    )
    def export_figures(png_clicks, pdf_clicks, html_clicks, sunburst_fig_data, image_fig_data, region_info):
        ctx = dash.callback_context
        if not ctx.triggered or sunburst_fig_data is None or image_fig_data is None or region_info is None:
            raise dash.exceptions.PreventUpdate
        selected_acronym = region_info.get('selected_acronym', "unknown")
        rgb_image_slice = region_info.get('rgb_image_slice')  # Extract rgb_image_slice from region_info
        if rgb_image_slice is None:
            return "Error: RGB image slice not found in region info."
        button_id = ctx.triggered[0]["prop_id"].split(".")[0]
        high_res_scale = 8
        sunburst_fig = go.Figure(sunburst_fig_data)
        image_fig = go.Figure(image_fig_data)
        if button_id == "export-png-button":
            sunburst_filename = os.path.join(outputpath, f"sunburst_{selected_acronym}.png")
            image_filename = os.path.join(outputpath, f"image_{selected_acronym}.tif")
            try:
                sunburst_fig.write_image(sunburst_filename, format="png", scale=high_res_scale)
                from tifffile import imwrite
                imwrite(image_filename, np.array(rgb_image_slice, dtype=np.uint8))  # Save as TIFF
                return f"PNG file saved: {sunburst_filename}, TIFF file saved: {image_filename}"
            except Exception as e:
                return f"Error saving files: {e}"
        elif button_id == "export-pdf-button":
            sunburst_filename = os.path.join(outputpath, f"sunburst_{selected_acronym}.pdf")
            image_filename = os.path.join(outputpath, f"image_{selected_acronym}.tif")
            try:
                sunburst_fig.write_image(sunburst_filename, format="pdf", scale=high_res_scale)
                from tifffile import imwrite
                imwrite(image_filename, np.array(rgb_image_slice, dtype=np.uint8))  # Save as TIFF
                return f"PDF file saved: {sunburst_filename}, TIFF file saved: {image_filename}"
            except Exception as e:
                return f"Error saving files: {e}"
        elif button_id == "export-html-button":
            sunburst_html = pio.to_html(sunburst_fig, full_html=True, include_plotlyjs='cdn')
            image_html = pio.to_html(image_fig, full_html=True, include_plotlyjs='cdn')
            sunburst_filename = os.path.join(outputpath, f"sunburst_{selected_acronym}.html")
            image_filename = os.path.join(outputpath, f"image_{selected_acronym}.html")
            try:
                with open(sunburst_filename, "w", encoding="utf-8") as f:
                    f.write(sunburst_html)
                with open(image_filename, "w", encoding="utf-8") as f:
                    f.write(image_html)
                return f"HTML files saved: {sunburst_filename} and {image_filename}"
            except Exception as e:
                return f"Error saving HTML files: {e}"
    return app

# ------------------------------
# Optional: Function to Run the App Directly
# ------------------------------

def run_app(data_by_region, data_by_img, annotated_atlas_img, outputpath, colormap=plt.cm.coolwarm, **kwargs):
    """
    Creates and runs the Dash app.
    Additional keyword arguments are passed to app.run_server.
    """
    app = create_dash_app(data_by_region, data_by_img, annotated_atlas_img, outputpath, colormap=colormap, **kwargs)
    app.run_server(**kwargs)

if __name__ == '__main__':
    # For testing, set your variables accordingly.
    # Example:
    # from your_module import TreeFDRF_df, effect_size_img, atlas_img, atlas_df
    # run_app(TreeFDRF_df, effect_size_img, atlas_img, "output_directory", atlas_df,
    #         colormap=plt.cm.coolwarm, sunburst_range_color=[0, 1], debug=True)
    pass
