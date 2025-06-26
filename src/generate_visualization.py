#!/usr/bin/env python3
"""
Voxel Model 3D Visualizer

Loads a text-based voxel mapping file and a metadata JSON file (containing mesh info
and tissue mapping) to generate an interactive 3D visualization.

Supports three modes:
1. Scatter plot of individual voxels (default, can be memory-intensive for large models).
2. Plotly surface rendering (--surface-render), generates an HTML file, more efficient for dense models.
3. PyVista surface rendering (--pyvista), opens a desktop window, most efficient for large models.

Usage:
    python voxel_visualizer.py <mapping_text_file> <metadata_json_file> [options]

Examples:
    # Scatter plot (may consume a lot of memory for large models)
    python voxel_visualizer.py my_model.mapping.out my_model_metadata.json my_model_3d_scatter

    # Plotly surface rendering (recommended for HTML output of large models)
    python voxel_visualizer.py my_model.mapping.out my_model_metadata.json --surface-render my_model_3d_plotly_surface

    # PyVista surface rendering (recommended for desktop window visualization of large models)
    python voxel_visualizer.py my_model.mapping.out my_model_metadata.json --pyvista
"""

import argparse
import json
import sys
import re # For parsing coordinates from the text file
from typing import Dict, Any, Tuple, List
import plotly.graph_objects as go
import numpy as np
from skimage import measure # For marching cubes
import pyvista as pv # For direct desktop visualization

def parse_text_mapping_file(filepath: str) -> Dict[str, str]:
    """
    Parses a text file with voxel mappings in the format (x, y, z): tissue_name.

    Args:
        filepath (str): Path to the text mapping file.

    Returns:
        Dict[str, str]: A dictionary mapping "x,y,z" string coordinates to tissue names.
    """
    voxel_tissue_map = {}
    # Regex to match (x, y, z): tissue_name. Allows for spaces around numbers and comma.
    # It also handles tissue names with spaces.
    line_pattern = re.compile(r"^\((\s*\d+)\s*,\s*(\d+)\s*,\s*(\d+)\s*\):\s*(.+)$")
    
    with open(filepath, 'r') as f:
        for line_num, line in enumerate(f, 1):
            line = line.strip()
            if not line:
                continue # Skip empty lines

            match = line_pattern.match(line)
            if match:
                try:
                    x = int(match.group(1))
                    y = int(match.group(2))
                    z = int(match.group(3))
                    tissue_name = match.group(4).strip()
                    voxel_tissue_map[f"{x},{y},{z}"] = tissue_name
                except ValueError as e:
                    print(f"Warning: Could not parse numbers on line {line_num}: '{line}' - {e}", file=sys.stderr)
            else:
                print(f"Warning: Skipping unparsable line {line_num}: '{line}'", file=sys.stderr)
    return voxel_tissue_map

def get_color_for_tissue(
    tissue_name: str, 
    tissue_mapping_values_to_names: Dict[str, str], 
    tissue_details: Dict[str, Any]
) -> str:
    """
    Determines a deterministic color for a given tissue name.
    Prioritizes explicit colors from tissue_details, falls back to HSL generation.
    """
    color_to_assign = None 

    # Find the tissue_value_str corresponding to the tissue_name
    tissue_value_str = None
    for val_s, name_s in tissue_mapping_values_to_names.items():
        if name_s == tissue_name:
            tissue_value_str = val_s
            break

    if tissue_value_str and tissue_value_str in tissue_details:
        detail = tissue_details[tissue_value_str]
        if 'color' in detail:
            color_value = detail['color']
            if isinstance(color_value, int):
                try:
                    # Convert integer color (0xRRGGBB) to hex string
                    color_to_assign = '#' + format(color_value, '06x')
                except Exception as e:
                    print(f"CRITICAL ERROR: Failed to format integer color '{color_value}' (type: {type(color_value)}) for tissue '{tissue_name}'. Error: {e}", file=sys.stderr)
                    color_to_assign = None # Fallback if formatting fails
    
    # If no explicit color was found or formatting failed, generate one using HSL
    if color_to_assign is None:
        tissue_value = None
        # Try to get the integer value from the name_to_value_mapping
        for val_s, name_s in tissue_mapping_values_to_names.items():
            if name_s == tissue_name:
                try:
                    tissue_value = int(val_s)
                    break
                except ValueError:
                    pass # Continue if value_str is not an integer

        if tissue_value is not None:
            hue = (tissue_value * 137) % 360 
        else:
            # For "Unknown_X" or other names not directly in the metadata's tissue_mapping
            try:
                if tissue_name.startswith("Unknown_"):
                    unknown_val_str = tissue_name.split('_')[1]
                    numeric_part = int(unknown_val_str)
                    hue = (numeric_part * 137 + 50) % 360 
                else: 
                    hue = hash(tissue_name) % 360 
            except (ValueError, IndexError):
                hue = hash(tissue_name) % 360 
        color_to_assign = f"hsl({hue}, 70%, 60%)"
    
    return color_to_assign


def extract_mesh_data_from_volume(
    voxel_tissue_map: Dict[str, str], 
    mesh_info: Dict[str, int],
    tissue_mapping_values_to_names: Dict[str, str],
    tissue_details: Dict[str, Any]
) -> List[Dict[str, Any]]:
    """
    Reconstructs a 3D voxel array and applies marching cubes to extract surface meshes
    for each tissue type.

    Args:
        voxel_tissue_map: Dictionary mapping "x,y,z" string coordinates to tissue names.
        mesh_info: Dictionary containing 'x_count', 'y_count', 'z_count' dimensions.
        tissue_mapping_values_to_names: Dictionary mapping tissue value (as string) to tissue name.
        tissue_details: Dictionary containing detailed information about each tissue type, including 'color'.

    Returns:
        List[Dict[str, Any]]: A list of dictionaries, each containing:
            - 'verts': numpy array of vertices
            - 'faces': numpy array of faces
            - 'tissue_name': string name of the tissue
            - 'color_hex': hex string color for the tissue
    """
    x_dim = mesh_info.get('x_count', 0)
    y_dim = mesh_info.get('y_count', 0)
    z_dim = mesh_info.get('z_count', 0)

    if x_dim == 0 or y_dim == 0 or z_dim == 0:
        print(f"❌ Invalid mesh dimensions: {x_dim}x{y_dim}x{z_dim}. Cannot create mesh data.", file=sys.stderr)
        return []

    print(f"  Reconstructing 3D voxel array for marching cubes...")
    # Initialize a 3D array with a background value (e.g., 0)
    # The order for numpy array should match the access pattern (z, y, x)
    volume = np.zeros((z_dim, y_dim, x_dim), dtype=np.uint8) 

    # Create a reverse mapping from tissue name to integer value (for populating volume array)
    name_to_value_int_mapping = {}
    for val_str, name in tissue_mapping_values_to_names.items():
        try:
            name_to_value_int_mapping[name] = int(val_str)
        except ValueError:
            pass # Continue if value_str is not an integer

    # Populate the 3D volume array with tissue integer values
    # If a tissue name doesn't have an explicit integer value, assign a temporary unique one
    # starting from a high number to avoid conflicts with known tissue values.
    temp_value_counter = 250 # Start assigning temporary values from here

    for key, tissue_name in voxel_tissue_map.items():
        x, y, z = map(int, key.split(','))
        
        # Ensure coordinates are within bounds
        if 0 <= x < x_dim and 0 <= y < y_dim and 0 <= z < z_dim:
            if tissue_name in name_to_value_int_mapping:
                volume[z, y, x] = name_to_value_int_mapping[tissue_name]
            else:
                # Assign a temporary unique integer value if not found in metadata mapping
                if tissue_name not in name_to_value_int_mapping:
                    name_to_value_int_mapping[tissue_name] = temp_value_counter
                    temp_value_counter += 1
                volume[z, y, x] = name_to_value_int_mapping[tissue_name]
        else:
            print(f"Warning: Voxel coordinate ({x},{y},{z}) is out of bounds for dimensions {x_dim}x{y_dim}x{z_dim}. Skipping.", file=sys.stderr)


    all_mesh_data = []
    unique_tissue_values_in_volume = np.unique(volume)

    print(f"  Extracting surfaces for {len(unique_tissue_values_in_volume) - (1 if 0 in unique_tissue_values_in_volume else 0)} tissue types...") # Subtract 1 if 0 is background

    # Iterate over each unique tissue value (excluding background 0)
    for tissue_value_int in unique_tissue_values_in_volume:
        if tissue_value_int == 0: # Assuming 0 is background/empty space
            continue

        # Create a binary mask for the current tissue type
        binary_mask = (volume == tissue_value_int)
        
        # Check if the tissue actually exists in the mask
        if not np.any(binary_mask):
            continue

        # Use marching_cubes to get the surface mesh
        # iso_value determines the surface boundary (anything above this is "inside")
        # For a binary mask, a value like 0.5 is typical.
        try:
            verts, faces, _, _ = measure.marching_cubes(binary_mask, level=0.5) # level 0.5 for binary mask
        except ValueError as e:
            # This can happen if the mask is entirely false (no tissue) or other issues
            print(f"Warning: Could not extract mesh for tissue value {tissue_value_int} (no surface found or error): {e}", file=sys.stderr)
            continue
        
        # If no vertices or faces are generated, skip
        if verts.size == 0 or faces.size == 0:
            print(f"  Skipping tissue value {tissue_value_int}: No surface generated by marching cubes.", file=sys.stderr)
            continue

        # Map the tissue_value_int back to tissue_name
        tissue_name = next((name for val_str, name in tissue_mapping_values_to_names.items() if int(val_str) == tissue_value_int), f"Unknown_{tissue_value_int}")
        # If the tissue_name is still "Unknown_X" and it was a temp value, try to find original name
        if tissue_name.startswith("Unknown_") and tissue_value_int >= 250: # Check if it's a temp assigned value
             tissue_name = next((name for name, val_int in name_to_value_int_mapping.items() if val_int == tissue_value_int), tissue_name)


        color_hex = get_color_for_tissue(tissue_name, tissue_mapping_values_to_names, tissue_details)

        all_mesh_data.append({
            'verts': verts,
            'faces': faces,
            'tissue_name': tissue_name,
            'color_hex': color_hex
        })

    if not all_mesh_data:
        print("  No meshes were generated for visualization. Ensure your data has distinct tissue boundaries.", file=sys.stderr)
        
    return all_mesh_data


def generate_3d_visualization_plotly_html(
    voxel_tissue_map: Dict[str, str], 
    mesh_info: Dict[str, int], 
    tissue_mapping_values_to_names: Dict[str, str], 
    tissue_details: Dict[str, Any], 
    output_filename_base: str,
    surface_render: bool = False,
    # This function no longer directly uses raw_mesh_data, but it's generated by main
    # and then passed depending on the render mode.
) -> None:
    """
    Generates an HTML file with an embedded Plotly 3D visualization.

    Args:
        voxel_tissue_map: Dictionary mapping "x,y,z" string coordinates to tissue names.
        mesh_info: Dictionary containing 'x_count', 'y_count', 'z_count' dimensions.
        tissue_mapping_values_to_names: Dictionary mapping tissue value (as string) to tissue name.
        tissue_details: Dictionary containing detailed information about each tissue type, including 'color'.
        output_filename_base: Base name for the output HTML file.
        surface_render: If True, uses Mesh3d for surface rendering; otherwise, uses Scatter3d for individual voxels.
    """
    x_dim = mesh_info.get('x_count', 0)
    y_dim = mesh_info.get('y_count', 0)
    z_dim = mesh_info.get('z_count', 0)

    if x_dim == 0 or y_dim == 0 or z_dim == 0:
        print(f"❌ Invalid mesh dimensions: {x_dim}x{y_dim}x{z_dim}. Cannot generate 3D visualization.", file=sys.stderr)
        return

    data_traces = []

    if surface_render:
        print("  Using surface rendering (Mesh3d) for Plotly...")
        # Call the mesh extraction directly here for Plotly's needs
        all_mesh_data = extract_mesh_data_from_volume(voxel_tissue_map, mesh_info, tissue_mapping_values_to_names, tissue_details)
        if not all_mesh_data:
            print("❌ No surface traces generated for Plotly. Exiting visualization.", file=sys.stderr)
            return

        for mesh_data_item in all_mesh_data:
            mesh_trace = go.Mesh3d(
                x=mesh_data_item['verts'][:, 0], 
                y=mesh_data_item['verts'][:, 1], 
                z=mesh_data_item['verts'][:, 2], 
                i=mesh_data_item['faces'][:, 0], 
                j=mesh_data_item['faces'][:, 1], 
                k=mesh_data_item['faces'][:, 2], 
                color=mesh_data_item['color_hex'],
                opacity=0.7, 
                name=mesh_data_item['tissue_name'],
                hoverinfo='name',
                showscale=False 
            )
            data_traces.append(mesh_trace)
    else:
        print("  Using individual voxel scatter plot (Scatter3d) for Plotly...")
        x_coords = []
        y_coords = []
        z_coords = []
        tissue_names_list = [] 

        for key, tissue_name in voxel_tissue_map.items():
            x, y, z = map(int, key.split(','))
            x_coords.append(x)
            y_coords.append(y)
            z_coords.append(z)
            tissue_names_list.append(tissue_name)

        colors_for_plot = [
            get_color_for_tissue(name, tissue_mapping_values_to_names, tissue_details) 
            for name in tissue_names_list
        ]
        
        trace = go.Scatter3d(
            x=x_coords,
            y=y_coords,
            z=z_coords,
            mode='markers',
            marker=dict(
                size=2,  
                color=colors_for_plot,
                opacity=0.7,
                line=dict(width=0) 
            ),
            text=tissue_names_list, 
            hoverinfo='text+x+y+z',
            name='Voxels'
        )
        data_traces.append(trace)

    # Define the layout
    layout = go.Layout(
        title=f"Voxelized Model 3D Visualization ({output_filename_base})",
        scene=dict(
            xaxis=dict(title='X', showbackground=False, zeroline=False),
            yaxis=dict(title='Y', showbackground=False, zeroline=False),
            zaxis=dict(title='Z', showbackground=False, zeroline=False),
            aspectmode='data', 
            camera=dict(
                eye=dict(x=x_dim*0.8/max(x_dim,y_dim,z_dim), y=y_dim*0.8/max(x_dim,y_dim,z_dim), z=z_dim*0.8/max(x_dim,y_dim,z_dim)) 
            ),
            bgcolor='rgb(34,34,34)' 
        ),
        margin=dict(l=0, r=0, b=0, t=40),
        height=800, 
        paper_bgcolor='rgb(34,34,34)', 
        font=dict(color='white') 
    )

    fig = go.Figure(data=data_traces, layout=layout)
    
    output_filepath = f"{output_filename_base}.html"
    try:
        fig.write_html(output_filepath, auto_open=False)
        print(f"✓ 3D visualization HTML saved to '{output_filepath}'")
        print(f"   Open '{output_filepath}' in your web browser to view the model.")
    except Exception as e:
        print(f"Error saving Plotly HTML to file '{output_filepath}': {e}", file=sys.stderr)

def generate_3d_visualization_pyvista_window(
    all_mesh_data: List[Dict[str, Any]], 
    mesh_info: Dict[str, int],
    window_title: str = "Voxel Model 3D Visualization (PyVista)"
) -> None:
    """
    Generates an interactive 3D visualization in a PyVista window.

    Args:
        all_mesh_data: List of dictionaries, each containing 'verts', 'faces', 'tissue_name', 'color_hex'.
        mesh_info: Dictionary containing 'x_count', 'y_count', 'z_count' dimensions.
        window_title: Title for the PyVista visualization window.
    """
    if not all_mesh_data:
        print("❌ No mesh data provided for PyVista visualization. Exiting.", file=sys.stderr)
        return

    x_dim = mesh_info.get('x_count', 0)
    y_dim = mesh_info.get('y_count', 0)
    z_dim = mesh_info.get('z_count', 0)

    if x_dim == 0 or y_dim == 0 or z_dim == 0:
        print(f"❌ Invalid mesh dimensions: {x_dim}x{y_dim}x{z_dim}. Cannot generate PyVista visualization.", file=sys.stderr)
        return

    plotter = pv.Plotter(window_size=[1024, 768], off_screen=False) # Open a window
    plotter.background_color = (0.13, 0.13, 0.13) # Dark background color for PyVista

    # Add each mesh to the plotter
    for mesh_data_item in all_mesh_data:
        verts = mesh_data_item['verts']
        faces = mesh_data_item['faces']
        tissue_name = mesh_data_item['tissue_name']
        color_hex = mesh_data_item['color_hex']

        # PyVista requires faces to be in a specific format:
        # [n_points_in_face, idx1, idx2, idx3, n_points_in_next_face, ...]
        # For triangles, it's [3, i, j, k, 3, i', j', k', ...]
        faces_pv = np.hstack([np.full((faces.shape[0], 1), 3), faces]).flatten()

        polydata = pv.PolyData(verts, faces_pv)
        
        # Add the mesh to the plotter
        # PyVista accepts HTML color strings (like #RRGGBB) directly
        plotter.add_mesh(
            polydata, 
            color=color_hex, 
            opacity=0.7, 
            show_edges=False, # Show edges for clarity or set to False for smoother appearance
            label=tissue_name
        )

    # Set camera position based on model dimensions
    center = np.array([x_dim / 2, y_dim / 2, z_dim / 2])
    distance = max(x_dim, y_dim, z_dim) * 1.5 # Distance from center
    
    # Simple camera position: looking at center from an angle
    plotter.camera_position = [
        center + distance, 
        center, 
        (0, 0, 1) # Up direction (Z-axis)
    ]
    plotter.camera.SetPosition(x_dim * 1.5, y_dim * 1.5, z_dim * 1.5) # Example
    plotter.camera.SetFocalPoint(x_dim / 2, y_dim / 2, z_dim / 2) # Focus on center

    # Add a title
    plotter.add_text(window_title, font_size=16, color='white', position='upper_left')

    # Show the plot
    print(f"✓ Opening PyVista 3D visualization window...")
    plotter.show()
    print(f"✓ PyVista window closed.")


def main():
    """Main command-line interface for the 3D voxel visualizer."""
    parser = argparse.ArgumentParser(
        description="Load a text-based voxel mapping file and a metadata JSON file to generate a 3D visualization.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Example:
  # Scatter plot (may consume a lot of memory for large models)
  python voxel_visualizer.py my_model.mapping.out my_model_metadata.json my_model_3d_scatter

  # Plotly surface rendering (recommended for HTML output of large models)
  python voxel_visualizer.py my_model.mapping.out my_model_metadata.json --surface-render my_model_3d_plotly_surface

  # PyVista surface rendering (recommended for desktop window visualization of large models)
  python voxel_visualizer.py my_model.mapping.out my_model_metadata.json --pyvista
        """
    )
    
    parser.add_argument('mapping_text_file', 
                        help='Path to the text-based mapping file (e.g., my_model.mapping.out).')
    parser.add_argument('metadata_json_file',
                        help='Path to the JSON metadata file containing "mesh_info" and "tissue_mapping".')
    parser.add_argument('output_html_filename', nargs='?', default='voxel_model_3d_plotly',
                        help='Base name for the output HTML visualization file (e.g., "my_model_3d_plotly" will save to "my_model_3d_plotly.html"). Only used if --surface-render is NOT used with --pyvista.')
    parser.add_argument('-s', '--surface-render', action='store_true',
                        help='Generate a surface mesh visualization (Plotly HTML output).')
    parser.add_argument('-p', '--pyvista', action='store_true',
                        help='Display a surface mesh visualization in a desktop window using PyVista (overrides --surface-render for output type).')


    args = parser.parse_args()
    
    try:
        print(f"Loading mapping data from '{args.mapping_text_file}'...")
        voxel_tissue_map = parse_text_mapping_file(args.mapping_text_file)
        if not voxel_tissue_map:
            raise ValueError("No valid voxel mappings found in the provided text file.")
        print(f"✓ Loaded {len(voxel_tissue_map):,} voxel mappings.")

        print(f"\nLoading metadata from '{args.metadata_json_file}'...")
        with open(args.metadata_json_file, 'r') as f:
            metadata_data = json.load(f)
        
        # Extract data from metadata JSON
        mesh_info = metadata_data.get("metadata", {}).get("mesh_info")
        tissue_mapping_values_to_names = metadata_data.get("metadata", {}).get("tissue_mapping", {})
        tissue_details = metadata_data.get("metadata", {}).get("tissue_details", {}) 

        if not mesh_info:
            raise ValueError("Invalid metadata file format. Missing 'metadata.mesh_info'.")
        if not tissue_mapping_values_to_names:
            print("Warning: 'metadata.tissue_mapping' not found in metadata JSON. Colors will be generated without explicit tissue value knowledge.", file=sys.stderr)
        if not tissue_details:
            print("Warning: 'metadata.tissue_details' not found in metadata JSON. Default HSL colors will be used as explicit colors are not available.", file=sys.stderr)


        print(f"✓ Metadata loaded successfully.")
        
        # Determine rendering mode
        if args.pyvista:
            print(f"\nGenerating 3D visualization using PyVista...")
            all_mesh_data = extract_mesh_data_from_volume(voxel_tissue_map, mesh_info, tissue_mapping_values_to_names, tissue_details)
            if all_mesh_data:
                generate_3d_visualization_pyvista_window(all_mesh_data, mesh_info)
            else:
                print("❌ No mesh data generated for PyVista visualization.", file=sys.stderr)
        elif args.surface_render:
            print(f"\nGenerating 3D visualization HTML using Plotly (surface render)...")
            generate_3d_visualization_plotly_html(
                voxel_tissue_map=voxel_tissue_map, 
                mesh_info=mesh_info, 
                tissue_mapping_values_to_names=tissue_mapping_values_to_names,
                tissue_details=tissue_details, 
                output_filename_base=args.output_html_filename,
                surface_render=True # Explicitly set to True for Plotly surface mode
            )
        else: # Default is Plotly scatter plot
            print(f"\nGenerating 3D visualization HTML using Plotly (scatter plot)...")
            generate_3d_visualization_plotly_html(
                voxel_tissue_map=voxel_tissue_map, 
                mesh_info=mesh_info, 
                tissue_mapping_values_to_names=tissue_mapping_values_to_names,
                tissue_details=tissue_details, 
                output_filename_base=args.output_html_filename,
                surface_render=False # Explicitly set to False for Plotly scatter mode
            )
        
        print(f"\n✓ Visualization generation complete!")
    
    except FileNotFoundError as e:
        print(f"❌ Error: File not found: {e.filename}", file=sys.stderr)
        sys.exit(1)
    except json.JSONDecodeError as e:
        print(f"❌ Error: Could not parse JSON from metadata file: {e}", file=sys.stderr)
        sys.exit(1)
    except ValueError as e:
        print(f"❌ Error: {e}", file=sys.stderr)
        sys.exit(1)
    except Exception as e:
        print(f"❌ An unexpected error occurred: {e}", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()
