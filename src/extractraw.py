#!/usr/bin/env python3
"""
Voxel Data Parser - Command Line Interface

Parses XML metadata and binary voxel data to create tissue mapping dictionaries.

Usage:
    python voxel_parser.py <xml_file> <raw_file> [options]

Example:
    python voxel_parser.py af_man_1mm.xml af_man_1mm.raw --output voxel_data.json
"""

import argparse
import json
import numpy as np
import xml.etree.ElementTree as ET
from pathlib import Path
from typing import Dict, Tuple, Optional, Any
import sys


class VoxelDataParser:
    """Parser for voxel XML metadata and binary data files."""
    
    def __init__(self):
        self.mesh_info = {}
        self.tissue_mapping = {} # Maps tissue_value (int) to tissue_name (str)
        self.tissue_details = {} # Maps tissue_value (int) to dict of details
        self.voxel_data = None
        self.voxel_tissue_map = {} # Maps (x,y,z) tuple to tissue_name (str)
    
    def parse_xml_metadata(self, xml_file_path: str) -> Dict:
        """Parse XML metadata file."""
        try:
            tree = ET.parse(xml_file_path)
            root = tree.getroot()
            
            # Parse mesh data
            mesh_data = root.find('MeshData')
            if mesh_data is not None:
                self.mesh_info = {
                    'x_count': int(mesh_data.get('XCount', 0)),
                    'y_count': int(mesh_data.get('YCount', 0)),
                    'z_count': int(mesh_data.get('ZCount', 0))
                }
                
                # Parse resolution
                resolution = mesh_data.find('Resolution')
                if resolution is not None:
                    self.mesh_info['resolution'] = {
                        'x': float(resolution.get('X', 1.0)),
                        'y': float(resolution.get('Y', 1.0)),
                        'z': float(resolution.get('Z', 1.0)),
                        'units': resolution.get('Units', 'mm')
                    }
                
            # Parse tissue data
            tissue_data = root.find('TissueData')
            if tissue_data is not None:
                tissues = tissue_data.findall('Tissue')
                
                for tissue in tissues:
                    name = tissue.get('Name')
                    value = int(tissue.get('Value'))
                    
                    # Store basic mapping (value to name)
                    self.tissue_mapping[value] = name
                    
                    # Store detailed information (value to details dict)
                    self.tissue_details[value] = {
                        'name': name,
                        'value': value,
                        'visible': tissue.get('Visible', 'Yes') == 'Yes',
                        'color': int(tissue.get('Color', 0)) if tissue.get('Color') else 0,
                        'density': float(tissue.get('Density', 0.0)) if tissue.get('Density') else 0.0,
                        'voxel_count': int(tissue.get('VoxelCount', 0)) if tissue.get('VoxelCount') else 0,
                        'user_data': tissue.get('UserData'),
                        'priority': int(tissue.get('Priority', 0)) if tissue.get('Priority') else 0
                    }
                
            print(f"✓ Parsed XML metadata: {len(self.tissue_mapping)} tissue types")
            return self.mesh_info
            
        except ET.ParseError as e:
            raise ValueError(f"Error parsing XML file: {e}")
        except FileNotFoundError:
            raise FileNotFoundError(f"XML file not found: {xml_file_path}")
    
    def load_binary_voxel_data(self, raw_file_path: str) -> np.ndarray:
        """Load binary voxel data."""
        try:
            # Get expected dimensions
            if not self.mesh_info:
                raise ValueError("Must parse XML metadata first to get dimensions")
            
            x_count = self.mesh_info['x_count']
            y_count = self.mesh_info['y_count']
            z_count = self.mesh_info['z_count']
            expected_voxels = x_count * y_count * z_count
            
            # Check file size
            file_path = Path(raw_file_path)
            file_size = file_path.stat().st_size
            
            print(f"Loading binary data:")
            print(f"    File: {raw_file_path}")
            print(f"    File size: {file_size:,} bytes")
            print(f"    Expected voxels: {expected_voxels:,}")
            print(f"    Dimensions: {x_count} × {y_count} × {z_count}")
            
            if file_size != expected_voxels:
                raise ValueError(f"File size mismatch: expected {expected_voxels} bytes, got {file_size}")
            
            # Load binary data as uint8 array
            with open(raw_file_path, 'rb') as f:
                raw_data = f.read()
            
            # Convert to numpy array
            voxel_array = np.frombuffer(raw_data, dtype=np.uint8)
            
            # Reshape to 3D array (assuming Z, Y, X order - common in medical imaging)
            self.voxel_data = voxel_array.reshape((z_count, y_count, x_count))
            
            print(f"✓ Loaded voxel data: shape {self.voxel_data.shape}")
            print(f"    Data type: {self.voxel_data.dtype}")
            print(f"    Min value: {self.voxel_data.min()}")
            print(f"    Max value: {self.voxel_data.max()}")
            print(f"    Unique values: {len(np.unique(self.voxel_data))}")
            
            return self.voxel_data
            
        except FileNotFoundError:
            raise FileNotFoundError(f"Binary file not found: {raw_file_path}")
    
    def get_voxel_tissue_mapping(self, filename: str = None) -> Dict[Tuple[int, int, int], str]:
        """
        Create dictionary mapping voxel coordinates to tissue names and optionally save to a file.

        Args:
            filename (str, optional): Base name for the output text file. If provided, the mapping
                                      will be saved to '{filename}.mapping.out'. Defaults to None.

        Returns:
            Dict[Tuple[int, int, int], str]: A dictionary mapping (x, y, z) voxel coordinates to tissue names.

        Raises:
            ValueError: If voxel data has not been loaded.
        """
        if self.voxel_data is None:
            raise ValueError("Must load voxel data first")
        
        print("Creating voxel coordinate to tissue mapping...")
        
        voxel_tissue_map = {}
        z_count, y_count, x_count = self.voxel_data.shape
        
        # Iterate through all voxels
        for z in range(z_count):
            if z % 100 == 0:  # Progress indicator
                print(f"        Processing slice {z}/{z_count}") 
            
            for y in range(y_count):
                for x in range(x_count):
                    tissue_value = self.voxel_data[z, y, x]
                    
                    # Skip empty/air voxels (value 0 or 1 typically)
                    if tissue_value > 1:  # Assuming 0/1 are air/empty
                        tissue_name = self.tissue_mapping.get(tissue_value, f"Unknown_{tissue_value}")
                        voxel_tissue_map[(x, y, z)] = tissue_name
        
        self.voxel_tissue_map = voxel_tissue_map # Store map with names
        print(f"✓ Created mapping for {len(voxel_tissue_map):,} non-empty voxels")

        # Save the mapping to a text file if filename is provided (outputs with names)
        if filename:
            output_filepath = f"{filename}.mapping.out" # Ensure .mapping.out extension for text file
            try:
                with open(output_filepath, 'w') as f:
                    for coord, tissue_name in voxel_tissue_map.items(): # Iterate over names
                        f.write(f"({coord[0]}, {coord[1]}, {coord[2]}): {tissue_name}\n")
                print(f"✓ Voxel tissue mapping saved to '{output_filepath}'")
            except IOError as e:
                print(f"Error saving mapping to file '{output_filepath}': {e}")
        
        return voxel_tissue_map
    
    def get_tissue_statistics(self) -> Dict[str, Any]:
        """Get statistics about tissue distribution."""
        if self.voxel_data is None:
            return {}
        
        unique_values, counts = np.unique(self.voxel_data, return_counts=True)
        
        stats = {
            'total_voxels': int(self.voxel_data.size),
            'unique_tissue_values': len(unique_values),
            'tissue_distribution': {}
        }
        
        for value, count in zip(unique_values, counts):
            tissue_name = self.tissue_mapping.get(int(value), f"Unknown_{value}")
            percentage = (count / self.voxel_data.size) * 100
            
            stats['tissue_distribution'][tissue_name] = {
                'voxel_value': int(value),
                'voxel_count': int(count),
                'percentage': float(percentage)
            }
        
        return stats
    
    def save_results(self, output_file: str, include_coordinates: bool = False):
        """
        Save parsing results to JSON file.
        If include_coordinates is True, voxel coordinates will be stored with tissue IDs.
        """
        results = {
            'metadata': {
                'mesh_info': self.mesh_info,
                'tissue_mapping': {str(k): v for k, v in self.tissue_mapping.items()},
                'tissue_details': {str(k): v for k, v in self.tissue_details.items()}
            },
            'statistics': self.get_tissue_statistics()
        }
        
        # --- MODIFIED: Include voxel coordinates with tissue IDs for JSON output ---
        if include_coordinates and self.voxel_data is not None: 
            print("  Including voxel coordinates by ID in JSON output...")
            coord_mapping_by_id = {}
            z_count, y_count, x_count = self.voxel_data.shape
            for z in range(z_count):
                for y in range(y_count):
                    for x in range(x_count):
                        tissue_id = int(self.voxel_data[z, y, x]) 
                        # Only include non-empty/air voxels (value > 1) for consistency
                        if tissue_id > 1:
                            coord_mapping_by_id[f"{x},{y},{z}"] = tissue_id # Store ID number
            results['voxel_coordinates'] = coord_mapping_by_id
        # -------------------------------------------------------------------------
        
        with open(output_file, 'w') as f:
            json.dump(results, f, indent=2)
        
        print(f"✓ Results saved to: {output_file}")
    
    def print_summary(self):
        """Print summary of parsed data."""
        print("\n" + "="*60)
        print("VOXEL DATA PARSING SUMMARY")
        print("="*60)
        
        if self.mesh_info:
            print(f"Mesh Dimensions: {self.mesh_info['x_count']} × {self.mesh_info['y_count']} × {self.mesh_info['z_count']}")
            if 'resolution' in self.mesh_info:
                res = self.mesh_info['resolution']
                print(f"Resolution: {res['x']} × {res['y']} × {res['z']} {res['units']}")
        
        if self.voxel_data is not None:
            print(f"Total Voxels: {self.voxel_data.size:,}")
            print(f"Data Shape: {self.voxel_data.shape}")
        
        print(f"Tissue Types: {len(self.tissue_mapping)}")
        
        # Show tissue distribution
        stats = self.get_tissue_statistics()
        if 'tissue_distribution' in stats:
            print(f"\nTissue Distribution:")
            sorted_tissues = sorted(stats['tissue_distribution'].items(), 
                                    key=lambda x: x[1]['voxel_count'], reverse=True)
            
            for tissue_name, info in sorted_tissues[:10]:  # Show top 10
                print(f"    {info['voxel_value']:2d}: {tissue_name:<20} "
                      f"{info['voxel_count']:>10,} voxels ({info['percentage']:5.1f}%)")
            
            if len(sorted_tissues) > 10:
                print(f"    ... and {len(sorted_tissues) - 10} more tissue types")


def main():
    """Main command-line interface."""
    parser = argparse.ArgumentParser(
        description="Parse voxel XML metadata and binary data files",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python extractraw.py metadata.xml data.raw
  python extractraw.py af_man_1mm.xml af_man_1mm.raw --output results.json
  python extractraw.py metadata.xml data.raw --coords --output full_data.json
  python extractraw.py metadata.xml data.raw --mapping my_output_map
        """
    )
    
    parser.add_argument('xml_file', help='Path to XML metadata file')
    parser.add_argument('raw_file', help='Path to binary voxel data file')
    parser.add_argument('-o', '--output', help='Output JSON file path for full results (e.g., results.json)')
    parser.add_argument('--coords', action='store_true', 
                        help='Include voxel coordinates (by ID) in output JSON (warning: can create very large files)')
    parser.add_argument('--stats-only', action='store_true',
                        help='Only generate statistics, skip coordinate mapping')
    parser.add_argument('-m', '--mapping', help="Output text-based mapping file path (e.g., 'my_output_map' will save to 'my_output_map.mapping.out').")
    args = parser.parse_args()
    
    try:
        # Initialize parser
        voxel_parser = VoxelDataParser()
        
        # Parse XML metadata
        print("Step 1: Parsing XML metadata...")
        voxel_parser.parse_xml_metadata(args.xml_file)
        
        # Load binary data
        print("\nStep 2: Loading binary voxel data...")
        voxel_parser.load_binary_voxel_data(args.raw_file)
        
        # Create coordinate mapping (unless stats-only), passing the mapping filename if provided
        # This step populates self.voxel_tissue_map (with names) and optionally writes the .mapping.out file
        if not args.stats_only: 
            print("\nStep 3: Creating voxel-to-tissue mapping...")
            voxel_parser.get_voxel_tissue_mapping(filename=args.mapping)
        
        # Print summary
        voxel_parser.print_summary()
        
        # Save results if output file specified
        # This will now include IDs for voxel_coordinates if --coords is used
        if args.output:
            print(f"\nStep 4: Saving results...")
            voxel_parser.save_results(args.output, include_coordinates=args.coords)
        
        print(f"\n✓ Processing complete!")
        
        # Show example usage (uses the internally stored voxel_tissue_map with names)
        if not args.stats_only:
            print(f"\nExample: Access tissue at coordinate (100, 50, 200):")
            if (100, 50, 200) in voxel_parser.voxel_tissue_map:
                tissue = voxel_parser.voxel_tissue_map[(100, 50, 200)]
                print(f"    Tissue: {tissue}")
            else:
                print(f"    No tissue at that coordinate (likely air/empty)")
    
    except Exception as e:
        print(f"❌ Error: {e}", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()