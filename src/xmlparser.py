import xml.etree.ElementTree as ET
from typing import Dict, Optional, Tuple


class VoxelModelParser:
    """Parser for voxelized biological model XML files."""
    
    def __init__(self):
        self.mesh_info = {}
        self.tissue_mapping = {}
        self.tissue_details = {}
    
    def parse_file(self, xml_file_path: str) -> Dict[int, str]:
        """
        Parse XML file and return dictionary mapping voxel values to tissue names.
        
        Args:
            xml_file_path: Path to the XML file
            
        Returns:
            Dictionary mapping voxel values (int) to tissue names (str)
        """
        try:
            tree = ET.parse(xml_file_path)
            root = tree.getroot()
            
            # Parse mesh data
            self._parse_mesh_data(root)
            
            # Parse tissue data
            self._parse_tissue_data(root)
            
            return self.tissue_mapping
            
        except ET.ParseError as e:
            raise ValueError(f"Error parsing XML file: {e}")
        except FileNotFoundError:
            raise FileNotFoundError(f"XML file not found: {xml_file_path}")
    
    def parse_string(self, xml_string: str) -> Dict[int, str]:
        """
        Parse XML string and return dictionary mapping voxel values to tissue names.
        
        Args:
            xml_string: XML content as string
            
        Returns:
            Dictionary mapping voxel values (int) to tissue names (str)
        """
        try:
            root = ET.fromstring(xml_string)
            
            # Parse mesh data
            self._parse_mesh_data(root)
            
            # Parse tissue data
            self._parse_tissue_data(root)
            
            return self.tissue_mapping
            
        except ET.ParseError as e:
            raise ValueError(f"Error parsing XML string: {e}")
    
    def _parse_mesh_data(self, root: ET.Element) -> None:
        """Parse mesh data section."""
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
    
    def _parse_tissue_data(self, root: ET.Element) -> None:
        """Parse tissue data section."""
        tissue_data = root.find('TissueData')
        if tissue_data is not None:
            tissues = tissue_data.findall('Tissue')
            
            for tissue in tissues:
                name = tissue.get('Name')
                value = int(tissue.get('Value'))
                
                # Store basic mapping
                self.tissue_mapping[value] = name
                
                # Store detailed information
                self.tissue_details[value] = {
                    'name': name,
                    'value': value,
                    'visible': tissue.get('Visible', 'Yes') == 'Yes',
                    'color': int(tissue.get('Color', 0)),
                    'density': float(tissue.get('Density', 0.0)),
                    'voxel_count': int(tissue.get('VoxelCount', 0)),
                    'user_data': tissue.get('UserData'),
                    'priority': int(tissue.get('Priority', 0))
                }
    
    def get_mesh_info(self) -> Dict:
        """Get mesh information."""
        return self.mesh_info
    
    def get_tissue_details(self) -> Dict:
        """Get detailed tissue information."""
        return self.tissue_details
    
    def get_tissue_by_value(self, value: int) -> Optional[str]:
        """Get tissue name by voxel value."""
        return self.tissue_mapping.get(value)
    
    def get_total_voxels(self) -> int:
        """Calculate total number of voxels in the mesh."""
        if self.mesh_info:
            return (self.mesh_info.get('x_count', 0) * 
                   self.mesh_info.get('y_count', 0) * 
                   self.mesh_info.get('z_count', 0))
        return 0
    
    def print_summary(self) -> None:
        """Print a summary of the parsed data."""
        print("=== Voxel Model Summary ===")
        print(f"Mesh dimensions: {self.mesh_info.get('x_count', 0)} x "
              f"{self.mesh_info.get('y_count', 0)} x {self.mesh_info.get('z_count', 0)}")
        
        if 'resolution' in self.mesh_info:
            res = self.mesh_info['resolution']
            print(f"Resolution: {res['x']} x {res['y']} x {res['z']} {res['units']}")
        
        print(f"Total voxels: {self.get_total_voxels():,}")
        print(f"Number of tissue types: {len(self.tissue_mapping)}")
        print("\nTissue mapping:")
        for value, name in sorted(self.tissue_mapping.items()):
            voxel_count = self.tissue_details[value]['voxel_count']
            print(f"  {value:2d}: {name} ({voxel_count:,} voxels)")


def parse_voxel_xml(xml_file_path: str) -> Dict[int, str]:
    """
    Convenience function to parse XML file and return voxel-to-tissue mapping.
    
    Args:
        xml_file_path: Path to the XML file
        
    Returns:
        Dictionary mapping voxel values to tissue names
    """
    parser = VoxelModelParser()
    return parser.parse_file(xml_file_path)


# Example usage
if __name__ == "__main__":
    # Example XML content (your provided sample)
    sample_xml = '''<!DOCTYPE TissueMeshData SYSTEM "TissueMeshData.dtd">
<TissueMeshData Name="AF 'man' Mesh- 1mm" FileName="af_man_1mm.raw.gz" DataType="Raw" >
    <MeshData XCount="586" YCount="340" ZCount="1878" >
        <Resolution X="1" Y="1" Z="1" Units="mm" />
    </MeshData>
    <TissueData Count="39" >
        <Tissue Name="bile" Value="2" Visible="Yes" Color="4718592" Density="1010.000000" VoxelCount="18991" UserData="17" Priority="10" />
        <Tissue Name="body fluid" Value="3" Visible="Yes" Color="7143424" Density="1010.000000" VoxelCount="369213" UserData="32" Priority="22" />
        <Tissue Name="eye cornea" Value="4" Visible="Yes" Color="9502720" Density="1076.000000" VoxelCount="239" UserData="36" Priority="3" />
        <Tissue Name="fat" Value="5" Visible="Yes" Color="11927552" Density="916.000000" VoxelCount="33374895" UserData="4" Priority="37" />
        <Tissue Name="lymph" Value="6" Visible="Yes" Color="14286848" Density="1040.000000" VoxelCount="63164" UserData="40" Priority="12" />
        <Tissue Name="mucous membrane" Value="7" Visible="Yes" Color="16711680" Density="1040.000000" VoxelCount="460314" UserData="34" Priority="25" />
        <Tissue Name="toe and finger nails" Value="8" Visible="Yes" Color="21760" Density="1030.000000" VoxelCount="3115" UserData="37" Priority="5" />
        <Tissue Name="nerve spine" Value="11" Visible="Yes" Color="7165184" Density="1038.000000" VoxelCount="347713" UserData="15" Priority="2" />
        <Tissue Name="muscle" Value="17" Visible="Yes" Color="2402816" Density="1046.900024" VoxelCount="42815299" UserData="8" Priority="38" />
    </TissueData>
</TissueMeshData>'''
    
    # Parse the sample XML
    parser = VoxelModelParser()
    tissue_mapping = parser.parse_string(sample_xml)
    
    # Print results
    print("Voxel value to tissue mapping:")
    for value, tissue in sorted(tissue_mapping.items()):
        print(f"{value}: {tissue}")
    
    print("\nDetailed summary:")
    parser.print_summary()
    
    # Example of using the convenience function with a file
    # tissue_mapping = parse_voxel_xml("your_file.xml")