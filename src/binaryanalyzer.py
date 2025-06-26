import struct
import numpy as np
from pathlib import Path
from typing import Dict, List, Tuple, Optional, Union
import collections
import sys

class VoxelBinaryAnalyzer:
    """Analyzer for binary voxel files to determine encoding format."""
    
    def __init__(self):
        self.file_size = 0
        self.header_info = {}
        self.data_analysis = {}
        self.possible_formats = []
    
    def analyze_file(self, file_path: str, expected_dimensions: Optional[Tuple[int, int, int]] = None) -> Dict:
        """
        Analyze a binary voxel file to determine its format and encoding.
        
        Args:
            file_path: Path to the binary file
            expected_dimensions: Expected (x, y, z) dimensions if known
            
        Returns:
            Dictionary with analysis results
        """
        file_path = Path(file_path)
        if not file_path.exists():
            raise FileNotFoundError(f"File not found: {file_path}")
        
        self.file_size = file_path.stat().st_size
        print(f"Analyzing file: {file_path}")
        print(f"File size: {self.file_size:,} bytes")
        
        with open(file_path, 'rb') as f:
            # Read first chunk for header analysis
            header_chunk = f.read(1024)
            
            # Read sample data from different positions
            f.seek(0)
            data_samples = self._sample_file_data(f)
        
        # Analyze header
        self._analyze_header(header_chunk)
        
        # Analyze data patterns
        self._analyze_data_patterns(data_samples)
        
        # Try to determine format based on expected dimensions
        if expected_dimensions:
            self._analyze_with_dimensions(data_samples, expected_dimensions)
        
        # Compile results
        results = {
            'file_size': self.file_size,
            'header_analysis': self.header_info,
            'data_analysis': self.data_analysis,
            'possible_formats': self.possible_formats,
            'recommendations': self._generate_recommendations(expected_dimensions)
        }
        
        return results
    
    def _sample_file_data(self, file_obj, sample_size: int = 10000) -> bytes:
        """Sample data from different positions in the file."""
        file_obj.seek(0, 2)  # Seek to end
        file_size = file_obj.tell()
        
        # Sample from beginning, middle, and end
        samples = []
        positions = [0, file_size // 4, file_size // 2, 3 * file_size // 4]
        
        for pos in positions:
            if pos + sample_size < file_size:
                file_obj.seek(pos)
                samples.append(file_obj.read(min(sample_size, 1000)))
        
        return b''.join(samples)
    
    def _analyze_header(self, header_data: bytes):
        """Analyze potential header information."""
        self.header_info = {
            'has_ascii_text': self._has_ascii_text(header_data),
            'potential_magic_numbers': self._find_magic_numbers(header_data),
            'first_bytes_hex': header_data[:64].hex(),
            'first_bytes_ascii': self._bytes_to_ascii(header_data[:64])
        }
        
        # Check for common voxel format signatures
        if header_data.startswith(b'#binvox'):
            self.header_info['format_detected'] = 'BINVOX'
            self._parse_binvox_header(header_data)
        elif b'SIMPLE' in header_data:
            self.header_info['format_detected'] = 'Simple Voxels'
        elif header_data.startswith(b'VOX '):
            self.header_info['format_detected'] = 'MagicaVoxel'
    
    def _analyze_data_patterns(self, data_samples: bytes):
        """Analyze data patterns to infer encoding."""
        if not data_samples:
            return
        
        # Analyze byte value distribution
        byte_counts = collections.Counter(data_samples)
        unique_values = len(byte_counts)
        most_common = byte_counts.most_common(10)
        
        # Analyze as different data types
        uint8_analysis = self._analyze_as_dtype(data_samples, np.uint8)
        uint16_analysis = self._analyze_as_dtype(data_samples, np.uint16)
        uint32_analysis = self._analyze_as_dtype(data_samples, np.uint32)
        
        self.data_analysis = {
            'unique_byte_values': unique_values,
            'most_common_bytes': most_common,
            'zero_percentage': (byte_counts[0] / len(data_samples)) * 100 if 0 in byte_counts else 0,
            'uint8_analysis': uint8_analysis,
            'uint16_analysis': uint16_analysis,
            'uint32_analysis': uint32_analysis,
            'entropy': self._calculate_entropy(data_samples)
        }
    
    def _analyze_as_dtype(self, data: bytes, dtype) -> Dict:
        """Analyze data as a specific numpy data type."""
        try:
            # Convert to numpy array
            byte_size = np.dtype(dtype).itemsize
            if len(data) < byte_size:
                return {'error': 'Insufficient data'}
            
            # Truncate to multiple of byte_size
            truncated_size = (len(data) // byte_size) * byte_size
            array = np.frombuffer(data[:truncated_size], dtype=dtype)
            
            return {
                'min_value': int(np.min(array)),
                'max_value': int(np.max(array)),
                'unique_values': len(np.unique(array)),
                'zero_count': int(np.sum(array == 0)),
                'non_zero_percentage': (np.sum(array != 0) / len(array)) * 100,
                'mean': float(np.mean(array)),
                'std': float(np.std(array))
            }
        except Exception as e:
            return {'error': str(e)}
    
    def _analyze_with_dimensions(self, data_samples: bytes, dimensions: Tuple[int, int, int]):
        """Analyze assuming specific voxel grid dimensions."""
        x, y, z = dimensions
        expected_voxels = x * y * z
        
        # Calculate bytes per voxel for different scenarios
        scenarios = []
        
        # Scenario 1: No header, direct voxel data
        for bytes_per_voxel in [1, 2, 4]:
            expected_size = expected_voxels * bytes_per_voxel
            if abs(self.file_size - expected_size) / self.file_size < 0.01:  # Within 1%
                scenarios.append({
                    'type': 'raw_voxels',
                    'bytes_per_voxel': bytes_per_voxel,
                    'header_size': 0,
                    'expected_size': expected_size,
                    'size_match': 'exact'
                })
        
        # Scenario 2: With various header sizes
        for header_size in [64, 128, 256, 512, 1024]:
            for bytes_per_voxel in [1, 2, 4]:
                expected_size = header_size + (expected_voxels * bytes_per_voxel)
                if abs(self.file_size - expected_size) / self.file_size < 0.01:
                    scenarios.append({
                        'type': 'header_plus_voxels',
                        'bytes_per_voxel': bytes_per_voxel,
                        'header_size': header_size,
                        'expected_size': expected_size,
                        'size_match': 'close'
                    })
        
        self.data_analysis['dimension_scenarios'] = scenarios
    
    def _has_ascii_text(self, data: bytes) -> bool:
        """Check if data contains readable ASCII text."""
        try:
            text = data.decode('ascii')
            return any(c.isprintable() and c != '\x00' for c in text)
        except UnicodeDecodeError:
            return False
    
    def _find_magic_numbers(self, data: bytes) -> List[str]:
        """Look for potential magic number patterns."""
        magic_numbers = []
        
        # Check first few bytes as different integer types
        if len(data) >= 4:
            magic_numbers.append(f"uint32_le: {struct.unpack('<I', data[:4])[0]}")
            magic_numbers.append(f"uint32_be: {struct.unpack('>I', data[:4])[0]}")
        
        if len(data) >= 2:
            magic_numbers.append(f"uint16_le: {struct.unpack('<H', data[:2])[0]}")
            magic_numbers.append(f"uint16_be: {struct.unpack('>H', data[:2])[0]}")
        
        return magic_numbers
    
    def _bytes_to_ascii(self, data: bytes) -> str:
        """Convert bytes to ASCII, replacing non-printable chars."""
        return ''.join(chr(b) if 32 <= b <= 126 else '.' for b in data)
    
    def _parse_binvox_header(self, header_data: bytes):
        """Parse BINVOX format header."""
        try:
            header_text = header_data.decode('ascii')
            lines = header_text.split('\n')
            
            for line in lines:
                if line.startswith('dim '):
                    dims = line.split()[1:]
                    self.header_info['binvox_dimensions'] = [int(d) for d in dims]
                elif line.startswith('translate '):
                    trans = line.split()[1:]
                    self.header_info['binvox_translate'] = [float(t) for t in trans]
                elif line.startswith('scale '):
                    self.header_info['binvox_scale'] = float(line.split()[1])
                elif line.startswith('data'):
                    break
        except Exception as e:
            self.header_info['binvox_parse_error'] = str(e)
    
    def _calculate_entropy(self, data: bytes) -> float:
        """Calculate Shannon entropy of the data."""
        if not data:
            return 0.0
        
        byte_counts = collections.Counter(data)
        length = len(data)
        entropy = 0.0
        
        for count in byte_counts.values():
            p = count / length
            if p > 0:
                entropy -= p * np.log2(p)
        
        return entropy
    
    def _generate_recommendations(self, expected_dimensions: Optional[Tuple[int, int, int]]) -> List[str]:
        """Generate recommendations based on analysis."""
        recommendations = []
        
        # Based on file size and expected dimensions
        if expected_dimensions and 'dimension_scenarios' in self.data_analysis:
            scenarios = self.data_analysis['dimension_scenarios']
            if scenarios:
                best_scenario = scenarios[0]
                recommendations.append(
                    f"Most likely format: {best_scenario['bytes_per_voxel']} bytes per voxel "
                    f"with {best_scenario['header_size']} byte header"
                )
        
        # Based on data patterns
        if 'unique_byte_values' in self.data_analysis:
            unique_vals = self.data_analysis['unique_byte_values']
            if unique_vals <= 256:
                recommendations.append("Data appears to be 8-bit (uint8) encoded")
            elif unique_vals <= 65536:
                recommendations.append("Data might be 16-bit (uint16) encoded")
        
        # Based on format detection
        if 'format_detected' in self.header_info:
            recommendations.append(f"Detected format: {self.header_info['format_detected']}")
        
        return recommendations
    
    def read_voxel_data(self, file_path: str, dimensions: Tuple[int, int, int], 
                       bytes_per_voxel: int = 1, header_size: int = 0,
                       dtype: str = 'uint8') -> np.ndarray:
        """
        Read voxel data based on determined format.
        
        Args:
            file_path: Path to the binary file
            dimensions: (x, y, z) dimensions
            bytes_per_voxel: Number of bytes per voxel
            header_size: Size of header to skip
            dtype: NumPy data type ('uint8', 'uint16', 'uint32')
            
        Returns:
            3D numpy array with voxel data
        """
        x, y, z = dimensions
        
        with open(file_path, 'rb') as f:
            # Skip header
            f.seek(header_size)
            
            # Read voxel data
            expected_bytes = x * y * z * bytes_per_voxel
            data = f.read(expected_bytes)
            
            if len(data) != expected_bytes:
                raise ValueError(f"Expected {expected_bytes} bytes, got {len(data)}")
            
            # Convert to numpy array
            voxel_array = np.frombuffer(data, dtype=getattr(np, dtype))
            
            # Reshape to 3D
            return voxel_array.reshape((z, y, x))  # Note: often stored as z, y, x
    
    def print_analysis_report(self, results: Dict):
        """Print a formatted analysis report."""
        print("\n" + "="*60)
        print("VOXEL FILE ANALYSIS REPORT")
        print("="*60)
        
        print(f"\nFile Size: {results['file_size']:,} bytes")
        
        print(f"\nHeader Analysis:")
        header = results['header_analysis']
        for key, value in header.items():
            print(f"  {key}: {value}")
        
        print(f"\nData Pattern Analysis:")
        data = results['data_analysis']
        for key, value in data.items():
            if key != 'dimension_scenarios':
                print(f"  {key}: {value}")
        
        if 'dimension_scenarios' in data and data['dimension_scenarios']:
            print(f"\nDimension-based Scenarios:")
            for i, scenario in enumerate(data['dimension_scenarios'][:3]):  # Show top 3
                print(f"  Scenario {i+1}: {scenario}")
        
        print(f"\nRecommendations:")
        for rec in results['recommendations']:
            print(f"  • {rec}")


# Example usage
if __name__ == "__main__":
    # Example analysis
    analyzer = VoxelBinaryAnalyzer()

    file = sys.argv[1]
    results = analyzer.analyze_file(file, (586, 340, 1878))
    results = analyzer.analyze_file(file, (586, 340, 1878))
    analyzer.print_analysis_report(results)
    voxel_data = analyzer.read_voxel_data(file, (586, 340, 1878), bytes_per_voxel=1)
    print(f'Voxel data shape: {voxel_data.shape}')
    print(f'Unique tissue values: {np.unique(voxel_data)}')