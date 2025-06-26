import gzip
import zipfile
import tarfile
import shutil
import os
from pathlib import Path
from typing import List, Optional


def extract_gz_file(gz_file_path: str, output_path: Optional[str] = None) -> str:
    """
    Extract a .gz file (single file compression).
    
    Args:
        gz_file_path: Path to the .gz file
        output_path: Optional output path. If None, removes .gz extension
        
    Returns:
        Path to extracted file
    """
    if output_path is None:
        output_path = gz_file_path.replace('.gz', '')
    
    with gzip.open(gz_file_path, 'rb') as gz_file:
        with open(output_path, 'wb') as output_file:
            shutil.copyfileobj(gz_file, output_file)
    
    print(f"Extracted: {gz_file_path} -> {output_path}")
    return output_path


def extract_zip_archive(zip_file_path: str, extract_to: str = ".") -> List[str]:
    """
    Extract a .zip archive.
    
    Args:
        zip_file_path: Path to the .zip file
        extract_to: Directory to extract to (default: current directory)
        
    Returns:
        List of extracted file paths
    """
    extracted_files = []
    
    with zipfile.ZipFile(zip_file_path, 'r') as zip_ref:
        # Get list of files in archive
        file_list = zip_ref.namelist()
        print(f"Files in archive: {file_list}")
        
        # Extract all files
        zip_ref.extractall(extract_to)
        
        # Build list of extracted file paths
        for file_name in file_list:
            extracted_path = os.path.join(extract_to, file_name)
            extracted_files.append(extracted_path)
            print(f"Extracted: {file_name}")
    
    return extracted_files


def extract_tar_archive(tar_file_path: str, extract_to: str = ".") -> List[str]:
    """
    Extract .tar, .tar.gz, .tar.bz2, or .tar.xz archives.
    
    Args:
        tar_file_path: Path to the tar archive
        extract_to: Directory to extract to (default: current directory)
        
    Returns:
        List of extracted file paths
    """
    extracted_files = []
    
    # Auto-detect compression type
    with tarfile.open(tar_file_path, 'r:*') as tar_ref:
        # Get list of files in archive
        file_list = tar_ref.getnames()
        print(f"Files in archive: {file_list}")
        
        # Extract all files
        tar_ref.extractall(extract_to)
        
        # Build list of extracted file paths
        for file_name in file_list:
            extracted_path = os.path.join(extract_to, file_name)
            extracted_files.append(extracted_path)
            print(f"Extracted: {file_name}")
    
    return extracted_files


def extract_archive(archive_path: str, extract_to: str = ".") -> List[str]:
    """
    Auto-detect archive type and extract accordingly.
    
    Args:
        archive_path: Path to the archive file
        extract_to: Directory to extract to (default: current directory)
        
    Returns:
        List of extracted file paths
    """
    archive_path = Path(archive_path)
    
    if not archive_path.exists():
        raise FileNotFoundError(f"Archive not found: {archive_path}")
    
    # Create extraction directory if it doesn't exist
    Path(extract_to).mkdir(parents=True, exist_ok=True)
    
    # Determine archive type by extension
    if archive_path.suffix.lower() == '.gz' and not archive_path.name.endswith('.tar.gz'):
        # Single file .gz compression
        output_path = extract_gz_file(str(archive_path), 
                                    os.path.join(extract_to, archive_path.stem))
        return [output_path]
    
    elif archive_path.suffix.lower() == '.zip':
        return extract_zip_archive(str(archive_path), extract_to)
    
    elif any(archive_path.name.endswith(ext) for ext in ['.tar', '.tar.gz', '.tar.bz2', '.tar.xz', '.tgz']):
        return extract_tar_archive(str(archive_path), extract_to)
    
    else:
        raise ValueError(f"Unsupported archive format: {archive_path.suffix}")


def list_archive_contents(archive_path: str) -> List[str]:
    """
    List contents of an archive without extracting.
    
    Args:
        archive_path: Path to the archive file
        
    Returns:
        List of file names in the archive
    """
    archive_path = Path(archive_path)
    
    if archive_path.suffix.lower() == '.zip':
        with zipfile.ZipFile(archive_path, 'r') as zip_ref:
            return zip_ref.namelist()
    
    elif any(archive_path.name.endswith(ext) for ext in ['.tar', '.tar.gz', '.tar.bz2', '.tar.xz', '.tgz']):
        with tarfile.open(archive_path, 'r:*') as tar_ref:
            return tar_ref.getnames()
    
    elif archive_path.suffix.lower() == '.gz':
        # For .gz files, we can't list contents without extracting
        return [archive_path.stem]
    
    else:
        raise ValueError(f"Unsupported archive format: {archive_path.suffix}")


# Example usage and command-line interface
if __name__ == "__main__":
    import sys
    
    # Example usage
    if len(sys.argv) > 1:
        archive_file = sys.argv[1]
        output_dir = sys.argv[2] if len(sys.argv) > 2 else "extracted"
        
        try:
            print(f"Extracting: {archive_file}")
            print(f"Output directory: {output_dir}")
            
            # List contents first
            print("\nArchive contents:")
            contents = list_archive_contents(archive_file)
            for item in contents:
                print(f"  {item}")
            
            # Extract
            print(f"\nExtracting...")
            extracted_files = extract_archive(archive_file, output_dir)
            
            print(f"\nExtraction complete!")
            print(f"Extracted {len(extracted_files)} files to: {output_dir}")
            
        except Exception as e:
            print(f"Error: {e}")
    
    else:
        # Demo with example files
        print("Archive Extraction Examples:")
        print("=" * 50)
        
        # Example for your specific case
        print("\nFor your voxel model files:")
        print("# Extract the .raw.gz file")
        print("extracted_files = extract_archive('af_man_1mm.raw.gz', 'extracted')")
        print("# This will create: extracted/af_man_1mm.raw")
        
        print("\nCommand line usage:")
        print("python extract_archives.py archive_file.zip [output_directory]")
        print("python extract_archives.py af_man_1mm.raw.gz voxel_data")