"""
This class is responsible for crawling a working directorty and finding files based on user specified criteria.
It provides methods for:
-finding files based on extension type
-finding files based on name patterns (e.g., containing 'phy' or 'msur')
-building merge_configs for BioDataProcessor based on found files and user specified merge keys
Structure:
-find_files method for crawling and filtering files
-build_merge_configs method for automating the creation of merge_configs for BioDataProcessor
Arguments:
-self: the instance of the class
-root_dir: the directory to crawl for files
-extension: the file extension to filter by (e.g., '.sas7bdat')
-name_pattern: a regex pattern to filter file names (e.g., 'phy|msur
"""

from pathlib import Path
import re

class DirectoryCrawler:
    def __init__(self, root_dir: str):
        # Path().resolve() automatically cleans up messy slashes and finds the absolute path
        self.root_dir = Path(root_dir).resolve()

    def find_files(self, extension: str = None, name_pattern: str = None) -> list:
        """
        Crawls the directory and its subfolders for files.
        - extension: e.g., '.sas7bdat' or '.csv'
        - name_pattern: A regex string to filter file names (e.g., 'phy|msur')
        """
        if not self.root_dir.exists() or not self.root_dir.is_dir():
            print(f"ERROR: Directory '{self.root_dir}' does not exist.")
            return []

        # rglob is the modern pathlib equivalent of os.walk (recursive glob)
        search_target = f"*{extension}" if extension else "*"

        matched_paths = []
        for file_path in self.root_dir.rglob(search_target):
            if file_path.is_file():
                # Filter by name if a pattern was provided
                if name_pattern:
                    if re.search(name_pattern, file_path.name, re.IGNORECASE):
                        # Convert the Path object back to a string for your DataLoader
                        matched_paths.append(str(file_path))
                else:
                    matched_paths.append(str(file_path))
        
        print(f"  DirectoryCrawler: Found {len(matched_paths)} file(s) matching criteria.")
        return matched_paths