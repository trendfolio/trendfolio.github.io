import json
import sys
from pathlib import Path

def check_json_file(file_path):
    """
    Check JSON file for validity and common issues.
    Returns tuple (is_valid, error_message)
    """
    try:
        # First try to read file as regular text to check for BOM
        with open(file_path, 'rb') as f:
            raw_data = f.read()
            
        # Check for BOM
        if raw_data.startswith(b'\xef\xbb\xbf'):
            return False, "File contains UTF-8 BOM marker. Please save the file without BOM."

        # Try to parse JSON
        with open(file_path, 'r', encoding='utf-8') as f:
            data = json.load(f)

        # Validate expected structure
        required_keys = {'metadata', 'instruments'}
        if not all(key in data for key in required_keys):
            return False, f"Missing required top-level keys. Expected {required_keys}"

        required_metadata = {'last_update', 'start_year', 'instruments'}
        if not all(key in data['metadata'] for key in required_metadata):
            return False, f"Missing required metadata keys. Expected {required_metadata}"

        # Check if instruments in metadata match instruments data
        metadata_instruments = set(data['metadata']['instruments'])
        actual_instruments = set(data['instruments'].keys())
        if metadata_instruments != actual_instruments:
            return False, (f"Mismatch between metadata instruments and actual data.\n"
                         f"In metadata: {metadata_instruments}\n"
                         f"In data: {actual_instruments}")

        return True, "JSON file is valid and matches expected structure."

    except json.JSONDecodeError as e:
        # Get more context around the error
        lines = raw_data.decode('utf-8', errors='replace').split('\n')
        error_line = e.lineno - 1  # zero-based index
        context = '\n'.join(lines[max(0, error_line-2):error_line+3])
        
        return False, (f"JSON parsing error: {str(e)}\n"
                      f"Error location: line {e.lineno}, column {e.colno}\n"
                      f"Context around error:\n{context}")
    
    except Exception as e:
        return False, f"Unexpected error: {str(e)}"

def main():
    if len(sys.argv) != 2:
        print("Usage: python check_json.py <path_to_json_file>")
        sys.exit(1)

    file_path = Path(sys.argv[1])
    if not file_path.exists():
        print(f"Error: File {file_path} does not exist")
        sys.exit(1)

    is_valid, message = check_json_file(file_path)
    print("Status:", "Valid" if is_valid else "Invalid")
    print("Message:", message)
    sys.exit(0 if is_valid else 1)

if __name__ == "__main__":
    main() 