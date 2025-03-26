import json

def clean_json_file():
    try:
        # Read the file in binary mode first to check for BOM
        with open('forecast_data.json', 'rb') as f:
            content = f.read()
        
        # Remove BOM if present
        if content.startswith(b'\xef\xbb\xbf'):
            content = content[3:]
        
        # Decode to string
        content_str = content.decode('utf-8')
        
        # Parse JSON to validate it
        data = json.loads(content_str)
        
        # Write back clean JSON
        with open('forecast_data.json', 'w', encoding='utf-8') as f:
            json.dump(data, f, indent=2)
            
        print("JSON file has been cleaned and validated successfully!")
        
    except json.JSONDecodeError as e:
        print(f"JSON Error: {str(e)}")
        print("\nFirst 50 characters of the file:")
        print(repr(content_str[:50]))
        
    except Exception as e:
        print(f"Error: {str(e)}")

if __name__ == "__main__":
    clean_json_file() 