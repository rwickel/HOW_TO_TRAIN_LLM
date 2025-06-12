import json
import re
import os
import csv
import io
from collections import defaultdict

class PDFContentExtractor:
    def __init__(self, main_json_file_path):
        """
        Initializes the extractor with the main JSON file path.

        Args:
            main_json_file_path (str): Path to the main JSON file from Adobe PDF Extract API.
        """
        self.main_json_file_path = os.path.normpath(main_json_file_path)
        self.base_dir = os.path.dirname(self.main_json_file_path)
        if not self.base_dir:  # If path is just a filename, current dir is base
            self.base_dir = "."
        
        self.document_data = self._load_json_data()
        self.all_doc_elements = self.document_data.get('elements', []) if self.document_data else []
        
        self.parsed_headings_with_indices = None # To store output of parse_all_document_headings
        self.headings_with_associated_text = None # To store output of associate_text_with_headings

    def _load_json_data(self):
        """Loads the main JSON data from the file path specified during initialization."""
        try:
            with open(self.main_json_file_path, 'r', encoding='utf-8') as f:
                data = json.load(f)
            print(f"Successfully loaded JSON from: {self.main_json_file_path}")
            return data
        except FileNotFoundError:
            print(f"Error: Main JSON file not found at '{self.main_json_file_path}'.")
        except json.JSONDecodeError as e:
            print(f"Error decoding main JSON from file '{self.main_json_file_path}': {e}")
        except Exception as e:
            print(f"An unexpected error occurred while loading '{self.main_json_file_path}': {e}")
        return {} # Return empty dict on error

    @staticmethod
    def _extract_heading_level_from_path(path_str):
        if not path_str: return None
        match = re.search(r'/H([1-6])', path_str)
        if match: return int(match.group(1))
        return None

    @staticmethod
    def _get_base_heading_path(path_str):
        if not path_str: return None
        match = re.match(r'(//Document/H[1-6](\[[0-9]+\])?)', path_str)
        if match: return match.group(1)
        return None

    def parse_all_document_headings(self, max_level=3):
        """
        Parses structured headings from the loaded document data up to specified level.
        Stores results in self.parsed_headings_with_indices and returns them.
        
        Args:
            max_level (int): Maximum heading level to include (default: 3)
        """
        if not self.all_doc_elements:
            print("No elements to parse for headings.")
            self.parsed_headings_with_indices = []
            return self.parsed_headings_with_indices

        extracted_headings = []
        current_heading_details = None 

        for element_idx, element in enumerate(self.all_doc_elements):
            path = element.get('Path')
            text = element.get('Text', '').strip() 
            page = element.get('Page')

            if not path or (not text and (path is None or '//Document/Figure' not in path)):
                if current_heading_details: 
                    extracted_headings.append(current_heading_details)
                    current_heading_details = None
                continue

            heading_level = self._extract_heading_level_from_path(path)
            
            # Skip headings beyond our max level
            if heading_level and heading_level > max_level:
                continue
                
            if heading_level: 
                element_base_path = self._get_base_heading_path(path)
                if not element_base_path: 
                    if current_heading_details:
                        extracted_headings.append(current_heading_details)
                        current_heading_details = None
                    continue

                if current_heading_details and \
                current_heading_details['base_path'] == element_base_path and \
                current_heading_details['page'] == page:
                    if text: 
                        if current_heading_details['title'] and \
                        not current_heading_details['title'].endswith(' ') and \
                        not text.startswith(' '):
                            current_heading_details['title'] += " "
                        current_heading_details['title'] += text
                    current_heading_details['last_element_idx_for_title'] = element_idx
                else:
                    if current_heading_details: 
                        extracted_headings.append(current_heading_details)
                    current_heading_details = {
                        'level': heading_level, 'title': text, 'page': page,
                        'base_path': element_base_path, 
                        'first_element_idx_for_title': element_idx,
                        'last_element_idx_for_title': element_idx
                    }
            else: 
                if current_heading_details: 
                    extracted_headings.append(current_heading_details)
                    current_heading_details = None
                    
        if current_heading_details: 
            extracted_headings.append(current_heading_details)

        final_headings_list = []
        for h_info in extracted_headings:
            cleaned_title = h_info['title'].strip()
            if cleaned_title or h_info.get('base_path'):
                final_headings_list.append({
                    'level': h_info['level'], 'title': cleaned_title, 'page': h_info['page'],
                    'path': h_info['base_path'],
                    'first_element_idx': h_info['first_element_idx_for_title'],
                    'last_element_idx': h_info['last_element_idx_for_title']
                })
        
        self.parsed_headings_with_indices = final_headings_list
        return self.parsed_headings_with_indices   
    
    def get_raw_csv_content_as_string(self, csv_file_path):
        """
        Reads the entire content of a CSV file and returns it as a single string,
        preserving all original formatting, commas, and line breaks.

        Args:
            csv_file_path (str): The full path to the CSV file.

        Returns:
            str: A single string containing the raw content of the CSV file,
                 or an error message string if the file cannot be read.
        """
        csv_filename = os.path.basename(csv_file_path)
        try:
            with open(csv_file_path, 'r', encoding='utf-8-sig') as file:
                raw_content = file.read()
            return raw_content
        except FileNotFoundError:
            print(f"Error: CSV file not found at {csv_file_path}")
            return f"[Raw content from CSV ({csv_filename}): File not found]"
        except Exception as e:
            print(f"Error reading raw content from CSV file {csv_file_path}: {e}")
            return f"[Raw content from CSV ({csv_filename}): Error reading file - {e}]"   
    
    def associate_text_with_headings(self, parsed_headings=None):
        """
        Associates text content (paragraphs, figures, stringified tables) 
        with each heading. Uses self.parsed_headings_with_indices if parsed_headings is None.
        """
        if parsed_headings is None:
            if self.parsed_headings_with_indices is None:
                self.parse_all_document_headings() # Ensure headings are parsed
            parsed_headings = self.parsed_headings_with_indices
        
        if not parsed_headings:
            print("No parsed headings available to associate text with.")
            self.headings_with_associated_text = []
            return self.headings_with_associated_text

        updated_headings = []
        for i, current_heading in enumerate(parsed_headings):
            heading_data_to_store = current_heading.copy()
            heading_data_to_store['associated_text_blocks'] = []

            if 'last_element_idx' not in current_heading or 'first_element_idx' not in current_heading:
                print(f"Warning: Heading '{current_heading.get('title')}' is missing index information. Skipping text association.")
                updated_headings.append(heading_data_to_store)
                continue

            content_start_index = current_heading['last_element_idx'] + 1
            content_end_index = len(self.all_doc_elements) 
            if i + 1 < len(parsed_headings):
                next_heading = parsed_headings[i+1]
                if 'first_element_idx' in next_heading:
                    content_end_index = next_heading['first_element_idx']
                else:
                    print(f"Warning: Next heading '{next_heading.get('title')}' is missing index. Content for '{current_heading.get('title')}' might extend too far.")

            for j in range(content_start_index, content_end_index):
                content_element = self.all_doc_elements[j]
                element_path = content_element.get('Path', '')
                element_text = content_element.get('Text', '').strip()
                block_to_add = None

                if element_path.startswith('//Document/L'): # ignore links
                    continue              
                
                if element_path and element_path.startswith("//Document/Table"):
                    table_text_rep = f"[Table: Error or no CSV data ({element_path})]"
                    file_paths = content_element.get('filePaths')
                    csv_rel_path = next((fp for fp in file_paths if fp.lower().endswith('.csv')), None) if file_paths else None
                    if csv_rel_path:
                        full_csv_path = os.path.normpath(os.path.join(self.base_dir, csv_rel_path))
                        table_text_rep = f"```csv\n{self.get_raw_csv_content_as_string(full_csv_path)}```"                       
                        block_to_add = {'path': element_path, 'text': table_text_rep, 'page': content_element.get('Page')}
                    else:
                        table_text_rep = f"[Table: No CSV file path found ({element_path})]"
                        block_to_add = None
                elif element_text:
                    block_to_add = {'path': element_path, 'text': element_text, 'page': content_element.get('Page')}
                elif element_path and "//Document/Figure" in element_path: 
                    block_to_add = {
                                    'path': element_path,
                                    'text': "",
                                    'url': content_element.get('filePaths', []),  # Store as list
                                    'alternate_text': content_element.get('alternate_text', ''),  # Store as string
                                    'page': content_element.get('Page')
                                    }
                if block_to_add:
                    heading_data_to_store['associated_text_blocks'].append(block_to_add)
                    block_to_add = None
            updated_headings.append(heading_data_to_store)
        
        self.headings_with_associated_text = updated_headings
        return self.headings_with_associated_text    

def print_hierarchy(headings, indent=0):
    prefix = "    " * indent + "- "
    for heading in headings:
        print(f"{prefix}{heading['title']} (p.{heading['page']})")
        print_hierarchy(heading['subheadings'], indent + 1)


def group_by_level(headings_data):
    levels = defaultdict(list)
    for heading in headings_data:
        levels[heading['level']].append({
            'title': heading['title'],
            'page': heading['page']
        })
    return dict(levels)

def organize_headings(headings_data):
    organized = []
    stack = []  # To keep track of current hierarchy
    
    for heading in headings_data:
        level = heading['level']
        title = heading['title']
        
        # Remove any existing higher or equal levels from stack
        while stack and stack[-1]['level'] >= level:
            stack.pop()
            
        # Create the new entry
        new_entry = {
            'title': title,
            'level': level,
            'page': heading['page'],
            'subheadings': []
        }
        
        # Add to the appropriate parent or root
        if stack:
            stack[-1]['subheadings'].append(new_entry)
        else:
            organized.append(new_entry)
            
        # Push to stack
        stack.append(new_entry)
    
    return organized

def print_hierarchy_up_to_level(headings, max_level=3, indent=0):
    """Print heading hierarchy up to specified level"""
    if indent//4 >= max_level:  # Each level is indented by 4 spaces
        return
    prefix = "    " * indent + "- "
    for heading in headings:
        print(f"{prefix}{heading['title']} (p.{heading['page']})")
        print_hierarchy_up_to_level(heading['subheadings'], max_level, indent + 1)

def extract_headings_up_to_level(headings_data, max_level=3):
    """
    Extract headings up to a specified level (default: level 3)
    
    Args:
        headings_data: List of heading dictionaries from associate_text_with_headings()
        max_level: Maximum heading level to include (default 3)
    
    Returns:
        List of filtered headings with their associated content
    """
    return [heading for heading in headings_data if heading.get('level', 1) <= max_level]        

# --- How to use the Class ---
if __name__ == "__main__":
    json_file_path_str = r"output\ExtractFromPDF\ISO_21448\ISO_21448.json" # Your JSON file path
    
    extractor = PDFContentExtractor(json_file_path_str)    

    if extractor.document_data:        
        headings_with_text = extractor.associate_text_with_headings() 

        # Filter to only include headings up to level 3
        level_3_headings = extract_headings_up_to_level(headings_with_text, max_level=3)

        #print_hierarchy(headings_with_text)
        
        if headings_with_text:
            print("\n--- Headings with Associated Text & Content ---")
            for heading_info in headings_with_text:
                indent = "  " * (heading_info.get('level', 1) - 1)
                title = heading_info.get('title', 'Untitled Heading')
                page = heading_info.get('page', -1) 
                path = heading_info.get('path', 'N/A')                
                print(f"\nLevel {heading_info.get('level', 'N/A')}: {title} (Page: {page + 1 if isinstance(page, int) else 'N/A'}, Path: {path})")
                
                associated_blocks = heading_info.get('associated_text_blocks', [])
                if associated_blocks:
                    print(f"{indent}  Associated Content Blocks ({len(associated_blocks)}):")                    
                    for block_idx, block in enumerate(associated_blocks):                        
                        print(f"\n{block['text']}")                        
        

        
    