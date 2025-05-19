# pdf_reader_utils.py

from typing import List, Tuple, Optional, Any, Dict
import pymupdf # PyMuPDF, also known as Fitz
import pandas as pd

def read_pdf(
    pdf_path: str,
    unwanted_texts: Optional[List[str]] = None,
    add_table_to_text: bool = False,
    page_limit: Optional[int] = None
) -> Tuple[List[Dict[str, Any]], List[str]]:
    """
    Reads a PDF file, extracts text and tables from its pages.

    Args:
        pdf_path (str): The file path to the PDF document.
        unwanted_texts (Optional[List[str]]): A list of strings to remove from the
                                             extracted text. Useful for removing common
                                             headers, footers, or watermarks.
        add_table_to_text (bool): If True, appends the string representation of
                                  extracted tables to the page's text. Defaults to False.
        page_limit (Optional[int]): An optional limit on the number of pages to process.
                                    If None, all pages are processed.

    Returns:
        Tuple[List[Dict[str, Any]], List[str]]:
            - A list of dictionaries, where each dictionary contains information
              about an extracted table (page, title, bbox, dataframe).
            - A list of strings, where each string is the text content of a page,
              after applying exclusions and optionally adding table text.

    Raises:
        FileNotFoundError: If the pdf_path does not point to an existing file.
        Exception: For other PDF parsing related errors (e.g., corrupted PDF).
    """
    try:
        doc = pymupdf.open(pdf_path)
    except RuntimeError as e:
        if "no such file or directory" in str(e).lower() or \
           "cannot open document" in str(e).lower():
            raise FileNotFoundError(f"Error: The PDF file was not found or cannot be opened at '{pdf_path}'. Original error: {e}")
        raise Exception(f"Error opening PDF '{pdf_path}': {e}")

    all_tables: List[Dict[str, Any]] = []
    all_text_pages: List[str] = [] # Changed variable name for clarity

    num_total_pages = len(doc)
    
    pages_to_iterate_count: int
    if page_limit is not None:
        if page_limit <= 0:
            pages_to_iterate_count = 0
        else:
            pages_to_iterate_count = min(page_limit, num_total_pages)
    else:
        pages_to_iterate_count = num_total_pages    

    for page_number in range(pages_to_iterate_count):
        page = doc[page_number]
        print(f"\n=== Processing Page {page_number + 1} ===")

        # Get all text blocks (sorted top to bottom) for potential table title extraction
        text_blocks = page.get_text("blocks")
        text_blocks = sorted(text_blocks, key=lambda b: b[1])  # b[1] = y0 (top coordinate)

        # Store all page text
        page_text_content = page.get_text() # Renamed for clarity
        
        # Remove unwanted text
        if unwanted_texts:
            for pattern in unwanted_texts:
                page_text_content = page_text_content.replace(pattern, "")
        
        current_page_texts = [page_text_content.strip()]        

        # Find tables
        # page.find_tables() returns a TableFinder object.
        # TableFinder.tables gives the list of Table objects.
        table_finder = page.find_tables()
        
        if table_finder.tables: # Check if tables were found
            print(f"Found {len(table_finder.tables)} table(s) on page {page_number + 1}.")
            for idx, table in enumerate(table_finder.tables, 1):
                bbox = table.bbox # (x0, y0, x1, y1)
                # Using pymupdf.Rect for consistency as pymupdf.open() is used.
                # rect = pymupdf.Rect(bbox) 
                table_top = bbox[1]
                table_left = bbox[0]
                table_right = bbox[2]

                # Extract table content
                extracted_data = table.extract()
                if not extracted_data: # Skip if table extraction yields nothing
                    print(f"Table {idx} on page {page_number + 1} had no extractable data.")
                    continue

                # Try to find nearest title block above the table
                title = "Unknown Table" # Default title
                if text_blocks: 
                    for block in reversed(text_blocks):
                        # block structure: (x0, y0, x1, y1, "lines in block", block_no, block_type)
                        # We need the text content which is block[4]
                        if len(block) >= 5: # Ensure block has at least 5 elements
                            block_x0, block_y0, block_x1, block_y1, text_content_lines = block[:5]
                            # Check if block is above the table and within a certain vertical distance
                            if block_y1 <= table_top and abs(table_top - block_y1) < 100: # 100px threshold
                                # Check for horizontal overlap or proximity
                                if block_x1 > table_left and block_x0 < table_right:
                                    title = text_content_lines.strip().replace("\n", " ") # Clean up title
                                    break # Found a plausible title
                        else:
                            print(f"Warning: Text block has unexpected structure: {block}")

                # Save table with metadata
                try:
                    df = pd.DataFrame(extracted_data[1:], columns=extracted_data[0]) # Assume first row is header
                except Exception as e:
                    print(f"Could not create DataFrame for table {idx} on page {page_number + 1}: {e}. Using raw data.")
                    df = pd.DataFrame(extracted_data) # Fallback

                table_info = {
                    "page": page_number + 1,
                    "title": title,
                    "bbox": bbox,
                    "dataframe": df
                }
                all_tables.append(table_info)                

                if add_table_to_text:
                    # Ensure title is a string before concatenation
                    table_string = f"\nTable Title: {str(title)}\n{df.to_string(index=False)}"
                    current_page_texts.append(table_string)
        else:
            print(f"No tables found on page {page_number + 1}.")
        
        all_text_pages.append("\n".join(current_page_texts))

    doc.close()
    print(f"\nFinished PDF processing. Extracted {len(all_tables)} tables and text from {len(all_text_pages)} pages.")
    return all_tables, all_text_pages
