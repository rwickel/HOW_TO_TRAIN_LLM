import pymupdf  # PyMuPDF
import fitz
import pandas as pd

def read_pdf(pdf_path: str, unwanted_texts: list = None, add_table_to_text: bool = False, page_limit: int = None):
    doc = pymupdf.open(pdf_path)
    
    all_tables = []
    all_text = []

    num_total_pages = len(doc)
    
    if page_limit is not None:
        if page_limit <= 0:
            # If page_limit is zero or negative, process no pages.
            # You could also raise an error here if preferred:
            # raise ValueError("page_limit must be a positive integer.")
            pages_to_iterate_count = 0
        else:
            # Process up to page_limit pages, but not more than available
            pages_to_iterate_count = min(page_limit, num_total_pages)
    else:
        # If no limit is specified, process all pages
        pages_to_iterate_count = num_total_pages

    for page_number in range(pages_to_iterate_count):
        page = doc[page_number]
        print(f"\n=== Processing Page {page_number + 1} ===")

        # Get all text blocks (sorted top to bottom)
        text_blocks = page.get_text("blocks")
        text_blocks = sorted(text_blocks, key=lambda b: b[1])  # b[1] = y0 (top)

        # Store all page text
        page_text = page.get_text()
        
        # Remove unwanted text
        if unwanted_texts: # Check if unwanted_texts is not None
            for pattern in unwanted_texts:
                page_text = page_text.replace(pattern, "")

        all_text.append(page_text.strip())      
        
        print(f"\nText:\n{page_text}")

        # Find tables
        table_list = page.find_tables()
        # The `tables` attribute might not exist if no tables are found, 
        # or table_list itself might be the list of tables depending on PyMuPDF version.
        # It's safer to check if table_list is not None and iterate directly if it's a list.
        # Assuming table_list.tables is the correct way for your version:
        if table_list: # Check if any table objects were found
            tables = table_list.tables # This attribute might be version specific.
                                      # Directly iterating over table_list might be an option
                                      # page.find_tables() returns a TableFinder object
                                      # and table_list.tables gets the actual list of Table objects
            
            for idx, table in enumerate(tables, 1):
                bbox = table.bbox
                # Use pymupdf.Rect for consistency if fitz is not separately imported
                # rect = pymupdf.Rect(bbox) 
                rect = fitz.Rect(bbox) # Using fitz.Rect as in original code
                table_top = bbox[1]
                table_left = bbox[0]
                table_right = bbox[2]

                # Extract table content
                extracted_data = table.extract()
                if not extracted_data:
                    continue

                # Try to find nearest title block above the table
                title = "Unknown"
                # Ensure text_blocks is available and contains items
                if text_blocks: 
                    for block in reversed(text_blocks):
                        # Ensure block has at least 5 elements before unpacking
                        if len(block) >= 5:
                            block_x0, block_y0, block_x1, block_y1, text_content = block[:5]
                            if block_y1 <= table_top and abs(table_top - block_y1) < 100:
                                if block_x1 > table_left and block_x0 < table_right:
                                    title = text_content.strip()
                                    break
                        else:
                            # Handle blocks with unexpected structure if necessary
                            print(f"Warning: Text block has unexpected structure: {block}")


                # Save table with metadata
                df = pd.DataFrame(extracted_data)
                table_info = {
                    "page": page_number + 1,
                    "title": title,
                    "bbox": bbox,
                    "dataframe": df
                }
                all_tables.append(table_info)

                # Display summary
                print(f"\n\nTable")
                print(f"Title: {title}")
                print(df.to_string(index=False))

                if add_table_to_text:
                    # Ensure title is a string before concatenation
                    all_text.append(f"\n{str(title)}\n{df.to_string(index=False)}")

    doc.close() # It's good practice to close the document
    return all_tables, all_text

exclude_text = [
            "© ISO 2022"
            "© ISO 2022 – All rights reserved",
            "Normen-Download-DIN Media-Robert Wickel-KdNr.8450525-ID.XEKoxl0127MPxe2YrRrlTXYDJGWZtEpnSwYwINY1-2025-03-31 10:21:30",
            "COPYRIGHT PROTECTED DOCUMENT",
            "All rights reserved. Unless otherwise specified, or required in the context of its implementation, no part of this publication may",
            "be reproduced or utilized otherwise in any form or by any means, electronic or mechanical, including photocopying, or posting on",
            "the internet or an intranet, without prior written permission. Permission can be requested from either ISO at the address below",
            "or ISO’s member body in the country of the requester.",
            "ISO copyright office",
            "CP 401 • Ch. de Blandonnet 8",
            "CH-1214 Vernier, Geneva",
            "Phone: +41 22 749 01 11",
            "Email: copyright@iso.org",
            "Website: www.iso.org",
            "Published in Switzerland",
        ]

if __name__ == "__main__":
    pdf_path = "./data/ISO21448.PDF"
    tables, text_pages = read_pdf(pdf_path, exclude_text)

    # No CSV saving
    # Optional: Save full text
    with open("full_text_extracted.txt", "w", encoding="utf-8") as f:
        f.write("\n\n".join(text_pages))
        print("\nSaved full text to 'full_text_extracted.txt'")
