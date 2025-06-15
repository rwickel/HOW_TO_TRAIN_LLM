import os
import string
import pandas as pd
from spire.pdf import PdfDocument, PdfTableExtractor, PdfTextFinder
from spire.pdf.common import *
import re 

class PdfExtractor:
    """
    A class to perform page-by-page extraction from a PDF document.

    For each page, it extracts:
    1. Tables as pandas DataFrames.
    2. Prose text, defined as text on the page whose words do NOT appear
       in any of the tables on that same page.
    """

    def __init__(self, pdf_file_path: str):
        """
        Initializes the extractor with the path to the PDF file.
        
        Args:
            pdf_file_path (str): The full path to the PDF file.
            
        Raises:
            FileNotFoundError: If the specified PDF file does not exist.
        """
        if not os.path.exists(pdf_file_path):
            raise FileNotFoundError(f"Error: The file '{pdf_file_path}' was not found.")
        self.pdf_file = pdf_file_path
        self.doc = PdfDocument()
        self.doc.LoadFromFile(pdf_file_path)             

    def extract_content(self, grouped_lines, texts_to_remove, char_width=3):
        texts_to_remove = [t.strip() for t in texts_to_remove if t.strip()] 
        final_page_text_block = []

        for line in grouped_lines:
            if not line:
                continue

            # raw line
            line_text = " ".join(frag.Text.strip() for frag in line)

            # --- REMOVE UNWANTED PHRASES ---------------------------------
            for rem in texts_to_remove:
                pattern = re.escape(rem)
                line_text = re.sub(pattern, '', line_text, flags=re.IGNORECASE)

            # if everything was stripped out, go to next line
            if not line_text.strip():
                continue
            # --------------------------------------------------------------

            # tidy whitespace that may be left behind
            line_text = re.sub(r' {2,}', ' ', line_text).strip()

            # ---------- formatting (unchanged) ----------
            font_size  = line[0].TextStates[0].FontSize
            font_name  = str(line[0].TextStates[0].FontName)
            start_x    = line[0].Bounds[0].X
            padding    = ' ' * int(start_x / char_width)

            if font_size > 13:
                formatted_line = f"\n{padding}# {line_text}\n"
            elif font_size > 12:
                formatted_line = f"\n{padding}## {line_text}\n"
            elif font_size > 11:
                formatted_line = f"\n{padding}### {line_text}\n"
            elif 'bold' in font_name.lower():
                formatted_line = f"\n{padding}**{line_text}**\n"
            elif 'italic' in font_name.lower() or 'oblique' in font_name.lower():
                formatted_line = f"\n{padding}*{line_text}*\n"
            else:
                formatted_line = f"{padding}{line_text}"

            final_page_text_block.append(formatted_line)
        
        return  final_page_text_block   
    
    def process(self, output_dir="output",texts_to_remove: list = None, pages_to_skip: list = None, content_pages=None, y_tolerance=2, char_width = 3):
        """
        Processes the PDF page by page in a single loop.
        
        For each page, it extracts tables, collects all words from those tables,
        and then extracts prose text by filtering out the table words.
        """         
        os.makedirs(output_dir, exist_ok=True)
        base_filename = os.path.splitext(os.path.basename(self.pdf_file))[0]
        is_content = False

        if texts_to_remove is None:
            texts_to_remove = []

        if pages_to_skip is None:
            pages_to_skip = []   

        if content_pages is None:
            content_pages = []     
        
        for i in range(self.doc.Pages.Count):
            
            current_page_number = i + 1
            
            # --- SKIPPING LOGIC ---
            if current_page_number in pages_to_skip:
                print(f"Skipping Page {current_page_number} as requested.")
                continue

            if current_page_number in  content_pages:
                is_content =True
            
            page = self.doc.Pages[i]            
            finder = PdfTextFinder(page)
            fragments = finder.FindAllText() 
                
            # 2. GROUPING: Group sorted fragments into lines.
            grouped_lines = []
            if fragments:
                current_line = [fragments[0]]
                # Use the Y-coordinate of the first fragment as the reference for the line.
                line_start_y = fragments[0].Bounds[0].Y

                for frag in fragments[1:]:
                    # Check if the fragment's Y is close enough to the current line's start Y.
                    if abs(frag.Bounds[0].Y - line_start_y) <= y_tolerance:
                        current_line.append(frag)
                    else:
                        # A new line starts. Store the completed line.
                        grouped_lines.append(current_line)
                        # Start a new line with the current fragment.
                        current_line = [frag]
                        line_start_y = frag.Bounds[0].Y
                
                # Don't forget to add the very last line after the loop finishes.
                grouped_lines.append(current_line)

            # 3. FORMATTING: Process each line to add padding and join text. 
            final_page_text_block = self.extract_content(grouped_lines, texts_to_remove, char_width)
            
            output_content = "\n".join(final_page_text_block)

            page_filename = f"{base_filename}_page_{i + 1}.txt"
            output_filepath = os.path.join(output_dir, page_filename)
            
            with open(output_filepath, "w", encoding="utf-8") as f:
                f.write(output_content)
            
            print(f"Successfully processed and saved Page {i + 1} to '{output_filepath}'")   





# --- Main Execution ---
if __name__ == "__main__":
    PDF_FILE = r"resources\ISO_21448.pdf"
    OUTPUT_DIRECTORY = "output_files"

    unwanted_texts = [                
        "© ISO 2022",        
        "©",
        "– All rights reserved",
        "Normen-Download-DIN",
        "Media-Robert",
        "Wickel-KdNr.8450525-ID.XEKoxl0127MPxe2YrRrlTXYDJGWZtEpnSwYwINY1-2025-03-31 10:21:30"
    ]    

    content_pages = [3, 4] 

    skip_these_pages = []    
    
    try:
        pdf_processor = PdfExtractor(PDF_FILE) 
        # Pass the list of unwanted texts to the process method
        pdf_processor.process(
            output_dir=OUTPUT_DIRECTORY,
            pages_to_skip=skip_these_pages,
            texts_to_remove=unwanted_texts,
            content_pages=content_pages
        )
        print(f"\nProcessing complete. All files saved in '{OUTPUT_DIRECTORY}' directory.")
            
    except FileNotFoundError as e:
        print(f"\n[ERROR] File not found: {e}")
    except Exception as e:
        print(f"\n[ERROR] An unexpected error occurred: {e}")