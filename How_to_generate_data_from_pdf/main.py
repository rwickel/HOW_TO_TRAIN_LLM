# main.py
import os
import json
from typing import List
from repl.llm import LLM
from reader.config import config
from dataclasses import asdict
from reader.types import PageQAData
from reader.pdf_reader_utils import read_pdf # This imports from the provided pdf_reader_utils.py
from reader.llm_interaction import execute_qa_extraction

def process_pdf_file(pdf_path: str, llm_client: LLM, output_dir: str, page_limit: int, add_tables_to_page_text: bool):
    print(f"\nProcessing PDF: {pdf_path}")

    filename = os.path.splitext(os.path.basename(pdf_path))[0]
    
    try:
        extracted_tables, pages_text_content = read_pdf(
            pdf_path=pdf_path,
            unwanted_texts=config.exclude_text,
            add_table_to_text=add_tables_to_page_text,
            page_limit=page_limit
        )
        print(f" - Text extracted from {len(pages_text_content)} page(s).")
        print(f" - Found {len(extracted_tables)} table(s).")

        all_qa_responses = []
        page_content: List[PageQAData] = []

        for i, page_text in enumerate(pages_text_content):
            print(f"\n  - Processing Page {i + 1}")
            qa_response = execute_qa_extraction(
                page_text=page_text,
                llm_client=llm_client,
                system_prompt=config.system_prompt.format(filename=filename, page=(i + 1)),
                response_schema=config.response_schema if config.response_schema else None
            )

            if qa_response['valid'] and qa_response['data']:
                current_page_data = PageQAData(
                    doc_id=pdf_path,
                    page_number=i + 1,
                    page_text=page_text,
                    qa_pairs=qa_response['data'],
                    raw_response=qa_response['raw_content']
                )
                all_qa_responses.extend(qa_response['data'])  # Flatten
                page_content.append(current_page_data)

        if all_qa_responses:   
            # Save flat Q/A
            all_qa_output_path = os.path.join(output_dir, f"{filename}_qa_pairs.json")
            with open(all_qa_output_path, "w", encoding="utf-8") as f:
                json.dump(all_qa_responses, f, indent=2, ensure_ascii=False)

            # Save structured page content
            structured_output_path = os.path.join(output_dir, f"{filename}_qa_results.json")
            with open(structured_output_path, "w", encoding="utf-8") as f:
                json.dump([asdict(p) for p in page_content], f, indent=2, ensure_ascii=False)

            print(f"  - Q/A saved: {all_qa_output_path}")
        else:
            print(f"  - No Q/A pairs extracted for {pdf_path}")

    except Exception as e:
        print(f"  [ERROR] Failed to process '{pdf_path}': {e}")

def main(model:str=None, output_dir: str = None):
    input_folder = config.pdf_folder  # Add this to your config
        
    if output_dir is None:
        output_dir = config.output_folder    
    
    if model is not None:
        config.model = model
        print(f"Model set to: {config.model}")
    
    if not os.path.exists(output_dir):
        os.makedirs(output_dir, exist_ok=True)

    page_limit = None
    add_tables_to_page_text = False

    try:
        llm_client = LLM(model=config.model)
        print(f"LLM initialized: {llm_client.model}")
    except Exception as e:
        print(f"[ERROR] LLM initialization failed: {e}")
        return

    pdf_files = [f for f in os.listdir(input_folder) if f.lower().endswith(".pdf")]

    if not pdf_files:
        print(f"No PDF files found in folder: {input_folder}")
        return

    print(f"Found {len(pdf_files)} PDF(s) in folder '{input_folder}'.")

    for pdf_file in pdf_files:
        pdf_path = os.path.join(input_folder, pdf_file)
        process_pdf_file(pdf_path, llm_client, output_dir, page_limit, add_tables_to_page_text)

if __name__ == "__main__":  
    #main("phi4:14b","./phi4_14b_output")   
    #main("qwen2.5:14b","./qwen2_5_14b_output")
    main("llama3.2:latest","./llama3_2_output")

    
    
