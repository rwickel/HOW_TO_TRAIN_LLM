from reader.extract_from_pdf import ExtractTextInfoFromPDF

if __name__ == "__main__":
    # Example usage:
    # Set env variables or replace with actual values if needed.
    client_id="a15c1..." # insert your ADOBE ID
    client_secret="p8..." # insert ADOBE KEY
    
    # Provide path to input PDF and desired output directory
    input_pdf = "resources/ISO_26262-3_746716_EN.pdf"
    output_directory = "output/ExtractFromPDF"

    #Step 1 Extract elements from PDF
    pdf_name = ExtractTextInfoFromPDF.run(input_pdf, output_directory, client_id, client_secret)
    print(f"Processed PDF name: {pdf_name}")

    


