from reader.extract_from_pdf import ExtractTextInfoFromPDF

if __name__ == "__main__":
    # Example usage:
    # Set env variables or replace with actual values if needed.
    client_id="a15c19e647c74c89b6837c5f40bbc4c3"
    client_secret="p8e-Y55EkNWC54xQ_7R-g7w0U_4XGY48UHPE"
    
    # Provide path to input PDF and desired output directory
    input_pdf = "resources/ISO_26262-3_746716_EN.pdf"
    output_directory = "output/ExtractFromPDF"

    #Step 1 Extract elements from PDF
    pdf_name = ExtractTextInfoFromPDF.run(input_pdf, output_directory, client_id, client_secret)
    print(f"Processed PDF name: {pdf_name}")

    #input_json = "output/ExtractTextInfoFromPDF/ISO_26262-6_746747_EN.json"


