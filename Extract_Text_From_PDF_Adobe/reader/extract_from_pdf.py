import logging
import os
import json
import zipfile
import io
from datetime import datetime

from adobe.pdfservices.operation.auth.service_principal_credentials import ServicePrincipalCredentials
from adobe.pdfservices.operation.exception.exceptions import ServiceApiException, ServiceUsageException, SdkException
from adobe.pdfservices.operation.pdf_services_media_type import PDFServicesMediaType
from adobe.pdfservices.operation.io.cloud_asset import CloudAsset
from adobe.pdfservices.operation.io.stream_asset import StreamAsset
from adobe.pdfservices.operation.pdf_services import PDFServices
from adobe.pdfservices.operation.pdfjobs.jobs.extract_pdf_job import ExtractPDFJob
from adobe.pdfservices.operation.pdfjobs.params.extract_pdf.extract_element_type import ExtractElementType
from adobe.pdfservices.operation.pdfjobs.params.extract_pdf.extract_pdf_params import ExtractPDFParams
from adobe.pdfservices.operation.pdfjobs.result.extract_pdf_result import ExtractPDFResult
from adobe.pdfservices.operation.pdfjobs.params.extract_pdf.table_structure_type import TableStructureType
from adobe.pdfservices.operation.pdfjobs.params.extract_pdf.extract_renditions_element_type import ExtractRenditionsElementType

# Initialize the logger
logging.basicConfig(level=logging.INFO)


class ExtractTextInfoFromPDF:
    def __init__(self, pdf_path: str, output_dir: str, client_id: str, client_secret: str):
        
        self.pdf_name = os.path.splitext(os.path.basename(pdf_path))[0]
        try:            

            # Load PDF file
            with open(pdf_path, 'rb') as file:
                input_stream = file.read()

            # Setup credentials from environment variables (recommended for security)
            credentials = ServicePrincipalCredentials(
                client_id=client_id,
                client_secret=client_secret
            )

            # Create PDFServices instance
            pdf_services = PDFServices(credentials=credentials)

            # Upload input file
            input_asset = pdf_services.upload(input_stream=input_stream, mime_type=PDFServicesMediaType.PDF)

            # Set extract parameters
            extract_pdf_params = ExtractPDFParams(
                elements_to_extract=[
                    ExtractElementType.TEXT,
                    ExtractElementType.TABLES,                    
                ],
                elements_to_extract_renditions=[
                #     ExtractRenditionsElementType.TABLES,
                     ExtractRenditionsElementType.FIGURES
                ],
                table_structure_type=TableStructureType.CSV,
                styling_info=False,
                add_char_info=False,
            )

            # Create and submit extract job
            extract_pdf_job = ExtractPDFJob(input_asset=input_asset, extract_pdf_params=extract_pdf_params)
            location = pdf_services.submit(extract_pdf_job)

            # Get job result
            pdf_services_response = pdf_services.get_job_result(location, ExtractPDFResult)

            # Download result
            result_asset: CloudAsset = pdf_services_response.get_result().get_resource()
            stream_asset: StreamAsset = pdf_services.get_content(result_asset)

            # Process ZIP content
            self.extract_zip_contents(stream_asset, output_dir, self.pdf_name )

        except (ServiceApiException, ServiceUsageException, SdkException) as e:
            logging.exception(f'Exception encountered while executing operation: {e}')

    @staticmethod
    def extract_zip_contents(stream_asset: StreamAsset, output_dir: str, source_pdf_name: str):
        zip_stream = io.BytesIO(stream_asset.get_input_stream())       
        output_path = os.path.join(output_dir, f"{source_pdf_name}")
        
        # Create base directories first
        os.makedirs(output_path, exist_ok=True)
        os.makedirs(os.path.join(output_path, "tables"), exist_ok=True)   
        os.makedirs(os.path.join(output_path, "figures"), exist_ok=True)     

        with zipfile.ZipFile(zip_stream, 'r') as zip_file:
            for file_name in zip_file.namelist():
                with zip_file.open(file_name) as file:
                    file_content = file.read()

                    if file_name.endswith(".json"):
                        custom_json_name = f"{source_pdf_name}.json"
                        file_output_path = os.path.join(output_path, custom_json_name)
                        with open(file_output_path, 'w', encoding='utf-8') as f:
                            json_data = json.loads(file_content.decode('utf-8'))
                            json.dump(json_data, f, indent=4, ensure_ascii=False)
                            logging.info(f"Saved structured JSON: {file_output_path}")

                    elif file_name.endswith('.csv'):
                        # Save CSV files in the tables subdirectory
                        csv_file_name = os.path.basename(file_name)
                        file_output_path = os.path.join(output_path, "tables", csv_file_name)
                        with open(file_output_path, 'w', encoding='utf-8') as f:
                            f.write(file_content.decode('utf-8'))
                            logging.info(f"Saved CSV: {file_output_path}")

                    elif file_name.endswith(('.jpg', '.png')):
                        # Save images in the figures subdirectory
                        image_file_name = os.path.basename(file_name)
                        file_output_path = os.path.join(output_path, "figures", image_file_name)
                        with open(file_output_path, 'wb') as f:
                            f.write(file_content)
                            logging.info(f"Saved image (table/figure): {file_output_path}")

                    else:
                        logging.warning(f"Unhandled file type: {file_name}")

    @staticmethod
    def run(pdf_path: str, output_dir: str, client_id: str, client_secret: str):
        extractor = ExtractTextInfoFromPDF(pdf_path, output_dir, client_id, client_secret)
        return extractor.pdf_name


if __name__ == "__main__":
    # Example usage:
    # Set env variables or replace with actual values if needed.
    client_id=""
    client_secret=""
    
    # Provide path to input PDF and desired output directory
    input_pdf = "resources/ISO_21448.pdf"
    output_directory = "output/ExtractTextInfoFromPDF"

    ExtractTextInfoFromPDF.run(input_pdf, output_directory, client_id, client_secret)
    