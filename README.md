Project Toolkit Overview

This repository is a collection of tools and scripts designed to support various stages of a machine learning pipeline, particularly focusing on data extraction from PDFs and preparation for training models. Below is a quick overview of the folder structure based on the image provided:



Folder Descriptions

📁 Extract_Text_From_PDF_Adobe

Extracts structured data (text, tables, and figures) from PDFs using Adobe PDF Services API.

Output includes structured JSON, CSV tables, and figure images.

Use this for robust extraction from research papers, datasheets, etc.

📁 How_to_create_custom_train_data

Guides and scripts for generating custom training datasets.

May include JSON or CSV format conversion, annotation helpers, or formatting for specific model training.

📁 How_to_generate_data_from_pdf

Focuses on simpler or alternative methods to extract information from PDFs (likely without Adobe API).

Useful for internal datasets or less structured PDFs.

📁 How_to_setup_benchmarks

Contains benchmarking tools or scripts for evaluating model performance.

May support metric logging, comparison reports, or visualization tools.

📁 How_to_train_data_from_HF

Guides and scripts for fine-tuning or training models using Hugging Face.

Likely includes integration with transformers, dataset formatting, and model evaluation.
