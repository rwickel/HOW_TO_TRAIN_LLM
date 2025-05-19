# PDF Q&A Extraction (Simple)

This project enables automatic extraction of Question-Answer pairs from PDF files using large language models (LLMs) like **LLaMA3**, **Qwen**, or **Phi**. The pipeline leverages a structured approach with PDF text parsing, table inclusion (optional), prompt engineering, and configurable output.

---

## Features

* **Versatile LLM Support**: Easily configure and switch between different LLMs (e.g., LLaMA3, Qwen, Phi).
* **Comprehensive PDF Parsing**: Extracts both textual content and tables from PDF documents using PyMuPDF.
* **Table Integration**: Optionally includes extracted table data directly into the text processed by the LLM.
* **Customizable Prompts**: Tailor the system prompt used for Q&A extraction to fit specific needs.
* **Text Exclusion**: Filter out irrelevant text sections like headers, footers, or specific disclaimers.
* **Structured Output**: Generates two types of JSON files:
    * A flat list of all Q&A pairs from a document.
    * A detailed page-by-page structure including page text, Q&A pairs, and raw LLM responses.
* **Response Validation**: Optionally validate LLM responses against a predefined JSON schema.
* **Configurable Processing**: Control aspects like page limits for processing.
* **Command-Line Interface**: Run the extraction process via `main.py` with arguments for model and output directory.

---

## Project Structure

├── main.py # Entry point for PDF Q&A extraction
├── reader/
│ ├── config.py # Global configuration (LLM model, folders, prompts)
│ ├── types.py # Data classes for structured output
│ ├── pdf_reader_utils.py # PDF parsing + table handling
│ └── llm_interaction.py # Q&A logic using LLM
├── repl/
│ └── llm.py # LLM client abstraction
├── output/ # Output folder for JSON results
├── input_pdfs/ # Folder containing PDFs to process
└── README.md

---

## How It Works

1.  **Initialization**: The `main.py` script initializes the configured LLM client (via `repl/llm.py`).
2.  **PDF Discovery**: It scans the `input_pdfs` folder (or a configured path) for PDF files.
3.  **PDF Reading & Parsing**: For each PDF file:
    * `pdf_reader_utils.py` is used to open and read the PDF.
    * Text content is extracted page by page.
    * Specified unwanted text patterns (e.g., footers, copyright notices from `reader/config.py`) are removed.
    * Tables are identified and extracted (optionally converted to string format and appended to the page text if `add_tables_to_page_text` is enabled).
4.  **LLM Prompting & Q&A Extraction**:
    * For each processed page's text, a system prompt (defined in `reader/config.py` and formatted with filename and page number) is constructed.
    * This prompt, along with the page text, is sent to the configured LLM via `llm_interaction.py`.
    * The LLM generates Q&A pairs based on the provided content.
    * (Optional) The LLM's response can be validated against a JSON schema defined in `reader/config.py`.
5.  **Structured Output Generation**:
    * The extracted Q&A pairs are organized.
    * Two JSON files are saved to the `output` folder (or a specified directory):
        * `filename_qa_pairs.json`: A flat list containing all Q&A pairs extracted from the document.
        * `filename_qa_results.json`: A structured list, where each item represents a page and includes its `doc_id`, `page_number`, `page_text`, extracted `qa_pairs` for that page, and the `raw_response` from the LLM.

---

## Prerequisites

* **Python 3.8+**
* **An accessible LLM**:
    * This project is designed to work with LLMs like LLaMA3, Qwen, or Phi. You'll need to have the chosen model running and accessible. For models like `llama3.2:latest`, this typically means having a local LLM serving tool like [Ollama](https://ollama.com/) installed and the model pulled (e.g., `ollama pull llama3.2`).
    * Ensure the `repl/llm.py` is correctly configured to communicate with your LLM setup.
* **Pandas** and **PyMuPDF**: These Python libraries are used for PDF processing and data handling.

---

## Installation

1.  **Clone the repository:**
    ```bash
    git clone <your-repository-url>
    cd <your-repository-name>
    ```

2.  **Install dependencies:**
    It's recommended to use a virtual environment.
    ```bash
    python -m venv venv
    source venv/bin/activate  # On Windows use `venv\Scripts\activate`
    pip install pymupdf pandas # Add other dependencies if llm.py requires them (e.g., ollama, litellm)
    ```
    *Note: Depending on your `repl/llm.py` implementation, you might need to install additional libraries (e.g., `ollama`, `requests`, `litellm`). Please check `repl/llm.py` for specific LLM client library requirements.*

---

## Configuration

All primary configurations are centralized in `reader/config.py`.

```python
# reader/config.py

# Default folder for input PDFs
pdf_folder = "./input_pdfs"

# Default folder for output JSON files
output_folder = "./output"

# --- LLM Configuration ---
# Specify the model identifier for the LLM
# Examples: "llama3.2:latest", "qwen2.5:14b", "phi3:latest"
# This identifier should match what your LLM client (repl/llm.py) expects.
model = "llama3.2:latest"

# System prompt used to instruct the LLM for Q&A extraction.
# Placeholders {filename}, {page}, and {text} will be filled automatically.
system_prompt: str = (
        "You are an expert PDF reader designed to extract and convert content from the ISO document {filename} page {page} "
        "into a structured list of multiple question-and-answer (Q&A) pairs. Your role is to deeply understand each section "
        "and produce technically accurate, well-referenced Q&A pairs for each relevant part.\n\n"

        "Your output must:\n"
        "- Be a **JSON array** of Q&A objects (each with a 'question' and an 'answer').\n"
        "- Contain **multiple Q&A pairs**, ideally 1–10 per page of substantive content.\n"
        "- Focus on **definitions, requirements, safety concepts, processes, key principles, and standard-specific terminology** found in {filename}.\n"
        "- Ensure that **each question and answer includes the phrase '{filename}'**.\n"
        "- Ensure that **each answer includes a clause referencing the source section or clause number within {filename}**, such as '(see {filename}, Clause 5.2)'.\n"
        "- Be **self-contained**, concise, and understandable without external context.\n"
        "- Avoid copying large blocks of text verbatim; paraphrase and clarify for better comprehension.\n"
        "- Use proper grammar and domain-specific clarity.\n\n"

        "Format example:\n"
        "[\n"
        "  {{\n"
        "    \"question\": \"What is the scope of {filename}?\",\n"
        "    \"answer\": \"{filename} focuses on the safety of the intended functionality of road vehicles and addresses potential hazards from insufficient specification or performance (see {filename}{page}, Clause 1).\"\n"
        "  }},\n"
        "  {{\n"
        "    \"question\": \"How does {filename} relate to other automotive safety standards?\",\n"
        "    \"answer\": \"{filename} complements ISO 26262 by addressing safety concerns not caused by hardware or software faults, but by performance limitations or misuse (see {filename}{page}, Clause A.2).\"\n"
        "  }}\n"
        "]\n\n"

        "IMPORTANT: If the provided content does not contain any relevant material for Q&A extraction related to {filename}, "
        "return only an empty JSON array (`[]`) without any additional text, explanation, or comments."
    )

# JSON schema defining the expected structure for the LLM's response.
response_schema: Dict[str, Any] = field(default_factory=lambda: {
    "type": "array",
    "items": {
        "type": "object",
        "properties": {
            "question": {
                "type": "string",
                "description": "The question extracted or formulated from the PDF content related to ISO ... ."
            },
            "answer": {
                "type": "string",
                "description": "The answer to the question, sourced from the PDF content and starting with 'ISO ...'."
            }
        },
        "required": ["question", "answer"]
    }
})
```

## Run
Place PDF Files: Add the PDF files you want to process into the input_pdfs folder (or the folder specified in config.pdf_folder).

Bash
```
python main.py
Specify LLM model and output directory:
```
Bash
```
python main.py "<model_identifier>" "./custom_output_directory"
```

Examples:
```
python main.py "llama3.2:latest" "./llama3_2_output"
python main.py "qwen2.5:14b" "./qwen_outputs"
python main.py "phi3:latest" "./phi3_results"
```


