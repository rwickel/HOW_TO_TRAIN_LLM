# repl/config.py

from dataclasses import dataclass, field
from typing import List, Dict, Any

@dataclass
class AppConfig:
    """
    Configuration class for the PDF Q&A extraction application.
    """
    # System prompt for the LLM, instructing it on how to process the PDF content
    # and the desired output format (JSON array of Q&A pairs).
    system_prompt: str = (
        "You are an expert PDF reader designed to extract and convert content from the document {filename} page {page} "
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
        "    \"question\": \"What is the scope of ISO 21448 ?\",\n"
        "    \"answer\": \"ISO 21448 focuses on the safety of the intended functionality of road vehicles and addresses potential hazards from insufficient specification or performance (see {filename} page:{page}, Clause 1).\"\n"
        "  }},\n"
        "  {{\n"
        "    \"question\": \"How does ISO 21448 relate to other automotive safety standards?\",\n"
        "    \"answer\": \"ISO 21448 complements ISO 26262 by addressing safety concerns not caused by hardware or software faults, but by performance limitations or misuse (see {filename} page: {page}, Clause A.2).\"\n"
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

    # A list of text strings to be excluded from the PDF content before processing.
    exclude_text: List[str] = field(default_factory=lambda: [
        
    ])

    # Default path to the PDF file that needs to be processed.
    pdf_folder: str = ".\data"

    output_folder: str = ".\output"

    model:str = "qwen2.5:14b"

# Create a default instance of the configuration
# Other modules will import this instance.
config = AppConfig()

if __name__ == "__main__":
    # Example of how to access the configuration
    config_instance = AppConfig()
    print(f"System Prompt (first 50 chars): {config_instance.system_prompt[:50]}...")
    print(f"Default PDF Path: {config_instance.pdf_path}")
    print(f"First item to exclude: {config_instance.exclude_text[0]}")
    print(f"Response schema type: {config_instance.response_schema.get('type')}")

