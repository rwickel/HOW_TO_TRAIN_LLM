# repl/llm.py
import re
import json
from openai import OpenAI
from typing import Dict, Any
from typing import  List, Dict, Optional, Union, Any, Callable
import inspect
from jsonschema import validate, ValidationError

class LLM:
    def __init__(
        self,
        model: str = "qwen2.5:14b",
        temperature: float = 0.2,
        top_p: float = 0.9,
        max_completion_tokens: int = 2000,
        client = OpenAI(base_url='http://localhost:11434/v1/', api_key='ollama', timeout=20.0),
    ):
        """
        Initialize the LLM class.

        Args:
            model (str): The model to use for completions. Default is "phi4:14b".
            temperature (float): Sampling temperature. Default is 0.1.
            top_p (float): Nucleus sampling parameter. Default is 0.5.
            max_completion_tokens (int): Maximum number of tokens to generate. Default is 1000.
            client (Optional[object]): The client object to interact with the API. Default is None.
        """
        self.model = model
        self.temperature = temperature
        self.top_p = top_p
        self.max_completion_tokens = max_completion_tokens
        self.client = client

    def get_chat_completion(
        self,
        messages: List[Dict[str, str]],
        functions: Optional[List[Callable]] = None, # Changed object to Callable
        stream: bool = False,
        response_format: Optional[Dict[str, str]] = None,
    ) -> Any: # Using Any as the exact response type can vary (e.g. openai.types.chat.ChatCompletion)
        """
        Get a chat completion from the model.

        Args:
            messages (List[Dict[str, str]]): List of message dictionaries with "role" and "content".
            functions (Optional[List[Callable]]): List of Python functions to describe for tool usage. Default is None.
            stream (bool): Whether to stream the response. Default is False.
            response_format (Optional[Dict[str, str]]): Format of the response (e.g., {"type": "json_object"}). Default is None.

        Returns:
            Any: The completion response from the model (typically an object from the OpenAI library).
        """
        if not self.client:
            # This case should ideally not be reached if __init__ ensures client creation
            raise ValueError("Client is not initialized. Please provide a client object or ensure __init__ creates one.")

        # Convert Python functions to JSON schema for tools if provided
        tools_json = [self.function_to_json(f) for f in functions] if functions else []

        # Filter valid messages (basic validation)
        # Consider if "" (empty string) content should be allowed by the API or your use case
        valid_messages = [
            msg for msg in messages
            if isinstance(msg, dict)
            and "role" in msg
            and "content" in msg
            and isinstance(msg["content"], str) # Content must be a string
            # and msg["content"].strip() # Removed: allows empty strings if API supports it; API usually handles this.
        ]
        if not valid_messages and messages: # If all messages were filtered out but some were provided
            raise ValueError("No valid messages to send after filtering. Ensure messages have 'role' and 'content' (string).")
        if not valid_messages and not messages: # No messages provided at all
             raise ValueError("No messages provided to get_chat_completion.")


        create_params = {
            "model": self.model,
            "temperature": self.temperature,
            "messages": valid_messages,
            "max_tokens": self.max_completion_tokens,
            "top_p": self.top_p,
            "stream": stream,
        }

        if response_format:
            create_params["response_format"] = response_format
        
        if tools_json:
            create_params["tools"] = tools_json
            # OpenAI API often requires tool_choice to be set if tools are provided,
            # e.g., "auto" or {"type": "function", "function": {"name": "my_function"}}
            # Depending on the Ollama/client version, this might be needed.
            # create_params["tool_choice"] = "auto" # Example

        return self.client.chat.completions.create(**create_params)
    
    def format_response(self, content: str, schema: Optional[Dict[str, Any]] = None) -> Union[List[Dict[str, Any]], Dict[str, Any]]:
        """
        Formats and validates the raw string response from the model based on an optional schema.

        Args:
            content (str): The raw response content from the model.
            schema (Optional[Dict[str, Any]]): The JSON schema to validate the content against.

        Returns:
            Union[List[Dict[str, Any]], Dict[str, Any]]: 
                - If schema validation is successful: the parsed and validated data.
                - If schema validation fails or JSON is invalid: an error dictionary.
                - If no schema provided: a dictionary with the raw text.
        """
        content = content.strip()

        if not schema:
            print("Warning: Schema provided to LLM.format_response, but jsonschema is not installed. Returning raw text.")
            return {"text": content, "warning": "jsonschema not available for validation"}

        if schema:
            try:                
                content = self.extract_json_block(content) # Extract JSON block if present	

                # Handle specific cases for "empty" JSON that should result in an empty list
                if content == "[]" and schema.get("type") == "array":                                  
                    return {"valid": True, "data":None, "raw_content": content} 
                
                if content == "{}" and schema.get("type") == "array":
                    return {"valid": True, "data": None, "raw_content": content}  

                data = json.loads(content)

                # Auto-wrap single object in list if schema expects an array and object matches item schema
                if schema.get("type") == "array" and isinstance(data, dict):
                    item_schema = schema.get("items")
                    if item_schema:
                        try:
                            validate(instance=data, schema=item_schema) # Check if the dict matches the item schema
                            print("INFO: LLM.format_response - Auto-wrapping single dictionary into a list as it matches item schema.")
                            data = [data]
                        except ValidationError:
                            # Don't wrap if the single dict doesn't match the item schema;
                            # let the main validation catch the type mismatch (dict vs array).
                            print("INFO: LLM.format_response - Single dictionary found, but it does not match item schema. Will not auto-wrap.")
                    else: # Schema is an array but no item schema? Unlikely for Q&A but possible.
                        print("Warning: LLM.format_response - Schema type is array, but no 'items' schema defined for auto-wrapping check.")

                validate(instance=data, schema=schema)
                # If validation passes, 'data' is the Python object (list or dict)
                return {
                    "valid": True,
                    "data": data,
                    "error": "",
                    "raw_content": content # Return original content, not stripped
                } 
            except json.JSONDecodeError:
                return {
                    "valid": False,
                    "data": None,
                    "error": "Invalid JSON format.",
                    "raw_content": content # Return original content, not stripped
                }
            except ValidationError as e:
                return {
                    "valid": False,
                    "data": None,
                    "error": "Schema validation failed.",
                    "details": e.message, # More specific error message
                    "raw_content": content
                }
            except Exception as e: # Catch any other unexpected errors
                return {
                    "valid": False,
                    "data": None,
                    "error": "Unexpected error during response formatting/validation.",
                    "details": str(e),
                    "raw_content": content
                }

        # If no schema is provided, or if jsonschema is not available but schema was given
        return {"valid": True, "text": content}  

    def extract_json_block(self, content: str) -> str:
        """
        Extracts the JSON content from a markdown-style code block.

        Args:
            content (str): The raw content possibly wrapped in ```json ... ```.

        Returns:
            str: The extracted JSON string, or the original content if no code block is found.
        """
        match = re.search(r"```json\s*(.*?)\s*```", content, re.DOTALL)
        if match:
            return match.group(1).strip()
        return content.strip()

        
    def function_to_json(self, func) -> dict:
        """
        Converts a Python function into a JSON-serializable dictionary
        that describes the function's signature, including its name,
        description, and parameters.

        Args:
            func: The function to be converted.

        Returns:
            A dictionary representing the function's signature in JSON format.
        """
        type_map = {
            str: "string",
            int: "integer",
            float: "number",
            bool: "boolean",
            list: "array",
            dict: "object",
            type(None): "null",
        }

        try:
            signature = inspect.signature(func)
        except ValueError as e:
            raise ValueError(
                f"Failed to get signature for function {func.__name__}: {str(e)}"
            )

        parameters = {}
        for param in signature.parameters.values():
            try:
                param_type = type_map.get(param.annotation, "string")
            except KeyError as e:
                raise KeyError(
                    f"Unknown type annotation {param.annotation} for parameter {param.name}: {str(e)}"
                )
            parameters[param.name] = {"type": param_type}

        required = [
            param.name
            for param in signature.parameters.values()
            if param.default == inspect._empty
        ]

        return {
            "type": "function",
            "function": {
                "name": func.__name__,
                "description": func.__doc__ or "",
                "parameters": {
                    "type": "object",
                    "properties": parameters,
                    "required": required,
                },
            },
        }
    
if __name__ == "__main__": 
    llm = LLM()
    response= llm.get_chat_completion(
        messages=[{"role": "user", "content": "Hello, how are you?"}],
        functions=[],
        stream=False,
        response_format=None
    )  
    print(response.choices[0].message.content)