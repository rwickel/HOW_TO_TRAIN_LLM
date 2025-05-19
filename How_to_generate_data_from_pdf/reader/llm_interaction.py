# repl/llm_interaction.py

from typing import Dict, Any, List, Optional
from repl.llm import LLM 
import json

def format_llm_response(llm_completion: Any, llm_client: LLM, response_schema: Optional[Dict] = None ) -> Optional[str]:
    """
    Extracts and prints the message content from the LLM completion object.

    Args:
        llm_completion (Any): The completion object returned by the LLM.

    Returns:
        Optional[str]: The content of the LLM's response as a string,
                       or a string indicating a tool call, or None if an error occurs
                       or no relevant content is found.
    """
    try:
        if llm_completion and hasattr(llm_completion, 'choices') and llm_completion.choices:
            message = llm_completion.choices[0].message
            if message:
                if hasattr(message, 'content') and message.content:
                    raw_content_str = message.content.strip()
                    # Primary path: successful response with content
                    print("\n--- LLM Response (Q/A Pairs) ---")                    
                    print(raw_content_str)
                    print("--- End LLM Response ---\n")                    
                    return llm_client.format_response(raw_content_str, response_schema)
                
                elif hasattr(message, 'tool_calls') and message.tool_calls:
                    # Path for tool usage (if your LLM supports it)
                    print("\n--- LLM Response (Tool Calls) ---")
                    for tool_call in message.tool_calls:
                        print(f"Tool Call ID: {tool_call.id}")
                        print(f"Function Name: {tool_call.function.name}")
                        print(f"Arguments: {tool_call.function.arguments}")
                    print("--- End LLM Response ---\n")
                    # For this application, we expect direct Q/A, not tool calls.
                    # However, returning info about the tool call might be useful for debugging.
                    # Depending on the setup, you might want to handle this as an unexpected response.
                    return f"Tool call requested: {message.tool_calls[0].function.name}"
                else:
                    print("LLM response message does not contain content or tool calls.")
            else:
                print("LLM response does not contain a message object in the first choice.")
        else:
            print("Invalid or empty LLM completion object received.")
            if llm_completion and hasattr(llm_completion, 'error'): # Check for an error attribute
                 print(f"API Error: {llm_completion.error}")

    except AttributeError as e:
        print(f"Error accessing attributes in LLM response: {e}")
        print(f"Raw LLM response object: {llm_completion}")
    except Exception as e:
        print(f"An unexpected error occurred in format_llm_response: {e}")
        print(f"Raw LLM response object: {llm_completion}")
    return None

def execute_qa_extraction(page_text: str, llm_client: LLM, system_prompt: str="", response_schema: Optional[Dict] = None) -> Optional[str]:
    """
    Sends a page's text to the LLM for Q/A extraction based on the system prompt.

    Args:
        page_text (str): The text content from a single PDF page.
        llm_client (LLM): An instance of the LLM class used for API calls.
        response_schema (Optional[Dict]): An optional JSON schema to guide the LLM's response format.
                                          If None, the LLM will rely solely on the prompt.

    Returns:
        Optional[str]: The content of the LLM's response (expected to be a JSON string
                       of Q&A pairs), or None if an error occurs or the page text is empty.
    """
    if not page_text.strip():
        print("Skipping Q/A extraction for empty page text.")
        return None

    # Construct the messages payload for the LLM API call
    user_message = {"role": "user", "content": page_text}
    
    messages_for_llm = [{"role": "system", "content": system_prompt}, user_message]

    response_format_param = None
    if response_schema:
        # This structure is common for OpenAI API's JSON mode.
        # Adjust if your LLM client or model uses a different format for specifying JSON output with a schema.
        response_format_param = {"type": "json_object", "json_schema": response_schema}  
    
    try:
        completion = llm_client.get_chat_completion(
            messages=messages_for_llm,
            stream=False,  # Assuming non-streaming for Q/A extraction 
            #response_format=response_format_param,  # when using json shema the amont of question answering paires is reduced by the llm itself
        )

        response = completion.choices[0].message.content.strip()
        print(response)

        if response_schema:
            return llm_client.format_response(response, response_schema)  
        else:
            # If no schema is provided, we still return the raw response
            return response      
    
    except Exception as e:
        print(f"Error during LLM API call in execute_qa_extraction: {e}")
        return None
