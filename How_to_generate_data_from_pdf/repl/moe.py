# repl/moe.py
import threading
from llm import LLM 
from typing import List, Dict, Any, Optional, Callable

# repl/moe.py
import threading
import json # For processing tool calls if needed
from llm import LLM # Assuming llm.py is in the same directory or PYTHON_PATH
from typing import List, Dict, Any, Optional, Callable

class MOE():
    def __init__(self, configs: List[Dict[str, Any]], max_threads: int = 2):
        self.configs = configs
        self.max_threads = max(1, max_threads)
        self.semaphore = threading.Semaphore(self.max_threads)
        # self.results will be populated by _run_expert_phase
        self.results: List[Optional[Dict[str, Any]]] = [None] * len(configs)


    def _handle_request(self, index: int, model_config: Dict[str, Any]):
        # This internal method remains largely the same as your last version
        with self.semaphore:
            try:
                # print(f"Expert thread for model '{model_config.get('model', 'unknown')}' (index {index}) starting.")
                llm_model = model_config.get('model', 'qwen2.5:14b')
                llm_temperature = model_config.get('temperature', 0.2)
                llm_top_p = model_config.get('top_p', 0.9)
                llm_max_tokens = model_config.get('max_completion_tokens', 2000)
                llm_base_url = model_config.get('base_url')
                llm_api_key = model_config.get('api_key')
                llm_timeout = model_config.get('timeout')

                llm_params = {
                    "model": llm_model, "temperature": llm_temperature,
                    "top_p": llm_top_p, "max_completion_tokens": llm_max_tokens,
                }
                if llm_base_url: llm_params["base_url"] = llm_base_url
                if llm_api_key: llm_params["api_key"] = llm_api_key
                if llm_timeout: llm_params["timeout"] = llm_timeout
                
                llm_instance = LLM(**llm_params)
                messages = []
                system_prompt = model_config.get('system_prompt')
                user_input = model_config.get('input')

                if not user_input: # Should be validated before threading ideally
                    raise ValueError(f"Missing 'input' for model config at index {index}")
                if system_prompt:
                    messages.append({"role": "system", "content": system_prompt})
                messages.append({"role": "user", "content": user_input})

                functions_to_use = model_config.get('tools', None)
                response_format_config = model_config.get('response_format', None)

                raw_response = llm_instance.get_chat_completion(
                    messages=messages, functions=functions_to_use,
                    response_format=response_format_config
                )
                choice_message = raw_response.choices[0].message if raw_response.choices else None
                
                self.results[index] = {
                    "config": model_config, "response_object": raw_response,
                    "message": choice_message, "error": None
                }
            except Exception as e:
                # print(f"Error in expert thread for model '{model_config.get('model', 'unknown')}' (index {index}): {e}")
                self.results[index] = {
                    "config": model_config, "response_object": None,
                    "message": None, "error": str(e)
                }
            # finally:
                # print(f"Expert thread for model '{model_config.get('model', 'unknown')}' (index {index}) finishing.")


    def _run_expert_phase(self) -> List[Dict[str, Any]]:
        """Runs the initial parallel LLM calls to the experts."""
        if not self.configs:
            return []
        
        # Reset results for this run if generate() is called multiple times on same instance
        self.results = [None] * len(self.configs)

        threads = []
        print(f"Starting expert phase with {len(self.configs)} configurations...")
        for index, model_config_item in enumerate(self.configs):
            thread = threading.Thread(target=self._handle_request, args=(index, model_config_item))
            threads.append(thread)
            thread.start()
        for thread in threads:
            thread.join()
        
        print("Expert phase complete.")
        expert_results_list = []
        for i, res_info in enumerate(self.results):
            if res_info is None: # Should ideally not happen if list pre-allocated & threads complete
                 expert_results_list.append({
                    "original_config_index": i, "config": self.configs[i],
                    "response_object": None, "message": None,
                    "error": "Result not captured (unexpected error in threading for expert)"
                })
            else:
                 expert_results_list.append({"original_config_index": i, **res_info})
        return expert_results_list

    def _run_synthesis_phase(
        self,
        expert_results_list: List[Dict[str, Any]], # Results from the expert phase
        original_input_for_context: str,
        synthesis_prompt_template: str,
        synthesis_prompt_template_vars: Dict[str, str], # Vars like task_description, output_format, filename
        synthesis_llm_config: Dict[str, Any],
        include_errors_in_synthesis_context: bool = False
    ) -> Dict[str, Any]:
        """Runs the synthesis LLM call using results from the expert phase."""
        print("Starting synthesis phase...")
        response_snippets = []
        has_successful_expert_response = False

        for result_info in expert_results_list: # Iterate over the formatted expert_results_list
            model_name = result_info['config'].get('model', f"expert_{result_info['original_config_index']}")
            if not result_info.get('error'):
                message = result_info.get('message')
                content_to_add = None
                if message and message.content:
                    content_to_add = f"Response from {model_name}:\n{message.content}"
                    has_successful_expert_response = True
                elif message and message.tool_calls:
                    tool_calls_str = json.dumps([tc.model_dump() for tc in message.tool_calls])
                    content_to_add = f"Tool calls from {model_name}:\n{tool_calls_str}"
                    has_successful_expert_response = True 
                
                if content_to_add:
                    response_snippets.append(content_to_add)

            elif include_errors_in_synthesis_context and result_info.get('error'):
                response_snippets.append(
                    f"Error from {model_name}:\n{result_info.get('error')}"
                )

        if not has_successful_expert_response and not (include_errors_in_synthesis_context and response_snippets):
            print("No successful expert responses or relevant error context to synthesize.")
            return {
                "config": {"source": "synthesis_engine", **synthesis_llm_config},
                "response_object": None, "message": None,
                "error": "No successful expert responses or relevant error context to synthesize."
            }

        all_responses_text = "\n\n---\n\n".join(response_snippets)

        full_template_vars = {
            "original_input_text": original_input_for_context,
            "all_expert_responses": all_responses_text,
            **synthesis_prompt_template_vars 
        }

        try:
            synthesizer_system_prompt = synthesis_prompt_template.format(**full_template_vars)
        except KeyError as e:
            print(f"Error formatting synthesis prompt template: Missing key {e}")
            return {
                "config": {"source": "synthesis_engine", **synthesis_llm_config},
                "error": f"Failed to format synthesis prompt template: Missing key {e}"
            }

        synthesizer_user_prompt = "Please perform the synthesis task as described in the system instructions."
        
        messages = [
            {"role": "system", "content": synthesizer_system_prompt},
            {"role": "user", "content": synthesizer_user_prompt}
        ]

        # Prepare LLM constructor parameters (excluding response_format)
        llm_constructor_params_for_synthesis = {
            "model": synthesis_llm_config.get('model', 'qwen2.5:14b'), 
            "temperature": synthesis_llm_config.get('temperature', 0.2),
            "top_p": synthesis_llm_config.get('top_p', 0.9),
            "max_completion_tokens": synthesis_llm_config.get('max_completion_tokens', 2000),
            "base_url": synthesis_llm_config.get('base_url'),
            "api_key": synthesis_llm_config.get('api_key'),
            "timeout": synthesis_llm_config.get('timeout'),
        }
        llm_constructor_params_for_synthesis = {
            k: v for k, v in llm_constructor_params_for_synthesis.items() if v is not None
        }
        
        # Get response_format separately for the get_chat_completion call
        response_format_for_synthesis_call = synthesis_llm_config.get('response_format')

        synthesis_llm = LLM(**llm_constructor_params_for_synthesis)

        try:
            print(f"Calling synthesis LLM ({synthesis_llm.model})...")
            raw_response = synthesis_llm.get_chat_completion(
                messages=messages,
                response_format=response_format_for_synthesis_call # Pass response_format here
            )
            choice_message = raw_response.choices[0].message if raw_response.choices else None
            print("Synthesis LLM call complete.")
            return {
                "config": {"source": "synthesis_engine", **synthesis_llm_config},
                "response_object": raw_response,
                "message": choice_message,
                "error": None
            }
        except Exception as e:
            print(f"Error during synthesis LLM call: {e}")
            return {
                "config": {"source": "synthesis_engine", **synthesis_llm_config},
                "response_object": None, "message": None,
                "error": f"Synthesis LLM call failed: {str(e)}"
            }

    def generate(
        self,
        # --- Parameters for Synthesis Step ---
        perform_synthesis: bool = False,
        # Input that was common to all experts, for context in synthesis prompt
        synthesis_original_input_for_context: Optional[str] = None,
        # The general prompt template for the synthesizer LLM
        synthesis_prompt_template: Optional[str] = None,
        # Variables to fill into the synthesis_prompt_template (e.g., task_description, filename)
        synthesis_prompt_template_vars: Optional[Dict[str, str]] = None,
        # LLM configuration for the synthesizer LLM
        synthesis_llm_config: Optional[Dict[str, Any]] = None,
        # Whether to include error messages from experts in the context for the synthesizer
        include_errors_in_synthesis_context: bool = False
    ) -> Dict[str, Any]:
        """
        Generates responses from all configured expert LLMs concurrently.
        Optionally, if perform_synthesis is True, it then runs a synthesis LLM call
        using the outputs of the experts to generate a final answer.

        Returns:
            Dict[str, Any]: A dictionary containing:
                - "expert_results": A list of results from individual expert LLMs.
                - "final_synthesized_answer": The result from the synthesis LLM call, or None/error.
        """
        expert_results_list = self._run_expert_phase()
        final_synthesized_answer_info = None

        if perform_synthesis:
            if not all([synthesis_original_input_for_context, synthesis_prompt_template,
                        synthesis_prompt_template_vars, synthesis_llm_config]):
                print("Warning: Synthesis requested but not all required synthesis parameters are provided. Skipping synthesis.")
                final_synthesized_answer_info = {"error": "Missing required synthesis parameters."}
            else:
                final_synthesized_answer_info = self._run_synthesis_phase(
                    expert_results_list=expert_results_list,
                    original_input_for_context=synthesis_original_input_for_context,
                    synthesis_prompt_template=synthesis_prompt_template,
                    synthesis_prompt_template_vars=synthesis_prompt_template_vars,
                    synthesis_llm_config=synthesis_llm_config,
                    include_errors_in_synthesis_context=include_errors_in_synthesis_context
                )
        
        return {
            "expert_results": expert_results_list,
            "final_synthesized_answer": final_synthesized_answer_info
        }


if __name__ == "__main__": 

    filename="ISO 21448"     

    GENERAL_SYNTHESIS_PROMPT_TEMPLATE = """\
        You are an advanced AI assistant responsible for reviewing and synthesizing information from multiple expert AI responses.

        The original context provided to these expert models was regarding the document '{filename}' and based on the following input text:
        --- ORIGINAL INPUT TEXT ---
        {original_input_text}
        --- END ORIGINAL INPUT TEXT ---

        The expert AI models produced the following responses:
        --- COLLECTED EXPERT RESPONSES ---
        {all_expert_responses}
        --- END COLLECTED EXPERT RESPONSES ---

        Your specific synthesis instructions are as follows:
        --- SYNTHESIS TASK ---
        {synthesis_task_description}
        --- END SYNTHESIS TASK ---

        Please ensure your final output strictly adheres to the following format description:
        --- OUTPUT FORMAT ---
        {synthesis_output_format_description}
        --- END OUTPUT FORMAT ---

        If the provided expert responses are insufficient to complete the task, or are mostly errors, please indicate this clearly in your response, adhering to the output format if possible (e.g., by returning an empty string or a specific message like "Insufficient data for synthesis.").
        """    
    
    specific_synthesis_task = (
        "1. Review all the provided responses. These responses are expected to be JSON arrays of Q&A pairs.\n"
        "2. Merge these JSON arrays into a single, comprehensive JSON array.\n"
        "3. De-duplicate questions. If multiple answers exist for the same question, select the most accurate and complete one, or synthesize them into a better single answer. Ensure the answer still adheres to the original constraint of including the phrase '{filename}'.\n"
        "4. If the collected responses are mostly errors or empty, or if no valid Q&A pairs can be extracted and merged, return an empty JSON array (`[]`)."
    )
    specific_output_format = ("Your final output is ONLY a valid JSON array of Q&A objects, containing no other text, explanations, or markdown formatting.")

    synthesis_template_variables = {
        "filename": filename, # For {filename} in the general template
        "synthesis_task_description": specific_synthesis_task.format(filename=filename), # Format filename into the task desc.
        "synthesis_output_format_description": specific_output_format
    }

    synthesis_llm_configuration = {
        'model': 'phi4:14b', # Or a powerful model for synthesis, e.g., GPT-4, Claude 3 Opus
        'temperature': 0.3,
        'max_completion_tokens': 3000, # Allow more for potentially large merged JSON
        'response_format': {"type": "json_object"} # Crucial for the synthesizer to output JSON
    }
    
    system_prompt: str = (
        "You are an expert PDF reader designed to extract and convert content from the ISO document {filename} "
        "into a structured list of multiple question-and-answer (Q&A) pairs. Your role is to deeply understand each page or section "
        "of text and produce a detailed, technically accurate set of Q&A pairs.\n\n"
        "Your output must:\n"
        "- Be a **JSON array** of Q&A objects (each with a 'question' and an 'answer').\n"
        "- Contain **multiple Q&A pairs**, ideally 1–10 per page of substantive content.\n"
        "- Focus on **definitions, requirements, safety concepts, processes, and key principles** of {filename}.\n"
        "- Ensure **each question and answer includes the phrase '{filename}'**.\n"
        "- Be **self-contained**, concise, and understandable without external context.\n"
        "- Avoid copying large blocks of text verbatim; paraphrase and clarify where helpful.\n\n"

        "Format example:\n"
        "[\n"
        "  {{\n"
        "    \"question\": \"What is the scope of {filename}?\",\n"
        "    \"answer\": \"{filename} focuses on the safety of the intended functionality of road vehicles and addresses potential hazards from insufficient specification or performance.\"\n"
        "  }},\n"
        "  {{\n"
        "    \"question\": \"How does {filename} relate to other automotive safety standards?\",\n"
        "    \"answer\": \"{filename} complements standards like ISO 26262 by addressing safety concerns not related to hardware or software failures.\"\n"
        "  }}\n"
        "]\n\n"

        "IMPORTANT: If the provided content does not contain any relevant material for Q&A extraction related to {filename}, "
        "return only an empty JSON array (`[]`) without any additional text, explanation, or comments."
    )

    input_text="""Road vehicles — Safety of the intended functionality
            1 Scope
            This document provides a general argument framework and guidance on measures to ensure the safety 
            of the intended functionality (SOTIF), which is the absence of unreasonable risk due to a hazard caused 
            by functional insufficiencies, i.e.:
            a) the insufficiencies of specification of the intended functionality at the vehicle level; or
            b) the insufficiencies of specification or performance insufficiencies in the implementation of electric 
            and/or electronic (E/E) elements in the system.
            This document provides guidance on the applicable design, verification and validation measures, as 
            well as activities during the operation phase, that are needed to achieve and maintain the SOTIF.
            This document is applicable to intended functionalities where proper situational awareness is essential 
            to safety and where such situational awareness is derived from complex sensors and processing 
            algorithms, especially functionalities of emergency intervention systems and systems having levels of 
            driving automation from 1 to 5[2].
            This document is applicable to intended functionalities that include one or more E/E systems installed 
            in series production road vehicles, excluding mopeds.
            Normen-Download-DIN Media-Robert Wickel-KdNr.8450525-ID.XEKoxl0127MPxe2YrRrlTXYDJGWZtEpnSwYwINY1-2025-03-31 10:21:30
            Reasonably foreseeable misuse is in the scope of this document. In addition, operation or assistance of a 
            vehicle by a remote user or communication with a back office that can affect vehicle decision making is 
            in scope of this document when it can lead to safety hazards.
            This document does not apply to:
            — faults covered by the ISO 26262 series;
            — cybersecurity threats;
            — hazards directly caused by the system technology (e.g. eye damage from the beam of a lidar);
            — hazards related to electric shock, fire, smoke, heat, radiation, toxicity, flammability, reactivity, 
            release of energy and similar hazards, unless directly caused by the intended functionality of E/E 
            systems; and
            — deliberate actions that clearly violate the system’s intended use, (which are considered feature 
            abuse).
            This document is not intended for functions of existing systems for which well-established and well
            trusted design, verification and validation (V&V) measures exist (e.g. dynamic stability control systems, 
            airbags).
            2 Normative references
            The following documents are referred to in the text in such a way that some or all of their content 
            constitutes requirements of this document. For dated references, only the edition cited applies. For 
            undated references, the latest edition of the referenced document (including any amendments) applies.
            ISO 26262-1, Road vehicles — Functional safety — Part 1: Vocabulary"""     

    model_configs = [
        {
            'model': 'llama3.2:latest',
            'input': input_text,
            'system_prompt': system_prompt.format(filename=filename),
        },
        {
            'model': 'llama3.2:latest',
            'input': input_text,
            'system_prompt': system_prompt.format(filename=filename), # Corrected: removed extra comma
        }
    ]

    print("Starting MOE generation...")
    moe_instance = MOE(model_configs, max_threads=2)
    all_results = moe_instance.generate(
        perform_synthesis=True,
        synthesis_original_input_for_context=input_text,
        synthesis_prompt_template=GENERAL_SYNTHESIS_PROMPT_TEMPLATE,
        synthesis_prompt_template_vars=synthesis_template_variables,
        synthesis_llm_config=synthesis_llm_configuration,
        include_errors_in_synthesis_context=False # Set to True to include expert errors in synthesis context
    )
    # --- Process and Print Results ---
    print("\n\n--- Individual Expert Results ---")
    print("===================================")
    for i, result_info in enumerate(all_results["expert_results"]):
        print(f"\nExpert {i+1} (Model: {result_info['config'].get('model')})")
        if result_info['error']:
            print(f"  Error: {result_info['error']}")
        elif result_info.get('message') and result_info['message'].content:
            print(f"  Content:\n{result_info['message'].content}")
        elif result_info.get('message') and result_info['message'].tool_calls:
            print(f"  Tool Calls: {result_info['message'].tool_calls}")
        else:
            print("  No content or tool calls.")
    
    
    print("\n\n--- Final Synthesized Answer ---")
    print("==============================")
    synthesized_info = all_results["final_synthesized_answer"]
    if synthesized_info:
        if synthesized_info.get('error'):
            print(f"Error in synthesis: {synthesized_info['error']}")
        elif synthesized_info.get('message') and synthesized_info['message'].content:
            final_content = synthesized_info['message'].content
            print(f"Synthesized Content:\n{final_content}")
            # Optionally validate if it's the expected JSON for Q&A
            if synthesis_llm_configuration.get('response_format', {}).get('type') == 'json_object':
                try:
                    parsed_json = json.loads(final_content)
                    print("\nSuccessfully parsed synthesized JSON.")
                    if isinstance(parsed_json, list):
                        print(f"Number of Q&A pairs in final list: {len(parsed_json)}")
                except json.JSONDecodeError as e:
                    print(f"Could not parse synthesized content as JSON: {e}")
        else:
            print("No content in the synthesized answer message.")
    else:
        print("Synthesis was not performed or resulted in no information.")
        
    print("\nProcessing finished.")