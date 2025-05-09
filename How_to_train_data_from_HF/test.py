import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

class Chatbot:
    def __init__(self, model_path="./best_model"):
        # Set device (GPU if available, else CPU)
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(f"Using device: {self.device}")

        # Load the model and tokenizer
        self.tokenizer = AutoTokenizer.from_pretrained(model_path)
        self.model = AutoModelForCausalLM.from_pretrained(
            model_path,
            torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32
        ).to(self.device)

        self.model.eval()  # Set the model to evaluation mode

        self.history = []

        # Check for pad_token_id and set if necessary
        if self.tokenizer.pad_token_id is None:
            self.tokenizer.pad_token_id = self.tokenizer.eos_token_id or 0

    def generate_response(self, user_input, memory=False):
        # Add user input to the conversation history
        messages = self.history + [{"role": "user", "content": user_input}]
        
        try:
            # Tokenize using chat template (if available)
            text = self.tokenizer.apply_chat_template(messages, 
                                                      tokenize=False, 
                                                      add_generation_prompt=True,
                                                      enable_thinking=False)
        except AttributeError:
            # Fallback if tokenizer has no chat template
            text = user_input if not self.history else " ".join(
                [f"{m['role']}: {m['content']}" for m in messages]
            )

        # Tokenize the input text for the model
        inputs = self.tokenizer(text, return_tensors="pt").to(self.device)

        # Generate the response from the model
        with torch.no_grad():
            response_ids = self.model.generate(
                **inputs,
                max_new_tokens=512,  # Adjust based on your needs
                do_sample=True,
                temperature=0.7,                
                top_p=0.9,                
                pad_token_id=self.tokenizer.pad_token_id  # Ensure pad token is used
            )[0][len(inputs["input_ids"][0]):].tolist()

        # Decode the response
        response = self.tokenizer.decode(response_ids, skip_special_tokens=True)

        if memory:
            # Add both user input and assistant response to history
            self.history.append({"role": "user", "content": user_input})
            self.history.append({"role": "assistant", "content": response})

        return response

if __name__ == "__main__":
    chatbot = Chatbot()

    print("💬 Chatbot is ready! Type 'exit' or 'quit' to stop.\n")
    while True:
        user_input = input("You: ").strip()
        if user_input.lower() in ["exit", "quit"]:
            print("👋 Goodbye!")
            break

        bot_response = chatbot.generate_response(user_input)
        print(f"Bot: {bot_response}\n")
