from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
import torch
from accelerate import init_empty_weights, infer_auto_device_map

quantization_config = BitsAndBytesConfig(load_in_8bit=True)

# PREQUISITES
# -----------------------------
# With transformers<4.51.0
# pip install --upgrade transformers accelerate bitsandbytes


class QwenChatbot:  # Load model directly

    def __init__(self, model_name="Qwen/Qwen3-1.7B"):
        max_memory = {i: "14GiB" for i in range(torch.cuda.device_count())}
        self.tokenizer = AutoTokenizer.from_pretrained(
            pretrained_model_name_or_path=model_name, trust_remote_code=True
        )
        self.model = AutoModelForCausalLM.from_pretrained(
            pretrained_model_name_or_path=model_name,
            torch_dtype=torch.bfloat16,
            # torch_dtype="auto",
            device_map="cuda",  ### "`tp_plan` and `device_map` are mutually exclusive. Choose either one for parallelization."
            attn_implementation="eager",
            # attn_implementation="sdpa",
            trust_remote_code=True,
            max_memory=max_memory,
            quantization_config=quantization_config,
        )
        self.history = []

    def _gc(self):
        import gc

        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    def generate_response(self, user_input, thinking: bool = True):
        messages = self.history + [{"role": "user", "content": user_input}]

        text = self.tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
            enable_thinking=True,  # Switches between thinking and non-thinking modes. Default is True.
        )
        model_inputs = self.tokenizer([text], return_tensors="pt").to(self.model.device)

        if thinking:
            # For thinking mode, use Temperature=0.6, TopP=0.95, TopK=20, and MinP=0 (the default setting in generation_config.json).
            # DO NOT use greedy decoding, as it can lead to performance degradation and endless repetitions.
            # For more detailed guidance, please refer to the Best Practices section.
            generated_ids = self.model.generate(
                **model_inputs,
                temperature=0.6,
                top_p=0.95,
                top_k=20,
                min_p=0,
                repetition_penalty=1.05,
                max_new_tokens=32768,
            )
        else:
            ## NON-THINKING
            # For non-thinking mode, we suggest using
            # Temperature=0.7, TopP=0.8, TopK=20, and MinP=0.
            # For more detailed guidance, please refer to the Best Practices section.
            generated_ids = self.model.generate(
                **model_inputs,
                temperature=0.7,
                top_p=0.8,
                top_k=20,
                min_p=0,
                repetition_penalty=1.05,
                max_new_tokens=32768,
            )

        ## decode output tokens
        output_ids = generated_ids[0][len(model_inputs.input_ids[0]) :].tolist()

        # parsing thinking content
        try:
            # rindex finding 151668 (</think>)
            index = len(output_ids) - output_ids[::-1].index(151668)
        except ValueError:
            index = 0

        thinking_content = self.tokenizer.decode(
            output_ids[:index], skip_special_tokens=True
        ).strip("\n")
        response = self.tokenizer.decode(
            output_ids[index:], skip_special_tokens=True
        ).strip("\n")

        self._gc()
        print(self.model.get_memory_footprint())

        print("thinking content:", thinking_content)
        # print("content:", response)

        # Update history
        self.history.append({"role": "user", "content": user_input})
        self.history.append({"role": "assistant", "content": response})

        return response


# Example Usage
if __name__ == "__main__":
    chatbot = QwenChatbot()

    # # default think
    # user_input_0 = "Give me a short introduction to large language model."
    # print(f"User: {user_input_0}")
    # response_0 = chatbot.generate_response(user_input_0)
    # print(f"Bot: {response_0}")
    # print("----------------------")

    # # nothink
    # user_input_0 = "Give me a short introduction to large language model. /no_think"
    # print(f"User: {user_input_0}")
    # response_0 = chatbot.generate_response(user_input_0)
    # print(f"Bot: {response_0}")
    # print("----------------------")

    # # First input (without /think or /no_think tags, thinking mode is enabled by default)
    # user_input_1 = "How many r's in strawberries?"
    # print(f"User: {user_input_1}")
    # response_1 = chatbot.generate_response(user_input_1)
    # print(f"Bot: {response_1}")
    # print("----------------------")

    # # Second input with /no_think
    # user_input_2 = "Then, how many r's in blueberries? /no_think"
    # print(f"User: {user_input_2}")
    # response_2 = chatbot.generate_response(user_input_2)
    # print(f"Bot: {response_2}")
    # print("----------------------")

    # # Third input with /think
    # user_input_3 = "Really? /think"
    # print(f"User: {user_input_3}")
    # response_3 = chatbot.generate_response(user_input_3)
    # print(f"Bot: {response_3}")

    # Fourth input with /think
    user_input_4 = (
        "compare strawberries and blueberries determine word has more r's? /think"
    )
    print(f"User: {user_input_4}")
    response_4 = chatbot.generate_response(user_input_4, thinking=True)
    print(f"Bot: {response_4}")

    print("=======================================")
    # Fourth input with /no_think
    user_input_4 = (
        "compare strawberries and blueberries determine word has more r's? /no_think"
    )
    print(f"User: {user_input_4}")
    response_4 = chatbot.generate_response(user_input_4, thinking=False)
    print(f"Bot: {response_4}")
