from transformers import AutoTokenizer, AutoModelForCausalLM
import torch

class GenerativeModel:
    def __init__(self, personality:str, maxTokens:int=50) -> None: 
        # declare our tokenizer
        self.tokenizer = AutoTokenizer.from_pretrained("teknium/OpenHermes-2.5-Mistral-7B")

        # declare out model
        self.model = AutoModelForCausalLM.from_pretrained("teknium/OpenHermes-2.5-Mistral-7B").to("cuda")

        # Padding is used to make all sequences (e.g., sentences or prompts) in a batch the same length.
        # It's mainly important when using batch processing, but setting it helps avoid size mismatch issues.
        # We're telling it to use the end-of-sequence token (eos) as the padding token.
        # EOS = End of Sequence. It marks the logical end of a generated output.
        self.tokenizer.pad_token = self.tokenizer.eos_token
        # Let the model know our pad token
        self.model.config.pad_token_id = self.tokenizer.eos_token_id
        # Clarification: Tokens are words or chunks of words. Token IDs are thier numerical form
        self.maxTokens=maxTokens

        # This is effectivly a running log of our conversation
        self.messages = [
            {"role": "system", "content": personality}
        ]
    
    def prompt(self, prompt:str, fullInput:bool=False) -> str:
        """Send a propmt and receive the model's response"""
        self.messages.append({"role": "user", "content": prompt})
        # Prepare inputs by tokenizing the message using a chat template.
        # We call it input_ids because its token IDs, which are integers representing tokens
        # Prepare input_ids by tokenizing a message with a specified templete
        input_ids = self.tokenizer.apply_chat_template(
            self.messages,                               # the string you want to generate text off of
            return_tensors="pt",                    # the shape of tensor you want back. pt = PyTorch (needed for this model)
            add_generation_prompt=True              # adds a marker so the model knows its time to "speak"
        ).to("cuda")
        # create a dictionary that holds our token IDs and attention mask
        # Attention mask controlls what the model pays attention to. (1 = attend, 0 = ignore)
        # in this case, all are marked with 1s because we're not padding anything out
        inputs = {
            "input_ids": input_ids,
            "attention_mask": torch.ones_like(input_ids)
        }
        
        output = self.model.generate(
            input_ids=inputs["input_ids"],              # input the token IDs to the model
            attention_mask=inputs["attention_mask"],    # pass the attention_mask
            pad_token_id=self.tokenizer.pad_token_id,   # pass the pad token id
            max_new_tokens=self.maxTokens               # set the max tokens it can return. This controlls how much it can say
        )
        
        # add prompt into conversation history. 
        input_length = inputs["input_ids"].shape[1]
        new_tokens = output[0][input_length:]
        response_text = self.tokenizer.decode(new_tokens, skip_special_tokens=True).strip()
        self.messages.append({"role": "assistant", "content": response_text})
        
        # return generated text
        if not(fullInput):
            return response_text
        
        return self.tokenizer.decode(output[0], skip_special_tokens=True)
        