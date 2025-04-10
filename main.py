# Load the tokenizer and model from Hugging Face
from transformers import AutoTokenizer, AutoModelForCausalLM
import torch
import time


# declare our tokenizer
tokenizer = AutoTokenizer.from_pretrained("teknium/OpenHermes-2.5-Mistral-7B")

# declare out model
model = AutoModelForCausalLM.from_pretrained("teknium/OpenHermes-2.5-Mistral-7B").to("cuda")



# Padding is used to make all sequences (e.g., sentences or prompts) in a batch the same length.
# It's mainly important when using batch processing, but setting it helps avoid size mismatch issues.
# We're telling it to use the end-of-sequence token (eos) as the padding token.
# EOS = End of Sequence. It marks the logical end of a generated output.
tokenizer.pad_token = tokenizer.eos_token
# Let the model know our pad token
model.config.pad_token_id = tokenizer.eos_token_id
# Clarification: Tokens are words or chunks of words. Token IDs are thier numerical form


# This is effectivly a running log of our conversation
messages = [
    {"role": "system", "content": "You are an old and wise wizard."}
]


while True:
    user_input = input("\nEnter your message (or '0' to exit): ")
    
    if user_input == '0':
        break
        
    messages.append({"role": "user", "content": user_input})
    
    
    start_time = time.time()

    # Prepare inputs by tokenizing the message using a chat template.
    # We call it input_ids because its token IDs, which are integers representing tokens
    # Prepare input_ids by tokenizing a message with a specified templete
    input_ids = tokenizer.apply_chat_template(
        messages,                               # the string you want to generate text off of
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

    generation_start = time.time()
    
    # run the model
    output = model.generate(
        input_ids=inputs["input_ids"],              # input the token IDs to the model
        attention_mask=inputs["attention_mask"],    # pass the attention_mask
        pad_token_id=tokenizer.pad_token_id,        # pass the pad token id
        max_new_tokens=50                           # set the max tokens it can return. This controlls how much it can say
    )
    
    generation_end = time.time()

    # output should be in a tensor with the shape [1, input_length + generated_tokens]
    
    
    # decode tokens back into human language.
    # skip special tokens because we dont care about things like <|pad|>, <|eos|>, ect. during decoding
    decoded = tokenizer.decode(output[0], skip_special_tokens=True) # whats the shape of the output? skip special tokens to avoid things like eod tokens?

    # Total time
    end_time = time.time()

    # add new tokens to conversation history
    input_length = inputs["input_ids"].shape[1]
    new_tokens = output[0][input_length:]

    # Decode only new response
    response_text = tokenizer.decode(new_tokens, skip_special_tokens=True).strip()

    # Append only the new response to the conversation
    messages.append({"role": "assistant", "content": response_text})



    # Print timing results and response
    print("\n=== Timing Results ===")
    print(f"Total time: {end_time - start_time:.2f} seconds")
    print(f"Generation time: {generation_end - generation_start:.2f} seconds")
    print("\n=== Generated Text ===")
    print(decoded)

print("Goodbye!")
