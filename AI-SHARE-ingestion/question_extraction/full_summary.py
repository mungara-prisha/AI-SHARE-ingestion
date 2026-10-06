import os
from transformers import AutoTokenizer, AutoModelForCausalLM
import torch
from huggingface_hub import login
import json
import process_pdf

print("CUDA available:", torch.cuda.is_available())
print("CUDA device count:", torch.cuda.device_count())

if torch.cuda.is_available():
    print("GPU:", torch.cuda.get_device_name(0))



login(token="hf_token") # add token here

model_name = "meta-llama/Llama-3.1-8B-Instruct"

tokenizer = AutoTokenizer.from_pretrained(model_name)

model = AutoModelForCausalLM.from_pretrained(
    model_name,
    torch_dtype=torch.float16,
    device_map="auto"

)



def generate_response(prompt):

    messages = [
        {
            "role": "user",
            "content": prompt
        }
    ]

    text = tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True
    )

    inputs = tokenizer(
        text,
        return_tensors="pt"
    ).to(model.device)

    outputs = model.generate(
        **inputs,
        max_new_tokens=150,
        temperature=0,
        do_sample=False
    )

    generated = outputs[0][inputs.input_ids.shape[1]:]

    response = tokenizer.decode(
        generated,
        skip_special_tokens=True
    )

    return response

if __name__ == "__main__":
    prompt = "Summarize the following text in 3 sentences:\n\n "
    prompt += process_pdf.extract_text_from_pdf("D0095__paper.pdf")
    answer = generate_response(prompt)
    print("Answer:", answer)
