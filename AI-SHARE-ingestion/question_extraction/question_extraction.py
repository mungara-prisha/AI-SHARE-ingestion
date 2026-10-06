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

def chunk_text(text, chunk_size=4000):

    chunks = []

    for i in range(
        0,
        len(text),
        chunk_size
    ):
        chunks.append(
            text[i:i+chunk_size]
        )

    return chunks





def find_survey_sections(chunk):

    prompt = f"""
You are analyzing an academic paper.

Determine whether this text contains survey questions.

Text:

{chunk}


Return JSON:

{{
"contains_survey_information": true/false,
"relevant_excerpt": ""
}}
"""

    return generate_response(prompt)

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


def extract_survey_questions(section_text):

    prompt = f"""
You are analyzing an academic research paper with survey questions.
Extract these survey questions verbatim, with no modifications, and return them in a JSON format as follows:

{{
"survey_questions": [
"example1",
"example2"
]   

}}
"""

    response = generate_response(prompt)

    try:
        return json.loads(response)

    except:
        return {
            "survey_questions": [],
            "raw_response": response
        }


def question_check(question_text):
    prompt = f"""
You are analyzing an academic research paper with survey questions.
Given the following survey question, determine whether it is related to technology, AI, or similar topics.
If it is, return true, otherwise return false. Only return the word true or false, with no additional text.
Survey Question:
{question_text}
"""
    response = generate_response(prompt)

    if response == "true":
        return 1
    elif response == "false":
        return 0
    else:
        return -1


def get_provenance(question_text, paper_text):
    prompt = f"""
You are analyzing an academic research paper with survey questions.
Given the following survey question, find the source of this question according to the provided text.
Answer using the following guide.
1 -Author developed
The authors state that the survey question or instrument was developed for the current study. The authors may not state this explicitly, but it is likely implied if they do not mention that the survey question was taken directly or adapted from another study or survey instrument. That is, you will probably code 1) if 2) and 3) below do not apply.
2 -Taken directly from an existing survey instrument
The authors state that the survey question or instrument was directly taken from an existing survey instrument, scale, index, or prior paper.
3 -Adapted from an existing survey instrument
The authors state that the survey question or instrument was adapted, modified, or based on an existing survey instrument, scale, index, or prior paper.
4 -Other
The authors provide a different explanation for the source of the survey question or instrument.
After determining the source, provide the DOI or citation of the source if it is available in the text.
Answer using the following JSON format:
{{
    "provenance": 1/2/3/4
    "source_text": "text from the paper that supports your answer"
}}
If you chose 1, fill the "source_text" field with "N/A".
Otherwise, if the source DOI can not be found, put "not provided" in the source field.
Survey Question:
{question_text}
Paper Text:
{paper_text}
"""
    response = generate_response(prompt)
    return response


def get_tech_scope(question_text, paper_text):
    prompt = f"""
You are analyzing an academic research paper with survey questions.
Given the following survey question, determine the technology scope of this question according to the provided text.
Answer using the following guide.
1) AI or automation generally, without mentioning anything more specific
If the question wording does not involve any categorization or classification of the task that the AI would or does perform or the purpose/context, it should be classified as 1. Some survey questions ask about opinions of AI generally, such as: “When thinking about society generally, do you think the benefits of AI outweigh the risks?”

2) Domain Artificial Intelligence, or ‘Narrow’ AI
If the question references a specific type of AI or a specific context/purpose for AI, it should be classified as 2.
Refers to AI systems designed and trained for a specific task. Examples include virtual personal assistants like Siri or Alexa, image and facial recognition systems, self-driving cars, automated surgeons, or chatbots. May also refer to AI systems deployed for specific purposes or in specific contexts. Examples include AI tools for student learning or automated decision systems for government agency programs.
If the question wording specifically categorizes and narrows down the task that the AI would or does perform, or the context in which it would be deployed, it should be classified as 2.

3) ‘Strong’ AI, AGI, advanced or human-level AI
If the question specifically mentions strong AI, artificial general intelligence (AGI), superintelligence, the singularity, or other related concepts, it should be classified as 3. 
Refers to a machine that can perform any intellectual task that a human being can and/or is expected to mimic (or exceed) human consciousness. It can understand, learn, and apply knowledge in different domains, reason through problems, have consciousness, and even possess emotional understanding.

4) Other: Please specify
May include questions just asking about technology generally                                                 
5) Technology not identified
Answer using the following JSON format:
{{
    "tech_scope": 1/2/3/4/5
}}
Survey Question:
{question_text}
Paper Text:
{paper_text}
"""
    response = generate_response(prompt)
    return response

def code_question(question_text):
    # call all functions here to code the question
    question_check_result = question_check(question_text)
    if question_check_result == 0:
        return {
            "question_text": question_text,
            "is_tech_related": False,
            "provenance": None,
            "tech_scope": None
        }
    # question_provenance = get_provenance(question_text, paper_text)
    tech_scope = get_tech_scope(question_text, paper_text)




if __name__ == "__main__":

    paper_text = process_pdf.extract_text_from_pdf("D0095__paper.pdf")

    questions = extract_survey_questions(paper_text)
    question_coding = code_question(questions["survey_questions"][0])
