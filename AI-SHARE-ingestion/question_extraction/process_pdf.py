# Consider using basic keyword search vs prompting to separate sections

import pdfplumber
import argparse

def extract_text_from_pdf(pdf_path):
    paper=pdfplumber.open(pdf_path)
    text = ""
    for page in paper.pages:
        text += page.extract_text()
    return text

def extract_method(text):
    sections = text.split("\n")
    method_section = ""
    capture = False

    for line in sections:
        if "method" in line.lower() or "materials and methods" in line.lower():
            capture = True
        elif any(keyword in line.lower() for keyword in ["results", "discussion", "conclusion"]):
            capture = False

        if capture:
            method_section += line + "\n"

    return method_section



if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("input", help="Path to the PDF file")
    args = parser.parse_args()

    method_text = extract_method(extract_text_from_pdf(args.input))
    print(method_text)
