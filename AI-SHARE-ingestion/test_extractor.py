from pathlib import Path
from extractor.extractor import SurveyExtractor
import json

#extractor = SurveyExtractor(model_name="microsoft/Phi-3-mini-4k-instruct")
extractor = SurveyExtractor(model_name="meta-llama/Llama-3.2-1B-Instruct")

# replace w file path & document id
result = extractor.extract(
    pdf_path=Path("data/pdfs/J of Consumer Behaviour - 2021 - Nguyen - The effect of AI quality on customer experience and brand relationship.pdf"),
    document_id="D0003"                         
)

print(json.dumps(result, indent=2))