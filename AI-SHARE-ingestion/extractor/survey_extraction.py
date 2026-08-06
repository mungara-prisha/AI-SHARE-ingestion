import os
import json
import pdfplumber
import torch
from pathlib import Path
from typing import Optional
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
from dotenv import load_dotenv


# surveyID is based on logic rules, not LLM

load_dotenv()

RESEARCH_TYPE_DESCRIPTIONS = """
1: Survey — Standardized questions, no manipulation, goal is to describe or generalize to a population.
2: Survey Experiment (non-conjoint) — Researchers manipulate an independent variable and measure its effect. Keywords: experiment, random assignment, treatment, manipulation, causality, effect, impact.
3: Conjoint Experiment — Respondents choose between profiles with varying attributes. Keywords: conjoint, attribute, profile, trade-off.
4: Longitudinal Survey — Same or similar survey conducted at multiple time points, no experiment. Goal: track changes over time.
5: Longitudinal Survey Experiment — Experiment administered over multiple waves, measuring pre/post treatment effects.
"""

#exp_type is only for research types 2, 3, and 5

EXPERIMENT_TYPE_DESCRIPTIONS = """
1: Online survey experiment — Experiment conducted entirely online via survey platform.
2: Online field experiment with survey components — Manipulation happens in the real world online (e.g., emails sent to participants), outcome measured via survey.
3: Offline lab experiment with survey components — Manipulation happens in a controlled physical lab setting, outcome measured via survey.
4: Offline field experiment with survey components — Manipulation happens in a real-world physical setting (e.g., classroom), outcome measured via survey.
"""


class SurveyExtractor:
    def __init__(self, model_name: str = "meta-llama/Llama-3.1-8B-Instruct"):
        self.model_name = model_name
        self.token = os.environ.get("HF_TOKEN")

        if not self.token:
            raise ValueError("HF_TOKEN not found. Check your .env file.")

        print("Loading model... this may take a few minutes.")
        self.model, self.tokenizer = self._load_model()
        print("Model loaded successfully.")

    def _load_model(self):
        model = AutoModelForCausalLM.from_pretrained(
            self.model_name,
            device_map="cpu",
            torch_dtype=torch.float32,
            token=self.token
        )
        tokenizer = AutoTokenizer.from_pretrained(
            self.model_name,
            token=self.token
        )
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token
        return model, tokenizer

    def extract_text(self, pdf_path: Path) -> str:
        """Extract raw text from a PDF file."""
        text = []
        with pdfplumber.open(pdf_path) as pdf:
            for page in pdf.pages:
                page_text = page.extract_text()
                if page_text:
                    text.append(page_text)
        full_text = "\n".join(text)

        if not full_text.strip():
            raise ValueError(f"No text could be extracted from {pdf_path}")

        return full_text

    def truncate_text(self, text: str, max_chars: int = 6000) -> str:
        """
        Use the first and last portion of the paper.
        Abstract/methods are at the start; design details often at the end.
        """
        if len(text) <= max_chars:
            return text
        half = max_chars // 2
        return text[:half] + "\n\n[...middle truncated...]\n\n" + text[-half:]

    def _run_model(self, prompt: str) -> str:
        """Run the LLM on a prompt and return the raw text output."""
        inputs = self.tokenizer(
            prompt,
            return_tensors="pt",
            truncation=True,
            max_length=4096
        ).to(self.model.device)

        with torch.no_grad():
            outputs = self.model.generate(
                **inputs,
                max_new_tokens=300,
                do_sample=False,
                eos_token_id=self.tokenizer.eos_token_id,
                pad_token_id=self.tokenizer.eos_token_id,
            )

        # Decode only the newly generated tokens, not the input prompt
        response_tokens = outputs[0][inputs["input_ids"].shape[-1]:]
        return self.tokenizer.decode(response_tokens, skip_special_tokens=True).strip()

    def _safe_json_load(self, text: str) -> Optional[dict]:
        """Attempt to parse JSON from LLM output robustly."""
        if not text:
            return None

        # Try direct parse first
        try:
            return json.loads(text)
        except Exception:
            pass

        # Try extracting first JSON block
        try:
            start = text.find("{")
            end = text.rfind("}")
            if start != -1 and end != -1:
                return json.loads(text[start:end + 1])
        except Exception:
            pass

        # Try stripping markdown code fences
        try:
            cleaned = text.replace("```json", "").replace("```", "").strip()
            start = cleaned.find("{")
            end = cleaned.rfind("}")
            if start != -1 and end != -1:
                return json.loads(cleaned[start:end + 1])
        except Exception:
            pass

        print(f"JSON parse failed. Raw output was:\n{text}")
        return None

    # ------------------------------------------------------------------ #
    #  RESEARCH TYPE
    # ------------------------------------------------------------------ #

    def extract_research_type(self, text: str) -> dict:
        """Extract research type(s) from paper text."""
        truncated = self.truncate_text(text)

        prompt = f"""You are a social science research assistant. Your job is to classify the research design of a paper.

        Classify the paper into one or more of these research types:
        {RESEARCH_TYPE_DESCRIPTIONS}

        A paper may have multiple research types if it contains multiple studies.
        Return ONLY valid JSON in exactly this format, nothing else:
        {{
        "research_types": [1],
        "confidence": 0.85,
        "evidence": "brief quote or observation from the paper that justifies your classification"
        }}

        Paper text:
        {truncated}
        """
        raw = self._run_model(prompt)
        parsed = self._safe_json_load(raw)

        if not parsed:
            return {"research_types": None, "confidence": None, "evidence": "parse failed", "raw_output": raw}

        # Validate values are in range 1-5
        types = parsed.get("research_types", [])
        if not isinstance(types, list) or not all(isinstance(t, int) and t in {1,2,3,4,5} for t in types):
            return {"research_types": None, "confidence": None, "evidence": "invalid types returned", "raw_output": raw}

        return parsed

    # ------------------------------------------------------------------ #
    #  EXPERIMENT TYPE
    # ------------------------------------------------------------------ #

    # Prompts are separate for the better "attention" in LLM

    def extract_experiment_type(self, text: str, research_types: list) -> dict:
        """
        Extract experiment type. Only runs if research type is 2, 3, or 5.
        Returns blank if research type is 1 or 4, matching codebook rules.
        """
        experimental_types = {2, 3, 5}

        if not research_types or not any(t in experimental_types for t in research_types):
            return {"experiment_type": None, "note": "Not applicable — research type is 1 or 4"}

        truncated = self.truncate_text(text)

        prompt = f"""You are a social science research assistant. Your job is to classify the experiment type of a paper.

        Classify the experiment into one of these types:
        {EXPERIMENT_TYPE_DESCRIPTIONS}

        Return ONLY valid JSON in exactly this format, nothing else:
        {{
        "experiment_type": 1,
        "confidence": 0.85,
        "evidence": "brief quote or observation from the paper that justifies your classification"
        }}

        Paper text:
        {truncated}
        """
        raw = self._run_model(prompt)
        parsed = self._safe_json_load(raw)

        if not parsed:
            return {"experiment_type": None, "confidence": None, "evidence": "parse failed", "raw_output": raw}

        exp_type = parsed.get("experiment_type")
        if exp_type not in {1, 2, 3, 4}:
            return {"experiment_type": None, "confidence": None, "evidence": "invalid experiment type returned", "raw_output": raw}

        return parsed

    # ------------------------------------------------------------------ #
    #  SURVEY ID
    # ------------------------------------------------------------------ #

    def construct_survey_id(self, document_id: str, num_surveys: int = 1) -> list:
        """
        Survey ID is constructed from Document ID, not extracted by LLM.
        Format: [DocumentID]S1, [DocumentID]S2, etc.
        """
        return [f"{document_id}S{i}" for i in range(1, num_surveys + 1)]

    # ------------------------------------------------------------------ #
    #  MAIN ENTRY POINT
    # ------------------------------------------------------------------ #

    def extract(self, pdf_path: Path, document_id: str) -> dict:
        """
        Run the full extraction pipeline for a single paper.
        Returns a dict of all extracted variables.
        """
        print(f"\nProcessing: {pdf_path.name}")

        text = self.extract_text(pdf_path)
        print(f"Extracted {len(text)} characters of text.")

        # Research Type
        print("Extracting research type...")
        research_type_result = self.extract_research_type(text)
        research_types = research_type_result.get("research_types") or []

        # Experiment Type (conditional on research type)
        print("Extracting experiment type...")
        experiment_type_result = self.extract_experiment_type(text, research_types)

        # Survey ID (logic-based, not LLM)
        # For now assume 1 survey per paper — can be updated later
        survey_ids = self.construct_survey_id(document_id, num_surveys=1)

        return {
            "document_id": document_id,
            "survey_ids": survey_ids,
            "research_type": research_type_result,
            "experiment_type": experiment_type_result,
        }