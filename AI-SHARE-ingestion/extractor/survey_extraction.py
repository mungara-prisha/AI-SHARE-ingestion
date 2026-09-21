import os
import json
import pdfplumber
import torch
from pathlib import Path
from typing import Optional
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
from dotenv import load_dotenv


load_dotenv()

RESEARCH_TYPE_DESCRIPTIONS = """
1: Survey — Standardized questions, no manipulation, goal is to describe or generalize to a population.
2: Survey Experiment (non-conjoint) — Researchers manipulate an independent variable and measure its effect. Keywords: experiment, random assignment, treatment, manipulation, causality, effect, impact.
3: Conjoint Experiment — Respondents choose between profiles with varying attributes. Keywords: conjoint, attribute, profile, trade-off.
4: Longitudinal Survey — Same or similar survey conducted at multiple time points, no experiment. Goal: track changes over time.
5: Longitudinal Survey Experiment — Experiment administered over multiple waves, measuring pre/post treatment effects.
"""
# IMP: Figure of multiple survey logic. Different surveys in same study -> different surveys. EXCEPTION - longitudinal survey exp. 

#exp_type is only for research types 2, 3, and 5 (non-conjoint exp, conjoint exp, longitudinal exp)
# can have multiple rows with ResearchType == 1 and == 2 for a single unique paper.

EXPERIMENT_TYPE_DESCRIPTIONS = """
1: Online survey experiment — Experiment conducted entirely online via survey platform.
2: Online field experiment with survey components — Manipulation happens in the real world online (e.g., emails sent to participants), outcome measured via follow-up survey.
3: Offline lab experiment with survey components — Manipulation happens in a controlled physical lab setting, outcome measured via follow-up survey. Researchers distribute hard-copy informational materials to undergraduate participants.
4: Offline field experiment with survey components — Manipulation happens in a real-world physical setting (e.g., classroom), outcome measured via follow-up survey. Researchers distribute hard-copy informational materials to undergraduate participants.
"""

TREATMENT_TYPE_DESCRIPTIONS = """
1: Informational Treatments — Treatments that involve (either real-world or hypothetical) evidence, statistics, or factual data.
   Examples: news articles presenting real or fabricated reports (e.g., about automation replacing jobs, AI regulation); expert reports and white papers; statistical infographics with charts, graphs, or tables; fact-based policy briefs outlining pros and cons; economic model projections presented in a neutral, data-driven way (e.g., "AI will automate 30% of jobs by 2050").

2: Narrative & Scenario-Based Treatments — Treatments that involve (either real-world or hypothetical) vignettes, narratives, or personalized scenarios.
   Examples: short fictional stories (e.g., a worker who lost their job to AI vs. one who benefits from upskilling); personalized consumer stories; testimony-style accounts (e.g., a doctor's perspective on AI diagnostics); near-future speculative scenarios (e.g., "Imagine it's 2050 and AI has replaced factory jobs"); role-playing exercises (e.g., imagine yourself as a policymaker); mock situations where an AI system evaluates the respondent.

3: Behavioral & Incentive-Based Treatments — Treatments that involve monetary incentives, behavioral nudges, or direct participation-based interventions.
   Examples: mock job applications where participants actively interact with an AI ranking system; sending real event invitations to measure attendance; having participants test and review a live AI chatbot and collect impressions.

Classification notes:
- If a treatment presents projections in a neutral, data-driven way, classify as 1 (Informational).
- If a treatment frames projections as a speculative or role-playing scenario, classify as 2 (Narrative).
- A paper may use multiple treatment types — return all that apply as a list.
"""

TREATMENT_MODALITY_DESCRIPTIONS = """
1: Text — The treatment is delivered via written text only (e.g., a written news article, a text-based vignette, written policy briefs).
   IMPORTANT: Textual instructions telling participants how to engage with the treatment do NOT count as text modality. Classify only based on the content of the treatment itself.

2: Audiovisual — The treatment is delivered via images, video, audio, or other visual/audio media (e.g., a video clip, an infographic, photographs).

3: Interactive — The treatment involves participants actively engaging with it in a controlled environment (e.g., interactive policy simulations, gamified decision-making exercises, live chatbot interactions where the participant receives real-time responses).

4: Other — The treatment modality does not fit any of the above. If selecting this, describe the modality in the evidence field.

A paper may use multiple modality types — return all that apply as a list.
"""

TREATMENT_ARMS_INSTRUCTIONS = """
Identify and label each experimental condition (treatment arm or factor level) used in this non-conjoint survey experiment.
Follow the authors' original terminology as closely as possible. Check the full paper including any appendix or supplementary materials.

Formatting rules:
- For experiments with discrete treatment arms: list each arm separated by commas.
  Example: "Control, Pro-AI Regulation, Anti-AI Regulation"
  Example: "Placebo, Negative, Positive"

- For factorial designs with multiple factors: use the format "[Factor]: [Level1] vs. [Level2]" per factor, separated by semicolons.
  Example: "Candidate Gender: Male vs. Female; Policy Position: Liberal vs. Conservative"
  Example: "Guideline compliance: Applying vs. Violating; AI performance: Optimal vs. Sub-optimal"

- For a scenario-based design with multiple named scenarios, list each scenario name separated by commas.
  Example: "Self-Driving Cars Enhance Human Judgment, Self-Driving Cars Substitute for Human Judgment, Armed Drones Enhance Human Judgment, Armed Drones Substitute for Human Judgment"
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
        Abstract/methods are at the start; design details and appendices often at the end.
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

        # add one-shot examples to the prompt ⭐️

        raw = self._run_model(prompt)
        parsed = self._safe_json_load(raw)

        if not parsed:
            return {"research_types": None, "confidence": None, "evidence": "parse failed", "raw_output": raw}

        # Validate values are in range 1-5
        types = parsed.get("research_types", [])
        if not isinstance(types, list) or not all(isinstance(t, int) and t in {1,2,3,4,5} for t in types):
            return {"research_types": None, "confidence": None, "evidence": "invalid types returned", "raw_output": raw}

        # check confidence logic - how is it calculated?

        return parsed

    # ------------------------------------------------------------------ #
    #  EXPERIMENT TYPE
    # ------------------------------------------------------------------ #

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

        # add one-shot examples to the prompt ⭐️

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

    # surveyID is based on logic rules, not LLM extraction.

    def construct_survey_id(self, document_id: str, num_surveys: int = 1) -> list:
        """
        Survey ID is constructed from Document ID, not extracted by LLM.
        Format: [DocumentID]S1, [DocumentID]S2, etc.
        """
        return [f"{document_id}S{i}" for i in range(1, num_surveys + 1)]

    # ------------------------------------------------------------------ #
    #  TREATMENT
    # ------------------------------------------------------------------ #

    def extract_treatment(self, text: str, research_types: list) -> dict:
        """
        Extract treatment type(s). Only runs if research type is 2, 3, or 5.
        Returns blank if research type is 1 or 4, matching codebook rules.
        Multiple treatment types are possible and returned as a list.
        """
        applicable_types = {2, 3, 5}

        if not research_types or not any(t in applicable_types for t in research_types):
            return {"treatment_types": None, "note": "Not applicable — research type is 1 or 4"}

        truncated = self.truncate_text(text)

        prompt = f"""You are a social science research assistant. Your job is to classify the treatment type(s) used in a survey experiment.

        Classify the treatment(s) into one or more of these types:
        {TREATMENT_TYPE_DESCRIPTIONS}

        Return ONLY valid JSON in exactly this format, nothing else:
        {{
        "treatment_types": [1],
        "confidence": 0.85,
        "evidence": "brief quote or observation from the paper that justifies your classification"
        }}

        Paper text:
        {truncated}
        """

        # add one-shot examples to the prompt ⭐️

        raw = self._run_model(prompt)
        parsed = self._safe_json_load(raw)

        if not parsed:
            return {"treatment_types": None, "confidence": None, "evidence": "parse failed", "raw_output": raw}

        types = parsed.get("treatment_types", [])
        if not isinstance(types, list) or not all(isinstance(t, int) and t in {1, 2, 3} for t in types):
            return {"treatment_types": None, "confidence": None, "evidence": "invalid treatment types returned", "raw_output": raw}

        return parsed

    # ------------------------------------------------------------------ #
    #  TREATMENT MODALITY
    # ------------------------------------------------------------------ #

    def extract_treatment_modality(self, text: str, research_types: list) -> dict:
        """
        Extract treatment modality type(s). Only runs if research type is 2, 3, or 5.
        Returns blank if research type is 1 or 4, matching codebook rules.
        Multiple modality types are possible and returned as a list.
        If type 4 (Other) is returned, the evidence field should describe the modality.
        """
        applicable_types = {2, 3, 5}

        if not research_types or not any(t in applicable_types for t in research_types):
            return {"modality_types": None, "note": "Not applicable — research type is 1 or 4"}

        truncated = self.truncate_text(text)

        prompt = f"""You are a social science research assistant. Your job is to classify the treatment modality type(s) used in a survey experiment.

        Classify the treatment modality into one or more of these types:
        {TREATMENT_MODALITY_DESCRIPTIONS}

        If you select type 4 (Other), describe the modality in the evidence field.

        Return ONLY valid JSON in exactly this format, nothing else:
        {{
        "modality_types": [1],
        "confidence": 0.85,
        "evidence": "brief quote or observation from the paper that justifies your classification"
        }}

        Paper text:
        {truncated}
        """

        # add one-shot examples to the prompt ⭐️

        raw = self._run_model(prompt)
        parsed = self._safe_json_load(raw)

        if not parsed:
            return {"modality_types": None, "confidence": None, "evidence": "parse failed", "raw_output": raw}

        types = parsed.get("modality_types", [])
        if not isinstance(types, list) or not all(isinstance(t, int) and t in {1, 2, 3, 4} for t in types):
            return {"modality_types": None, "confidence": None, "evidence": "invalid modality types returned", "raw_output": raw}

        return parsed

    # ------------------------------------------------------------------ #
    #  TREATMENT ARMS / FACTORS
    # ------------------------------------------------------------------ #

    def extract_treatment_arms(self, text: str, research_types: list) -> dict:
        """
        Extract treatment arms/factors. Only runs if research type is 2 or 5.
        Not applicable for conjoint experiments (RT 3), which define conditions via
        attributes rather than discrete arms. Also not applicable for RT 1 or 4.

        Returns a free-text string following codebook formatting conventions:
        - Discrete arms: comma-separated labels (e.g., "Control, Treatment A, Treatment B")
        - Factorial designs: "[Factor]: [L1] vs. [L2]; [Factor]: [L1] vs. [L2]"
        - Named scenarios: comma-separated scenario names

        Note: treatment arm details often appear in appendices, which are captured
        by the tail of truncate_text. Increase max_chars if arms are being missed.
        """
        applicable_types = {2, 5}

        if not research_types or not any(t in applicable_types for t in research_types):
            return {"treatment_arms": None, "note": "Not applicable — research type is 1, 3, or 4"}

        # Use a larger window to improve appendix capture
        truncated = self.truncate_text(text, max_chars=8000)

        prompt = f"""You are a social science research assistant. Your job is to identify and label all experimental conditions in a non-conjoint survey experiment.

        {TREATMENT_ARMS_INSTRUCTIONS}

        Return ONLY valid JSON in exactly this format, nothing else:
        {{
        "treatment_arms": "Control, Treatment A, Treatment B",
        "confidence": 0.85,
        "evidence": "brief quote or observation from the paper that justifies your classification"
        }}

        Paper text:
        {truncated}
        """

        # add one-shot examples to the prompt ⭐️

        raw = self._run_model(prompt)
        parsed = self._safe_json_load(raw)

        if not parsed:
            return {"treatment_arms": None, "confidence": None, "evidence": "parse failed", "raw_output": raw}

        arms = parsed.get("treatment_arms")
        if not isinstance(arms, str) or not arms.strip():
            return {"treatment_arms": None, "confidence": None, "evidence": "invalid treatment arms returned", "raw_output": raw}

        return parsed

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

        # Experiment Type (conditional: RT 2, 3, 5)
        print("Extracting experiment type...")
        experiment_type_result = self.extract_experiment_type(text, research_types)

        # Treatment (conditional: RT 2, 3, 5)
        print("Extracting treatment...")
        treatment_result = self.extract_treatment(text, research_types)

        # Treatment Modality (conditional: RT 2, 3, 5)
        print("Extracting treatment modality...")
        treatment_modality_result = self.extract_treatment_modality(text, research_types)

        # Treatment Arms/Factors (conditional: RT 2, 5 only — not conjoint)
        print("Extracting treatment arms/factors...")
        treatment_arms_result = self.extract_treatment_arms(text, research_types)

        # Survey ID
        survey_ids = self.construct_survey_id(document_id, num_surveys=1)

        return {
            "document_id": document_id,
            "survey_ids": survey_ids,
            "research_type": research_type_result,
            "experiment_type": experiment_type_result,
            "treatment": treatment_result,
            "treatment_modality": treatment_modality_result,
            "treatment_arms": treatment_arms_result,
        }