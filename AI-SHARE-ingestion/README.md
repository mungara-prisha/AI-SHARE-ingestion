# AI-SHARE-ingestion
Repository for AI SHARE's automated ingestion

Previous repository:
https://github.com/mungara-prisha/AI-SHARE

# Updates (starting 7/30):

## Shireen (8/5):

### Overall Architecture: extractor.py

LLMs have a well-documented tendency to degrade in accuracy as prompts get longer and more complex. One prompt per variable means each extraction is independently debuggable, improvable, and evaluable.

###Text Extraction: pdfplumber

The extract_text method simply opens the PDF, iterates through every page, pulls the text, and joins it all into one long string. The if page_text check is there because some pages, cover pages, pages with only images or figures, return None or empty strings, and joining those would introduce garbage into text.

### Truncation Strategy: First and Last, Not Middle

Academic papers are typically much longer than what can be safely feed into an LLM prompt without running into context window limits or degraded performance.
Research design information such as how the study was conducted, what kind of experiment it was, how participants were sampled appears primarily in the Methods section, which is usually in the first third of the paper. But some details, like wave structure in longitudinal studies or experiment type clarifications, sometimes appear in later sections, so this strategy is subject to change.

The truncate_text method takes the first half and the last half of the paper text and discards the middle. The middle of academic papers is typically Results and Discussion. This is useful for human readers but not for extracting methodological variables. This gives Abstract + Introduction + Methods at the start, and Conclusion + Appendix details at the end, which together contain almost everything that's need.

The max_chars=6000 default choice as it keeps the total prompt length manageable for the 8B model without being so restrictive that the key information is missed.

### The Prompt Structure:
Each prompt follows the same pattern deliberately.

[Role assignment]
[Task description]
[Valid categories with descriptions]
[Strict output format instruction]
[Paper text]

### Clean JSON formatting:
The _safe_json_load method tries three increasingly lenient approaches to parse the model's output:

First, direct json.loads() works if the model perfectly followed instructions. Second, finding the first { and last } and extracting what's between them handles cases where the model added a sentence before or after the JSON. Third, stripping markdown code fences (```json ```) handles cases where the model wrapped its JSON in a code block despite being told not to.

### Experiment Type: Conditional Logic

python
if not research_types or not any(t in experimental_types for t in research_types):
    return {"experiment_type": None, "note": "Not applicable. Research type is 1 or 4"}

This directly encodes the codebook rule: Experiment Type is only relevant if Research Type is 2, 3, or 5. If the paper is a plain survey (1) or longitudinal survey (4), there is no experiment type to extract, and asking the model to extract it anyway would produce meaningless output.

The main extract() method is deliberately thin. It calls the individual extraction methods in the right order and assembles the results into a single dictionary. The ordering matters: Research Type must be extracted before Experiment Type, because Experiment Type's logic depends on the Research Type result.

This orchestrator pattern also makes it easy to add new variables later.