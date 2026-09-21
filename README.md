# AI-SHARE-ingestion
Repository for AI SHARE's automated ingestion

Previous repository:
https://github.com/mungara-prisha/AI-SHARE

Updates (starting 7/30):

# 8/5/2026:

Questions to revisit:
How to scan for certain variables in a long paper? Chunk by chunk? Looking in only certain predefined places? Analysis of random lines in a paragraph?

Tasks: Prisha will automate question-level variables including question extraction, question topic, concept, and response information.
Shireen will automate survey-level variables including experiment type, treatment, treatment modality, and respondent information.

File updates: new code for question extraction. has yet to be fully tested and refined.


# 9/21/2026:

Shireen:
Added code prompts for Treatment, Treatment Modality, and Treatment Arms/Factors.
- Need to add one-shot examples for all prompts -> then send final prompts in Slack gc for team to review.
- Need to test using anvil, check if model needs changing.
- Revisit text extraction logic (8-bit quantization etc) since Anvil allows for higher bandwidth.

