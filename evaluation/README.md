# Small RAG Evaluation

## Dataset and method

Two fictional documents describe a cafe and a library.
The dataset contains eight answerable questions and two questions
whose answers are absent from the documents.

Run from the repository root:

    .venv\Scripts\python.exe evaluation\run.py

The runner builds an isolated temporary Chroma database and records
answers, source excerpts and response times in results.json.
It makes up to ten requests to the configured LLM provider.

The reviewed run is preserved in baseline_results.json.
Manual verdicts compare answers with the expected answers and documents.

## Baseline results

Run date: 2 October 2026.
Provider: Groq.
Model: openai/gpt-oss-20b.
Retrieval setting: top_k = 4.

| Measure | Result |
|---|---|
| Manually judged correct responses | 9/10 |
| Correct answers to answerable questions | 7/8 |
| Correct refusals for unanswerable questions | 2/2 |
| Mean document-source recall on answerable questions | 87.5% |
| Request errors | 0 |

Document-source recall measures the fraction of expected document
filenames returned for each answerable question, averaged across
those questions. It does not measure chunk relevance or citation accuracy.

## Failure analysis

The library renewal question received an insufficient-information
response despite the answer being present in the library document.

A retrieval diagnostic showed:
- An unrelated cafe chunk had distance 1.4838.
- The library chunk containing the answer had distance 1.5549.
- The current distance filter accepts only scores below 1.5.

The filter therefore retained the cafe chunk and excluded the relevant
library chunk. The threshold was left unchanged. Future tuning should
use separate development and held-out evaluation questions.

## Limitations

This is a small synthetic smoke evaluation, not a general accuracy benchmark.
It uses two TXT documents and one run with manual answer review.
It does not evaluate PDFs, large collections or adversarial inputs.
Source excerpts are truncated and may omit supporting sentences.
The distance metric is not established by this report; the retrieval
code's existing cosine-distance comment should not be treated as verified.
Model responses and timings may vary between runs.
