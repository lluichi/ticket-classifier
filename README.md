# Smart Customer Support Ticket Classifier

MVP prototype for AI-powered customer support ticket classification. Uses **Gemini Flash** with structured output for zero-shot classification of customer messages across multiple channels.

## What It Does

The classifier receives a raw customer message and returns a structured classification:

- **Urgency** — critical / high / medium / low
- **Intent** — complaint, question, request, feedback, bug_report
- **Product area** — billing, technical, onboarding, general
- **Language** — auto-detected (ISO code)
- **Confidence score** — 0 to 1
- **Suggested reply** — draft response for the customer
- **Human escalation flag** — `true` if urgency is critical or confidence < 0.7

### Example Output

| Ticket | Urgency | Intent | Product | Lang | Human? | Latency |
|--------|---------|--------|---------|------|--------|---------|
| Platform down, 2h deadline | CRITICAL | complaint | technical | en | Yes | ~350ms |
| How to export PDF report? | LOW | question | general | en | No | ~280ms |
| Double charge, refund request | HIGH | complaint | billing | en | Yes | ~310ms |
| Spanish: plan features inquiry | LOW | question | general | es | No | ~290ms |
| Positive feedback + dark mode | LOW | feedback | general | en | No | ~260ms |

## Architecture Context

In the full MVP, this classifier sits inside a **Google ADK orchestrator** as part of an event-driven pipeline:

```
Incoming message → PII Anonymizer → Cloud Pub/Sub → Orchestrator (ADK)
    → Classifier (this code) → RAG (Vertex AI Search) → Response
    → Confidence routing → Auto-reply or Human Agent Dashboard
```

This prototype extracts the classification step as a standalone, runnable script.

## Setup

### Prerequisites

- Python 3.10+
- A Google AI API key (free at [aistudio.google.com/apikey](https://aistudio.google.com/apikey))

### Installation

```bash
# Clone the repository
git clone https://github.com/lluichi/ticket-classifier.git
cd ticket-classifier

# Install dependencies
pip install google-genai pydantic python-dotenv pytest
```

### Configuration

Create a `.env` file in the project root:

```env
GOOGLE_API_KEY=your_key_here
GEMINI_MODEL=gemini-2.0-flash
```

The model is configurable via the `GEMINI_MODEL` environment variable, allowing easy upgrades to newer Gemini versions.

## Usage

### Run the classifier

```bash
python classifier.py
```

Processes 5 sample tickets covering different scenarios: critical outage, FAQ question, billing complaint, Spanish-language inquiry, and positive feedback.

### Run tests

```bash
python -m pytest classifier.py -v
```

Three tests validate:
- **Schema compliance** — output matches the Pydantic model
- **Critical escalation** — critical tickets always flag `needs_human=True`
- **Multilingual detection** — Spanish messages are detected as `lang: es`

## Key Design Decisions

**Zero-shot classification** — No training data required to start. The system uses Gemini's structured output with a typed JSON schema (Pydantic) to enforce classification consistency. Few-shot examples can be added incrementally as real data accumulates.

**Business rules override LLM** — Deterministic logic (e.g., critical urgency always escalates) is enforced in code after LLM response, not delegated to the model. The LLM classifies; the code decides.

**Retry with exponential backoff** — Up to 3 attempts with `2^attempt` second delays. Failed classifications return `needs_human: True` as a safe fallback.

**Pydantic validation** — Every LLM response is validated against a strict schema. Invalid responses trigger a retry, not a silent failure.

## Project Structure

```
.
├── classifier.py       # Main classifier with tests
├── .env                # API key configuration (not committed)
├── .gitignore
└── README.md
```

## Evolution Path

This prototype is Phase 1 (zero-shot). The planned evolution:

- **Phase 2** — Few-shot prompting refined with highest-rated human corrections
- **Phase 3** — LoRA fine-tuning on Vertex AI using 500+ accumulated correction pairs, A/B tested before production promotion

## License

This project was built as part of a technical assessment.
