import json, os, sys, time, logging
from pydantic import BaseModel, Field, ValidationError
from enum import Enum
from typing import Optional
from dotenv import load_dotenv
load_dotenv()

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)


class Urgency(str, Enum):
    CRITICAL = "critical"
    HIGH = "high"
    MEDIUM = "medium"
    LOW = "low"


class Intent(str, Enum):
    COMPLAINT = "complaint"
    QUESTION = "question"
    REQUEST = "request"
    FEEDBACK = "feedback"
    BUG_REPORT = "bug_report"


class TicketClassification(BaseModel):
    urgency: Urgency = Field(description="Urgency level of the ticket")
    intent: Intent = Field(description="Primary intent of the message")
    product_area: str = Field(description="Product area: billing, technical, onboarding, general")
    language: str = Field(description="Detected language ISO code")
    confidence: float = Field(description="Classification confidence 0-1")
    suggested_reply: str = Field(description="Draft reply for the customer")
    needs_human: bool = Field(description="Whether human review is needed")


MODEL = os.environ.get("GEMINI_MODEL", "gemini-2.0-flash")
API_KEY = os.environ.get("GOOGLE_API_KEY")
MAX_RETRIES = 3
TIMEOUT_SECONDS = 30

SYSTEM_PROMPT = """You are a B2B SaaS customer support classifier.
Analyze each ticket and classify it accurately.
For suggested_reply: write a professional, empathetic response.
Set needs_human=true if: urgency is critical, confidence < 0.7,
or the issue requires account-specific investigation."""

SAMPLE_TICKETS = [
    {"channel": "email", "message": "Our entire team cannot access the platform since this morning. We have a client presentation in 2 hours. Please help ASAP!"},
    {"channel": "whatsapp", "message": "Hi, how do I export my monthly report to PDF?"},
    {"channel": "email", "message": "We have been charged twice for the Pro plan this month. Invoice #INV-2024-1847. Please refund immediately."},
    {"channel": "whatsapp", "message": "Hola, me gustaria saber si tienen soporte en espanol y si el plan basico incluye integracion con WhatsApp."},
    {"channel": "email", "message": "The new dashboard redesign is fantastic! Much easier to navigate. One suggestion: add dark mode support."},
]


def classify_ticket(message: str, channel: str) -> dict:
    """Classify a support ticket using Gemini structured output with retry logic."""
    from google import genai
    from google.genai import types

    if not API_KEY:
        logger.error("GOOGLE_API_KEY environment variable not set.")
        raise EnvironmentError("Missing GOOGLE_API_KEY")

    client = genai.Client(api_key=API_KEY)
    last_error = None

    for attempt in range(1, MAX_RETRIES + 1):
        try:
            start = time.time()
            response = client.models.generate_content(
                model=MODEL,
                contents=f"Channel: {channel}\nMessage: {message}",
                config=types.GenerateContentConfig(
                    system_instruction=SYSTEM_PROMPT,
                    response_mime_type="application/json",
                    response_schema=TicketClassification,
                    temperature=0.2,
                ),
            )
            latency = time.time() - start
            result = json.loads(response.text)
            # Validate against Pydantic schema
            TicketClassification(**result)
            logger.info(f"Classified in {latency:.2f}s (attempt {attempt})")
            result["latency_ms"] = round(latency * 1000)

            # Enforce business rules (don't trust LLM for deterministic logic)
            if result.get("urgency") == "critical" or result.get("confidence", 1) < 0.7:
                result["needs_human"] = True

            return result
        except (json.JSONDecodeError, ValidationError) as e:
            logger.warning(f"Schema validation failed (attempt {attempt}): {e}")
            last_error = e
        except Exception as e:
            logger.warning(f"API error (attempt {attempt}): {e}")
            last_error = e
            if attempt < MAX_RETRIES:
                time.sleep(2 ** attempt)  # Exponential backoff

    logger.error(f"All {MAX_RETRIES} attempts failed: {last_error}")
    return {"error": str(last_error), "needs_human": True}


# — Minimal tests (run with: python -m pytest classifier.py -v) —

def test_classification_schema():
    """Verify classification output complies with Pydantic schema."""
    ticket = SAMPLE_TICKETS[0]
    result = classify_ticket(ticket["message"], ticket["channel"])
    validated = TicketClassification(**result)
    assert 0 <= validated.confidence <= 1

def test_critical_tickets_need_human():
    """Critical urgency tickets must always escalate to human."""
    ticket = SAMPLE_TICKETS[0]  # Platform down = critical
    result = classify_ticket(ticket["message"], ticket["channel"])
    assert result["needs_human"] is True

def test_multilingual_detection():
    """Spanish ticket must be detected as Spanish."""
    ticket = SAMPLE_TICKETS[3]
    result = classify_ticket(ticket["message"], ticket["channel"])
    assert result["language"] == "es"

if __name__ == "__main__":
    if not API_KEY:
        print("ERROR: GOOGLE_API_KEY not set.")
        print("Get a free key: aistudio.google.com/apikey")
        sys.exit(1)

    for i, ticket in enumerate(SAMPLE_TICKETS):
        if i > 0:
            time.sleep(2)  # Respetar rate limits
        print(f"\n=== Ticket {i+1} ({ticket['channel']}) ===")
        print(f"  Message: {ticket['message'][:80]}...")
        result = classify_ticket(ticket["message"], ticket["channel"])
        for key, value in result.items():
            print(f"  {key}: {value}")
