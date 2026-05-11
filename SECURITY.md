# Security Mitigations

## Mandatory Technical Disclosures

### LLM Chosen
| Field | Detail |
|---|---|
| Model | Llama 3.2 3B (default) / Llama 3 8B (optional) |
| Provider | Meta (via Ollama, self-hosted on localhost) |
| Version | Latest stable via `ollama pull llama3.2:3b` |
| Why | 100% local (zero data leakage), free, no API costs, fits in 4GB VRAM (RTX 3050), reliable structured JSON output |

### Agent Framework
| Field | Detail |
|---|---|
| Framework | LangGraph 0.2.x (LangChain ecosystem) |
| Architecture | Multi-agent stateful graph |
| Pattern | Plan-and-Execute with Critic re-score loop |
| Flow | Parser → Scorer → Critic (optional) → Report |

### Prompt Design
- **System prompts** per agent with role, task, rules, and JSON schema
- **Guardrails**: Input sanitisation, max length enforcement, JSON-only output enforcement
- **Anti-hallucination**: "Only score based on evidence found in the resume"
- **Structured output**: Pydantic validation on every LLM output before pipeline continues

---

## Security Risk Mitigation Table

| Risk | Description | Mitigation | Implementation Status |
|---|---|---|---|
| **Prompt Injection** | Malicious resume content manipulating LLM behaviour | `_sanitise_input()` strips dangerous tokens: `ignore previous instructions`, `<<SYS>>`, `[INST]`, `<system>`, `<\|im_start\|>`. All output forced to JSON schema via `format="json"`. Pydantic validates every response. | **Implemented** — `agents/parser_agent.py` lines 85-101 |
| **Data Privacy / PII** | Resumes contain names, emails, phone numbers, DOB | 100% local Ollama (no data leaves machine). `anonymise_text()` strips name, email, phone, gender pronouns, DOB, age indicators before scoring. Only anonymised text sent to Scorer. Raw text stored locally for audit only. | **Implemented** — `services/bias_detector.py` |
| **API Key Exposure** | Langfuse / external API keys leaking in code | `.env` file loaded via `python-dotenv`. All secrets read via `os.getenv()`. `.env` is listed in `.gitignore` (line 2). `.env.example` contains placeholders only. No hardcoded secrets anywhere in codebase. | **Implemented** — `.gitignore`, `.env.example` |
| **Hallucination Risk** | LLM fabricating skills or wrong scores | `format="json"` + Pydantic schema enforcement. Critic Agent validates every score against resume text. Confidence below 0.6 triggers automatic re-score (max 2 retries). Human override tab with audit log. | **Implemented** — `agents/critic_agent.py`, `models/schemas.py` |
| **Unauthorised Access** | Anyone triggering the agent pipeline | App runs on `localhost:8501` only. Optional password gate via `st.secrets["APP_PASSWORD"]`. Rate limiting: max 10 pipeline runs per session (`MAX_RUNS_PER_SESSION = 10` in `app.py`). | **Implemented** — `app.py` lines 25-51 |
| **Bias Risk** | Scoring influenced by candidate demographics | PII anonymisation pre-scoring (names, pronouns, age stripped). Evidence-only prompts ("only score on what is in the resume"). Post-hoc fairness audit in `generate_fairness_report()`. Transparent 5-dimension rubric. Human override with audit trail. | **Implemented** — `services/bias_detector.py` |
| **Output Integrity** | Generated reports contain wrong data | All reports generated from validated Pydantic models (`ShortlistReport`). Score overrides stored in SQLite with reviewer name, reason, and timestamp. Audit log shown in Overrides tab. | **Implemented** — `models/database.py`, `services/report_generator.py` |

---

## Implementation Evidence

### 1. Prompt Injection Guard — `agents/parser_agent.py`
```python
def _sanitise_input(text: str) -> str:
    dangerous_patterns = [
        "ignore previous instructions", "ignore all previous",
        "disregard above", "<|im_start|>", "<|im_end|>",
        "<<SYS>>", "[INST]", "<system>",
    ]
    cleaned = text
    for pattern in dangerous_patterns:
        cleaned = cleaned.replace(pattern, "[REDACTED]")
        cleaned = cleaned.replace(pattern.upper(), "[REDACTED]")
    return cleaned
```

### 2. PII Anonymisation — `services/bias_detector.py`
```python
def anonymise_text(text: str) -> str:
    # Strips: candidate name (first line), gender pronouns,
    # age/DOB indicators, email addresses, phone numbers, URLs
    # Returns anonymised text safe for unbiased scoring
```

### 3. API Key Protection — `.gitignore`
```
.env          # Line 2 of .gitignore — secrets never committed
```

### 4. Pydantic Output Validation — `models/schemas.py`
```python
class CandidateProfile(BaseModel):
    name: str
    email: str | None
    linkedin_url: str | None
    # ... all fields validated on every LLM output
```

### 5. Confidence Threshold + Re-score Loop — `agents/critic_agent.py`
```
Critic assigns confidence: float (0-1)
If confidence < 0.6: candidate added to retry_candidates
LangGraph routes back to Scorer (max 2 retries)
```

### 6. Password Gate — `app.py`
```python
# Add to .streamlit/secrets.toml to enable:
# APP_PASSWORD = "your-password-here"
_app_password = st.secrets.get("APP_PASSWORD", "")
# If set, shows login screen before allowing any access
```

### 7. Rate Limiting — `app.py`
```python
MAX_RUNS_PER_SESSION = 10
# Enforced before every pipeline execution
# Prevents runaway GPU usage and abuse
```

---

## How to Enable Password Protection

Create `.streamlit/secrets.toml`:
```toml
APP_PASSWORD = "your-secure-password-here"
```

Then restart the app with `streamlit run app.py`. A login screen will appear before any access is granted.

This file must also be added to `.gitignore` before committing:
```
.streamlit/secrets.toml
```

---

## Data Flow — Security Perspective

```
User uploads resume PDF
    -> detect_and_parse_bytes()  [no external calls]
    -> _sanitise_input()         [prompt injection stripped]
    -> parse_resume() via Ollama [localhost only, no cloud]
    -> anonymise_text()          [PII removed before scoring]
    -> score_candidate()         [only anonymised text used]
    -> Pydantic validation       [schema enforced]
    -> Critic validation         [confidence checked]
    -> Human override UI         [full audit trail in SQLite]
    -> Report generation         [from validated Pydantic models]

No resume data ever leaves the local machine.
```
