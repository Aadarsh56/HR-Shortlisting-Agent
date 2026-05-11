# HR Resume Shortlisting Agent — Full Project Explanation

## What This Project Does

This is an AI-powered HR tool that automates the process of screening job candidates. Instead of an HR professional manually reading hundreds of resumes and trying to remember how each one compares, this system reads every resume automatically, scores each candidate against the job description, and produces a ranked shortlist with explanations for every decision.

The entire process runs locally on your machine. No resume data is sent to any cloud service. The AI runs through Ollama, which hosts a local language model (Llama 3) on your GPU.

---

## The Problem It Solves

Manual resume screening has three major problems:

1. **Fatigue** — After reading 50 resumes, a recruiter's judgment degrades. The 51st candidate is evaluated less carefully than the first.
2. **Inconsistency** — Different reviewers apply different standards. The same resume gets different scores depending on who reads it.
3. **Unconscious bias** — Names, universities, and demographic signals can influence decisions even when the reviewer tries to be objective.

This system addresses all three by standardising the evaluation criteria, applying the same rubric to every candidate, and anonymising resumes before they are scored.

---

## Architecture Overview

The system is built as a multi-agent pipeline using LangGraph. Think of it as an assembly line where each station does one specific job and passes the result to the next station.

```
Job Description + Resumes
         |
    [ Parser Agent ]
    Reads and structures all input into clean data
         |
    [ Scorer Agent ]
    Applies 5-dimension rubric to each candidate
         |
    [ Critic Agent ] (optional)
    Validates scores, checks for bias, flags low-confidence results
    If confidence is below 60%, sends candidate back to Scorer
         |
    [ Report Agent ]
    Computes skills gap, generates final rankings
         |
    Streamlit Dashboard
    Visual charts, download reports, human overrides
```

---

## Component-by-Component Breakdown

### 1. Parser Agent — `agents/parser_agent.py`

**What it does:** Takes raw text (from a PDF, DOCX, or LinkedIn JSON) and asks the language model to extract structured information from it.

**For the Job Description, it extracts:**
- Job title
- Required skills (must-have)
- Preferred skills (nice-to-have)
- Minimum and maximum years of experience
- Education requirements
- Certifications required
- Key responsibilities
- Industry domain

**For each resume, it extracts:**
- Candidate full name
- Email address
- Phone number
- LinkedIn profile URL
- GitHub profile URL
- Skills list (normalised to lowercase)
- Work experience (company, title, duration, description)
- Total years of experience
- Education history
- Projects and portfolio
- Certifications
- Professional summary

**How it handles failure:** The LLM sometimes returns malformed JSON. The parser tries up to 2 times, and uses a regex fallback to extract JSON from markdown code blocks if the model wraps its response that way. If parsing still fails, the candidate is skipped with a warning.

**Bias protection:** Before any resume text is sent for scoring, the PII anonymiser strips out the candidate's name, email, phone number, and any other identifying information. The scorer never sees who the candidate is — only their skills and experience.

**LinkedIn/GitHub extraction:** The parser first asks the LLM to extract these URLs, then applies regex patterns as a backup. So even if the LLM misses them, `github.com/username` patterns in the PDF text will still be caught.

---

### 2. Scorer Agent — `agents/scorer_agent.py`

**What it does:** Evaluates each candidate across five dimensions and assigns a score from 0 to 10 for each. A weighted total is then computed.

**The 5-Dimension Rubric:**

| Dimension | Weight | What It Measures |
|---|---|---|
| Skills Match | 30% | How many required and preferred skills the candidate has |
| Experience Relevance | 25% | Years of experience and how relevant their past roles are |
| Education and Certifications | 15% | Degree level, field of study, relevant certifications |
| Projects and Portfolio | 20% | Quality and relevance of projects and demonstrated work |
| Communication Quality | 10% | Clarity, structure, and professionalism of the resume itself |

**Weighted Total Formula:**
```
Total = sum(score x weight x 10) for each dimension
```
So a perfect 10/10 on all 5 dimensions gives a score of 100.

**Recommendation thresholds:**
- 75 and above: Hire
- 50 to 74: Maybe
- Below 50: No Hire

**Failure handling:** If the LLM returns bad scores, each value is clamped to the 0-10 range and any None values are defaulted to 5. If the entire scoring call fails, all dimensions get a default score of 5 with an auto-generated note.

---

### 3. Critic Agent — `agents/critic_agent.py`

**What it does:** Acts as a quality control step after scoring. It reviews the Scorer's output and decides whether the scores are trustworthy.

**It checks for:**
- Hallucinations — did the Scorer claim the candidate has a skill that is not in the resume?
- Overconfidence — did the Scorer give a 9/10 for a skill the candidate only mentioned once?
- Bias signals — are scores inconsistent in a way that suggests demographic information leaked through?
- Score consistency — do the individual dimension scores add up to a coherent overall picture?

**The re-scoring loop:** If the Critic's confidence score is below 0.6 (60%), it marks that candidate for re-scoring and sends them back to the Scorer agent. The maximum number of re-score attempts is 2 to prevent infinite loops.

**This agent is optional.** In the sidebar, you can toggle it off. Disabling it removes one LLM call per candidate, cutting processing time by roughly 30%.

---

### 4. Report Agent — `agents/report_agent.py`

**What it does:** Collects all evaluations and produces the final output.

**Specific tasks:**
- Runs the skills gap analysis for each candidate
- Sorts candidates by weighted total score (highest first)
- Assigns rank numbers (1st, 2nd, 3rd, etc.)
- Stores results in session state for the dashboard to display

---

### 5. Skills Gap Service — `services/skills_gap.py`

**What it does:** For each candidate, compares their skills against the JD's required and preferred skills, and categorises every skill into one of three buckets.

**Three buckets:**
1. **Matched** — Skills the JD requires that the candidate has
2. **Missing** — Skills the JD requires that the candidate does not have
3. **Bonus** — Skills the candidate has that the JD did not ask for

**Fuzzy matching:** The comparison is not just exact string matching. A skill normaliser handles common variations, and an alias map handles abbreviations and synonyms. For example:
- `pytest` matches `unit testing`
- `k8s` matches `kubernetes`
- `rest api design` matches `rest api`
- `rabbitmq` matches `message queues`
- `github actions` matches `ci/cd`

**Learning resources:** For every missing skill, the system looks up a curated resource from its built-in database (40+ skills covered) or generates a generic search recommendation. Each resource includes an estimated learning time.

---

### 6. File Parser Service — `services/file_parser.py`

Reads PDF and DOCX files and extracts their plain text content using `PyMuPDF` for PDFs and `python-docx` for DOCX files. Returns the raw text that is then passed to the Parser Agent.

**Important limitation:** If a PDF was created by scanning a physical document (an image scan), the file contains no machine-readable text. The parser will return an empty string, and the candidate will fail to parse. The resume must be a native digital PDF (created directly from Word, Google Docs, or a similar tool) to work correctly.

---

### 7. Bias Detector — `services/bias_detector.py`

Two functions:

1. `anonymise_text` — Strips PII from resume text before scoring. Removes: full name, email addresses, phone numbers, and common demographic signals.

2. `generate_fairness_report` — After all candidates are scored, analyses the distribution of scores across the candidate pool. Flags if any unusual patterns are detected that might indicate systematic bias.

---

### 8. Database — `models/database.py`

Uses SQLite (a file-based database, no setup required) to persistently store score overrides made by HR reviewers. Every time a reviewer changes a score in the Override tab, the change is recorded with: the candidate name, the dimension changed, the original score, the new score, the reason given, and the reviewer's name and timestamp.

This creates an audit trail — a record of every human intervention in the process.

---

### 9. LangGraph Pipeline — `agents/graph.py`

LangGraph is the orchestration layer. It defines the agents as nodes in a state machine and manages the flow of data between them.

**State:** A single dictionary (`PipelineState`) is passed through every agent. Each agent receives the full current state, adds or modifies its relevant fields, and returns the updated state. This means every agent has access to everything that happened before it.

**Conditional routing:** The connection between the Critic and the Scorer is not a one-way edge. It is a conditional edge that checks: does the Critic want any candidates re-scored? If yes, route back to Scorer. If no, proceed to Report. This is what creates the re-scoring loop.

---

### 10. Streamlit Dashboard — `app.py`

The web interface. Five tabs:

| Tab | Purpose |
|---|---|
| Job Description | Paste or upload the JD |
| Upload Resumes | Upload one or more PDF/DOCX files, or paste LinkedIn JSON |
| Results and Charts | Ranking bar chart, radar chart, heatmap, gauge charts, full score tables |
| Skills Gap | Per-candidate matched/missing/bonus skills with learning resources |
| Overrides | Human-in-the-loop score adjustments with audit logging |

**Charts:**
- **Radar chart** — Overlays all candidates on a pentagon shape, one axis per dimension. Makes it immediately clear who is strong in which area.
- **Ranking bar chart** — Horizontal bars sorted by total score, colour-coded green/amber/red by recommendation.
- **Heatmap** — A grid of candidates vs dimensions, colour-coded so weaknesses are immediately visible.
- **Gauge charts** — One per candidate, showing their total score on a 0-100 dial.

---

## Data Flow — End to End

```
User pastes JD text
    -> Stored in st.session_state.jd_text

User uploads resume PDFs
    -> detect_and_parse_bytes() extracts plain text from each PDF
    -> Stored as list of {filename, content, source}

User clicks Run Analysis
    -> run_pipeline(jd_text, resumes) is called
    -> LangGraph starts the state machine

[Parser Node]
    -> parse_jd(jd_text) calls Llama 3, returns JobDescription object
    -> For each resume: parse_resume(text) calls Llama 3, returns CandidateProfile object
    -> LinkedIn/GitHub URLs extracted via LLM + regex fallback
    -> Failed candidates are skipped with a warning
    -> State now has: jd_parsed, candidates[]

[Scorer Node]
    -> For each candidate: score_candidate(jd, candidate) calls Llama 3
    -> Returns 5 DimensionScore objects + weighted_total + recommendation
    -> State now has: evaluations[]

[Critic Node] (if enabled)
    -> For each evaluation: critique_evaluation() calls Llama 3
    -> If confidence < 0.6: adds candidate to retry_candidates[]
    -> LangGraph checks: should_rescore()
    -> If retry_candidates not empty: routes back to Scorer

[Report Node]
    -> For each evaluation: analyse_skills_gap() computes matched/missing/bonus
    -> Sorts evaluations by weighted_total descending
    -> Assigns rank numbers
    -> State now has: complete final results

Dashboard renders charts and tables from final state
```

---

## Configuration

All configuration is in the `.env` file:

```
OLLAMA_BASE_URL=http://localhost:11434   # Where Ollama is running
OLLAMA_MODEL=llama3.2:3b                # Which model to use
```

The model can also be changed live from the sidebar in the dashboard without restarting.

---

## Model Recommendations for RTX 3050 (4GB VRAM)

| Model | VRAM | Speed | Quality |
|---|---|---|---|
| llama3.2:1b | 1 GB | 30-60 sec per candidate | Lower |
| llama3.2:3b | 2 GB | 1-2 min per candidate | Good |
| phi3:mini | 2.3 GB | 45-90 sec per candidate | Good |
| llama3 | 4.5 GB | May overflow VRAM | Best |

The recommended model is `llama3.2:3b`. It fits comfortably in 4GB VRAM, produces reliable JSON output, and processes each candidate in 1-2 minutes on the GPU.

---

## How to Run

1. Start Ollama: open a terminal and run `ollama serve`
2. Pull a model: `ollama pull llama3.2:3b`
3. Start the app: `streamlit run app.py` from the project directory
4. Open `http://localhost:8501` in your browser

---

## Project File Structure

```
hr, resume shortlister/
    app.py                      Main Streamlit dashboard
    requirements.txt            Python dependencies
    .env.example                Configuration template
    README.md                   Quick start guide
    PROJECT_EXPLANATION.md      This file

    agents/
        graph.py                LangGraph pipeline orchestrator
        parser_agent.py         JD and resume parsing node
        scorer_agent.py         5-dimension scoring node
        critic_agent.py         Validation and bias checking node
        report_agent.py         Skills gap and ranking node

    models/
        schemas.py              Pydantic data models
        database.py             SQLite override storage

    services/
        file_parser.py          PDF and DOCX text extraction
        linkedin_parser.py      LinkedIn JSON to CandidateProfile
        skills_gap.py           Skills matching and gap analysis
        bias_detector.py        PII anonymisation and fairness audit
        report_generator.py     PDF, HTML, JSON report generation

    prompts/
        jd_parser.txt           System prompt for JD parsing
        resume_parser.txt       System prompt for resume parsing
        scorer.txt              System prompt for scoring
        guardrails.txt          Safety and constraint instructions

    templates/
        report.html             Jinja2 template for HTML reports

    output/                     Generated reports saved here
    tests/                      Unit tests
```

---

## Security Considerations

- All data stays on the local machine. Nothing is sent to external APIs.
- Resume text is anonymised before scoring to prevent name/demographic bias.
- Input sanitisation removes prompt injection attempts before text is sent to the LLM.
- Override actions are logged immutably in SQLite.
- The SECURITY.md file in the project root documents the full security model.

---

## Known Limitations

1. **Scanned PDFs will not work.** The PDF must contain machine-readable text. Image-only scans return empty text.
2. **LLM output is non-deterministic.** The same resume run twice may produce slightly different scores. Temperature is set to 0 to minimise this, but it cannot be fully eliminated.
3. **Small VRAM may cause overflow.** If the model partially runs on CPU, inference time increases significantly.
4. **Skills gap is keyword-based.** Even with fuzzy matching, the system may miss skills that are described in unusual ways in the resume.
