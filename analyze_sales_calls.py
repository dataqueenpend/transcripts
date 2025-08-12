import csv
import os
import re
from collections import Counter, defaultdict
from dataclasses import dataclass, field
from typing import List, Tuple, Dict, Optional
import json
import urllib.request
import urllib.error


@dataclass
class ObjectionEvent:
    objection_sentence: str
    rebuttal_sentences: List[str]
    technique: str
    client_followup_sentences: List[str]
    client_reaction: str  # 'accept', 'resist', 'neutral'


@dataclass
class ConversationAnalysis:
    filename: str
    is_sales_like: bool
    sales_score: int
    objections: List[ObjectionEvent] = field(default_factory=list)
    summary: str = ""


SALES_KEYWORDS = [
    # product/benefit words
    "medicare", "benefit", "benefits", "plan", "coverage", "coverages",
    "dental", "vision", "hearing", "transportation", "prescription",
    "medications", "co pay", "copay", "premium", "paperwork",
    # value/price words
    "no additional cost", "no cost", "save", "money back", "lower premiums",
    # process/authority words
    "eligible", "qualified", "upgrade", "upgraded", "updated",
    "connect", "connecting", "specialist", "licensed", "agent", "advisor",
]

AGENT_LEXICON = [
    "my name is", "we can", "we will", "i will", "let you know", "benefits include",
    "you will", "you are qualified", "you are eligible", "no additional cost",
    "i'm connecting", "i will connect", "please stay online", "paperwork",
    "licensed", "agent", "advisor", "specialist",
]

OBJECTION_PATTERNS = [
    r"\bi don't\b", r"\bi do not\b", r"\bi can't\b", r"\bi cannot\b",
    r"\bi won't\b", r"\bnot interested\b", r"\bno\b", r"\bi thought\b",
    r"\bi'm not\b", r"\bwhen\b", r"\bwhy\b", r"\bhow\b", r"\bwhat\b",
    r"\bdon't understand\b", r"\bnot sure\b", r"\bconfused\b",
]

TECHNIQUE_RULES: List[Tuple[str, List[str]]] = [
    ("assurance/cost removal", ["no additional cost", "no co pay", "no copay", "no payment"]),
    ("feature/benefit reframing", ["benefit", "benefits include", "dental", "vision", "hearing", "transportation", "prescription", "medications"]),
    ("authority/qualification", ["qualified", "eligible", "congrats", "big congrats"]),
    ("transfer to expert/authority", ["connect", "connecting", "licensed", "specialist", "advisor", "agent"]),
    ("minimize effort/foot-in-the-door", ["just", "quick", "minute", "moments", "stay online"]),
    ("value proposition", ["save", "money back", "lower premiums", "reduce", "help you"]),
    ("reassurance/empathy", ["don't worry", "appreciate your time", "glad to hear", "good to know"]),
]

ACCEPTANCE_TOKENS = ["okay", "ok", "right", "yes", "correct", "alright", "fine", "yeah"]
RESISTANCE_TOKENS = ["no", "not", "don't", "cannot", "can't", "busy", "stop", "hang up"]

STOPWORDS = set(
    "a an the and or but to of in on for with at from by as is are was were be been being have has had do does did not no yes i you he she it we they my your his her our their this that these those there here then than so such if because just very can will would could should may might must".split()
)


def normalize_text(text: str) -> str:
    return re.sub(r"\s+", " ", text).strip()


def sentence_split(text: str) -> List[str]:
    # Simple sentence splitter respecting ., ?, !
    # Keep punctuation at the end
    text = normalize_text(text)
    sentences = re.split(r"(?<=[.!?])\s+", text)
    return [s.strip() for s in sentences if s.strip()]


def contains_any(text: str, terms: List[str]) -> bool:
    lower = text.lower()
    return any(term in lower for term in terms)


def regex_any(text: str, patterns: List[str]) -> bool:
    return any(re.search(p, text, flags=re.IGNORECASE) for p in patterns)


def sales_score(text: str) -> int:
    lower = text.lower()
    return sum(1 for term in SALES_KEYWORDS if term in lower)


def looks_like_agent(text: str) -> bool:
    lower = text.lower()
    return contains_any(lower, AGENT_LEXICON)


def classify_technique(text: str) -> str:
    lower = text.lower()
    for name, markers in TECHNIQUE_RULES:
        if any(m in lower for m in markers):
            return name
    # Fallback
    return "generic rebuttal"


def assess_client_reaction(sentences: List[str]) -> str:
    # Score acceptance vs resistance from provided sentences window
    acceptance = 0
    resistance = 0
    for s in sentences:
        lower = s.lower()
        acceptance += sum(1 for t in ACCEPTANCE_TOKENS if re.search(rf"\\b{re.escape(t)}\\b", lower))
        resistance += sum(1 for t in RESISTANCE_TOKENS if re.search(rf"\\b{re.escape(t)}\\b", lower))
    if acceptance > resistance and acceptance > 0:
        return "accept"
    if resistance > acceptance and resistance > 0:
        return "resist"
    return "neutral"


def extractive_summary(text: str, max_sentences: int = 3) -> str:
    # Simple frequency-based extractive summarization
    sents = sentence_split(text)
    if not sents:
        return ""
    # Build term frequencies excluding stopwords
    word_freq: Counter = Counter()
    for s in sents:
        words = re.findall(r"[A-Za-z']+", s.lower())
        for w in words:
            if w in STOPWORDS:
                continue
            word_freq[w] += 1
    if not word_freq:
        # return first sentences
        return " ".join(sents[:max_sentences])
    max_freq = max(word_freq.values())
    # Normalize frequencies
    for w in list(word_freq.keys()):
        word_freq[w] = word_freq[w] / max_freq
    # Score sentences
    sent_scores: List[Tuple[float, int, str]] = []  # (score, index, sentence)
    for idx, s in enumerate(sents):
        words = re.findall(r"[A-Za-z']+", s.lower())
        score = sum(word_freq.get(w, 0.0) for w in words)
        sent_scores.append((score, idx, s))
    # Pick top-N by score, but keep original order
    top = sorted(sent_scores, key=lambda x: x[0], reverse=True)[:max_sentences]
    top_sorted = [s for _, _, s in sorted(top, key=lambda x: x[1])]
    return " ".join(top_sorted)


def detect_objections_and_rebuttals(sentences: List[str]) -> List[ObjectionEvent]:
    events: List[ObjectionEvent] = []
    n = len(sentences)
    for i, s in enumerate(sentences):
        if regex_any(s, OBJECTION_PATTERNS):
            # Candidate objection
            # Look ahead for agent rebuttal within next 3 sentences
            rebuttals: List[str] = []
            technique: Optional[str] = None
            for j in range(i + 1, min(i + 4, n)):
                if looks_like_agent(sentences[j]) or contains_any(sentences[j].lower(), ["benefit", "you will", "we will", "no additional cost", "qualified", "eligible", "connect"]):
                    rebuttals.append(sentences[j])
                    if technique is None:
                        technique = classify_technique(sentences[j])
                # Stop if we collected two rebuttal sentences
                if len(rebuttals) >= 2:
                    break
            if not rebuttals:
                continue
            # Look further for client's immediate reaction (next 1-2 sentences after rebuttal block)
            k_start = i + 1 + len(rebuttals)
            followup = sentences[k_start:k_start + 2]
            reaction = assess_client_reaction(followup)
            events.append(ObjectionEvent(
                objection_sentence=sentences[i],
                rebuttal_sentences=rebuttals,
                technique=technique or "generic rebuttal",
                client_followup_sentences=followup,
                client_reaction=reaction,
            ))
    return events


def analyze_conversation(filename: str, text: str) -> ConversationAnalysis:
    text_norm = normalize_text(text)
    sents = sentence_split(text_norm)
    s_score = sales_score(text_norm)
    is_sales = s_score >= 3 and ("medicare" in text_norm.lower() or "benefit" in text_norm.lower() or "plan" in text_norm.lower())
    events = detect_objections_and_rebuttals(sents) if is_sales else []
    # Build focus text for summary: objection+rebuttal snippets if any, otherwise general summary
    focus_parts: List[str] = []
    for ev in events[:3]:
        focus_parts.append(ev.objection_sentence)
        focus_parts.extend(ev.rebuttal_sentences)
    focus_text = " ".join(focus_parts) if focus_parts else text_norm
    summ = extractive_summary(focus_text, max_sentences=3)
    return ConversationAnalysis(
        filename=filename,
        is_sales_like=is_sales,
        sales_score=s_score,
        objections=events,
        summary=summ,
    )


def load_csv(path: str) -> List[Dict[str, str]]:
    rows = []
    # utf-8-sig to strip BOM from first field name
    with open(path, newline='', encoding='utf-8-sig') as f:
        reader = csv.DictReader(f)
        for row in reader:
            rows.append(row)
    return rows


def generate_markdown_reports(conversations: List[ConversationAnalysis], output_analysis: str, output_task: str) -> None:
    # Aggregate stats
    sales_convos = [c for c in conversations if c.is_sales_like]
    num_sales = len(sales_convos)
    total = len(conversations)

    technique_counter: Counter = Counter()
    technique_accept: Counter = Counter()
    technique_resist: Counter = Counter()
    total_objections = 0

    for conv in sales_convos:
        for ev in conv.objections:
            total_objections += 1
            technique_counter[ev.technique] += 1
            if ev.client_reaction == "accept":
                technique_accept[ev.technique] += 1
            elif ev.client_reaction == "resist":
                technique_resist[ev.technique] += 1

    def fmt_rate(numer: int, denom: int) -> str:
        if denom == 0:
            return "0.0%"
        return f"{(100.0 * numer / denom):.1f}%"

    # Build analysis markdown
    lines: List[str] = []
    lines.append("## Podsumowanie analizy rozmów\n")
    lines.append(f"- **Liczba wszystkich transkryptów**: {total}")
    lines.append(f"- **Rozmowy sprzedażowe (heurystyka słów-kluczy)**: {num_sales}")
    lines.append(f"- **Liczba wykrytych obiekcji**: {total_objections}\n")

    lines.append("### Techniki zbijania obiekcji i ich skuteczność")
    if technique_counter:
        lines.append("")
        for tech, cnt in technique_counter.most_common():
            acc = technique_accept[tech]
            res = technique_resist[tech]
            lines.append(f"- **{tech}**: {cnt} zdarzeń; akceptacja: {fmt_rate(acc, cnt)}; opór: {fmt_rate(res, cnt)}")
        lines.append("")
    else:
        lines.append("- Brak wykrytych technik.")

    lines.append("### Metodologia")
    lines.append("- **Identyfikacja rozmów sprzedażowych**: reguły oparte na słowach-kluczach (np. ‘Medicare’, ‘benefits’, ‘plan’, ‘no additional cost’, ‘eligible’, ‘connect’). Próg >= 3 dopasowań + obecność rdzeniowych słów (benefit/plan/Medicare).")
    lines.append("- **Segmentacja**: proste dzielenie na zdania po znakach `.?!`.")
    lines.append("- **Obiekcje**: wykrywane wzorcami (np. ‘I don't’, ‘not interested’, pytania ‘why/when/how/what’, negacje i niepewność).")
    lines.append("- **Zbijanie obiekcji (rebuttal)**: zdania następujące po obiekcji, zawierające leksykon doradcy (np. ‘no additional cost’, ‘you are eligible’, ‘I will connect...’). Przypisywanie techniki na podstawie słów-wyzwalaczy.")
    lines.append("- **Wpływ na klienta**: natychmiastowa reakcja w kolejnych 1–2 zdaniach (tokeny akceptacji: ‘okay/yes/right’, oporu: ‘no/not/busy’).")
    lines.append("- **Podsumowania**: ekstrakcyjny skrót zdań (rankowanie częstotliwościowe bez zależności od zewnętrznego LLM).\n")

    lines.append("### Ograniczenia i możliwe ulepszenia")
    lines.append("- **Brak oznaczeń ról mówców** utrudnia precyzyjne przypisanie, kto zgłasza obiekcje. Można poprawić przez diarization ASR lub LLM, które klasyfikuje role zdań.")
    lines.append("- **Reguły słów-kluczowych** są szybkie, ale kruche. Warto dodać klasyfikator uczenia maszynowego (np. BERT/LoRA) albo promptowany LLM do etykietowania: ‘sprzedaż’, ‘obiekcja’, ‘rebuttal’, ‘reakcja klienta’.")
    lines.append("- **Ocena wpływu** jest bardzo lokalna (1–2 zdania). Można modelować sekwencję stanów (‘opór’→‘akceptacja’) i mierzyć ‘czas do akceptacji’.\n")

    lines.append("### Architektura modułowa dla wielu business case’ów")
    lines.append("- **Ingestion**: moduł loaderów (CSV/Parquet/DB) + normalizacja pól (tekst, czas, meta).")
    lines.append("- **Segmentacja i role**: interfejs `Segmenter` (regułowy/LLM) i `SpeakerAttribution` (heurystyki, VAD/diarization, LLM).")
    lines.append("- **Detekcja eventów**: `EventDetector` z pluginami: ‘objection’, ‘rebuttal’, ‘question’, ‘compliance’. Każdy plugin ma własne reguły/LLM/klasyfikator.")
    lines.append("- **Taksonomia technik**: konfigurowalny słownik → etykiety. Można trzymać w YAML i wersjonować.")
    lines.append("- **Ewaluacja wpływu**: `ImpactScorer` (akceptacja/opór, sentiment, przejście stanu, eskalacja).")
    lines.append("- **Summarizer**: strategia `Extractive` lub `LLM` z adapterem (OpenAI/Azure/Ollama) + retry/batching.")
    lines.append("- **Orkiestracja**: pipeline w stylu scikit-learn/Prefect (fit/transform) + artefakty (JSON/Parquet) + raporty Markdown/HTML.\n")

    lines.append("### Przykładowe rozmowy i skróty (wybrane)")
    for conv in sales_convos[:10]:
        lines.append(f"- **{conv.filename}** — obiekcje: {len(conv.objections)} — skrót: {conv.summary}")

    analysis_md = "\n".join(lines) + "\n"
    with open(output_analysis, "w", encoding="utf-8") as f:
        f.write(analysis_md)

    # Recruitment task markdown
    task_lines: List[str] = []
    task_lines.append("## Zadanie rekrutacyjne — Zespół Text Analytics & AI")
    task_lines.append("Zbuduj modułowy pipeline analityczny do wykrywania obiekcji i technik ich zbijania w rozmowach (call transcripts). Pipeline ma być reużywalny dla różnych domen (bankowość, ubezpieczenia, sprzedaż).\n")
    task_lines.append("### Zakres")
    task_lines.append("- Ingestion danych (CSV/Parquet) i standaryzacja pól.")
    task_lines.append("- Segmentacja tekstu na zdania i przypisanie ról mówców (heurystyki + opcjonalnie LLM).")
    task_lines.append("- Detekcja obiekcji oraz identyfikacja zbijania (techniki) z możliwością łatwej konfiguracji słowników i reguł.")
    task_lines.append("- Ocena wpływu na klienta (akceptacja/opór) w krótkim oknie oraz metryki efektywności per technika.")
    task_lines.append("- Generowanie podsumowań rozmów (strategia: ekstrakcyjna lub LLM — adapter).")
    task_lines.append("- Raport końcowy w Markdown (metodologia, metryki, przykłady).\n")

    task_lines.append("### Wymagania niefunkcjonalne")
    task_lines.append("- Modułowość (interfejsy: `Segmenter`, `EventDetector`, `ImpactScorer`, `Summarizer`).")
    task_lines.append("- Konfigurowalność (YAML/JSON dla słowników i progów).")
    task_lines.append("- Testowalność (unit/integration), deterministyczny tryb offline.")
    task_lines.append("- Skalowalność (batch processing, streaming opcjonalnie).\n")

    task_lines.append("### Dane wejściowe")
    task_lines.append("- Przykładowy plik CSV z kolumnami: `filename`, `full_text`, `confidence`, `audio_duration_seconds`, `word_count`, `redacted_pii_policies`.\n")

    task_lines.append("### Kryteria akceptacji")
    task_lines.append("- >80% trafności w klasyfikacji ‘rozmowa sprzedażowa’ na danych validacyjnych (heurystyka lub model).")
    task_lines.append("- Wykrycie ≥70% obiekcji w zestawie adnotowanym (może być stworzony przez kandydata).")
    task_lines.append("- Raport efektywności technik: tabela z liczbą zdarzeń i wskaźnikami akceptacji/oporu.")
    task_lines.append("- Dwa tryby podsumowań: ekstrakcyjny (offline) i LLM (mock lub adapter).\n")

    task_lines.append("### Co dostarczyć")
    task_lines.append("- Repozytorium z kodem (Python), `README.md`, przykładowe konfiguracje, testy.")
    task_lines.append("- Skrypt CLI do uruchomienia pipeline'u i wygenerowania raportu.")
    task_lines.append("- Krótkie omówienie kompromisów projektowych i planu rozwoju.\n")

    task_md = "\n".join(task_lines) + "\n"
    with open(output_task, "w", encoding="utf-8") as f:
        f.write(task_md)


def try_llm_summaries(conversations: List[ConversationAnalysis], output_analysis: str, max_items: int = 10) -> None:
    """
    Generate LLM-based summaries for top sales conversations if an API is configured.
    Adapter supports:
    - offline:extractive (default)
    - openai:<model> via Chat Completions API (requires OPENAI_API_KEY)
    Environment:
      SUMMARIZER_MODEL, OPENAI_API_KEY, OPENAI_BASE_URL
    """
    model = os.environ.get("SUMMARIZER_MODEL", "offline:extractive")
    sales = [c for c in conversations if c.is_sales_like][:max_items]

    section_lines: List[str] = []
    section_lines.append("### Podsumowania LLM")

    if model.startswith("offline:") or not sales:
        # Fallback to existing extractive summaries
        if not sales:
            section_lines.append("- Brak rozmów sprzedażowych do podsumowania.")
        else:
            for conv in sales:
                section_lines.append(f"- **{conv.filename}** — {conv.summary}")
        # Note about configuration
        section_lines.append("\nUwaga: aby włączyć LLM, ustaw `SUMMARIZER_MODEL=openai:gpt-4o-mini` oraz `OPENAI_API_KEY`.\n")
    elif model.startswith("openai:"):
        api_key = os.environ.get("OPENAI_API_KEY")
        if not api_key:
            section_lines.append("- Brak `OPENAI_API_KEY`. Użyto fallbacku offline.")
            for conv in sales:
                section_lines.append(f"- **{conv.filename}** — {conv.summary}")
        else:
            base_url = os.environ.get("OPENAI_BASE_URL", "https://api.openai.com/v1")
            chat_url = base_url.rstrip('/') + "/chat/completions"
            model_name = model.split(":", 1)[1] or "gpt-4o-mini"
            headers = {
                "Authorization": f"Bearer {api_key}",
                "Content-Type": "application/json",
            }
            section_lines.append(f"Model: `{model_name}`\n")
            for conv in sales:
                prompt = (
                    "You are an analyst for call center sales conversations. Summarize in 2-3 sentences: "
                    "the offer, key objections, advisor's rebuttal techniques, and client's reaction. "
                    "Be concise, factual, and avoid PII.\n\nConversation snippet:\n" + conv.summary
                )
                body = {
                    "model": model_name,
                    "messages": [
                        {"role": "system", "content": "You are a concise, factual enterprise analyst."},
                        {"role": "user", "content": prompt},
                    ],
                    "temperature": 0.2,
                    "max_tokens": 160,
                }
                try:
                    req = urllib.request.Request(chat_url, data=json.dumps(body).encode("utf-8"), headers=headers)
                    with urllib.request.urlopen(req, timeout=30) as resp:
                        data = json.loads(resp.read().decode("utf-8"))
                        content = data.get("choices", [{}])[0].get("message", {}).get("content", "")
                except Exception as e:
                    content = f"[LLM error: {e}] Fallback: {conv.summary}"
                section_lines.append(f"- **{conv.filename}** — {content}")
    else:
        # Unknown adapter -> fallback
        section_lines.append(f"- Nieznany adapter `{model}`. Użyto fallbacku offline.")
        for conv in sales:
            section_lines.append(f"- **{conv.filename}** — {conv.summary}")

    # Append to analysis file
    with open(output_analysis, "a", encoding="utf-8") as f:
        f.write("\n" + "\n".join(section_lines) + "\n")


def main():
    input_csv = os.environ.get("INPUT_CSV", "/workspace/transcripts_combined.csv")
    output_analysis = os.environ.get("OUTPUT_ANALYSIS", "/workspace/analiza.md")
    output_task = os.environ.get("OUTPUT_TASK", "/workspace/zadanie_rekrutacyjne.md")

    rows = load_csv(input_csv)
    conversations: List[ConversationAnalysis] = []
    for row in rows:
        filename = row.get("filename") or row.get("\ufefffilename") or ""
        text = row.get("full_text", "")
        conversations.append(analyze_conversation(filename, text))

    generate_markdown_reports(conversations, output_analysis, output_task)
    # LLM summaries appended
    try_llm_summaries(conversations, output_analysis)
    print(f"Wygenerowano: {output_analysis} oraz {output_task}")


if __name__ == "__main__":
    main()