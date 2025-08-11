### Slide 1 — Cel i schemat rozwiązania
- **Pytanie biznesowe**: Jak doradcy radzą sobie z obiekcjami klientów w rozmowach sprzedażowych?
- **Dane**: CSV z transkrypcjami (zredagowane PII). Kolumny m.in. `filename`, `full_text`, `confidence`, `audio_duration_seconds`, `word_count`.
- **Schemat**:
  - Detekcja rozmów sprzedażowych: reguły słów kluczowych (+ opcjonalny klasyczny ML TF-IDF).
  - Detekcja obiekcji: segmentacja na zdania + reguły lingwistyczne (negacje, odmowy, „za drogo”, „nie potrzebuję”…) + opcjonalny klasyfikator.
  - Zbijanie obiekcji: powiązanie zdań doradcy następujących po obiekcji i kategoryzacja odpowiedzi (korzyści, rekompozycja kosztu, alternatywa, doprecyzowanie, social proof, zamknięcie warunkowe).
  - Wyniki: CSV z listą obiekcji i odpowiedzi, dashboard/metryki.

### Slide 2 — Optymalizacja i metryki jakości
- **Efektywność**: reguły i TF-IDF ograniczają liczbę zapytań do LLM; ewentualne LLM tylko dla trudnych przypadków (active learning / fallback).
- **Metryki**:
  - Jakość detekcji (Precision/Recall/F1) na próbce ręcznie oznakowanej (JSONL).
  - Jakość kategoryzacji obiekcji/odpowiedzi (macro/micro F1 + confusion matrix).
  - Stabilność: statystyki dystrybucji kategorii między wersjami (drifty). 
- **Porównywalność między modelami**:
  - Zapis konfiguracji: `model_version`, `config_hash`, `data_version` w artefaktach CSV/JSON.
  - Identyczny kontrakt klas `ObjectionClassifier`/`RebuttalClassifier`, plug-and-play.

### Slide 3 — Reużywalność i rozszerzenia
- **Reużywalny pipeline**: te same kroki dla innych zadań (powody odmów, jak formułują zapytania sprzedażowe, itp.).
- **Łatwe rozszerzenia**:
  - Nowe kategorie → dodanie reguł + trenowalny klasyfikator.
  - Inne domeny → podmiana list słów kluczowych i przykładów.
  - LLM w chmurze → tylko na zanonimizowanych danych, batched, z cache.
- **Artyfakty**: notebook (EDA+pipeline), `outputs/*.csv`, etykiety `data/labels.jsonl`, `requirements.txt`. 