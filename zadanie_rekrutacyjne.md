## Zadanie rekrutacyjne — Zespół Text Analytics & AI
Zbuduj modułowy pipeline analityczny do wykrywania obiekcji i technik ich zbijania w rozmowach (call transcripts). Pipeline ma być reużywalny dla różnych domen (bankowość, ubezpieczenia, sprzedaż).

### Zakres
- Ingestion danych (CSV/Parquet) i standaryzacja pól.
- Segmentacja tekstu na zdania i przypisanie ról mówców (heurystyki + opcjonalnie LLM).
- Detekcja obiekcji oraz identyfikacja zbijania (techniki) z możliwością łatwej konfiguracji słowników i reguł.
- Ocena wpływu na klienta (akceptacja/opór) w krótkim oknie oraz metryki efektywności per technika.
- Generowanie podsumowań rozmów (strategia: ekstrakcyjna lub LLM — adapter).
- Raport końcowy w Markdown (metodologia, metryki, przykłady).

### Wymagania niefunkcjonalne
- Modułowość (interfejsy: `Segmenter`, `EventDetector`, `ImpactScorer`, `Summarizer`).
- Konfigurowalność (YAML/JSON dla słowników i progów).
- Testowalność (unit/integration), deterministyczny tryb offline.
- Skalowalność (batch processing, streaming opcjonalnie).

### Dane wejściowe
- Przykładowy plik CSV z kolumnami: `filename`, `full_text`, `confidence`, `audio_duration_seconds`, `word_count`, `redacted_pii_policies`.

### Kryteria akceptacji
- >80% trafności w klasyfikacji ‘rozmowa sprzedażowa’ na danych validacyjnych (heurystyka lub model).
- Wykrycie ≥70% obiekcji w zestawie adnotowanym (może być stworzony przez kandydata).
- Raport efektywności technik: tabela z liczbą zdarzeń i wskaźnikami akceptacji/oporu.
- Dwa tryby podsumowań: ekstrakcyjny (offline) i LLM (mock lub adapter).

### Co dostarczyć
- Repozytorium z kodem (Python), `README.md`, przykładowe konfiguracje, testy.
- Skrypt CLI do uruchomienia pipeline'u i wygenerowania raportu.
- Krótkie omówienie kompromisów projektowych i planu rozwoju.

