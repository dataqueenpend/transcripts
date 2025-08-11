# Analiza rozmów i zbijania obiekcji klientów

Repo zawiera prototyp do wykrywania rozmów sprzedażowych, obiekcji klientów oraz sposobów zbijania obiekcji w transkryptach CSV.

## Uruchomienie
1. Zainstaluj zależności: `pip install -r requirements.txt`
2. Otwórz notebook: `notebooks/objections_analysis.ipynb`
3. Plik danych: `/workspace/transcripts_combined - sample2.csv` (lub `transcripts_combined_sample2.csv` jeśli dostępny).

## Wyniki
- Wyniki analizy zapisywane są do `outputs/objections_analysis_sample2.csv` oraz metryk w `outputs/metrics.json`.

## Ewaluacja i porównywalność
- Etykiety referencyjne (opcjonalnie): `data/labels.jsonl` (format JSON Lines).
- Metryki: Precision/Recall/F1 dla detekcji obiekcji i kategorii odpowiedzi.
- Porównania między modelami: stały interfejs `ObjectionClassifier` / `RebuttalClassifier`, wersjonowanie `model_version`, `config_hash`, `data_version` w artefaktach.