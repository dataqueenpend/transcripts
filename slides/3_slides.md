---
## 1) Schemat działania i optymalizacja procesu
- Wejście: CSV z transkryptami (anonimowe)
- Pipeline:
  1. Heurystyki rozmów sprzedażowych (regex/jargon)
  2. Segmentacja na zdania + heurystyki mówiącego
  3. Detekcja obiekcji (reguły)
  4. Reakcje doradców (okno po obiekcji) → klasyfikacja taktyk (reguły + embeddingi)
  5. Raporty (tabele, crosstab)
- Optymalizacja kosztu: minimalizacja wywołań LLM, batchowe embeddingi, cache, sampling

---
## 2) Ewaluacja i porównanie modeli
- Metryki: Precision/Recall/F1 dla: is_sales, is_objection, tactic
- Zbiór walidacyjny: próbka losowa zdarzeń do adnotacji (CSV)
- Harness: stały kontrakt funkcji `categorize_response()` → zamienialny model (LLM/klasyczny)
- Porównanie A/B: zapisz predykcje i metryki per model/wersja, wykresy trendów

---
## 3) Reużywalność rozwiązania
- Konfiguracja reguł (patterns) i taksonomii taktyk w jednym miejscu
- Możliwość zastosowania do: powodów odmów, formułowania ofert, identyfikacji pytań sprzedażowych
- Artefakty: notebook, raporty CSV, paczka z regułami → szybka adaptacja do nowych domen