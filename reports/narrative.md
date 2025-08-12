# Narracja realizacji zadania

- Co się udało: Zbudowano prototyp pipeline’u wykrywania rozmów sprzedażowych, obiekcji i taktyk reakcji. Wyniki eksportowane do CSV, wygenerowano próbkę do adnotacji.
- Trudności: Brak jawnych ról mówców → heurystyki; brak etykiet → potrzeba lekkiej adnotacji; minimalizacja kosztu modeli → embeddingi zamiast LLM.
- Iteracje: Start od reguł, następnie dodanie embeddingów do rozstrzygania taktyk; wstępne strojenie list wzorców.
- Błędy i poprawki: Doprecyzowanie wzorców językowych (różne formy), ograniczenie okna reakcji; weryfikacja poprawności segmentacji zdań.
- Plan vs. rzeczywistość: Plan 6h: 1h eksploracja, 2h prototyp pipeline, 1h ewaluacja/harness, 1h raporty, 1h slajdy; w praktyce więcej czasu na heurystyki mówców i dopasowanie wzorców.