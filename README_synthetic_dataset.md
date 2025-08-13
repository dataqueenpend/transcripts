# Syntetyczny Dataset Rozmów Santander

## Opis

Ten dataset zawiera 4000 syntetycznie wygenerowanych rozmów na infolinii banku Santander. Rozmowy zostały stworzone w sposób realistyczny, naśladując naturalne konwersacje między doradcami bankowymi a klientami.

## Struktura Datasetu

Dataset został zapisany w pliku `synthetic_conversations_dataset.csv` i zawiera następujące kolumny:

- **id** - unikalny identyfikator rozmowy
- **filename** - nazwa pliku z transkrypcją
- **full_text** - pełny tekst rozmowy
- **confidence** - poziom pewności transkrypcji (0.85-0.99)
- **audio_duration_seconds** - czas trwania rozmowy w sekundach (60-600)
- **word_count** - liczba słów w rozmowie
- **is_sales** - czy jest to rozmowa sprzedażowa (True/False)
- **is_rebuttal** - czy doradca zbija obiekcje klienta (True/False)
- **is_obj** - czy klient zgłasza obiekcje (True/False)
- **redacted_pii_policies** - polityki anonimizacji danych osobowych

## Typy Rozmów

Dataset zawiera różnorodne typy rozmów:

1. **Rozmowy sprzedażowe** (19.8%) - dotyczące produktów bankowych:
   - Santander Premium Card (karta kredytowa)
   - Santander FlexiLimit (limit kredytowy)
   - Santander Protect Plus (ubezpieczenie)

2. **Rozmowy serwisowe** - informacje o aktualizacjach kont, kart, aplikacji

3. **Rozmowy informacyjne** - zmiany w regulaminie, zasady bezpieczeństwa

4. **Rozmowy reklamacyjne** - rozwiązywanie problemów klientów

5. **Rozmowy konsultacyjne** - doradztwo finansowe

## Charakterystyka Rozmów

### Struktura Standardowa
Każda rozmowa zawiera standardowe elementy:
1. Przedstawienie się doradcy
2. Informacja o nagrywaniu rozmowy
3. Wyjaśnienie celu rozmowy
4. Rozwinięcie tematu
5. Zakończenie

### Różnorodność Scenariuszy
- Klienci zainteresowani ofertą
- Klienci sceptyczni
- Klienci z obiekcjami (15.0%)
- Rozmowy ze zbić obiekcji (3.8%)
- Smalltalk i przypadki graniczne

### Produkty Bankowe
- **Santander Premium Card** - karta kredytowa z atrakcyjnym oprocentowaniem
- **Santander FlexiLimit** - elastyczny limit kredytowy
- **Santander Protect Plus** - kompleksowe ubezpieczenie

## Statystyki Datasetu

- **Łączna liczba rozmów**: 4000
- **Rozmowy sprzedażowe**: 793 (19.8%)
- **Rozmowy ze zbić obiekcji**: 152 (3.8%)
- **Rozmowy z obiekcjami**: 602 (15.0%)
- **Średni czas rozmowy**: 1-10 minut
- **Zakres słów**: 30-150 słów na rozmowę

## Anonimizacja

Wszystkie rozmowy są zanonimizowane zgodnie z politykami RODO:
- Imiona zastąpione przez [PERSON_NAME]
- Numery telefonów przez [PHONE_NUMBER]
- Daty urodzenia przez [DATE_OF_BIRTH]
- Lokalizacje przez [LOCATION]
- Kwoty przez [MONEY_AMOUNT]

## Użycie

Dataset może być wykorzystywany do:
- Trenowania modeli NLP do analizy rozmów
- Analizy skuteczności sprzedaży
- Badania technik zbić obiekcji
- Analizy sentymentu klientów
- Testowania systemów automatycznej transkrypcji

## Autor

Dataset został wygenerowany przez Mateusza w ramach zadania dotyczącego tworzenia syntetycznych danych rozmów na infolinii bankowej.

## Data Utworzenia

Dataset został wygenerowany w dniu: 13.08.2025