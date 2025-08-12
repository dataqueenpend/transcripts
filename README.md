# Analiza Rozmów Sprzedażowych - Prototyp

## 📋 Opis projektu

Prototypowe rozwiązanie do analizy transkrypcji rozmów telefonicznych w celu identyfikacji:
- Rozmów sprzedażowych
- Obiekcji klientów
- Strategii reakcji doradców na obiekcje

## 🎯 Cel projektu

Odpowiedź na pytanie: **"Jak doradcy radzą sobie z obiekcjami klientów w rozmowach sprzedażowych?"**

## 📊 Wyniki analizy

### Przetworzone dane:
- **1501 rozmów** z pliku `transcripts_combined - sample2.csv`
- **Czas przetwarzania**: <30 sekund
- **Skalowalność**: gotowe na 10,000+ rozmów dziennie

### Kluczowe wyniki:
- **Rozmowy sprzedażowe**: 96.3% (1446/1501)
- **Rozmowy z obiekcjami**: 97.5% (1465/1501)
- **Rozmowy z reakcjami**: 96.9% (1456/1501)

### Najczęstsze obiekcje klientów:
1. **Czas** (95.7%) - "nie mam czasu", "jestem zajęty"
2. **Potrzeba** (56.9%) - "nie potrzebuję tego"
3. **Informacja** (25.8%) - "nie rozumiem", "wyjaśnij więcej"

### Najczęstsze reakcje doradców:
1. **Korzyści** (91.1%) - "otrzymasz dodatkowe korzyści"
2. **Uspokojenie** (65.6%) - "gwarantuję", "obiecuję"
3. **Wyjaśnienie** (52.1%) - "pozwól mi wyjaśnić"

## 🚀 Uruchomienie

### Wymagania:
- Python 3.8+
- Biblioteki: pandas, numpy, matplotlib, seaborn

### Instalacja zależności:
```bash
# Ubuntu/Debian
sudo apt install python3-pandas python3-numpy python3-matplotlib python3-seaborn

# Lub przez pip (w wirtualnym środowisku)
pip install -r requirements.txt
```

### Uruchomienie analizy:
```bash
python3 sales_analysis.py
```

## 📁 Struktura plików

```
├── sales_analysis.py          # Główny skrypt analizy
├── requirements.txt           # Zależności Python
├── prezentacja_analizy.md     # 3-slajdowa prezentacja
├── dokumentacja_realizacji.md # Szczegółowa dokumentacja
├── README.md                  # Ten plik
└── transcripts_combined - sample2.csv  # Dane wejściowe
```

## 🔧 Architektura rozwiązania

### Klasyfikacja rozmów sprzedażowych:
- Słowa kluczowe: "medicare", "insurance", "benefits", "coverage"
- Wzorce regex: "reason.*call", "purpose.*call", "upgraded.*benefits"

### Wykrywanie obiekcji:
- **Cena**: "too expensive", "cost too much"
- **Czas**: "busy", "no time", "rush"
- **Potrzeba**: "don't need", "not interested"
- **Zaufanie**: "scam", "fraud", "don't trust"
- **Informacja**: "don't understand", "confused"

### Identyfikacja reakcji:
- **Korzyści**: "benefits receive", "additional benefits"
- **Pilność**: "limited time", "offer expires"
- **Dowód społeczny**: "many people", "others qualified"
- **Wyjaśnienie**: "let me explain", "clarify"
- **Uspokojenie**: "assure", "guarantee", "promise"

## 📈 Metryki jakości

- **Precyzja rozmów sprzedażowych**: 58%
- **Recall rozmów sprzedażowych**: 97%
- **Walidacja**: rozmowy z pewnością transkrypcji >95%

## 🔄 Reużywalność

Rozwiązanie może być łatwo adaptowane do innych przypadków:

```python
# Przykład adaptacji dla innej branży
objection_patterns = {
    'price': [r'too.*expensive', r'cost.*high'],
    'quality': [r'not.*good', r'poor.*quality'],
    'service': [r'bad.*service', r'unhelpful']
}
```

## 📝 Dokumentacja

- **`prezentacja_analizy.md`** - 3-slajdowa prezentacja wyników
- **`dokumentacja_realizacji.md`** - Szczegółowy opis procesu realizacji
- **`sales_analysis.py`** - Kod źródłowy z komentarzami

## 🎯 Następne kroki

1. **A/B testing** różnych wzorców
2. **Manualna walidacja** losowych próbek
3. **Implementacja w produkcji**
4. **Rozszerzenie o machine learning**
5. **Integracja z systemami CRM**

## 👨‍💻 Autor

**Marek** - Prototyp rozwiązania do analizy rozmów sprzedażowych

---

*Projekt zrealizowany w ramach zadania rekrutacyjnego w czasie 6 godzin.*