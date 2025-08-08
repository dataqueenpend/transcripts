## Podsumowanie analizy rozmów

- **Liczba wszystkich transkryptów**: 4005
- **Rozmowy sprzedażowe (heurystyka słów-kluczy)**: 3967
- **Liczba wykrytych obiekcji**: 18640

### Techniki zbijania obiekcji i ich skuteczność

- **feature/benefit reframing**: 9764 zdarzeń; akceptacja: 0.0%; opór: 0.0%
- **transfer to expert/authority**: 3794 zdarzeń; akceptacja: 0.0%; opór: 0.0%
- **assurance/cost removal**: 2117 zdarzeń; akceptacja: 0.0%; opór: 0.0%
- **generic rebuttal**: 1900 zdarzeń; akceptacja: 0.0%; opór: 0.0%
- **minimize effort/foot-in-the-door**: 591 zdarzeń; akceptacja: 0.0%; opór: 0.0%
- **authority/qualification**: 283 zdarzeń; akceptacja: 0.0%; opór: 0.0%
- **value proposition**: 179 zdarzeń; akceptacja: 0.0%; opór: 0.0%
- **reassurance/empathy**: 12 zdarzeń; akceptacja: 0.0%; opór: 0.0%

### Metodologia
- **Identyfikacja rozmów sprzedażowych**: reguły oparte na słowach-kluczach (np. ‘Medicare’, ‘benefits’, ‘plan’, ‘no additional cost’, ‘eligible’, ‘connect’). Próg >= 3 dopasowań + obecność rdzeniowych słów (benefit/plan/Medicare).
- **Segmentacja**: proste dzielenie na zdania po znakach `.?!`.
- **Obiekcje**: wykrywane wzorcami (np. ‘I don't’, ‘not interested’, pytania ‘why/when/how/what’, negacje i niepewność).
- **Zbijanie obiekcji (rebuttal)**: zdania następujące po obiekcji, zawierające leksykon doradcy (np. ‘no additional cost’, ‘you are eligible’, ‘I will connect...’). Przypisywanie techniki na podstawie słów-wyzwalaczy.
- **Wpływ na klienta**: natychmiastowa reakcja w kolejnych 1–2 zdaniach (tokeny akceptacji: ‘okay/yes/right’, oporu: ‘no/not/busy’).
- **Podsumowania**: ekstrakcyjny skrót zdań (rankowanie częstotliwościowe bez zależności od zewnętrznego LLM).

### Ograniczenia i możliwe ulepszenia
- **Brak oznaczeń ról mówców** utrudnia precyzyjne przypisanie, kto zgłasza obiekcje. Można poprawić przez diarization ASR lub LLM, które klasyfikuje role zdań.
- **Reguły słów-kluczowych** są szybkie, ale kruche. Warto dodać klasyfikator uczenia maszynowego (np. BERT/LoRA) albo promptowany LLM do etykietowania: ‘sprzedaż’, ‘obiekcja’, ‘rebuttal’, ‘reakcja klienta’.
- **Ocena wpływu** jest bardzo lokalna (1–2 zdania). Można modelować sekwencję stanów (‘opór’→‘akceptacja’) i mierzyć ‘czas do akceptacji’.

### Architektura modułowa dla wielu business case’ów
- **Ingestion**: moduł loaderów (CSV/Parquet/DB) + normalizacja pól (tekst, czas, meta).
- **Segmentacja i role**: interfejs `Segmenter` (regułowy/LLM) i `SpeakerAttribution` (heurystyki, VAD/diarization, LLM).
- **Detekcja eventów**: `EventDetector` z pluginami: ‘objection’, ‘rebuttal’, ‘question’, ‘compliance’. Każdy plugin ma własne reguły/LLM/klasyfikator.
- **Taksonomia technik**: konfigurowalny słownik → etykiety. Można trzymać w YAML i wersjonować.
- **Ewaluacja wpływu**: `ImpactScorer` (akceptacja/opór, sentiment, przejście stanu, eskalacja).
- **Summarizer**: strategia `Extractive` lub `LLM` z adapterem (OpenAI/Azure/Ollama) + retry/batching.
- **Orkiestracja**: pipeline w stylu scikit-learn/Prefect (fit/transform) + artefakty (JSON/Parquet) + raporty Markdown/HTML.

### Przykładowe rozmowy i skróty (wybrane)
- **2013404947_transcript.json** — obiekcje: 3 — skrót: So the reason for my call is to let you know your Medicare insurance has been upgraded and you will be receiving more benefit from your Medicare plan at no additional cost. And you will be getting dental, vision, hearing glasses from Transportation benefit, which will help you in getting pickup and drop services from the ambulance in emergency. And you will be getting dental, vision, hearing glasses from Transportation benefit, which will help you in getting pickup and drop services from the ambulance in emergency.
- **2014616758_transcript.json** — obiekcje: 5 — skrót: My name is [PERSON_NAME]. So the reason of my call is to let you know that your Medicare has been upgraded and you are receiving some additional Medicare benefits at no additional cost to you. These are additional Medicare benefits which will be added to your existing plan.
- **2015694145_transcript.json** — obiekcje: 4 — skrót: Okay, so dear, the reason today I gave you call is to let you know that your Medicare insurance has been updated for the year [DATE_INTERVAL] and you are receiving some additional benefits on your Medicare plan. He just verify your details and directly get this call connected to the [OCCUPATION] [OCCUPATION] who will be from your same area and zip code, that is [LOCATION]. In details and all about the paperwork within [DURATION] [DURATION] [DURATION] [DURATION] of your precious time.
- **2015698147_transcript.json** — obiekcje: 4 — skrót: So right now what I will do, I will just like connect your call, ma' am, to the Medicare [OCCUPATION] [OCCUPATION], like, who is from the same area, ma' am, from the same zip code. And he will just going to let you know, ma' am, when you will be receiving the paperwork about these benefits so you can read about the benefits as well. And he will just going to let you know, ma' am, when you will be receiving the paperwork about these benefits so you can read about the benefits as well.
- **2025464247_transcript.json** — obiekcje: 2 — skrót: Like after providing these benefits to you, your all expenses gonna taken care by Medicare part A and Medicare part B. I will quickly get this call connected to one of my [OCCUPATION] [OCCUPATION]. He, he just going to verify your details and directly get this call connected to the [OCCUPATION] [OCCUPATION] who will be from your same area and zip code.
- **203-748-1950_transcript.json** — obiekcje: 4 — skrót: So ma' am, the purpose of my call is to let you know that your Medicare is getting updated today. So ma' am, like after these benefits you don't need to know co payments to your [OCCUPATION]. And apart from this, you were also qualified to receive the medication co payments benefit.
- **2032810169_transcript.json** — obiekcje: 8 — skrót: The reason of my call is to let you know that your Medicare has been upgraded and you are qualified to receive some additional Medicare benefits at no additional cost to you. The reason of my call is to let you know that your Medicare has been upgraded and you are qualified to receive some additional Medicare benefits at no additional cost to you. So as I can see, you are all qualified to receive all these benefits at [MONEY_AMOUNT] [MONEY_AMOUNT] premium to you.
- **2035961822_transcript.json** — obiekcje: 6 — skrót: And ma' am, the reason for my call is to let you know that your Medicare is getting updated for this year and you are receiving much more additional Medicare benefits at no additional cost to you. Now, I will get your call connected to the [OCCUPATION] [OCCUPATION] who will be from your same zip code, and he will be the right person to let you know what benefits you are entitled to receive this year at no additional cost to you. And do you receive any [OCCUPATION] or any kind of [ORGANIZATION] benefits like TRICARE for life?
- **2037239090_transcript.json** — obiekcje: 3 — skrót: The reason of my call is just to let you know today Medicare has been updated and you have been qualified to receive some additional Medicare benefits at no additional cost. All right, madam, so once we provide those benefits to you, you no longer have to pay any co payments to your [OCCUPATION] [OCCUPATION] or another day hospital. And all these benefits will be added to your Medicare plan with no additional cost to you.
- **2038789478_transcript.json** — obiekcje: 5 — skrót: And ma' am, the reason for my call is to let you know that your Medicare is getting updated for this year and you're receiving much more additional Medicare benefits at no additional cost to you. Like, we are just here to help you to get some more additional benefits at no additional cost to you. Like, we are just here to help you to get some more additional benefits at no additional cost to you.

### Podsumowania LLM
- **2013404947_transcript.json** — So the reason for my call is to let you know your Medicare insurance has been upgraded and you will be receiving more benefit from your Medicare plan at no additional cost. And you will be getting dental, vision, hearing glasses from Transportation benefit, which will help you in getting pickup and drop services from the ambulance in emergency. And you will be getting dental, vision, hearing glasses from Transportation benefit, which will help you in getting pickup and drop services from the ambulance in emergency.
- **2014616758_transcript.json** — My name is [PERSON_NAME]. So the reason of my call is to let you know that your Medicare has been upgraded and you are receiving some additional Medicare benefits at no additional cost to you. These are additional Medicare benefits which will be added to your existing plan.
- **2015694145_transcript.json** — Okay, so dear, the reason today I gave you call is to let you know that your Medicare insurance has been updated for the year [DATE_INTERVAL] and you are receiving some additional benefits on your Medicare plan. He just verify your details and directly get this call connected to the [OCCUPATION] [OCCUPATION] who will be from your same area and zip code, that is [LOCATION]. In details and all about the paperwork within [DURATION] [DURATION] [DURATION] [DURATION] of your precious time.
- **2015698147_transcript.json** — So right now what I will do, I will just like connect your call, ma' am, to the Medicare [OCCUPATION] [OCCUPATION], like, who is from the same area, ma' am, from the same zip code. And he will just going to let you know, ma' am, when you will be receiving the paperwork about these benefits so you can read about the benefits as well. And he will just going to let you know, ma' am, when you will be receiving the paperwork about these benefits so you can read about the benefits as well.
- **2025464247_transcript.json** — Like after providing these benefits to you, your all expenses gonna taken care by Medicare part A and Medicare part B. I will quickly get this call connected to one of my [OCCUPATION] [OCCUPATION]. He, he just going to verify your details and directly get this call connected to the [OCCUPATION] [OCCUPATION] who will be from your same area and zip code.
- **203-748-1950_transcript.json** — So ma' am, the purpose of my call is to let you know that your Medicare is getting updated today. So ma' am, like after these benefits you don't need to know co payments to your [OCCUPATION]. And apart from this, you were also qualified to receive the medication co payments benefit.
- **2032810169_transcript.json** — The reason of my call is to let you know that your Medicare has been upgraded and you are qualified to receive some additional Medicare benefits at no additional cost to you. The reason of my call is to let you know that your Medicare has been upgraded and you are qualified to receive some additional Medicare benefits at no additional cost to you. So as I can see, you are all qualified to receive all these benefits at [MONEY_AMOUNT] [MONEY_AMOUNT] premium to you.
- **2035961822_transcript.json** — And ma' am, the reason for my call is to let you know that your Medicare is getting updated for this year and you are receiving much more additional Medicare benefits at no additional cost to you. Now, I will get your call connected to the [OCCUPATION] [OCCUPATION] who will be from your same zip code, and he will be the right person to let you know what benefits you are entitled to receive this year at no additional cost to you. And do you receive any [OCCUPATION] or any kind of [ORGANIZATION] benefits like TRICARE for life?
- **2037239090_transcript.json** — The reason of my call is just to let you know today Medicare has been updated and you have been qualified to receive some additional Medicare benefits at no additional cost. All right, madam, so once we provide those benefits to you, you no longer have to pay any co payments to your [OCCUPATION] [OCCUPATION] or another day hospital. And all these benefits will be added to your Medicare plan with no additional cost to you.
- **2038789478_transcript.json** — And ma' am, the reason for my call is to let you know that your Medicare is getting updated for this year and you're receiving much more additional Medicare benefits at no additional cost to you. Like, we are just here to help you to get some more additional benefits at no additional cost to you. Like, we are just here to help you to get some more additional benefits at no additional cost to you.

Uwaga: aby włączyć LLM, ustaw `SUMMARIZER_MODEL=openai:gpt-4o-mini` oraz `OPENAI_API_KEY`.

