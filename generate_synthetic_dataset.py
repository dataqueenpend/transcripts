import csv
import random
from datetime import datetime, timedelta

# Nazwy produktów Santander
PRODUCTS = {
    "karta_kredytowa": "Santander Premium Card",
    "limit": "Santander FlexiLimit", 
    "ubezpieczenie": "Santander Protect Plus"
}

# Imiona doradców
ADVISOR_NAMES = [
    "Anna", "Marek", "Katarzyna", "Piotr", "Joanna", "Tomasz", "Magdalena", "Michał",
    "Agnieszka", "Jan", "Ewa", "Andrzej", "Monika", "Grzegorz", "Barbara", "Łukasz",
    "Dorota", "Paweł", "Elżbieta", "Marcin", "Małgorzata", "Adam", "Iwona", "Robert"
]

# Imiona klientów
CUSTOMER_NAMES = [
    "Jan", "Anna", "Piotr", "Maria", "Andrzej", "Katarzyna", "Tomasz", "Agnieszka",
    "Marek", "Ewa", "Michał", "Joanna", "Grzegorz", "Monika", "Łukasz", "Barbara",
    "Adam", "Dorota", "Marcin", "Elżbieta", "Paweł", "Małgorzata", "Robert", "Iwona"
]

# Miasta
CITIES = [
    "Warszawa", "Kraków", "Łódź", "Wrocław", "Poznań", "Gdańsk", "Szczecin", "Bydgoszcz",
    "Lublin", "Katowice", "Białystok", "Gdynia", "Częstochowa", "Radom", "Sosnowiec",
    "Toruń", "Kielce", "Rzeszów", "Gliwice", "Zabrze", "Bytom", "Bielsko-Biała",
    "Olsztyn", "Ruda Śląska", "Rybnik", "Tychy", "Dąbrowa Górnicza", "Opole"
]

# Typy rozmów
CONVERSATION_TYPES = [
    "sprzedażowa",
    "informacyjna", 
    "serwisowa",
    "reklamacyjna",
    "konsultacyjna"
]

def generate_conversation_id():
    """Generuje unikalne ID rozmowy"""
    timestamp = datetime.now().strftime("%Y%m%d%H%M%S")
    random_num = random.randint(1000, 9999)
    return f"conv_{timestamp}_{random_num}"

def generate_sales_conversation():
    """Generuje rozmowę sprzedażową"""
    product = random.choice(list(PRODUCTS.keys()))
    product_name = PRODUCTS[product]
    advisor = random.choice(ADVISOR_NAMES)
    customer = random.choice(CUSTOMER_NAMES)
    city = random.choice(CITIES)
    
    # Podstawowe elementy rozmowy
    intro = f"Dzień dobry, nazywam się {advisor} i dzwonię z banku Santander. Rozmowa jest nagrywana. Dzwonię w sprawie specjalnej oferty dotyczącej {product_name}."
    
    # Różne scenariusze rozmów sprzedażowych
    scenarios = [
        # Scenariusz 1: Klient zainteresowany
        f"{intro} Czy mogę rozmawiać z panem/panią {customer}? Tak, to ja. Świetnie! Chciałbym przedstawić panu/pani specjalną ofertę {product_name}. To produkt, który może bardzo pomóc w codziennych finansach. Czy byłby pan/pani zainteresowany/a poznaniem szczegółów? Tak, chętnie posłucham. Doskonale! {product_name} oferuje bardzo atrakcyjne warunki...",
        
        # Scenariusz 2: Klient sceptyczny
        f"{intro} Czy mogę rozmawiać z panem/panią {customer}? Tak, ale nie jestem zainteresowany/a żadnymi ofertami. Rozumiem, ale {product_name} to naprawdę wyjątkowy produkt. Może chociaż posłucha pan/pani przez minutę? Hmm, no dobrze, ale szybko. Dziękuję! {product_name} to...",
        
        # Scenariusz 3: Klient z obiekcjami
        f"{intro} Czy mogę rozmawiać z panem/panią {customer}? Tak, słucham. Chciałbym przedstawić panu/pani ofertę {product_name}. Nie, dziękuję, nie jestem zainteresowany/a. Rozumiem, ale może warto poznać szczegóły? {product_name} oferuje bardzo niskie oprocentowanie. Ile wynosi? Tylko 5.9% w skali roku. To brzmi drogo. W porównaniu do innych banków to bardzo atrakcyjna oferta. Dodatkowo nie ma ukrytych kosztów. Hmm, a jakie są warunki?",
        
        # Scenariusz 4: Klient pytający o szczegóły
        f"{intro} Czy mogę rozmawiać z panem/panią {customer}? Tak, słucham. Chciałbym przedstawić panu/pani ofertę {product_name}. Co to za produkt? {product_name} to nasz flagowy produkt finansowy. Jakie ma zalety? Oferuje bardzo niskie oprocentowanie, brak ukrytych kosztów i elastyczne warunki spłaty. Brzmi interesująco. Czy mogę poznać szczegóły? Oczywiście!",
        
        # Scenariusz 5: Klient z doświadczeniem
        f"{intro} Czy mogę rozmawiać z panem/panią {customer}? Tak, ale już mam podobny produkt w innym banku. Rozumiem, ale {product_name} może oferować lepsze warunki. Jakie konkretnie? Oprocentowanie tylko 5.9%, brak opłat za prowadzenie i możliwość zwiększenia limitu. To rzeczywiście lepiej niż mam teraz. Czy mogę poznać więcej szczegółów?",
    ]
    
    conversation = random.choice(scenarios)
    
    # Dodanie zakończenia
    endings = [
        "Dziękuję za rozmowę. Czy mogę przesłać panu/pani szczegółowe informacje?",
        "Czy chciałby pan/pani, żebym skontaktował się ponownie w dogodnym terminie?",
        "Dziękuję za poświęcony czas. Mam nadzieję, że oferta była interesująca.",
        "Czy mogę zapisać panu/panią wizytę w oddziale, żeby omówić szczegóły?",
        "Dziękuję za rozmowę. Życzę miłego dnia!"
    ]
    
    conversation += " " + random.choice(endings)
    
    return {
        "is_sales": True,
        "is_rebuttal": "obiekcje" in conversation.lower() or "drogo" in conversation.lower(),
        "is_obj": "nie jestem zainteresowany" in conversation.lower() or "dziękuję" in conversation.lower() or "drogo" in conversation.lower(),
        "conversation": conversation
    }

def generate_service_conversation():
    """Generuje rozmowę serwisową"""
    advisor = random.choice(ADVISOR_NAMES)
    customer = random.choice(CUSTOMER_NAMES)
    
    scenarios = [
        f"Dzień dobry, nazywam się {advisor} i dzwonię z banku Santander. Rozmowa jest nagrywana. Dzwonię w sprawie pana/pani {customer} konta. Czy mogę rozmawiać z panem/panią? Tak, to ja. Dziękuję. Chciałbym poinformować, że pana/pani konto zostało zaktualizowane i teraz oferuje nowe funkcje. Czy chciałby pan/pani poznać szczegóły? Tak, proszę. Świetnie! Nowe funkcje obejmują...",
        
        f"Dzień dobry, nazywam się {advisor} i dzwonię z banku Santander. Rozmowa jest nagrywana. Dzwonię w sprawie pana/pani {customer} karty płatniczej. Czy mogę rozmawiać z panem/panią? Tak, słucham. Chciałbym poinformować, że pana/pani karta została zastąpiona nową wersją z lepszymi zabezpieczeniami. Kiedy otrzymam nową kartę? W ciągu 5-7 dni roboczych. Dziękuję za informację.",
        
        f"Dzień dobry, nazywam się {advisor} i dzwonię z banku Santander. Rozmowa jest nagrywana. Dzwonię w sprawie pana/pani {customer} aplikacji mobilnej. Czy mogę rozmawiać z panem/panią? Tak, to ja. Chciałbym poinformować o nowej aktualizacji aplikacji, która poprawia bezpieczeństwo i dodaje nowe funkcje. Czy chciałby pan/pani poznać szczegóły? Tak, proszę. Nowa wersja oferuje...",
    ]
    
    conversation = random.choice(scenarios)
    
    return {
        "is_sales": False,
        "is_rebuttal": False,
        "is_obj": False,
        "conversation": conversation
    }

def generate_informational_conversation():
    """Generuje rozmowę informacyjną"""
    advisor = random.choice(ADVISOR_NAMES)
    customer = random.choice(CUSTOMER_NAMES)
    
    scenarios = [
        f"Dzień dobry, nazywam się {advisor} i dzwonię z banku Santander. Rozmowa jest nagrywana. Dzwonię w sprawie pana/pani {customer} konta. Czy mogę rozmawiać z panem/panią? Tak, słucham. Chciałbym poinformować o zmianach w regulaminie banku, które wchodzą w życie od przyszłego miesiąca. Czy chciałby pan/pani poznać szczegóły? Tak, proszę. Zmiany dotyczą głównie opłat za prowadzenie konta...",
        
        f"Dzień dobry, nazywam się {advisor} i dzwonię z banku Santander. Rozmowa jest nagrywana. Dzwonię w sprawie pana/pani {customer} bezpieczeństwa transakcji. Czy mogę rozmawiać z panem/panią? Tak, to ja. Chciałbym przypomnieć o ważnych zasadach bezpieczeństwa przy korzystaniu z bankowości internetowej. Czy chciałby pan/pani poznać szczegóły? Tak, proszę. Pamiętaj, aby nigdy nie podawać swoich danych logowania...",
        
        f"Dzień dobry, nazywam się {advisor} i dzwonię z banku Santander. Rozmowa jest nagrywana. Dzwonię w sprawie pana/pani {customer} konta. Czy mogę rozmawiać z panem/panią? Tak, słucham. Chciałbym poinformować o nowych funkcjach w bankowości internetowej, które mogą być dla pana/pani przydatne. Czy chciałby pan/pani poznać szczegóły? Tak, proszę. Nowe funkcje obejmują...",
    ]
    
    conversation = random.choice(scenarios)
    
    return {
        "is_sales": False,
        "is_rebuttal": False,
        "is_obj": False,
        "conversation": conversation
    }

def generate_complaint_conversation():
    """Generuje rozmowę reklamacyjną"""
    advisor = random.choice(ADVISOR_NAMES)
    customer = random.choice(CUSTOMER_NAMES)
    
    scenarios = [
        f"Dzień dobry, nazywam się {advisor} i dzwonię z banku Santander. Rozmowa jest nagrywana. Dzwonię w sprawie pana/pani {customer} reklamacji. Czy mogę rozmawiać z panem/panią? Tak, to ja. Dziękuję za zgłoszenie problemu z kartą płatniczą. Chciałbym przeprosić za niedogodności i poinformować, że sprawa została rozwiązana. Czy karta działa już prawidłowo? Tak, teraz wszystko w porządku. Świetnie! Czy mogę jeszcze w czymś pomóc?",
        
        f"Dzień dobry, nazywam się {advisor} i dzwonię z banku Santander. Rozmowa jest nagrywana. Dzwonię w sprawie pana/pani {customer} problemu z aplikacją mobilną. Czy mogę rozmawiać z panem/panią? Tak, słucham. Chciałbym poinformować, że problem z logowaniem do aplikacji został rozwiązany. Czy aplikacja działa już prawidłowo? Tak, teraz mogę się zalogować bez problemu. Doskonale! Przepraszamy za niedogodności.",
        
        f"Dzień dobry, nazywam się {advisor} i dzwonię z banku Santander. Rozmowa jest nagrywana. Dzwonię w sprawie pana/pani {customer} błędnej transakcji. Czy mogę rozmawiać z panem/panią? Tak, to ja. Chciałbym poinformować, że błędna transakcja została zwrócona na pana/pani konto. Czy otrzymał pan/pani potwierdzenie? Tak, widzę zwrot na koncie. Świetnie! Przepraszamy za błąd i dziękujemy za cierpliwość.",
    ]
    
    conversation = random.choice(scenarios)
    
    return {
        "is_sales": False,
        "is_rebuttal": False,
        "is_obj": False,
        "conversation": conversation
    }

def generate_consultation_conversation():
    """Generuje rozmowę konsultacyjną"""
    advisor = random.choice(ADVISOR_NAMES)
    customer = random.choice(CUSTOMER_NAMES)
    
    scenarios = [
        f"Dzień dobry, nazywam się {advisor} i dzwonię z banku Santander. Rozmowa jest nagrywana. Dzwonię w sprawie pana/pani {customer} finansów. Czy mogę rozmawiać z panem/panią? Tak, słucham. Chciałbym zaproponować bezpłatną konsultację finansową, która może pomóc w lepszym zarządzaniu pieniędzmi. Czy byłby pan/pani zainteresowany/a? Tak, to może być przydatne. Świetnie! Konsultacja trwa około 30 minut i jest całkowicie bezpłatna...",
        
        f"Dzień dobry, nazywam się {advisor} i dzwonię z banku Santander. Rozmowa jest nagrywana. Dzwonię w sprawie pana/pani {customer} planowania finansowego. Czy mogę rozmawiać z panem/panią? Tak, to ja. Chciałbym zaproponować spotkanie z naszym doradcą finansowym, który może pomóc w planowaniu przyszłości finansowej. Czy byłby pan/pani zainteresowany/a? Tak, to może być pomocne. Doskonale! Spotkanie jest bezpłatne i trwa około godziny...",
        
        f"Dzień dobry, nazywam się {advisor} i dzwonię z banku Santander. Rozmowa jest nagrywana. Dzwonię w sprawie pana/pani {customer} inwestycji. Czy mogę rozmawiać z panem/panią? Tak, słucham. Chciałbym zaproponować konsultację dotyczącą możliwości inwestycyjnych, które mogą pomóc w pomnażaniu oszczędności. Czy byłby pan/pani zainteresowany/a? Tak, chętnie poznam ofertę. Świetnie! Nasz doradca inwestycyjny może przedstawić różne opcje...",
    ]
    
    conversation = random.choice(scenarios)
    
    return {
        "is_sales": False,
        "is_rebuttal": False,
        "is_obj": False,
        "conversation": conversation
    }

def generate_conversation():
    """Generuje pojedynczą rozmowę"""
    conversation_type = random.choice(CONVERSATION_TYPES)
    
    if conversation_type == "sprzedażowa":
        return generate_sales_conversation()
    elif conversation_type == "serwisowa":
        return generate_service_conversation()
    elif conversation_type == "informacyjna":
        return generate_informational_conversation()
    elif conversation_type == "reklamacyjna":
        return generate_complaint_conversation()
    else:  # konsultacyjna
        return generate_consultation_conversation()

def main():
    """Główna funkcja generująca dataset"""
    conversations = []
    
    print("Generowanie syntetycznego datasetu z 4000 rozmów...")
    
    for i in range(4000):
        if i % 500 == 0:
            print(f"Wygenerowano {i} rozmów...")
        
        conv_data = generate_conversation()
        
        # Generowanie dodatkowych danych
        confidence = round(random.uniform(0.85, 0.99), 6)
        duration = random.randint(60, 600)  # 1-10 minut
        word_count = len(conv_data["conversation"].split())
        
        # Generowanie daty w ostatnich 6 miesiącach
        end_date = datetime.now()
        start_date = end_date - timedelta(days=180)
        random_date = start_date + timedelta(
            days=random.randint(0, (end_date - start_date).days)
        )
        filename = f"synthetic_{random_date.strftime('%Y%m%d')}_{random.randint(1000, 9999)}.json"
        
        conversations.append({
            "id": generate_conversation_id(),
            "filename": filename,
            "full_text": conv_data["conversation"],
            "confidence": confidence,
            "audio_duration_seconds": duration,
            "word_count": word_count,
            "is_sales": conv_data["is_sales"],
            "is_rebuttal": conv_data["is_rebuttal"],
            "is_obj": conv_data["is_obj"],
            "redacted_pii_policies": "person_name; phone_number; date_of_birth; location; money_amount"
        })
    
    # Zapisywanie do CSV
    output_file = "synthetic_conversations_dataset.csv"
    
    with open(output_file, 'w', newline='', encoding='utf-8') as csvfile:
        fieldnames = [
            "id", "filename", "full_text", "confidence", "audio_duration_seconds", 
            "word_count", "is_sales", "is_rebuttal", "is_obj", "redacted_pii_policies"
        ]
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        
        writer.writeheader()
        for conv in conversations:
            writer.writerow(conv)
    
    print(f"Dataset został zapisany do pliku: {output_file}")
    print(f"Wygenerowano {len(conversations)} rozmów")
    
    # Statystyki
    sales_count = sum(1 for conv in conversations if conv["is_sales"])
    rebuttal_count = sum(1 for conv in conversations if conv["is_rebuttal"])
    obj_count = sum(1 for conv in conversations if conv["is_obj"])
    
    print(f"\nStatystyki:")
    print(f"Rozmowy sprzedażowe: {sales_count} ({sales_count/len(conversations)*100:.1f}%)")
    print(f"Rozmowy ze zbić obiekcji: {rebuttal_count} ({rebuttal_count/len(conversations)*100:.1f}%)")
    print(f"Rozmowy z obiekcjami: {obj_count} ({obj_count/len(conversations)*100:.1f}%)")

if __name__ == "__main__":
    main()