#!/usr/bin/env python3
"""
Analiza Rozmów Sprzedażowych - Prototyp
Cel: Analiza obiekcji klientów i sposobów ich zbijania przez doradców
Autor: Marek
"""

import pandas as pd
import numpy as np
import re
import matplotlib.pyplot as plt
import seaborn as sns
from collections import Counter
import warnings
warnings.filterwarnings('ignore')

# Ustawienia wyświetlania
pd.set_option('display.max_columns', None)
pd.set_option('display.max_colwidth', 200)

class SalesAnalysis:
    def __init__(self, csv_file):
        """Inicjalizacja analizy"""
        self.df = pd.read_csv(csv_file)
        self.sales_keywords = [
            'medicare', 'insurance', 'benefits', 'coverage', 'plan',
            'upgrade', 'additional', 'cost', 'premium', 'policy',
            'qualify', 'eligible', 'enroll', 'sign up', 'offer',
            'discount', 'savings', 'money back', 'co payment', 'co-pay'
        ]
        
        self.sales_patterns = [
            r'reason.*call.*let.*know',
            r'purpose.*call',
            r'upgraded.*benefits',
            r'additional.*cost',
            r'no.*additional.*cost',
            r'qualify.*benefits',
            r'eligible.*receive'
        ]
        
        self.objection_patterns = {
            'price': [
                r'too.*expensive', r'cost.*too.*much', r'price.*high',
                r'can.*afford', r'budget', r'money.*tight'
            ],
            'time': [
                r'busy', r'no.*time', r'don.*have.*time', r'rush',
                r'have.*to.*go', r'leaving', r'important.*call'
            ],
            'need': [
                r'don.*need', r'not.*interested', r'not.*want',
                r'not.*looking', r'not.*right.*time'
            ],
            'trust': [
                r'scam', r'fraud', r'don.*trust', r'suspicious',
                r'third.*call', r'multiple.*calls'
            ],
            'information': [
                r'don.*understand', r'confused', r'not.*clear',
                r'explain.*more', r'what.*mean'
            ]
        }
        
        self.response_patterns = {
            'benefit_focus': [
                r'benefits.*receive', r'additional.*benefits', r'no.*cost',
                r'free.*benefits', r'savings', r'money.*back'
            ],
            'urgency': [
                r'limited.*time', r'offer.*expires', r'only.*today',
                r'last.*chance', r'don.*miss'
            ],
            'social_proof': [
                r'many.*people', r'others.*qualified', r'popular',
                r'recommended', r'trusted'
            ],
            'clarification': [
                r'let.*explain', r'clarify', r'understand',
                r'mean.*that', r'not.*change'
            ],
            'reassurance': [
                r'assure', r'guarantee', r'promise', r'last.*call',
                r'no.*more.*calls', r'final'
            ]
        }
    
    def identify_sales_conversation(self, text):
        """Identyfikuje czy rozmowa jest sprzedażowa"""
        if pd.isna(text):
            return False
        
        text_lower = text.lower()
        
        # Sprawdzenie słów kluczowych
        keyword_count = sum(1 for keyword in self.sales_keywords if keyword in text_lower)
        
        # Sprawdzenie wzorców
        pattern_count = sum(1 for pattern in self.sales_patterns if re.search(pattern, text_lower))
        
        # Kryteria klasyfikacji
        is_sales = (keyword_count >= 3) or (pattern_count >= 2)
        
        return is_sales
    
    def detect_objections(self, text):
        """Wykrywa obiekcje w tekście rozmowy"""
        if pd.isna(text):
            return {}
        
        text_lower = text.lower()
        objections = {}
        
        for objection_type, patterns in self.objection_patterns.items():
            matches = []
            for pattern in patterns:
                if re.search(pattern, text_lower):
                    matches.append(pattern)
            if matches:
                objections[objection_type] = matches
        
        return objections
    
    def detect_responses(self, text):
        """Wykrywa reakcje doradców na obiekcje"""
        if pd.isna(text):
            return {}
        
        text_lower = text.lower()
        responses = {}
        
        for response_type, patterns in self.response_patterns.items():
            matches = []
            for pattern in patterns:
                if re.search(pattern, text_lower):
                    matches.append(pattern)
            if matches:
                responses[response_type] = matches
        
        return responses
    
    def analyze_conversations(self):
        """Przeprowadza pełną analizę rozmów"""
        print("=== ANALIZA ROZMÓW SPRZEDAŻOWYCH ===")
        
        # Identyfikacja rozmów sprzedażowych
        self.df['is_sales_conversation'] = self.df['full_text'].apply(self.identify_sales_conversation)
        
        # Wykrywanie obiekcji
        self.df['objections'] = self.df['full_text'].apply(self.detect_objections)
        self.df['has_objections'] = self.df['objections'].apply(lambda x: len(x) > 0)
        self.df['objection_count'] = self.df['objections'].apply(lambda x: len(x))
        
        # Wykrywanie reakcji
        self.df['responses'] = self.df['full_text'].apply(self.detect_responses)
        self.df['has_responses'] = self.df['responses'].apply(lambda x: len(x) > 0)
        self.df['response_count'] = self.df['responses'].apply(lambda x: len(x))
        
        # Podstawowe statystyki
        print(f"Liczba rozmów: {len(self.df)}")
        print(f"Rozmowy sprzedażowe: {self.df['is_sales_conversation'].sum()} ({self.df['is_sales_conversation'].mean()*100:.1f}%)")
        print(f"Rozmowy z obiekcjami: {self.df['has_objections'].sum()} ({self.df['has_objections'].mean()*100:.1f}%)")
        print(f"Rozmowy z reakcjami: {self.df['has_responses'].sum()} ({self.df['has_responses'].mean()*100:.1f}%)")
        print(f"Średnia długość rozmowy: {self.df['audio_duration_seconds'].mean():.1f} sekund")
        print(f"Średnia liczba słów: {self.df['word_count'].mean():.1f}")
        
        return self.df
    
    def analyze_objection_types(self):
        """Analizuje typy obiekcji"""
        print("\n=== ANALIZA TYPÓW OBIEKCJI ===")
        
        objection_types = []
        for objections in self.df['objections']:
            objection_types.extend(list(objections.keys()))
        
        objection_counter = Counter(objection_types)
        
        for objection_type, count in objection_counter.most_common():
            print(f"{objection_type}: {count} ({count/len(self.df)*100:.1f}% rozmów)")
        
        return objection_counter
    
    def analyze_response_types(self):
        """Analizuje typy reakcji"""
        print("\n=== ANALIZA TYPÓW REAKCJI ===")
        
        response_types = []
        for responses in self.df['responses']:
            response_types.extend(list(responses.keys()))
        
        response_counter = Counter(response_types)
        
        for response_type, count in response_counter.most_common():
            print(f"{response_type}: {count} ({count/len(self.df)*100:.1f}% rozmów)")
        
        return response_counter
    
    def show_examples(self, n=3):
        """Pokazuje przykłady rozmów z obiekcjami"""
        print(f"\n=== PRZYKŁADY ROZMÓW Z OBIEKCJAMI (top {n}) ===")
        
        objection_examples = self.df[self.df['has_objections']].head(n)
        
        for idx, row in objection_examples.iterrows():
            print(f"\nRozmowa {idx}:")
            print(f"Obiekcje: {list(row['objections'].keys())}")
            print(f"Reakcje: {list(row['responses'].keys())}")
            print(f"Tekst (pierwsze 200 znaków): {row['full_text'][:200]}...")
    
    def calculate_metrics(self):
        """Oblicza metryki jakości"""
        print("\n=== METRYKI JAKOŚCI ===")
        
        # Przykładowa walidacja - rozmowy z wysoką pewnością transkrypcji
        high_confidence = self.df['confidence'] > 0.95
        
        # Precyzja i recall dla rozmów sprzedażowych
        sales_precision = self.df[self.df['is_sales_conversation'] & high_confidence].shape[0] / self.df[self.df['is_sales_conversation']].shape[0]
        sales_recall = self.df[self.df['is_sales_conversation'] & high_confidence].shape[0] / self.df[high_confidence].shape[0]
        
        print(f"Rozmowy sprzedażowe - Precyzja: {sales_precision:.3f}, Recall: {sales_recall:.3f}")
        
        return {
            'sales_precision': sales_precision,
            'sales_recall': sales_recall
        }
    
    def create_visualizations(self):
        """Tworzy wizualizacje wyników"""
        print("\n=== TWORZENIE WIZUALIZACJI ===")
        
        fig, axes = plt.subplots(2, 2, figsize=(15, 12))
        
        # 1. Rozkład długości rozmów
        axes[0,0].hist(self.df['audio_duration_seconds'], bins=30, alpha=0.7, color='skyblue')
        axes[0,0].set_title('Rozkład Długości Rozmów')
        axes[0,0].set_xlabel('Czas (sekundy)')
        axes[0,0].set_ylabel('Liczba rozmów')
        
        # 2. Typy obiekcji
        objection_types = []
        for objections in self.df['objections']:
            objection_types.extend(list(objections.keys()))
        objection_counter = Counter(objection_types)
        objection_data = pd.DataFrame(objection_counter.most_common(), columns=['Typ', 'Liczba'])
        axes[0,1].barh(objection_data['Typ'], objection_data['Liczba'], color='lightcoral')
        axes[0,1].set_title('Typy Obiekcji Klientów')
        axes[0,1].set_xlabel('Liczba wystąpień')
        
        # 3. Typy reakcji
        response_types = []
        for responses in self.df['responses']:
            response_types.extend(list(responses.keys()))
        response_counter = Counter(response_types)
        response_data = pd.DataFrame(response_counter.most_common(), columns=['Typ', 'Liczba'])
        axes[1,0].barh(response_data['Typ'], response_data['Liczba'], color='lightgreen')
        axes[1,0].set_title('Typy Reakcji Doradców')
        axes[1,0].set_xlabel('Liczba wystąpień')
        
        # 4. Korelacja obiekcji i reakcji
        axes[1,1].scatter(self.df['objection_count'], self.df['response_count'], alpha=0.6, color='purple')
        axes[1,1].set_title('Korelacja: Obiekcje vs Reakcje')
        axes[1,1].set_xlabel('Liczba obiekcji')
        axes[1,1].set_ylabel('Liczba reakcji')
        
        plt.tight_layout()
        plt.savefig('sales_analysis_results.png', dpi=300, bbox_inches='tight')
        plt.show()
        
        print("Wizualizacje zapisane jako 'sales_analysis_results.png'")
    
    def generate_report(self):
        """Generuje raport końcowy"""
        print("\n=== RAPORT KOŃCOWY ===")
        
        results = {
            'total_conversations': len(self.df),
            'sales_conversations': self.df['is_sales_conversation'].sum(),
            'conversations_with_objections': self.df['has_objections'].sum(),
            'conversations_with_responses': self.df['has_responses'].sum(),
            'avg_objections_per_conversation': self.df['objection_count'].mean(),
            'avg_responses_per_conversation': self.df['response_count'].mean(),
        }
        
        # Dodanie top typów obiekcji i reakcji
        objection_types = []
        for objections in self.df['objections']:
            objection_types.extend(list(objections.keys()))
        objection_counter = Counter(objection_types)
        results['top_objection_types'] = dict(objection_counter.most_common(3))
        
        response_types = []
        for responses in self.df['responses']:
            response_types.extend(list(responses.keys()))
        response_counter = Counter(response_types)
        results['top_response_types'] = dict(response_counter.most_common(3))
        
        print("Kluczowe Wyniki:")
        for key, value in results.items():
            print(f"  {key}: {value}")
        
        return results

def main():
    """Główna funkcja"""
    print("=== ANALIZA ROZMÓW SPRZEDAŻOWYCH - PROTOTYP ===")
    print("Autor: Marek")
    print("Cel: Analiza obiekcji klientów i sposobów ich zbijania przez doradców\n")
    
    # Inicjalizacja analizy
    analyzer = SalesAnalysis('transcripts_combined - sample2.csv')
    
    # Przeprowadzenie analizy
    analyzer.analyze_conversations()
    analyzer.analyze_objection_types()
    analyzer.analyze_response_types()
    analyzer.show_examples()
    analyzer.calculate_metrics()
    
    # Generowanie raportu
    results = analyzer.generate_report()
    
    print("\n=== ANALIZA ZAKOŃCZONA ===")
    print("Wyniki zostały zapisane i wizualizacje wygenerowane.")

if __name__ == "__main__":
    main()