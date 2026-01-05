"""
Data Processor for Disaster Relevance Classification
Tüm datasetleri yükler, temizler ve birleştirir
"""
import csv
import json
import re
from pathlib import Path
from typing import List, Tuple
import random
import glob
import os


class DataProcessor:
    """Veri setlerini yükleyip işleyen sınıf"""
    
    def __init__(self, data_dir: str = None):
        """
        DataProcessor başlat
        
        Args:
            data_dir: Veri setlerinin bulunduğu dizin
        """
        if data_dir is None:
            # Mevcut dosyanın bulunduğu dizini al
            current_dir = Path(__file__).parent
            self.data_dir = current_dir / "data"
        else:
            self.data_dir = Path(data_dir)
    
    def clean_text(self, text: str) -> str:
        """Metni temizle"""
        if not text:
            return ""
        
        # URL'leri kaldır
        text = re.sub(r'http\S+|www\.\S+', '', text)
        
        # Fazla boşlukları temizle
        text = re.sub(r'\s+', ' ', text)
        
        # Başta/sonda boşlukları kaldır
        text = text.strip()
        
        return text
    
    def load_crisislext26(self) -> List[Tuple[str, int, str]]:
        """CrisisLexT26 klasöründeki tüm event'leri yükle"""
        data = []
        crisislext26_dir = self.data_dir / "CrisisLexT26"
        
        if not crisislext26_dir.exists():
            return data
        
        print("CrisisLexT26 veri setleri yukleniyor...")
        
        # Tüm event klasörlerini bul
        event_dirs = [d for d in crisislext26_dir.iterdir() if d.is_dir()]
        
        for event_dir in event_dirs:
            event_name = event_dir.name
            tweets_file = event_dir / f"{event_name}-tweets_labeled.csv"
            
            if not tweets_file.exists():
                continue
            
            try:
                with open(tweets_file, 'r', encoding='utf-8') as f:
                    reader = csv.DictReader(f)
                    count = 0
                    for row in reader:
                        # ' Tweet Text' kolonunu kontrol et (başında boşluk var)
                        text = row.get(' Tweet Text', '').strip()
                        if not text:
                            text = row.get('Tweet Text', '').strip()
                        
                        if not text or len(text) < 3:
                            continue
                        
                        cleaned_text = self.clean_text(text)
                        if len(cleaned_text) < 3:
                            continue
                        
                        # ' Informativeness' kolonunu kontrol et
                        informativeness = row.get(' Informativeness', '').strip()
                        if not informativeness:
                            informativeness = row.get('Informativeness', '').strip()
                        
                        # Informativeness: "Related and informative" = 1, diğerleri = 0
                        binary_label = 1 if informativeness.lower() == "related and informative" else 0
                        
                        data.append((cleaned_text, binary_label, f"crisislext26_{event_name}"))
                        count += 1
                
                print(f"  [OK] {event_name}: {count} kayit")
            except Exception as e:
                print(f"  [ERROR] {event_name} yuklenirken hata: {e}")
        
        print(f"  [OK] Toplam {len(data)} CrisisLexT26 kayit yuklendi")
        return data
    
    def load_tweets_csv(self) -> List[Tuple[str, int, str]]:
        """tweets.csv dosyasını yükle (Turkey earthquake tweets)"""
        data = []
        tweets_file = self.data_dir / "tweets.csv"
        
        if not tweets_file.exists():
            return data
        
        print("Tweets.csv veri seti yukleniyor...")
        
        try:
            # Disaster-related keyword'ler (tweets.csv'de label yok, keyword-based)
            disaster_keywords = [
                'earthquake', 'deprem', 'disaster', 'afet', 'emergency', 'acil',
                'rescue', 'kurtarma', 'damage', 'hasar', 'collapse', 'çökme',
                'türkiye', 'turkey', 'syria', 'suriye', 'hatay', 'kahramanmaraş'
            ]
            
            with open(tweets_file, 'r', encoding='utf-8') as f:
                reader = csv.DictReader(f)
                count = 0
                for row in reader:
                    text = row.get('content', '').strip()
                    
                    if not text or len(text) < 3:
                        continue
                    
                    cleaned_text = self.clean_text(text)
                    if len(cleaned_text) < 3:
                        continue
                    
                    # Keyword-based label (tweets.csv'de label yok)
                    text_lower = cleaned_text.lower()
                    is_disaster = any(keyword in text_lower for keyword in disaster_keywords)
                    binary_label = 1 if is_disaster else 0
                    
                    data.append((cleaned_text, binary_label, 'tweets_csv'))
                    count += 1
                    
                    # İlk 50,000 tweet'i al
                    if count >= 50000:
                        break
        except Exception as e:
            print(f"  [ERROR] Tweets.csv yuklenirken hata: {e}")
        
        print(f"  [OK] {len(data)} kayit yuklendi")
        return data
    
    def load_turkish_dataset(self) -> List[Tuple[str, int, str]]:
        """turkish_dataset.csv dosyasını yükle"""
        data = []
        turkish_file = self.data_dir / "turkish_dataset.csv"
        
        if not turkish_file.exists():
            return data
        
        print("Turkish dataset yukleniyor...")
        
        try:
            with open(turkish_file, 'r', encoding='utf-8') as f:
                reader = csv.DictReader(f)
                count = 0
                for row in reader:
                    text = row.get('Tweets', '').strip()
                    class_label = row.get('Class', '').strip()
                    
                    if not text or len(text) < 3:
                        continue
                    
                    cleaned_text = self.clean_text(text)
                    if len(cleaned_text) < 3:
                        continue
                    
                    # Class: "1" veya "disaster" = 1, diğerleri = 0
                    binary_label = 1 if class_label.lower() in ['1', 'disaster', 'yes'] else 0
                    
                    data.append((cleaned_text, binary_label, 'turkish_dataset'))
                    count += 1
        except Exception as e:
            print(f"  [ERROR] Turkish dataset yuklenirken hata: {e}")
        
        print(f"  [OK] {len(data)} kayit yuklendi")
        return data
    
    def load_sample_prccd(self) -> List[Tuple[str, int, str]]:
        """sample_prccd.csv dosyasını yükle"""
        data = []
        prccd_file = self.data_dir / "sample_prccd.csv"
        
        if not prccd_file.exists():
            return data
        
        print("Sample PRCCD veri seti yukleniyor...")
        
        try:
            with open(prccd_file, 'r', encoding='utf-8') as f:
                reader = csv.DictReader(f)
                count = 0
                for row in reader:
                    text = row.get('item', '').strip()
                    label = row.get('label', '').strip()
                    
                    if not text or len(text) < 3:
                        continue
                    
                    cleaned_text = self.clean_text(text)
                    if len(cleaned_text) < 3:
                        continue
                    
                    # Label mapping: PRCCD'de "Other Useful Information" = disaster related (1)
                    # Diğer label'lar = not related (0)
                    label_lower = label.lower()
                    if 'useful' in label_lower and 'information' in label_lower:
                        binary_label = 1
                    else:
                        binary_label = 0
                    
                    data.append((cleaned_text, binary_label, 'sample_prccd'))
                    count += 1
        except Exception as e:
            print(f"  [ERROR] Sample PRCCD yuklenirken hata: {e}")
        
        print(f"  [OK] {len(data)} kayit yuklendi")
        return data
    
    def load_socialmedia_disaster_tweets(self) -> List[Tuple[str, int, str]]:
        """socialmedia-disaster-tweets-DFE.csv dosyasını yükle"""
        data = []
        dfe_file = self.data_dir / "socialmedia-disaster-tweets-DFE.csv"
        
        if not dfe_file.exists():
            return data
        
        print("SocialMedia Disaster Tweets DFE yukleniyor...")
        
        try:
            with open(dfe_file, 'r', encoding='utf-8') as f:
                reader = csv.DictReader(f)
                count = 0
                for row in reader:
                    text = row.get('text', '').strip()
                    choose_one = row.get('choose_one', '').strip()
                    
                    if not text or len(text) < 3:
                        continue
                    
                    cleaned_text = self.clean_text(text)
                    if len(cleaned_text) < 3:
                        continue
                    
                    # choose_one: "Yes" = 1, "No" = 0
                    binary_label = 1 if choose_one.lower() == 'yes' else 0
                    
                    data.append((cleaned_text, binary_label, 'socialmedia_disaster_tweets_dfe'))
                    count += 1
        except Exception as e:
            print(f"  [ERROR] SocialMedia Disaster Tweets DFE yuklenirken hata: {e}")
        
        print(f"  [OK] {len(data)} kayit yuklendi")
        return data
    
    def load_disaster_related_100k(self) -> List[Tuple[str, int, str]]:
        """disaster_related_dataset_en_100k.csv veri setini yükle"""
        data = []
        disaster_100k_file = self.data_dir / "disaster_related_dataset_en_100k.csv"
        
        if not disaster_100k_file.exists():
            return data
        
        print("Disaster Related 100k dataset yukleniyor...")
        
        try:
            with open(disaster_100k_file, 'r', encoding='utf-8') as f:
                reader = csv.DictReader(f)
                count = 0
                for row in reader:
                    text = row.get('text', '').strip()
                    label = row.get('label_disaster_related', '').strip()
                    scenario = row.get('scenario', '').strip()
                    
                    if not text or len(text) < 3:
                        continue
                    
                    cleaned_text = self.clean_text(text)
                    if len(cleaned_text) < 3:
                        continue
                    
                    # Label: "1" = disaster related, "0" = not related
                    try:
                        binary_label = int(label) if label else 0
                    except ValueError:
                        binary_label = 0
                    
                    # Scenario'ya göre source adı oluştur
                    source_name = f"disaster_100k_{scenario}" if scenario else "disaster_100k"
                    
                    data.append((cleaned_text, binary_label, source_name))
                    count += 1
        except Exception as e:
            print(f"  [ERROR] Disaster Related 100k yuklenirken hata: {e}")
        
        print(f"  [OK] {len(data)} kayit yuklendi")
        return data
    
    def load_hard_negatives(self) -> List[Tuple[str, int, str]]:
        """hard_negatives.json dosyasını yükle"""
        data = []
        hard_negatives_file = self.data_dir / "hard_negatives.json"
        
        if not hard_negatives_file.exists():
            return data
        
        print("Hard negatives yukleniyor...")
        
        try:
            with open(hard_negatives_file, 'r', encoding='utf-8') as f:
                hard_negatives = json.load(f)
                
                for item in hard_negatives.get('hard_negatives', []):
                    text = item.get('text', '').strip()
                    expected = item.get('expected', False)
                    
                    if not text or len(text) < 3:
                        continue
                    
                    cleaned_text = self.clean_text(text)
                    if len(cleaned_text) < 3:
                        continue
                    
                    # expected: True = disaster related (1), False = not related (0)
                    binary_label = 1 if expected else 0
                    
                    data.append((cleaned_text, binary_label, 'hard_negative_mining'))
        except Exception as e:
            print(f"  [ERROR] Hard negatives yuklenirken hata: {e}")
        
        print(f"  [OK] {len(data)} kayit yuklendi")
        return data
    
    def load_archive_tsv_files(self) -> List[Tuple[str, int, str]]:
        """archive (1) klasöründeki TSV dosyalarını yükle"""
        data = []
        archive_dir = self.data_dir / "archive (1)"
        
        if not archive_dir.exists():
            return data
        
        print("Archive TSV dosyalari yukleniyor...")
        
        # Tüm TSV dosyalarını bul
        tsv_files = list(archive_dir.glob("*.tsv"))
        
        for tsv_file in tsv_files:
            try:
                with open(tsv_file, 'r', encoding='utf-8') as f:
                    reader = csv.DictReader(f, delimiter='\t')
                    count = 0
                    for row in reader:
                        text = row.get('tweet_text', '').strip()
                        label = row.get('label', '').strip()
                        
                        if not text or len(text) < 3:
                            continue
                        
                        cleaned_text = self.clean_text(text)
                        if len(cleaned_text) < 3:
                            continue
                        
                        # Label mapping: disaster-related = 1, not_related = 0
                        # "other_useful_information", "injured_or_dead_people", "donation_needs_or_offers_or_volunteering_services" = 1
                        # "not_related_or_irrelevant" = 0
                        disaster_labels = ['other_useful_information', 'injured_or_dead_people', 
                                          'donation_needs_or_offers_or_volunteering_services']
                        binary_label = 1 if label.lower() in [l.lower() for l in disaster_labels] else 0
                        
                        event_name = tsv_file.stem.replace('_CF_labeled_data', '').replace('_cl_labeled_data', '').replace('_en_CF_labeled_data', '')
                        data.append((cleaned_text, binary_label, f"archive_{event_name}"))
                        count += 1
                
                print(f"  [OK] {tsv_file.name}: {count} kayit")
            except Exception as e:
                print(f"  [ERROR] {tsv_file.name} yuklenirken hata: {e}")
        
        print(f"  [OK] Toplam {len(data)} archive TSV kayit yuklendi")
        return data
    
    def load_crisis_benchmarks(self) -> List[Tuple[str, int, str]]:
        """crisis_datasets_benchmarks_v1.0 klasöründeki TSV dosyalarını yükle (sadece İngilizce)"""
        data = []
        benchmarks_dir = self.data_dir / "crisis_datasets_benchmarks_v1.0" / "data" / "all_data_en"
        
        if not benchmarks_dir.exists():
            return data
        
        print("Crisis Benchmarks dataset yukleniyor...")
        
        # Tüm TSV dosyalarını bul
        tsv_files = list(benchmarks_dir.glob("*.tsv"))
        
        for tsv_file in tsv_files:
            try:
                with open(tsv_file, 'r', encoding='utf-8') as f:
                    reader = csv.DictReader(f, delimiter='\t')
                    count = 0
                    for row in reader:
                        text = row.get('text', '').strip() if row.get('text') else ''
                        lang = row.get('lang', '').strip().lower() if row.get('lang') else ''
                        class_label = row.get('class_label', '').strip() if row.get('class_label') else ''
                        
                        # Sadece İngilizce
                        if lang != 'en':
                            continue
                        
                        if not text or len(text) < 3:
                            continue
                        
                        cleaned_text = self.clean_text(text)
                        if len(cleaned_text) < 3:
                            continue
                        
                        # class_label'a göre binary label
                        # Informativeness task: "informative" = 1, "not_informative" = 0
                        # Humanitarian task: disaster-related categories = 1, "not_humanitarian" = 0
                        class_label_lower = class_label.lower()
                        if 'informative' in class_label_lower and 'not' not in class_label_lower:
                            # Informativeness: informative = disaster related
                            binary_label = 1
                        elif 'not_informative' in class_label_lower or 'not_humanitarian' in class_label_lower:
                            # Not informative/not humanitarian = not related
                            binary_label = 0
                        elif any(cat in class_label_lower for cat in ['affected', 'infrastructure', 'requests', 'displaced', 'rescue', 'injured', 'missing', 'caution', 'sympathy']):
                            # Humanitarian categories = disaster related
                            binary_label = 1
                        else:
                            # Default: not related
                            binary_label = 0
                        
                        event = row.get('event', 'unknown')
                        source = row.get('source', 'unknown')
                        data.append((cleaned_text, binary_label, f"crisis_benchmarks_{event}_{source}"))
                        count += 1
                
                print(f"  [OK] {tsv_file.name}: {count} kayit")
            except Exception as e:
                print(f"  [ERROR] {tsv_file.name} yuklenirken hata: {e}")
        
        print(f"  [OK] Toplam {len(data)} crisis benchmarks kayit yuklendi")
        return data
    
    def load_humaid_dataset(self) -> List[Tuple[str, int, str]]:
        """HumAID_data_events_set1_47K klasöründeki TSV dosyalarını yükle"""
        data = []
        humaid_dir = self.data_dir / "HumAID_data_events_set1_47K" / "events_set1"
        
        if not humaid_dir.exists():
            return data
        
        print("HumAID dataset yukleniyor...")
        
        # Tüm event klasörlerini bul
        event_dirs = [d for d in humaid_dir.iterdir() if d.is_dir()]
        
        for event_dir in event_dirs:
            event_name = event_dir.name
            # Train, dev, test dosyalarını yükle
            for split in ['train', 'dev', 'test']:
                tsv_file = event_dir / f"{event_name}_{split}.tsv"
                
                if not tsv_file.exists():
                    continue
                
                try:
                    with open(tsv_file, 'r', encoding='utf-8') as f:
                        reader = csv.DictReader(f, delimiter='\t')
                        count = 0
                        for row in reader:
                            text = row.get('tweet_text', '').strip()
                            class_label = row.get('class_label', '').strip()
                            
                            if not text or len(text) < 3:
                                continue
                            
                            cleaned_text = self.clean_text(text)
                            if len(cleaned_text) < 3:
                                continue
                            
                            # class_label mapping: disaster-related = 1, not_related = 0
                            # HumAID humanitarian categories = disaster related
                            disaster_labels = [
                                'other_relevant_information', 
                                'displaced_people_and_evacuations',
                                'rescue_volunteering_or_donation_effort', 
                                'injured_or_dead_people',
                                'infrastructure_and_utility_damage', 
                                'missing_or_found_people',
                                'caution_and_advice',
                                'sympathy_and_support',
                                'affected_individual',
                                'infrastructure_and_utilities_damage',
                                'requests_or_needs'
                            ]
                            not_related_labels = ['not_humanitarian']
                            
                            class_label_lower = class_label.lower()
                            if class_label_lower in [l.lower() for l in not_related_labels]:
                                binary_label = 0
                            elif class_label_lower in [l.lower() for l in disaster_labels]:
                                binary_label = 1
                            else:
                                # Default: not related (güvenli tarafta)
                                binary_label = 0
                            
                            data.append((cleaned_text, binary_label, f"humaid_{event_name}_{split}"))
                            count += 1
                    
                    print(f"  [OK] {event_name}_{split}: {count} kayit")
                except Exception as e:
                    print(f"  [ERROR] {event_name}_{split} yuklenirken hata: {e}")
        
        print(f"  [OK] Toplam {len(data)} HumAID kayit yuklendi")
        return data
    
    def load_crisis_csv(self) -> List[Tuple[str, int, str]]:
        """archive (2)/crisis.csv dosyasını yükle"""
        data = []
        crisis_file = self.data_dir / "archive (2)" / "crisis.csv"
        
        if not crisis_file.exists():
            return data
        
        print("Crisis CSV dataset yukleniyor...")
        
        try:
            with open(crisis_file, 'r', encoding='utf-8') as f:
                reader = csv.DictReader(f)
                count = 0
                for row in reader:
                    text = row.get('text', '').strip()
                    relevance_label = row.get('relevance_label', '').strip()
                    
                    if not text or len(text) < 3:
                        continue
                    
                    cleaned_text = self.clean_text(text)
                    if len(cleaned_text) < 3:
                        continue
                    
                    # relevance_label: "1" = disaster related, "0" = not related
                    try:
                        binary_label = int(relevance_label) if relevance_label else 0
                    except ValueError:
                        binary_label = 0
                    
                    crisis_type = row.get('crisis_type', 'unknown')
                    data.append((cleaned_text, binary_label, f"crisis_csv_{crisis_type}"))
                    count += 1
            
            print(f"  [OK] {len(data)} kayit yuklendi")
        except Exception as e:
            print(f"  [ERROR] Crisis CSV yuklenirken hata: {e}")
        
        return data
    
    def process_all_datasets(self) -> Tuple[List[str], List[int], List[str]]:
        """
        Tüm datasetleri yükle ve birleştir
        
        Returns:
            Tuple[List[str], List[int], List[str]]: (texts, labels, sources)
        """
        all_texts = []
        all_labels = []
        all_sources = []
        
        # Tüm datasetleri yükle
        datasets = [
            self.load_crisislext26,
            self.load_tweets_csv,
            self.load_turkish_dataset,
            self.load_sample_prccd,
            self.load_socialmedia_disaster_tweets,
            self.load_disaster_related_100k,
            self.load_hard_negatives,
            self.load_archive_tsv_files,  # Yeni
            self.load_crisis_benchmarks,  # Yeni
            self.load_humaid_dataset,     # Yeni
            self.load_crisis_csv,         # Yeni
        ]
        
        for load_func in datasets:
            try:
                dataset_data = load_func()
                for text, label, source in dataset_data:
                    all_texts.append(text)
                    all_labels.append(label)
                    all_sources.append(source)
            except Exception as e:
                print(f"  [ERROR] Dataset yuklenirken hata: {e}")
        
        print(f"\n{'='*60}")
        print(f"TOPLAM {len(all_texts)} kayit yuklendi")
        print(f"  - Disaster related: {sum(all_labels)}")
        print(f"  - Not related: {len(all_labels) - sum(all_labels)}")
        print(f"{'='*60}")
        
        return all_texts, all_labels, all_sources


if __name__ == "__main__":
    # Test için
    import sys
    import codecs
    # Windows için Unicode desteği
    if sys.platform == 'win32':
        sys.stdout = codecs.getwriter('utf-8')(sys.stdout.buffer, 'strict')
        sys.stderr = codecs.getwriter('utf-8')(sys.stderr.buffer, 'strict')
    
    processor = DataProcessor()
    texts, labels, sources = processor.process_all_datasets()
    print(f"\nToplam {len(texts)} kayit islendi")
