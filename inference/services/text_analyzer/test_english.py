"""
Comprehensive English Test for Disaster Relevance Classification
Sadece İngilizce metinlerle test yapar
"""
import sys
import codecs
import json
import pickle
from pathlib import Path
from typing import List, Tuple, Dict

import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression


# Windows için Unicode desteği
if sys.platform == 'win32':
    sys.stdout = codecs.getwriter('utf-8')(sys.stdout.buffer, 'strict')
    sys.stderr = codecs.getwriter('utf-8')(sys.stderr.buffer, 'strict')


class ModelTester:
    """Model test sınıfı"""
    
    def __init__(self):
        """Model ve vectorizer'ı yükle"""
        models_dir = Path(__file__).parent / "models"
        model_path = models_dir / "model.pkl"
        vectorizer_path = models_dir / "vectorizer.pkl"
        
        if not model_path.exists() or not vectorizer_path.exists():
            raise FileNotFoundError("Model dosyalari bulunamadi! Once model egitimi yapin.")
        
        with open(model_path, 'rb') as f:
            self.model = pickle.load(f)
        
        with open(vectorizer_path, 'rb') as f:
            self.vectorizer = pickle.load(f)
        
        print("[OK] Model ve vectorizer yuklendi")
    
    def classify(self, text: str) -> Tuple[bool, float]:
        """
        Metni sınıflandır
        
        Returns:
            Tuple[bool, float]: (is_disaster_related, confidence_score)
        """
        if not text or len(text.strip()) == 0:
            return False, 0.0
        
        # TF-IDF vectorization
        text_vectorized = self.vectorizer.transform([text])
        
        # Prediction
        prediction = self.model.predict(text_vectorized)[0]
        probability = self.model.predict_proba(text_vectorized)[0]
        
        # Class 0: not_related, Class 1: disaster_related
        is_related = bool(prediction == 1)
        confidence = float(probability[1])  # Disaster related probability
        
        return is_related, confidence


def get_test_cases() -> Dict[str, List[Tuple[str, bool]]]:
    """
    İngilizce test case'leri
    
    Returns:
        Dict[str, List[Tuple[str, bool]]]: {category: [(text, expected_label), ...]}
    """
    test_cases = {
        "1. Real Disasters": [
            ("A massive earthquake hit the city center, causing widespread damage.", True),
            ("Wildfire spreading rapidly in the forest, evacuation orders issued.", True),
            ("Flood waters rising in downtown area, residents advised to evacuate.", True),
            ("Tornado touched down near the highway, multiple injuries reported.", True),
            ("Volcano eruption forces thousands to flee their homes.", True),
            ("Hurricane approaching the coast, emergency shelters opening.", True),
            ("Building collapse in downtown, rescue teams on scene.", True),
            ("Tsunami warning issued after major earthquake.", True),
            ("Avalanche buried several houses in the mountain village.", True),
            ("Landslide blocked the main road, traffic diverted.", True),
            ("Severe storm caused power outages across the region.", True),
            ("Explosion at chemical plant, toxic fumes spreading.", True),
            ("Bridge collapse during rush hour, multiple casualties.", True),
            ("Dam burst threatens downstream communities.", True),
            ("Gas leak forces evacuation of entire neighborhood.", True),
            ("Mudslide destroyed homes in the hillside area.", True),
            ("Ice storm knocks down power lines, thousands without electricity.", True),
            ("Drought conditions worsen, water rationing begins.", True),
            ("Heat wave causes multiple deaths, hospitals overwhelmed.", True),
            ("Blizzard traps hundreds of motorists on highway.", True),
        ],
        
        "2. False Alarms / Not Disasters": [
            ("The movie was a disaster at the box office.", False),
            ("My presentation was a complete disaster.", False),
            ("The party was a disaster, nobody showed up.", False),
            ("This is a disaster of epic proportions!", False),
            ("The earthquake drill was successfully completed.", False),
            ("I watched a documentary about earthquakes.", False),
            ("The fire alarm went off during the test.", False),
            ("The project was a financial disaster.", False),
            ("My cooking attempt was a complete disaster.", False),
            ("The meeting was a disaster, nothing got done.", False),
            ("Her hair was a disaster after the haircut.", False),
            ("The exam was a disaster for most students.", False),
            ("The earthquake simulation exercise went well.", False),
            ("I read a book about natural disasters.", False),
            ("The fire safety training was informative.", False),
            ("We practiced evacuation procedures today.", False),
            ("The disaster movie was really scary.", False),
            ("I'm studying disaster management in school.", False),
            ("The fire drill was scheduled for 2 PM.", False),
            ("Emergency preparedness workshop was helpful.", False),
        ],
        
        "3. Figurative Language": [
            ("The flood of emails overwhelmed my inbox.", False),
            ("A storm of criticism hit the company.", False),
            ("The earthquake of emotions shook her.", False),
            ("Work came like a flood today.", False),
            ("The fire in his eyes was intense.", False),
            ("A tsunami of complaints flooded customer service.", False),
            ("The hurricane of change swept through the organization.", False),
            ("Her anger was like a volcano about to erupt.", False),
            ("The avalanche of work buried me this week.", False),
            ("A tornado of activity in the office today.", False),
            ("The floodgates of information opened.", False),
            ("A storm is brewing in the boardroom.", False),
            ("The earthquake of news shook the industry.", False),
            ("Fire in the belly to succeed.", False),
            ("A perfect storm of events led to this.", False),
        ],
        
        "4. Ambiguous Context": [
            ("There was a fire at the restaurant.", True),
            ("Fire broke out in the building.", True),
            ("The fire department responded quickly.", True),
            ("I saw fire in the distance.", True),
            ("Fire alarm system needs maintenance.", False),
            ("Fire destroyed three homes last night.", True),
            ("Firefighters are battling the blaze.", True),
            ("Fire spread to adjacent buildings.", True),
            ("Fire caused extensive damage.", True),
            ("Fire is under control now.", True),
            ("Fire safety regulations updated.", False),
            ("Fire insurance policy covers this.", False),
            ("Fire prevention measures discussed.", False),
            ("Fire drill scheduled for next week.", False),
            ("Fire extinguisher needs refilling.", False),
        ],
        
        "5. Nonsensical Combinations": [
            ("Earthquake became a potato.", False),
            ("Flood turned into a banana.", False),
            ("Fire is now a car.", False),
            ("The disaster became a table.", False),
            ("Hurricane turned into a book.", False),
            ("Earthquake is a chair now.", False),
            ("Flood became a computer.", False),
            ("Fire turned into a phone.", False),
            ("Disaster is now a lamp.", False),
            ("Tornado became a pencil.", False),
            ("Volcano turned into a cup.", False),
            ("Tsunami is now a door.", False),
            ("Avalanche became a window.", False),
            ("Landslide turned into a spoon.", False),
            ("Explosion is now a plate.", False),
        ],
        
        "6. Normal Text": [
            ("I went to the store today.", False),
            ("The weather is nice today.", False),
            ("I had pizza for dinner.", False),
            ("The movie was really good.", False),
            ("I'm learning Python programming.", False),
            ("I love reading books in my free time.", False),
            ("The coffee shop is crowded this morning.", False),
            ("I need to buy groceries for the week.", False),
            ("My favorite color is blue.", False),
            ("The concert was amazing last night.", False),
            ("I enjoy playing tennis on weekends.", False),
            ("The restaurant has great food.", False),
            ("I'm planning a vacation next month.", False),
            ("The book I'm reading is interesting.", False),
            ("I like listening to music while working.", False),
        ],
        
        "7. Indirect Disaster Relevance": [
            ("Emergency services are responding to the scene.", True),
            ("Rescue teams have been deployed.", True),
            ("Evacuation centers are being set up.", True),
            ("Medical supplies are urgently needed.", True),
            ("Search and rescue operations underway.", True),
            ("Ambulances rushing to the location.", True),
            ("Helicopters conducting aerial search.", True),
            ("Temporary shelters opened for displaced families.", True),
            ("Red Cross providing aid to victims.", True),
            ("Emergency hotline established for assistance.", True),
            ("Field hospital set up near the disaster zone.", True),
            ("Volunteers distributing food and water.", True),
            ("Emergency response teams on standby.", True),
            ("Disaster relief fund accepting donations.", True),
            ("Crisis management center activated.", True),
        ],
        
        "8. Short Social Media Posts": [
            ("Earthquake!", True),
            ("Fire!", True),
            ("Flood warning", True),
            ("Evacuate now!", True),
            ("Help needed", True),
            ("Nice weather today", False),
            ("Going shopping", False),
            ("Tornado warning!", True),
            ("Building collapse!", True),
            ("Emergency evacuation", True),
            ("Rescue needed", True),
            ("Disaster area", True),
            ("Having lunch", False),
            ("Watching TV", False),
            ("At the gym", False),
            ("Wildfire spreading", True),
            ("Tsunami alert", True),
            ("Volcano erupting", True),
            ("Just woke up", False),
            ("Coffee time", False),
        ],
        
        "9. Negation / Negative Expressions": [
            ("There is no earthquake here.", False),
            ("No fire in the area.", False),
            ("Not a disaster, just a drill.", False),
            ("This is not an emergency.", False),
            ("No evacuation needed.", False),
            ("There was no flood damage.", False),
            ("No tornado warnings issued.", False),
            ("This is not a real disaster.", False),
            ("No emergency services required.", False),
            ("There is no danger here.", False),
            ("Not an actual earthquake, just a test.", False),
            ("No fire threat in this area.", False),
            ("This is not a real emergency situation.", False),
            ("No evacuation necessary at this time.", False),
            ("There is no disaster happening here.", False),
        ],
        
        "10. Passive Structures": [
            ("There was an earthquake in the region.", True),
            ("A fire broke out in the building.", True),
            ("Flooding occurred in the downtown area.", True),
            ("An explosion happened at the factory.", True),
            ("A collapse was reported at the construction site.", True),
            ("There was a tornado in the county.", True),
            ("A landslide occurred on the mountain road.", True),
            ("Flooding was reported in several neighborhoods.", True),
            ("An avalanche was triggered by heavy snow.", True),
            ("A wildfire broke out in the national park.", True),
            ("There was a gas leak in the building.", True),
            ("A dam failure was reported upstream.", True),
            ("An ice storm hit the region last night.", True),
            ("A heat wave was recorded in the area.", True),
            ("A blizzard was forecast for tomorrow.", True),
        ],
        
        "11. Location Patterns": [
            ("Fire in my neighborhood.", True),
            ("Earthquake at my location.", True),
            ("Flood near my house.", True),
            ("Disaster behind my building.", True),
            ("Emergency right next to me.", True),
            ("Fire in my backyard.", True),
            ("Earthquake near my home.", True),
            ("Flood in my street.", True),
            ("Disaster close to my area.", True),
            ("Emergency in my vicinity.", True),
            ("Fire right behind my house.", True),
            ("Earthquake in my city.", True),
            ("Flood near my workplace.", True),
            ("Disaster in my region.", True),
            ("Emergency at my location.", True),
        ],
        
        "12. Spelling Errors": [
            ("Erthquake hit the city.", True),
            ("Fier spreading rapidly.", True),
            ("Flud waters rising.", True),
            ("Tornadoe touched down.", True),
            ("Volcano erupion forces evacuation.", True),
            ("Erthquak damaged buildings.", True),
            ("Fier broke out in forest.", True),
            ("Flud warning issued.", True),
            ("Tornadoe warning active.", True),
            ("Volcano erupion ongoing.", True),
            ("Erthquake aftershocks felt.", True),
            ("Fier department responding.", True),
            ("Flud evacuation ordered.", True),
            ("Tornadoe watch in effect.", True),
            ("Volcano erupion imminent.", True),
        ],
        
        "13. Mixed Disaster Types": [
            ("Multiple disasters struck the region simultaneously.", True),
            ("Earthquake followed by tsunami devastated the coast.", True),
            ("Wildfire combined with drought created crisis.", True),
            ("Hurricane and flooding affected millions.", True),
            ("Volcano eruption caused ash fall and evacuations.", True),
            ("Earthquake triggered landslides in mountainous areas.", True),
            ("Flooding and mudslides blocked all roads.", True),
            ("Storm surge and high winds damaged infrastructure.", True),
            ("Wildfire spread due to strong winds and dry conditions.", True),
            ("Multiple tornadoes touched down across the state.", True),
        ],
        
        "14. Time-based Expressions": [
            ("The earthquake happened yesterday.", True),
            ("Fire broke out this morning.", True),
            ("Flooding occurred last night.", True),
            ("The disaster happened an hour ago.", True),
            ("Emergency declared just now.", True),
            ("The earthquake will happen tomorrow.", False),
            ("Fire might break out later.", False),
            ("Flooding could occur next week.", False),
            ("The disaster may happen someday.", False),
            ("Emergency might be declared.", False),
        ],
        
        "15. Intensity and Severity": [
            ("Minor earthquake felt in the area.", True),
            ("Massive wildfire burning out of control.", True),
            ("Severe flooding inundated the city.", True),
            ("Devastating tornado destroyed everything.", True),
            ("Catastrophic earthquake caused widespread destruction.", True),
            ("Small fire in the kitchen.", True),
            ("Major disaster declared by authorities.", True),
            ("Extreme weather conditions expected.", True),
            ("Critical situation developing rapidly.", True),
            ("Unprecedented disaster unfolding.", True),
        ],
    }
    
    return test_cases


def run_comprehensive_test():
    """Kapsamlı test çalıştır"""
    print("="*70)
    print("COMPREHENSIVE ENGLISH TEST - DISASTER RELEVANCE CLASSIFICATION")
    print("="*70)
    
    # Model yükle
    try:
        tester = ModelTester()
    except Exception as e:
        print(f"[ERROR] Model yuklenemedi: {e}")
        return
    
    # Test case'leri al
    test_cases = get_test_cases()
    
    # Sonuçları sakla
    results = {}
    total_correct = 0
    total_tests = 0
    
    print("\n" + "="*70)
    print("TEST RESULTS BY CATEGORY")
    print("="*70 + "\n")
    
    # Her kategori için test yap
    for category, cases in test_cases.items():
        category_correct = 0
        category_total = len(cases)
        category_results = []
        
        print(f"{category}:")
        print("-" * 70)
        
        for text, expected in cases:
            is_related, confidence = tester.classify(text)
            predicted = is_related
            
            is_correct = (predicted == expected)
            if is_correct:
                category_correct += 1
                total_correct += 1
            
            total_tests += 1
            
            status = "✓" if is_correct else "✗"
            expected_str = "DISASTER" if expected else "NOT DISASTER"
            predicted_str = "DISASTER" if predicted else "NOT DISASTER"
            
            category_results.append({
                "text": text,
                "expected": expected_str,
                "predicted": predicted_str,
                "confidence": round(confidence, 4),
                "correct": is_correct
            })
            
            print(f"  {status} [{confidence:.3f}] {predicted_str:12} | Expected: {expected_str:12} | {text[:50]}")
        
        accuracy = (category_correct / category_total) * 100 if category_total > 0 else 0
        results[category] = {
            "accuracy": accuracy,
            "correct": category_correct,
            "total": category_total,
            "results": category_results
        }
        
        print(f"  Accuracy: {accuracy:.1f}% ({category_correct}/{category_total})")
        print()
    
    # Genel özet
    overall_accuracy = (total_correct / total_tests) * 100 if total_tests > 0 else 0
    
    print("="*70)
    print("OVERALL SUMMARY")
    print("="*70)
    print(f"Total Tests: {total_tests}")
    print(f"Correct Predictions: {total_correct}")
    print(f"Overall Accuracy: {overall_accuracy:.2f}%")
    print("="*70)
    
    # Kategori bazında özet
    print("\nCategory-wise Accuracy:")
    print("-" * 70)
    for category, result in results.items():
        print(f"  {category:40} {result['accuracy']:6.1f}% ({result['correct']}/{result['total']})")
    
    # Sonuçları JSON'a kaydet
    output_file = Path(__file__).parent / "data" / "accuracy_test_results_english.json"
    output_data = {
        "overall_accuracy": overall_accuracy,
        "total_tests": total_tests,
        "total_correct": total_correct,
        "categories": results
    }
    
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(output_data, f, indent=2, ensure_ascii=False)
    
    print(f"\n[OK] Test sonuclari kaydedildi: {output_file}")
    
    return results


if __name__ == "__main__":
    run_comprehensive_test()

