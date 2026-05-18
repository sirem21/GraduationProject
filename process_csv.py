import csv

def to_binary(val):
    v = str(val).strip().lower()
    if v in ['1', '1.0', 'true']:
        return 1
    if v in ['0', '0.0', 'false']:
        return 0
    if v in ['hayır', 'no', 'hiç', 'hiç kullanmadım', 'hiç içmedim', 'bıraktım']:
        return 0
    # For anything that means 'yes' or positive presence
    if 'evet' in v or 'yes' in v or 'var' in v or 'bazen' in v or 'ara sıra' in v or 'her gün' in v or 'sık' in v or 'düzenli' in v:
        return 1
    # For numeric stress (1-10) -> >=5 is 1, else 0
    try:
        if float(val) > 0:
            return 1
    except:
        pass
    return 0

features = [
    "sudden_onset", "paralysed_area", "speech_difficulty", "face_numbness",
    "atrial_fibrillation", "age", "smoking", "family_history_stroke",
    "frequent_urination", "glucose_lvl", "excessive_thrist", "family_history_diabetes",
    "bmi", "frequent_infection", "unintended_weight_loss", "appitide", "blood_p",
    "family_history_hypertension", "nose_bleed", "cholesterol", "chest_pain",
    "physical_activity", "headache", "alcohol", "dizziness", "stress"
]

with open('sample_dataset.csv', 'r', encoding='utf-8') as f_in, open('sample_dataset_cleared.csv', 'w', encoding='utf-8', newline='') as f_out:
    reader = csv.DictReader(f_in)
    writer = csv.DictWriter(f_out, fieldnames=features)
    writer.writeheader()
    
    for row in reader:
        out_row = {}
        out_row['sudden_onset'] = to_binary(row.get('sudden_onset', 0))
        out_row['paralysed_area'] = to_binary(row.get('Vücudun bir tarafında güç kaybı veya felç oldu mu?', 0))
        out_row['speech_difficulty'] = to_binary(row.get('Konuşma güçlüğü veya konuşulanı anlamada zorluk yaşadınız mı?', 0))
        out_row['face_numbness'] = to_binary(row.get('Yüzde ani uyuşma, karıncalanma veya kayma yaşadınız mı?', 0))
        out_row['atrial_fibrillation'] = to_binary(row.get('Kalp ritim bozukluğu (Atriyal Fibrilasyon) teşhisi konuldu mu?', 0))
        out_row['age'] = row.get('Age', 0)
        out_row['smoking'] = to_binary(row.get('Sigara kullanma durumunuz nedir?', 0))
        out_row['family_history_stroke'] = to_binary(row.get('family_history_stroke', 0))
        out_row['frequent_urination'] = to_binary(row.get('frequent_urination', 0))
        out_row['glucose_lvl'] = 100 # Default average
        out_row['excessive_thrist'] = to_binary(row.get('excessive_thrist', 0))
        out_row['family_history_diabetes'] = to_binary(row.get('family_history_diabetes', 0))
        
        # Calculate BMI
        height_cm = float(row.get('Boyunuz (cm)', 170) or 170)
        weight_kg = float(row.get('Kilonuz (kg)', 70) or 70)
        bmi = weight_kg / ((height_cm / 100) ** 2)
        out_row['bmi'] = round(bmi, 2)
        
        out_row['frequent_infection'] = to_binary(row.get('frequent_infection', 0))
        out_row['unintended_weight_loss'] = to_binary(row.get('unexplained_weight_loss', 0))
        out_row['appitide'] = to_binary(row.get('appitide', 0))
        out_row['blood_p'] = 120 # Default average
        out_row['family_history_hypertension'] = to_binary(row.get('family_history_hypertension', 0))
        out_row['nose_bleed'] = to_binary(row.get('nose_bleed', 0))
        out_row['cholesterol'] = to_binary(row.get('Yüksek kolesterolünüz var mı?', 0))
        out_row['chest_pain'] = to_binary(row.get('chest_pain', 0))
        out_row['physical_activity'] = to_binary(row.get('Düzenli fiziksel aktivite yapıyor musunuz?', 0))
        out_row['headache'] = to_binary(row.get('headache', 0))
        out_row['alcohol'] = to_binary(row.get('alcohol', 0))
        out_row['dizziness'] = to_binary(row.get('dizziness', 0))
        
        # Stress level handling > 5 is 1
        try:
            stress_val = int(row.get('Günlük hayatınızdaki genel stres seviyenizi nasıl puanlarsınız?', 0))
            out_row['stress'] = 1 if stress_val >= 5 else 0
        except:
            out_row['stress'] = to_binary(row.get('Günlük hayatınızdaki genel stres seviyenizi nasıl puanlarsınız?', 0))

        writer.writerow(out_row)

print("Veri seti oluşturuldu.")
