# -*- coding: utf-8 -*-
import pandas as pd
from App import l1_model_predict, l2_model_predict

def generate_report():
    print("Loading dataset...")
    df = pd.read_csv("sample_dataset_cleared.csv")
    df_raw = pd.read_csv("sample_dataset.csv")
    
    results = []
    
    total = len(df)
    print(f"Processing {total} records...")
    
    for index, row in df.iterrows():
        features = row.to_dict()
        raw_row = df_raw.iloc[index]
        
        # Gercek verilerden hastalik tespiti (sample_dataset.csv)
        gercek_hastaliklar = []
        if str(raw_row.get('Stroke', '')) == 'Evet':
            gercek_hastaliklar.append('stroke')
        if str(raw_row.get('Diabetes_012', '')) in ['Diyabet', 'Prediyabet (Gizli Şeker)']:
            gercek_hastaliklar.append('diabetes')
        if str(raw_row.get('Has_Hypertension', '')) == 'Evet':
            gercek_hastaliklar.append('hypertension')
            
        if not gercek_hastaliklar:
            gercek_durum = "yok"
        else:
            gercek_durum = ", ".join(gercek_hastaliklar)
        
        l1_result = l1_model_predict(features)
        l1_scores = l1_result["scores"]
        predicted_disease_kaba = l1_result["predicted_disease"]

        l1_thresholds = {
            'stroke': 8.1,
            'diabetes': 9.8,
            'hypertension': 17.7
        }

        # Debug: ilk 5 hastayı yazdır
        if index < 5:
            print(f"\n--- Hasta {index+1} ---")
            print(f"predicted_disease_kaba: {predicted_disease_kaba}")
            print(f"L1 scores: {l1_scores}")

        if predicted_disease_kaba and l1_scores[predicted_disease_kaba] >= l1_thresholds.get(predicted_disease_kaba, 0):
            predicted_disease = predicted_disease_kaba
            l2_result = l2_model_predict(predicted_disease, features)
            risk_level = l2_result["risk_level"].upper()
            risk_score = l2_result["risk_score"]
            s_score = l1_scores.get('stroke', 0)
            d_score = l1_scores.get('diabetes', 0)
            h_score = l1_scores.get('hypertension', 0)
        else:
            predicted_disease = "YOK"
            risk_level = "RİSK YOK"
            risk_score = 0.0
            s_score = 0.0
            d_score = 0.0
            h_score = 0.0

        results.append({
            "Hasta No": index + 1,
            "Gerçek Hastalık": gercek_durum.upper(),
            "Tahmin Edilen": predicted_disease.upper(),
            "L1 İnme (%)": round(s_score, 1),
            "L1 Diyabet (%)": round(d_score, 1),
            "L1 Tansiyon (%)": round(h_score, 1),
            "L2 Risk Puanı": round(risk_score, 1),
            "Risk Seviyesi": risk_level
        })
        
        if (index + 1) % 100 == 0:
            print(f"Processed {index + 1}/{total} records...")
            
    results_df = pd.DataFrame(results)
    
    print("\n" + "="*40)
    print("             REPORT SUMMARY")
    print("="*40)
    
    print("\n1. Disease Distribution:")
    print(results_df["Tahmin Edilen"].value_counts().to_string())
    
    print("\n2. Risk Level Distribution:")
    print(results_df["Risk Seviyesi"].value_counts().to_string())
    
    print("\n3. Risk Level by Disease:")
    crosstab = pd.crosstab(results_df["Tahmin Edilen"], results_df["Risk Seviyesi"])
    print(crosstab.to_string())
    
    print("\n4. Average Risk Score by Disease:")
    avg_scores = results_df.groupby("Tahmin Edilen")["L2 Risk Puanı"].mean().round(2)
    print(avg_scores.to_string())
    
    output_file = "evaluation_report.csv"
    results_df.to_csv(output_file, index=False, encoding="utf-8-sig")
    print(f"\n[+] Detailed evaluation report saved to {output_file}")

if __name__ == "__main__":
    generate_report()
