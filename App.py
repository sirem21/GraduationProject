import os
import joblib
import pandas as pd
import numpy as np
from flask import Flask, render_template, request, jsonify
from datetime import datetime
import holidays

app = Flask(__name__)

# ===== HOSPITAL DISPATCH SYSTEM =====
MAX_DIST_ANKARA = 50.0
tr_holidays = holidays.Turkey()

WEIGHTS = {
    'red':    {'w1': 0.95, 'w2': 0.05},
    'yellow': {'w1': 0.60, 'w2': 0.40},
    'green':  {'w1': 0.25, 'w2': 0.75}
}

class Hospital:
    def __init__(self, id, x, y, capacity, current_occupancy=None, has_ct_scan=True, is_vascular_center=False):
        self.id = id
        self.location = np.array([x, y])
        self.total_capacity = capacity
        self.current_occupancy = current_occupancy if current_occupancy is not None else int(capacity * 0.7)
        self.has_ct_scan = has_ct_scan
        self.is_vascular_center = is_vascular_center

class Patient:
    def __init__(self, id, x, y, risk_level, disease):
        self.id = id
        self.location = np.array([x, y])
        self.risk_level = risk_level
        self.disease = disease

def get_env_factors(date, weather_type='clear'):
    if date in tr_holidays or date.weekday() >= 5:
        holiday_factor = 1.0
    else:
        if 8 <= date.hour <= 10 or 17 <= date.hour <= 19:
            holiday_factor = 2.2
        else:
            holiday_factor = 1.2

    weather_factor_map = {'clear': 1.0, 'rain': 1.3, 'snow': 2.5, 'fog': 1.5}
    weather_factor = weather_factor_map.get(weather_type, 1.0)

    if date in tr_holidays:
        shift_rate = 2.0
    elif date.weekday() >= 5:
        shift_rate = 1.6
    elif date.hour < 8 or date.hour > 18:
        shift_rate = 1.4
    else:
        shift_rate = 1.0

    return holiday_factor, weather_factor, shift_rate

def dispatch_main_algo(patient, hospitals, current_time, weather='clear'):
    holiday_factor, weather_factor, shift_rate = get_env_factors(current_time, weather)
    results = []

    for h in hospitals:
        dist_grid = np.linalg.norm(patient.location - h.location)

        if patient.disease == 'stroke' and not h.has_ct_scan:
            score = np.nan
            status = "CT BROKEN"
        else:
            status = "OK"
            result_dist = dist_grid * holiday_factor * weather_factor
            dist_norm = min(result_dist / MAX_DIST_ANKARA, 2.0)
            occupancy_rate = h.current_occupancy / h.total_capacity
            effective_load = min(occupancy_rate * shift_rate, 1.5)

            is_critical = (patient.disease == 'stroke' or patient.risk_level == 'red')
            if is_critical:
                w = WEIGHTS['red']
            else:
                w = WEIGHTS[patient.risk_level]

            spec_bonus = 0.8 if (patient.disease == 'stroke' and h.is_vascular_center) else 1.0
            score = ((w['w1'] * dist_norm) + (w['w2'] * effective_load)) * spec_bonus

        results.append({
            "id": h.id,
            "score": score,
            "status": status,
            "dist_km": round(dist_grid, 2),
            "occupancy": f"{h.current_occupancy}/{h.total_capacity}",
            "occupancy_pct": round((h.current_occupancy / h.total_capacity) * 100, 1),
            "vascular": h.is_vascular_center,
            "ct_scan": h.has_ct_scan,
            "hospital_obj": h
        })

    df = pd.DataFrame(results).sort_values("score", ascending=True, na_position='last')
    return df, holiday_factor, weather_factor, shift_rate


# --- Load Models and Scalers ---
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODELS_DIR = os.path.join(BASE_DIR, "models_without_confusion")

loaded_model_important   = joblib.load(os.path.join(MODELS_DIR, "model_important.joblib"))
loaded_model_stroke      = joblib.load(os.path.join(MODELS_DIR, "model_stroke.joblib"))
loaded_model_diabetes    = joblib.load(os.path.join(MODELS_DIR, "model_diabetes.joblib"))
loaded_model_hypertension = joblib.load(os.path.join(MODELS_DIR, "model_hypertension.joblib"))

loaded_scaler_main         = joblib.load(os.path.join(MODELS_DIR, "scaler_main.joblib"))
loaded_scaler_stroke       = joblib.load(os.path.join(MODELS_DIR, "scaler_stroke.joblib"))
loaded_scaler_diabetes     = joblib.load(os.path.join(MODELS_DIR, "scaler_diabetes.joblib"))
loaded_scaler_hypertension = joblib.load(os.path.join(MODELS_DIR, "scaler_hypertension.joblib"))

refined_thresholds = {
        'stroke_percentile':       8.1,
        'diabetes_percentile':     9.8,
        'hypertension_percentile': 17.7,
    }

#Risk stratification thresholds were determined using ROC analysis with Youden's J statistic (J = TPR − FPR), which maximizes the balance between sensitivity and specificity. For stroke, the optimal high-risk threshold was identified at 45.73 (TPR=0.990, FPR=0.030).


L2_RISK_THRESHOLDS = {
    "stroke": {
        "medium": 25.64,
        "high":   45.73,
    },
    "diabetes": {
        "medium": 25.02,
        "high":   46.89,
    },
    "hypertension": {
        "medium": 28.19,
        "high":   47.53,
    },
}

SAMPLE_PATIENTS = [
    {
        "id": 1, "name": "Patient Alpha", "location": {"x": 32.78, "y": 39.70},
        "profile": "72 y/o male · abrupt left hemiparesis and facial asymmetry · slurred speech onset within 40 min · known atrial fibrillation history · smoker with high cerebrovascular risk",
        "tag": "High-Risk Stroke",
        "expected_disease": "stroke",
        "l1": {"sudden_onset": 1, "paralysed_area": 1, "speech_difficulty": 1, "face_numbness": 1,
               "frequent_urination": 0, "glucose_lvl": 108, "excessive_thrist": 0, "family_history_diabetes": 0,
               "blood_p": 162, "family_history_hypertension": 1, "nose_bleed": 1, "cholesterol": 1},
        "l2": {"atrial_fibrillation": 1, "age": 72, "smoking": 1, "family_history_stroke": 1}
    },
    {
        "id": 2, "name": "Patient Beta", "location": {"x": 32.90, "y": 39.66},
        "profile": "48 y/o female · fasting glucose around 318 mg/dL with frequent urination · persistent excessive thirst and dry mouth · BMI 34.2 with metabolic-risk pattern · family history positive for diabetes",
        "tag": "Diabetic Profile",
        "expected_disease": "diabetes",
        "l1": {"sudden_onset": 0, "paralysed_area": 0, "speech_difficulty": 0, "face_numbness": 0,
               "frequent_urination": 1, "glucose_lvl": 318, "excessive_thrist": 1, "family_history_diabetes": 1,
               "blood_p": 122, "family_history_hypertension": 0, "nose_bleed": 0, "cholesterol": 1},
        "l2": {"bmi": 34.2, "frequent_infection": 1, "unintended_weight_loss": 1, "appitide": 0}
    },
    {
        "id": 3, "name": "Patient Gamma", "location": {"x": 32.70, "y": 39.90},
        "profile": "55 y/o male · repeated blood pressure spikes near 168 mmHg · recurrent headache and intermittent dizziness · chronic psychosocial stress exposure · positive family history of hypertension",
        "tag": "Hypertensive Case",
        "expected_disease": "hypertension",
        "l1": {"sudden_onset": 0, "paralysed_area": 0, "speech_difficulty": 0, "face_numbness": 0,
               "frequent_urination": 0, "glucose_lvl": 98, "excessive_thrist": 0, "family_history_diabetes": 0,
               "blood_p": 168, "family_history_hypertension": 1, "nose_bleed": 1, "cholesterol": 1},
        "l2": {"chest_pain": 1, "physical_activity": 0, "headache": 1, "alcohol": 1, "dizziness": 1, "stress": 1}
    },
    {
        "id": 4, "name": "Dataset Patient (Stroke)", "location": {"x": 32.8, "y": 39.7},
        "profile": "Dataset-derived case (row 2) · mild unilateral facial numbness without sudden collapse · age 48 with stroke family history signal · no smoking history · mixed baseline risk factors requiring triage review",
        "tag": "Dataset Sample (Stroke)",
        "expected_disease": "stroke",
        "l1": {"sudden_onset": 0, "paralysed_area": 0, "speech_difficulty": 0, "face_numbness": 1,
               "frequent_urination": 0, "glucose_lvl": 100, "excessive_thrist": 0, "family_history_diabetes": 1,
               "blood_p": 120, "family_history_hypertension": 1, "nose_bleed": 0, "cholesterol": 0},
        "l2": {"atrial_fibrillation": 0, "age": 48.0, "smoking": 0, "family_history_stroke": 1,
               "bmi": 27.76, "frequent_infection": 0, "unintended_weight_loss": 0, "appitide": 0,
               "chest_pain": 0, "physical_activity": 0, "headache": 0, "alcohol": 0, "dizziness": 0, "stress": 0}
    },
    {
        "id": 5, "name": "Dataset Patient (Diabetes)", "location": {"x": 32.85, "y": 39.75},
        "profile": "Dataset-derived case (row 3) · glucose near 300+ mg/dL with excessive thirst · BMI around 31.7 indicating obesity-related burden · middle-age profile with limited activity markers · diabetes-focused follow-up recommended",
        "tag": "Dataset Sample (Diabetes)",
        "expected_disease": "diabetes",
        "l1": {"sudden_onset": 0, "paralysed_area": 0, "speech_difficulty": 0, "face_numbness": 0,
            "frequent_urination": 0, "glucose_lvl": 300, "excessive_thrist": 1, "family_history_diabetes": 0,
            "blood_p": 120, "family_history_hypertension": 1, "nose_bleed": 0, "cholesterol": 0},
        "l2": {"atrial_fibrillation": 0, "age": 54.0, "smoking": 0, "family_history_stroke": 0,
            "bmi": 31.65, "frequent_infection": 0, "unintended_weight_loss": 0, "appitide": 0,
            "blood_p": 120, "family_history_hypertension": 1, "nose_bleed": 0, "cholesterol": 0,
            "chest_pain": 1, "physical_activity": 0, "headache": 0, "alcohol": 0, "dizziness": 0, "stress": 0}
    },
    {
        "id": 6, "name": "Dataset Patient (Hypertension)", "location": {"x": 32.9, "y": 39.8},
        "profile": "Dataset-derived case (row 45) · age 44 with smoking exposure and stress-related lifestyle pattern · borderline-high pressure tendencies with family history support · no acute neurologic deficit reported · hypertension pathway likely",
        "tag": "Dataset Sample (Hypertension)",
        "expected_disease": "hypertension",
        "l1": {"sudden_onset": 0, "paralysed_area": 0, "speech_difficulty": 0, "face_numbness": 0,
               "frequent_urination": 0, "glucose_lvl": 100, "excessive_thrist": 0, "family_history_diabetes": 1,
               "blood_p": 120, "family_history_hypertension": 1, "nose_bleed": 0, "cholesterol": 0},
        "l2": {"atrial_fibrillation": 0, "age": 44.0, "smoking": 1, "family_history_stroke": 0,
               "bmi": 25.73, "frequent_infection": 0, "unintended_weight_loss": 0, "appitide": 0,
               "chest_pain": 0, "physical_activity": 1, "headache": 0, "alcohol": 1, "dizziness": 0, "stress": 0}
        },
        {
         "id": 7, "name": "Dataset Patient (Healthy A)", "location": {"x": 32.84, "y": 39.77},
        "profile": "23 y/o male sample · no neurologic warning signs (no sudden onset, no speech deficit, no facial numbness) · blood pressure around 90 mmHg and lean BMI 21.16 · overall low cardiometabolic symptom burden",
         "tag": "Dataset Sample (Healthy)",
         "expected_disease": "healthy",
         "l1": {"sudden_onset": 0, "paralysed_area": 0, "speech_difficulty": 0, "face_numbness": 0,
             "frequent_urination": 0, "glucose_lvl": 150, "excessive_thrist": 0, "family_history_diabetes": 0,
             "blood_p": 90, "family_history_hypertension": 0, "nose_bleed": 0, "cholesterol": 0},
         "l2": {"atrial_fibrillation": 0, "age": 23.0, "smoking": 1, "family_history_stroke": 0,
             "bmi": 21.16, "frequent_infection": 0, "unintended_weight_loss": 0, "appitide": 0,
             "chest_pain": 0, "physical_activity": 0, "headache": 1, "alcohol": 1, "dizziness": 1, "stress": 0}
        },
        {
         "id": 8, "name": "Dataset Patient (Healthy B)", "location": {"x": 32.88, "y": 39.72},
        "profile": "25 y/o male sample · asymptomatic neurologic screening with zero acute stroke signs · glucose ~143 mg/dL and BMI 21.04 with otherwise balanced indicators · no family history flags for stroke/diabetes/hypertension",
         "tag": "Dataset Sample (Healthy)",
         "expected_disease": "healthy",
         "l1": {"sudden_onset": 0, "paralysed_area": 0, "speech_difficulty": 0, "face_numbness": 0,
             "frequent_urination": 0, "glucose_lvl": 143, "excessive_thrist": 0, "family_history_diabetes": 0,
             "blood_p": 141, "family_history_hypertension": 0, "nose_bleed": 0, "cholesterol": 0},
         "l2": {"atrial_fibrillation": 0, "age": 25.0, "smoking": 1, "family_history_stroke": 0,
             "bmi": 21.04, "frequent_infection": 0, "unintended_weight_loss": 0, "appitide": 0,
             "chest_pain": 0, "physical_activity": 0, "headache": 0, "alcohol": 1, "dizziness": 1, "stress": 0}
    }
]

# GPS: Ankara bölgesi — lon ~32.5-33.5, lat ~39.5-40.5
HOSPITALS = [
    {"id": 1, "location": {"x": 32.74, "y": 39.66}, "total_capacity": 1200, "current_occupancy": 950,  "has_ct_scan": True,  "is_vascular_center": True},
    {"id": 2, "location": {"x": 32.86, "y": 39.80}, "total_capacity": 640,  "current_occupancy": 520,  "has_ct_scan": True,  "is_vascular_center": False},
    {"id": 3, "location": {"x": 32.66, "y": 39.60}, "total_capacity": 3700, "current_occupancy": 2850, "has_ct_scan": True,  "is_vascular_center": True},
    {"id": 4, "location": {"x": 32.94, "y": 40.00}, "total_capacity": 430,  "current_occupancy": 340,  "has_ct_scan": False, "is_vascular_center": False},
    {"id": 5, "location": {"x": 32.80, "y": 39.74}, "total_capacity": 900,  "current_occupancy": 720,  "has_ct_scan": True,  "is_vascular_center": True},
    {"id": 6, "location": {"x": 33.00, "y": 39.90}, "total_capacity": 350,  "current_occupancy": 210,  "has_ct_scan": False, "is_vascular_center": False},
    {"id": 7, "location": {"x": 32.70, "y": 39.62}, "total_capacity": 480,  "current_occupancy": 340,  "has_ct_scan": True,  "is_vascular_center": False},
]


def prepare_patient_data(features, l1_features=None):
    if l1_features is None:
        l1_features = {}

    all_f = {**l1_features, **features}

    # Yazım hatası düzeltmeleri (model eski yazımı bekliyor)
    if 'excessive_thirst' in all_f:
        all_f['excessive_thrist'] = all_f.pop('excessive_thirst')
    if 'appetite_loss' in all_f:
        all_f['appitide'] = all_f.pop('appetite_loss')

    return pd.DataFrame([all_f])


def _scale_and_predict(model, scaler, patient_df):
    """Eksik sütunları doldur, scale et, tahmin döndür."""
    df = patient_df.copy()
    for col in model.feature_names_in_:
        if col not in df.columns:
            df[col] = 0
    scaler_cols = list(scaler.feature_names_in_)
    df[scaler_cols] = scaler.transform(df[scaler_cols])
    return float(model.predict(df[model.feature_names_in_])[0])


def _assign_risk_level(score, disease):
    """L2 skoru → low/medium/high."""
    t = L2_RISK_THRESHOLDS.get(disease, {"medium": 26.16, "high": 40.88})
    if score >= t["high"]:
        return "high"
    elif score >= t["medium"]:
        return "medium"
    else:
        return "low"


def l1_model_predict(features):
    """
    Ana (triage) modeli çalıştır.
    - L1 skoru refined_thresholds eşiğini geçen hastalıkları döndür.
    - predicted_disease: eşiği geçenler arasında en yüksek skoru olan hastalık;
      hiçbiri geçmiyorsa yine de en yüksek skoru olan hastalık (fallback).
    """
    patient_df = prepare_patient_data(features)

    for col in loaded_model_important.feature_names_in_:
        if col not in patient_df.columns:
            patient_df[col] = 0

    df_main = patient_df.copy()
    num_cols = list(loaded_scaler_main.feature_names_in_)
    df_main[num_cols] = loaded_scaler_main.transform(df_main[num_cols])
    preds = loaded_model_important.predict(df_main[loaded_model_important.feature_names_in_])[0]

    scores = {
        "stroke":       float(np.clip(preds[0], 0, 99)),
        "diabetes":     float(np.clip(preds[1], 0, 99)),
        "hypertension": float(np.clip(preds[2], 0, 99)),
    }

    # refined_thresholds: L1'den L2'ye yönlendirme eşiği
    # joblib dict → anahtarlar 'stroke_percentile' formatında olabilir
    def _get_thresh(disease_key):
        long_key = f"{disease_key}_percentile"
        if isinstance(refined_thresholds, dict):
            return refined_thresholds.get(long_key,
                   refined_thresholds.get(disease_key, 5.0))
        return 5.0  # fallback

    l1_thresholds = {
        'stroke':       _get_thresh('stroke'),
        'diabetes':     _get_thresh('diabetes'),
        'hypertension': _get_thresh('hypertension'),
    }

    exceeds = {k: v for k, v in scores.items() if v >= l1_thresholds[k]}

    # Eğer herhangi bir hastalık L1 eşiğini geçmiyorsa
    # frontend'in bunu açıkça göstermesi için `predicted_disease` None döndür.
    if not exceeds:
        return {
            "scores": scores,
            "predicted_disease": None,
            "needs_l2": False,                # L2 gerekli değil
            "l1_thresholds": l1_thresholds,   # debug amaçlı
        }

    predicted_disease = max(exceeds, key=exceeds.get)

    return {
        "scores": scores,
        "predicted_disease": predicted_disease,
        "needs_l2": True,                   # en az birisi eşiği geçti
        "l1_thresholds": l1_thresholds,     # debug amaçlı
    }


def l2_model_predict(disease, features, l1_features=None):
    """
    Hastalığa özgü detay modelini çalıştır → risk skoru + seviye.
    """
    patient_df = prepare_patient_data(features, l1_features)

    try:
        if disease == "stroke":
            risk_score = _scale_and_predict(loaded_model_stroke, loaded_scaler_stroke, patient_df)
        elif disease == "diabetes":
            risk_score = _scale_and_predict(loaded_model_diabetes, loaded_scaler_diabetes, patient_df)
        elif disease == "hypertension":
            risk_score = _scale_and_predict(loaded_model_hypertension, loaded_scaler_hypertension, patient_df)
        else:
            risk_score = 0.0
    except Exception as e:
        print(f"L2 Model Error [{disease}]:", e)
        risk_score = 0.0

    risk_score = float(np.clip(risk_score, 0, 100))
    risk_level = _assign_risk_level(risk_score, disease)

    return {"risk_score": round(risk_score, 1), "risk_level": risk_level}


def rank_hospitals(disease, risk_level, weather='clear', datetime_obj=None,
                   patient_lon=32.86, patient_lat=39.93):
    """
    Dispatch algoritmasıyla hastaneleri sırala.
    Koordinatlar: lon (x ekseni ~32.x), lat (y ekseni ~39.x)
    """
    if datetime_obj is None:
        datetime_obj = datetime.now()

    risk_mapping = {'high': 'red', 'medium': 'yellow', 'low': 'green'}
    risk_color = risk_mapping.get(risk_level, 'yellow')

    # Patient: x=lon, y=lat
    patient = Patient("current", patient_lon, patient_lat, risk_color, disease)

    hospitals_objs = [
        Hospital(h['id'],
                 h['location']['x'],   # lon
                 h['location']['y'],   # lat
                 h['total_capacity'],
                 h['current_occupancy'],
                 h['has_ct_scan'],
                 h['is_vascular_center'])
        for h in HOSPITALS
    ]

    df_results, _, _, _ = dispatch_main_algo(patient, hospitals_objs, datetime_obj, weather)
    valid_results = df_results[df_results['status'] == 'OK'].head(5)

    ranked = []
    for _, row in valid_results.iterrows():
        h_obj = row['hospital_obj']
        ranked.append({
            'id': h_obj.id,
            'location': {
                'x': round(float(h_obj.location[0]), 4),
                'y': round(float(h_obj.location[1]), 4),
            },
            'total_capacity':    h_obj.total_capacity,
            'current_occupancy': h_obj.current_occupancy,
            'has_ct_scan':       h_obj.has_ct_scan,
            'is_vascular_center': h_obj.is_vascular_center,
            # score: 0-100 arası normalize (orijinal 0-2 arasında olabilir)
            'score':        round(float(row['score']) * 100, 1) if pd.notnull(row['score']) else 0,
            'dist_km':      row['dist_km'],
            'occupancy_pct': row['occupancy_pct'],
        })

    # Hiç geçerli hastane yoksa (stroke + CT yok gibi) en yakını fallback
    if not ranked:
        fallback = df_results.dropna(subset=['score']).sort_values('dist_km')
        if fallback.empty:
            fallback = df_results.sort_values('dist_km')
        if not fallback.empty:
            row = fallback.iloc[0]
            h_obj = row['hospital_obj']
            ranked = [{
                'id': h_obj.id,
                'location': {'x': round(float(h_obj.location[0]), 4), 'y': round(float(h_obj.location[1]), 4)},
                'total_capacity':    h_obj.total_capacity,
                'current_occupancy': h_obj.current_occupancy,
                'has_ct_scan':       h_obj.has_ct_scan,
                'is_vascular_center': h_obj.is_vascular_center,
                'score': 0,
                'dist_km': row['dist_km'],
                'occupancy_pct': row['occupancy_pct'],
            }]

    return ranked


# ===== ROUTES =====

@app.route("/")
def index():
    return render_template("index.html")


@app.route("/api/samples")
def get_samples():
    return jsonify(SAMPLE_PATIENTS)


@app.route("/api/l1_predict", methods=["POST"])
def l1_predict():
    data = request.json
    return jsonify(l1_model_predict(data.get("features", {})))


@app.route("/api/l2_predict", methods=["POST"])
def l2_predict():
    data = request.json
    result = l2_model_predict(
        data.get("disease", "stroke"),
        data.get("features", {}),
        data.get("l1_features", data.get("features", {}))
    )
    return jsonify(result)


@app.route("/api/hospitals", methods=["POST"])
def get_hospitals():
    data = request.json
    disease    = data.get("disease", "stroke")
    risk_level = data.get("risk_level", "medium")
    weather    = data.get("weather", "clear")
    datetime_str = data.get("datetime", None)

    # Koordinat isimlendirmesi: API'den lon/lat veya x/y gelebilir
    # Ankara: lon ~32.86, lat ~39.93
    patient_lon = data.get("patient_lon", data.get("patient_x", 32.86))
    patient_lat = data.get("patient_lat", data.get("patient_y", 39.93))

    datetime_obj = datetime.now()
    if datetime_str:
        try:
            datetime_obj = datetime.fromisoformat(datetime_str)
        except Exception:
            pass

    return jsonify(rank_hospitals(disease, risk_level, weather, datetime_obj, patient_lon, patient_lat))


@app.route("/api/debug", methods=["POST"])
def debug_predict():
    """
    Geliştirme sırasında ham tahminleri görmek için debug endpoint.
    Production'da kaldırabilirsin.
    """
    data = request.json
    features = data.get("features", {})
    patient_df = prepare_patient_data(features)

    for col in loaded_model_important.feature_names_in_:
        if col not in patient_df.columns:
            patient_df[col] = 0

    num_cols = list(loaded_scaler_main.feature_names_in_)
    raw_vals = patient_df[num_cols].to_dict(orient='records')[0]

    df_scaled = patient_df.copy()
    df_scaled[num_cols] = loaded_scaler_main.transform(df_scaled[num_cols])
    scaled_vals = df_scaled[num_cols].to_dict(orient='records')[0]

    preds = loaded_model_important.predict(df_scaled[loaded_model_important.feature_names_in_])[0]

    return jsonify({
        "raw_inputs":  raw_vals,
        "scaled_inputs": scaled_vals,
        "raw_predictions": {
            "stroke":       round(float(preds[0]), 3),
            "diabetes":     round(float(preds[1]), 3),
            "hypertension": round(float(preds[2]), 3),
        },
        "refined_thresholds_loaded": refined_thresholds if isinstance(refined_thresholds, dict) else str(refined_thresholds),
    })


if __name__ == "__main__":
    app.run(debug=True, port=5000)