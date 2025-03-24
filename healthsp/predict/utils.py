import pandas as pd
import numpy as np
from sklearn.naive_bayes import GaussianNB

# Load dataset with tab delimiter
data = pd.read_csv("Training.csv", sep='\t')  # Use tab as separator
data.columns = data.columns.str.strip()  # Clean column names
x = data.drop('prognosis', axis=1)
y = data['prognosis']


# Train the Naive Bayes model
gnb = GaussianNB()
gnb.fit(x, y)


# Symptom list and descriptions
symptom_data = {
    'itching': 'Persistent urge to scratch the skin.',
    'skin_rash': 'Red, inflamed patches on the skin.',
    'nodal_skin_eruptions': 'Small, raised bumps on the skin.',
    'continuous_sneezing': 'Uncontrollable, repeated sneezing.',
    'shivering': 'Involuntary trembling or shaking.',
    'chills': 'Feeling cold with goosebumps.',
    'joint_pain': 'Ache or stiffness in joints.',
    'stomach_pain': 'Discomfort or ache in the abdomen.',
    'acidity': 'Burning sensation in the chest or stomach.',
    'ulcers_on_tongue': 'Painful sores on the tongue.',
    'muscle_wasting': 'Loss of muscle mass or strength.',
    'vomiting': 'Forceful expulsion of stomach contents.',
    'burning_micturition': 'Pain or burning during urination.',
    'spotting_ urination': 'Blood or spots in urine.',
    'fatigue': 'Extreme tiredness or lack of energy.',
    'weight_gain': 'Unexplained increase in body weight.',
    'anxiety': 'Feelings of nervousness or worry.',
    'cold_hands_and_feets': 'Unusually cold hands and feet.',
    'mood_swings': 'Rapid changes in emotional state.',
    'weight_loss': 'Unexplained decrease in body weight.',
    'restlessness': 'Inability to relax or stay still.',
    'lethargy': 'Sluggishness or lack of motivation.',
    'patches_in_throat': 'White or red spots in the throat.',
    'irregular_sugar_level': 'Fluctuating blood sugar levels.',
    'cough': 'Sudden expulsion of air from the lungs.',
    'high_fever': 'Elevated body temperature above normal.',
    'sunken_eyes': 'Eyes appearing deep-set or hollow.',
    'breathlessness': 'Difficulty breathing or shortness of breath.',
    'sweating': 'Excessive perspiration.',
    'dehydration': 'Lack of adequate body fluids.',
    'indigestion': 'Discomfort or pain during digestion.',
    'headache': 'Pain in the head or neck area.',
    'yellowish_skin': 'Yellow tint to the skin.',
    'dark_urine': 'Urine appearing darker than usual.',
    'nausea': 'Feeling of sickness or urge to vomit.',
    'loss_of_appetite': 'Reduced desire to eat.',
    'pain_behind_the_eyes': 'Ache or pressure behind the eyes.',
    'back_pain': 'Discomfort in the back region.',
    'constipation': 'Difficulty or infrequent bowel movements.',
    'abdominal_pain': 'Pain in the stomach or belly area.',
    'diarrhoea': 'Frequent, loose, or watery stools.',
    'mild_fever': 'Slightly elevated body temperature.',
    'yellow_urine': 'Yellowish discoloration of urine.',
    'yellowing_of_eyes': 'Yellow tint in the whites of the eyes.',
    'acute_liver_failure': 'Sudden loss of liver function.',
    'fluid_overload': 'Excess fluid accumulation in the body.',
    'swelling_of_stomach': 'Bloating or enlargement of the abdomen.',
    'swelled_lymph_nodes': 'Enlarged glands under the skin.',
    'malaise': 'General feeling of discomfort or illness.',
    'blurred_and_distorted_vision': 'Unclear or warped vision.',
    'phlegm': 'Thick mucus in the throat or lungs.',
    'throat_irritation': 'Scratchy or sore feeling in the throat.',
    'redness_of_eyes': 'Red or bloodshot eyes.',
    'sinus_pressure': 'Pressure or pain in the sinus areas.',
    'runny_nose': 'Excess nasal discharge.',
    'congestion': 'Blocked nose or chest.',
    'chest_pain': 'Discomfort or pain in the chest.',
    'weakness_in_limbs': 'Reduced strength in arms or legs.',
    'fast_heart_rate': 'Rapid or irregular heartbeat.',
    'pain_during_bowel_movements': 'Discomfort during defecation.',
    'pain_in_anal_region': 'Pain around the anus.',
    'bloody_stool': 'Blood in the feces.',
    'irritation_in_anus': 'Itching or discomfort around the anus.',
    'neck_pain': 'Ache or stiffness in the neck.',
    'dizziness': 'Feeling lightheaded or unsteady.',
    'cramps': 'Sudden, painful muscle contractions.',
    'bruising': 'Discoloration from broken blood vessels.',
    'obesity': 'Excessive body fat accumulation.',
    'swollen_legs': 'Enlargement of the legs due to fluid.',
    'swollen_blood_vessels': 'Visible, enlarged veins.',
    'puffy_face_and_eyes': 'Swelling around the face and eyes.',
    'enlarged_thyroid': 'Swollen thyroid gland in the neck.',
    'brittle_nails': 'Fragile or easily broken nails.',
    'swollen_extremeties': 'Swelling in hands, feet, or limbs.',
    'excessive_hunger': 'Unusually strong appetite.',
    'extra_marital_contacts': 'History of multiple sexual partners.',
    'drying_and_tingling_lips': 'Dry, prickling sensation on lips.',
    'slurred_speech': 'Difficulty speaking clearly.',
    'knee_pain': 'Ache or stiffness in the knees.',
    'hip_joint_pain': 'Discomfort in the hip area.',
    'muscle_weakness': 'Reduced muscle strength.',
    'stiff_neck': 'Limited neck movement or rigidity.',
    'swelling_joints': 'Enlarged or painful joints.',
    'movement_stiffness': 'Difficulty moving joints or limbs.',
    'spinning_movements': 'Sensation of spinning or vertigo.',
    'loss_of_balance': 'Unsteadiness or difficulty standing.',
    'unsteadiness': 'Lack of stability while moving.',
    'weakness_of_one_body_side': 'Weakness on one side of the body.',
    'loss_of_smell': 'Reduced or absent sense of smell.',
    'bladder_discomfort': 'Pain or pressure in the bladder.',
    'foul_smell_of urine': 'Unpleasant odor in urine.',
    'continuous_feel_of_urine': 'Persistent urge to urinate.',
    'passage_of_gases': 'Excessive flatulence.',
    'internal_itching': 'Itching sensation inside the body.',
    'toxic_look_(typhos)': 'Feverish, unwell appearance.',
    'depression': 'Persistent sadness or low mood.',
    'irritability': 'Easily annoyed or agitated state.',
    'muscle_pain': 'Ache or soreness in muscles.',
    'altered_sensorium': 'Changes in mental awareness.',
    'red_spots_over_body': 'Red marks or dots on the skin.',
    'belly_pain': 'Pain in the lower abdomen.',
    'abnormal_menstruation': 'Irregular or painful periods.',
    'dischromic _patches': 'Discolored skin patches.',
    'watering_from_eyes': 'Excessive tearing.',
    'increased_appetite': 'Heightened desire to eat.',
    'polyuria': 'Frequent urination.',
    'family_history': 'Genetic predisposition to illness.',
    'mucoid_sputum': 'Thick, sticky mucus coughed up.',
    'rusty_sputum': 'Brownish or blood-tinged mucus.',
    'lack_of_concentration': 'Difficulty focusing or thinking.',
    'visual_disturbances': 'Blurred or altered vision.',
    'receiving_blood_transfusion': 'History of blood transfusion.',
    'receiving_unsterile_injections': 'Exposure to unclean needles.',
    'coma': 'State of unconsciousness.',
    'stomach_bleeding': 'Blood in vomit or stools.',
    'distention_of_abdomen': 'Swollen or bloated belly.',
    'history_of_alcohol_consumption': 'Past excessive alcohol use.',
    'fluid_overload': 'Excess fluid in the body.',
    'blood_in_sputum': 'Blood coughed up in mucus.',
    'prominent_veins_on_calf': 'Visible veins on the lower leg.',
    'palpitations': 'Rapid or irregular heartbeats.',
    'painful_walking': 'Discomfort while walking.',
    'pus_filled_pimples': 'Pimples with pus.',
    'blackheads': 'Dark, clogged pores on the skin.',
    'scurring': 'Dry, flaky skin.',
    'skin_peeling': 'Shedding of outer skin layers.',
    'silver_like_dusting': 'Silvery scales on the skin.',
    'small_dents_in_nails': 'Pits or depressions in nails.',
    'inflammatory_nails': 'Red, swollen nail beds.',
    'blister': 'Fluid-filled bump on the skin.',
    'red_sore_around_nose': 'Painful red area near the nose.',
    'yellow_crust_ooze': 'Yellowish discharge from skin sores.'
}

# List of symptoms (ensure this matches your dataset columns)
l1 = ['itching', 'skin_rash', 'nodal_skin_eruptions', 'continuous_sneezing', 'shivering', 'chills', 'joint_pain',
      'stomach_pain', 'acidity', 'ulcers_on_tongue', 'muscle_wasting', 'vomiting', 'burning_micturition', 'spotting_ urination', 'fatigue',
      'weight_gain', 'anxiety', 'cold_hands_and_feets', 'mood_swings', 'weight_loss', 'restlessness', 'lethargy', 'patches_in_throat',
      'irregular_sugar_level', 'cough', 'high_fever', 'sunken_eyes', 'breathlessness', 'sweating', 'dehydration', 'indigestion',
      'headache', 'yellowish_skin', 'dark_urine', 'nausea', 'loss_of_appetite', 'pain_behind_the_eyes', 'back_pain', 'constipation',
      'abdominal_pain', 'diarrhoea', 'mild_fever', 'yellow_urine', 'yellowing_of_eyes', 'acute_liver_failure', 'fluid_overload',
      'swelling_of_stomach', 'swelled_lymph_nodes', 'malaise', 'blurred_and_distorted_vision', 'phlegm', 'throat_irritation',
      'redness_of_eyes', 'sinus_pressure', 'runny_nose', 'congestion', 'chest_pain', 'weakness_in_limbs', 'fast_heart_rate',
      'pain_during_bowel_movements', 'pain_in_anal_region', 'bloody_stool', 'irritation_in_anus', 'neck_pain', 'dizziness', 'cramps',
      'bruising', 'obesity', 'swollen_legs', 'swollen_blood_vessels', 'puffy_face_and_eyes', 'enlarged_thyroid', 'brittle_nails',
      'swollen_extremeties', 'excessive_hunger', 'extra_marital_contacts', 'drying_and_tingling_lips', 'slurred_speech', 'knee_pain', 'hip_joint_pain',
      'muscle_weakness', 'stiff_neck', 'swelling_joints', 'movement_stiffness', 'spinning_movements', 'loss_of_balance', 'unsteadiness', 'weakness_of_one_body_side',
      'loss_of_smell', 'bladder_discomfort', 'foul_smell_of urine', 'continuous_feel_of_urine', 'passage_of_gases', 'internal_itching', 'toxic_look_(typhos)',
      'depression', 'irritability', 'muscle_pain', 'altered_sensorium', 'red_spots_over_body', 'belly_pain', 'abnormal_menstruation', 'dischromic _patches',
      'watering_from_eyes', 'increased_appetite', 'polyuria', 'family_history', 'mucoid_sputum', 'rusty_sputum', 'lack_of_concentration', 'visual_disturbances',
      'receiving_blood_transfusion', 'receiving_unsterile_injections', 'coma', 'stomach_bleeding', 'distention_of_abdomen', 'history_of_alcohol_consumption',
      'fluid_overload', 'blood_in_sputum', 'prominent_veins_on_calf', 'palpitations', 'painful_walking', 'pus_filled_pimples', 'blackheads', 'scurring', 'skin_peeling',
      'silver_like_dusting', 'small_dents_in_nails', 'inflammatory_nails', 'blister', 'red_sore_around_nose', 'yellow_crust_ooze']



# Disease-to-department mapping
disease_department_mapping = {
    "Fungal infection": "Dermatology",
    "Allergy": "Immunology/Allergy Specialist",
    "GERD": "Gastroenterology",
    "Chronic cholestasis": "Hepatology/Gastroenterology",
    "Drug Reaction": "Dermatology/Immunology",
    "Peptic ulcer disease": "Gastroenterology",
    "AIDS": "Infectious Diseases",
    "Diabetes": "Endocrinology",
    "Gastroenteritis": "Gastroenterology",
    "Bronchial Asthma": "Pulmonology/Immunology",
    "Hypertension": "Cardiology",
    "Migraine": "Neurology",
    "Cervical spondylosis": "Orthopedics/Neurology",
    "Paralysis (brain hemorrhage)": "Neurology/Neurosurgery",
    "Jaundice": "Hepatology/Gastroenterology",
    "Malaria": "Infectious Diseases",
    "Chicken pox": "Infectious Diseases/Dermatology",
    "Dengue": "Infectious Diseases",
    "Typhoid": "Infectious Diseases",
    "Hepatitis A": "Hepatology",
    "Hepatitis B": "Hepatology",
    "Hepatitis C": "Hepatology",
    "Hepatitis D": "Hepatology",
    "Hepatitis E": "Hepatology",
    "Alcoholic hepatitis": "Hepatology",
    "Tuberculosis": "Pulmonology/Infectious Diseases",
    "Common Cold": "General Medicine",
    "Pneumonia": "Pulmonology",
    "Dimorphic hemorrhoids (piles)": "Proctology/Surgery",
    "Heart attack": "Cardiology",
    "Varicose veins": "Vascular Surgery",
    "Hypothyroidism": "Endocrinology",
    "Hyperthyroidism": "Endocrinology",
    "Hypoglycemia": "Endocrinology",
    "Osteoarthritis": "Orthopedics",
    "Arthritis": "Rheumatology",
    "(Vertigo) Paroxysmal Positional Vertigo": "Neurology/Ear, Nose, Throat (ENT)",
    "Acne": "Dermatology",
    "Urinary tract infection": "Urology",
    "Psoriasis": "Dermatology",
    "Impetigo": "Dermatology"
}


disease_precaution_mapping = {
    "Fungal infection": "Keep skin dry and clean, avoid sharing personal items, use antifungal creams.",
    "Allergy": "Avoid allergens (e.g., pollen, dust), use antihistamines, keep windows closed during high pollen seasons.",
    "GERD": "Avoid spicy/acidic foods, eat smaller meals, elevate head while sleeping.",
    "Chronic cholestasis": "Limit fatty foods, avoid alcohol, follow a low-fat diet.",
    "Drug Reaction": "Discontinue suspected medication, consult a doctor immediately, avoid self-medicating.",
    "Peptic ulcer disease": "Avoid spicy foods and alcohol, eat smaller meals, reduce stress.",
    "AIDS": "Practice safe sex, avoid sharing needles, adhere to antiretroviral therapy.",
    "Diabetes": "Monitor blood sugar, maintain a balanced diet, exercise regularly.",
    "Gastroenteritis": "Stay hydrated, avoid contaminated food/water, wash hands frequently.",
    "Bronchial Asthma": "Avoid triggers (e.g., smoke, dust), use inhalers as prescribed, keep air clean.",
    "Hypertension": "Reduce salt intake, manage stress, exercise daily.",
    "Migraine": "Avoid triggers (e.g., loud noise, bright lights), stay hydrated, rest in a quiet place.",
    "Cervical spondylosis": "Maintain good posture, avoid heavy lifting, perform neck exercises.",
    "Paralysis (brain hemorrhage)": "Monitor blood pressure, avoid head injuries, follow rehabilitation therapy.",
    "Jaundice": "Avoid alcohol, eat a balanced diet, stay hydrated.",
    "Malaria": "Use mosquito nets, apply insect repellent, take antimalarial medication if prescribed.",
    "Chicken pox": "Avoid scratching, keep skin clean, isolate to prevent spread.",
    "Dengue": "Use mosquito repellent, stay hydrated, avoid aspirin for fever.",
    "Typhoid": "Drink clean water, avoid raw foods, wash hands frequently.",
    "Hepatitis A": "Wash hands before eating, avoid contaminated water, get vaccinated if possible.",
    "Hepatitis B": "Practice safe sex, avoid sharing needles, get vaccinated.",
    "Hepatitis C": "Avoid sharing personal items (e.g., razors), practice safe injections, monitor liver health.",
    "Hepatitis D": "Prevent Hepatitis B (co-infection risk), avoid risky behaviors, consult a specialist.",
    "Hepatitis E": "Drink boiled water, avoid raw shellfish, maintain hygiene.",
    "Alcoholic hepatitis": "Stop alcohol consumption, eat a nutritious diet, seek medical support.",
    "Tuberculosis": "Cover mouth when coughing, take full course of medication, improve ventilation.",
    "Common Cold": "Rest well, stay hydrated, avoid spreading germs (e.g., cover sneezes).",
    "Pneumonia": "Get vaccinated, avoid smoking, rest and hydrate adequately.",
    "Dimorphic hemorrhoids (piles)": "Increase fiber intake, stay hydrated, avoid straining during bowel movements.",
    "Heart attack": "Adopt a heart-healthy diet, exercise regularly, avoid smoking.",
    "Varicose veins": "Elevate legs, wear compression stockings, avoid prolonged standing.",
    "Hypothyroidism": "Take thyroid medication as prescribed, eat iodine-rich foods, monitor symptoms.",
    "Hyperthyroidism": "Avoid caffeine, follow medication regimen, reduce stress.",
    "Hypoglycemia": "Eat regular small meals, carry sugar sources, monitor blood sugar levels.",
    "Osteoarthritis": "Maintain a healthy weight, exercise gently, use pain relief as needed.",
    "Arthritis": "Stay active, apply heat/cold packs, avoid overexertion.",
    "(Vertigo) Paroxysmal Positional Vertigo": "Avoid sudden head movements, perform repositioning exercises, rest when dizzy.",
    "Acne": "Keep skin clean, avoid touching face, use non-comedogenic products.",
    "Urinary tract infection": "Drink plenty of water, urinate frequently, maintain hygiene.",
    "Psoriasis": "Moisturize skin, avoid triggers (e.g., stress), use medicated creams.",
    "Impetigo": "Keep sores clean, avoid touching them, use prescribed antibiotics."
}

def predict_disease(selected_symptoms):
    list_c = [0] * len(l1)
    for symptom in selected_symptoms:
        if symptom in l1:
            list_c[l1.index(symptom)] = 1
    test = np.array(list_c).reshape(1, -1)
    prediction = gnb.predict(test)[0]
    department = disease_department_mapping.get(prediction, "General Medicine")
    precaution = disease_precaution_mapping.get(prediction, "Consult a healthcare professional for advice.")
    print("Prediction:", prediction)  # Debug
    return prediction, department, precaution


def get_prediction_stats(history=None):
    if not history:
        print("Prediction history is empty in utils")
        return {}
    from collections import Counter
    stats = dict(Counter(history))
    print("Prediction stats in utils:", stats)
    return stats

