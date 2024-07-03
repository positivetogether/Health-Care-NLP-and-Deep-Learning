import pandas as pd
import re
import gender_guesser.detector as gender

def extract_age(text):
    patterns = [
        r'(\d+)-year-old',        # Pattern: "xx-year-old"
        r'(\d+) years old',       # Pattern: "xx years old"
        r'(\d+) yrs old',         # Pattern: "xx yrs old"
        r'age (\d+)',             # Pattern: "age xx"
        r'(\d+) yo',              # Pattern: "xx yo"
        r'(\d+)-yr-old'           # Pattern: "xx-yr-old"
    ]
    for pattern in patterns:
        match = re.search(pattern, text, re.IGNORECASE)
        if match:
            return int(match.group(1))
    return None

def has_diabetes(text):
    if pd.isnull(text):
        return False
    return 'diabetes' in text.lower()

def has_hypertension(text):
    if pd.isnull(text):
        return False
    return 'hypertension' in text.lower()

def has_heart_disease(text):
    if pd.isnull(text):
        return False
    return any(keyword in text.lower() for keyword in ['heart disease', 'cardiovascular', 'cardiac', 'myocardial'])

def extract_heart_disease_type(text):
    if pd.isnull(text):
        return None
    for keyword in ['heart disease', 'cardiovascular', 'cardiac', 'myocardial']:
        if keyword in text.lower():
            return keyword
    return None

def extract_year(text):
    match = re.search(r'\b(19|20)\d{2}\b', text)
    return int(match.group(0)) if match else None

d = gender.Detector()

def extract_gender_from_name(first_name):
    if pd.isnull(first_name):
        return None
    gender_guess = d.get_gender(first_name)
    if gender_guess in ['male', 'mostly_male']:
        return 'Male'
    elif gender_guess in ['female', 'mostly_female']:
        return 'Female'
    else:
        return None

def extract_gender(text):
    if pd.isnull(text):
        return None
    text = text.lower()
    male_keywords = ['male', 'man', 'boy', 'mr', 'sir']
    female_keywords = ['female', 'woman', 'girl', 'mrs', 'ms', 'madam', 'lady', 'she', 'her', 'hers']
    
    male_count = sum(text.count(keyword) for keyword in male_keywords)
    female_count = sum(text.count(keyword) for keyword in female_keywords)
    
    if male_count > female_count:
        return 'Male'
    elif female_count > male_count:
        return 'Female'
    else:
        return None
    
def combine_genders(row):
    genders = row.dropna().tolist()
    if genders:
        male_count = genders.count('Male')
        female_count = genders.count('Female')
        if male_count > female_count:
            return 'Male'
        elif female_count > male_count:
            return 'Female'
        else:
            return genders[0] 
    else:
        return None

def extract_visit_type(text):
    if pd.isnull(text):
        return None
    text = text.lower()
    if 'emergency' in text or 'er' in text:
        return 'Emergency'
    elif 'follow-up' in text or 'follow up' in text:
        return 'Follow-up'
    elif 'routine' in text or 'checkup' in text:
        return 'Routine Checkup'
    else:
        return 'Other'

def extract_procedure_type(text):
    if pd.isnull(text):
        return None
    text = text.lower()
    if 'surgery' in text or 'operation' in text:
        return 'Surgery'
    elif 'consultation' in text or 'consult' in text:
        return 'Consultation'
    elif 'therapy' in text or 'treatment' in text:
        return 'Therapy'
    else:
        return 'Other'

def extract_admission_type(text):
    if pd.isnull(text):
        return None
    text = text.lower()
    if 'inpatient' in text or 'admitted' in text:
        return 'Inpatient'
    elif 'outpatient' in text or 'discharged' in text:
        return 'Outpatient'
    else:
        return 'Other'

body_parts = [
    'abdomen', 'adrenal gland', 'ankle', 'appendix', 'arm', 'artery', 'back', 'bladder', 'blood vessel', 
    'bone', 'brain', 'breast', 'bronchus', 'buttock', 'calf', 'cervix', 'chest', 'colon', 'ear', 
    'elbow', 'endometrium', 'esophagus', 'eye', 'face', 'fallopian tube', 'finger', 'foot', 'gallbladder', 
    'hand', 'heart', 'hip', 'intestine', 'jaw', 'kidney', 'knee', 'larynx', 'liver', 'lung', 'lymph node', 
    'mouth', 'muscle', 'neck', 'nerve', 'nose', 'ovary', 'pancreas', 'pelvis', 'penis', 'pharynx', 'prostate', 
    'rectum', 'rib', 'scalp', 'scapula', 'shoulder', 'sinus', 'skin', 'spine', 'spleen', 'stomach', 'testis', 
    'throat', 'thyroid', 'tongue', 'tooth', 'ureter', 'urethra', 'uterus', 'vagina', 'vein', 'wrist'
]

def extract_body_parts(text):
    if pd.isnull(text):
        return None
    text = text.lower()
    found_parts = [part for part in body_parts if part in text]
    return ', '.join(found_parts) if found_parts else None

symptom_keywords = [
    'fever', 'pyrexia', 'high temperature', 'febrile',
    'cough', 'dry cough', 'productive cough', 'barking cough',
    'shortness of breath', 'dyspnea', 'breathlessness',
    'fatigue', 'tiredness', 'exhaustion', 'lethargy',
    'headache', 'migraine', 'cephalgia', 'head pain',
    'chest pain', 'sternum pain', 'thoracic pain', 'chest discomfort',
    'nausea', 'queasiness', 'sickness',
    'vomiting', 'emesis', 'throwing up',
    'diarrhea', 'loose stools', 'frequent bowel movements',
    'constipation', 'difficulty passing stools', 'hard stools',
    'abdominal pain', 'stomach pain', 'bellyache', 'gastric pain',
    'dizziness', 'lightheadedness', 'vertigo',
    'sore throat', 'pharyngitis', 'throat pain',
    'runny nose', 'rhinorrhea', 'nasal discharge',
    'sneezing', 'sternutation',
    'joint pain', 'arthralgia', 'joint discomfort',
    'muscle pain', 'myalgia', 'muscle ache',
    'rash', 'dermatitis', 'skin eruption',
    'itching', 'pruritus', 'skin itching',
    'swelling', 'edema', 'swollen tissue',
    'weight loss', 'unintentional weight loss', 'slimming',
    'weight gain', 'obesity', 'weight increase',
    'anxiety', 'nervousness', 'apprehension',
    'depression', 'depressive disorder', 'low mood',
    'insomnia', 'sleeplessness', 'difficulty sleeping',
    'palpitations', 'heart palpitations', 'irregular heartbeat',
    'back pain', 'dorsalgia', 'backache',
    'leg pain', 'crural pain', 'lower limb pain',
    'arm pain', 'brachial pain', 'upper limb pain',
    'ear pain', 'otalgia', 'earache',
    'eye pain', 'ocular pain', 'eye discomfort',
    'neck pain', 'cervicalgia', 'neckache',
    'hip pain', 'pelvic pain', 'hip discomfort',
    'bleeding', 'hemorrhage', 'bleeding out',
    'blurred vision', 'vision impairment', 'visual disturbances',
    'burning sensation', 'burning feeling', 'burning pain',
    'chills', 'shivering', 'cold chills',
    'cold sweats', 'night sweats', 'sweating episodes',
    'confusion', 'disorientation', 'mental confusion',
    'congestion', 'nasal congestion', 'stuffiness',
    'cramps', 'muscle cramps', 'stomach cramps',
    'difficulty swallowing', 'dysphagia', 'swallowing difficulties',
    'dry mouth', 'xerostomia', 'mouth dryness',
    'dry skin', 'xerosis', 'skin dryness',
    'ear discharge', 'otorrhea', 'ear drainage',
    'earache', 'otalgia', 'ear pain',
    'excessive thirst', 'polydipsia', 'extreme thirst',
    'fainting', 'syncope', 'passing out',
    'flushing', 'blushing', 'redness',
    'hair loss', 'alopecia', 'baldness',
    'hallucinations', 'delusions', 'seeing things',
    'hearing loss', 'auditory impairment', 'hearing impairment',
    'heartburn', 'pyrosis', 'acid reflux',
    'hives', 'urticaria', 'skin welts',
    'hoarseness', 'dysphonia', 'voice loss',
    'increased appetite', 'hyperphagia', 'excessive hunger',
    'indigestion', 'dyspepsia', 'upset stomach',
    'irregular heartbeat', 'arrhythmia', 'irregular pulse',
    'itchy eyes', 'ocular pruritus', 'eye itchiness',
    'jaundice', 'icterus', 'yellow skin',
    'loss of appetite', 'anorexia', 'reduced appetite',
    'memory loss', 'amnesia', 'forgetfulness',
    'mood swings', 'emotional lability', 'mood changes',
    'mouth sores', 'stomatitis', 'oral ulcers',
    'muscle cramps', 'spasms', 'muscle contractions',
    'nausea and vomiting', 'N/V', 'sickness and vomiting',
    'night sweats', 'nocturnal sweating', 'sweating at night',
    'nosebleeds', 'epistaxis', 'nasal bleeding',
    'numbness', 'paresthesia', 'loss of sensation',
    'painful urination', 'dysuria', 'burning urination',
    'paralysis', 'plegia', 'loss of movement',
    'rapid heartbeat', 'tachycardia', 'fast heart rate',
    'red eyes', 'conjunctival injection', 'eye redness',
    'seizures', 'convulsions', 'fits',
    'skin discoloration', 'hyperpigmentation', 'skin changes',
    'sleep disturbances', 'dyssomnia', 'sleep problems',
    'slow healing', 'delayed healing', 'prolonged healing',
    'sore muscles', 'muscle soreness', 'muscle pain',
    'sore tongue', 'glossodynia', 'tongue pain',
    'speech difficulties', 'dysarthria', 'speech problems',
    'stiff neck', 'nuchal rigidity', 'neck stiffness',
    'stomach cramps', 'abdominal cramps', 'belly cramps',
    'stuffy nose', 'nasal congestion', 'blocked nose',
    'sweating', 'diaphoresis', 'excessive sweating',
    'swollen glands', 'lymphadenopathy', 'gland swelling',
    'taste changes', 'dysgeusia', 'altered taste',
    'tremors', 'shaking', 'trembling',
    'trouble concentrating', 'attention difficulties', 'focus issues',
    'trouble sleeping', 'insomnia', 'difficulty falling asleep',
    'unusual bruising', 'ecchymosis', 'bruising easily',
    'urinary frequency', 'polyuria', 'frequent urination',
    'urinary incontinence', 'enuresis', 'loss of bladder control',
    'vision problems', 'visual impairment', 'sight issues',
    'vomiting blood', 'hematemesis', 'bloody vomit',
    'weakness', 'asthenia', 'lack of strength',
    'wheezing', 'stridor', 'breath sound',
    'appetite loss', 'anorexia', 'loss of appetite',
    'body aches', 'myalgia', 'generalized pain',
    'burning pain', 'neuropathic pain', 'burning sensation',
    'chest tightness', 'thoracic tightness', 'chest constriction',
    'cold feet', 'cold extremities', 'cool feet',
    'cold hands', 'cold extremities', 'cool hands',
    'dry cough', 'nonproductive cough', 'hacking cough',
    'feeling cold', 'chills', 'cold sensation',
    'flank pain', 'side pain', 'lateral pain',
    'frequent urination', 'polyuria', 'increased urination',
    'general discomfort', 'malaise', 'general unease',
    'groin pain', 'inguinal pain', 'groin discomfort',
    'joint stiffness', 'articular rigidity', 'joint inflexibility',
    'lightheadedness', 'dizziness', 'faintness',
    'loss of balance', 'disequilibrium', 'balance problems',
    'lower back pain', 'lumbago', 'low back pain',
    'malaise', 'general malaise', 'discomfort',
    'muscle weakness', 'myasthenia', 'weak muscles',
    'neck stiffness', 'nuchal rigidity', 'stiff neck',
    'nightmares', 'bad dreams', 'night terrors',
    'nocturia', 'nighttime urination', 'frequent night urination',
    'painful joints', 'arthralgia', 'joint pain',
    'pelvic pain', 'pelvic discomfort', 'lower abdominal pain',
    'pressure in chest', 'chest pressure', 'thoracic pressure',
    'shakiness', 'trembling', 'tremors',
    'skin rash', 'dermatitis', 'skin eruption',
    'sore joints', 'arthralgia', 'joint pain',
    'spasms', 'cramps', 'muscle spasms',
    'stomach pain', 'gastric pain', 'abdominal pain',
    'throat pain', 'pharyngitis', 'sore throat',
    'tingling', 'paresthesia', 'pins and needles',
    'tiredness', 'fatigue', 'exhaustion',
    'upper abdominal pain', 'epigastric pain', 'upper stomach pain',
    'waking up tired', 'morning fatigue', 'tired upon waking',
    'loss of energy', 'fatigue', 'lethargy',
    'lethargy', 'fatigue', 'lack of energy',
    'muscle soreness', 'muscle pain', 'myalgia',
    'throat irritation', 'pharyngeal irritation', 'throat itchiness',
    'nasal congestion', 'stuffy nose', 'blocked nose',
    'stiffness', 'rigidity', 'muscle stiffness',
    'eye discharge', 'ocular discharge', 'eye drainage',
    'chest discomfort', 'thoracic discomfort', 'chest unease',
    'bruising easily', 'ecchymosis', 'easy bruising',
    'persistent cough', 'chronic cough', 'long-lasting cough',
    'dry throat', 'throat dryness', 'parched throat',
    'rapid breathing', 'tachypnea', 'fast breathing',
    'shallow breathing', 'hypopnea', 'superficial breathing',
    'skin redness', 'erythema', 'red skin',
    'irritability', 'moodiness', 'irritable',
    'restlessness', 'agitation', 'restless feeling',
    'low energy', 'fatigue', 'lack of energy',
    'exhaustion', 'extreme tiredness', 'fatigue',
    'sensitivity to light', 'photophobia', 'light sensitivity',
    'sensitivity to sound', 'phonophobia', 'sound sensitivity',
    'run-down feeling', 'malaise', 'general unease',
    'burning eyes', 'ocular burning', 'eye discomfort',
    'tearing eyes', 'lacrimation', 'watery eyes',
    'difficulty focusing', 'concentration issues', 'focus problems',
    'difficulty sleeping', 'insomnia', 'sleep problems',
    'decreased appetite', 'anorexia', 'loss of appetite',
    'painful breathing', 'pleuritic pain', 'breathing pain',
    'tight chest', 'chest tightness', 'thoracic constriction',
    'stomach discomfort', 'gastric unease', 'abdominal discomfort',
    'stomach upset', 'dyspepsia', 'gastric distress',
    'heart palpitations', 'palpitations', 'irregular heartbeats',
    'irregular pulse', 'arrhythmia', 'irregular heartbeat',
    'difficult urination', 'dysuria', 'urinary difficulty',
    'frequent urination at night', 'nocturia', 'nighttime urination',
    'weight gain', 'weight increase', 'weight rise',
    'weight loss', 'slimming', 'weight reduction',
    # Expanded keywords
    'anal pain', 'rectal pain', 'buttock pain', 'genital pain', 'painful erection',
    'foot pain', 'ankle pain', 'heel pain', 'toe pain', 'heel spur', 'plantar fasciitis',
    'hip pain', 'groin pain', 'sciatica', 'thigh pain', 'hamstring pain', 'quadriceps pain',
    'jaw pain', 'temporal pain', 'facial pain', 'toothache', 'gum pain', 'mouth pain',
    'wrist pain', 'hand pain', 'finger pain', 'carpal tunnel syndrome', 'trigger finger',
    'shoulder pain', 'rotator cuff pain', 'shoulder blade pain', 'scapular pain',
    'ankle swelling', 'foot swelling', 'knee swelling', 'hand swelling', 'face swelling',
    'shoulder swelling', 'arm swelling', 'elbow swelling', 'finger swelling', 'leg swelling',
    'foot numbness', 'toe numbness', 'ankle numbness', 'leg numbness', 'knee numbness',
    'hip numbness', 'hand numbness', 'finger numbness', 'wrist numbness', 'elbow numbness',
    'shoulder numbness', 'neck numbness', 'face numbness', 'lip numbness', 'tongue numbness',
    'arm weakness', 'leg weakness', 'hand weakness', 'foot weakness', 'shoulder weakness',
    'hip weakness', 'back weakness', 'neck weakness', 'jaw weakness', 'facial weakness',
    'muscle fatigue', 'joint fatigue', 'eye fatigue', 'mental fatigue', 'emotional fatigue',
    'bone pain', 'spine pain', 'rib pain', 'sternum pain', 'collarbone pain', 'pelvic bone pain',
    'shin pain', 'calf pain', 'forearm pain', 'upper arm pain', 'upper back pain', 'middle back pain',
    'lower back pain', 'coccyx pain', 'tailbone pain', 'abdomen pain', 'abdominal bloating',
    'chest tightness', 'chest heaviness', 'heart discomfort', 'abdominal discomfort', 'groin discomfort',
    'rectal discomfort', 'anal discomfort', 'genital discomfort', 'vaginal discomfort', 'penile discomfort',
    'testicular discomfort', 'scrotal discomfort', 'inguinal discomfort', 'perineal discomfort', 'pelvic discomfort',
    'anal itching', 'rectal itching', 'buttock itching', 'genital itching', 'vaginal itching', 'penile itching',
    'scrotal itching', 'testicular itching', 'perineal itching', 'inguinal itching', 'buttock rash', 'genital rash',
    'vaginal rash', 'penile rash', 'scrotal rash', 'testicular rash', 'perineal rash', 'inguinal rash', 'rectal rash',
    'anal rash', 'foot rash', 'leg rash', 'arm rash', 'hand rash', 'finger rash', 'toe rash', 'ankle rash', 'knee rash',
    'elbow rash', 'wrist rash', 'neck rash', 'chest rash', 'abdomen rash', 'back rash', 'buttock rash', 'groin rash',
    'scalp rash', 'facial rash', 'lip rash', 'mouth rash', 'eye rash', 'ear rash', 'nose rash', 'cheek rash',
    'forehead rash', 'chin rash', 'jaw rash', 'temple rash', 'neck itchiness', 'eye itchiness', 'ear itchiness',
    'nose itchiness', 'lip itchiness', 'cheek itchiness', 'forehead itchiness', 'chin itchiness', 'jaw itchiness',
    'temple itchiness', 'chest itchiness', 'abdomen itchiness', 'back itchiness', 'arm itchiness', 'leg itchiness',
    'hand itchiness', 'finger itchiness', 'toe itchiness', 'ankle itchiness', 'knee itchiness', 'elbow itchiness',
    'wrist itchiness', 'buttock itchiness', 'groin itchiness', 'scalp itchiness', 'perineal itchiness',
    'genital pain', 'vaginal pain', 'penile pain', 'testicular pain', 'scrotal pain', 'inguinal pain',
    'perineal pain', 'genital swelling', 'vaginal swelling', 'penile swelling', 'testicular swelling', 'scrotal swelling',
    'inguinal swelling', 'perineal swelling', 'genital discharge', 'vaginal discharge', 'penile discharge', 'testicular discharge',
    'scrotal discharge', 'inguinal discharge', 'perineal discharge', 'genital odor', 'vaginal odor', 'penile odor', 'testicular odor',
    'scrotal odor', 'inguinal odor', 'perineal odor', 'genital redness', 'vaginal redness', 'penile redness', 'testicular redness',
    'scrotal redness', 'inguinal redness', 'perineal redness', 'genital dryness', 'vaginal dryness', 'penile dryness',
    'testicular dryness', 'scrotal dryness', 'inguinal dryness', 'perineal dryness', 'genital irritation', 'vaginal irritation',
    'penile irritation', 'testicular irritation', 'scrotal irritation', 'inguinal irritation', 'perineal irritation', 'genital bleeding',
    'vaginal bleeding', 'penile bleeding', 'testicular bleeding', 'scrotal bleeding', 'inguinal bleeding', 'perineal bleeding',
    'genital sores', 'vaginal sores', 'penile sores', 'testicular sores', 'scrotal sores', 'inguinal sores', 'perineal sores',
    'genital ulcers', 'vaginal ulcers', 'penile ulcers', 'testicular ulcers', 'scrotal ulcers', 'inguinal ulcers', 'perineal ulcers',
    'genital lesions', 'vaginal lesions', 'penile lesions', 'testicular lesions', 'scrotal lesions', 'inguinal lesions', 'perineal lesions'
]

def extract_symptoms(text):
    if pd.isnull(text):
        return None
    text = text.lower()
    found_symptoms = [symptom for symptom in symptom_keywords if symptom in text]
    return ', '.join(found_symptoms) if found_symptoms else None

disease_keywords = {
    'diabetes': ['diabetes', 'diabetic', 'type 1 diabetes', 'type 2 diabetes', 'gestational diabetes', 'insulin resistance'],
    'hypertension': ['hypertension', 'high blood pressure', 'essential hypertension', 'secondary hypertension'],
    'heart disease': ['heart disease', 'cardiovascular', 'cardiac', 'heart attack', 'myocardial infarction', 'coronary heart disease', 'angina', 'arrhythmia', 'heart failure'],
    'cancer': ['cancer', 'malignancy', 'carcinoma', 'sarcoma', 'leukemia', 'lymphoma', 'melanoma', 'chemotherapy', 'radiation therapy'],
    'stroke': ['stroke', 'cerebrovascular accident', 'ischemic stroke', 'hemorrhagic stroke', 'transient ischemic attack', 'brain attack'],
    'asthma': ['asthma', 'wheezing', 'bronchial asthma', 'asthma attack', 'inhaler', 'bronchodilator'],
    'arthritis': ['arthritis', 'osteoarthritis', 'rheumatoid arthritis', 'joint pain', 'joint inflammation'],
    'pneumonia': ['pneumonia', 'lung infection', 'bacterial pneumonia', 'viral pneumonia', 'aspiration pneumonia', 'antibiotics'],
    'tuberculosis': ['tuberculosis', 'tb', 'pulmonary tuberculosis', 'extrapulmonary tuberculosis', 'latent tb infection', 'tb treatment'],
    'anemia': ['anemia', 'anemic', 'iron deficiency anemia', 'sickle cell anemia', 'vitamin B12 deficiency', 'blood transfusion'],
    'allergy': ['allergy', 'allergic', 'allergic reaction', 'hay fever', 'food allergy', 'anaphylaxis', 'antihistamines'],
    'depression': ['depression', 'depressive disorder', 'major depressive disorder', 'clinical depression', 'antidepressants', 'cognitive behavioral therapy'],
    'anxiety': ['anxiety', 'anxious', 'generalized anxiety disorder', 'panic disorder', 'social anxiety disorder', 'anxiolytics'],
    'obesity': ['obesity', 'overweight', 'morbid obesity', 'bariatric surgery', 'weight loss'],
    'migraine': ['migraine', 'headache', 'migraine attack', 'migraine with aura', 'tension headache', 'triptans'],
    'epilepsy': ['epilepsy', 'seizure', 'epileptic', 'grand mal seizure', 'petit mal seizure', 'antiepileptic drugs'],
    'hepatitis': ['hepatitis', 'liver inflammation', 'hepatitis A', 'hepatitis B', 'hepatitis C', 'hepatitis vaccine'],
    'cirrhosis': ['cirrhosis', 'liver disease', 'alcoholic cirrhosis', 'nonalcoholic fatty liver disease', 'liver transplantation'],
    'kidney disease': ['kidney disease', 'renal disease', 'chronic kidney disease', 'acute kidney injury', 'dialysis'],
    'lung disease': ['lung disease', 'pulmonary disease', 'chronic obstructive pulmonary disease', 'interstitial lung disease', 'pulmonary fibrosis'],
    'prostate cancer': ['prostate cancer', 'prostate adenocarcinoma', 'prostatectomy', 'radiation therapy', 'hormone therapy'],
    'breast cancer': ['breast cancer', 'mastectomy', 'lumpectomy', 'chemotherapy', 'radiation therapy'],
    'colon cancer': ['colon cancer', 'colorectal cancer', 'colonoscopy', 'chemotherapy', 'radiation therapy'],
    'leukemia': ['leukemia', 'acute lymphoblastic leukemia', 'chronic myeloid leukemia', 'bone marrow transplant'],
    'lymphoma': ['lymphoma', 'Hodgkin lymphoma', 'non-Hodgkin lymphoma', 'chemotherapy', 'radiation therapy'],
    'melanoma': ['melanoma', 'skin cancer', 'malignant melanoma', 'melanoma in situ', 'immunotherapy'],
    'hyperlipidemia': ['hyperlipidemia', 'high cholesterol', 'dyslipidemia', 'statins', 'lipid-lowering drugs'],
    'chronic kidney disease': ['chronic kidney disease', 'chronic renal disease', 'end-stage renal disease', 'dialysis', 'kidney transplantation'],
    'chronic obstructive pulmonary disease': ['chronic obstructive pulmonary disease', 'copd', 'emphysema', 'chronic bronchitis', 'bronchodilators'],
    'coronary artery disease': ['coronary artery disease', 'cad', 'coronary heart disease', 'atherosclerosis', 'angioplasty', 'stent'],
    "alzheimer's disease": ["alzheimer's disease", 'dementia', 'memory loss', 'cognitive decline', 'Alzheimer’s treatment'],
    "parkinson's disease": ["parkinson's disease", 'parkinsonism', 'dopamine', 'motor symptoms', 'deep brain stimulation'],
    'multiple sclerosis': ['multiple sclerosis', 'ms', 'relapsing-remitting ms', 'primary progressive ms', 'ms treatment'],
    'hiv': ['hiv', 'aids', 'human immunodeficiency virus', 'antiretroviral therapy', 'hiv prevention'],
    'syphilis': ['syphilis', 'treponema pallidum', 'primary syphilis', 'secondary syphilis', 'penicillin'],
    'chlamydia': ['chlamydia', 'chlamydia trachomatis', 'chlamydia infection', 'antibiotic treatment'],
    'gonorrhea': ['gonorrhea', 'neisseria gonorrhoeae', 'gonococcal infection', 'antibiotic treatment'],
    'herpes': ['herpes', 'herpes simplex virus', 'hsv', 'cold sores', 'genital herpes', 'antiviral medication'],
    'hpv': ['hpv', 'human papillomavirus', 'hpv infection', 'hpv vaccine', 'cervical cancer'],
    'hepatitis a': ['hepatitis a', 'hep a', 'hepatitis a virus', 'hav infection', 'hepatitis a vaccine'],
    'hepatitis b': ['hepatitis b', 'hep b', 'hepatitis b virus', 'hbv infection', 'hepatitis b vaccine'],
    'hepatitis c': ['hepatitis c', 'hep c', 'hepatitis c virus', 'hcv infection', 'hepatitis c treatment'],
    'pancreatitis': ['pancreatitis', 'acute pancreatitis', 'chronic pancreatitis', 'pancreatic inflammation', 'pancreatic enzymes'],
    'ulcerative colitis': ['ulcerative colitis', 'inflammatory bowel disease', 'ibd', 'colon inflammation', 'biologic therapy'],
    "crohn's disease": ["crohn's disease", 'inflammatory bowel disease', 'ibd', 'intestinal inflammation', 'crohn’s treatment'],
    'gastritis': ['gastritis', 'stomach inflammation', 'chronic gastritis', 'acute gastritis', 'antacids'],
    'esophagitis': ['esophagitis', 'esophageal inflammation', 'reflux esophagitis', 'eosinophilic esophagitis'],
    'gallstones': ['gallstones', 'cholelithiasis', 'biliary colic', 'cholecystectomy'],
    'kidney stones': ['kidney stones', 'renal calculi', 'nephrolithiasis', 'urolithiasis', 'extracorporeal shock wave lithotripsy'],
    'gout': ['gout', 'gouty arthritis', 'uric acid crystals', 'allopurinol'],
    'osteoporosis': ['osteoporosis', 'bone density loss', 'bisphosphonates', 'calcium supplements'],
    'fibromyalgia': ['fibromyalgia', 'chronic pain syndrome', 'widespread pain', 'fibromyalgia treatment'],
    'lupus': ['lupus', 'systemic lupus erythematosus', 'sle', 'autoimmune disease', 'hydroxychloroquine'],
    'scleroderma': ['scleroderma', 'systemic sclerosis', 'skin thickening', 'immunosuppressants'],
    'eczema': ['eczema', 'atopic dermatitis', 'skin inflammation', 'topical steroids'],
    'psoriasis': ['psoriasis', 'psoriatic arthritis', 'skin plaques', 'biologic therapy'],
    'acne': ['acne', 'acne vulgaris', 'pimples', 'benzoyl peroxide'],
    'rosacea': ['rosacea', 'facial redness', 'topical antibiotics'],
    'dermatitis': ['dermatitis', 'contact dermatitis', 'atopic dermatitis', 'eczema'],
    'vitiligo': ['vitiligo', 'skin depigmentation', 'autoimmune skin disorder', 'topical corticosteroids'],
    'alopecia': ['alopecia', 'hair loss', 'alopecia areata', 'androgenetic alopecia'],
    'anorexia': ['anorexia', 'anorexia nervosa', 'eating disorder', 'nutritional rehabilitation'],
    'bulimia': ['bulimia', 'bulimia nervosa', 'binge eating', 'cognitive behavioral therapy'],
    'bipolar disorder': ['bipolar disorder', 'manic depression', 'mood swings', 'lithium'],
    'schizophrenia': ['schizophrenia', 'psychosis', 'antipsychotics', 'schizophrenic'],
    'ptsd': ['ptsd', 'post-traumatic stress disorder', 'trauma', 'ptsd treatment'],
    'ocd': ['ocd', 'obsessive-compulsive disorder', 'obsessions', 'compulsions', 'cognitive behavioral therapy'],
    'adhd': ['adhd', 'attention deficit hyperactivity disorder', 'hyperactivity', 'stimulant medications'],
    'glaucoma': ['glaucoma', 'increased intraocular pressure', 'optic nerve damage', 'eye drops'],
    'cataracts': ['cataracts', 'cloudy lens', 'cataract surgery', 'vision impairment'],
    'macular degeneration': ['macular degeneration', 'age-related macular degeneration', 'amd', 'vision loss'],
    'retinopathy': ['retinopathy', 'diabetic retinopathy', 'retinal damage', 'laser therapy'],
    'conjunctivitis': ['conjunctivitis', 'pink eye', 'eye infection', 'allergic conjunctivitis'],
    'keratitis': ['keratitis', 'corneal inflammation', 'corneal ulcer', 'antibiotic eye drops'],
    'sinusitis': ['sinusitis', 'sinus infection', 'chronic sinusitis', 'acute sinusitis'],
    'tonsillitis': ['tonsillitis', 'tonsil infection', 'recurrent tonsillitis', 'tonsillectomy'],
    'pharyngitis': ['pharyngitis', 'sore throat', 'bacterial pharyngitis', 'viral pharyngitis'],
    'laryngitis': ['laryngitis', 'vocal cord inflammation', 'hoarseness', 'acute laryngitis'],
    'bronchitis': ['bronchitis', 'chronic bronchitis', 'acute bronchitis', 'cough', 'bronchodilators'],
    'pleurisy': ['pleurisy', 'pleural inflammation', 'pleuritic chest pain'],
    'emphysema': ['emphysema', 'chronic obstructive pulmonary disease', 'copd', 'alveolar damage'],
    'meningitis': ['meningitis', 'bacterial meningitis', 'viral meningitis', 'meningococcal infection'],
    'encephalitis': ['encephalitis', 'brain inflammation', 'viral encephalitis', 'herpes encephalitis'],
    'cerebral palsy': ['cerebral palsy', 'cp', 'motor disability', 'spastic cerebral palsy'],
    'muscular dystrophy': ['muscular dystrophy', 'duchenne muscular dystrophy', 'dmd', 'muscle weakness'],
    'myasthenia gravis': ['myasthenia gravis', 'mg', 'neuromuscular disorder', 'muscle fatigue'],
    'rheumatoid arthritis': ['rheumatoid arthritis', 'ra', 'joint inflammation', 'autoimmune arthritis'],
    'osteoarthritis': ['osteoarthritis', 'oa', 'degenerative joint disease', 'joint pain'],
    'vasculitis': ['vasculitis', 'blood vessel inflammation', 'autoimmune vasculitis'],
    'thrombosis': ['thrombosis', 'blood clot', 'deep vein thrombosis', 'dvt', 'anticoagulants'],
    'hemophilia': ['hemophilia', 'bleeding disorder', 'factor viii deficiency', 'factor ix deficiency'],
    'myeloma': ['myeloma', 'multiple myeloma', 'plasma cell cancer', 'bone marrow cancer'],
    'polycythemia vera': ['polycythemia vera', 'pv', 'blood cancer', 'increased red blood cells'],
    'thrombocytopenia': ['thrombocytopenia', 'low platelet count', 'immune thrombocytopenia'],
    'sickle cell disease': ['sickle cell disease', 'sickle cell anemia', 'hemoglobinopathy', 'blood transfusion'],
    'huntington\'s disease': ['huntington\'s disease', 'hd', 'neurodegenerative disease', 'chorea'],
    'cystic fibrosis': ['cystic fibrosis', 'cf', 'genetic disorder', 'mucus buildup'],
    'tay-sachs disease': ['tay-sachs disease', 'tay-sachs', 'genetic disorder', 'lysosomal storage disease'],
    'thalassemia': ['thalassemia', 'beta thalassemia', 'alpha thalassemia', 'blood transfusion'],
    'asbestosis': ['asbestosis', 'asbestos exposure', 'pulmonary fibrosis'],
    'silicosis': ['silicosis', 'silica dust exposure', 'lung fibrosis'],
    'coal worker\'s pneumoconiosis': ['coal worker\'s pneumoconiosis', 'black lung disease', 'coal dust exposure'],
    'vasectomy': ['vasectomy', 'male sterilization', 'contraception'],
    'orchiopexy': ['orchiopexy', 'undescended testicle', 'cryptorchidism'],
    'herniorrhaphy': ['herniorrhaphy', 'hernia repair', 'inguinal hernia'],
    'scrotal pain': ['scrotal pain', 'testicular pain', 'epididymitis'],
    'bladder tumor': ['bladder tumor', 'bladder cancer', 'transitional cell carcinoma'],
    'benign prostatic hyperplasia': ['benign prostatic hyperplasia', 'bph', 'enlarged prostate', 'tamsulosin'],
    'epididymitis': ['epididymitis', 'testicular infection', 'testicular pain'],
    'urethral stricture': ['urethral stricture', 'narrowed urethra', 'urinary retention'],
    'nephrolithiasis': ['nephrolithiasis', 'kidney stones', 'renal calculi'],
    'hydronephrosis': ['hydronephrosis', 'kidney swelling', 'urinary tract obstruction'],
    'testicular torsion': ['testicular torsion', 'twisted testicle', 'scrotal pain'],
    'urinary retention': ['urinary retention', 'inability to urinate', 'bladder dysfunction'],
    'prostate adenocarcinoma': ['prostate adenocarcinoma', 'prostate cancer', 'gleason score'],
    'cystitis': ['cystitis', 'bladder infection', 'urinary tract infection', 'uti'],
    'voiding dysfunction': ['voiding dysfunction', 'urinary incontinence', 'bladder control'],
    'penile mass': ['penile mass', 'penile tumor', 'penile cancer'],
    'penile prosthesis': ['penile prosthesis', 'erectile dysfunction', 'penile implant'],
    'penile skin bridges': ['penile skin bridges', 'penile adhesion'],
    'urinary obstruction': ['urinary obstruction', 'blocked urine flow', 'hydronephrosis'],
    'ureteral calculus': ['ureteral calculus', 'ureteral stone', 'urinary stone'],
    'ureteropelvic junction obstruction': ['ureteropelvic junction obstruction', 'upj obstruction', 'hydronephrosis'],
    'inguinal hernia': ['inguinal hernia', 'groin hernia', 'herniorrhaphy'],
    'hydrocele': ['hydrocele', 'scrotal swelling', 'fluid accumulation'],
    'pyeloureteroscopy': ['pyeloureteroscopy', 'kidney stone removal', 'ureteroscopic lithotripsy'],
    'bladder neck': ['bladder neck', 'bladder neck obstruction', 'urinary retention'],
    'capsule': ['capsule', 'joint capsule', 'fibrous capsule'],
    'hemorrhage': ['hemorrhage', 'bleeding', 'internal bleeding', 'hemorrhagic'],
    'urosepsis': ['urosepsis', 'urinary tract infection', 'sepsis'],
    'orchiectomy': ['orchiectomy', 'testicle removal', 'testicular cancer'],
    'urothelial carcinoma': ['urothelial carcinoma', 'bladder cancer', 'transitional cell carcinoma'],
    'hematuria': ['hematuria', 'blood in urine', 'gross hematuria'],
    'bronchiectasis': ['bronchiectasis', 'chronic lung disease', 'airway dilation'],
    'pulmonary fibrosis': ['pulmonary fibrosis', 'lung scarring', 'interstitial lung disease'],
    'sarcoidosis': ['sarcoidosis', 'granulomatous disease', 'multisystem inflammation'],
    'interstitial lung disease': ['interstitial lung disease', 'ild', 'lung fibrosis'],
    'pneumothorax': ['pneumothorax', 'collapsed lung', 'chest tube'],
    'pulmonary embolism': ['pulmonary embolism', 'pe', 'blood clot in lung', 'anticoagulants'],
    'lyme disease': ['lyme disease', 'borrelia burgdorferi', 'tick-borne illness', 'erythema migrans'],
    'rocky mountain spotted fever': ['rocky mountain spotted fever', 'rmsf', 'tick-borne illness', 'rickettsia rickettsii'],
    'ehrlichiosis': ['ehrlichiosis', 'tick-borne illness', 'ehrlichia infection'],
    'anaplasmosis': ['anaplasmosis', 'tick-borne illness', 'anaplasma infection'],
    'babesiosis': ['babesiosis', 'tick-borne illness', 'babesia infection'],
    'rabies': ['rabies', 'rabies virus', 'lyssavirus', 'post-exposure prophylaxis'],
    'west nile virus': ['west nile virus', 'wnv', 'mosquito-borne illness'],
    'zika virus': ['zika virus', 'zika infection', 'mosquito-borne illness'],
    'ebola virus': ['ebola virus', 'ebola hemorrhagic fever', 'ebola outbreak'],
    'lassa fever': ['lassa fever', 'arenavirus', 'viral hemorrhagic fever'],
    'dengue': ['dengue', 'dengue fever', 'mosquito-borne illness'],
    'yellow fever': ['yellow fever', 'flavivirus', 'mosquito-borne illness'],
    'malaria': ['malaria', 'plasmodium infection', 'mosquito-borne illness'],
    'leprosy': ['leprosy', 'hansen\'s disease', 'mycobacterium leprae'],
    'typhoid fever': ['typhoid fever', 'salmonella typhi', 'enteric fever'],
    'cholera': ['cholera', 'vibrio cholerae', 'severe diarrhea'],
    'plague': ['plague', 'bubonic plague', 'yersinia pestis'],
    'tetanus': ['tetanus', 'clostridium tetani', 'muscle stiffness'],
    'botulism': ['botulism', 'clostridium botulinum', 'muscle paralysis'],
    'anthrax': ['anthrax', 'bacillus anthracis', 'cutaneous anthrax'],
    'brucellosis': ['brucellosis', 'brucella infection', 'undulant fever'],
    'leptospirosis': ['leptospirosis', 'leptospira infection', 'weil\'s disease'],
    'tularemia': ['tularemia', 'francisella tularensis', 'rabbit fever'],
    'q fever': ['q fever', 'coxiella burnetii', 'zoonotic infection'],
    'melioidosis': ['melioidosis', 'burkholderia pseudomallei', 'whitmore\'s disease'],
    'glanders': ['glanders', 'burkholderia mallei', 'zoonotic infection'],
    'smallpox': ['smallpox', 'variola virus', 'eradicated disease'],
    'chickenpox': ['chickenpox', 'varicella', 'varicella-zoster virus'],
    'measles': ['measles', 'rubeola', 'paramyxovirus'],
    'mumps': ['mumps', 'paramyxovirus', 'parotitis'],
    'rubella': ['rubella', 'german measles', 'rubella virus'],
    'pertussis': ['pertussis', 'whooping cough', 'bordetella pertussis'],
    'diphtheria': ['diphtheria', 'corynebacterium diphtheriae', 'throat infection'],
    'scarlet fever': ['scarlet fever', 'group a streptococcus', 'strep throat'],
    'rheumatic fever': ['rheumatic fever', 'streptococcus infection', 'heart valve damage'],
    'polio': ['polio', 'poliomyelitis', 'poliovirus'],
    'hand foot and mouth disease': ['hand foot and mouth disease', 'coxsackievirus', 'enterovirus'],
    'croup': ['croup', 'laryngotracheobronchitis', 'barking cough'],
    'respiratory syncytial virus': ['respiratory syncytial virus', 'rsv', 'bronchiolitis'],
    'influenza': ['influenza', 'flu', 'influenza virus'],
    'covid-19': ['covid-19', 'coronavirus', 'sars-cov-2', 'pandemic'],
    'mers': ['mers', 'middle east respiratory syndrome', 'mers-cov'],
    'sars': ['sars', 'severe acute respiratory syndrome', 'sars-cov'],
}

def extract_disease(text):
    if pd.isnull(text):
        return None
    text = text.lower()
    found_diseases = set()
    for disease, keywords in disease_keywords.items():
        for keyword in keywords:
            pattern = r'\b' + re.escape(keyword) + r'\b'
            if re.search(pattern, text):
                found_diseases.add(disease)
    return ', '.join(found_diseases) if found_diseases else None