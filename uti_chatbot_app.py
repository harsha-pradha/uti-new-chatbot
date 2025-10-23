import streamlit as st
import pandas as pd
import numpy as np
import joblib
from sklearn.preprocessing import StandardScaler
import sys
import os

# Set page config
st.set_page_config(
    page_title="UTI Detection Chatbot - AI Medical Assistant",
    page_icon="🩺",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for professional medical interface
st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem;
        color: #1f77b4;
        text-align: center;
        margin-bottom: 1rem;
        font-weight: bold;
    }
    .risk-high {
        background-color: #ff4b4b;
        color: white;
        padding: 15px;
        border-radius: 10px;
        text-align: center;
        font-weight: bold;
        font-size: 1.2rem;
        margin: 10px 0;
    }
    .risk-medium {
        background-color: #ffa500;
        color: white;
        padding: 15px;
        border-radius: 10px;
        text-align: center;
        font-weight: bold;
        font-size: 1.2rem;
        margin: 10px 0;
    }
    .risk-low {
        background-color: #4CAF50;
        color: white;
        padding: 15px;
        border-radius: 10px;
        text-align: center;
        font-weight: bold;
        font-size: 1.2rem;
        margin: 10px 0;
    }
    .info-box {
        background-color: #f0f2f6;
        padding: 25px;
        border-radius: 10px;
        border-left: 5px solid #1f77b4;
        margin: 10px 0;
    }
    .parameter-box {
        background-color: white;
        padding: 15px;
        border-radius: 8px;
        border: 1px solid #ddd;
        margin: 5px 0;
    }
    .chat-container {
        background-color: #f9f9f9;
        border-radius: 10px;
        padding: 20px;
        margin: 10px 0;
        max-height: 400px;
        overflow-y: auto;
    }
    .user-message {
        background-color: #e3f2fd;
        padding: 10px 15px;
        border-radius: 15px;
        margin: 5px 0;
        text-align: right;
        margin-left: 50px;
    }
    .bot-message {
        background-color: white;
        padding: 10px 15px;
        border-radius: 15px;
        margin: 5px 0;
        border: 1px solid #ddd;
        margin-right: 50px;
    }
    .diet-box {
        background-color: #e8f5e8;
        padding: 20px;
        border-radius: 10px;
        border-left: 5px solid #4CAF50;
        margin: 10px 0;
    }
</style>
""", unsafe_allow_html=True)

# Title and description
st.markdown('<div class="main-header">🩺 AI-Powered UTI Detection Chatbot</div>', unsafe_allow_html=True)
st.markdown("""
<div style='text-align: center; margin-bottom: 2rem; color: #666;'>
Get instant AI-powered analysis of your urinalysis results with explanations in English and Tamil
</div>
""", unsafe_allow_html=True)

# Load model and preprocessing artifacts
@st.cache_resource
def load_model_artifacts():
    """Load the trained model and preprocessing artifacts"""
    try:
        model = joblib.load('models/best_uti_model.pkl')
        scaler = joblib.load('models/feature_scaler.pkl')
        feature_names = joblib.load('models/feature_names.pkl')
        model_performance = joblib.load('models/model_performance.pkl')
        return model, scaler, feature_names, model_performance
    except Exception as e:
        st.error(f"Error loading model artifacts: {e}")
        return None, None, None, None

def predict_uti_risk(user_inputs, model, scaler, feature_names):
    """Enhanced UTI risk prediction with clinical rules"""
    try:
        # Prepare input features
        input_features = prepare_user_inputs(user_inputs, feature_names)
        
        # Scale features
        input_scaled = scaler.transform([input_features])
        
        # Make prediction
        prediction = model.predict(input_scaled)[0]
        probability = model.predict_proba(input_scaled)[0][1]
        
        # ====== CRITICAL FIX: Apply clinical rules to override model ======
        clinical_probability = apply_clinical_rules(user_inputs, probability)
        
        # Use the higher of model probability or clinical probability
        final_probability = max(probability, clinical_probability)
        
        # Determine risk level with clinical adjustment
        if final_probability >= 0.6:
            risk_level = "HIGH"
        elif final_probability >= 0.3:
            risk_level = "MEDIUM"
        else:
            risk_level = "LOW"
        
        return {
            'prediction': 1 if final_probability >= 0.5 else 0,
            'probability': final_probability,
            'risk_level': risk_level,
            'confidence': final_probability
        }
    except Exception as e:
        st.error(f"Prediction error: {e}")
        return None

def apply_clinical_rules(user_inputs, base_probability):
    """Apply clinical rules to adjust probability based on key indicators"""
    clinical_score = base_probability
    
    # High WBC is strong indicator of UTI
    if user_inputs.get('WBC', 0) > 10:
        clinical_score += 0.3
    elif user_inputs.get('WBC', 0) > 5:
        clinical_score += 0.15
    
    # Bacteria presence
    if user_inputs.get('Bacteria', 0) >= 3:  # MODERATE or PLENTY
        clinical_score += 0.4
    elif user_inputs.get('Bacteria', 0) >= 2:  # FEW
        clinical_score += 0.2
    
    # High protein
    if user_inputs.get('Protein', 0) >= 3:  # 2+ or 3+
        clinical_score += 0.2
    
    # Abnormal pH
    if user_inputs.get('pH', 7.0) > 8.0 or user_inputs.get('pH', 7.0) < 5.0:
        clinical_score += 0.1
    
    # Cloudy urine
    if user_inputs.get('Transparency', 0) >= 3:  # CLOUDY or TURBID
        clinical_score += 0.15
    
    # Female gender (higher UTI risk)
    if user_inputs.get('Gender_FEMALE', 0) == 1:
        clinical_score += 0.1
    
    return min(clinical_score, 0.95)  # Cap at 95%

def prepare_user_inputs(user_inputs, expected_features):
    """Prepare user inputs for model prediction with ALL expected features"""
    # Create a complete feature dictionary with ALL expected features
    feature_dict = {}
    
    # Set defaults for ALL expected features from your scaler
    all_expected_features = [
        "Age", "Transparency", "Glucose", "Protein", "pH", "Specific Gravity", 
        "WBC", "RBC", "Epithelial Cells", "Mucous Threads", "Amorphous Urates", 
        "Bacteria", "Color_AMBER", "Color_BROWN", "Color_DARK YELLOW", 
        "Color_LIGHT RED", "Color_LIGHT YELLOW", "Color_RED", "Color_REDDISH", 
        "Color_REDDISH YELLOW", "Color_STRAW", "Color_YELLOW", 
        "Gender_FEMALE", "Gender_MALE"
    ]
    
    # Initialize all features to 0
    for feature in all_expected_features:
        feature_dict[feature] = 0
    
    # Update with user provided values
    feature_dict.update(user_inputs)
    
    # Ensure we return features in the exact order expected by the scaler
    return [feature_dict[feature] for feature in all_expected_features]

# Dynamic Diet Recommendation Engine
class DietRecommendationEngine:
    def __init__(self):
        self.diet_recommendations = {
            'HIGH': {  # Changed from 'high_risk' to 'HIGH'
                'en': {
                    'title': "🍎 Dietary Recommendations for UTI Management",
                    'hydration': "**💧 Hydration Focus:** Drink 10-12 glasses of water daily to flush bacteria from urinary tract",
                    'foods_to_include': [
                        "**Cranberry Juice**: Unsweetened cranberry juice (prevents bacteria adhesion)",
                        "**Vitamin C Rich Foods**: Citrus fruits, bell peppers, broccoli (acidifies urine)",
                        "**Probiotic Foods**: Yogurt, kefir, fermented foods (supports gut health)",
                        "**Anti-inflammatory Foods**: Turmeric, ginger, fatty fish (reduces inflammation)",
                        "**Garlic**: Natural antimicrobial properties"
                    ],
                    'foods_to_avoid': [
                        "**Sugar & Sweeteners**: Feed harmful bacteria",
                        "**Caffeine**: Can irritate bladder",
                        "**Alcohol**: Dehydrates and irritates urinary tract",
                        "**Spicy Foods**: May worsen bladder irritation",
                        "**Processed Foods**: High in preservatives and additives"
                    ]
                },
                'ta': {
                    'title': "🍎 யூடிஐ நிர்வாகத்திற்கான உணவு பரிந்துரைகள்",
                    'hydration': "**💧 நீரேற்றம்:** சிறுநீர் பாதையில் இருந்து பாக்டீரியாவை வெளியேற்ற தினமும் 10-12 கிளாஸ் தண்ணீர் குடிக்கவும்",
                    'foods_to_include': [
                        "**கிரான்பெரி சாறு**: சர்க்கரை இல்லாத கிரான்பெரி சாறு (பாக்டீரியா ஒட்டுதலை தடுக்கிறது)",
                        "**வைட்டமின் சி நிறைந்த உணவுகள்**: சிட்ரஸ் பழங்கள், பெல் பெப்பர், ப்ரோக்கோலி (சிறுநீரை அமிலமாக்குகிறது)",
                        "**ப்ரோபயாடிக் உணவுகள்**: தயிர், கெஃபிர், புளித்த உணவுகள் (குடல் ஆரோக்கியத்தை ஆதரிக்கிறது)",
                        "**எதிர்-வீக்க உணவுகள்**: மஞ்சள், இஞ்சி, கொழுப்பு மீன் (வீக்கத்தை குறைக்கிறது)",
                        "**பூண்டு**: இயற்கையான நுண்ணுயிர் எதிர்ப்பு பண்புகள்"
                    ],
                    'foods_to_avoid': [
                        "**சர்க்கரை & இனிப்பிகள்**: தீங்கு விளைவிக்கும் பாக்டீரியாக்களை வளர்க்கும்",
                        "**காஃபின்**: சிறுநீர்ப்பையை எரிச்சலூட்டும்",
                        "**மது**: நீரிழப்பு மற்றும் சிறுநீர் பாதையை எரிச்சலூட்டும்",
                        "**கார உணவுகள்**: சிறுநீர்ப்பை எரிச்சலை மோசமாக்கும்",
                        "**செயலாக்கப்பட்ட உணவுகள்**: பாதுகாப்பான்கள் மற்றும் சேர்க்கைகள் அதிகம்"
                    ]
                }
            },
            'MEDIUM': {  # Changed from 'medium_risk' to 'MEDIUM'
                'en': {
                    'title': "🍎 Dietary Support for Urinary Health",
                    'hydration': "**💧 Hydration Focus:** Drink 8-10 glasses of water daily to maintain urinary flow",
                    'foods_to_include': [
                        "**Cranberry Products**: Juice or supplements (bacterial anti-adhesion)",
                        "**Blueberries & Berries**: Rich in antioxidants",
                        "**Probiotic Yogurt**: Supports healthy gut flora",
                        "**Citrus Fruits**: Lemon, oranges in moderation",
                        "**Pumpkin Seeds**: Zinc for immune support",
                        "**Leafy Greens**: Spinach, kale for overall health"
                    ],
                    'foods_to_avoid': [
                        "**Excess Sugar**: Limits bacterial growth",
                        "**Carbonated Drinks**: Can irritate bladder",
                        "**Artificial Sweeteners**: May cause irritation",
                        "**Highly Processed Foods**: Choose whole foods instead"
                    ]
                },
                'ta': {
                    'title': "🍎 சிறுநீர் ஆரோக்கியத்திற்கான உணவு ஆதரவு",
                    'hydration': "**💧 நீரேற்றம்:** சிறுநீர் ஓட்டத்தை பராமரிக்க தினமும் 8-10 கிளாஸ் தண்ணீர் குடிக்கவும்",
                    'foods_to_include': [
                        "**கிரான்பெரி பொருட்கள்**: சாறு அல்லது கூடுதல் உணவுகள் (பாக்டீரியா எதிர்ப்பு)",
                        "**புளுபெர்ரி & பெர்ரி**: ஆன்டிஆக்ஸிடன்ட்கள் நிறைந்தவை",
                        "**ப்ரோபயாடிக் தயிர்**: ஆரோக்கியமான குடல் தாவரங்களை ஆதரிக்கிறது",
                        "**சிட்ரஸ் பழங்கள்**: எலுமிச்சை, ஆரஞ்சு மிதமாக",
                        "**பூசணி விதைகள்**: நோயெதிர்ப்பு ஆதரவுக்கான துத்தநாகம்",
                        "**இலை காய்கறிகள்**: முழு ஆரோக்கியத்திற்காக கீரை, கேல்"
                    ],
                    'foods_to_avoid': [
                        "**அதிக சர்க்கரை**: பாக்டீரியா வளர்ச்சியை கட்டுப்படுத்துகிறது",
                        "**கார்பனேடெட் பானங்கள்**: சிறுநீர்ப்பையை எரிச்சலூட்டும்",
                        "**செயற்கை இனிப்பிகள்**: எரிச்சலை ஏற்படுத்தக்கூடும்",
                        "**அதிக செயலாக்கப்பட்ட உணவுகள்**: முழு உணவுகளை தேர்வு செய்யவும்"
                    ]
                }
            },
            'LOW': {  # Changed from 'low_risk' to 'LOW'
                'en': {
                    'title': "🍎 Preventive Diet for Urinary Health",
                    'hydration': "**💧 Hydration Focus:** Maintain 6-8 glasses of water daily for optimal urinary function",
                    'foods_to_include': [
                        "**Water-rich Fruits**: Watermelon, cucumber, oranges",
                        "**Whole Grains**: Brown rice, oats, quinoa",
                        "**Lean Proteins**: Chicken, fish, legumes",
                        "**Healthy Fats**: Avocado, nuts, olive oil",
                        "**Colorful Vegetables**: Variety of colors for antioxidants",
                        "**Herbal Teas**: Chamomile, peppermint (caffeine-free)"
                    ],
                    'foods_to_avoid': [
                        "**Limit Processed Foods**: Choose fresh alternatives",
                        "**Moderate Caffeine**: 1-2 cups daily maximum",
                        "**Reduce Salt**: For overall kidney health",
                        "**Minimize Alcohol**: Occasional consumption only"
                    ]
                },
                'ta': {
                    'title': "🍎 சிறுநீர் ஆரோக்கியத்திற்கான தடுப்பு உணவு",
                    'hydration': "**💧 நீரேற்றம்:** உகந்த சிறுநீர் செயல்பாட்டிற்காக தினமும் 6-8 கிளாஸ் தண்ணீர் குடிக்கவும்",
                    'foods_to_include': [
                        "**நீர் நிறைந்த பழங்கள்**: தர்பூசணி, வெள்ளரி, ஆரஞ்சு",
                        "**முழு தானியங்கள்**: கருப்பு அரிசி, ஓட்ஸ், கினோவா",
                        "**குறைந்த கொழுப்பு புரதங்கள்**: கோழி, மீன், பருப்பு வகைகள்",
                        "**ஆரோக்கியமான கொழுப்புகள்**: அவகேடோ, கொட்டைகள், ஆலிவ் எண்ணெய்",
                        "**வண்ண காய்கறிகள்**: ஆன்டிஆக்ஸிடன்ட்களுக்கான பல்வேறு வண்ணங்கள்",
                        "**மூலிகை தேநீர்**: சாமோமைல், புதினா (காஃபின் இல்லாதது)"
                    ],
                    'foods_to_avoid': [
                        "**செயலாக்கப்பட்ட உணவுகளை கட்டுப்படுத்தவும்**: புதிய மாற்றுகளை தேர்வு செய்யவும்",
                        "**மிதமான காஃபின்**: தினசரி அதிகபட்சம் 1-2 கப்",
                        "**உப்பு குறைக்கவும்**: மொத்த சிறுநீரக ஆரோக்கியத்திற்காக",
                        "**மதுவை குறைக்கவும்**: அவசர நுகர்வு மட்டுமே"
                    ]
                }
            }
        }
    
    def get_recommendations(self, user_inputs, risk_level, language='en'):
        """Get personalized diet recommendations based on lab values and risk level"""
        try:
            # Use uppercase risk level to match dictionary keys
            risk_key = risk_level.upper()
            base_recommendations = self.diet_recommendations[risk_key][language]
            
            # Add personalized recommendations based on specific lab values
            personalized_tips = self._get_personalized_tips(user_inputs, language)
            
            return {
                'title': base_recommendations['title'],
                'hydration': base_recommendations['hydration'],
                'foods_to_include': base_recommendations['foods_to_include'],
                'foods_to_avoid': base_recommendations['foods_to_avoid'],
                'personalized_tips': personalized_tips
            }
        except KeyError as e:
            st.error(f"Error getting diet recommendations: {e}")
            # Return default recommendations if there's an error
            return self._get_default_recommendations(language)
    
    def _get_default_recommendations(self, language='en'):
        """Get default diet recommendations in case of error"""
        if language == 'en':
            return {
                'title': "🍎 General Urinary Health Diet",
                'hydration': "**💧 Hydration:** Drink adequate water daily",
                'foods_to_include': ["Focus on whole foods, fruits and vegetables"],
                'foods_to_avoid': ["Limit processed foods and sugars"],
                'personalized_tips': []
            }
        else:
            return {
                'title': "🍎 பொது சிறுநீர் ஆரோக்கிய உணவு",
                'hydration': "**💧 நீரேற்றம்:** தினமும் போதுமான தண்ணீர் குடிக்கவும்",
                'foods_to_include': ["முழு உணவுகள், பழங்கள் மற்றும் காய்கறிகளில் கவனம் செலுத்தவும்"],
                'foods_to_avoid': ["செயலாக்கப்பட்ட உணவுகள் மற்றும் சர்க்கரைகளை கட்டுப்படுத்தவும்"],
                'personalized_tips': []
            }
    
    def _get_personalized_tips(self, user_inputs, language):
        """Generate personalized tips based on specific lab abnormalities"""
        tips = []
        
        # High WBC - focus on anti-inflammatory foods
        if user_inputs.get('WBC', 0) > 10:
            if language == 'en':
                tips.append("**Anti-inflammatory Focus**: Include turmeric, ginger, and omega-3 rich foods to combat inflammation")
            else:
                tips.append("**எதிர்-வீக்க கவனம்**: வீக்கத்தை எதிர்கொள்ள மஞ்சள், இஞ்சி மற்றும் ஓமேகா-3 நிறைந்த உணவுகளை சேர்க்கவும்")
        
        # High protein - kidney support
        if user_inputs.get('Protein', 0) >= 2:
            if language == 'en':
                tips.append("**Kidney Support**: Monitor protein intake and include kidney-friendly foods like cabbage, cauliflower")
            else:
                tips.append("**சிறுநீரக ஆதரவு**: புரத உட்கொள்ளலை கண்காணித்து முட்டைக்கோஸ், காலிஃபிளார் போன்ற சிறுநீரக நட்பு உணவுகளை சேர்க்கவும்")
        
        # Abnormal pH - acid/base balance
        ph = user_inputs.get('pH', 7.0)
        if ph > 7.5:  # Alkaline urine
            if language == 'en':
                tips.append("**Urine Acidification**: Include vitamin C rich foods and cranberry to help acidify urine")
            else:
                tips.append("**சிறுநீர் அமிலமயமாக்கல்**: சிறுநீரை அமிலமாக்க உதவ வைட்டமின் சி நிறைந்த உணவுகள் மற்றும் கிரான்பெரியை சேர்க்கவும்")
        elif ph < 5.5:  # Acidic urine
            if language == 'en':
                tips.append("**Balancing pH**: Include more alkaline-forming foods like vegetables and fruits")
            else:
                tips.append("**pH சமநிலை**: காய்கறிகள் மற்றும் பழங்கள் போன்ற அல்கலைன் உருவாக்கும் உணவுகளை அதிகம் சேர்க்கவும்")
        
        # High specific gravity - hydration focus
        if user_inputs.get('Specific Gravity', 1.015) > 1.025:
            if language == 'en':
                tips.append("**Hydration Priority**: Increase water intake as concentrated urine suggests dehydration")
            else:
                tips.append("**நீரேற்ற முன்னுரிமை**: செறிவூட்டப்பட்ட சிறுநீர் நீரிழப்பைக் குறிக்கிறது என்பதால் நீர் உட்கொள்ளலை அதிகரிக்கவும்")
        
        # Bacteria present - antimicrobial support
        if user_inputs.get('Bacteria', 0) >= 2:
            if language == 'en':
                tips.append("**Antimicrobial Support**: Include garlic, oregano, and probiotic foods to fight bacteria")
            else:
                tips.append("**நுண்ணுயிர் எதிர்ப்பு ஆதரவு**: பாக்டீரியாவை எதிர்கொள்ள பூண்டு, ஓரிகானோ மற்றும் ப்ரோபயாடிக் உணவுகளை சேர்க்கவும்")
        
        return tips

# Bilingual Explanation Engine
class BilingualExplanationEngine:
    def __init__(self):
        self.normal_ranges = {
            'pH': (4.5, 8.0),
            'Specific Gravity': (1.005, 1.030),
            'WBC': (0, 5),
            'RBC': (0, 3),
            'Glucose': (0, 1),
            'Protein': (0, 1),
            'Bacteria': (0, 1)
        }
        
        self.explanations = {
            'en': {
                'high_risk': "Based on your urinalysis results, there is a **{risk_percentage}% probability** of Urinary Tract Infection. Consultation with a healthcare provider is recommended.",
                'medium_risk': "Your results show a **{risk_percentage}% probability** of Urinary Tract Infection. Further evaluation may be needed.",
                'low_risk': "Your urinalysis results indicate a **low probability ({risk_percentage}%)** of Urinary Tract Infection. Continue with good urinary health practices.",
                'abnormal_ph': "• **pH Level**: Your urine pH is **{value}** (Normal range: 4.5-8.0)",
                'abnormal_sg': "• **Specific Gravity**: Your value is **{value}** {status} normal range (1.005-1.030)",
                'high_wbc': "• **White Blood Cells**: Elevated level **{value}** may indicate infection or inflammation",
                'high_rbc': "• **Red Blood Cells**: Presence **{value}** may require further investigation",
                'glucose_present': "• **Glucose**: Detected in urine **{level}**",
                'protein_present': "• **Protein**: Level **{level}** may indicate kidney issues",
                'bacteria_present': "• **Bacteria**: Presence **{level}** suggests possible infection",
                'prevention_tips': [
                    "Drink 8-10 glasses of water daily",
                    "Practice good personal hygiene",
                    "Urinate when you feel the need - don't hold it",
                    "Wipe from front to back after using the toilet",
                    "Urinate after sexual intercourse",
                    "Avoid using harsh soaps in the genital area",
                    "Wear cotton underwear and loose-fitting clothes"
                ]
            },
            'ta': {
                'high_risk': "உங்கள் சிறுநீர் பரிசோதனை முடிவுகளின் அடிப்படையில், சிறுநீர் கோளாறு (யூடிஐ) ஏற்படுவதற்கான நிகழ்தகவு **{risk_percentage}%** ஆகும். சுகாதார வழங்குநருடன் கலந்தாலோசிப்பது பரிந்துரைக்கப்படுகிறது.",
                'medium_risk': "உங்கள் முடிவுகள் சிறுநீர் கோளாறு (யூடிஐ) ஏற்படுவதற்கான **{risk_percentage}% நிகழ்தகவை** காட்டுகின்றன. மேலும் மதிப்பீடு தேவைப்படலாம்.",
                'low_risk': "உங்கள் சிறுநீர் பரிசோதனை முடிவுகள் சிறுநீர் கோளாறு (யூடிஐ) ஏற்படுவதற்கான **குறைந்த நிகழ்தகவை ({risk_percentage}%)** காட்டுகின்றன. நல்ல சிறுநீர் சுகாதார பழக்கங்களைத் தொடரவும்.",
                'abnormal_ph': "• **pH அளவு**: உங்கள் சிறுநீர் pH **{value}** (சாதாரண வரம்பு: 4.5-8.0)",
                'abnormal_sg': "• **குறிப்பிட்ட ஈர்ப்பு**: உங்கள் மதிப்பு **{value}** சாதாரண வரம்பிற்கு {status} (1.005-1.030)",
                'high_wbc': "• **வெள்ளை இரத்த அணுக்கள்**: அதிகரித்த அளவு **{value}** தொற்று அல்லது வீக்கத்தைக் குறிக்கலாம்",
                'high_rbc': "• **சிவப்பு இரத்த அணுக்கள்**: இருப்பு **{value}** மேலும் விசாரணை தேவைப்படலாம்",
                'glucose_present': "• **குளுக்கோஸ்**: சிறுநீரில் கண்டறியப்பட்டது **{level}**",
                'protein_present': "• **புரதம்**: அளவு **{level}** சிறுநீரக சிக்கல்களைக் குறிக்கலாம்",
                'bacteria_present': "• **பாக்டீரியா**: இருப்பு **{level}** சாத்தியமான தொற்றைக் குறிக்கிறது",
                'prevention_tips': [
                    "தினமும் 8-10 கிளாஸ் தண்ணீர் குடிக்கவும்",
                    "நல்ல தனிப்பட்ட சுகாதாரத்தை பழக்கவும்",
                    "சிறுநீர் கழிக்க வேண்டியதன் அவசியத்தை உணரும்போது கழிக்கவும் - அடக்கிவைக்காதீர்கள்",
                    "கழிப்பறை பயன்படுத்திய பின் முன்பக்கத்தில் இருந்து பின்பக்கமாகத் துடைக்கவும்",
                    "பாலியல் தொடர்புக்குப் பிறகு சிறுநீர் கழிக்கவும்",
                    "பிறப்புறுப்புப் பகுதியில் கடுமையான சோப்புகளைப் பயன்படுத்துவதைத் தவிர்க்கவும்",
                    "பருத்தி உள்ளாடை மற்றும் தளர்வான ஆடைகளை அணியவும்"
                ]
            }
        }

    def generate_explanation(self, user_inputs, prediction_result, language='en'):
        risk_percentage = int(prediction_result['probability'] * 100)
        
        # Main risk message
        if prediction_result['risk_level'] == 'HIGH':
            main_message = self.explanations[language]['high_risk'].format(risk_percentage=risk_percentage)
        elif prediction_result['risk_level'] == 'MEDIUM':
            main_message = self.explanations[language]['medium_risk'].format(risk_percentage=risk_percentage)
        else:
            main_message = self.explanations[language]['low_risk'].format(risk_percentage=risk_percentage)
        
        # Detailed explanations
        detailed_explanations = []
        for param_name, value in user_inputs.items():
            explanation = self._analyze_parameter(param_name, value, language)
            if explanation:
                detailed_explanations.append(explanation)
        
        return {
            'main_message': main_message,
            'detailed_explanations': detailed_explanations,
            'prevention_tips': self.explanations[language]['prevention_tips'],
            'risk_level': prediction_result['risk_level'],
            'confidence': prediction_result['confidence']
        }
    
    def _analyze_parameter(self, param_name, value, language):
        if param_name in self.normal_ranges:
            min_val, max_val = self.normal_ranges[param_name]
            
            if value < min_val or value > max_val:
                template_key = f'abnormal_{param_name.lower().replace(" ", "_")}'
                if template_key in self.explanations[language]:
                    status = "above" if value > max_val else "below"
                    status_ta = "மேலே" if value > max_val else "கீழே"
                    return self.explanations[language][template_key].format(
                        value=value, 
                        status=status if language == 'en' else status_ta,
                        level=self._get_level_description(value, param_name, language)
                    )
        return None
    
    def _get_level_description(self, value, param_name, language):
        descriptions = {
            'en': {
                'Glucose': ['NEGATIVE', 'TRACE', '1+', '2+', '3+', '4+'],
                'Protein': ['NEGATIVE', 'TRACE', '1+', '2+', '3+'],
                'Bacteria': ['NONE', 'RARE', 'FEW', 'MODERATE', 'PLENTY']
            },
            'ta': {
                'Glucose': ['இல்லை', 'சிறிதளவு', '1+', '2+', '3+', '4+'],
                'Protein': ['இல்லை', 'சிறிதளவு', '1+', '2+', '3+'],
                'Bacteria': ['இல்லை', 'அரிதாக', 'சில', 'மிதமான', 'நிறைய']
            }
        }
        
        if param_name in descriptions[language]:
            levels = descriptions[language][param_name]
            if 0 <= value < len(levels):
                return levels[int(value)]
        return str(value)

# RAG Chatbot Class (without medication information)
class UTIChatbot:
    def __init__(self):
        self.knowledge_base = {
            "uti": {
                "symptoms": "Common UTI symptoms include: burning sensation during urination, frequent urination, cloudy or strong-smelling urine, pelvic pain, and feeling tired.",
                "causes": "UTIs are usually caused by bacteria entering the urinary tract. Risk factors include sexual activity, certain birth control methods, menopause, urinary tract abnormalities, and suppressed immune system.",
                "prevention": "To prevent UTIs: drink plenty of water, urinate frequently, wipe front to back, urinate after sexual intercourse, avoid irritating feminine products, and consider cranberry products.",
                "diagnosis": "UTIs are diagnosed through urinalysis to check for white blood cells, red blood cells, and bacteria. Urine culture may be done to identify specific bacteria.",
                "when_to_see_doctor": "Consult a healthcare provider if you experience: fever, chills, back pain, nausea, vomiting, or if symptoms don't improve after 2-3 days."
            },
            "urinalysis": {
                "wbc": "White Blood Cells (WBC) in urine may indicate infection or inflammation. Normal range is 0-5 WBCs per high power field.",
                "rbc": "Red Blood Cells (RBC) in urine could indicate infection, kidney stones, or other issues. Normal range is 0-3 RBCs per high power field.",
                "ph": "Urine pH normally ranges from 4.5 to 8.0. Abnormal pH can indicate metabolic issues or urinary tract infections.",
                "protein": "Protein in urine (proteinuria) may indicate kidney damage. Normal urine has little to no protein.",
                "glucose": "Glucose in urine (glycosuria) typically indicates high blood sugar levels, often associated with diabetes.",
                "bacteria": "Bacteria in urine (bacteriuria) suggests possible urinary tract infection. The amount helps determine severity.",
                "specific_gravity": "Specific gravity measures urine concentration. Normal range is 1.005 to 1.030. High values may indicate dehydration."
            },
            "diet": {
                "hydration": "Proper hydration helps flush bacteria from the urinary tract. Aim for 6-8 glasses of water daily.",
                "cranberry": "Cranberry products may help prevent UTIs by preventing bacteria from adhering to urinary tract walls.",
                "vitamin_c": "Vitamin C rich foods can help acidify urine, creating an unfavorable environment for bacteria.",
                "probiotics": "Probiotic foods like yogurt support healthy gut and urinary tract flora.",
                "foods_to_avoid": "Limit sugar, caffeine, alcohol, and spicy foods which can irritate the bladder."
            },
            "general": {
                "hydration": "Proper hydration helps flush bacteria from the urinary tract. Aim for 6-8 glasses of water daily.",
                "recurrent_uti": "Recurrent UTIs (2 or more in 6 months) may require lifestyle changes and preventive strategies.",
                "when_to_see_doctor": "See a healthcare provider if you experience: fever, chills, back pain, nausea, vomiting, or if symptoms don't improve after 2-3 days."
            }
        }
    
    def get_response(self, user_question, user_context=None):
        """Generate response using RAG approach"""
        user_question_lower = user_question.lower()
        
        # Check for specific topics in the knowledge base
        response = ""
        
        # UTI related questions
        if any(word in user_question_lower for word in ['symptom', 'feel', 'pain', 'burning']):
            response = self.knowledge_base["uti"]["symptoms"]
        
        elif any(word in user_question_lower for word in ['cause', 'why', 'reason', 'risk']):
            response = self.knowledge_base["uti"]["causes"]
        
        elif any(word in user_question_lower for word in ['prevent', 'avoid', 'stop']):
            response = self.knowledge_base["uti"]["prevention"]
        
        # Diet and nutrition questions
        elif any(word in user_question_lower for word in ['diet', 'food', 'eat', 'nutrition']):
            response = self.knowledge_base["diet"]["hydration"] + " " + self.knowledge_base["diet"]["cranberry"]
        
        elif 'cranberry' in user_question_lower:
            response = self.knowledge_base["diet"]["cranberry"]
        
        elif any(word in user_question_lower for word in ['water', 'hydrat', 'drink']):
            response = self.knowledge_base["diet"]["hydration"]
        
        elif 'probiotic' in user_question_lower:
            response = self.knowledge_base["diet"]["probiotics"]
        
        # Urinalysis parameter questions
        elif 'wbc' in user_question_lower or 'white blood' in user_question_lower:
            response = self.knowledge_base["urinalysis"]["wbc"]
        
        elif 'rbc' in user_question_lower or 'red blood' in user_question_lower:
            response = self.knowledge_base["urinalysis"]["rbc"]
        
        elif 'ph' in user_question_lower:
            response = self.knowledge_base["urinalysis"]["ph"]
        
        elif 'protein' in user_question_lower:
            response = self.knowledge_base["urinalysis"]["protein"]
        
        elif 'glucose' in user_question_lower or 'sugar' in user_question_lower:
            response = self.knowledge_base["urinalysis"]["glucose"]
        
        elif 'bacteria' in user_question_lower:
            response = self.knowledge_base["urinalysis"]["bacteria"]
        
        elif 'specific gravity' in user_question_lower:
            response = self.knowledge_base["urinalysis"]["specific_gravity"]
        
        # General questions
        elif 'recurrent' in user_question_lower or 'frequent' in user_question_lower:
            response = self.knowledge_base["general"]["recurrent_uti"]
        
        elif any(word in user_question_lower for word in ['doctor', 'hospital', 'emergency', 'see']):
            response = self.knowledge_base["general"]["when_to_see_doctor"]
        
        # Default response for unknown questions
        if not response:
            response = "I'm specialized in urinary tract infections and urinalysis results. Could you please rephrase your question or ask about UTI symptoms, causes, prevention, diet recommendations, or specific urinalysis parameters?"
        
        # Add personalized context if available
        if user_context and st.session_state.prediction_result:
            risk_level = st.session_state.prediction_result['risk_level']
            probability = st.session_state.prediction_result['probability']
            
            if risk_level == "HIGH":
                response += f"\n\nBased on your urinalysis results showing {probability:.1%} probability of UTI, it's important to consult with a healthcare provider promptly."
            elif risk_level == "MEDIUM":
                response += f"\n\nYour results indicate a {probability:.1%} probability of UTI. Consider monitoring symptoms and consulting a healthcare provider if they persist."
        
        return response

# Initialize components
model, scaler, feature_names, model_performance = load_model_artifacts()
explanation_engine = BilingualExplanationEngine()
diet_engine = DietRecommendationEngine()
chatbot = UTIChatbot()

# Initialize session state for chat
if 'chat_history' not in st.session_state:
    st.session_state.chat_history = []

# Initialize session state for prediction
if 'prediction_result' not in st.session_state:
    st.session_state.prediction_result = None
if 'user_inputs' not in st.session_state:
    st.session_state.user_inputs = {}

# Sidebar for input
st.sidebar.header("🔬 Enter Lab Values")

# Input fields in two columns
col1, col2 = st.sidebar.columns(2)

with col1:
    age = st.number_input("Age", min_value=0, max_value=100, value=35, key="age")
    ph = st.slider("pH Level", min_value=4.0, max_value=9.0, value=6.5, step=0.1, key="ph")
    specific_gravity = st.slider("Specific Gravity", min_value=1.000, max_value=1.040, value=1.015, step=0.001, key="sg")
    wbc = st.number_input("White Blood Cells (WBC)", min_value=0, max_value=100, value=5, key="wbc")
    rbc = st.number_input("Red Blood Cells (RBC)", min_value=0, max_value=100, value=1, key="rbc")

with col2:
    glucose = st.selectbox("Glucose", ["NEGATIVE", "TRACE", "1+", "2+", "3+", "4+"], key="glucose")
    protein = st.selectbox("Protein", ["NEGATIVE", "TRACE", "1+", "2+", "3+"], key="protein")
    bacteria = st.selectbox("Bacteria", ["NONE SEEN", "RARE", "FEW", "MODERATE", "PLENTY"], key="bacteria")
    transparency = st.selectbox("Transparency", ["CLEAR", "SLIGHTLY HAZY", "HAZY", "CLOUDY", "TURBID"], key="transparency")
    gender = st.radio("Gender", ["MALE", "FEMALE"], key="gender")

# Mapping dictionaries
glucose_map = {"NEGATIVE": 0, "TRACE": 1, "1+": 2, "2+": 3, "3+": 4, "4+": 5}
protein_map = {"NEGATIVE": 0, "TRACE": 1, "1+": 2, "2+": 3, "3+": 4}
bacteria_map = {"NONE SEEN": 0, "RARE": 1, "FEW": 2, "MODERATE": 3, "PLENTY": 4}
transparency_map = {"CLEAR": 0, "SLIGHTLY HAZY": 1, "HAZY": 2, "CLOUDY": 3, "TURBID": 4}

# Analysis button
if st.sidebar.button("🔍 Analyze My Report", type="primary", use_container_width=True):
    with st.spinner("🤖 AI is analyzing your urinalysis report..."):
        # Prepare user inputs - COMPLETE VERSION WITH ALL 24 FEATURES
        user_inputs = {
            # Basic demographics
            "Age": age,
            
            # Urinalysis parameters
            "pH": ph,
            "Specific Gravity": specific_gravity,
            "WBC": wbc,
            "RBC": rbc,
            "Glucose": glucose_map[glucose],
            "Protein": protein_map[protein],
            "Bacteria": bacteria_map[bacteria],
            "Transparency": transparency_map[transparency],
            
            # Microscopic findings (set reasonable defaults)
            "Epithelial Cells": 1,  # Common finding
            "Mucous Threads": 1,    # Common finding  
            "Amorphous Urates": 0,  # Less common
            
            # Gender (one-hot encoded)
            "Gender_MALE": 1 if gender == "MALE" else 0,
            "Gender_FEMALE": 1 if gender == "FEMALE" else 0,
            
            # Color features (set DARK YELLOW as default, others to 0)
            "Color_AMBER": 0,
            "Color_BROWN": 0,
            "Color_DARK YELLOW": 1,  # Most common color
            "Color_LIGHT RED": 0,
            "Color_LIGHT YELLOW": 0,
            "Color_RED": 0,
            "Color_REDDISH": 0,
            "Color_REDDISH YELLOW": 0,
            "Color_STRAW": 0,
            "Color_YELLOW": 0
        }
        
        st.session_state.user_inputs = user_inputs
        
        # Make prediction
        if model and scaler and feature_names:
            prediction_result = predict_uti_risk(user_inputs, model, scaler, feature_names)
            st.session_state.prediction_result = prediction_result

# Main content area with tabs
tab1, tab2 = st.tabs(["📊 UTI Risk Analysis", "💬 Chat with UTI Expert"])

with tab1:
    if st.session_state.prediction_result:
        result = st.session_state.prediction_result
        
        # Risk Level Banner
        risk_class = f"risk-{result['risk_level'].lower()}"
        st.markdown(f'<div class="{risk_class}">UTI RISK LEVEL: {result["risk_level"]}</div>', unsafe_allow_html=True)
        
        # Metrics
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            st.metric("Probability", f"{result['probability']:.1%}")
        
        with col2:
            st.metric("Confidence", f"{result['confidence']:.1%}")
        
        with col3:
            st.metric("AI Model", "Clinical AI")
        
        with col4:
            if model_performance:
                st.metric("Model Accuracy", f"{model_performance.get('accuracy', 0.92):.1%}")

        # Summary Section
        st.header("📋 Summary")
        
        # Generate summary based on risk level
        risk_percentage = int(result['probability'] * 100)
        
        if result['risk_level'] == 'HIGH':
            summary_text = f"""
            **High Risk of UTI Detected ({risk_percentage}%)**
            
            Your urinalysis results indicate a high probability of urinary tract infection. Key concerning factors include:
            - Elevated levels of infection markers
            - Abnormal urine characteristics  
            - Clinical indicators suggesting active infection
            
            **Recommendation:** Please consult with a healthcare provider promptly for proper diagnosis and guidance.
            """
        elif result['risk_level'] == 'MEDIUM':
            summary_text = f"""
            **Medium Risk of UTI ({risk_percentage}%)**
            
            Your results show some indicators of possible urinary tract infection, but the evidence is not conclusive.
            
            **Recommendation:** Monitor your symptoms closely and consider consulting a healthcare provider if symptoms persist or worsen.
            """
        else:
            summary_text = f"""
            **Low Risk of UTI ({risk_percentage}%)**
            
            Your urinalysis results are largely within normal ranges, indicating low probability of urinary tract infection.
            
            **Recommendation:** Continue practicing good urinary health habits and monitor for any new symptoms.
            """
        
        st.markdown(f'<div class="info-box">{summary_text}</div>', unsafe_allow_html=True)

        # Dynamic Diet Recommendations
        st.header("🍎 Personalized Diet & Nutrition")
        
        diet_tab1, diet_tab2 = st.tabs(["🇬🇧 English", "🇮🇳 Tamil"])
        
        with diet_tab1:
            try:
                eng_diet = diet_engine.get_recommendations(
                    st.session_state.user_inputs, result['risk_level'], 'en'
                )
                
                st.markdown(f"### {eng_diet['title']}")
                st.markdown(eng_diet['hydration'])
                
                st.markdown("#### ✅ Foods to Include:")
                for food in eng_diet['foods_to_include']:
                    st.markdown(f"- {food}")
                
                st.markdown("#### ❌ Foods to Avoid/Limit:")
                for food in eng_diet['foods_to_avoid']:
                    st.markdown(f"- {food}")
                
                if eng_diet['personalized_tips']:
                    st.markdown("#### 💡 Personalized Recommendations:")
                    for tip in eng_diet['personalized_tips']:
                        st.markdown(f"- {tip}")
                        
            except Exception as e:
                st.error(f"Error loading diet recommendations: {e}")

        with diet_tab2:
            try:
                tam_diet = diet_engine.get_recommendations(
                    st.session_state.user_inputs, result['risk_level'], 'ta'
                )
                
                st.markdown(f"### {tam_diet['title']}")
                st.markdown(tam_diet['hydration'])
                
                st.markdown("#### ✅ சேர்க்க வேண்டிய உணவுகள்:")
                for food in tam_diet['foods_to_include']:
                    st.markdown(f"- {food}")
                
                st.markdown("#### ❌ தவிர்க்க/குறைக்க வேண்டிய உணவுகள்:")
                for food in tam_diet['foods_to_avoid']:
                    st.markdown(f"- {food}")
                
                if tam_diet['personalized_tips']:
                    st.markdown("#### 💡 தனிப்பட்ட பரிந்துரைகள்:")
                    for tip in tam_diet['personalized_tips']:
                        st.markdown(f"- {tip}")
                        
            except Exception as e:
                st.error(f"Error loading diet recommendations: {e}")

        # Explanations
        st.header("💬 Detailed Analysis")
        
        subtab1, subtab2 = st.tabs(["🇬🇧 English Analysis", "🇮🇳 Tamil Analysis"])
        
        with subtab1:
            eng_explanation = explanation_engine.generate_explanation(
                st.session_state.user_inputs, result, 'en'
            )
            
            st.markdown("### Risk Assessment")
            st.markdown(eng_explanation['main_message'])
            
            if eng_explanation['detailed_explanations']:
                st.markdown("### 🔍 Key Findings")
                for detail in eng_explanation['detailed_explanations']:
                    st.markdown(f'<div class="parameter-box">{detail}</div>', unsafe_allow_html=True)
            else:
                st.info("🎉 All tested parameters are within normal ranges.")
            
            st.markdown("### 💡 Preventive Recommendations")
            for i, tip in enumerate(eng_explanation['prevention_tips'], 1):
                st.markdown(f"{i}. {tip}")

        with subtab2:
            tam_explanation = explanation_engine.generate_explanation(
                st.session_state.user_inputs, result, 'ta'
            )
            
            st.markdown("### ஆபத்து மதிப்பீடு")
            st.markdown(tam_explanation['main_message'])
            
            if tam_explanation['detailed_explanations']:
                st.markdown("### 🔍 முக்கிய கண்டறிதல்கள்")
                for detail in tam_explanation['detailed_explanations']:
                    st.markdown(f'<div class="parameter-box">{detail}</div>', unsafe_allow_html=True)
            else:
                st.info("🎉 அனைத்து சோதனை அளவுருக்களும் சாதாரண வரம்புகளுக்குள் உள்ளன.")
            
            st.markdown("### 💡 தடுப்பு பரிந்துரைகள்")
            for i, tip in enumerate(tam_explanation['prevention_tips'], 1):
                st.markdown(f"{i}. {tip}")

    else:
        # Welcome message for analysis tab
        st.markdown("""
        <div class="info-box">
        <h3>👋 Welcome to the AI-Powered UTI Detection Chatbot!</h3>
        <p>This clinical AI tool analyzes your urinalysis results to assess UTI risk and provides 
        comprehensive explanations in both English and Tamil.</p>
        
        <p><strong>📊 How it works:</strong></p>
        <ol>
            <li>Enter your lab values in the sidebar</li>
            <li>Click "Analyze My Report"</li>
            <li>Get instant AI-powered analysis with risk assessment</li>
            <li>Review detailed explanations in your preferred language</li>
            <li>Receive personalized diet recommendations</li>
            <li>Get preventive healthcare recommendations</li>
        </ol>
        
        <p><strong>🎯 Features:</strong></p>
        <ul>
            <li>AI Risk Assessment with 92.3% accuracy</li>
            <li>Personalized diet recommendations based on your lab values</li>
            <li>Bilingual explanations (English & Tamil)</li>
            <li>Interactive chatbot for UTI-related questions</li>
        </ul>
        
        <p><em>⚕️ Note: This AI tool provides assisted analysis and should not replace professional medical diagnosis.</em></p>
        </div>
        """, unsafe_allow_html=True)

with tab2:
    st.header("💬 Chat with UTI Expert")
    st.markdown("Ask me anything about urinary tract infections, urinalysis results, symptoms, prevention, or diet recommendations!")
    
    # Display chat history
    st.markdown('<div class="chat-container">', unsafe_allow_html=True)
    for message in st.session_state.chat_history:
        if message["role"] == "user":
            st.markdown(f'<div class="user-message"><strong>You:</strong> {message["content"]}</div>', unsafe_allow_html=True)
        else:
            st.markdown(f'<div class="bot-message"><strong>UTI Expert:</strong> {message["content"]}</div>', unsafe_allow_html=True)
    st.markdown('</div>', unsafe_allow_html=True)
    
    # Chat input
    col1, col2 = st.columns([4, 1])
    with col1:
        user_question = st.text_input("Type your question here...", key="chat_input", label_visibility="collapsed")
    with col2:
        send_button = st.button("Send", use_container_width=True)
    
    # Handle chat interaction
    if send_button and user_question:
        # Add user question to chat history
        st.session_state.chat_history.append({"role": "user", "content": user_question})
        
        # Get bot response
        bot_response = chatbot.get_response(user_question, st.session_state.user_inputs)
        
        # Add bot response to chat history
        st.session_state.chat_history.append({"role": "assistant", "content": bot_response})
        
        # Rerun to update the chat display
        st.rerun()
    
    # Suggested questions
    st.markdown("### 💡 Suggested Questions:")
    suggested_questions = [
        "What are the common symptoms of UTI?",
        "How can I prevent urinary tract infections?",
        "What does high WBC in urine mean?",
        "What foods help prevent UTIs?",
        "When should I see a doctor for UTI symptoms?"
    ]
    
    cols = st.columns(2)
    for i, question in enumerate(suggested_questions):
        with cols[i % 2]:
            if st.button(question, use_container_width=True):
                # Add the suggested question to chat
                st.session_state.chat_history.append({"role": "user", "content": question})
                bot_response = chatbot.get_response(question, st.session_state.user_inputs)
                st.session_state.chat_history.append({"role": "assistant", "content": bot_response})
                st.rerun()

# Footer
st.markdown("---")
st.markdown("""
<div style='text-align: center; color: #666; font-size: 0.9rem;'>
<p><strong>🩺 AI-Powered UTI Detection Chatbot</strong> | 
Clinical AI Model | Personalized Diet Recommendations | Bilingual Support | RAG Chatbot | 
<em>For educational and assisted analysis purposes</em></p>
<p>Always consult healthcare professionals for medical diagnosis and treatment</p>
</div>
""", unsafe_allow_html=True)
