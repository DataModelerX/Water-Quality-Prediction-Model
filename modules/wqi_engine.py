"""
Water Quality Index (WQI) Engine and CPCB Designated Best Use Classification.
Recognized Methodology: Weighted Arithmetic Water Quality Index (WA-WQI) (Brown et al., 1970; Horton, 1965)
aligned with Bureau of Indian Standards (BIS 10500:2012) and Central Pollution Control Board (CPCB) standards.
"""

import numpy as np
import pandas as pd

# Standard BIS 10500:2012 / CPCB Primary Water Quality Criteria for Indian Surface Water
# Si = Recommended Standard Permissible Value
# V_ideal = Ideal Value for natural pure water
# Si weight proportional factor
WQI_STANDARDS = {
    "pH": {
        "standard": 8.5,
        "ideal": 7.0,
        "weight_factor": 8.5,
        "unit": "pH units",
        "description": "Acidity / Basicity (BIS 10500: 6.5 - 8.5)"
    },
    "DO": {
        "standard": 5.0,
        "ideal": 14.6,
        "weight_factor": 5.0,
        "unit": "mg/L",
        "description": "Dissolved Oxygen (CPCB Class B/C: >= 5.0 mg/L; pure saturated: 14.6 mg/L)"
    },
    "BOD": {
        "standard": 3.0,
        "ideal": 0.0,
        "weight_factor": 3.0,
        "unit": "mg/L",
        "description": "Biochemical Oxygen Demand 3 days at 27C (CPCB: <= 3.0 mg/L)"
    },
    "Conductivity": {
        "standard": 1000.0,
        "ideal": 0.0,
        "weight_factor": 1000.0,
        "unit": "µmho/cm",
        "description": "Electrical Conductivity (WHO/CPCB drinking limit: 1000 µmho/cm)"
    },
    "Nitrate": {
        "standard": 45.0,
        "ideal": 0.0,
        "weight_factor": 45.0,
        "unit": "mg/L",
        "description": "Nitrate as NO3 (BIS 10500: 45.0 mg/L)"
    },
    "Total_Coliform": {
        "standard": 5000.0,
        "ideal": 0.0,
        "weight_factor": 5000.0,
        "unit": "MPN/100ml",
        "description": "Total Coliforms (CPCB Class C criterion: <= 5000 MPN/100ml)"
    }
}

# Compute proportionality constant K = 1 / sum(1 / Si)
_sum_inv = sum(1.0 / v["weight_factor"] for v in WQI_STANDARDS.values())
K_CONSTANT = 1.0 / _sum_inv

# Unit weight Wi = K / Si
UNIT_WEIGHTS = {p: K_CONSTANT / v["weight_factor"] for p, v in WQI_STANDARDS.items()}

def calculate_sub_index(param, value):
    """
    Computes quality rating sub-index Qi for a parameter:
    Qi = 100 * |Vi - V_ideal| / |Si - V_ideal|
    """
    if pd.isnull(value):
        return np.nan
        
    std_info = WQI_STANDARDS.get(param)
    if not std_info:
        return np.nan
        
    s_val = std_info["standard"]
    ideal_val = std_info["ideal"]
    
    if param == "pH":
        # pH ideal is 7.0, tolerance band is 7.0 to 8.5
        q = 100.0 * abs(value - 7.0) / (8.5 - 7.0)
    elif param == "DO":
        # DO ideal is 14.6 mg/L (saturated), standard is 5.0 mg/L
        q = 100.0 * abs(14.6 - value) / (14.6 - 5.0)
    else:
        # Ideal is 0
        q = 100.0 * abs(value - ideal_val) / abs(s_val - ideal_val)
        
    return max(0.0, float(q))

def compute_row_wqi(row):
    """
    Calculates overall Weighted Arithmetic WQI for a data row or dictionary:
    WQI = sum(Qi * Wi) / sum(Wi)
    """
    q_w_sum = 0.0
    w_sum = 0.0
    sub_indices = {}
    
    for param in WQI_STANDARDS.keys():
        val = row.get(param)
        if pd.notnull(val):
            q_i = calculate_sub_index(param, float(val))
            if not np.isnan(q_i):
                w_i = UNIT_WEIGHTS[param]
                q_w_sum += q_i * w_i
                w_sum += w_i
                sub_indices[param] = q_i
                
    if w_sum == 0.0:
        return np.nan, sub_indices
        
    wqi = q_w_sum / w_sum
    return round(float(wqi), 2), sub_indices

def get_wqi_classification(wqi):
    """
    Categorizes WQI into standard Brown et al. (1970) scale:
    - 0 to 25: Excellent Water Quality (Grade A)
    - 26 to 50: Good Water Quality (Grade B)
    - 51 to 75: Poor Water Quality (Grade C)
    - 76 to 100: Very Poor Water Quality (Grade D)
    - > 100: Unsuitable for Drinking / Highly Polluted (Grade E)
    """
    if pd.isnull(wqi):
        return {
            "grade": "N/A",
            "category": "Unknown",
            "color": "#94a3b8",
            "suitability": "Data unavailable",
            "description": "Insufficient water quality parameters to compute WQI."
        }
    wqi = float(wqi)
    if wqi <= 25.0:
        return {
            "grade": "Grade A",
            "category": "Excellent",
            "color": "#10b981", # Emerald
            "suitability": "Pristine, suitable for drinking and sensitive aquatic life without conventional treatment.",
            "description": "Water quality is in pristine ecological state with minimal anthropogenic disturbance."
        }
    elif wqi <= 50.0:
        return {
            "grade": "Grade B",
            "category": "Good",
            "color": "#3b82f6", # Blue
            "suitability": "Suitable for drinking after conventional treatment and disinfection, bathing, and irrigation.",
            "description": "Minor organic or nutrient enrichment, fully supports diverse freshwater ecosystems."
        }
    elif wqi <= 75.0:
        return {
            "grade": "Grade C",
            "category": "Poor",
            "color": "#f59e0b", # Amber
            "suitability": "Requires extensive filtration, coagulation, and disinfection before domestic use.",
            "description": "Elevated organic load or bacterial contamination detected. Aquatic habitat stressed."
        }
    elif wqi <= 100.0:
        return {
            "grade": "Grade D",
            "category": "Very Poor",
            "color": "#f97316", # Orange
            "suitability": "Unsuitable for domestic supply without advanced treatment; restricted irrigation only.",
            "description": "Significant degradation; high bacterial count, low DO, and elevated organic impurities."
        }
    else:
        return {
            "grade": "Grade E",
            "category": "Unsuitable",
            "color": "#ef4444", # Red
            "suitability": "Hazardous for drinking and public contact; requires ecological remediation.",
            "description": "Critically polluted surface water body exceeding permissible toxicological thresholds."
        }

def classify_cpcb_pwqc(do, ph, bod, fc=None):
    """
    Classifies water according to official Central Pollution Control Board (CPCB)
    Primary Water Quality Criteria (PWQC) for Designated Best Use:
    - Class A: Drinking water source without conventional treatment but after disinfection
    - Class B: Outdoor bathing (organized)
    - Class C: Drinking water source after conventional treatment and disinfection
    - Class D: Propagation of Wild life and Fisheries
    - Class E: Irrigation, Industrial Cooling, Controlled Waste disposal
    - Below Class E: Not meeting minimum criteria
    """
    if pd.isnull(do) or pd.isnull(ph) or pd.isnull(bod):
        return {
            "class_code": "Unknown",
            "name": "Insufficient Data",
            "color": "#94a3b8",
            "use": "Cannot determine without DO, pH, and BOD."
        }
        
    do, ph, bod = float(do), float(ph), float(bod)
    fc = float(fc) if pd.notnull(fc) else None
    
    # Class A: Total Coliform <= 50, DO >= 6 mg/L, BOD <= 2 mg/L, pH 6.5 - 8.5
    if do >= 6.0 and 6.5 <= ph <= 8.5 and bod <= 2.0 and (fc is None or fc <= 50):
        return {
            "class_code": "Class A",
            "name": "Drinking water source without conventional treatment (Disinfection only)",
            "color": "#10b981",
            "use": "Direct human consumption after disinfection; pristine Himalayan and upstream reaches."
        }
    # Class B: Fecal Coliform <= 500-2500, DO >= 5 mg/L, BOD <= 3 mg/L, pH 6.5 - 8.5
    elif do >= 5.0 and 6.5 <= ph <= 8.5 and bod <= 3.0 and (fc is None or fc <= 2500):
        return {
            "class_code": "Class B",
            "name": "Outdoor Bathing (Organized)",
            "color": "#06b6d4",
            "use": "Safe for public immersion, swimming, bathing, and water recreation."
        }
    # Class C: DO >= 4 mg/L, BOD <= 3 mg/L, pH 6.0 - 9.0
    elif do >= 4.0 and 6.0 <= ph <= 9.0 and bod <= 3.0:
        return {
            "class_code": "Class C",
            "name": "Drinking water source after conventional treatment and disinfection",
            "color": "#f59e0b",
            "use": "Municipal supply after coagulation, sedimentation, rapid sand filtration, and chlorination."
        }
    # Class D: DO >= 4 mg/L, pH 6.5 - 8.5
    elif do >= 4.0 and 6.5 <= ph <= 8.5:
        return {
            "class_code": "Class D",
            "name": "Propagation of Wild life and Fisheries",
            "color": "#8b5cf6",
            "use": "Supports indigenous fish species, benthic invertebrates, and aquatic biodiversity."
        }
    # Class E: pH 6.0 - 8.5
    elif 6.0 <= ph <= 8.5:
        return {
            "class_code": "Class E",
            "name": "Irrigation, Industrial Cooling, Controlled Waste disposal",
            "color": "#f97316",
            "use": "Agricultural crop irrigation and industrial cooling towers."
        }
    else:
        return {
            "class_code": "Below Class E",
            "name": "Heavily Polluted (Non-compliant)",
            "color": "#ef4444",
            "use": "Severe organic stress; water body unfit for beneficial designated uses."
        }
