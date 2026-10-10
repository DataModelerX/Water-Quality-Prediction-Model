"""
Report Generation Module for Water Quality Prediction System.
Produces downloadable CSV records and formatted assessment certificates.
"""

from datetime import datetime
import pandas as pd
import json

def generate_prediction_report(sample_input, predictions, wqi_info, cpcb_info, xai_info=None):
    """
    Constructs a detailed structured report dictionary and markdown text for export.
    """
    now_str = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    
    report_dict = {
        "Report_ID": f"WQR-{int(datetime.now().timestamp())}",
        "Generated_At": now_str,
        "Sampling_Location": sample_input.get("Location", "Field Test Sample"),
        "State": sample_input.get("State", "Custom Location"),
        "River_Basin": sample_input.get("River", "Not Specified"),
        
        # Parameters (Inputs and Predictions)
        "pH": predictions.get("pH", sample_input.get("pH")),
        "DO_mg_L": predictions.get("DO", sample_input.get("DO")),
        "BOD_mg_L": predictions.get("BOD", sample_input.get("BOD")),
        "COD_mg_L": predictions.get("COD", sample_input.get("COD")),
        "EC_umho_cm": predictions.get("Conductivity", sample_input.get("Conductivity")),
        "TDS_mg_L": predictions.get("TDS", sample_input.get("TDS")),
        "Temperature_C": sample_input.get("Temperature"),
        "Nitrate_mg_L": sample_input.get("Nitrate"),
        
        # Indices & Classification
        "WQI_Score": wqi_info.get("wqi"),
        "WQI_Category": wqi_info.get("category"),
        "WQI_Grade": wqi_info.get("grade"),
        "CPCB_Designated_Class": cpcb_info.get("class_code"),
        "Primary_Use": cpcb_info.get("name"),
        "Suitability_Remarks": wqi_info.get("suitability")
    }
    
    # Format markdown report
    md_content = f"""# Indian Surface Water Quality Assessment Report
**Report ID:** `{report_dict['Report_ID']}`  
**Date of Generation:** {now_str}  
**Location:** {report_dict['Sampling_Location']} | **State:** {report_dict['State']} | **River:** {report_dict['River_Basin']}  

---

## 1. Overall Water Quality Classification
- **Water Quality Index (WA-WQI):** **{report_dict['WQI_Score']}** ({report_dict['WQI_Grade']} - {report_dict['WQI_Category']})
- **CPCB Designated Best Use:** **{report_dict['CPCB_Designated_Class']}** ({report_dict['Primary_Use']})
- **Status Summary:** {report_dict['Suitability_Remarks']}

---

## 2. Chemical & Physical Parameter Profile
| Parameter | Value | Standard (CPCB/BIS) | Compliance Status |
| :--- | :--- | :--- | :--- |
| **pH** | {report_dict['pH']} pH units | 6.5 - 8.5 | {'Compliant' if 6.5 <= float(report_dict['pH'] or 7) <= 8.5 else 'Non-Compliant'} |
| **Dissolved Oxygen (DO)** | {report_dict['DO_mg_L']} mg/L | >= 5.0 mg/L | {'Compliant' if float(report_dict['DO_mg_L'] or 0) >= 5.0 else 'Deficient'} |
| **Biochemical Oxygen Demand (BOD)** | {report_dict['BOD_mg_L']} mg/L | <= 3.0 mg/L | {'Compliant' if float(report_dict['BOD_mg_L'] or 0) <= 3.0 else 'Excessive Organic Load'} |
| **Chemical Oxygen Demand (COD)** | {report_dict['COD_mg_L']} mg/L | <= 10.0 mg/L (Empirical) | {'Compliant' if float(report_dict['COD_mg_L'] or 0) <= 10.0 else 'High Oxidizable Matter'} |
| **Electrical Conductivity (EC)** | {report_dict['EC_umho_cm']} µmho/cm | <= 1000 µmho/cm | {'Compliant' if float(report_dict['EC_umho_cm'] or 0) <= 1000.0 else 'Elevated Salinity'} |
| **Total Dissolved Solids (TDS)** | {report_dict['TDS_mg_L']} mg/L | <= 500 mg/L | {'Compliant' if float(report_dict['TDS_mg_L'] or 0) <= 500.0 else 'Elevated Dissolved Solids'} |
| **Nitrate (NO3)** | {report_dict['Nitrate_mg_L']} mg/L | <= 45.0 mg/L | {'Compliant' if float(report_dict['Nitrate_mg_L'] or 0) <= 45.0 else 'Elevated Nutrients'} |

---

## 3. Methodology & Governance
- **WQI Methodology:** Weighted Arithmetic Index Method (Brown et al., 1970; Horton, 1965) incorporating unit weights inversely proportional to BIS 10500 standards.
- **CPCB PWQC Classification:** Governed by Primary Water Quality Criteria notified under the Environment (Protection) Rules, 1986.
- **Predictive AI Framework:** Multi-algorithm gradient boosting and ensemble trees trained on CPCB National Water Quality Monitoring Programme (NWMP) river surveillance dataset.
"""
    return report_dict, md_content

def dataframe_to_csv_download(df):
    """
    Returns CSV string for streamlit download button.
    """
    return df.to_csv(index=False).encode('utf-8')
