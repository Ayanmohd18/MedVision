from typing import Dict, List
from langchain.prompts import PromptTemplate

RADIOLOGY_REPORT_TEMPLATE = """
You are an expert radiologist generating a structured chest X-ray report. 

PATIENT INFORMATION:
- Age: {age}
- Gender: {gender}
- View Position: {view_position}
- Clinical History: {clinical_history}

DETECTED FINDINGS:
{findings_list}

CONFIDENCE SCORES:
{confidence_scores}

Generate a professional radiology report with the following sections:

FINDINGS:
Describe the radiological findings in clear, medical terminology. Focus on the detected pathologies and their characteristics.

IMPRESSION:
Provide a concise clinical interpretation of the findings. Prioritize the most significant abnormalities.

RECOMMENDATIONS:
Suggest appropriate follow-up actions or additional studies if needed.

Use standard radiological terminology and maintain professional medical language throughout.
"""

FINDINGS_TEMPLATE = """
Based on the chest X-ray analysis, the following pathologies were detected:

{pathology_findings}

Image quality: {image_quality}
Technical factors: Adequate penetration and positioning.
"""

def create_findings_text(pathologies: List[str], probabilities: List[float], 
                        confidence_threshold: float = 0.7) -> str:
    """Create findings text from detected pathologies"""
    findings = []
    
    for pathology, prob in zip(pathologies, probabilities):
        if prob > confidence_threshold:
            confidence_level = "high" if prob > 0.8 else "moderate"
            findings.append(f"- {pathology}: Present ({confidence_level} confidence, {prob:.2f})")
    
    if not findings:
        return "No significant pathological findings detected."
    
    return "\n".join(findings)

def create_report_prompt(patient_data: Dict, findings: List[Dict]) -> PromptTemplate:
    """Create structured prompt for report generation"""
    
    # Format findings
    findings_text = ""
    confidence_text = ""
    
    for finding in findings:
        pathology = finding['pathology']
        probability = finding['probability']
        if probability > 0.5:
            findings_text += f"- {pathology} (confidence: {probability:.2f})\n"
            confidence_text += f"{pathology}: {probability:.2f}\n"
    
    if not findings_text:
        findings_text = "No significant abnormalities detected."
        confidence_text = "All pathologies below detection threshold."
    
    prompt = PromptTemplate(
        input_variables=["age", "gender", "view_position", "clinical_history", 
                        "findings_list", "confidence_scores"],
        template=RADIOLOGY_REPORT_TEMPLATE
    )
    
    return prompt

QUALITY_CHECK_TEMPLATE = """
Review the following radiology report for medical accuracy and completeness:

{report_text}

Check for:
1. Appropriate medical terminology
2. Logical consistency between findings and impression
3. Complete coverage of detected abnormalities
4. Professional language and structure

Provide feedback on any issues or confirm if the report meets clinical standards.
"""