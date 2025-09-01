from langchain.llms import OpenAI
from langchain.chains import LLMChain
from langchain.schema import BaseOutputParser
import openai
from typing import Dict, List, Any
import os
from .prompt_templates import create_report_prompt, QUALITY_CHECK_TEMPLATE

class RadiologyReportGenerator:
    def __init__(self, openai_api_key: str = None, model: str = "gpt-4", temperature: float = 0.3):
        self.api_key = openai_api_key or os.getenv("OPENAI_API_KEY")
        openai.api_key = self.api_key
        
        self.llm = OpenAI(
            model_name=model,
            temperature=temperature,
            max_tokens=1000,
            openai_api_key=self.api_key
        )
    
    def generate_report(self, patient_data: Dict, findings: List[Dict]) -> Dict[str, Any]:
        """Generate structured radiology report"""
        
        # Create prompt
        prompt = create_report_prompt(patient_data, findings)
        
        # Create chain
        chain = LLMChain(llm=self.llm, prompt=prompt)
        
        # Prepare input data
        input_data = {
            "age": patient_data.get("age", "Unknown"),
            "gender": patient_data.get("gender", "Unknown"),
            "view_position": patient_data.get("view_position", "PA"),
            "clinical_history": patient_data.get("clinical_history", "Not provided"),
            "findings_list": self._format_findings(findings),
            "confidence_scores": self._format_confidence_scores(findings)
        }
        
        # Generate report
        try:
            report_text = chain.run(input_data)
            
            # Parse structured sections
            structured_report = self._parse_report_sections(report_text)
            
            return {
                "success": True,
                "report": structured_report,
                "raw_text": report_text,
                "metadata": {
                    "patient_id": patient_data.get("patient_id"),
                    "generated_by": "AI_Assistant",
                    "findings_count": len([f for f in findings if f["probability"] > 0.5])
                }
            }
            
        except Exception as e:
            return {
                "success": False,
                "error": str(e),
                "report": None
            }
    
    def _format_findings(self, findings: List[Dict]) -> str:
        """Format findings for prompt"""
        if not findings:
            return "No significant abnormalities detected."
        
        formatted = []
        for finding in findings:
            if finding["probability"] > 0.5:
                formatted.append(f"- {finding['pathology']}: {finding['probability']:.2f}")
        
        return "\n".join(formatted) if formatted else "No significant abnormalities detected."
    
    def _format_confidence_scores(self, findings: List[Dict]) -> str:
        """Format confidence scores for prompt"""
        scores = []
        for finding in findings:
            if finding["probability"] > 0.3:  # Include lower threshold for context
                scores.append(f"{finding['pathology']}: {finding['probability']:.2f}")
        
        return "\n".join(scores) if scores else "All findings below significance threshold."
    
    def _parse_report_sections(self, report_text: str) -> Dict[str, str]:
        """Parse report into structured sections"""
        sections = {
            "findings": "",
            "impression": "",
            "recommendations": ""
        }
        
        lines = report_text.split('\n')
        current_section = None
        
        for line in lines:
            line = line.strip()
            if line.upper().startswith('FINDINGS'):
                current_section = 'findings'
            elif line.upper().startswith('IMPRESSION'):
                current_section = 'impression'
            elif line.upper().startswith('RECOMMENDATIONS'):
                current_section = 'recommendations'
            elif current_section and line:
                sections[current_section] += line + " "
        
        # Clean up sections
        for key in sections:
            sections[key] = sections[key].strip()
        
        return sections

    def validate_report_quality(self, report_text: str) -> Dict[str, Any]:
        """Validate generated report quality"""
        validation_prompt = QUALITY_CHECK_TEMPLATE.format(report_text=report_text)
        
        try:
            validation_result = self.llm(validation_prompt)
            return {
                "validation_passed": "meets clinical standards" in validation_result.lower(),
                "feedback": validation_result,
                "issues_found": "issues" in validation_result.lower()
            }
        except Exception as e:
            return {
                "validation_passed": False,
                "feedback": f"Validation failed: {str(e)}",
                "issues_found": True
            }