import os
import sys
import json
from agno.agent import Agent
from agno.media import Image
from pathlib import Path
from agno.models.google import Gemini
from dotenv import load_dotenv
from pydantic import BaseModel
from typing import List

current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)
project_root = os.path.dirname(parent_dir)

sys.path.append(parent_dir)
sys.path.append(project_root)

from Tools.crop_disease_detection import detect_crop_disease
from agno.tools.tavily import TavilyTools
load_dotenv()


class CropDiseaseOutput(BaseModel):
    diseases: List[str] = []
    disease_probabilities: List[float] = []
    symptoms: List[str] = []
    Treatments: List[str] = []
    prevention_tips: List[str] = []


class CropDiseaseAgent:
    def __init__(self, model_id="gemini-2.0-flash"):
        self.agent = Agent(
            model=Gemini(id=model_id),
            markdown=True,
            debug_mode=False,
            tools=[detect_crop_disease, TavilyTools()],
            instructions="""
You are an advanced crop disease analysis agent. Your task is to analyze crop images for disease symptoms and provide a clear diagnosis and actionable recommendations.

CRITICAL: You MUST return EXACTLY this JSON structure:
{
    "diseases": ["Disease1", "Disease2", "Disease3"],
    "disease_probabilities": [0.85, 0.70, 0.60],
    "symptoms": ["symptom1", "symptom2", "symptom3"],
    "Treatments": ["treatment1", "treatment2", "treatment3"],
    "prevention_tips": ["tip1", "tip2", "tip3"]
}

RULES:
1. ALWAYS provide at least 3 items in each list
2. disease_probabilities should be decimals between 0.0 and 1.0
3. If no image provided, use general crop disease information
4. Each prevention tip should be maximum 10 words
5. Make treatments specific and actionable
"""
        )

    def analyze_disease(self, query: str, image_path=None) -> CropDiseaseOutput:
        if image_path and os.path.exists(image_path):
            image = Image(filepath=Path(image_path))
            prompt = f"Analyze this crop image for disease symptoms and provide structured JSON output: {query}"
            result = self.agent.run(prompt, images=[image])
        else:
            prompt = f"No image provided. Analyze based on context only. Provide JSON structured output for: {query}"
            result = self.agent.run(prompt)

        response = result.content if hasattr(result, "content") else result

        # Try parsing JSON
        try:
            parsed = json.loads(response)
            return CropDiseaseOutput(**parsed)
        except Exception as e:
            # fallback if parsing fails
            return CropDiseaseOutput(
                diseases=["Unable to determine"],
                disease_probabilities=[0.0],
                symptoms=[f"Parsing error: {e}. Raw output: {response}"],
                Treatments=["Consult local agricultural expert"],
                prevention_tips=["Regular crop monitoring recommended"]
            )


if __name__ == "__main__":
    agent = CropDiseaseAgent()

    # Test with image
    image_path = os.path.join(project_root, "uploads", "Screenshot 2025-08-16 at 12.22.35 PM.png")
    print(f"Looking for image at: {image_path}")
    print(f"Image exists: {os.path.exists(image_path)}")

    result = agent.analyze_disease(query="analyze this crop for diseases", image_path=image_path)
    print("Diseases:", result.diseases)
    print("Disease probabilities:", result.disease_probabilities)
    print("Symptoms:", result.symptoms)
    print("Treatments:", result.Treatments)
    print("Prevention tips:", result.prevention_tips)

    # Test without image
    query = "What are the common diseases affecting wheat crops?"
    result = agent.analyze_disease(query=query)
    print("Diseases:", result.diseases)
    print("Disease probabilities:", result.disease_probabilities)
    print("Symptoms:", result.symptoms)
    print("Treatments:", result.Treatments)
    print("Prevention tips:", result.prevention_tips)
