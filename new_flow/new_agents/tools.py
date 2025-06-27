from crewai.tools import tool
from pydantic import Field
from typing import Optional
from crewai.tools import BaseTool
from transformers import pipeline

class MentalHealthTools:
    """Tools for mental health chatbot"""
    
    @tool("Bhutanese Helplines")
    def get_bhutanese_helplines():
        """
        Provides a list of mental health helplines in Bhutan.
        """
        helplines = """
        Here are some mental health helplines in Bhutan:
        - National Mental Health Program Hotline: 1717 (24/7)
        - Jigme Dorji Wangchuck National Referral Hospital (JDWNRH) Psychiatry Department: +975-2-322137
        - Khesar Gyalpo University of Medical Sciences of Bhutan (KGUMSB) Counseling Services: Contact university administration for current numbers.
        - Youth HelpLine (for young people): 1769 / 1768
        - Druk Trace App (for general emergency contacts, including health): Available on app stores.
        """
        return helplines

class TextClassifierTool(BaseTool):
    name: str = "Text Classifier"
    description: str = (
        "A tool that classifies text into predefined categories. "
        "Input should be the text to classify."
    )
            
    def _run(self, text: str) -> str:
        """
        Classifies the given text using the Hugging Face model.
        Returns the classification label and score.
        """
        try:
            # Initialize the pipeline here (will happen on every tool call)
            classifier = pipeline("sentiment-analysis", model="sentinet/suicidality")
            result = classifier(text)
            if result:
                label = result[0]['label']
                score = result[0]['score']
                return f"Classification: {label} (Score: {score:.4f})"
            return "Could not classify the text."
        except Exception as e:
            return f"Error during text classification: {e}"
