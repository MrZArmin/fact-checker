from typing import List, Dict, Optional, Any
import spacy
from spacy.language import Language
from spacy.tokens import Doc
import os
import json
from pathlib import Path
from openai import OpenAI
from django.core.management.base import CommandError
import huspacy

class EntityExtractor:
    def __init__(self):
        #huspacy.download("hu_core_news_lg")
        self.nlp = spacy.load("hu_core_news_lg")

        self.api_key = os.getenv('OPENAI_API_KEY')
        self.client = OpenAI(api_key=self.api_key)
        self.prompts_dir = Path(__file__).parent.parent / "prompts"

    def _load_prompt(self, filename: str) -> str:
        """Load prompt from file."""
        try:
            prompt_path = self.prompts_dir / filename
            with open(prompt_path, "r", encoding="utf-8") as f:
                return f.read().strip()
        except Exception as e:
            raise FileNotFoundError(f"Could not load prompt {filename}: {str(e)}")

    def get_query_structure_llm(self, question: str) -> Dict[str, Any]:
        prompt = self._load_prompt("get_entities_prompt.txt")

        try:
            response = self.client.chat.completions.create(
                model="gpt-4o-mini",
                messages=[
                    {"role": "system", "content": "You are a helpful assistant that converts questions to structured JSON for knowledge graph queries."},
                    {"role": "user", "content": prompt + question}
                ],
                temperature=0
            )
            
            return json.loads(response.choices[0].message.content)
            
        except Exception as e:
            raise CommandError(f'Failed to generate query structure: {str(e)}')
        
    def get_query_structure_nlp(self, question: str) -> Dict[str, Any]:
        doc = self.nlp(question)
        entities = [ent.text for ent in doc.ents]
        relationships = []
        expected_type = None

        return {
            "entities": entities,
            "relationships": relationships,
        }