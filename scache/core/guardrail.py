from pathlib import Path
import sys

# Add project root to sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

import spacy

nlp = spacy.load("en_core_web_sm")

class EntityGuardrail:

    def __init__(self) -> None:
        pass
    
    def extract_entities(self, query_text: str) -> set[str]:
        """
            returns a set of lower case strings which 
            are named entities
        """
        doc = nlp(query_text)
        memo = set()
        for ent in doc.ents:
            memo.add(ent.text.lower())

        return memo
    

    def verify(self, query_text: str, cached_text: str) -> bool:
        """
            so even if we get empty sets the return value is 
            going to be true. empty set as no entities were
            detected
        """
        query_entity = self.extract_entities(query_text)
        cached_entity = self.extract_entities(cached_text)
        
        for ent in query_entity:
            if ent not in cached_text.lower():
                return False

        for ent in cached_entity:
            if ent not in query_text.lower():
                return False

        return True



