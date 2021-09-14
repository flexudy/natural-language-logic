from flexudy.conceptor.conceptor_service.conceptor_service import ConceptorService
from typing import List, Optional, Dict


class ConceptInferenceMachine:

    def __init__(self, conceptor_service: ConceptorService):
        self.__conceptor_service = conceptor_service

    def infer_concepts(self, terms: List[str], language: Optional[str] = None, batch_size: int = 2,
                       beam_size: int = 4) -> Dict[str, List[str]]:

        concepts = self.__conceptor_service.infer_concepts(terms, language, batch_size, beam_size)

        return concepts
