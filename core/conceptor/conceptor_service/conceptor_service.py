from typing import List, Optional, Dict


class ConceptorService:
    def infer_concepts(self, terms: List[str], language: Optional[str] = None, batch_size: int = 2,
                       beam_size: int = 4) -> Dict[str, List[str]]:
        pass
