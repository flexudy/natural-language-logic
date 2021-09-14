from core.conceptor.start import FlexudyConceptInferenceMachineFactory
from typing import List, Optional


def infer_concepts_by_example(terms: List[str], language: Optional[str] = None) -> None:
    concept_inference_machine = FlexudyConceptInferenceMachineFactory.get_concept_inference_machine()

    concepts = concept_inference_machine.infer_concepts(terms, language=language)

    print("Language: {0}".format(language))

    print("_________________")

    print(concepts)

    print("_________________\n")


def infer_english_concepts() -> None:
    terms = ["snake", "door", "economics and sociology", "chair", "public company"]

    infer_concepts_by_example(terms, "en")


def infer_german_concepts() -> None:
    terms = ["Deutschland", "McDonald", "Firmenstruktur", "Finanzdienstleistung", "Mietvertrag"]

    infer_concepts_by_example(terms, "de")


def infer_french_concepts() -> None:
    terms = ["cour d'anglais", "Seconde Guerre mondiale", "1945", "Vertèbre", "Tête au pied"]

    infer_concepts_by_example(terms, "fr")


def infer_unknown_language_concepts() -> None:
    terms = ["Imaginary Airways", "Freundschaft", "Voiture", "schöne Blumen", "Ordinateur"]

    infer_concepts_by_example(terms)


if __name__ == "__main__":
    infer_english_concepts()

    infer_german_concepts()

    infer_french_concepts()

    infer_unknown_language_concepts()
