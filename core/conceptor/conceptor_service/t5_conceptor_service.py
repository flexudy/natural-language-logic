from typing import List, Optional, Dict, Tuple
from core.conceptor.conceptor_service.conceptor_service import ConceptorService
from core.conceptor.resource_management.resource_helper import ResourceHelper
from langdetect import detect


class T5ConceptorService(ConceptorService):
    __MAX_IN_SEQUENCE_LEN = 512

    __MAX_OUT_SEQUENCE_LEN = 64

    def __init__(self, resource_helper: ResourceHelper):
        super().__init__()

        self.__resource_helper = resource_helper

        self.__model_prefix = "flexudy:"

        self.__supported_languages = {"en", "de", "fr"}

    def infer_concepts(self, terms: List[str], language: Optional[str] = None, batch_size: int = 2,
                       beam_size: int = 4) -> Dict[str, List[str]]:

        terms = self.__get_valid_terms_and_languages(terms, language)

        if len(terms) == 0:
            return dict()

        generated_concepts = dict()

        i = 0

        while i < len(terms):
            batch = self.__get_batch_for_generation(terms, i, batch_size)

            concepts = self.__generate_concepts(batch, beam_size)

            generated_concepts.update(concepts)

            i += batch_size

        return generated_concepts

    def __get_valid_terms_and_languages(self, terms: List[str],
                                        language: Optional[str] = None) -> List[Tuple[str, str]]:

        selected_terms_and_languages = list()

        for term in terms:

            term = term.strip().lower()

            if len(term) == 0:
                continue

            detected_language = self.__get_language(term, language)

            selected_terms_and_languages.append((term, detected_language))

        return selected_terms_and_languages

    def __get_language(self, term: str, current_language: Optional[str] = None) -> str:

        if current_language is None:
            try:
                current_language = detect(term)
            except:
                current_language = "en"

        current_language = current_language.strip().lower()

        if current_language not in self.__supported_languages:
            current_language = "en"

        return current_language

    def __get_batch_for_generation(self, terms: List[Tuple[str, str]], position: int,
                                   batch_size: int) -> List[Tuple[str, str]]:

        batch = terms[position: position + batch_size]

        return batch

    def __generate_concepts(self, terms_and_languages: List[Tuple[str, str]], beam_size: int) -> Dict[str, List[str]]:

        model_inputs = self.__preprocess_terms(terms_and_languages)

        tokenizer, model = self.__resource_helper.get_tokenizer_and_model()

        inputs = tokenizer.batch_encode_plus(model_inputs, return_tensors="pt", truncation=True, padding=True)

        input_ids = inputs["input_ids"]

        if self.__resource_helper.get_cuda_is_available():
            input_ids = input_ids.to("cuda")

        outputs = model.generate(
            input_ids=input_ids,
            max_length=self.__MAX_IN_SEQUENCE_LEN,
            num_beams=beam_size,
            length_penalty=1.0,
            early_stopping=True,
            repetition_penalty=1.0
        )

        generated_concepts = tokenizer.batch_decode(outputs, skip_special_tokens=True,
                                                    clean_up_tokenization_spaces=True)

        concepts = self.__get_concepts(generated_concepts, terms_and_languages)

        return concepts

    def __preprocess_terms(self, terms_and_languages: List[Tuple[str, str]]) -> List[str]:

        model_inputs = list()

        for term, language in terms_and_languages:
            model_inputs.append(self.__model_prefix + " [" + language + "] " + term + " </s>")

        return model_inputs

    def __get_concepts(self, generated_concepts: List[str],
                       terms_and_languages: List[Tuple[str, str]]) -> Dict[str, List[str]]:

        selected_concepts = dict()

        i = 0

        while i < len(generated_concepts):

            concepts = generated_concepts[i].strip()

            term = terms_and_languages[i][0]  # Term is at position zero of the tuple

            if len(concepts) > 0:
                concept_collection = concepts.split("   ")

                concept_collection = [concept for concept in concept_collection if concept.strip() != term]

                if len(concept_collection) == 0:
                    concept_collection = [term]

                selected_concepts[term] = list(set(concept_collection))

            i += 1

        return selected_concepts
