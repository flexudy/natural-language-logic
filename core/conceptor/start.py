from core.conceptor.concept_inference import ConceptInferenceMachine
from core.conceptor.conceptor_service.t5_conceptor_service import T5ConceptorService
from core.conceptor.resource_management.resource_helper import ResourceHelper


class FlexudyConceptInferenceMachineFactory:

    @staticmethod
    def get_concept_inference_machine(path_to_model: str = "flexudy-conceptor-4") -> ConceptInferenceMachine:
        resource_helper = ResourceHelper(path_to_model)

        concept_inference_machine = T5ConceptorService(resource_helper)

        return ConceptInferenceMachine(concept_inference_machine)
