from flexudy.conceptor.concept_inference import ConceptInferenceMachine
from flexudy.conceptor.conceptor_service.t5_conceptor_service import T5ConceptorService
from flexudy.conceptor.resource_management.resource_helper import ResourceHelper


class FlexudyConceptInferenceMachineFactory:

    @staticmethod
    def get_concept_inference_machine(path_to_model: str = "flexudy/conceptor-t5-base") -> ConceptInferenceMachine:
        resource_helper = ResourceHelper(path_to_model)

        concept_inference_machine = T5ConceptorService(resource_helper)

        return ConceptInferenceMachine(concept_inference_machine)
