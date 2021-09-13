import os
from torch import cuda
from typing import Tuple
from transformers import T5TokenizerFast, T5ForConditionalGeneration


class ResourceHelper:

    def __init__(self, path_to_model_folder: str):
        self.__cuda_is_available = cuda.is_available()

        path_to_model_folder = self.__get_resource_path(path_to_model_folder + "/")

        self.__tokenizer_and_model = self.__load_model_and_tokenizer(path_to_model_folder)

    def get_tokenizer_and_model(self):
        return self.__tokenizer_and_model

    def get_cuda_is_available(self) -> bool:
        return self.__cuda_is_available

    def __get_resource_path(self, file_name: str) -> str:
        return os.path.join(os.path.dirname(os.path.realpath(__file__)), file_name)

    def __load_model_and_tokenizer(self, path_to_model_folder: str) -> Tuple[T5TokenizerFast,
                                                                             T5ForConditionalGeneration]:
        tokenizer = T5TokenizerFast.from_pretrained(path_to_model_folder)

        model = T5ForConditionalGeneration.from_pretrained(path_to_model_folder)

        if self.__cuda_is_available:
            model.to("cuda")

        return tokenizer, model
