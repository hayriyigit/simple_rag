from abc import ABC, abstractmethod
from typing import Any

from langchain_ollama.llms import BaseLLM


class ProviderBase(ABC):
    def __call__(self, *args: tuple, **kwargs: dict[str, Any]) -> dict:
        if not hasattr(self, "model") or self.model is None:
            msg = "Subclasses must define the 'model' attribute."
            raise NotImplementedError(msg)
        return self.model.invoke(*args, **kwargs)

    @abstractmethod
    def create(self) -> BaseLLM:
        pass

    @abstractmethod
    def set_history_memory(self, llm: BaseLLM, max_token: int) -> None:
        pass

    @abstractmethod
    def set_prompt(self, template: str) -> None:
        pass
