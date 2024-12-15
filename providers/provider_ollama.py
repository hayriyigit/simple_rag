from langchain.chains import ConversationChain
from langchain.chains.conversation.memory import ConversationSummaryBufferMemory
from langchain_core.prompts import PromptTemplate
from langchain_ollama import OllamaLLM
from langchain_ollama.llms import BaseLLM

from providers.provider_base import ProviderBase


class OllamaModel(ProviderBase):
    def __init__(self):
        super().__init__()
        self.memory = None
        self.prompt = """
You are a helpfull asistant
Current conversation:
{history}
Human: {input}
AI:
  """

    def set_history_memory(self, llm: BaseLLM, max_token: int) -> None:
        self.memory = ConversationSummaryBufferMemory(
            llm=llm,
            max_token_limit=max_token,
        )

    def set_prompt(self, template: str) -> None:
        if template:
            self.prompt = PromptTemplate.from_template(template=template)

    def create(
        self,
        model: str,
        temperature: float,
        top_k: int = 40,
        top_p: float = 0.9,
        num_predict: int = 128,
        repeat_penalty: float = 1.1,
        max_token: int = 2048,
        template: str | None = None,
    ):
        llm = OllamaLLM(
            model=model,
            temperature=temperature,
            top_k=top_k,
            top_p=top_p,
            num_predict=num_predict,
            repeat_penalty=repeat_penalty,
        )

        self.set_history_memory(llm=llm, max_token=max_token)
        self.set_prompt(template=template)

        return ConversationChain(
            llm=llm,
            prompt=self.prompt,
            memory=self.memory,
        )
