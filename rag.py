"""
Ядро RAG: поиск по базе, сборка промпта, запрос к языковой модели.

Тексты промптов лежат в папке prompts/ и загружаются оттуда. Правила
поведения ассистента вынесены в системное сообщение, а сообщение
пользователя несёт только данные - контекст и вопрос. В версии из урока
правила ехали в пользовательском сообщении вместе с контекстом: там их
проще перебить содержимым документов.
"""

from dataclasses import dataclass, field
from typing import List, Optional

from openai import OpenAI

from config import AppConfig, CHUNK_PREVIEW_LENGTH
from embeddings import EmbeddingStore, SearchResult
from prompts import build_context_message, build_system_prompt
from timing import STAGE_LLM_GENERATION, StageTimer, TimingReport

# Заголовок фрагмента в контексте. Формат читает системный промпт,
# поэтому менять его в отрыве от prompts/system.md нельзя.
CONTEXT_ITEM_TEMPLATE = "[Документ {number} - {source}]\n{text}"

# Текст, уходящий в контекст, когда поиск ничего не нашёл
EMPTY_CONTEXT_TEXT = "Релевантных фрагментов в базе знаний не найдено."

# Роли сообщений OpenAI
ROLE_SYSTEM = "system"
ROLE_USER = "user"


@dataclass
class TokenUsage:
    """Расход токенов на один запрос к модели."""
    prompt_tokens: int = 0
    completion_tokens: int = 0
    total_tokens: int = 0


@dataclass
class RAGResponse:
    """Результат полного цикла RAG."""
    answer: str
    results: List[SearchResult] = field(default_factory=list)
    timing: Optional[TimingReport] = None
    usage: TokenUsage = field(default_factory=TokenUsage)
    failed: bool = False


class RAGAssistant:
    """Ассистент, отвечающий на вопросы по локальной базе знаний."""

    def __init__(self, embedding_store: EmbeddingStore, config: AppConfig, verbose: bool = True):
        self.embedding_store = embedding_store
        self.config = config
        self.verbose = verbose

        # Адрес API передаётся явно, см. комментарий к OPENAI_API_URL в config.py
        self.client = OpenAI(
            api_key=config.openai_api_key,
            base_url=config.openai_base_url,
            timeout=config.openai_timeout,
            max_retries=config.openai_max_retries,
        )

        # Системный промпт собирается один раз: он не зависит от запроса
        self.system_prompt = build_system_prompt(max_tokens=config.llm_max_tokens)

        if verbose:
            print(f"✓ RAG-ассистент готов (модель {config.llm_model})")

    # -- сборка контекста --------------------------------------------------

    def _format_context(self, results: List[SearchResult]) -> str:
        """Превращает найденные фрагменты в текст контекста."""
        if not results:
            return EMPTY_CONTEXT_TEXT

        parts = [
            CONTEXT_ITEM_TEMPLATE.format(number=index, source=item.source, text=item.text)
            for index, item in enumerate(results, start=1)
        ]
        return "\n\n".join(parts)

    # -- основной цикл -----------------------------------------------------

    def generate_response(
        self,
        query: str,
        top_k: Optional[int] = None,
        source_filter: Optional[str] = None,
        verbose: Optional[bool] = None,
    ) -> RAGResponse:
        """
        Полный цикл: поиск, сборка промпта, генерация ответа.

        Args:
            query: вопрос пользователя
            top_k: сколько фрагментов брать, по умолчанию из конфига
            source_filter: подстрока имени источника, по умолчанию из конфига
            verbose: печатать ли подробности хода работы

        Returns:
            RAGResponse с ответом, найденными фрагментами, таймингами и расходом токенов
        """
        show = self.verbose if verbose is None else verbose
        effective_top_k = top_k if top_k is not None else self.config.top_k

        timer = StageTimer()

        # Шаг 1. Поиск в векторной базе
        if show:
            print(f"\n🔍 Поиск по базе знаний (top_k={effective_top_k})")

        results, search_timing = self.embedding_store.search(
            query=query,
            top_k=effective_top_k,
            source_filter=source_filter,
        )

        # Переносим замеры поиска в общий отчёт
        for stage in search_timing.stages:
            timer.add(stage.name, stage.seconds)

        if show and results:
            print(f"\n📚 Найдено фрагментов: {len(results)}")
            for index, item in enumerate(results, start=1):
                preview = " ".join(item.text.split())[:CHUNK_PREVIEW_LENGTH]
                print(f"  {index}. [{item.source}] похожесть: {item.similarity:.3f}")
                print(f"     {preview}...")
        elif show:
            print("\n📚 Подходящих фрагментов не найдено")

        # Шаг 2. Сборка сообщения с контекстом
        context = self._format_context(results)
        user_message = build_context_message(context=context, question=query)

        # Шаг 3. Запрос к модели
        if show:
            print(f"\n🤖 Генерация ответа моделью {self.config.llm_model}")

        try:
            with timer.stage(STAGE_LLM_GENERATION):
                response = self.client.chat.completions.create(
                    model=self.config.llm_model,
                    messages=[
                        {"role": ROLE_SYSTEM, "content": self.system_prompt},
                        {"role": ROLE_USER, "content": user_message},
                    ],
                    temperature=self.config.llm_temperature,
                    max_tokens=self.config.llm_max_tokens,
                )

            answer = (response.choices[0].message.content or "").strip()

            usage = TokenUsage()
            if response.usage is not None:
                usage = TokenUsage(
                    prompt_tokens=response.usage.prompt_tokens or 0,
                    completion_tokens=response.usage.completion_tokens or 0,
                    total_tokens=response.usage.total_tokens or 0,
                )

            return RAGResponse(
                answer=answer,
                results=results,
                timing=timer.finish(),
                usage=usage,
            )

        except Exception as error:
            message = f"Ошибка при обращении к модели: {error}"
            print(f"❌ {message}")
            return RAGResponse(
                answer=message,
                results=results,
                timing=timer.finish(),
                failed=True,
            )
