"""
Конфигурация проекта.

Единственное место, где заданы значения по умолчанию. В остальных модулях
числовых и строковых литералов настройки нет: всё приходит сюда из .env,
а при отсутствии переменной берётся константа из блока DEFAULT_*.

Порядок приоритета: переменная окружения -> константа по умолчанию.
"""

import os
from dataclasses import dataclass
from typing import List, Optional

from dotenv import load_dotenv

# ---------------------------------------------------------------------------
# ЗНАЧЕНИЯ ПО УМОЛЧАНИЮ
# Меняются либо здесь, либо (предпочтительно) в файле .env
# ---------------------------------------------------------------------------

# --- Доступ к OpenAI ---
# Штатный адрес OpenAI. Задаётся явной константой, а не пустой строкой.
# Причина: библиотека openai читает переменную окружения OPENAI_BASE_URL сама.
# Строка OPENAI_BASE_URL= в файле .env задаёт переменную с пустым значением,
# библиотека считает её заданной и собирает запрос без протокола, после чего
# падает с "Request URL is missing an 'http://' or 'https://' protocol".
# Поэтому адрес всегда передаётся в клиент явно.
OPENAI_API_URL = "https://api.openai.com/v1"
DEFAULT_OPENAI_BASE_URL = OPENAI_API_URL
DEFAULT_OPENAI_TIMEOUT = 60.0         # секунд на один HTTP-запрос
DEFAULT_OPENAI_MAX_RETRIES = 2        # повторов при сетевой ошибке

# --- Языковая модель ---
DEFAULT_LLM_MODEL = "gpt-3.5-turbo"
DEFAULT_LLM_TEMPERATURE = 0.7
DEFAULT_LLM_MAX_TOKENS = 500

# --- Эмбеддинги ---
DEFAULT_EMBEDDING_MODEL = "text-embedding-3-small"
DEFAULT_EMBEDDING_BATCH_SIZE = 100    # текстов в одном запросе к API

# --- Векторное хранилище ChromaDB ---
DEFAULT_CHROMA_PERSIST_DIR = "./chroma_db"
DEFAULT_CHROMA_COLLECTION = "rag_documents"
DEFAULT_CHROMA_DISTANCE = "cosine"    # cosine | l2 | ip
DEFAULT_CHROMA_TELEMETRY = False      # отправка анонимной статистики в Chroma

# --- Разбивка документов на чанки ---
DEFAULT_CHUNK_SIZE = 500              # символов
DEFAULT_CHUNK_OVERLAP = 50            # символов перекрытия

# --- Поиск ---
DEFAULT_TOP_K = 3                     # сколько чанков уходит в контекст
DEFAULT_SOURCE_FILTER = ""            # пусто = искать по всем источникам

# --- Кеш ответов ---
DEFAULT_CACHE_FILE = "cache.json"
DEFAULT_CACHE_ENABLED = True
DEFAULT_CACHE_KEY_INCLUDES_PARAMS = True

# --- Отображение времени ответа ---
DEFAULT_TIME_PRECISION = 5            # знаков после запятой (0.00001 с)
DEFAULT_SHOW_TIMING_BREAKDOWN = True  # показывать разбивку по этапам

# --- Демонстрационный режим и режим замеров ---
DEFAULT_DEMO_QUESTIONS = [
    "Привет, что ты умеешь?",
    "Что такое RAG?",
    "Что такое ИИ?",
]

# --- Оформление вывода в терминал ---
SEPARATOR_WIDTH = 70                  # ширина линий-разделителей
CACHE_PREVIEW_LENGTH = 50             # сколько символов запроса печатать в логе
CHUNK_PREVIEW_LENGTH = 100            # сколько символов чанка показывать в выдаче

# --- Служебные значения ---
CACHE_FORMAT_VERSION = 2              # версия структуры файла кеша
EXIT_COMMANDS = ("exit", "quit", "выход", "q")
DEMO_QUESTIONS_SEPARATOR = "|"        # разделитель вопросов в переменной окружения


# ---------------------------------------------------------------------------
# ЧТЕНИЕ ПЕРЕМЕННЫХ ОКРУЖЕНИЯ
# ---------------------------------------------------------------------------

def _get_str(name: str, default: str) -> str:
    """Читает строку. Пустое значение считается незаданным."""
    raw = os.getenv(name)
    if raw is None or raw.strip() == "":
        return default
    return raw.strip()


def _get_int(name: str, default: int) -> int:
    """Читает целое число, при нечисловом значении падает с понятным текстом."""
    raw = os.getenv(name)
    if raw is None or raw.strip() == "":
        return default
    try:
        return int(raw.strip())
    except ValueError:
        raise ValueError(
            f"Переменная {name} должна быть целым числом, получено: '{raw}'"
        )


def _get_float(name: str, default: float) -> float:
    """Читает дробное число, принимает и точку, и запятую."""
    raw = os.getenv(name)
    if raw is None or raw.strip() == "":
        return default
    try:
        return float(raw.strip().replace(",", "."))
    except ValueError:
        raise ValueError(
            f"Переменная {name} должна быть числом, получено: '{raw}'"
        )


def _get_bool(name: str, default: bool) -> bool:
    """Читает флаг. Истина: true/1/yes/on/да. Ложь: false/0/no/off/нет."""
    raw = os.getenv(name)
    if raw is None or raw.strip() == "":
        return default

    value = raw.strip().lower()
    if value in ("true", "1", "yes", "on", "да"):
        return True
    if value in ("false", "0", "no", "off", "нет"):
        return False

    raise ValueError(
        f"Переменная {name} должна быть true или false, получено: '{raw}'"
    )


def _get_list(name: str, default: List[str], separator: str) -> List[str]:
    """Читает список строк, разделённых указанным символом."""
    raw = os.getenv(name)
    if raw is None or raw.strip() == "":
        return list(default)

    items = [part.strip() for part in raw.split(separator)]
    return [item for item in items if item]


# ---------------------------------------------------------------------------
# ОБЪЕКТ КОНФИГУРАЦИИ
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class AppConfig:
    """Собранные настройки приложения. Неизменяемый объект."""

    # OpenAI
    openai_api_key: str
    openai_base_url: str
    openai_timeout: float
    openai_max_retries: int

    # LLM
    llm_model: str
    llm_temperature: float
    llm_max_tokens: int

    # Эмбеддинги
    embedding_model: str
    embedding_batch_size: int

    # ChromaDB
    chroma_persist_dir: str
    chroma_collection: str
    chroma_distance: str
    chroma_telemetry: bool

    # Чанки
    chunk_size: int
    chunk_overlap: int

    # Поиск
    top_k: int
    source_filter: str

    # Кеш
    cache_file: str
    cache_enabled: bool
    cache_key_includes_params: bool

    # Отображение времени
    time_precision: int
    show_timing_breakdown: bool

    # Демо-режим
    demo_questions: List[str]

    def search_signature(self) -> str:
        """
        Строка параметров, влияющих на текст ответа.

        Входит в ключ кеша: при смене модели, температуры, top_k или фильтра
        старый ответ перестаёт подходить, и кеш обязан промахнуться. Без этого
        эксперимент "поменял TOP_K и сравнил ответы" не работает - кеш вернёт
        ответ, посчитанный со старым значением.
        """
        return "|".join([
            self.llm_model,
            f"t={self.llm_temperature}",
            f"max={self.llm_max_tokens}",
            f"emb={self.embedding_model}",
            f"k={self.top_k}",
            f"src={self.source_filter or 'all'}",
        ])

    def describe(self) -> str:
        """Человекочитаемая сводка настроек для вывода при запуске."""
        if self.openai_base_url == OPENAI_API_URL:
            base_url = f"{OPENAI_API_URL} (стандартный OpenAI)"
        else:
            base_url = self.openai_base_url
        source = self.source_filter or "все источники"
        cache_state = "включён" if self.cache_enabled else "выключен"

        lines = [
            f"  Модель LLM             : {self.llm_model}",
            f"  Температура            : {self.llm_temperature}",
            f"  Лимит токенов ответа   : {self.llm_max_tokens}",
            f"  Модель эмбеддингов     : {self.embedding_model}",
            f"  Адрес API              : {base_url}",
            f"  Коллекция ChromaDB     : {self.chroma_collection}",
            f"  Папка базы             : {self.chroma_persist_dir}",
            f"  Метрика расстояния     : {self.chroma_distance}",
            f"  Размер чанка/перекрытие: {self.chunk_size} / {self.chunk_overlap}",
            f"  TOP_K                  : {self.top_k}",
            f"  Фильтр источников      : {source}",
            f"  Кеш                    : {cache_state} ({self.cache_file})",
            f"  Точность таймера       : {self.time_precision} знаков после запятой",
        ]
        return "\n".join(lines)


def load_config(env_file: Optional[str] = None) -> AppConfig:
    """
    Загружает .env и собирает объект конфигурации.

    Args:
        env_file: путь к файлу .env, по умолчанию ищется в текущей папке

    Returns:
        AppConfig

    Raises:
        ValueError: значение переменной некорректно или не прошло проверку
    """
    if env_file:
        load_dotenv(env_file)
    else:
        load_dotenv()

    config = AppConfig(
        openai_api_key=_get_str("OPENAI_API_KEY", ""),
        openai_base_url=_get_str("OPENAI_BASE_URL", DEFAULT_OPENAI_BASE_URL),
        openai_timeout=_get_float("OPENAI_TIMEOUT", DEFAULT_OPENAI_TIMEOUT),
        openai_max_retries=_get_int("OPENAI_MAX_RETRIES", DEFAULT_OPENAI_MAX_RETRIES),

        llm_model=_get_str("LLM_MODEL", DEFAULT_LLM_MODEL),
        llm_temperature=_get_float("LLM_TEMPERATURE", DEFAULT_LLM_TEMPERATURE),
        llm_max_tokens=_get_int("LLM_MAX_TOKENS", DEFAULT_LLM_MAX_TOKENS),

        embedding_model=_get_str("EMBEDDING_MODEL", DEFAULT_EMBEDDING_MODEL),
        embedding_batch_size=_get_int("EMBEDDING_BATCH_SIZE", DEFAULT_EMBEDDING_BATCH_SIZE),

        chroma_persist_dir=_get_str("CHROMA_PERSIST_DIR", DEFAULT_CHROMA_PERSIST_DIR),
        chroma_collection=_get_str("CHROMA_COLLECTION", DEFAULT_CHROMA_COLLECTION),
        chroma_distance=_get_str("CHROMA_DISTANCE", DEFAULT_CHROMA_DISTANCE).lower(),
        chroma_telemetry=_get_bool("CHROMA_TELEMETRY", DEFAULT_CHROMA_TELEMETRY),

        chunk_size=_get_int("CHUNK_SIZE", DEFAULT_CHUNK_SIZE),
        chunk_overlap=_get_int("CHUNK_OVERLAP", DEFAULT_CHUNK_OVERLAP),

        top_k=_get_int("TOP_K", DEFAULT_TOP_K),
        source_filter=_get_str("SOURCE_FILTER", DEFAULT_SOURCE_FILTER),

        cache_file=_get_str("CACHE_FILE", DEFAULT_CACHE_FILE),
        cache_enabled=_get_bool("CACHE_ENABLED", DEFAULT_CACHE_ENABLED),
        cache_key_includes_params=_get_bool(
            "CACHE_KEY_INCLUDES_PARAMS", DEFAULT_CACHE_KEY_INCLUDES_PARAMS
        ),

        time_precision=_get_int("TIME_PRECISION", DEFAULT_TIME_PRECISION),
        show_timing_breakdown=_get_bool(
            "SHOW_TIMING_BREAKDOWN", DEFAULT_SHOW_TIMING_BREAKDOWN
        ),

        demo_questions=_get_list(
            "DEMO_QUESTIONS", DEFAULT_DEMO_QUESTIONS, DEMO_QUESTIONS_SEPARATOR
        ),
    )

    _validate(config)
    return config


def _validate(config: AppConfig) -> None:
    """Проверяет значения, при которых программа не сможет работать осмысленно."""
    problems: List[str] = []

    if config.top_k < 1:
        problems.append("TOP_K должен быть не меньше 1")

    if config.chunk_size < 1:
        problems.append("CHUNK_SIZE должен быть не меньше 1")

    if config.chunk_overlap < 0:
        problems.append("CHUNK_OVERLAP не может быть отрицательным")

    if config.chunk_overlap >= config.chunk_size:
        problems.append(
            "CHUNK_OVERLAP должен быть меньше CHUNK_SIZE, иначе разбивка на чанки "
            "зациклится и текст будет резаться бесконечно"
        )

    if not 0.0 <= config.llm_temperature <= 2.0:
        problems.append("LLM_TEMPERATURE должна быть в диапазоне от 0 до 2")

    if config.llm_max_tokens < 1:
        problems.append("LLM_MAX_TOKENS должен быть не меньше 1")

    if config.embedding_batch_size < 1:
        problems.append("EMBEDDING_BATCH_SIZE должен быть не меньше 1")

    if config.chroma_distance not in ("cosine", "l2", "ip"):
        problems.append("CHROMA_DISTANCE должен быть cosine, l2 или ip")

    if not 0 <= config.time_precision <= 9:
        problems.append("TIME_PRECISION должен быть в диапазоне от 0 до 9")

    if not config.openai_base_url.startswith(("http://", "https://")):
        problems.append(
            "OPENAI_BASE_URL должен начинаться с http:// или https://, "
            f"получено: '{config.openai_base_url}'. Оставьте строку закомментированной, "
            f"чтобы использовать стандартный адрес {OPENAI_API_URL}"
        )

    if config.openai_timeout <= 0:
        problems.append("OPENAI_TIMEOUT должен быть больше нуля")

    if config.openai_max_retries < 0:
        problems.append("OPENAI_MAX_RETRIES не может быть отрицательным")

    if problems:
        raise ValueError(
            "Ошибки в настройках (.env):\n  - " + "\n  - ".join(problems)
        )
