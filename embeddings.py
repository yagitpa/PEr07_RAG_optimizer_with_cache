"""
Векторное хранилище ChromaDB и эмбеддинги OpenAI.

Отличия от версии из урока:
- все параметры приходят из config.py, литералов настройки в модуле нет;
- метрика расстояния задаётся явно (по умолчанию cosine), иначе показатель
  similarity = 1 - distance не имеет смысла: у метрики L2 расстояние не
  ограничено единицей;
- при несовпадении метрики уже созданной коллекции с настройкой выводится
  предупреждение, а не тихое расхождение;
- поиск умеет фильтровать по источнику через метаданные;
- поиск отдельно замеряет время эмбеддинга запроса и время запроса к базе.
"""

from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import chromadb
from chromadb.config import Settings
from openai import OpenAI

from config import AppConfig
from timing import STAGE_QUERY_EMBEDDING, STAGE_VECTOR_SEARCH, StageTimer, TimingReport

# Ключ метаданных ChromaDB, задающий метрику расстояния
CHROMA_DISTANCE_KEY = "hnsw:space"

# Описание коллекции, попадает в метаданные
COLLECTION_DESCRIPTION = "Документы RAG-ассистента"

# Ключи метаданных чанка
META_SOURCE = "source"
META_CHUNK_LENGTH = "chunk_length"

# Префикс идентификаторов чанков
CHUNK_ID_PREFIX = "chunk_"

# Формат эмбеддингов, который запрашиваем у API
EMBEDDING_ENCODING_FORMAT = "float"


@dataclass
class SearchResult:
    """Один найденный фрагмент."""
    text: str
    source: str
    distance: float

    @property
    def similarity(self) -> float:
        """
        Похожесть в диапазоне примерно от 0 до 1.

        Корректно считается для метрики cosine. При l2 значение будет
        уходить в минус, поэтому метрика и вынесена в настройку с проверкой.
        """
        return 1.0 - self.distance


class EmbeddingStore:
    """Хранилище векторов на ChromaDB с эмбеддингами через OpenAI API."""

    def __init__(self, config: AppConfig, verbose: bool = True):
        self.config = config
        self.verbose = verbose

        if verbose:
            print(f"Инициализация ChromaDB: {config.chroma_persist_dir}")

        self.client = chromadb.PersistentClient(
            path=config.chroma_persist_dir,
            settings=Settings(anonymized_telemetry=config.chroma_telemetry),
        )

        # Адрес API передаётся всегда. Полагаться на то, что библиотека сама
        # возьмёт штатный адрес, нельзя: она читает переменную окружения
        # OPENAI_BASE_URL, и пустое значение из .env считает заданным адресом.
        self.openai_client = OpenAI(
            api_key=config.openai_api_key,
            base_url=config.openai_base_url,
            timeout=config.openai_timeout,
            max_retries=config.openai_max_retries,
        )

        self.collection = self.client.get_or_create_collection(
            name=config.chroma_collection,
            metadata={
                "description": COLLECTION_DESCRIPTION,
                CHROMA_DISTANCE_KEY: config.chroma_distance,
            },
        )

        self._warn_on_distance_mismatch()

        if verbose:
            print(f"Модель эмбеддингов: {config.embedding_model}")
            print(f"✓ ChromaDB готова. Чанков в коллекции: {self.count()}")

    # -- служебное ---------------------------------------------------------

    def _warn_on_distance_mismatch(self) -> None:
        """
        Сверяет метрику существующей коллекции с настройкой.

        ChromaDB задаёт метрику один раз при создании коллекции. Если папка
        chroma_db осталась от прошлого запуска с другой метрикой,
        get_or_create вернёт старую коллекцию и настройка будет
        проигнорирована молча.
        """
        metadata = self.collection.metadata or {}
        actual = metadata.get(CHROMA_DISTANCE_KEY)

        if actual and actual != self.config.chroma_distance:
            print(
                f"⚠ Коллекция '{self.config.chroma_collection}' создана с метрикой "
                f"'{actual}', а в настройках указано '{self.config.chroma_distance}'. "
                f"Метрику существующей коллекции сменить нельзя. Чтобы применить "
                f"настройку, удалите папку {self.config.chroma_persist_dir} и "
                f"переиндексируйте базу командой reindex."
            )

    def count(self) -> int:
        """Количество чанков в коллекции."""
        return self.collection.count()

    def list_sources(self) -> Dict[str, int]:
        """
        Возвращает словарь "источник -> число чанков".

        Читает метаданные всей коллекции. Для учебной базы это дёшево, на
        большой базе такой обход делать не стоит.
        """
        if self.count() == 0:
            return {}

        records = self.collection.get(include=["metadatas"])
        counts: Dict[str, int] = {}

        for metadata in records.get("metadatas") or []:
            source = (metadata or {}).get(META_SOURCE, "без источника")
            counts[source] = counts.get(source, 0) + 1

        return counts

    def resolve_sources(self, pattern: str) -> List[str]:
        """
        Находит источники, чьё имя содержит подстроку pattern.

        ChromaDB умеет фильтровать метаданные только точным совпадением или
        списком значений, поэтому поиск по части имени делается здесь, а в
        базу уходит уже готовый список.
        """
        needle = pattern.strip().lower()
        if not needle:
            return []
        return [name for name in self.list_sources() if needle in name.lower()]

    # -- индексация --------------------------------------------------------

    @staticmethod
    def _normalize_text(text: str) -> str:
        """
        Убирает отступы кода и лишние пустые строки.

        Документы заданы многострочными литералами внутри Python-файла, и
        каждая строка приходит с отступом в 8-12 пробелов. Без очистки эти
        пробелы попадают в чанки, занимают место в чанке и оплачиваются как
        токены при каждом запросе к модели.
        """
        lines = [line.strip() for line in text.strip().splitlines()]

        cleaned: List[str] = []
        for line in lines:
            # Схлопываем подряд идущие пустые строки в одну
            if not line and cleaned and not cleaned[-1]:
                continue
            cleaned.append(line)

        return "\n".join(cleaned)

    def _create_chunks(self, text: str) -> List[str]:
        """Режет текст на чанки заданного размера с перекрытием."""
        chunk_size = self.config.chunk_size
        overlap = self.config.chunk_overlap

        text = self._normalize_text(text)

        chunks: List[str] = []
        start = 0

        while start < len(text):
            end = start + chunk_size
            chunk = text[start:end].strip()
            if chunk:
                chunks.append(chunk)
            start = end - overlap

        return chunks

    def _create_embeddings(self, texts: List[str]) -> List[List[float]]:
        """Считает эмбеддинги для списка текстов одним запросом к API."""
        response = self.openai_client.embeddings.create(
            model=self.config.embedding_model,
            input=texts,
            encoding_format=EMBEDDING_ENCODING_FORMAT,
        )
        return [item.embedding for item in response.data]

    def add_documents(self, documents: List[Tuple[str, str]]) -> int:
        """
        Добавляет документы в хранилище.

        Args:
            documents: список пар (имя документа, текст)

        Returns:
            Сколько чанков добавлено
        """
        all_chunks: List[str] = []
        all_metadatas: List[dict] = []
        all_ids: List[str] = []

        chunk_id = self.count()

        if self.verbose:
            print(f"\nДобавление {len(documents)} документов в ChromaDB")

        for doc_name, doc_text in documents:
            chunks = self._create_chunks(doc_text)

            if self.verbose:
                print(f"  • {doc_name}: {len(chunks)} чанков")

            for chunk in chunks:
                all_chunks.append(chunk)
                all_metadatas.append({
                    META_SOURCE: doc_name,
                    META_CHUNK_LENGTH: len(chunk),
                })
                all_ids.append(f"{CHUNK_ID_PREFIX}{chunk_id}")
                chunk_id += 1

        if not all_chunks:
            print("⚠ Документы пустые, добавлять нечего")
            return 0

        if self.verbose:
            print(
                f"\nСоздание эмбеддингов для {len(all_chunks)} чанков "
                f"(модель {self.config.embedding_model})"
            )

        batch_size = self.config.embedding_batch_size
        all_embeddings: List[List[float]] = []

        for index in range(0, len(all_chunks), batch_size):
            batch = all_chunks[index:index + batch_size]
            if self.verbose:
                last = min(index + batch_size, len(all_chunks))
                print(f"  Чанки {index + 1}-{last} из {len(all_chunks)}")
            all_embeddings.extend(self._create_embeddings(batch))

        self.collection.add(
            embeddings=all_embeddings,
            documents=all_chunks,
            metadatas=all_metadatas,
            ids=all_ids,
        )

        if self.verbose:
            print(f"✓ Добавлено {len(all_chunks)} чанков. Всего в базе: {self.count()}")

        return len(all_chunks)

    def clear_collection(self) -> None:
        """Удаляет коллекцию и создаёт её заново пустой."""
        self.client.delete_collection(self.config.chroma_collection)
        self.collection = self.client.get_or_create_collection(
            name=self.config.chroma_collection,
            metadata={
                "description": COLLECTION_DESCRIPTION,
                CHROMA_DISTANCE_KEY: self.config.chroma_distance,
            },
        )
        print("✓ Коллекция очищена")

    # -- поиск -------------------------------------------------------------

    def search(
        self,
        query: str,
        top_k: Optional[int] = None,
        source_filter: Optional[str] = None,
    ) -> Tuple[List[SearchResult], TimingReport]:
        """
        Семантический поиск по базе.

        Args:
            query: текст запроса
            top_k: сколько фрагментов вернуть, по умолчанию из конфига
            source_filter: подстрока имени источника, по умолчанию из конфига

        Returns:
            Пара (список результатов, отчёт о времени этапов)
        """
        effective_top_k = top_k if top_k is not None else self.config.top_k
        effective_filter = (
            source_filter if source_filter is not None else self.config.source_filter
        )

        timer = StageTimer()

        if self.count() == 0:
            print("⚠ Коллекция пуста, искать нечего")
            return [], timer.finish()

        # Собираем условие по метаданным, если задан фильтр источников
        where: Optional[dict] = None
        if effective_filter:
            matched = self.resolve_sources(effective_filter)
            if not matched:
                print(
                    f"⚠ Под фильтр '{effective_filter}' не подошёл ни один источник, "
                    f"поиск идёт по всей базе"
                )
            else:
                where = {META_SOURCE: {"$in": matched}}
                if self.verbose:
                    print(f"• Фильтр по источникам: {', '.join(matched)}")

        with timer.stage(STAGE_QUERY_EMBEDDING):
            query_embedding = self._create_embeddings([query])[0]

        with timer.stage(STAGE_VECTOR_SEARCH):
            response = self.collection.query(
                query_embeddings=[query_embedding],
                n_results=min(effective_top_k, self.count()),
                where=where,
            )

        results: List[SearchResult] = []
        documents = response.get("documents") or [[]]

        if documents and documents[0]:
            for index in range(len(documents[0])):
                metadata = response["metadatas"][0][index] or {}
                results.append(SearchResult(
                    text=documents[0][index],
                    source=metadata.get(META_SOURCE, "без источника"),
                    distance=response["distances"][0][index],
                ))

        return results, timer.finish()


def get_sample_documents() -> List[Tuple[str, str]]:
    """
    Тестовая база знаний.

    Тексты те же, что в проекте урока: основы Python, машинное обучение и RAG,
    векторные базы данных. Добавлять свои документы нужно сюда.
    """
    return [
        (
            "Python Основы",
            """
            Python - это высокоуровневый язык программирования общего назначения.
            Он был создан Гвидо ван Россумом и впервые выпущен в 1991 году.

            Python известен своей простотой и читаемостью кода. Философия языка
            подчеркивает важность читаемости кода и позволяет программистам
            выражать концепции в меньшем количестве строк кода, чем это было бы
            возможно в других языках.

            Основные возможности Python включают:
            - Динамическую типизацию
            - Автоматическое управление памятью
            - Обширную стандартную библиотеку
            - Поддержку множественных парадигм программирования

            Python широко используется в веб-разработке, анализе данных,
            машинном обучении, автоматизации и научных вычислениях.
            """
        ),
        (
            "Машинное обучение и AI",
            """
            Машинное обучение (Machine Learning) - это подраздел искусственного
            интеллекта, который изучает алгоритмы и статистические модели,
            позволяющие компьютерам выполнять задачи без явного программирования.

            Основные типы машинного обучения:

            1. Обучение с учителем (Supervised Learning)
            В этом подходе модель обучается на размеченных данных, где каждый
            пример имеет известный правильный ответ. Примеры: классификация
            изображений, предсказание цен на недвижимость.

            2. Обучение без учителя (Unsupervised Learning)
            Модель ищет закономерности в неразмеченных данных. Примеры:
            кластеризация клиентов, обнаружение аномалий.

            3. Обучение с подкреплением (Reinforcement Learning)
            Агент обучается принимать решения, взаимодействуя со средой и
            получая награды или штрафы.

            RAG (Retrieval-Augmented Generation) - это техника, которая улучшает
            качество ответов языковых моделей, дополняя их внешними знаниями из
            базы данных. Это позволяет модели давать более точные и актуальные
            ответы, основанные на конкретных документах.
            """
        ),
        (
            "Векторные базы данных",
            """
            Векторные базы данных - это специализированные системы хранения данных,
            оптимизированные для хранения и поиска векторных эмбеддингов.

            Что такое эмбеддинги?
            Эмбеддинги - это векторные представления данных (текста, изображений,
            аудио) в многомерном пространстве. Семантически похожие объекты
            располагаются близко друг к другу в этом пространстве.

            ChromaDB - это открытая векторная база данных, разработанная специально
            для работы с эмбеддингами в приложениях с искусственным интеллектом.

            Преимущества ChromaDB:
            - Простота использования и встраивания в приложения
            - Поддержка персистентного хранения данных
            - Встроенная поддержка различных моделей эмбеддингов
            - Быстрый семантический поиск
            - Возможность работы как локально, так и в клиент-серверном режиме

            Векторные базы данных критически важны для RAG-систем, так как они
            позволяют быстро находить релевантные документы на основе семантического
            сходства запроса с содержимым базы данных.

            OpenAI предоставляет мощные модели для создания эмбеддингов, такие как
            text-embedding-3-small и text-embedding-3-large. Эти модели создают
            высококачественные векторные представления текста, которые отлично
            работают для семантического поиска в различных языках, включая русский.
            """
        ),
    ]
