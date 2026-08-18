"""
Точка входа RAG-ассистента.

Три режима работы:
  1. Интерактивный - вопросы вводит пользователь.
  2. Демонстрационный - прогоняются вопросы из настройки DEMO_QUESTIONS.
  3. Замеры - каждый вопрос задаётся дважды (мимо кеша и из кеша),
     результат сводится в таблицу со временем и ускорением.

Время ответа показывается после каждого ответа с точностью, заданной
настройкой TIME_PRECISION (по умолчанию 5 знаков, то есть 0.00001 с).
"""

import sys
from dataclasses import dataclass
from typing import List, Optional, Tuple

from cache import ResponseCache
from config import (
    AppConfig,
    EXIT_COMMANDS,
    SEPARATOR_WIDTH,
    load_config,
)
from embeddings import EmbeddingStore, get_sample_documents
from rag import RAGAssistant
from timing import (
    STAGE_CACHE_LOOKUP,
    STAGE_CACHE_WRITE,
    StageTimer,
    format_ratio,
    format_seconds,
    speedup_line,
)

# Разделители вывода
LINE = "=" * SEPARATOR_WIDTH
THIN_LINE = "-" * SEPARATOR_WIDTH

# Команды интерактивного режима
CMD_HELP = "help"
CMD_CACHE = "cache"
CMD_CLEAR_CACHE = "clear_cache"
CMD_STATS = "stats"
CMD_SOURCES = "sources"
CMD_FILTER = "filter"
CMD_TOPK = "topk"
CMD_REINDEX = "reindex"
FILTER_OFF = "off"

# Ответы, считающиеся согласием
YES_ANSWERS = ("y", "yes", "д", "да", "")


@dataclass
class Session:
    """
    Изменяемые в рантайме параметры сессии.

    Конфигурация неизменяема, а top_k и фильтр источников удобно менять на
    ходу командами. Значения стартуют из конфигурации.
    """
    config: AppConfig
    store: EmbeddingStore
    assistant: RAGAssistant
    cache: ResponseCache
    top_k: int
    source_filter: str

    def signature(self) -> str:
        """Строка параметров для ключа кеша с учётом рантайм-изменений."""
        if not self.config.cache_key_includes_params:
            return ""
        base = self.config.search_signature()
        return f"{base}|runtime_k={self.top_k}|runtime_src={self.source_filter or 'all'}"


# ---------------------------------------------------------------------------
# Инициализация
# ---------------------------------------------------------------------------

def initialize_system(config: AppConfig) -> Session:
    """Поднимает все компоненты и при пустой базе заполняет её примерами."""
    print(LINE)
    print("🚀 ЗАПУСК RAG-АССИСТЕНТА")
    print(LINE)

    print("\nНастройки:")
    print(config.describe())

    print("\n[1/3] Кеш")
    cache = ResponseCache(cache_file=config.cache_file, enabled=config.cache_enabled)

    print("\n[2/3] Векторное хранилище")
    store = EmbeddingStore(config=config)

    if store.count() == 0:
        print("\n📝 База пуста, добавляем тестовые документы")
        store.add_documents(get_sample_documents())
    else:
        print(f"✓ В базе уже есть {store.count()} чанков")

    print("\n[3/3] RAG-ассистент")
    assistant = RAGAssistant(embedding_store=store, config=config)

    print("\n" + LINE)
    print("✅ СИСТЕМА ГОТОВА")
    print(LINE)

    return Session(
        config=config,
        store=store,
        assistant=assistant,
        cache=cache,
        top_k=config.top_k,
        source_filter=config.source_filter,
    )


# ---------------------------------------------------------------------------
# Обработка одного вопроса
# ---------------------------------------------------------------------------

def answer_question(query: str, session: Session, verbose: bool = True) -> Tuple[str, float, bool]:
    """
    Отвечает на вопрос: сначала кеш, при промахе полный цикл RAG.

    Returns:
        Кортеж (ответ, общее время в секундах, взят ли ответ из кеша)
    """
    config = session.config
    precision = config.time_precision
    signature = session.signature()

    print("\n" + LINE)
    print(f"❓ ВОПРОС: {query}")
    print(LINE)

    timer = StageTimer()

    # Шаг 1. Кеш
    print("\n[Шаг 1] Проверка кеша")
    with timer.stage(STAGE_CACHE_LOOKUP):
        entry = session.cache.get(query, signature=signature)

    if entry is not None:
        report = timer.finish(from_cache=True)

        print("\n💾 ОТВЕТ ИЗ КЕША:")
        print(THIN_LINE)
        print(entry.answer)
        print(THIN_LINE)
        print()
        print(report.render(precision, config.show_timing_breakdown))

        if entry.generation_seconds:
            print(f"   ⚡ {speedup_line(entry.generation_seconds, report.total_seconds, precision)}")

        return entry.answer, report.total_seconds, True

    # Шаг 2. Полный цикл RAG
    print("\n[Шаг 2] Полный цикл RAG: поиск и генерация")

    response = session.assistant.generate_response(
        query=query,
        top_k=session.top_k,
        source_filter=session.source_filter,
        verbose=verbose,
    )

    if response.timing is not None:
        for stage in response.timing.stages:
            timer.add(stage.name, stage.seconds)

    # Шаг 3. Запись в кеш. Неудачные ответы не кешируются.
    if not response.failed:
        print("\n[Шаг 3] Сохранение в кеш")
        generation_seconds = timer.elapsed()
        with timer.stage(STAGE_CACHE_WRITE):
            session.cache.set(
                query=query,
                answer=response.answer,
                generation_seconds=generation_seconds,
                signature=signature,
            )

    report = timer.finish(from_cache=False)

    print("\n💡 ОТВЕТ:")
    print(THIN_LINE)
    print(response.answer)
    print(THIN_LINE)
    print()
    print(report.render(precision, config.show_timing_breakdown))

    if response.usage.total_tokens:
        print(
            f"   🧾 Токены: {response.usage.prompt_tokens} на вход, "
            f"{response.usage.completion_tokens} на ответ, "
            f"{response.usage.total_tokens} всего"
        )

    return response.answer, report.total_seconds, False


# ---------------------------------------------------------------------------
# Команды интерактивного режима
# ---------------------------------------------------------------------------

def print_help() -> None:
    """Печатает список команд."""
    print("\nКоманды:")
    print(f"  {CMD_HELP}          - этот список")
    print(f"  {CMD_STATS}         - статистика системы")
    print(f"  {CMD_CACHE}         - размер кеша и попадания за сессию")
    print(f"  {CMD_CLEAR_CACHE}   - очистить кеш")
    print(f"  {CMD_SOURCES}       - источники в базе и число чанков")
    print(f"  {CMD_FILTER} <имя>  - искать только по источникам с таким именем")
    print(f"  {CMD_FILTER} {FILTER_OFF}     - снять фильтр")
    print(f"  {CMD_TOPK} <число>  - сменить top_k на время сессии")
    print(f"  {CMD_REINDEX}       - очистить базу и проиндексировать документы заново")
    print(f"  {'/'.join(EXIT_COMMANDS)} - выход")


def print_stats(session: Session) -> None:
    """Печатает состояние системы."""
    cache = session.cache
    rate = cache.hit_rate()

    print("\n📊 СТАТИСТИКА")
    print(f"  Чанков в ChromaDB     : {session.store.count()}")
    print(f"  Записей в кеше        : {cache.size()}")
    print(f"  Попаданий в кеш       : {cache.hits}")
    print(f"  Промахов              : {cache.misses}")
    if rate is not None:
        print(f"  Доля попаданий        : {rate:.0%}")
    print(f"  Модель LLM            : {session.config.llm_model}")
    print(f"  Модель эмбеддингов    : {session.config.embedding_model}")
    print(f"  TOP_K (текущий)       : {session.top_k}")
    print(f"  Фильтр источников     : {session.source_filter or 'нет'}")


def print_sources(session: Session) -> None:
    """Печатает источники в базе."""
    sources = session.store.list_sources()

    if not sources:
        print("\nБаза пуста")
        return

    print("\n📁 Источники в базе:")
    for name, count in sorted(sources.items()):
        print(f"  • {name}: {count} чанков")


def handle_filter(argument: str, session: Session) -> None:
    """Ставит или снимает фильтр по источнику."""
    value = argument.strip()

    if not value or value.lower() == FILTER_OFF:
        session.source_filter = ""
        print("✓ Фильтр снят, поиск идёт по всей базе")
        return

    matched = session.store.resolve_sources(value)
    if not matched:
        available = ", ".join(sorted(session.store.list_sources())) or "база пуста"
        print(f"⚠ Источников с именем '{value}' нет. Доступные: {available}")
        return

    session.source_filter = value
    print(f"✓ Фильтр установлен: {', '.join(matched)}")


def handle_topk(argument: str, session: Session) -> None:
    """Меняет top_k на время сессии."""
    try:
        value = int(argument.strip())
    except ValueError:
        print(f"⚠ Нужно целое число, получено: '{argument.strip()}'")
        return

    if value < 1:
        print("⚠ top_k должен быть не меньше 1")
        return

    session.top_k = value
    print(f"✓ TOP_K на время сессии: {value}")
    print("  Ключи кеша учитывают top_k, поэтому те же вопросы пойдут полным циклом")


def handle_reindex(session: Session) -> None:
    """Пересобирает базу с нуля."""
    confirm = input("Очистить базу и проиндексировать документы заново? (y/n): ").strip().lower()
    if confirm not in YES_ANSWERS:
        print("Отменено")
        return

    session.store.clear_collection()
    session.store.add_documents(get_sample_documents())
    print("✓ Переиндексация завершена")


def handle_command(user_input: str, session: Session) -> bool:
    """
    Обрабатывает служебную команду.

    Returns:
        True, если ввод был командой и вопрос задавать не нужно
    """
    lowered = user_input.lower()
    parts = user_input.split(maxsplit=1)
    keyword = parts[0].lower() if parts else ""
    argument = parts[1] if len(parts) > 1 else ""

    if lowered == CMD_HELP:
        print_help()
        return True

    if lowered == CMD_STATS:
        print_stats(session)
        return True

    if lowered == CMD_CACHE:
        print(f"\n📊 В кеше {session.cache.size()} записей")
        print(f"   За сессию: {session.cache.hits} попаданий, {session.cache.misses} промахов")
        return True

    if lowered == CMD_CLEAR_CACHE:
        session.cache.clear()
        return True

    if lowered == CMD_SOURCES:
        print_sources(session)
        return True

    if lowered == CMD_REINDEX:
        handle_reindex(session)
        return True

    if keyword == CMD_FILTER:
        handle_filter(argument, session)
        return True

    if keyword == CMD_TOPK:
        handle_topk(argument, session)
        return True

    return False


# ---------------------------------------------------------------------------
# Режимы работы
# ---------------------------------------------------------------------------

def interactive_mode(session: Session) -> None:
    """Диалог с пользователем."""
    print("\n" + LINE)
    print("💬 ИНТЕРАКТИВНЫЙ РЕЖИМ")
    print(LINE)
    print_help()

    while True:
        try:
            user_input = input("\n👤 Вы: ").strip()

            if user_input.lower() in EXIT_COMMANDS or user_input == "":
                print("\n👋 До свидания")
                break

            if handle_command(user_input, session):
                continue

            answer_question(user_input, session)

        except KeyboardInterrupt:
            print("\n\n👋 Прервано, до свидания")
            break
        except Exception as error:
            print(f"\n❌ Ошибка: {error}")


def demo_mode(session: Session) -> None:
    """Прогон заранее заданных вопросов."""
    questions: List[str] = list(session.config.demo_questions)

    print("\n" + LINE)
    print("🎬 ДЕМОНСТРАЦИОННЫЙ РЕЖИМ")
    print(LINE)
    print(f"\nВопросов в списке: {len(questions)}")
    print("Список задаётся настройкой DEMO_QUESTIONS в .env")

    # Последним идёт повтор первого вопроса: на нём видно работу кеша
    if questions:
        questions.append(questions[0])

    for index, question in enumerate(questions, start=1):
        print(f"\n\n{'#' * SEPARATOR_WIDTH}")
        print(f"ВОПРОС {index} из {len(questions)}")
        print("#" * SEPARATOR_WIDTH)

        answer_question(question, session)

        if index < len(questions):
            input("\n[Enter для следующего вопроса]")

    print("\n\n" + LINE)
    print("✅ ДЕМОНСТРАЦИЯ ЗАВЕРШЕНА")
    print(LINE)


def benchmark_mode(session: Session) -> None:
    """
    Режим замеров.

    Каждый вопрос задаётся дважды: первый раз мимо кеша, второй раз из кеша.
    Перед первым прогоном запись по этому вопросу удаляется, иначе замер
    полного цикла получится нечестным.
    """
    questions: List[str] = list(session.config.demo_questions)
    precision = session.config.time_precision

    print("\n" + LINE)
    print("📐 РЕЖИМ ЗАМЕРОВ")
    print(LINE)
    print(f"\nВопросов: {len(questions)}. Каждый задаётся дважды.")
    print("Записи кеша по этим вопросам будут удалены перед первым прогоном.")

    if not session.config.cache_enabled:
        print("\n⚠ Кеш выключен настройкой CACHE_ENABLED, второй прогон не ускорится")

    confirm = input("\nПродолжить? (y/n): ").strip().lower()
    if confirm not in YES_ANSWERS:
        print("Отменено")
        return

    rows: List[Tuple[str, float, float]] = []

    for index, question in enumerate(questions, start=1):
        print(f"\n\n{'#' * SEPARATOR_WIDTH}")
        print(f"ЗАМЕР {index} из {len(questions)}: {question}")
        print("#" * SEPARATOR_WIDTH)

        session.cache.invalidate(question, signature=session.signature())

        print("\n--- Прогон 1: полный цикл RAG ---")
        _, cold_seconds, _ = answer_question(question, session)

        print("\n--- Прогон 2: тот же вопрос ---")
        _, warm_seconds, from_cache = answer_question(question, session, verbose=False)

        if not from_cache:
            print("⚠ Второй прогон не попал в кеш, сравнение будет некорректным")

        rows.append((question, cold_seconds, warm_seconds))

    print("\n\n" + LINE)
    print("📊 ИТОГИ ЗАМЕРОВ")
    print(LINE)

    for question, cold, warm in rows:
        ratio = cold / warm if warm > 0 else 0
        print(f"\n• {question}")
        print(f"    полный цикл : {format_seconds(cold, precision)}")
        print(f"    из кеша     : {format_seconds(warm, precision)}")
        print(f"    ускорение   : в {format_ratio(ratio)} раз")

    if rows:
        avg_cold = sum(row[1] for row in rows) / len(rows)
        avg_warm = sum(row[2] for row in rows) / len(rows)
        print(f"\nСреднее по {len(rows)} вопросам:")
        print(f"    полный цикл : {format_seconds(avg_cold, precision)}")
        print(f"    из кеша     : {format_seconds(avg_warm, precision)}")

    print("\n" + LINE)


# ---------------------------------------------------------------------------
# Запуск
# ---------------------------------------------------------------------------

def choose_mode() -> str:
    """Спрашивает режим работы."""
    print("\n" + LINE)
    print("ВЫБОР РЕЖИМА")
    print(LINE)
    print("\n1. Интерактивный - свои вопросы")
    print("2. Демонстрационный - вопросы из DEMO_QUESTIONS")
    print("3. Замеры - каждый вопрос дважды, таблица со временем")
    return input("\nРежим (1, 2 или 3, по умолчанию 1): ").strip()


def main() -> None:
    """Главная функция."""
    try:
        config = load_config()
    except ValueError as error:
        print(f"❌ {error}")
        sys.exit(1)

    if not config.openai_api_key:
        print("❌ Не задан OPENAI_API_KEY.")
        print("   Скопируйте env.example в .env и впишите ключ:")
        print("   OPENAI_API_KEY=sk-...")
        print("   Без ключа не работают ни эмбеддинги, ни генерация ответов.")
        sys.exit(1)

    try:
        session = initialize_system(config)
    except Exception as error:
        print(f"\n❌ Не удалось запустить систему: {error}")
        import traceback
        traceback.print_exc()
        sys.exit(1)

    mode = choose_mode()

    if mode == "2":
        demo_mode(session)
        answer = input("\nПерейти в интерактивный режим? (y/n): ").strip().lower()
        if answer in YES_ANSWERS:
            interactive_mode(session)
    elif mode == "3":
        benchmark_mode(session)
        answer = input("\nПерейти в интерактивный режим? (y/n): ").strip().lower()
        if answer in YES_ANSWERS:
            interactive_mode(session)
    else:
        interactive_mode(session)


if __name__ == "__main__":
    main()
