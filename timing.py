"""
Замер и отображение времени ответа.

Используется time.perf_counter() - монотонный счётчик с максимальным
разрешением, доступным в системе. Он не зависит от перевода системных часов,
в отличие от time.time().

Про точность. Отображение идёт с точностью до 0.00001 секунды, как требует
задание. Само измерение при этом точнее вывода (perf_counter на типичной
машине даёт разрешение порядка десятков-сотен наносекунд), но реальная
воспроизводимость ниже: сетевой запрос к OpenAI плавает на десятки
миллисекунд от прогона к прогону. То есть последние знаки честно показывают
измеренное значение, но повторить его до последней цифры не получится.
"""

from contextlib import contextmanager
from dataclasses import dataclass, field
from time import perf_counter
from typing import Iterator, List, Optional

# Подпись единицы измерения в выводе
TIME_UNIT = "с"

# Названия этапов. Собраны здесь, чтобы одинаково выглядеть во всех режимах.
STAGE_CACHE_LOOKUP = "проверка кеша"
STAGE_QUERY_EMBEDDING = "эмбеддинг запроса"
STAGE_VECTOR_SEARCH = "поиск в ChromaDB"
STAGE_LLM_GENERATION = "генерация ответа LLM"
STAGE_CACHE_WRITE = "запись в кеш"

# Символы дерева для разбивки по этапам
TREE_BRANCH = "├─"
TREE_LAST = "└─"


def format_seconds(seconds: float, precision: int) -> str:
    """
    Форматирует секунды с заданным числом знаков после запятой.

    Args:
        seconds: измеренное время
        precision: знаков после запятой (5 = точность 0.00001 с)

    Returns:
        Строка вида "1.83472 с"
    """
    return f"{seconds:.{precision}f} {TIME_UNIT}"


@dataclass
class Stage:
    """Один измеренный этап обработки запроса."""
    name: str
    seconds: float


@dataclass
class TimingReport:
    """
    Набор замеров по одному запросу.

    total считается отдельным счётчиком от начала до конца обработки, а не
    суммой этапов: между этапами есть неучтённое время (форматирование
    контекста, сборка промпта), и его лучше видеть как расхождение, чем
    прятать.
    """
    stages: List[Stage] = field(default_factory=list)
    total_seconds: float = 0.0
    from_cache: bool = False

    def add(self, name: str, seconds: float) -> None:
        """Добавляет готовый замер."""
        self.stages.append(Stage(name=name, seconds=seconds))

    def get(self, name: str) -> Optional[float]:
        """Возвращает время этапа по имени или None, если этапа не было."""
        for stage in self.stages:
            if stage.name == name:
                return stage.seconds
        return None

    def render(self, precision: int, show_breakdown: bool = True) -> str:
        """
        Собирает блок с временем ответа для вывода в терминал.

        Args:
            precision: знаков после запятой
            show_breakdown: показывать ли разбивку по этапам

        Returns:
            Готовый многострочный текст
        """
        source_note = " (ответ из кеша)" if self.from_cache else ""
        header = f"⏱  Время ответа: {format_seconds(self.total_seconds, precision)}{source_note}"

        if not show_breakdown or not self.stages:
            return header

        # Выравниваем названия этапов по самому длинному
        width = max(len(stage.name) for stage in self.stages)

        lines = [header]
        for index, stage in enumerate(self.stages):
            is_last = index == len(self.stages) - 1
            prefix = TREE_LAST if is_last else TREE_BRANCH
            padded_name = stage.name.ljust(width)
            lines.append(
                f"   {prefix} {padded_name} : {format_seconds(stage.seconds, precision)}"
            )

        # Показываем, сколько времени не попало ни в один измеренный этап
        measured = sum(stage.seconds for stage in self.stages)
        unaccounted = self.total_seconds - measured
        if unaccounted > 0:
            lines.append(
                f"      {'прочее'.ljust(width)} : {format_seconds(unaccounted, precision)}"
            )

        return "\n".join(lines)


class StageTimer:
    """
    Секундомер с этапами.

    Пример:
        timer = StageTimer()
        with timer.stage("поиск"):
            do_search()
        report = timer.finish()
    """

    def __init__(self) -> None:
        self._start = perf_counter()
        self._report = TimingReport()

    @contextmanager
    def stage(self, name: str) -> Iterator[None]:
        """Замеряет блок кода и записывает результат под указанным именем."""
        started = perf_counter()
        try:
            yield
        finally:
            self._report.add(name, perf_counter() - started)

    def add(self, name: str, seconds: float) -> None:
        """Добавляет замер, сделанный где-то ещё."""
        self._report.add(name, seconds)

    def elapsed(self) -> float:
        """Сколько прошло с момента создания секундомера."""
        return perf_counter() - self._start

    def finish(self, from_cache: bool = False) -> TimingReport:
        """Останавливает общий счётчик и возвращает отчёт."""
        self._report.total_seconds = self.elapsed()
        self._report.from_cache = from_cache
        return self._report


def format_ratio(value: float) -> str:
    """
    Форматирует кратность с пробелом как разделителем тысяч.

    Замена делается только в самом числе: если применить её ко всей фразе,
    пострадают запятые обычного текста.
    """
    return f"{value:,.0f}".replace(",", " ")


def speedup_line(slow_seconds: float, fast_seconds: float, precision: int) -> str:
    """
    Строка сравнения двух замеров.

    Args:
        slow_seconds: время полного цикла RAG
        fast_seconds: время ответа из кеша
        precision: знаков после запятой

    Returns:
        Текст вида "быстрее в 9 656 раз (было 1.83472 с, стало 0.00019 с)"
    """
    if fast_seconds <= 0:
        return "ускорение посчитать не удалось: измеренное время равно нулю"

    ratio = format_ratio(slow_seconds / fast_seconds)
    return (
        f"быстрее в {ratio} раз "
        f"(было {format_seconds(slow_seconds, precision)}, "
        f"стало {format_seconds(fast_seconds, precision)})"
    )
