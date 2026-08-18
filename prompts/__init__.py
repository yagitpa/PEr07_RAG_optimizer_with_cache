"""
Загрузка текстов промптов из файлов.

Промпты вынесены в отдельные .md файлы, а не в код: их правят чаще, чем
логику, и правка не должна требовать чтения Python-файла.

Подстановка идёт по плейсхолдерам вида {{ИМЯ}}, а не через str.format().
Причина: в контекст попадают куски документов, где могут встретиться
одиночные фигурные скобки, и format() на них падает с KeyError.
"""

from pathlib import Path
from typing import Dict
import re

# Директория с файлами промптов (эта же папка)
PROMPTS_DIR = Path(__file__).parent

# Имена файлов промптов
SYSTEM_PROMPT_FILE = "system.md"
CONTEXT_MESSAGE_FILE = "context_message.md"

# Регулярка для поиска незакрытых плейсхолдеров после подстановки
PLACEHOLDER_PATTERN = re.compile(r"\{\{[A-Z_]+\}\}")

# Кеш прочитанных файлов, чтобы не читать диск на каждый запрос
_FILE_CACHE: Dict[str, str] = {}


def load_prompt(file_name: str, use_cache: bool = True) -> str:
    """
    Читает файл промпта из директории prompts/.

    Args:
        file_name: имя файла, например "system.md"
        use_cache: брать ли из памяти, если файл уже читался

    Returns:
        Текст промпта

    Raises:
        FileNotFoundError: файла нет на диске
    """
    if use_cache and file_name in _FILE_CACHE:
        return _FILE_CACHE[file_name]

    path = PROMPTS_DIR / file_name

    if not path.exists():
        raise FileNotFoundError(
            f"Не найден файл промпта: {path}. "
            f"Проверьте, что папка prompts/ на месте и файл не переименован."
        )

    text = path.read_text(encoding="utf-8").strip()
    _FILE_CACHE[file_name] = text

    return text


def render(template: str, **values: str) -> str:
    """
    Подставляет значения в плейсхолдеры {{ИМЯ}} шаблона.

    После подстановки проверяет, что незаполненных плейсхолдеров не осталось.
    Это детерминированная проверка: опечатка в имени плейсхолдера ловится
    здесь, а не превращается в кривой промпт, уехавший в модель.

    Args:
        template: текст шаблона
        **values: пары ИМЯ=значение (имя без фигурных скобок)

    Returns:
        Готовый текст

    Raises:
        ValueError: остался незаполненный плейсхолдер
    """
    result = template

    for key, value in values.items():
        result = result.replace("{{" + key + "}}", str(value))

    leftovers = PLACEHOLDER_PATTERN.findall(result)
    if leftovers:
        raise ValueError(
            f"В промпте остались незаполненные плейсхолдеры: {', '.join(sorted(set(leftovers)))}. "
            f"Переданы значения: {', '.join(sorted(values.keys())) or '(нет)'}"
        )

    return result


def build_system_prompt(max_tokens: int) -> str:
    """Собирает системный промпт с подставленным лимитом токенов."""
    return render(load_prompt(SYSTEM_PROMPT_FILE), MAX_TOKENS=max_tokens)


def build_context_message(context: str, question: str) -> str:
    """Собирает сообщение пользователя с контекстом и вопросом."""
    return render(
        load_prompt(CONTEXT_MESSAGE_FILE),
        CONTEXT=context,
        QUESTION=question,
    )
