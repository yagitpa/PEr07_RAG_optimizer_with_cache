"""
Кеш ответов LLM.

Кеш хранит не только текст ответа, но и время, за которое этот ответ был
получен в первый раз. Это нужно, чтобы при попадании в кеш показывать
пользователю честное сравнение: столько заняло вычисление, столько занимает
выдача из кеша.

Ключ кеша - SHA-256 от нормализованного запроса и (по настройке) от строки
параметров поиска. Второе принципиально: если top_k или модель поменялись,
старый ответ уже не соответствует текущим настройкам, и кеш обязан
промахнуться.
"""

import hashlib
import json
import os
from dataclasses import dataclass, asdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, Optional

from config import CACHE_FORMAT_VERSION, CACHE_PREVIEW_LENGTH


@dataclass
class CacheEntry:
    """Одна запись кеша."""
    answer: str
    generation_seconds: Optional[float]  # сколько занял полный цикл RAG
    created_at: str                      # время создания записи, ISO 8601, UTC
    signature: str                       # параметры, при которых получен ответ

    def to_dict(self) -> dict:
        return asdict(self)

    @staticmethod
    def from_dict(raw: dict) -> "CacheEntry":
        return CacheEntry(
            answer=raw.get("answer", ""),
            generation_seconds=raw.get("generation_seconds"),
            created_at=raw.get("created_at", ""),
            signature=raw.get("signature", ""),
        )


class ResponseCache:
    """
    Кеш ответов на диске в формате JSON.

    Структура файла:
        {
          "version": 2,
          "entries": {
            "<sha256>": {"answer": "...", "generation_seconds": 1.83,
                         "created_at": "...", "signature": "..."}
          }
        }
    """

    def __init__(self, cache_file: str, enabled: bool = True, verbose: bool = True):
        """
        Args:
            cache_file: путь к JSON-файлу кеша
            enabled: если False, кеш всегда промахивается и ничего не пишет
            verbose: печатать ли сообщения о попадании и промахе
        """
        self.cache_file = Path(cache_file)
        self.enabled = enabled
        self.verbose = verbose
        self.entries: Dict[str, CacheEntry] = {}

        # Счётчики за текущую сессию
        self.hits = 0
        self.misses = 0

        self._load()

    # -- ключи -------------------------------------------------------------

    @staticmethod
    def build_key(query: str, signature: str = "") -> str:
        """
        Считает ключ кеша.

        Запрос нормализуется: нижний регистр и схлопывание пробелов. То есть
        "Что такое RAG?" и "что  такое   rag?" дадут один ключ.
        """
        normalized = " ".join(query.lower().split())
        payload = f"{normalized}::{signature}" if signature else normalized
        return hashlib.sha256(payload.encode("utf-8")).hexdigest()

    # -- чтение и запись ---------------------------------------------------

    def get(self, query: str, signature: str = "") -> Optional[CacheEntry]:
        """Возвращает запись из кеша или None."""
        if not self.enabled:
            if self.verbose:
                print("• Кеш выключен настройкой CACHE_ENABLED, идём полным циклом")
            self.misses += 1
            return None

        key = self.build_key(query, signature)
        entry = self.entries.get(key)

        preview = query[:CACHE_PREVIEW_LENGTH]
        if entry is not None:
            self.hits += 1
            if self.verbose:
                print(f"✓ Кеш найден для запроса: '{preview}'")
            return entry

        self.misses += 1
        if self.verbose:
            print(f"✗ Кеш не найден для запроса: '{preview}', выполняем RAG")
        return None

    def set(
        self,
        query: str,
        answer: str,
        generation_seconds: Optional[float] = None,
        signature: str = "",
    ) -> None:
        """Сохраняет ответ в кеш и сразу пишет файл на диск."""
        if not self.enabled:
            return

        key = self.build_key(query, signature)
        self.entries[key] = CacheEntry(
            answer=answer,
            generation_seconds=generation_seconds,
            created_at=datetime.now(timezone.utc).isoformat(timespec="seconds"),
            signature=signature,
        )
        self._save()

        if self.verbose:
            print("✓ Ответ сохранён в кеш")

    def invalidate(self, query: str, signature: str = "") -> bool:
        """
        Удаляет одну запись. Возвращает True, если запись была.

        Нужно режиму замеров: чтобы честно измерить полный цикл, запись по
        этому вопросу должна отсутствовать.
        """
        key = self.build_key(query, signature)
        if key in self.entries:
            del self.entries[key]
            self._save()
            return True
        return False

    def clear(self) -> None:
        """Полностью очищает кеш и удаляет файл."""
        self.entries = {}
        if self.cache_file.exists():
            self.cache_file.unlink()
        if self.verbose:
            print("✓ Кеш очищен")

    def size(self) -> int:
        """Количество записей."""
        return len(self.entries)

    def hit_rate(self) -> Optional[float]:
        """Доля попаданий за сессию, None если обращений не было."""
        total = self.hits + self.misses
        if total == 0:
            return None
        return self.hits / total

    # -- работа с файлом ---------------------------------------------------

    def _save(self) -> None:
        """
        Пишет кеш на диск.

        Запись идёт во временный файл с последующей заменой: если процесс
        прервать посередине, целый старый файл останется на месте вместо
        обрезанного нового.
        """
        payload = {
            "version": CACHE_FORMAT_VERSION,
            "entries": {key: entry.to_dict() for key, entry in self.entries.items()},
        }

        temp_path = self.cache_file.with_suffix(self.cache_file.suffix + ".tmp")

        try:
            with open(temp_path, "w", encoding="utf-8") as handle:
                json.dump(payload, handle, ensure_ascii=False, indent=2)
            os.replace(temp_path, self.cache_file)
        except Exception as error:
            print(f"⚠ Не удалось сохранить кеш: {error}")
            if temp_path.exists():
                try:
                    temp_path.unlink()
                except OSError:
                    pass

    def _load(self) -> None:
        """Читает кеш с диска, понимает старый плоский формат из урока."""
        if not self.cache_file.exists():
            return

        try:
            with open(self.cache_file, "r", encoding="utf-8") as handle:
                raw = json.load(handle)
        except Exception as error:
            print(f"⚠ Не удалось прочитать кеш, начинаем с пустого: {error}")
            self.entries = {}
            return

        # Формат версии 2: словарь с ключами version и entries
        if isinstance(raw, dict) and "entries" in raw:
            self.entries = {
                key: CacheEntry.from_dict(value)
                for key, value in raw["entries"].items()
            }
        # Старый формат из урока: плоский словарь "хеш -> строка ответа"
        elif isinstance(raw, dict):
            self.entries = {
                key: CacheEntry(
                    answer=value if isinstance(value, str) else str(value),
                    generation_seconds=None,
                    created_at="",
                    signature="",
                )
                for key, value in raw.items()
            }
            print(
                f"• Кеш прочитан в старом формате ({len(self.entries)} записей), "
                f"при первой записи файл будет перезаписан в новом формате"
            )
        else:
            print("⚠ Файл кеша имеет неизвестную структуру, начинаем с пустого")
            self.entries = {}
            return

        if self.verbose and self.entries:
            print(f"✓ Загружен кеш: {len(self.entries)} записей")
