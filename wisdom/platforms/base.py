"""Platform interface — every platform implements this ABC."""
from __future__ import annotations

from abc import ABC, abstractmethod

from wisdom.schemas import PlatformResult, PostMeta


class BasePlatform(ABC):
    name: str = "base"

    @abstractmethod
    def post_video(self, video: bytes, thumbnail: bytes, meta: PostMeta) -> PlatformResult:
        ...

    @abstractmethod
    def post_image(self, image: bytes, meta: PostMeta) -> PlatformResult:
        ...

    def available(self) -> bool:
        return True
