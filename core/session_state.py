"""Runtime session state storage."""

from __future__ import annotations

import time
from collections import deque
from typing import Iterable

from .models import RecentImageRecord, SessionContext, SessionRuntimeState
from .user_settings import user_settings

class SessionStateStore:
    """In-memory runtime state. Session overrides are intentionally ephemeral."""

    def __init__(self) -> None:
        self._states: dict[str, SessionRuntimeState] = {}

    def get(self, session: SessionContext) -> SessionRuntimeState:
        """获取会话状态，优先加载用户独立设置"""
        state = self._states.setdefault(session.session_key, SessionRuntimeState())

        # ==================== 新增：用户独立设置优先 ====================
        user_id = None
        if hasattr(session, 'user_id') and session.user_id:
            user_id = str(session.user_id)
        elif hasattr(session, 'sender') and session.sender and hasattr(session.sender, 'user_id'):
            user_id = str(session.sender.user_id)

        if user_id:
            user_data = user_settings.get(user_id)
            if user_data:
                # 覆盖用户独立设置（只覆盖已实现的三项）
                if user_data.get("selected_model"):
                    state.selected_model = user_data["selected_model"]
                if user_data.get("selected_artist_index") is not None:
                    state.selected_artist_index = user_data["selected_artist_index"]
                if user_data.get("selected_size"):
                    state.selected_size = user_data["selected_size"]
        # ============================================================

        return state

    def track_image(
        self,
        session: SessionContext,
        message_id: str,
        prompt: str,
    ) -> None:
        state = self.get(session)
        state.recent_images.appendleft(
            RecentImageRecord(
                message_id=str(message_id),
                prompt=prompt,
                created_at=time.time(),
            )
        )

    def recent_images(self, session: SessionContext) -> Iterable[RecentImageRecord]:
        return tuple(self.get(session).recent_images)

    def find_recent_image(
        self,
        session: SessionContext,
        message_id: str,
    ) -> RecentImageRecord | None:
        for item in self.get(session).recent_images:
            if item.message_id == str(message_id):
                return item
        return None

    def latest_image(self, session: SessionContext) -> RecentImageRecord | None:
        state = self.get(session)
        return state.recent_images[0] if state.recent_images else None

    def prune_expired_images(
        self,
        session: SessionContext,
        max_age_seconds: float,
    ) -> None:
        state = self.get(session)
        now = time.time()
        valid = [
            item
            for item in state.recent_images
            if (now - item.created_at) <= max_age_seconds
        ]
        state.recent_images = deque(valid, maxlen=20)
