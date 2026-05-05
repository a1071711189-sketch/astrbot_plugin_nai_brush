import json
import os
from typing import Dict, Any

class UserSettingsStore:
    """最小用户设置存储（只存最常用3项）"""
    
    def __init__(self, data_dir: str = "data"):
        self.file_path = os.path.join(data_dir, "nai_user_settings.json")
        os.makedirs(data_dir, exist_ok=True)
        self._data: Dict[str, Dict] = self._load()

    def _load(self) -> Dict:
        if os.path.exists(self.file_path):
            try:
                with open(self.file_path, "r", encoding="utf-8") as f:
                    return json.load(f)
            except Exception:
                return {}
        return {}

    def _save(self):
        try:
            with open(self.file_path, "w", encoding="utf-8") as f:
                json.dump(self._data, f, ensure_ascii=False, indent=2)
        except Exception:
            pass

    def get(self, user_id: str) -> Dict[str, Any]:
        """获取用户设置"""
        key = str(user_id)
        return self._data.get(key, {})

    def set(self, user_id: str, key: str, value: Any):
        """保存单个设置"""
        user_key = str(user_id)
        if user_key not in self._data:
            self._data[user_key] = {}
        self._data[user_key][key] = value
        self._save()

# 全局实例
user_settings = UserSettingsStore()
