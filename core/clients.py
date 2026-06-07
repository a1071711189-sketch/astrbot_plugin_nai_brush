"""Async HTTP clients for NAI and Danbooru."""

from __future__ import annotations

import base64
from collections import Counter
import json
import random
import re
from typing import Any

import httpx

from astrbot.api import logger

from .constants import (
    DEFAULT_NAI_ENDPOINT,
    DEFAULT_TIMEOUT_SECONDS,
    MIN_RECOMMENDED_ARTIST_POST_COUNT,
)

DANBOORU_API_BASE = "https://danbooru.donmai.us"


class NaiWebClient:
    def __init__(self) -> None:
        self._client = httpx.AsyncClient(
            timeout=DEFAULT_TIMEOUT_SECONDS,
            verify=False,
            trust_env=True,
            follow_redirects=True,
        )

    async def close(self) -> None:
        await self._client.aclose()

    async def generate_image(
        self,
        *,
        prompt: str,
        model_config: dict[str, Any],
        size: str | None = None,
    ) -> tuple[bool, str]:
        base_url = str(model_config.get("base_url") or "").rstrip("/")
        if not base_url:
            logger.error("[nai_pic] generate_image: base_url 未配置")
            return False, "base_url 未配置"

        endpoint = str(model_config.get("nai_endpoint") or DEFAULT_NAI_ENDPOINT)
        if not endpoint.startswith("/"):
            endpoint = f"/{endpoint}"

        prompt_add = str(model_config.get("custom_prompt_add") or "").strip()
        full_prompt = f"{prompt_add}, {prompt}" if prompt_add else prompt

        artist_prompt = str(model_config.get("nai_artist_prompt") or "").strip()
        if artist_prompt:
            full_prompt = f"{artist_prompt}, {full_prompt}"

        token = str(model_config.get("api_key") or "").strip()
        if token.lower().startswith("bearer "):
            token = token.split(" ", 1)[1]

        final_size = str(model_config.get("nai_size") or size or "832x1216").strip()
        try:
            width_str, height_str = final_size.split("x")
            width = int(width_str)
            height = int(height_str)
        except (ValueError, AttributeError):
            width, height = 832, 1216

        negative = str(model_config.get("negative_prompt_add") or "").strip()
        sampler = str(model_config.get("sampler") or "k_euler_ancestral").strip()
        steps = model_config.get("num_inference_steps", 28)

        cfg_value = model_config.get("nai_cfg")
        if cfg_value is not None and float(cfg_value) != 0:
            scale = float(cfg_value)
        else:
            scale = float(model_config.get("guidance_scale", 5.0) or 5.0)

        noise_schedule = str(
            model_config.get("noise_schedule") or model_config.get("nai_noise_schedule") or "karras"
        ).strip()

        model_name = model_config.get("default_model", "nai-diffusion-4-5-full")

        parameters: dict[str, Any] = {
            "width": width,
            "height": height,
            "sampler": sampler,
            "steps": steps,
            "scale": scale,
            "n_samples": 1,
            "seed": random.randint(0, 4294967295),
        }
        if negative:
            parameters["negative_prompt"] = negative
        if noise_schedule:
            parameters["noise_schedule"] = noise_schedule

        extra_params = model_config.get("nai_extra_params") or {}
        if isinstance(extra_params, dict):
            for key, value in extra_params.items():
                if value not in (None, ""):
                    parameters[str(key)] = value

        body: dict[str, Any] = {
            "input": full_prompt,
            "model": model_name,
            "parameters": parameters,
        }

        url = f"{base_url}{endpoint}"
        headers: dict[str, str] = {"Content-Type": "application/json"}
        if token:
            headers["Authorization"] = f"Bearer {token}"

        # ── 调试日志 ──
        logger.info(f"[nai_pic] ======== 生图请求 ========")
        logger.info(f"[nai_pic] URL: {url}")
        masked_token = f"{token[:8]}...{token[-4:]}" if token and len(token) > 12 else (token or "(未设置)")
        logger.info(f"[nai_pic] Token: {masked_token}")
        logger.info(f"[nai_pic] Model: {model_name}")
        logger.info(f"[nai_pic] Size: {width}x{height}")
        logger.info(f"[nai_pic] Prompt: {full_prompt[:200]}{'...' if len(full_prompt) > 200 else ''}")
        logger.info(f"[nai_pic] Negative: {negative[:200]}{'...' if len(negative) > 200 else ''}")
        logger.info(f"[nai_pic] Sampler: {sampler}, Steps: {steps}, Scale: {scale}")
        logger.info(f"[nai_pic] Request Body: {json.dumps(body, ensure_ascii=False)[:500]}")

        try:
            response = await self._client.post(url, json=body, headers=headers)
            logger.info(f"[nai_pic] 响应状态码: {response.status_code}")
            logger.info(f"[nai_pic] 响应 Content-Type: {response.headers.get('content-type', 'unknown')}")
            logger.info(f"[nai_pic] 响应体前500字符: {response.text[:500]}")
            response.raise_for_status()
        except httpx.HTTPStatusError as exc:
            logger.error(f"[nai_pic] HTTP 错误: {exc.response.status_code}, 响应体: {exc.response.text[:500]}")
            return False, f"网络请求失败:NovelAI服务暂时不可用，请稍后再试。"
        except httpx.HTTPError as exc:
            logger.error(f"[nai_pic] 网络请求异常: {type(exc).__name__}: {exc}")
            return False, f"网络请求失败:NovelAI服务暂时不可用，请稍后再试。"

        content_type = response.headers.get("content-type", "")
        if "application/json" in content_type:
            try:
                data = response.json()
            except Exception as exc:
                logger.error(f"[nai_pic] JSON 解析失败: {exc}, 原始内容: {response.text[:500]}")
                data = {}

            if "data" in data:
                inner = data["data"]
                if isinstance(inner, str) and inner.strip():
                    logger.info(f"[nai_pic] 获取到图片 (data 字符串, {len(inner)} 字符)")
                    return True, inner.strip()
                if isinstance(inner, dict):
                    for key in ("image", "url", "image_url"):
                        value = inner.get(key)
                        if isinstance(value, str) and value.strip():
                            logger.info(f"[nai_pic] 获取到图片 (data.{key}, {len(value)} 字符)")
                            return True, value.strip()
                if isinstance(inner, list) and inner:
                    first = inner[0]
                    if isinstance(first, dict):
                        for key in ("image", "url", "image_url"):
                            value = first.get(key)
                            if isinstance(value, str) and value.strip():
                                logger.info(f"[nai_pic] 获取到图片 (data[0].{key}, {len(value)} 字符)")
                                return True, value.strip()
                    elif isinstance(first, str):
                        logger.info(f"[nai_pic] 获取到图片 (data[0] 字符串, {len(first)} 字符)")
                        return True, first

            for key in ("image", "url", "image_url"):
                value = data.get(key)
                if isinstance(value, str) and value.strip():
                    logger.info(f"[nai_pic] 获取到图片 (顶层.{key}, {len(value)} 字符)")
                    return True, value.strip()

            error_msg = str(data.get("message") or data.get("error") or "未返回图片数据")
            logger.error(f"[nai_pic] 响应中未找到图片数据，完整响应: {json.dumps(data, ensure_ascii=False)[:800]}")
            return False, error_msg

        logger.info(f"[nai_pic] 非 JSON 响应，按二进制图片处理 ({len(response.content)} bytes)")
        return True, base64.b64encode(response.content).decode("utf-8")


class DanbooruClient:
    def __init__(self, timeout: int = 15) -> None:
        self._client = httpx.AsyncClient(
            timeout=timeout,
            trust_env=True,
            headers={"User-Agent": "astrbot_plugin_nai_pic/1.0"},
        )

    async def close(self) -> None:
        await self._client.aclose()

    async def _get_json(self, path: str, params: dict[str, Any]) -> Any:
        try:
            response = await self._client.get(f"{DANBOORU_API_BASE}{path}", params=params)
            response.raise_for_status()
            return response.json()
        except Exception as exc:
            logger.warning(f"[nai_pic] Danbooru 请求失败 {path}: {exc}")
            return None

    async def search_artist(self, name: str) -> dict[str, Any] | None:
        artist_name = (name or "").strip().lower()
        if not artist_name:
            return None

        payload = await self._get_json(
            "/tags.json",
            {
                "search[category]": 1,
                "search[name_matches]": artist_name,
                "search[hide_empty]": "true",
                "limit": 5,
            },
        )
        if isinstance(payload, list) and payload:
            for item in payload:
                if item.get("name", "").lower() == artist_name:
                    return item
            if payload[0].get("post_count", 0) > 0:
                return payload[0]

        payload = await self._get_json(
            "/tags.json",
            {"search[category]": 1, "search[name]": artist_name, "limit": 1},
        )
        if isinstance(payload, list) and payload:
            return payload[0]
        return None

    async def search_tag(self, tag_name: str) -> dict[str, Any] | None:
        payload = await self._get_json(
            "/tags.json",
            {"search[name]": tag_name.lower().replace(" ", "_"), "limit": 1},
        )
        if isinstance(payload, list) and payload:
            return payload[0]
        return None

    async def fuzzy_search_tag(self, partial_name: str, limit: int = 10) -> list[dict[str, Any]]:
        payload = await self._get_json(
            "/tags.json",
            {
                "search[name_matches]": f"*{partial_name.lower().replace(' ', '_')}*",
                "search[order]": "count",
                "search[hide_empty]": "true",
                "limit": min(limit, 20),
            },
        )
        return payload if isinstance(payload, list) else []

    async def fuzzy_search_artist(
        self,
        partial_name: str,
        limit: int = 10,
    ) -> list[dict[str, Any]]:
        payload = await self._get_json(
            "/tags.json",
            {
                "search[category]": 1,
                "search[name_matches]": f"*{partial_name.lower()}*",
                "search[order]": "count",
                "limit": min(limit, 50),
            },
        )
        return payload if isinstance(payload, list) else []

    async def get_related_artists(
        self,
        artist_name: str,
        limit: int = 10,
    ) -> list[dict[str, Any]]:
        payload = await self._get_json(
            "/related_tag.json",
            {"query": artist_name.lower(), "category": 1},
        )
        if isinstance(payload, dict):
            related = payload.get("related_tags") or []
            if isinstance(related, list):
                return [
                    item
                    for item in related[:limit]
                    if item.get("tag", {}).get("name") != artist_name.lower()
                ]
        if isinstance(payload, list):
            return payload[:limit]
        return []

    async def get_artist_style_tags(
        self,
        artist_name: str,
        sample_size: int = 20,
    ) -> dict[str, list[str]]:
        payload = await self._get_json(
            "/posts.json",
            {"tags": artist_name.lower(), "limit": min(sample_size, 50)},
        )
        if not isinstance(payload, list) or not payload:
            return {"common_tags": [], "common_characters": [], "common_copyrights": []}

        general_counter: Counter[str] = Counter()
        character_counter: Counter[str] = Counter()
        copyright_counter: Counter[str] = Counter()

        for post in payload:
            general_counter.update(str(post.get("tag_string_general") or "").split())
            character_counter.update(str(post.get("tag_string_character") or "").split())
            copyright_counter.update(str(post.get("tag_string_copyright") or "").split())

        trivial_tags = {
            "1girl",
            "1boy",
            "solo",
            "highres",
            "absurdres",
            "commentary_request",
            "commentary",
            "translated",
            "translation_request",
            "simple_background",
            "white_background",
        }
        return {
            "common_tags": [
                tag
                for tag, _ in general_counter.most_common(30)
                if tag not in trivial_tags
            ][:15],
            "common_characters": [tag for tag, _ in character_counter.most_common(5)],
            "common_copyrights": [tag for tag, _ in copyright_counter.most_common(5)],
        }

    async def search_artists_by_tags(
        self,
        tags: list[str],
        sample_size: int = 100,
        min_artist_count: int = 2,
    ) -> list[dict[str, Any]]:
        if not tags:
            return []

        payload = await self._get_json(
            "/posts.json",
            {"tags": " ".join(tags[:2]), "limit": min(sample_size, 200)},
        )
        if not isinstance(payload, list) or not payload:
            return []

        counter: Counter[str] = Counter()
        for post in payload:
            artist_tag = str(post.get("tag_string_artist") or "").strip()
            if not artist_tag:
                continue
            for item in artist_tag.split():
                counter[item] += 1

        filtered = [(name, count) for name, count in counter.items() if count >= min_artist_count]
        if not filtered:
            filtered = counter.most_common(30)

        results: list[dict[str, Any]] = []
        for artist_name, count in sorted(filtered, key=lambda item: -item[1])[:25]:
            artist_info = await self.search_artist(artist_name)
            if not artist_info:
                continue
            post_count = int(artist_info.get("post_count") or 0)
            if post_count < MIN_RECOMMENDED_ARTIST_POST_COUNT:
                continue
            style_info = await self.get_artist_style_tags(artist_name, sample_size=15)
            results.append(
                {
                    "name": artist_name,
                    "count": count,
                    "post_count": post_count,
                    "style_tags": style_info.get("common_tags", [])[:6],
                }
            )
        return results

    async def validate_and_correct_tags(self, tags: list[str]) -> list[str]:
        valid_tags: list[str] = []
        for tag in tags:
            exact = await self.search_tag(tag)
            if exact and int(exact.get("post_count") or 0) > 0:
                valid_tags.append(str(exact.get("name") or tag))
                continue
            fuzzy = await self.fuzzy_search_tag(tag, 3)
            if fuzzy:
                valid_tags.append(str(fuzzy[0].get("name") or tag))
                continue
        return valid_tags


def extract_artist_names_from_prompt(artist_prompt: str) -> list[str]:
    matches = set(re.findall(r"artist:([a-zA-Z0-9_\-\(\)]+)", (artist_prompt or "").lower()))
    return list(matches)


def get_artist_quality_score(artist_info: dict[str, Any]) -> str:
    post_count = int(artist_info.get("post_count") or 0)
    if post_count >= 5000:
        return "S"
    if post_count >= 2000:
        return "A"
    if post_count >= 500:
        return "B"
    if post_count >= 100:
        return "C"
    return "D"
