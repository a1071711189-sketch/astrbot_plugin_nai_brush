"""Async HTTP clients for NAI and Danbooru."""

from __future__ import annotations

import asyncio
import base64
from collections import Counter
import io
import json
import random
import re
from typing import Any
import zipfile

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
            timeout=httpx.Timeout(connect=30.0, read=180.0, write=30.0, pool=30.0),
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

        body: dict[str, Any] = {
            "prompt": full_prompt,
            "model": model_name,
            "width": width,
            "height": height,
            "sampler": sampler,
            "steps": steps,
            "scale": scale,
            "n_samples": 1,
            "seed": random.randint(0, 4294967295),
        }
        if negative:
            body["negative_prompt"] = negative
        if noise_schedule:
            body["noise_schedule"] = noise_schedule

        extra_params = model_config.get("nai_extra_params") or {}
        if isinstance(extra_params, dict):
            for key, value in extra_params.items():
                if value not in (None, ""):
                    body[str(key)] = value

        url = f"{base_url}{endpoint}"
        headers: dict[str, str] = {"Content-Type": "application/json"}
        if token:
            headers["Authorization"] = f"Bearer {token}"

        proxy = str(model_config.get("proxy") or "").strip()

        # ── 调试日志 ──
        logger.info(f"[nai_pic] ======== 生图请求 ========")
        logger.info(f"[nai_pic] 完整请求 URL: {url}")
        logger.info(f"[nai_pic] 请求方法: POST")
        logger.info(f"[nai_pic] 代理: {proxy or '(未设置)'}")
        masked_token = f"{token[:8]}...{token[-4:]}" if token and len(token) > 12 else (token or "(未设置)")
        logger.info(f"[nai_pic] Token: {masked_token}")
        logger.info(f"[nai_pic] Model: {model_name}")
        logger.info(f"[nai_pic] Size: {width}x{height}")
        logger.info(f"[nai_pic] Prompt: {full_prompt}")
        logger.info(f"[nai_pic] Negative: {negative}")
        logger.info(f"[nai_pic] Sampler: {sampler}, Steps: {steps}, Scale: {scale}")
        logger.info(f"[nai_pic] 完整 Request Body ({len(json.dumps(body, ensure_ascii=False))} 字符): {json.dumps(body, ensure_ascii=False)}")

        client = self._client
        if proxy:
            client = httpx.AsyncClient(
                proxy=proxy,
                timeout=httpx.Timeout(connect=30.0, read=180.0, write=30.0, pool=30.0),
                verify=False,
                trust_env=True,
                follow_redirects=True,
            )

        try:
            try:
                response = await client.post(url, json=body, headers=headers)
                logger.info(f"[nai_pic] 响应状态码: {response.status_code}")
                logger.info(f"[nai_pic] 响应 Content-Type: {response.headers.get('content-type', 'unknown')}")
                response.raise_for_status()
            except httpx.ConnectTimeout:
                logger.error(f"[nai_pic] 连接超时 (无法建立到 image.novelai.net 的 TCP 连接，可能需要代理)")
                return False, "无法连接到 NovelAI 服务器，请检查网络或使用代理。"
            except httpx.ReadTimeout:
                logger.error(f"[nai_pic] 读取超时 (服务端生成图片时间过长)")
                return False, "NovelAI 生成超时，请稍后重试。"
            except httpx.HTTPStatusError as exc:
                logger.error(f"[nai_pic] HTTP 错误: {exc.response.status_code}, 完整响应: {exc.response.text[:1000]}")
                return False, "NovelAI服务暂时不可用，请稍后再试。"
            except httpx.HTTPError as exc:
                logger.error(f"[nai_pic] 网络请求异常: {type(exc).__name__}: {exc}")
                return False, "NovelAI服务暂时不可用，请稍后再试。"

            content_type = response.headers.get("content-type", "").lower()

            if "application/json" in content_type:
                try:
                    data = response.json()
                except Exception as exc:
                    logger.error(f"[nai_pic] JSON 解析失败: {exc}, 原始内容: {response.text}")
                    return False, "无法解析服务器响应"
                error_msg = str(data.get("message") or data.get("error") or "未知错误")
                logger.error(f"[nai_pic] 服务器返回错误: {error_msg}, 完整: {json.dumps(data, ensure_ascii=False)[:800]}")
                return False, error_msg

            # 官方 API 返回 ZIP（含 PNG）或纯二进制图片
            if "zip" in content_type or "octet-stream" in content_type:
                logger.info(f"[nai_pic] 收到 ZIP 响应 ({len(response.content)} bytes)，提取图片...")
                try:
                    with zipfile.ZipFile(io.BytesIO(response.content)) as zf:
                        names = zf.namelist()
                        logger.info(f"[nai_pic] ZIP 内文件: {names}")
                        for name in names:
                            if name.lower().endswith(".png"):
                                img_bytes = zf.read(name)
                                logger.info(f"[nai_pic] 提取 {name} ({len(img_bytes)} bytes)")
                                return True, base64.b64encode(img_bytes).decode("utf-8")
                        if names:
                            img_bytes = zf.read(names[0])
                            logger.info(f"[nai_pic] 提取 {names[0]} ({len(img_bytes)} bytes)")
                            return True, base64.b64encode(img_bytes).decode("utf-8")
                        return False, "ZIP 中没有图片文件"
                except Exception as exc:
                    logger.error(f"[nai_pic] ZIP 解压失败: {exc}")
                    return False, "图片数据解压失败"

            # 其他二进制格式（如 image/png 等），直接当图片
            logger.info(f"[nai_pic] 收到二进制图片 ({len(response.content)} bytes)")
            return True, base64.b64encode(response.content).decode("utf-8")
        finally:
            if proxy:
                await client.aclose()


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
