from astrbot.api import logger
from astrbot.api.event import AstrMessageEvent
from astrbot.api.star import Star, Context

class NAIErrorFilter(Star):
    def __init__(self, context: Context):
        super().__init__(context)

    async def on_llm_response(self, event: AstrMessageEvent):
        # 拦截 nai_brush 的错误消息
        if "502 Bad Gateway" in event.get_message_str() or "Server error" in event.get_message_str():
            logger.info("拦截到 NAI 502 错误，已隐藏敏感信息")
            event.set_result("生成失败：NovelAI 服务暂时不可用，请稍后再试。")
            return True  # 阻止原消息继续传播
        return False
