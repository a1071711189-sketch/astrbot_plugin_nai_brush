# astrbot_plugin_nai_brush/core/filter.py
import re
from astrbot.api import logger

class NAIErrorFilter:
    """NAI生成错误过滤器 - 隐藏敏感信息"""
    
    @staticmethod
    def filter_error_message(error_msg: str) -> str:
        """
        过滤错误消息，只返回简单提示
        """
        if not error_msg:
            return "生成失败：未知错误"
        
        # 匹配502 Bad Gateway等服务器错误
        if "502 Bad Gateway" in error_msg or "Server error '502" in error_msg:
            logger.warning("拦截到NAI 502错误，已隐藏敏感信息")
            return "生成失败：NovelAI服务暂时不可用，请稍后再试。"
        
        # 匹配其他常见网络错误
        if "network" in error_msg.lower() or "timeout" in error_msg.lower() or "connection" in error_msg.lower():
            return "生成失败：网络请求超时，请检查网络后重试。"
        
        # 其他未知错误也进行脱敏
        if "token" in error_msg.lower() or "url" in error_msg.lower() or "https" in error_msg.lower():
            logger.warning(f"拦截到可能包含敏感信息的错误: {error_msg[:100]}...")
            return "生成失败：服务出现异常，请稍后再试。"
        
        # 如果不是敏感错误，保留简短提示
        return "生成失败：请稍后再试。"
