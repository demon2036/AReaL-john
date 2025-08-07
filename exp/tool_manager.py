# tool_manager.py
from typing import List, Dict, Any, Optional, Callable
from dataclasses import dataclass
import json

from mcp import ClientSession


@dataclass
class Tool:
    """工具的统一表示"""
    name: str
    original_name: str
    description: str
    parameters: dict
    server_index: Optional[int] = None
    is_internal: bool = False
    handler: Optional[Callable] = None


class ToolRegistry:
    """工具注册表"""

    def __init__(self):
        self._tools: Dict[str, Tool] = {}

    def register_internal(self, name: str, description: str,
                          parameters: dict, handler: Callable) -> None:
        """注册内部工具"""
        tool = Tool(
            # name=f"internal_{name}",
            name=f"{name}",
            original_name=name,
            description=f"[Internal] {description}",
            parameters=parameters,
            is_internal=True,
            handler=handler
        )
        self._tools[tool.name] = tool

    def register_external(self, tools: List[Dict], server_index: int) -> None:
        """批量注册外部工具"""
        for tool_data in tools:
            tool = Tool(
                name=f"server_{server_index}_{tool_data['name']}",
                original_name=tool_data['name'],
                description=f"[Server {server_index}] {tool_data['description']}",
                parameters=tool_data['inputSchema'],
                server_index=server_index,
                is_internal=False
            )
            self._tools[tool.name] = tool

    def get_all_schemas(self) -> List[Dict]:
        """获取所有工具的OpenAI格式schema"""
        return [{
            "type": "function",
            "function": {
                "name": tool.name,
                "description": tool.description,
                "parameters": tool.parameters
            }
        } for tool in self._tools.values()]

    def find(self, tool_name: str) -> Optional[Tool]:
        """查找工具"""
        return self._tools.get(tool_name)


class ToolExecutor:
    """工具执行器"""

    async def execute(self, tool: Tool, arguments: Dict[str, Any],
                      sessions: Optional[List[ClientSession]] = None) -> str:
        """执行工具并返回字符串结果"""
        if tool.is_internal:
            result = await tool.handler(**arguments)
            return str(result)
        else:
            if sessions is None or tool.server_index >= len(sessions):
                raise ValueError(f"No session for server {tool.server_index}")

            session = sessions[tool.server_index]
            response = await session.call_tool(tool.original_name, arguments)

            # print(response)
            # print(tool)
            # print(await session.list_tools(),sessions)
            # while True:
            #     pass

            # 统一处理各种返回格式
            return self._extract_text_content(response)

    def _extract_text_content(self, response: Any) -> str:
        """从各种响应格式中提取文本内容"""
        # 如果是字符串，直接返回
        if isinstance(response, str):
            return response

        # 如果有content属性
        if hasattr(response, 'content'):
            content = response.content

            # 如果content是列表（如[TextContent(...)]）
            if isinstance(content, list):
                for item in content:
                    # 优先使用text属性
                    if hasattr(item, 'text'):
                        return item.text
                    # 如果有type和text的字典
                    elif isinstance(item, dict) and 'text' in item:
                        return item['text']

                # 如果没找到text，返回第一个元素的字符串形式
                if content:
                    return str(content[0])

            # 如果content不是列表，转字符串
            return str(content)

        # 其他情况，直接转字符串
        return str(response)


class ToolManager:
    """工具管理器 - 支持可选的引用处理"""

    def __init__(self, reference_handler=None):
        self.registry = ToolRegistry()
        self.executor = ToolExecutor()
        self.sessions: List[ClientSession] = []

        # 可选的引用处理器
        self.reference_handler = reference_handler

    def register_internal_tool(self, name: str, description: str,
                               parameters: dict, handler: Callable) -> None:
        """注册内部工具"""
        self.registry.register_internal(name, description, parameters, handler)

    def register_external_tools(self, tools: List[Dict], server_index: int) -> None:
        """注册外部工具"""
        self.registry.register_external(tools, server_index)

    def get_tools_schema(self) -> List[Dict]:
        """获取所有工具的schema"""
        return self.registry.get_all_schemas()

    async def execute_tool_call(self, tool_call) -> str:
        """执行单个工具调用"""


        tool_name = tool_call.function.name
        tool_args = json.loads(tool_call.function.arguments)

        # 如果有引用处理器，处理输入
        # if self.reference_handler:
        #     tool_args = self.reference_handler.process_tool_input(tool_name, tool_args)

        tool = self.registry.find(tool_name)
        if tool is None:
            raise ValueError(f"Unknown tool: {tool_name}")

        # 执行工具
        result = await self.executor.execute(tool, tool_args, self.sessions)

        # 如果有引用处理器，处理输出
        # if self.reference_handler:
        #     result = self.reference_handler.process_tool_output(tool_name, result)

        return result