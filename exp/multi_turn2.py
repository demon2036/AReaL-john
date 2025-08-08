import json
import uuid
from contextlib import AsyncExitStack

import torch
from mcp import ClientSession
from mcp.client.sse import sse_client
from mcp.client.streamable_http import streamablehttp_client
from openai.types.chat import ChatCompletionMessageToolCall
from tensordict import TensorDict
from transformers import PreTrainedTokenizerFast

from areal.api.cli_args import GenerationHyperparameters
from areal.api.engine_api import InferenceEngine
from areal.api.io_struct import LLMRequest
from areal.api.workflow_api import RolloutWorkflow
from areal.utils.data import concat_padded_tensors
from exp.simple_agent_prompt import get_system_prompt
from exp.tool_manager import ToolManager
from exp.utils import parse_tool_calls,_check_task_done,reward_fn



class MultiTurnMCPWorkflow(RolloutWorkflow):
    def __init__(
            self,
            reward_fn,
            gconfig: GenerationHyperparameters,
            tokenizer: PreTrainedTokenizerFast,
            max_turns: int,
            turn_discount: float,
            enable_thinking: bool,
            dump_dir: str | None = None,
            servers  =  None
    ):
        self.reward_fn = reward_fn
        self.gconfig = gconfig
        self.tokenizer = tokenizer
        self.max_turns = max_turns
        self.turn_discount = turn_discount
        self.enable_thinking = True
        self.dump_dir = dump_dir
        self.tool_manager = ToolManager()
        self.servers=servers
        self.system_prompt=get_system_prompt()


    async def _setup_internal_tools(self, tool_manager: ToolManager):
        """注册Agent特定的工具"""
        tool_manager.register_internal_tool(
            name="task_done",
            description="Mark the current task as completed",
            parameters={
                "type": "object",
                "properties": {
                    "success": {"type": "boolean"},
                    "task_completion_description": {"type": "string"},
                    "attachments_descriptions": {"type": "string"}
                },
                "required": ["success", "task_completion_description"]
            },
            handler=self._handle_task_done
        )

    async def _handle_task_done(self, success: bool,
                                task_completion_description: str,
                                attachments_descriptions: str = "") -> str:
        """处理任务完成"""
        self.task_completed = True
        self.task_result = {
            "success": success,
            "task_completion_description": task_completion_description,
            "attachments_descriptions": attachments_descriptions
        }
        return "Task marked as completed"


    async def setup(self, stack: AsyncExitStack,tool_manager) -> None:
        """Initialize tool connections"""
        for i, server_url in enumerate(self.servers):
            client = sse_client if '/sse' in server_url else streamablehttp_client
            *connection_args, = await stack.enter_async_context(client(url=server_url))
            read, write = connection_args[:2]

            session = await stack.enter_async_context(ClientSession(read, write))
            await session.initialize()

            tools_response = await session.list_tools()
            # print(tools_response)
            tool_manager.register_external_tools(
                [tool.model_dump() for tool in tools_response.tools],
                server_index=i
            )
            tool_manager.sessions.append(session)


    async def _execute_tool_calls(self, tool_calls,tool_manager):
        """Execute tool calls and format results"""
        results = []
        for tool_call in tool_calls:
            try:
                # tool_call['function']['arguments']=json.dumps(tool_call['function']['arguments'],ensure_ascii=False)
                result = await tool_manager.execute_tool_call(ChatCompletionMessageToolCall(**tool_call))
                results.append({
                    "tool_call_id": tool_call["id"],
                    "role": "tool",
                    "name": tool_call["function"]["name"],
                    "content": result
                })
            except Exception as e:
                print(e)
                results.append({
                    "tool_call_id": tool_call["id"],
                    "role": "tool",
                    "name": tool_call["function"]["name"],
                    "content": f"Error: {str(e)}"
                })

        return results

    async def _arun_episode(self, engine: InferenceEngine, data):
        seq, logprobs, loss_mask, versions = [], [], [], []
        messages = [{'role': 'system', 'content': self.system_prompt}] + data["messages"]
        rid = uuid.uuid4().hex
        tool_manager = ToolManager()

        async with AsyncExitStack() as stack:
            await self.setup(stack, tool_manager)
            await self._setup_internal_tools(tool_manager)

            # 关键技巧：计算系统提示的长度
            # 通过传入空消息列表，apply_chat_template只会返回系统提示部分的tokens
            # 这让我们知道后续处理工具响应时需要切掉多少重复部分
            system_prompt_tokens = self.tokenizer.apply_chat_template(
                [{}],  # 空消息，只生成系统提示
                add_generation_prompt=False,
                tokenize=True
            )
            system_prompt_length = len(system_prompt_tokens)

            # 初始化：对完整消息列表进行一次性编码
            # 这是整个流程中唯一一次对完整messages进行编码
            # 后续都是通过累积tokens来避免重新编码
            accumulated_tokens = self.tokenizer.apply_chat_template(
                messages,
                tools=tool_manager.get_tools_schema(),
                tokenize=True,
                add_generation_prompt=True,
                # enable_thinking=self.enable_thinking,
            )

            # 保存最终的消息历史，用于验证和返回
            final_messages = list(messages)

            for turn_num in range(self.max_turns):

                # self.gconfig.stop_token_ids=None
                # print(f'{self.gconfig.stop=}',f'{self.gconfig.stop_token_ids=}')

                # Generate response
                # 使用累积的tokens而不是重新编码messages
                # 这是避免encode/decode不一致的核心
                req = LLMRequest(
                    rid=rid,
                    input_ids=accumulated_tokens,  # 使用累积的tokens
                    gconfig=self.gconfig.new(n_samples=1),
                )
                resp = await engine.agenerate(req)

                completions_str = self.tokenizer.decode(resp.output_tokens)
                completions_str = completions_str.replace(self.tokenizer.eos_token, "")

                # Update sequence data
                # 计算新增的输入长度（处理输入重叠）
                input_len = len(resp.input_tokens) - len(seq)
                seq += resp.input_tokens[-input_len:] + resp.output_tokens
                logprobs += [0.0] * input_len + resp.output_logprobs
                loss_mask += [0] * input_len + [1] * resp.output_len
                versions += [-1] * input_len + resp.output_versions

                # 核心操作：直接累积响应tokens
                # 不decode再encode，保持原始tokens不变
                if len(accumulated_tokens)+len(resp.output_tokens)>16384:
                    break
                accumulated_tokens += resp.output_tokens

                # Check for tool calls
                content,tool_calls = parse_tool_calls(completions_str)

                final_messages.append({
                    "role": "assistant",
                    "content": completions_str,
                    # "tool_calls": to
                })


                if not tool_calls or _check_task_done(tool_calls):
                    # No tool calls, end conversation
                    messages = final_messages
                    break

                # Execute tools and continue conversation
                tool_results = await self._execute_tool_calls(tool_calls, tool_manager)

                # 更新消息历史
                final_messages.extend(tool_results)

                # 处理工具响应的关键步骤
                if tool_results:
                    temp_messages_for_tools =  tool_results

                    # 单独编码工具响应
                    # apply_chat_template会为工具响应添加完整格式（包括系统提示）
                    tool_response_full = self.tokenizer.apply_chat_template(
                        temp_messages_for_tools,
                        tools=tool_manager.get_tools_schema(),
                        add_generation_prompt=True,  # 为下一轮生成做准备
                        tokenize=True
                    )

                    # 智能切片：去掉重复的系统提示
                    # 因为accumulated_tokens已经包含了系统提示，
                    # 我们只需要工具响应的实际内容部分
                    tool_response_tokens = tool_response_full[system_prompt_length:]

                    if len(accumulated_tokens) + len(tool_response_tokens) > 16384:
                        break

                    # 累积工具响应tokens
                    accumulated_tokens += tool_response_tokens

                    # 更新序列追踪数据
                    # 工具响应作为输入的一部分，loss_mask设为0
                    seq += tool_response_tokens
                    logprobs += [0.0] * len(tool_response_tokens)
                    loss_mask += [0] * len(tool_response_tokens)
                    versions += [-1] * len(tool_response_tokens)
            print('**' * 5)
            # Final reward calculation
            prompt_str = self.tokenizer.decode(accumulated_tokens)


        del tool_manager

        reward=await reward_fn(data["messages"][0]['content'],messages)
        print(f'{reward=}')
        print('**' * 5)

        res = dict(
            input_ids=torch.tensor(accumulated_tokens),
            logprobs=torch.tensor(logprobs),
            loss_mask=torch.tensor(loss_mask),
            versions=torch.tensor(versions),
            rewards=torch.tensor(float(reward)),
            attention_mask=torch.ones(len(accumulated_tokens), dtype=torch.bool),
        )
        res = {k: v.unsqueeze(0) for k, v in res.items()}

        return concat_padded_tensors([TensorDict(res, batch_size=[1])])

    async def arun_episode(self, engine: InferenceEngine, data):
            return await self._arun_episode(engine, data,)