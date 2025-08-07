import json
import re
import uuid

import httpx
from openai import AsyncOpenAI
from tenacity import stop_after_attempt, wait_exponential, retry


def parse_tool_calls(text: str):
    """Parse tool calls from text and convert to OpenAI format, return both content and tool_calls"""

    # 方法1: 修改正则表达式，包含前面的换行符
    tool_regex = re.compile(r"\n<tool_call>(.*?)</tool_call>", re.DOTALL)
    matches = tool_regex.findall(text)

    tool_calls = []
    for i, match in enumerate(matches):
        try:
            tool_data = json.loads(match.strip())
            tool_calls.append({
                "id": f"call_{uuid.uuid4().hex[:24]}",
                "type": "function",
                "function": {
                    "name": tool_data.get("name", ""),
                    "arguments": tool_data.get("arguments", {}),
                }
            })
        except Exception as e:
            print(e)
            continue

    # Remove tool call tokens from text to get remaining content
    content = tool_regex.sub("", text).strip()

    return content, tool_calls


def _check_task_done( tool_calls):
    """Check if task_done is in tool calls"""
    return any(tc["function"]["name"] == "task_done" for tc in tool_calls)


client = AsyncOpenAI(api_key='1',
timeout=httpx.Timeout(180.0),
                base_url='https://ms-shpc7pdz-100034032793-sw.gw.ap-shanghai.ti.tencentcs.com/ms-shpc7pdz/v1')


@retry(stop=stop_after_attempt(10), wait=wait_exponential(min=1, max=10))
async def send_message(prompt):
    global client
    response = await client.chat.completions.create(
        model="ms-shpc7pdz",
        messages=[{"role": "user", "content": prompt}],
        temperature=0.6,
        response_format={"type": "json_object"}
    )
    return response


async def reward_fn(prompt, messages):
    # Loop through messages from back to front
    for i in range(len(messages) - 1, -1, -1):
        msg = messages[i]
        # Check if it's an assistant message
        if msg.get('role') == 'assistant' and msg.get('content'):
            # Parse tool calls from the content
            tool_calls = parse_tool_calls(msg['content'])

            # If there are tool calls and they're not task_done
            if tool_calls and not _check_task_done(tool_calls):
                if i + 1 < len(messages):
                    next_msg = messages[i + 1]
                else:
                    print("\nNo next message found!")
                    continue

                # Check if download_info exists in the result
                result_content = next_msg.get('content', '')
                if "'download_info'" not in result_content:
                    print("\nDownload failed: 'download_info' key not found in result")
                    # print(messages)
                    return 0.0

                evaluation_prompt = f"""Given this user prompt and assistant's tool call, evaluate if the tool usage is appropriate.

                User Prompt: {prompt}

                Result: {next_msg.get('content')}

                Carefully analyze:
                1. Does the result actually fulfill what the user requested?
                2. For download requests: Is this the actual downloadable file or just a webpage about it?
                3. Check file metadata consistency:
                   - Is the filename appropriate for what was requested?
                   - Does the file size make sense? (e.g., browsers/apps are typically MB/GB, not KB)
                   - Is the content type correct? (executables/installers vs HTML pages)
                   - For downloads: Did it fetch the actual file or just webpage metadata?

                Output a JSON with:
                - "appropriate": boolean (true ONLY if the tool call successfully addresses the user's actual need)
                - "reason": string (specific explanation noting any mismatches or issues found)

                Only output the JSON, nothing else."""

                response = await send_message(evaluation_prompt)

                response = re.sub(r'<think>.*?</think>', '', response.choices[0].message.content,
                                  flags=re.DOTALL).strip()

                evaluation = json.loads(response)
                print(f"\nEvaluation: {evaluation_prompt}")
                print(f"\nEvaluation: {evaluation}")

                if evaluation['appropriate']:
                    return 1.0
                else:
                    return 0.0

    return 0.0

