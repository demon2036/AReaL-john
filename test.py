from transformers import AutoTokenizer
import json

# 加载Qwen2.5-7B的tokenizer
model_name = "Qwen/Qwen2.5-7B-Instruct"
tokenizer = AutoTokenizer.from_pretrained(model_name)

# 定义天气预报tool
weather_tool = {
    "type": "function",
    "function": {
        "name": "get_weather_forecast",
        "description": "获取指定城市的天气预报信息",
        "parameters": {
            "type": "object",
            "properties": {
                "city": {
                    "type": "string",
                    "description": "城市名称，例如：北京、上海、深圳"
                },
                "days": {
                    "type": "integer",
                    "description": "预报天数，1-7天",
                    "minimum": 1,
                    "maximum": 7
                }
            },
            "required": ["city"]
        }
    }
}

# 创建对话消息
messages = [
    {
        "role": "tool",
        "tool_call_id": "call_weather_001",
        "name": "get_weather_forecast",
        "content": json.dumps({
            "city": "北京",
            "date": "2025-08-07",
            "weather": "晴",
            "temperature": {
                "high": 32,
                "low": 22
            },
            "humidity": "45%",
            "wind": "西南风2-3级"
        }, ensure_ascii=False)
    },

]




# 应用chat template并编码
# 编码消息
tool_responses = tokenizer.apply_chat_template(
    messages,
    # tools=[weather_tool],
    add_generation_prompt=True,
    tokenize=True,
    return_tensors="pt"
)

print(f"编码后的token数量: {tool_responses.shape[1]}")
print(f"编码结果形状: {tool_responses.shape}")
print()

# 打印部分token
print("=== 编码后的tokens (前50个) ===")
print(tool_responses[0].tolist())
print()

# 解码回文本查看效果
print("=== 解码后的文本 ===")
decoded_text = tokenizer.decode(tool_responses[0], skip_special_tokens=False)
print(decoded_text)
print()




print("=== 解码后的文本 ===")
decoded_text = tokenizer.apply_chat_template([{}], add_generation_prompt=False, tokenize=False,skip_special_tokens=False)
print(decoded_text)
print()

# # 不带特殊token的解码
# print("=== 解码后的文本 (无特殊token) ===")
# decoded_clean = tokenizer.decode(tool_responses[0], skip_special_tokens=True)
# print(decoded_clean)

