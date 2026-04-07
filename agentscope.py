import torch
import asyncio
from transformers import pipeline
from agentscope.agent import ReActAgent
from agentscope.message import Msg
from agentscope.model import ChatResponse 
from agentscope.formatter import OllamaChatFormatter

pipe = pipeline("text-generation", model="unsloth/Llama-3.2-1B-Instruct", 
                dtype=torch.bfloat16, device_map="auto")

async def local_llama(messages, **kwargs):
    prompt = [
        {"role": m['role'] if isinstance(m, dict) else m.role, 
         "content": m['content'] if isinstance(m, dict) else m.content} 
        for m in messages
    ]
    output = pipe(prompt, max_new_tokens=256)[0]['generated_text'][-1]['content']
    return ChatResponse(content=[{"type": "text", "text": output}])

local_llama.format = lambda msgs: msgs
local_llama.stream = False          
local_llama.config_name = "local"   

agent = ReActAgent(name="Assistant", sys_prompt="You are a helpful, concise AI.", model=local_llama, formatter=OllamaChatFormatter())

print("\n🤖 Assistant ready! \n" + "─"*20)
while True:
    query = input("\nYou: ")
    response = asyncio.run(agent(Msg(name="user", role="user", content=query)))
    print(f"\nAssistant: {response.content}")