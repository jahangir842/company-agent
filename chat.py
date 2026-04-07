import torch
from transformers import pipeline

pipe = pipeline("text-generation", model="unsloth/Llama-3.2-1B-Instruct", 
                dtype=torch.bfloat16, device_map="auto")

messages = [{"role": "system", "content": "You are a helpful, concise AI."}]

print("\n🤖 Assistant ready! \n" + "─"*20)

while True:
    user_input = input("\nYou: ")
    messages.append({"role": "user", "content": user_input})
    
    # Generate response
    output = pipe(messages, max_new_tokens=256)[0]['generated_text']
    assistant_text = output[-1]['content']
    
    print(f"\nAssistant: {assistant_text}")
    
    # Keep history for context
    messages.append({"role": "assistant", "content": assistant_text})
