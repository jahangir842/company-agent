import asyncio
from flask import Flask, request, jsonify
from agentscope.model import OpenAIChatModel
from agentscope.agent import ReActAgent  
from agentscope.memory import InMemoryMemory
from agentscope.tool import Toolkit
from agentscope.message import Msg

app = Flask(__name__)

# 1. Define a simple Tool function
def calculate_monthly_salary(annual_salary: int) -> str:
    """
    Calculate the monthly take-home pay from an annual salary.
    Args:
        annual_salary (int): The yearly salary amount.
    """
    monthly = annual_salary / 12
    return f"The monthly salary is ${monthly:.2f}"

# 2. Setup the Toolkit
hiring_tools = Toolkit()
hiring_tools.add(calculate_monthly_salary)

# 3. Setup the Model
model = OpenAIChatModel(
    model_name="/app/local_model/3.1-8b-instruct",
    api_key="key",
    client_kwargs={"base_url": "http://192.168.3.73:8080/v1"},
)

async def analyze(query):
    # We define the agent inside so it gets a fresh memory per request
    agent = ReActAgent(
        name="HiringAgent",
        sys_prompt="You are a hiring assistant. Use the provided tools for calculations.",
        model=model,
        memory=InMemoryMemory(),
        toolkit=hiring_tools, # Tools are now active!
    )
    
    msg = Msg(name="user", content=query, role="user")
    response = await agent(msg)
    return response.get_text_content()

@app.route('/api/analyze', methods=['POST'])
def api_analyze():
    query = request.json.get('query', '')
    result = asyncio.run(analyze(query))
    return jsonify({'response': result})

if __name__ == '__main__':
    app.run(port=5000)