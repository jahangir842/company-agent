import asyncio
import csv
from flask import Flask, render_template, request, jsonify
from agentscope.model import OpenAIChatModel
from agentscope.formatter import OllamaChatFormatter
from agentscope.agent import ReActAgent  
from agentscope.memory import InMemoryMemory
from agentscope.message import Msg

app = Flask(__name__, template_folder='templates')

# Read candidates from CSV
def load_candidates():
    candidates = []
    with open('candidates.csv', 'r') as f:
        reader = csv.DictReader(f)
        for row in reader:
            candidates.append(row)
    return candidates

# Company data
COMPANY_DATA = """
HIRING: 7 open positions, 45 candidates, 12 interviews scheduled
PROJECTS: Mobile Redesign, API Migration, Security Audit
METRICS: 156 employees, $2.5M revenue, 94.2% retention
"""

CANDIDATES_DATA = "\n".join([f"- {c['name']}: {c['position']}, {c['experience']} exp, Skills: {c['skills']}, Status: {c['status']}, Salary: {c['salary_expectation']}" for c in load_candidates()])

async def analyze(query):
    """Get AI analysis"""
    full_context = f"""Company Data:
{COMPANY_DATA}

Candidate Pool:
{CANDIDATES_DATA}"""
    
    agent = ReActAgent(
        name="HiringAgent",
        sys_prompt=f"You are a hiring expert. Use this data to answer questions:\n{full_context}\nAnswer concisely with specific names and details.",
        model=OpenAIChatModel(
            model_name="/app/local_model/3.1-8b-instruct",
            api_key="key",
            client_kwargs={"base_url": "http://192.168.3.73:8080/v1"},
            stream=False,
        ),
        memory=InMemoryMemory(),
        formatter=OllamaChatFormatter(),
    )
    msg = Msg(name="user", role="user", content=query)
    return (await agent(msg)).get_text_content()


@app.route('/')
def index():
    return render_template('hiring.html')

@app.route('/api/analyze', methods=['POST'])
def api_analyze():
    try:
        query = request.json.get('query', '')
        result = asyncio.run(analyze(query))
        return jsonify({'response': result})
    except Exception as e:
        return jsonify({'error': str(e)})

if __name__ == '__main__':
    print("\n╔════════════════════════════════════════════╗")
    print("║  AgentScope Company Agent                 ║")
    print("║                                            ║")
    print("║  Open: http://localhost:5000              ║")
    print("╚════════════════════════════════════════════╝\n")
    app.run(debug=False, port=5000)
