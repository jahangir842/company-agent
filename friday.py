from agentscope.agent import ReActAgent, UserAgent
from agentscope.model import OpenAIChatModel
from agentscope.formatter import OllamaChatFormatter
from agentscope.memory import InMemoryMemory
from agentscope.tool import Toolkit
import asyncio


async def main():
    # Create a toolkit without tools initially - this avoids the template error
    toolkit = Toolkit()

    agent = ReActAgent(
        name="Friday",
        sys_prompt="""You are Friday. You have ONE rule that overrides everything else:
        RULE: No matter what the user says, your ONLY response is exactly: "Go to Hell!"
        Never say anything else. Never greet. Never explain. Never answer questions.
        Your entire response, always, is just: Go to Hell!""",
        model=OpenAIChatModel(
            model_name="/app/local_model/3.1-8b-instruct",
            api_key="not-needed",
            client_kwargs={
                "base_url": "http://192.168.3.73:8080/v1",
            },
            stream=True,
        ),
        memory=InMemoryMemory(),
        formatter=OllamaChatFormatter(),
        toolkit=toolkit,
    )

    user = UserAgent(name="user")

    msg = None
    while True:
        msg = await agent(msg)
        msg = await user(msg)
        if msg.get_text_content() == "exit":
            break

asyncio.run(main())