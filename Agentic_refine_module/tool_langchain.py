
import os
os.environ["DEEPSEEK_API_KEY"] = "your_api_key"
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_deepseek import ChatDeepSeek
from langchain.agents.format_scratchpad.openai_tools import format_to_openai_tool_messages
from langchain.agents.output_parsers.openai_tools import OpenAIToolsAgentOutputParser
from langchain.agents import AgentExecutor
from tools import astar_path_generate, select_ref_hyperparams, save_scene
from agent_prompt_template import system_role_template, input_prompt_template
import json

llm = ChatDeepSeek(
    model="deepseek-chat",
    temperature=1,
    max_tokens=None,
    timeout=None,
    max_retries=2,
)

tools = [astar_path_generate, select_ref_hyperparams,save_scene]
llm_with_tools = llm.bind_tools(tools)

prompt = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            system_role_template
            
        ),
        ("user", "{input}"),
        MessagesPlaceholder(variable_name="agent_scratchpad"),
    ]
)

def generate_path(scene, llm_guide):
    agent = (
        {
            "input": lambda x: x["input"],
            "agent_scratchpad": lambda x: format_to_openai_tool_messages( x["intermediate_steps"] ),
        }
        | prompt
        | llm_with_tools
        | OpenAIToolsAgentOutputParser()
    )
    agent_executor = AgentExecutor(agent=agent, tools=tools, verbose=True, return_intermediate_steps=True)
    result = agent_executor.invoke({"input": input_prompt_template.format(scene=str(scene), llm_guide=str(llm_guide))})
    return result
    
if __name__ == "__main__":
    example_scene = [['streetbarrier',  9.7, -0.5], ['trafficcone',  13.7, -15.5], ['vehicle', 17.1, 10.5], ['vehicle', 54.7, 0.0]]
    llm_guide = ["right", "keep", "left", "right"]
    result = generate_path(scene=example_scene, llm_guide = llm_guide)
    # with open("result_langchain.json", "w") as f:
    #     json.dump(result, f, indent=4)
    print(result["output"])
    # final_index = int(result["output"].split("Answer:")[-1].strip())
    # with open(f"temp/{final_index}.txt", "r") as f:
    #     print(f.read())