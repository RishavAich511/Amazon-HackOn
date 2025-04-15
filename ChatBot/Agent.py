import sys
import os

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import asyncio
from langchain_mistralai import ChatMistralAI
from langchain import hub
from langchain.agents import create_tool_calling_agent
from langchain.agents import AgentExecutor
from langchain import PromptTemplate
from env import *
from tools import tool
from langchain_community.chat_message_histories import ChatMessageHistory
from langchain_core.chat_history import BaseChatMessageHistory
from langchain_core.runnables.history import RunnableWithMessageHistory

store = {}
# print(os.getenv("MISTRAL_API_KEY"))

# Initialize language model
llm = ChatMistralAI(model="mistral-large-latest", streaming=True)

# Bind the tool to the model
llm = llm.bind_tools(tool)

def get_session_history(session_id: str) -> BaseChatMessageHistory:
    if session_id not in store:
        store[session_id] = ChatMessageHistory()
    return store[session_id]

# Define agent
agent = create_tool_calling_agent(llm, tool, hub.pull("hwchase17/openai-functions-agent"))
agent_executor = AgentExecutor(agent=agent, tools=tool)

agent_with_chat_history = RunnableWithMessageHistory(
    agent_executor,
    get_session_history,
    input_messages_key="input",
    history_messages_key="chat_history",
)

async def async_agent_call(user_needs, user_attributes, user_type, question):
    def create_prompt_template():
        template = """
            AMAZON CUSTOMER SERVICE AGENT - PAYMENT ISSUES RESOLUTION

                I. ROLE OVERVIEW
                Your primary responsibility is to efficiently resolve payment-related issues for Amazon customers.

                II. INFORMATION RETRIEVAL TOOLS
                A. Payment Queries
                1. 'payment_query_search': General payment information
                2. 'Amazon-Pay-Services': Specific Amazon Pay services
                3. 'Amazon-Pay-FAQs': Amazon Pay-specific questions
                4. 'Amazon-Pay-Services-Faqs': Amazon Pay service inquiries
                5. 'Amazon-AWS-Billing-FAQs': AWS billing and payment questions

                B. Order and Account Information
                1. 'order_confirmation': Order status (requires transaction ID)
                2. 'Prime-Members': Amazon Prime membership inquiries

                C. Policy and Customer Experience
                1. 'Amazon_policy': Policy questions
                2. 'financial_management': Financial data queries
                3. 'Customer-pain-point': Assess customer emotions

                III. PROMPT STRUCTURE
                Question: {question}
                User Profile: {profile}
                User Needs: {needs}

                IV. RESPONSE GUIDELINES
                A. Content
                1. Tailor responses to user's profile and needs
                2. Provide clear, concise, and helpful information
                3. Focus on solutions and recommendations

                B. Format
                1. Use bullet points for clarity
                2. Keep conversations brief and focused
                3. Format text with clear spacing for readability
                4. Highlight important information

                C. Special Considerations
                1. Embed links in relevant words (e.g., "Click [here](link)")
                2. For missing transaction IDs, assure customer and explain order may not be confirmed yet
                3. Respond empathetically, appropriate to customer's emotional state

                V. IMPORTANT NOTES
                A. Exclude tool invocation commands from responses
                B. Do not mention user types or tools used in the response
                C. Focus solely on providing informative and helpful replies

                VI. INTERACTION FLOW
                1. Analyze user question, profile, and needs
                2. Select appropriate information retrieval tool(s)
                3. Formulate response according to guidelines
                4. Deliver clear, concise, and tailored answer to user
            """
        return PromptTemplate.from_template(template=template)

    def format_prompt(prompt_template, question, users, needs, types):
        users_str = "\n".join(users)
        needs_str = "\n".join(needs)
        types_str = "\n".join(types)
        return prompt_template.format(
            question=question,
            profile = users_str,
            user_type=types_str,
            needs=needs_str
        )

    # Prepare the prompt
    prompt_template = create_prompt_template()
    formatted_prompt = format_prompt(prompt_template, question, user_attributes, user_needs, user_type)

    # Asynchronously invoke the agent
    response = await asyncio.to_thread(agent_with_chat_history.invoke, {"input": formatted_prompt}, {"configurable": {"session_id": "<foo>"}})

    return response['output']

