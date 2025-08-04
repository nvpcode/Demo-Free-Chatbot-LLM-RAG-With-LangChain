import os
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.retrievers import EnsembleRetriever
from langchain_community.retrievers import BM25Retriever
from langchain.tools.retriever import create_retriever_tool
from langchain.agents import AgentExecutor, create_openai_functions_agent
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder  # Xử lý prompt

from dotenv import load_dotenv
load_dotenv()
GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")

def get_retriever(vector_store):
    """
    Kết hợp FAISS retriever (dựa trên embedding) và BM25 retriever (dựa trên từ khóa)
    theo tỷ trọng 7:3.
    """
    ## FAISS-based retriever (embedding)
    retriever_faiss = vector_store.as_retriever(
        search_type="similarity", search_kwargs={"k": 4}
    )

    ## BM25-based retriever (keyword)

        #Lấy 100 documents gần nhất từ FAISS
    top_docs = vector_store.similarity_search("", k=100)
        # Tạo BM25 retriever từ 100 tài liệu đã lấy
    retriever_bm25 = BM25Retriever.from_documents(top_docs)
    retriever_bm25.k = 4

    ## Kết hợp hai retriever
    ensemble_retriever = EnsembleRetriever(
        retrievers=[retriever_faiss, retriever_bm25],
        weights=[0.7, 0.3]
    )

    return ensemble_retriever


def get_llm_and_agent(model_choice, retriever):
    """
    Khởi tạo LLM của Google Gemini
    """
    # Khởi tạo ChatGoogleGenerativeAI
    llm = ChatGoogleGenerativeAI(
        model= model_choice,
        temperature=0.01,
        google_api_key=GOOGLE_API_KEY,
        model_kwargs={"streaming": True},
    )

    # Tạo công cụ tìm kiếm cho agent
    tool = create_retriever_tool(
        retriever,
        "find",
        "Search for information of a question in the knowledge base."
    )
    tools = [tool]

    # Prompt template cho agent
    system = """Bạn là chuyên gia về AI. Tên bạn là NVP-Chatbot.
                Hãy trả lời dựa trên thông tin được cung cấp. 
                Nếu không đủ thông tin, hãy nói: "Tôi không chắc chắn về tài liệu được cung cấp"."""

    prompt = ChatPromptTemplate.from_messages([
        ("system", system),
        ("system", "You have access to the following tools: {tool_names}. Use them if needed."),
        MessagesPlaceholder(variable_name="chat_history"),
        ("human", "{input}"),
        MessagesPlaceholder(variable_name="agent_scratchpad"),
    ])


    # Tạo agent
    agent = create_openai_functions_agent(llm=llm, tools=tools, prompt=prompt)

    return AgentExecutor(agent=agent, tools=tools, verbose=True)
