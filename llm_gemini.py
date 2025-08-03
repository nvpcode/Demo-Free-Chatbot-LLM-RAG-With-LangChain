import os
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.prompts import PromptTemplate
from langchain.retrievers import EnsembleRetriever
from langchain_community.retrievers import BM25Retriever

from dotenv import load_dotenv
load_dotenv()
GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")

def func_retriever(vector_store):
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

def get_llm(question, retriever, model_choice):
    """
    Khởi tạo LLM của Google Gemini
    """
    # Khởi tạo ChatGoogleGenerativeAI
    llm = ChatGoogleGenerativeAI(
        model= model_choice,
        temperature=0.01,
        google_api_key=GOOGLE_API_KEY,
        streaming=True,
    )

    retrieved_docs = retriever.invoke(question)
    context_text = "\n\n".join(doc.page_content for doc in retrieved_docs)

    # Thiết lập prompt template
    prompt = PromptTemplate(
        template="""
        Bạn là một trợ lý AI thân thiện.
        Trả lời câu hỏi DỰA TRÊN thông tin dưới đây.
        Nếu không đủ thông tin, hãy nói "Tôi không chắc từ tài liệu đã cho".

        {context}

        Câu hỏi: {question}
        """,
        input_variables=['context', 'question']
    )

    final_prompt = prompt.format(context=context_text, question=question)

    return llm, final_prompt
