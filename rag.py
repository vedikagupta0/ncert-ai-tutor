import os
import shutil
from dotenv import load_dotenv
from langchain_community.vectorstores import FAISS
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough
from langchain_groq import ChatGroq
from langchain_core.runnables import Runnable

load_dotenv()
os.environ["GROQ_API_KEY"] = os.getenv("GROQ_API_KEY")
os.environ["GOOGLE_API_KEY"] = os.getenv("GOOGLE_API_KEY")

# #Extract the zip backup 
# backup_zip = "faiss_index_6_12ncert.zip" 
# restore_path = "faiss_index" 

# shutil.unpack_archive(backup_zip, restore_path) 
# print(f"Extracted vector DB backup to {restore_path}")

embeddings = GoogleGenerativeAIEmbeddings(
    model="models/text-embedding-004"
)

def format_docs(docs):
    return "\n\n".join(doc.page_content for doc in docs)

def load_rag():
    vector_store = FAISS.load_local(
        "faiss_index",
        embeddings,
        allow_dangerous_deserialization=True
    )

    retriever = vector_store.as_retriever(search_kwargs={"k": 4})

    llm = ChatGroq(
        model="llama-3.1-8b-instant",
        temperature=0
    )

    prompt = PromptTemplate.from_template(
        """
        You are a helpful AI assistant that answers questions strictly from NCERT context.

        Rules: 
        **You are AI Tutor for class 6th-12th NCERT content**
        1. Only use information from the provided context to answer questions and not any other informartion. 
        2. If the context doesn't contain enough information, then admit it directly in your response. 
        3. If it is a general question/greeting or fun elements, answer accordingly as long as it is not some knowledge based question, do not say "not enough data is provided" for greetings, general conversation inputs.
        Context:
        4. Keep your answers clear and concise. 
        5. If you're unsure, then say that you did not get enough data for the question. 
        
        {context}

        Question:
        {question}
        """
    )

    
    def question_extractor(inputs):
        return inputs["question"]

    # Manual chaining in a function
    def rag_chain_invoke(inputs, chat_history):
        query = inputs["question"]

        # Get docs
        docs = retriever.invoke(query)
        context = format_docs(docs)

        prompt_input = {
            "chat_history": chat_history,
            "context": context,
            "question": query
        }

        prompt_text = prompt.format(**prompt_input)

        answer = llm.invoke(prompt_text)

        # Extract response string
        response_text = answer.content if hasattr(answer, "content") else str(answer)

        # Update chat history with current turn
        updated_history = chat_history + [f"User: {query}", f"AI: {response_text}"]

        return response_text, updated_history


    return rag_chain_invoke
