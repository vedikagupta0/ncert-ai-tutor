import os
import shutil
from dotenv import load_dotenv
from langchain_community.vectorstores import FAISS
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain_core.prompts import PromptTemplate
from langchain_groq import ChatGroq

load_dotenv()

os.environ["GROQ_API_KEY"] = os.getenv("GROQ_API_KEY")
os.environ["GOOGLE_API_KEY"] = os.getenv("GOOGLE_API_KEY")

# Extract the zip backup 
# backup_zip = "faiss_index_6_12ncert.zip" 
# restore_path = "faiss_index" 

# shutil.unpack_archive(backup_zip, restore_path) 
# print(f"Extracted vector DB backup to {restore_path}")

embeddings = GoogleGenerativeAIEmbeddings(
    model="models/text-embedding-004"
)

def format_chat_history(chat_history, max_turns=6):
    return "\n".join(chat_history[-max_turns:])

def format_docs(docs):
    return "\n\n".join(doc.page_content for doc in docs)

def load_rag():
    vector_store = FAISS.load_local(
        "faiss_index",
        embeddings,
        allow_dangerous_deserialization=True
    )

    retriever = vector_store.as_retriever(search_kwargs={"k": 3})

    llm = ChatGroq(
        model="llama-3.1-8b-instant",
        temperature=0
    )

    rewrite_prompt = PromptTemplate.from_template(
        """
    You are a query rewriter.

    Given the conversation and the latest question,
    rewrite the question so that it is fully self-contained
    and explicitly mentions the subject.

    Conversation:
    {chat_history}

    User question:
    {question}

    Rewritten question:
    """
    )

    prompt = PromptTemplate.from_template(
        """
        You are a helpful AI assistant that answers questions strictly from NCERT context.

        Rules: 
        **You are AI Tutor for class 6th-12th NCERT content**
        1. If the question refers to unclarified names/pronouns, priortize referring to the conversation so far if you do not understand the context.
        2. Only use information from the provided context to answer questions and not any other informartion unless the condition is Condition-7. 
        3. If the context doesn't contain enough information, then admit it directly in your response. 
        4. If it is a general question/greeting or fun elements, answer accordingly as long as it is not some knowledge based question, do not say "not enough data is provided" for greetings, general conversation inputs.
        5. Keep your answers clear and concise. 
        6. If you're unsure, then admit you did not get enough data for the question. 
        7. If the user asks for examples, real life use cases, MCQs, you can use your intelligence to answer these if the provided context and chat history is not enough.
        Conversation so far:
        {chat_history}

        NCERT Context:
        {context}

        Student Question:
        {question}
        """
    )

    # Manual chaining in a function
    def rag_chain_invoke(inputs, chat_history):
        query = inputs["question"]

        # chat_history is a LIST here
        chat_history_text = format_chat_history(chat_history)

        # Rewrite question using chat history
        rewritten_query = llm.invoke(
            rewrite_prompt.format(
                chat_history=chat_history_text,
                question=query
            )
        ).content.strip()

        # Retrieve using rewritten query
        docs = retriever.invoke(rewritten_query)
        context = format_docs(docs)

        # Final answer prompt
        prompt_text = prompt.format(
            chat_history=chat_history_text,
            context=context,
            question=query
        )

        answer = llm.invoke(prompt_text)
        response_text = answer.content if hasattr(answer, "content") else str(answer)

        return response_text

    return rag_chain_invoke
