from langchain_community.document_loaders import PyPDFLoader, DirectoryLoader
from langchain.prompts import PromptTemplate
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_community.llms import CTransformers
from langchain.chains import RetrievalQA
import chainlit as cl
from typing import Dict, Any
import asyncio
from functools import lru_cache

DB_FAISS_PATH = 'vectorstore/db_faiss'

# Simplified prompt template
custom_prompt_template = """Context: {context}
Question: {question}
Answer:"""

@lru_cache(maxsize=1)
def set_custom_prompt() -> PromptTemplate:
    return PromptTemplate(
        template=custom_prompt_template,
        input_variables=['context', 'question']
    )

@lru_cache(maxsize=1)
def load_llm() -> CTransformers:
    # Optimized LLM settings
    return CTransformers(
        model="TheBloke/Llama-2-7B-Chat-GGML",
        model_type="llama",
        max_new_tokens=256,
        temperature=0.5,
        download_model=True,
        threads=8,  # Increased threads for better performance
        gpu_layers=0,
        top_k=20,
        top_p=0.95,
        context_length=2048
    )

@lru_cache(maxsize=1)
def load_embeddings() -> HuggingFaceEmbeddings:
    return HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-MiniLM-L6-v2",
        model_kwargs={'device': 'cpu'},
        encode_kwargs={'normalize_embeddings': True},  # Added for better embedding quality
        cache_folder="./model_cache"
    )

def create_retriever(db: FAISS):
    return db.as_retriever(
        search_kwargs={
            'k': 3,  # Increased for better context
            'fetch_k': 5,
            'score_threshold': 0.5  # Lowered threshold for more matches
        }
    )

def qa_bot():
    embeddings = load_embeddings()
    db = FAISS.load_local(
        DB_FAISS_PATH,
        embeddings,
        allow_dangerous_deserialization=True
    )
    
    retriever = create_retriever(db)
    qa_chain = RetrievalQA.from_chain_type(
        llm=load_llm(),
        chain_type='stuff',
        retriever=retriever,
        return_source_documents=True,
        chain_type_kwargs={
            'prompt': set_custom_prompt(),
            'verbose': True  # Enable verbose mode for debugging
        }
    )
    return qa_chain

@cl.on_chat_start
async def start():
    chain = qa_bot()
    cl.user_session.set("chain", chain)
    await cl.Message(content="Hi, Welcome to Medical Bot. I'm ready to help with your questions!").send()

@cl.on_message
async def main(message: cl.Message):
    chain = cl.user_session.get("chain")
    if not chain:
        await cl.Message(content="Please refresh the page to restart the bot.").send()
        return

    try:
        cb = cl.AsyncLangchainCallbackHandler(
            stream_final_answer=True,
            answer_prefix_tokens=["FINAL", "ANSWER"]
        )
        cb.answer_reached = True

        # Increased timeout and added response processing
        async with asyncio.timeout(120):
            response = await chain.ainvoke(
                {"query": message.content},
                callbacks=[cb]
            )
            
            answer = response.get("result", "I apologize, I couldn't generate an answer. Please try rephrasing your question.")
            
            # Basic answer validation
            if len(answer.strip()) < 10:
                answer = "I apologize, but I need more context to provide a meaningful answer. Could you please elaborate on your question?"

            await cl.Message(content=answer).send()

    except asyncio.TimeoutError:
        await cl.Message(
            content="I apologize, but I'm taking longer than expected to process your request. Please try asking a more specific question."
        ).send()
    except Exception as e:
        await cl.Message(
            content=f"An error occurred while processing your request: {str(e)}. Please try again with a different question."
        ).send()
        