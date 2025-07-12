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


# Modified prompt to encourage concise responses
custom_prompt_template = """Using the context below, provide a brief and complete answer in 3-5 sentences maximum.
Context: {context}
Question: {question}
Concise answer:"""


@lru_cache(maxsize=1)
def set_custom_prompt() -> PromptTemplate:
    return PromptTemplate(
        template=custom_prompt_template,
        input_variables=['context', 'question']
    )


@lru_cache(maxsize=1)
def load_llm() -> CTransformers:
    return CTransformers(
        model="TheBloke/Llama-2-7B-Chat-GGML",
        model_type="llama",
        max_new_tokens=128,  # Significantly reduced for shorter responses
        temperature=0.1,     # Lower temperature for more focused responses
        download_model=True,
        threads=6,
        gpu_layers=0,
        top_k=10,
        top_p=0.9,
        context_length=1024  # Reduced context length
    )


@lru_cache(maxsize=1)
def load_embeddings() -> HuggingFaceEmbeddings:
    return HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-MiniLM-L6-v2",
        model_kwargs={'device': 'cpu'},
        cache_folder="./model_cache"
    )


def create_retriever(db: FAISS):
    return db.as_retriever(
        search_kwargs={
            'k': 1,  # Reduced to get more focused context
            'fetch_k': 2,
            'score_threshold': 0.7
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
            'verbose': False
        }
    )
    return qa_chain


@cl.on_chat_start
async def start():
    try:
        chain = qa_bot()
        msg = cl.Message(content="Initializing medical bot...")
        await msg.send()
        msg.content = "Hi, Welcome to Medical Bot. Please ask your question, and I'll provide a concise answer."
        await msg.update()
        cl.user_session.set("chain", chain)
    except Exception as e:
        error_msg = f"Error initializing bot: {str(e)}"
        await cl.Message(content=error_msg).send()


@cl.on_message
async def main(message: cl.Message):
    try:
        chain = cl.user_session.get("chain")
        if not chain:
            await cl.Message(content="Error: Bot not properly initialized. Please restart.").send()
            return


        processing_msg = cl.Message(content="Processing your query...")
        await processing_msg.send()


        cb = cl.AsyncLangchainCallbackHandler(
            stream_final_answer=True,
            answer_prefix_tokens=["FINAL", "ANSWER"]
        )
        cb.answer_reached = True


        async with asyncio.timeout(60):  # Reduced timeout
            res = await chain.ainvoke(
                {"query": message.content},
                callbacks=[cb]
            )
           
        answer = res.get("result", "No answer generated")
       
        await processing_msg.remove()
        await cl.Message(content=answer).send()


    except asyncio.TimeoutError:
        await processing_msg.remove()
        await cl.Message(content="I apologize for the delay. Please ask your question again in a simpler way.").send()
    except Exception as e:
        await processing_msg.remove()
        await cl.Message(content=f"Error: {str(e)}. Please try again.").send()
