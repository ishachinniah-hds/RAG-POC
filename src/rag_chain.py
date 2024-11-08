import os
from dotenv import load_dotenv
from langchain.prompts import PromptTemplate
from langchain_community.vectorstores import FAISS
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough
from transformers import pipeline, AutoTokenizer, AutoModelForCausalLM
from sentence_transformers import SentenceTransformer

# Load the API key from env variables
load_dotenv()
api_key = os.getenv("HUGGINGFACE_API_KEY")

RAG_PROMPT_TEMPLATE = """
You are a helpful coding assistant that can answer questions about the provided context. The context is usually a PDF document or an image (screenshot) of a code file. Augment your answers with code snippets from the context if necessary.

If you don't know the answer, say you don't know.

Context: {context}
Question: {question}
"""
PROMPT = PromptTemplate.from_template(RAG_PROMPT_TEMPLATE)

def format_docs(docs):
    return "\n\n".join(doc.page_content for doc in docs)

def create_rag_chain(chunks):
    embeddings_model = SentenceTransformer('sentence-transformers/all-MiniLM-L6-v2')
    doc_search = FAISS.from_documents(chunks, embeddings_model)
    retriever = doc_search.as_retriever(search_type="similarity", search_kwargs={"k":5})

    tokenizer = AutoTokenizer.from_pretrained("gpt-2")
    model = AutoModelForCausalLM.from_pretrained("gpt-2")  

    rag_chain = (
        {"context": retriever | format_docs, "question": RunnablePassthrough()}
        | PROMPT
        | (lambda input: model.generate(input, tokenizer))
        | StrOutputParser()
    )

    return rag_chain

