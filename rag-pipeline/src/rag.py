import os
import dotenv
from langchain.text_splitter import CharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain.chains import RetrievalQA
from langchain.prompts import PromptTemplate
from langchain_ollama import OllamaLLM

dotenv.load_dotenv()

def load_data(filepath):
    """Load text data from a file."""
    with open(filepath, 'r', encoding='utf-8') as f:
        return f.read()

def create_chunks(text, chunk_size=1000, chunk_overlap=200):
    """Split text into chunks."""
    text_splitter = CharacterTextSplitter(
        separator="\n",
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        length_function=len
    )
    return text_splitter.split_text(text)

def create_embeddings_and_vectorstore(chunks, embedding_model="sentence-transformers/all-MiniLM-L6-v2"):
    """Create embeddings and vectorstore from text chunks."""
    embeddings = HuggingFaceEmbeddings(model_name=embedding_model)
    vectorstore = FAISS.from_texts(texts=chunks, embedding=embeddings)
    return vectorstore

def create_prompt_template():
    """Create prompt template for QA."""
    template = """Используй следующие контексты для ответа на вопрос. Если ответ нельзя найти в контексте, скажи "Извини, я не знаю".

    Контекст:
    {context}

    Вопрос: {question}
    Ответ:
    """

    return PromptTemplate(template=template, input_variables=["context", "question"])

def create_retrieval_qa_chain(vectorstore, prompt_template):
    """Create a RetrievalQA chain with a language model (Ollama Mistral)."""
    llm = OllamaLLM(model="mistral")
    qa_chain = RetrievalQA.from_chain_type(
        llm=llm,
        chain_type="stuff",
        retriever=vectorstore.as_retriever(),
        chain_type_kwargs={"prompt": prompt_template},
    )
    return qa_chain

def main():
    # Load data
    file_path = "C:/VLADIK/MFTI/rag-projects/rag-pipeline/data/lectures.txt"
    text_data = load_data(file_path)

    # Create chunks
    chunks = create_chunks(text_data)

    # Create embeddings and vectorstore
    vectorstore = create_embeddings_and_vectorstore(chunks)

    # Create prompt template
    prompt_template = create_prompt_template()

    # Create RetrievalQA chain
    qa_chain = create_retrieval_qa_chain(vectorstore, prompt_template)


    while True:
        query = input("Введите запрос (или 'выход' для завершения): ")
        if query.lower() == 'выход':
            break
        result = qa_chain.invoke(query)
        print("Ответ:", result)

if __name__ == "__main__":
    main()
