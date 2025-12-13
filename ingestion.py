import os
from dotenv import load_dotenv

from langchain_community.document_loaders import TextLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.documents import Document
from langchain_openai import OpenAIEmbeddings
from langchain_pinecone import PineconeVectorStore

load_dotenv()

def main():
    print("Hello from vector-bd-into!")
    loader = TextLoader("./nup.txt")
    document= loader.load()

    # Split the document robustly (handles long lines / no newlines better)
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=150,
        separators=["\n\n", "\n", ". ", " ", ""],
    )
    texts = text_splitter.split_documents(document)
    print(f"created {len(texts)} chunks")

    # IMPORTANT: Pinecone index dimension must match embedding dimension.
    # Your Pinecone index is dimension=512, so we must request 512-d embeddings.
    embeddings = OpenAIEmbeddings(
        openai_api_key=os.environ.get("OPENAI_API_KEY"),
        model="text-embedding-3-small",
        dimensions=1536,
    )

    print("ingesting")
    PineconeVectorStore.from_documents(
        texts,
        embeddings,
        index_name=os.environ.get("INDEX_NAME"),
    )

if __name__ == "__main__":
    main()

