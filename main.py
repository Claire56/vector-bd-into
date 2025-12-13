import os
from dotenv import load_dotenv

from langchain_core.prompts import ChatPromptTemplate
from langchain_classic.chains import create_retrieval_chain
from langchain_classic.chains.combine_documents import create_stuff_documents_chain

from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain_pinecone import PineconeVectorStore

load_dotenv()

if __name__ == "__main__":
    index_name = os.environ.get("INDEX_NAME")
    if not index_name:
        raise ValueError("Missing INDEX_NAME in environment variables")

    embeddings = OpenAIEmbeddings(
        model="text-embedding-3-small",
        dimensions=1536,  # must match your Pinecone index dimension
        openai_api_key=os.environ.get("OPENAI_API_KEY"),
    )

    llm = ChatOpenAI(
        model="gpt-4o-mini",
        temperature=0,
        openai_api_key=os.environ.get("OPENAI_API_KEY"),
    )

    query = "what is the new Uganda?"

    vectorstore = PineconeVectorStore.from_existing_index(
        index_name=index_name,
        embedding=embeddings,
    )

    rag_prompt = ChatPromptTemplate.from_messages([
        ("system", "Answer using ONLY the provided context. If it's not in the context, say you don't know.\n\nContext:\n{context}"),
        ("human", "{input}"),
    ])

    combine_docs_chain = create_stuff_documents_chain(llm, rag_prompt)

    retrieval_chain = create_retrieval_chain(
        retriever=vectorstore.as_retriever(search_kwargs={"k": 4}),
        combine_docs_chain=combine_docs_chain,
    )

    result = retrieval_chain.invoke({"input": query})
    print("\nANSWER:\n", result.get("answer", result))