import os
from dotenv import load_dotenv
load_dotenv()

from langchain_community.document_loaders import ReadTheDocsLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import Pinecone as PineconeLangchain
#from langchain_openai import OpenAIEmbeddings
from langchain_openai import OpenAIEmbeddings
from langchain.vectorstores import Pinecone
import pinecone

from consts import INDEX_NAME

def ingest_docs()->None:
    loader = ReadTheDocsLoader(
       path= "langchain-docs/langchain.readthedocs.io/en/latest", encoding='utf8'
    )

    raw_documents = loader.load()
    print(f"loaded {len(raw_documents)} documents")
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=600, chunk_overlap=50, separators=["\n\n", "\n", " ", ""])
    documents = text_splitter.split_documents(raw_documents)
    print(f"Splitted into {len(documents)} chuncks")

    for doc in documents:
        new_url = doc.metadata["source"]
        # the url for langchain docs have been changed. hold for later practice.
        new_url = new_url.replace("langchain-docs", "https:/")
        ## customize for windows.
        new_url = new_url.replace("\\", "/")
        doc.metadata.update({"source": new_url})

    print(f"Going to add {len(documents)} to Pinecone")

    embeddings=OpenAIEmbeddings(openai_api_key=os.environ.get("OPENAI_API_KEY"))
    Pinecone.from_documents(documents, embeddings, index_name=INDEX_NAME)
    print("****Loading to vectorstore done ***")


if __name__ == "__main__":
    ingest_docs()