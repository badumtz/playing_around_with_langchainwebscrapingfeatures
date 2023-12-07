import openai
import langchain
from langchain.document_loaders import AsyncChromiumLoader
from langchain.document_transformers import BeautifulSoupTransformer
from langchain.llms import OpenAI
from langchain.chat_models import ChatOpenAI
from langchain.schema import HumanMessage
from langchain.chains import create_extraction_chain
import os
import pprint
from langchain.text_splitter import RecursiveCharacterTextSplitter

api_key = os.environ.get('OPENAI_API_KEY')
llm = ChatOpenAI(temperature=0, model="gpt-3.5-turbo-0613")
schema = {
    "properties": {
        "news_article_title": {"type": "string"},
        "news_article_summary": {"type": "string"},

    },
    "required": ["news_article_title", "news_article_summary"],
}


def extract(content: str, schema : dict):
    return create_extraction_chain(schema=schema, llm=llm).run(content)
def scrape_with_playwright(urls, schema):
    loader= AsyncChromiumLoader(urls)
    docs= loader.load()
    bs_transformer= BeautifulSoupTransformer()
    docs_transformed= bs_transformer.transform_documents(
        docs, tags_to_extract=["h2", "h3", "a"]
    )

    print("Extracting w LLM")

    splitter= RecursiveCharacterTextSplitter.from_tiktoken_encoder(
        chunk_size= 500, chunk_overlap=0
    )
    splits= splitter.split_documents(docs_transformed)

    extracted_content = splitter.split_documents(docs_transformed)

    pprint.pprint(extracted_content)
    return extracted_content

urls=["https://tazz.ro/timisoara/restaurante"]
extracted_content = scrape_with_playwright(urls, schema=schema)
'''restaurant_names = {}

for doc in extracted_content:
    # Extract names from the page_content
    names = [line.strip() for line in doc.page_content.split('\n') if line.strip()]

    # Use the source URL as the key in the dictionary
    key = doc.metadata['source']

    # Store the names in the dictionary
    restaurant_names[key] = names

# Print or use the dictionary as needed
for url, names in restaurant_names.items():
    print(f"Restaurants from {url}:")
    for name in names:
        print(f"- {name}")
    print()'''