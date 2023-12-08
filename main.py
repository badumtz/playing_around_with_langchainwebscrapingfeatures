from eventlet import Queue
import eventlet
import openai
import langchain
from bs4 import BeautifulSoup
from langchain.document_loaders import AsyncChromiumLoader
from langchain.document_transformers import BeautifulSoupTransformer
from langchain.llms import OpenAI
from langchain.chat_models import ChatOpenAI
from langchain.schema import HumanMessage
from langchain.chains import create_extraction_chain, llm
import os
import pprint
from langchain.text_splitter import RecursiveCharacterTextSplitter
from dotenv import find_dotenv, load_dotenv
import asyncio
from playwright.async_api import BrowserContext, Page

AsyncChromiumLoader

load_dotenv(find_dotenv())


async def scrape_url(url, schema, llm, max_tokens):
    loader = AsyncChromiumLoader(urls=[url])
    extracted_menus = []

    # Scrape the menu from the specified URL
    response = await asyncio.get_event_loop().run_in_executor(
        None, loader.open, url
    )
    menu_content = await response.body()

    # Extract menu items and prices
    soup = BeautifulSoup(menu_content, 'html.parser')
    menu_items = soup.find_all('li', class_='item-name')
    prices = soup.find_all('span', class_='price')

    token_count = 0
    menu_data = []

    # Extract and process menu items
    for item, price in zip(menu_items, prices):
        menu_item_text = item.text
        price_text = price.text
        encoded_text = llm.encode(
            human_message=menu_item_text, schema=schema
        )
        token_count += len(encoded_text)
        if token_count >= max_tokens:
            break

        menu_item_data = {
            'menu_item': menu_item_text,
            'price': price_text
        }
        menu_data.append(menu_item_data)

    restaurant_data = {
        'restaurant_name': url,
        'menu_data': menu_data
    }
    extracted_menus.append(restaurant_data)

    return extracted_menus


if __name__ == '__main__':
    # Create an event loop
    loop = asyncio.new_event_loop()

    # Create a queue to communicate with the greenlets
    queue = Queue()

    # URLs to scrape
    urls = ["https://tazz.ro/timisoara/restaurante"]

    # Run the scrape_url function for each URL using eventlet
    pool = eventlet.pool.Pool()

    for url in urls:
        pool.spawn(scrape_url, url, schema="restaurant", llm=OpenAI(), max_tokens=10000)

    # Collect and process scraped data
    scraped_data = []
    for result in pool.imap(scrape_url, urls):
        scraped_data.extend(result)

    pprint.pprint(scraped_data)

    # Close the event loop
    pool.close()

'''schema = {
    "properties": {
        "restaurant_name": {"type": "string"},
        "restaurant_links": {"type": "string"},
    },
    "required": ["restaurant_name", "restaurant_links"],
}

if __name__ == "__main__":
    api_key = os.environ.get('OPENAI_API_KEY')
    llm = ChatOpenAI(temperature=0, model="gpt-3.5-turbo-16k-0613")
    urls = ["https://tazz.ro/timisoara/restaurante"]
    extracted_content = scrape_with_playwright(urls, schema=schema)'''
