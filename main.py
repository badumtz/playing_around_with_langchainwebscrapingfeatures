import gevent as gevent
from gevent_queue import Queue
from gevent.redis import Redis
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


class AsyncChromiumLoader:
    def __init__(self, urls):
        self.urls = urls

    async def open(self, url):
        browser = BrowserContext().new_context()
        page = await browser.new_page()
        await page.goto(url)
        response = await page.content()
        browser.close()
        return response


def extract(content: str, schema: dict):
    return create_extraction_chain(schema=schema, llm=llm).run(content)


async def scrape_menu_with_playwright(urls, schema, llm, max_tokens):
    # Replace asyncio.get_event_loop() with creating a new event loop
    loop = asyncio.new_event_loop()

    loader = AsyncChromiumLoader(urls)
    extracted_menus = []

    for url in urls:
        response = await loop.run_in_executor(None, loader.open, url)
        menu_content = await response.body()

        soup = BeautifulSoup(menu_content, 'html.parser')
        menu_items = soup.find_all('li', class_='item-name')
        prices = soup.find_all('span', class_='price')

        token_count = 0
        menu_data = []

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

        # pprint.pprint(extracted_menus)


if __name__ == '__main__':
    # Create a queue to communicate with the greenlet
    queue = Queue()

    # Run the scrape_menu_with_playwright coroutine in a greenlet
    def worker():
        urls = ["https://tazz.ro/timisoara/restaurante"]
        schema = "restaurant"
        llm = OpenAI()
        max_tokens = 10000

        result = scrape_menu_with_playwright(urls, schema, llm, max_tokens)
        queue.put(result)

    g = gevent.spawn(worker)

    # Wait for the greenlet to finish
    try:
        result = queue.get()
        pprint.pprint(result)
    finally:
        gevent.kill(g)

    # Close the event loop
    gevent.shutdown()
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
