"""
Crawl clothing positions from Amazon.com
"""

from time import sleep
from selenium import webdriver
from bs4 import BeautifulSoup

from sqlmodel import Session, select
from database.database import create_db_and_tables, fill_table, engine
from database.db_tables import CrawlData, ClothesCategory

browser = webdriver.Firefox(executable_path="geckodriver.exe")


def open_page(link):
    """Open web-page link in webdriver and return it's source code"""

    browser.get(link)
    return browser.page_source


def get_amazon_data(product, category):
    """Crawl Amazon data"""

    name = product.h2.text.strip()
    url = "http://amazon.com" + product.h2.a.get("href")

    try:
        price = product.find("span", class_="a-offscreen").text
    except AttributeError:
        price = "None"

    img = product.find(
        "div", class_="a-section aok-relative s-image-square-aspect"
    ).img["src"]

    return {
        "name": name,
        "url": url,
        "price": price,
        "image_link": img,
        "category": category,
    }


def read_amazon_pages():
    """Read selected amazon pages and find desired products"""

    with Session(engine) as session:
        statement = select(ClothesCategory.extended_category).where(
            ClothesCategory.extended_category != "sling"
        )
        clothes_categories = set(session.exec(statement))

    for clothes_category in clothes_categories:
        for page in range(1, 8):
            link_amazon = (
                "https://www.amazon.com/s?k=" + clothes_category + f"&page={page}"
            )
            html_text = open_page(link_amazon)
            soup = BeautifulSoup(html_text, "html.parser")
            products = soup.find_all(
                "div", {"data-asin": True, "data-component-type": "s-search-result"}
            )
            for product in products:
                yield get_amazon_data(product, clothes_category)


if __name__ == "__main__":
    create_db_and_tables()
    fill_table(CrawlData, read_amazon_pages(), truncate=True)

sleep(5)
browser.close()
