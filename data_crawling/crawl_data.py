from selenium import webdriver
from time import sleep
import json
from bs4 import BeautifulSoup

data = []
browser = webdriver.Firefox(executable_path="data_crawling/geckodriver.exe")


def exportjson(data):
    with open("datasets/crawlfile.json", "w") as f:
        json.dump(data, f, indent=4)
        print("Data extracted")


def openpage(link):
    browser.get(link)
    return browser.page_source


def getamazondata(product, cate):
    name = product.h2.text.strip()
    url = "http://amazon.com" + product.h2.a.get("href")
    try:
        price = product.find("span", class_="a-offscreen").text
    except:
        price = "None"
    img = product.find(
        "div", class_="a-section aok-relative s-image-square-aspect"
    ).img["src"]

    record = {"Name": name,"URL": url,"Price": price,"ImageLink": img,"Category": cate,
    }
    data.append(record)


def main():
    clothes_categories = ["shirt","outwear","shorts","skirt","dress","jacket", "pants",]
    for i in range(len(clothes_categories)):
        for j in range(1, 8):
            linkamazon = (
                "https://www.amazon.com/s?k=" + clothes_categories[i] + f"&page={j}"
            )
            html_text = openpage(linkamazon)
            soup = BeautifulSoup(html_text, "html.parser")
            products = soup.find_all(
                "div", {"data-asin": True, "data-component-type": "s-search-result"}
            )
            cate = clothes_categories[i]
            if cate == "jacket":
                cate = "outwear"
            elif cate == "pants":
                cate = "short"
            elif cate == "shorts":
                cate = "short"

            for product in products:
                getamazondata(product, cate)
                
    exportjson(data)


if __name__ == "__main__":
    main()

sleep(5)
browser.close()
