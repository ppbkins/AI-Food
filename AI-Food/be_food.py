from selenium import webdriver
from selenium.webdriver.chrome.service import Service
from selenium.webdriver.common.by import By
from time import sleep
import random
import pandas as pd
from pymongo import MongoClient

# 1. Connect to MongoDB
try:
    client = MongoClient("mongodb://localhost:27017/")
    db = client["FoodyDB"]
    collection = db["HaNoiRestaurants"]
    print("--- Đã kết nối tới MongoDB thành công ---")
except Exception as e:
    print(f"Lỗi kết nối MongoDB: {e}")

# 2. Declare WebDriver
driver = webdriver.Chrome()

# 3. Craw data
def crawl_foody(limit_items=1000):
    try:
        url = "https://www.foody.vn/ha-noi"
        driver.get(url)
        print(f"Đang kết nối tới: {url}")
        sleep(random.randint(5, 10))

        data_list = []

        while len(data_list) < limit_items:
            items = driver.find_elements(By.CSS_SELECTOR, ".filter-results .item")
            for item in items:
                if len(data_list) >= limit_items:
                    break       
                try:
                    name = item.find_element(By.CSS_SELECTOR, "h2").text
                    address = item.find_element(By.CSS_SELECTOR, ".address").text
                    if {"Name": name, "Address": address} not in data_list:
                        data_list.append({"Name": name, "Address": address})
                        print(f"[{len(data_list)}] {name}")
                except:
                    continue
            driver.execute_script("window.scrollTo(0, document.body.scrollHeight);")
            sleep(2) 

        df = pd.DataFrame(data_list)
        df.to_csv("foody_hanoi.csv", index=False, encoding="utf-8-sig")
        print("\n--- Đã lưu dữ liệu vào file foody_hanoi.csv ---")
    except Exception as e:
        print(f"Lỗi: {e}")
    finally:
        driver.quit()

if __name__ == "__main__":
    crawl_foody(limit_items=1000) 