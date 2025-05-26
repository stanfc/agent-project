import requests

# 你的 Google API key 與 Search Engine ID（請換成你自己的）
API_KEY = 'AIzaSyCKArAlrm41kxln52-_HvQzuoIlSCIqgjM'
SEARCH_ENGINE_ID = '268aba22ab0f94ac9'

def search_google(query, num=5):
    url = f'https://www.googleapis.com/customsearch/v1'
    params = {
        'key': API_KEY,
        'cx': SEARCH_ENGINE_ID,
        'q': query,
        'num': 10  #
    }
    response = requests.get(url, params=params)
    results = response.json()
    
    for i, item in enumerate(results.get('items', []), 1):
        print(f'{i}. {item["title"]}')
        print(f'網址: {item["link"]}')
        print(f'摘要: {item.get("snippet", "無摘要")}\n')

# 測試輸入產品名稱
product_name = 'iPhone 15'
search_google(product_name)