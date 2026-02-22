import requests
import json
import csv
import time
import os
import re
import random

TARGET_RECORDS = 5500
OUTPUT_FILE = os.path.join(os.path.dirname(__file__), '..', 'data', 'mobiles.csv')

LOCATIONS = [
    "colombo", "gampaha", "kandy", "kurunegala", "kalutara",
    "galle", "matara", "puttalam", "kegalle", "anuradhapura",
    "ratnapura", "badulla", "matale", "nuwara-eliya", "ampara",
    "batticaloa", "jaffna", "hambantota", "polonnaruwa", "moneragala",
    "trincomalee", "vavuniya", "kilinochchi", "mullaitivu", "mannar",
    "sri-lanka" # Catch-all
]

QUERIES = [
    "", "apple", "iphone", "samsung", "galaxy", "xiaomi", "redmi", "poco",
    "vivo", "oppo", "huawei", "nokia", "sony", "realme", "google", "pixel",
    "oneplus", "infinix", "honor", "itel", "tecno", "zte", "motorola"
]

headers = {
    'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36',
    'Accept': 'text/html,application/xhtml+xml,application/xml;q=0.9,image/avif,image/webp,image/apng,*/*;q=0.8',
    'Accept-Language': 'en-US,en;q=0.9',
}

def extract_json_data(html_content):
    pattern = re.compile(r'window\.initialData\s*=\s*(\{.*?\})\s*</script>', re.DOTALL)
    match = pattern.search(html_content)
    if match:
        try:
            return json.loads(match.group(1))
        except json.JSONDecodeError:
            return None
    return None

def parse_ads(json_data):
    ads_list = []
    if not json_data:
        return ads_list
    
    try:
        ads_array = json_data.get('serp', {}).get('ads', {}).get('data', {}).get('ads', [])
        for ad in ads_array:
            if ad.get('type') == 'app-download':
                continue
                
            parsed_ad = {
                'id': ad.get('id', ''),
                'title': ad.get('title', ''),
                'price': ad.get('price', ''),
                'location': ad.get('location', ''),
                'description': ad.get('description', ''),
                'membershipLevel': ad.get('membershipLevel', ''),
                'shopName': ad.get('shopName', ''),
                'isVerified': ad.get('isVerified', False),
                'timeStamp': ad.get('timeStamp', ''),
                'URL': f"https://ikman.lk/en/ad/{ad.get('slug', '')}" if ad.get('slug') else ''
            }
            if parsed_ad['title']:
                ads_list.append(parsed_ad)
    except Exception:
        pass
        
    return ads_list

def save_to_csv(data, filename, is_first_batch):
    if not data:
        return
        
    fieldnames = ['id', 'title', 'price', 'location', 'description', 'membershipLevel', 'shopName', 'isVerified', 'timeStamp', 'URL']
    mode = 'w' if is_first_batch else 'a'
    with open(filename, mode=mode, newline='', encoding='utf-8') as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        if is_first_batch:
            writer.writeheader()
        writer.writerows(data)

def main():
    print(f"Starting grid-search JSON scraper. Target: {TARGET_RECORDS} records.")
    os.makedirs(os.path.dirname(OUTPUT_FILE), exist_ok=True)
    
    total_scraped = 0
    seen_ids = set()
    session = requests.Session()
    session.headers.update(headers)
    
    # Shuffle to distribute load
    random.shuffle(LOCATIONS)
    random.shuffle(QUERIES)

    for loc in LOCATIONS:
        if total_scraped >= TARGET_RECORDS: break
        
        for q in QUERIES:
            if total_scraped >= TARGET_RECORDS: break
            
            # Ikman usually blocks or limits beyond page 4 for unauthenticated
            for page in range(1, 5): 
                if total_scraped >= TARGET_RECORDS: break
                
                url = f"https://ikman.lk/en/ads/{loc}/mobiles?page={page}"
                if q:
                    url += f"&query={q}"
                
                #print(f"Fetching: {loc} | query: {q} | page {page}")
                try:
                    response = session.get(url, timeout=10)
                    if response.status_code != 200:
                        break # Go to next query on failure (like 404 or 500)
                        
                    json_data = extract_json_data(response.text)
                    if not json_data:
                        break # Probably blocked or structure changed
                        
                    ads = parse_ads(json_data)
                    if not ads:
                        break # End of results for this combo
                    
                    unique_ads = []
                    for ad in ads:
                        ad_id = ad.get('id')
                        if ad_id and ad_id not in seen_ids:
                            seen_ids.add(ad_id)
                            unique_ads.append(ad)
                    
                    if not unique_ads:
                        # All duplicates, might just be top ads repeating
                        continue
                        
                    if total_scraped + len(unique_ads) > TARGET_RECORDS:
                        unique_ads = unique_ads[:(TARGET_RECORDS - total_scraped)]
                    
                    is_first_batch = (total_scraped == 0)
                    save_to_csv(unique_ads, OUTPUT_FILE, is_first_batch)
                    
                    total_scraped += len(unique_ads)
                    if total_scraped % 100 == 0 or total_scraped > TARGET_RECORDS - 100:
                         print(f"Progress: {total_scraped}/{TARGET_RECORDS}")
                    
                    time.sleep(random.uniform(0.5, 1.5))
                except requests.RequestException:
                    time.sleep(2)
                    break # Skip query on network error
                    
    print(f"Scraping completed. Successfully saved {total_scraped} unique records to {OUTPUT_FILE}.")

if __name__ == "__main__":
    main()

