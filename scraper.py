import cloudscraper
from bs4 import BeautifulSoup
import pandas as pd
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
import csv
import os

BASE_URL = "https://riyasewana.com/search/cars"
CSV_FILE = "vehicles.csv"
TARGET_COUNT = 5000

# Initialize cloudscraper
scraper = cloudscraper.create_scraper()

def get_listing_details(url):
    """Scrapes individual listing page for details."""
    try:
        if url.startswith("//"):
            url = "https:" + url
        elif url.startswith("/"):
            url = "https://riyasewana.com" + url

        response = scraper.get(url, timeout=15)
        if response.status_code != 200:
            return None
        
        soup = BeautifulSoup(response.text, 'html.parser')
        details = {'URL': url}
        
        # Get Title
        title_tag = soup.find('h1')
        details['Title'] = title_tag.text.strip() if title_tag else "N/A"
        
        # Get Price
        price_tag = soup.find('span', {'style': 'font-weight:bold;color:red;'})
        if not price_tag:
            price_tag = soup.find('h2', style="color:red;font-weight:bold;") # Alternate price selector
        if not price_tag:
            for span in soup.find_all(['span', 'h2']):
                if 'Rs.' in span.text or 'Negotiable' in span.text:
                    price_tag = span
                    break
        details['Price'] = price_tag.text.strip() if price_tag else "N/A"

        # Robust Detail Extraction
        all_tds = soup.find_all('td')
        for td in all_tds:
            p_tag = td.find('p', class_='moreh')
            if p_tag:
                key = p_tag.text.strip().replace(':', '')
                next_td = td.find_next_sibling('td')
                if next_td:
                    val = next_td.text.strip()
                    # Mapping to standard headers
                    mapped_key = key
                    if "Mileage" in key: mapped_key = "Mileage (km)"
                    if "Engine" in key: mapped_key = "Engine (cc)"
                    if "Fuel" in key: mapped_key = "Fuel Type"
                    if "Gear" in key: mapped_key = "Gear"
                    details[mapped_key] = val
        
        return details
    except Exception as e:
        print(f"Error fetching {url}: {e}")
        return None

def scrape_riyasewana():
    page = 1
    scraped_data_count = 0
    
    headers = ['Title', 'Price', 'Make', 'Model', 'YOM', 'Mileage (km)', 'Gear', 'Fuel Type', 'Engine (cc)', 'URL']
    
    if not os.path.exists(CSV_FILE):
        with open(CSV_FILE, 'w', newline='', encoding='utf-8') as f:
            writer = csv.DictWriter(f, fieldnames=headers, extrasaction='ignore')
            writer.writeheader()
    else:
        try:
            df = pd.read_csv(CSV_FILE)
            scraped_data_count = len(df)
            print(f"Continuing from existing {scraped_data_count} records.")
        except:
            pass

    while scraped_data_count < TARGET_COUNT:
        # Paging 1 usually works both ways, but let's try ?page=1
        url = f"{BASE_URL}?page={page}"
        print(f"Scraping page {page}: {url}")
        
        try:
            response = scraper.get(url, timeout=15)
            if response.status_code != 200:
                print(f"Failed to fetch page {page}: Status {response.status_code}")
                break
            
            soup = BeautifulSoup(response.text, 'html.parser')
            # Narrow down to search results container if possible
            # Based on logs, it's just 'li' with class 'item'
            listings = soup.find_all('li', class_='item')
            
            print(f"Found {len(listings)} listings on page {page}")
            
            if not listings:
                if page > 1: # End of results
                    print("No more listings found.")
                    break
                else: # Something went wrong on page 1
                    print("Initial page scan failed to find listings.")
                    break
            
            current_page_links = []
            for item in listings:
                link_tag = item.find('a')
                if link_tag and 'href' in link_tag.attrs:
                    current_page_links.append(link_tag['href'])
            
            with ThreadPoolExecutor(max_workers=5) as executor:
                future_to_url = {executor.submit(get_listing_details, link): link for link in current_page_links}
                for future in as_completed(future_to_url):
                    data = future.result()
                    if data:
                        scraped_data_count += 1
                        with open(CSV_FILE, 'a', newline='', encoding='utf-8', errors='ignore') as f:
                            writer = csv.DictWriter(f, fieldnames=headers, extrasaction='ignore')
                            writer.writerow(data)
                        
                        if scraped_data_count >= TARGET_COUNT:
                            break
            
            print(f"Overall Progress: {scraped_data_count} / {TARGET_COUNT}")
            page += 1
            time.sleep(3) 
            
        except Exception as e:
            print(f"Error on page {page}: {e}")
            break

    print(f"Finished. Total records in {CSV_FILE}: {scraped_data_count}")

if __name__ == "__main__":
    scrape_riyasewana()
