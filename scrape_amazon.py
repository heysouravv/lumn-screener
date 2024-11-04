import requests
from bs4 import BeautifulSoup
import pandas as pd
import time
import random
from fake_useragent import UserAgent
import json
import logging
from typing import List, Dict
import re
from datetime import datetime

class AmazonSkincareScraper:
    def __init__(self):
        self.base_url = "https://www.amazon.in"
        self.headers = self._get_headers()
        self.logger = self._setup_logger()
        self.products = []
        
    def _setup_logger(self):
        """Configure logging"""
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(levelname)s - %(message)s',
            filename='amazon_scraping.log'
        )
        return logging.getLogger(__name__)
    
    def _get_headers(self) -> Dict:
        """Generate random headers to avoid detection"""
        ua = UserAgent()
        return {
            'User-Agent': ua.random,
            'Accept': 'text/html,application/xhtml+xml,application/xml;q=0.9,image/webp,*/*;q=0.8',
            'Accept-Language': 'en-US,en;q=0.5',
            'Connection': 'keep-alive',
        }
    
    def _extract_price(self, price_element) -> float:
        """Extract and clean price"""
        try:
            if price_element:
                price_text = price_element.text.strip()
                price = re.findall(r'\d+\.?\d*', price_text)
                return float(price[0]) if price else None
            return None
        except Exception as e:
            self.logger.error(f"Error extracting price: {str(e)}")
            return None
    
    def _extract_rating(self, rating_element) -> float:
        """Extract and clean rating"""
        try:
            if rating_element:
                rating_text = rating_element.text.strip()
                rating = re.findall(r'\d+\.?\d*', rating_text)
                return float(rating[0]) if rating else None
            return None
        except Exception as e:
            self.logger.error(f"Error extracting rating: {str(e)}")
            return None
    
    def _extract_review_count(self, review_element) -> int:
        """Extract and clean review count"""
        try:
            if review_element:
                review_text = review_element.text.strip()
                reviews = re.findall(r'\d+,?\d*', review_text)
                if reviews:
                    return int(reviews[0].replace(',', ''))
            return 0
        except Exception as e:
            self.logger.error(f"Error extracting review count: {str(e)}")
            return 0
    
    def _get_product_details(self, product_url: str) -> Dict:
        """Scrape detailed product information"""
        try:
            response = requests.get(
                product_url,
                headers=self._get_headers(),
                timeout=10
            )
            soup = BeautifulSoup(response.content, 'html.parser')
            
            # Extract product details
            details = {
                'url': product_url,
                'description': '',
                'ingredients': '',
                'brand': '',
                'size': '',
                'item_form': '',
                'skin_type': '',
                'about_item': []
            }
            
            # Get product description
            description_elem = soup.find('div', {'id': 'productDescription'})
            if description_elem:
                details['description'] = description_elem.text.strip()
            
            # Get product features/about item
            feature_bullets = soup.find('div', {'id': 'feature-bullets'})
            if feature_bullets:
                details['about_item'] = [li.text.strip() 
                                       for li in feature_bullets.find_all('li')]
            
            # Get technical details
            detail_section = soup.find('div', {'id': 'detailBullets_feature_div'})
            if detail_section:
                for item in detail_section.find_all('li'):
                    text = item.text.strip()
                    if 'Brand' in text:
                        details['brand'] = text.split(':')[-1].strip()
                    elif 'Size' in text:
                        details['size'] = text.split(':')[-1].strip()
                    elif 'Form' in text:
                        details['item_form'] = text.split(':')[-1].strip()
                    elif 'Skin Type' in text:
                        details['skin_type'] = text.split(':')[-1].strip()
            
            # Get ingredients
            ingredients_elem = soup.find('div', {'id': 'ingredients-section'})
            if ingredients_elem:
                details['ingredients'] = ingredients_elem.text.strip()
            
            return details
            
        except Exception as e:
            self.logger.error(f"Error getting product details: {str(e)}")
            return {}
    
    def scrape_products(self, num_pages: int = 5):
        """
        Scrape skincare products from Amazon
        Args:
            num_pages: Number of pages to scrape
        """
        base_search_url = f"{self.base_url}/s?k=skincare+products&i=beauty"
        
        for page in range(1, num_pages + 1):
            try:
                self.logger.info(f"Scraping page {page}")
                
                # Add page parameter
                url = f"{base_search_url}&page={page}"
                
                # Get page content
                response = requests.get(
                    url,
                    headers=self._get_headers(),
                    timeout=10
                )
                
                if response.status_code != 200:
                    self.logger.error(f"Error accessing page {page}: {response.status_code}")
                    continue
                
                soup = BeautifulSoup(response.content, 'html.parser')
                
                # Find all product containers
                products = soup.find_all('div', {'data-component-type': 's-search-result'})
                
                for product in products:
                    try:
                        # Extract basic product information
                        product_info = {
                            'timestamp': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
                            'asin': product.get('data-asin', ''),
                            'name': '',
                            'price': None,
                            'rating': None,
                            'review_count': 0,
                            'is_sponsored': False
                        }
                        
                        # Get product name
                        name_elem = product.find('span', {'class': 'a-text-normal'})
                        if name_elem:
                            product_info['name'] = name_elem.text.strip()
                        
                        # Get price
                        price_elem = product.find('span', {'class': 'a-price-whole'})
                        product_info['price'] = self._extract_price(price_elem)
                        
                        # Get rating
                        rating_elem = product.find('span', {'class': 'a-icon-alt'})
                        product_info['rating'] = self._extract_rating(rating_elem)
                        
                        # Get review count
                        review_elem = product.find('span', {'class': 'a-size-base'})
                        product_info['review_count'] = self._extract_review_count(review_elem)
                        
                        # Check if sponsored
                        sponsored_elem = product.find('span', {'class': 'a-color-secondary'})
                        if sponsored_elem and 'Sponsored' in sponsored_elem.text:
                            product_info['is_sponsored'] = True
                        
                        # Get product URL and detailed information
                        product_link = product.find('a', {'class': 'a-link-normal'})
                        if product_link:
                            product_url = self.base_url + product_link.get('href')
                            detailed_info = self._get_product_details(product_url)
                            product_info.update(detailed_info)
                        
                        self.products.append(product_info)
                        
                    except Exception as e:
                        self.logger.error(f"Error processing product: {str(e)}")
                        continue
                
                # Random delay between requests
                time.sleep(random.uniform(2, 5))
                
            except Exception as e:
                self.logger.error(f"Error processing page {page}: {str(e)}")
                continue
    
    def save_to_csv(self, filename: str = 'amazon_skincare_products.csv'):
        """Save scraped data to CSV"""
        try:
            df = pd.DataFrame(self.products)
            df.to_csv(f'data/{filename}', index=False)
            self.logger.info(f"Saved {len(self.products)} products to {filename}")
        except Exception as e:
            self.logger.error(f"Error saving to CSV: {str(e)}")
    
    def save_to_json(self, filename: str = 'amazon_skincare_products.json'):
        """Save scraped data to JSON"""
        try:
            with open(f'data/{filename}', 'w') as f:
                json.dump(self.products, f, indent=2)
            self.logger.info(f"Saved {len(self.products)} products to {filename}")
        except Exception as e:
            self.logger.error(f"Error saving to JSON: {str(e)}")

def main():
    # Create data directory if it doesn't exist
    import os
    if not os.path.exists('data'):
        os.makedirs('data')
    
    # Initialize scraper
    scraper = AmazonSkincareScraper()
    
    # Scrape products
    scraper.scrape_products(num_pages=5)  # Start with 5 pages
    
    # Save data
    scraper.save_to_csv()
    scraper.save_to_json()
    
    print(f"Scraped {len(scraper.products)} products")

if __name__ == "__main__":
    main()