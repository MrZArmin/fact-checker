import os
import ssl
import time
import threading
from queue import Queue
from concurrent.futures import ThreadPoolExecutor, as_completed
from selenium import webdriver
from selenium.webdriver.chrome.service import Service
from selenium.webdriver.common.by import By
from selenium.webdriver.support.ui import WebDriverWait 
from selenium.webdriver.support import expected_conditions as EC
from selenium.webdriver.common.keys import Keys
from dotenv import load_dotenv
from django.core.management.base import BaseCommand
from django.db import transaction
from django.utils.dateparse import parse_date
from factChecker.models import Link, Article, ArticleKeyword, Keyword

MAX_WORKERS = 5

class Command(BaseCommand):
    help = 'Scrape news data from the Nemzeti Archivum website'

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.driver_pool = Queue()

    def handle(self, *args, **options):
        load_dotenv()
        ssl._create_default_https_context = ssl._create_unverified_context
        
        # Initialize and login drivers
        for _ in range(MAX_WORKERS):
            driver = self.initialize_driver()
            self.login(driver)
            self.driver_pool.put(driver)

        links = Link.objects.filter(scraped=False)
        
        with ThreadPoolExecutor(max_workers=MAX_WORKERS) as executor:
            future_to_link = {executor.submit(self.process_link, link): link for link in links}
            for future in as_completed(future_to_link):
                link = future_to_link[future]
                try:
                    future.result()
                except Exception as exc:
                    self.stdout.write(self.style.ERROR(f'Link {link.url} generated an exception: {exc}'))
                else:
                    self.stdout.write(self.style.SUCCESS(f'Successfully processed {link.url}'))

        # Quit all drivers after processing is complete
        while not self.driver_pool.empty():
            driver = self.driver_pool.get()
            driver.quit()

    def process_link(self, link):
        driver = self.driver_pool.get()
        try:
            news_data = self.extract_news_data(driver, link.url)
            self.save_to_db(news_data, link)
        finally:
            self.driver_pool.put(driver)

    def initialize_driver(self) -> webdriver.Chrome:
        options = webdriver.ChromeOptions()
        options.add_argument("--headless")
        options.add_argument("--no-sandbox")
        options.add_argument("--disable-dev-shm-usage")
        options.add_argument("--disable-gpu")
        options.add_argument("--window-size=1920,1080")
        
        service = Service("/usr/local/bin/chromedriver")
        driver = webdriver.Chrome(service=service, options=options)
        return webdriver.Chrome(options=options)
    
    def login(self, driver: webdriver.Chrome):
        wait = WebDriverWait(driver, 10)
        driver.get("https://nemzetiarchivum.hu/news_bank")
        login_btn = wait.until(EC.element_to_be_clickable((By.XPATH, "/html/body/div[1]/div/div/div[2]/div/div/div[2]/div[2]/a[1]")))
        login_btn.click()

        email = os.getenv("ARCHIVUM_USERNAME")
        password = os.getenv("ARCHIVUM_PASSWORD")

        email_input = wait.until(EC.presence_of_element_located((By.XPATH, "/html/body/div[1]/div/div/div[2]/div/form/div[1]/div/input")))
        email_input.send_keys(email)
        password_input = driver.find_element(By.XPATH, "/html/body/div[1]/div/div/div[2]/div/form/div[2]/div/input")
        password_input.send_keys(password)
        submit_btn = driver.find_element(By.XPATH, "/html/body/div[1]/div/div/div[2]/div/form/div[3]/div/input")
        submit_btn.click()

    def return_shadow_root(self, driver: webdriver.Chrome, xpath: str) -> webdriver.Chrome:
        wait = WebDriverWait(driver, 10)
        shadow_host = wait.until(EC.presence_of_element_located((By.XPATH, xpath)))
        shadow_root = driver.execute_script('return arguments[0].shadowRoot', shadow_host)
        return shadow_root

    def extract_news_data(self, driver: webdriver.Chrome, link: str) -> dict:
        driver.get(link)
        shadow_root = self.return_shadow_root(driver, "/html/body/div[2]/mtva-hiradatbank")
        wait = WebDriverWait(shadow_root, 10)
        
        title = wait.until(EC.presence_of_element_located((By.CSS_SELECTOR, "div[class^='_title']"))).text
        tags = shadow_root.find_element(By.CSS_SELECTOR, "div[class^='_slug']").text if shadow_root.find_elements(By.CSS_SELECTOR, "div[class^='_slug']") else ""
        source = shadow_root.find_element(By.CSS_SELECTOR, "div[class^='_source']").text if shadow_root.find_elements(By.CSS_SELECTOR, "div[class^='_source']") else ""
        lead = shadow_root.find_element(By.CSS_SELECTOR, "div[class^='_lead']").text if shadow_root.find_elements(By.CSS_SELECTOR, "div[class^='_lead']") else ""
        date_id = shadow_root.find_elements(By.CSS_SELECTOR, "div[class^='_top_item']")[1].text if len(shadow_root.find_elements(By.CSS_SELECTOR, "div[class^='_top_item']")) > 1 else ""
        date = date_id.split(":")[0] if date_id else ""
        article_id = date_id.split(":")[-1] if date_id else ""
        
        content = shadow_root.find_element(By.CSS_SELECTOR, "div[class^='_item']") if shadow_root.find_elements(By.CSS_SELECTOR, "div[class^='_item']") else ""
        paragraphs = content.find_elements(By.CSS_SELECTOR, "div > div > div > p")[1:]
        text = " ".join([p.text for p in paragraphs])

        keyword_buttons = shadow_root.find_elements(By.CSS_SELECTOR, "button[class^='ov_button ov__meta ov_meta ov__pct']")
        keywords = self.extract_keywords(keyword_buttons)
        
        return {
            'id': article_id.strip(),
            'date': date.strip(),
            'tags': tags,
            'title': title,
            'source': source,
            'lead': lead,
            'text': text,
            'link': link,
            'iptc_codes': keywords
        }

    def extract_keywords(self, keyword_buttons):
        keywords = []
        for button in keyword_buttons:
            name = button.find_element(By.CSS_SELECTOR, "div.ov_meta_pct_label").text
            weight = button.find_element(By.CSS_SELECTOR, "div.ov_meta_pct_percent").text
            weight = int(weight.strip('%'))  # Convert percentage string to integer
            keywords.append({'name': name, 'weight': weight})
        return keywords
    
    @transaction.atomic
    def save_to_db(self, data: dict, link: Link):
        article = Article(
            date=parse_date(data['date']),
            tags=data['tags'],
            title=data['title'],
            lead=data['lead'],
            text=data['text'],
            link=data['link']
        )
        article.save()
        
        # Update the Link object to mark it as scraped
        link.scraped = True
        link.save()

        # Handle Keywords codes
        for keyword in data['keywords']:
            keyword, created = Keyword.objects.get_or_create(
                name=keyword['name'],
                defaults={'weight': keyword['weight']}
            )
            ArticleKeyword.objects.create(
                article=article,
                iptc_code=keyword,
                weight=keyword['weight']
            )