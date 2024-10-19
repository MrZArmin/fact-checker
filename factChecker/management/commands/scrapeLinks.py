import os
import ssl
import time
from selenium import webdriver
from selenium.webdriver.chrome.service import Service
from selenium.webdriver.common.by import By
from selenium.webdriver.support.ui import WebDriverWait 
from selenium.webdriver.support import expected_conditions as EC
from dotenv import load_dotenv
from django.core.management.base import BaseCommand
from django.db import transaction
from factChecker.models import Link
from django.db.utils import IntegrityError

class Command(BaseCommand):
    help = 'Scrape news links from the Nemzeti Archivum website'

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.driver = None
        self.processed_links = set()

    def handle(self, *args, **options):
        load_dotenv()
        ssl._create_default_https_context = ssl._create_unverified_context
        
        try:
            self.driver = self.initialize_driver()
            self.login_to_filter_page()

            # Apply filter after logging in
            self.apply_filter(self.driver, "2014.01.01.", "2015.01.01.")

            total_links = self.get_number_of_news()
            self.stdout.write(self.style.SUCCESS(f"Total links to scrape: {total_links}"))

            self.get_news_links(total_links)
            
            self.stdout.write(self.style.SUCCESS(f"Total unique links processed: {len(self.processed_links)}"))
        except Exception as e:
            self.stdout.write(self.style.ERROR(f"An error occurred: {str(e)}"))
        finally:
            if self.driver:
                self.driver.quit()

    def initialize_driver(self) -> webdriver.Chrome:
        options = webdriver.ChromeOptions()
        options.add_argument("--headless")
        options.add_argument("--no-sandbox")
        options.add_argument("--disable-dev-shm-usage")
        options.add_argument("--disable-gpu")
        options.add_argument("--window-size=1920,1080")
        
        service = Service("/usr/local/bin/chromedriver")
        driver = webdriver.Chrome(service=service, options=options)
        self.stdout.write(self.style.SUCCESS("Driver initialized"))
        return driver
    
    def login_to_filter_page(self):
        wait = WebDriverWait(self.driver, 20)
        self.driver.get("https://nemzetiarchivum.hu/news_bank?p=search&s=eyJhcmVPcGVyYXRvcnNFbmFibGVkIjpmYWxzZSwiZmlsdGVycyI6W3siZGF0ZUZyb20iOiIyMDI0LjA5LjI2LiAxODozODowMCIsImRhdGVUbyI6IiIsImlzRXhhY3REYXRlIjpmYWxzZSwia2luZCI6MjV9XSwia2luZCI6MTcwLCJsYW5ndWFnZSI6MCwib3JkZXIiOjAsInF1ZXJ5IjoiIn0%3D")
        login_btn = wait.until(EC.element_to_be_clickable((By.XPATH, "/html/body/div[1]/div/div/div[2]/div/div/div[2]/div[2]/a[1]")))
        login_btn.click()

        email = os.getenv("ARCHIVUM_USERNAME")
        password = os.getenv("ARCHIVUM_PASSWORD")

        email_input = wait.until(EC.presence_of_element_located((By.XPATH, "/html/body/div[1]/div/div/div[2]/div/form/div[1]/div/input")))
        email_input.send_keys(email)
        password_input = self.driver.find_element(By.XPATH, "/html/body/div[1]/div/div/div[2]/div/form/div[2]/div/input")
        password_input.send_keys(password)
        submit_btn = self.driver.find_element(By.XPATH, "/html/body/div[1]/div/div/div[2]/div/form/div[3]/div/input")
        submit_btn.click()
        self.stdout.write(self.style.SUCCESS("Logged in"))

    def apply_filter(self, driver: webdriver.Chrome, start_date: str, end_date: str):
        shadow_root = self.return_shadow_root(driver, "/html/body/div[2]/mtva-hiradatbank")
        wait = WebDriverWait(shadow_root, 10)

        #apply filter
        filters = wait.until(EC.presence_of_element_located((By.CLASS_NAME, 'ov_search_form_query_row_end')))
        filter_btn = filters.find_elements(By.TAG_NAME, 'button')[1]
        filter_btn.click()

        start_input = wait.until(EC.element_to_be_clickable((By.CLASS_NAME, 'ov_range_start_input input')))
        driver.execute_script("arguments[0].value = '';", start_input)
        start_input.send_keys(start_date)
        
        end_input_div = wait.until(EC.presence_of_element_located((By.CLASS_NAME, 'ov_range_end_input')))
        end_input = end_input_div.find_element(By.TAG_NAME, 'input')
        end_input.clear()
        end_input.send_keys(end_date)

    def return_shadow_root(self, xpath: str):
        wait = WebDriverWait(self.driver, 20)
        shadow_host = wait.until(EC.presence_of_element_located((By.XPATH, xpath)))
        shadow_root = self.driver.execute_script('return arguments[0].shadowRoot', shadow_host)
        return shadow_root
    
    def get_number_of_news(self) -> int:
        time.sleep(2)
        shadow_root = self.return_shadow_root("/html/body/div[2]/mtva-hiradatbank")
        wait = WebDriverWait(shadow_root, 20)
        is_present = wait.until(EC.text_to_be_present_in_element((By.CLASS_NAME, 'ov_header_title'),' találat'))
        if is_present:
            number_of_news = shadow_root.find_element(By.CLASS_NAME,'ov_header_title').text
        number_of_news = number_of_news.split()[0]
        return int(number_of_news.replace(',', ''))

    def get_news_links(self, total_links: int):
        shadow_root = self.return_shadow_root("/html/body/div[2]/mtva-hiradatbank")
        processed_count = 0

        while processed_count < total_links:
            current_links = shadow_root.find_elements(By.CLASS_NAME, "ov_result_title_link")
            current_links_href = [link.get_attribute("href") for link in current_links]
            
            new_links = [link for link in current_links_href if link not in self.processed_links]
            self.save_to_db(new_links)
            self.processed_links.update(new_links)
            processed_count += len(new_links)

            self.stdout.write(self.style.SUCCESS(f"Processed {processed_count} out of {total_links} links"))

            if processed_count < total_links:
                try:
                    load_more_btn = WebDriverWait(shadow_root, 10).until(
                        EC.element_to_be_clickable((By.CLASS_NAME, "ov__footer_long"))
                    )
                    self.driver.execute_script("arguments[0].scrollIntoView();", load_more_btn)
                    load_more_btn.click()
                except Exception as e:
                    self.stdout.write(self.style.WARNING(f"Failed to load more links: {str(e)}"))
                    break

    @transaction.atomic
    def save_to_db(self, links: list):
        for url in links:
            try:
                Link.objects.create(url=url, scraped=False)
            except IntegrityError:
                pass