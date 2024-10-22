import os
import ssl
import time
from datetime import datetime, timedelta
import concurrent.futures
from typing import List, Tuple
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
    help = 'Parallel scraping of news links from the Nemzeti Archivum website'

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.NUM_DRIVERS = 10
        self.processed_links = set()

    def add_arguments(self, parser):
        parser.add_argument('year', type=int, help='Year to scrape')

    def handle(self, *args, **options):
        load_dotenv()
        ssl._create_default_https_context = ssl._create_unverified_context
        
        year = options['year']
        date_ranges = self.generate_date_ranges(year)
        
        self.stdout.write(self.style.SUCCESS(f"Generated {len(date_ranges)} date ranges for year {year}"))
        
        with concurrent.futures.ThreadPoolExecutor(max_workers=self.NUM_DRIVERS) as executor:
            executor.map(self.scrape_date_range, date_ranges)
        
        self.stdout.write(self.style.SUCCESS(f"Completed scraping for year {year}"))

    def generate_date_ranges(self, year: int) -> List[Tuple[str, str]]:
        """Generate weekly date ranges for the given year."""
        start_date = datetime(year, 1, 1)
        end_date = datetime(year + 1, 1, 1)
        date_ranges = []
        
        current_date = start_date
        while current_date < end_date:
            range_start = current_date
            range_end = min(current_date + timedelta(days=6), end_date - timedelta(days=1))
            
            date_ranges.append((
                range_start.strftime("%Y.%m.%d."),
                range_end.strftime("%Y.%m.%d.")
            ))
            current_date = range_end + timedelta(days=1)
            
        return date_ranges

    def initialize_driver(self) -> webdriver.Chrome:
        """Initialize a Chrome driver with headless options."""
        options = webdriver.ChromeOptions()
        options.add_argument("--headless")
        options.add_argument("--no-sandbox")
        options.add_argument("--disable-dev-shm-usage")
        options.add_argument("--disable-gpu")
        options.add_argument("--window-size=1920,1080")
        
        service = Service("/usr/local/bin/chromedriver")
        driver = webdriver.Chrome(service=service, options=options)
        return driver

    def scrape_date_range(self, date_range: Tuple[str, str]) -> None:
        """Scrape news links for a specific date range."""
        driver = None
        try:
            driver = self.initialize_driver()
            self.login_to_filter_page(driver)
            self.apply_filter(driver, date_range[0], date_range[1])
            
            total_links = self.get_number_of_news(driver)
            self.stdout.write(self.style.SUCCESS(
                f"Processing date range {date_range[0]} - {date_range[1]}, found {total_links} links"
            ))
            
            self.get_news_links(driver, total_links)
            
        except Exception as e:
            self.stdout.write(self.style.ERROR(
                f"Error processing range {date_range[0]} - {date_range[1]}: {str(e)}"
            ))
        finally:
            if driver:
                driver.quit()

    def login_to_filter_page(self, driver: webdriver.Chrome):
        """Log in to the website."""
        wait = WebDriverWait(driver, 20)
        driver.get("https://nemzetiarchivum.hu/news_bank?p=search&s=eyJhcmVPcGVyYXRvcnNFbmFibGVkIjpmYWxzZSwiZmlsdGVycyI6W3siZGF0ZUZyb20iOiIyMDI0LjA5LjI2LiAxODozODowMCIsImRhdGVUbyI6IiIsImlzRXhhY3REYXRlIjpmYWxzZSwia2luZCI6MjV9XSwia2luZCI6MTcwLCJsYW5ndWFnZSI6MCwib3JkZXIiOjAsInF1ZXJ5IjoiIn0%3D")
        
        login_btn = wait.until(EC.element_to_be_clickable(
            (By.XPATH, "/html/body/div[1]/div/div/div[2]/div/div/div[2]/div[2]/a[1]")
        ))
        login_btn.click()

        email = os.getenv("ARCHIVUM_USERNAME")
        password = os.getenv("ARCHIVUM_PASSWORD")

        email_input = wait.until(EC.presence_of_element_located(
            (By.XPATH, "/html/body/div[1]/div/div/div[2]/div/form/div[1]/div/input")
        ))
        email_input.send_keys(email)
        
        password_input = driver.find_element(
            By.XPATH, "/html/body/div[1]/div/div/div[2]/div/form/div[2]/div/input"
        )
        password_input.send_keys(password)
        
        submit_btn = driver.find_element(
            By.XPATH, "/html/body/div[1]/div/div/div[2]/div/form/div[3]/div/input"
        )
        submit_btn.click()

    def return_shadow_root(self, driver: webdriver.Chrome, xpath: str) -> webdriver.Chrome:
        """Return the shadow root element."""
        wait = WebDriverWait(driver, 10)
        shadow_host = wait.until(EC.presence_of_element_located((By.XPATH, xpath)))
        shadow_root = driver.execute_script('return arguments[0].shadowRoot', shadow_host)
        return shadow_root

    def apply_filter(self, driver: webdriver.Chrome, start_date: str, end_date: str):
        """Apply date filter to the search."""
        shadow_root = self.return_shadow_root(driver, "/html/body/div[2]/mtva-hiradatbank")
        wait = WebDriverWait(shadow_root, 10)

        filters = wait.until(EC.presence_of_element_located(
            (By.CLASS_NAME, 'ov_search_form_query_row_end')
        ))
        filter_btn = filters.find_elements(By.TAG_NAME, 'button')[1]
        filter_btn.click()

        start_input = wait.until(EC.element_to_be_clickable(
            (By.CLASS_NAME, 'ov_range_start_input input')
        ))
        driver.execute_script("arguments[0].value = '';", start_input)
        start_input.send_keys(start_date)
        
        end_input_div = wait.until(EC.presence_of_element_located(
            (By.CLASS_NAME, 'ov_range_end_input')
        ))
        end_input = end_input_div.find_element(By.TAG_NAME, 'input')
        end_input.clear()
        end_input.send_keys(end_date)

    def get_number_of_news(self, driver: webdriver.Chrome) -> int:
        """Get the total number of news items for the current filter."""
        time.sleep(2)
        shadow_root = self.return_shadow_root(driver, "/html/body/div[2]/mtva-hiradatbank")
        wait = WebDriverWait(shadow_root, 20)
        is_present = wait.until(EC.text_to_be_present_in_element(
            (By.CLASS_NAME, 'ov_header_title'), ' találat'
        ))
        if is_present:
            number_of_news = shadow_root.find_element(By.CLASS_NAME, 'ov_header_title').text
        parts = number_of_news.split()
        result = ''.join([part for part in parts if part.isdigit()])
        return int(result)

    def get_news_links(self, driver: webdriver.Chrome, total_links: int):
        """Get all news links for the current filter."""
        shadow_root = self.return_shadow_root(driver, "/html/body/div[2]/mtva-hiradatbank")
        processed_count = 0

        while processed_count < total_links:
            current_links = shadow_root.find_elements(By.CLASS_NAME, "ov_result_title_link")
            current_links = current_links[processed_count:]
            current_links_href = [link.get_attribute("href") for link in current_links]
            
            new_links = [link for link in current_links_href if link not in self.processed_links]

            if new_links:
                self.save_to_db(new_links)
                self.processed_links.update(new_links)
                processed_count += len(new_links)

            if processed_count < total_links:
                try:
                    load_more_btn = WebDriverWait(shadow_root, 10).until(
                        EC.element_to_be_clickable((By.CLASS_NAME, "ov__footer_long"))
                    )
                    driver.execute_script("arguments[0].scrollIntoView();", load_more_btn)
                    WebDriverWait(driver, 30).until(EC.visibility_of(load_more_btn))
                    load_more_btn.click()
                except Exception as e:
                    self.stdout.write(self.style.WARNING(f"Failed to load more links: {str(e)}"))
                    break

    def save_to_db(self, links: list):
        """Save links to the database."""
        for url in links:
            try:
                with transaction.atomic():
                    Link.objects.create(url=url, scraped=False)
            except IntegrityError:
                pass