import re
import time
import random
import pandas as pd
from selenium import webdriver
from selenium.webdriver.chrome.service import Service
from selenium.webdriver.common.by import By
from selenium.webdriver.chrome.options import Options
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
from selenium.common.exceptions import NoSuchElementException, WebDriverException
from selenium.webdriver.common.action_chains import ActionChains
from selenium_stealth import stealth

class JobScraper:
    def __init__(self):
        self.user_agents = [
            "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36",
            "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/605.1.15 (KHTML, like Gecko) Version/14.1.1 Safari/605.1.15",
            "Mozilla/5.0 (Windows NT 10.0; Win64; x64; rv:89.0) Gecko/20100101 Firefox/89.0",
        ]
        self.job_data = []

        # Set up Selenium with Chrome
        self.chrome_options = Options()
        self.chrome_options.add_argument("--disable-blink-features=AutomationControlled")
        self.chrome_options.add_argument("--disable-extensions")
        self.chrome_options.add_argument("--disable-popup-blocking")
        self.chrome_options.add_argument("--disable-notifications")
        self.chrome_options.add_argument("--disable-infobars")
        self.chrome_options.add_argument("--disable-dev-shm-usage")
        self.chrome_options.add_argument("--no-sandbox")
        self.chrome_options.add_argument(f"user-agent={random.choice(self.user_agents)}")

        # Path to your ChromeDriver
        self.service = Service(r'E:\Programming\chromedriver-134.0.6998.35-win64\chromedriver.exe', log_path="chromedriver.log")
        self.driver = webdriver.Chrome(service=self.service, options=self.chrome_options)

        # Apply stealth settings
        stealth(
            self.driver,
            languages=["en-US", "en"],
            vendor="Google Inc.",
            platform="Win32",
            webgl_vendor="Intel Inc.",
            renderer="Intel Iris OpenGL Engine",
            fix_hairline=True,
        )

    def _get_random_user_agent(self):
        """Return a random user agent from the list."""
        return random.choice(self.user_agents)

    def _random_delay(self):
        """Add a random delay to mimic human behavior."""
        time.sleep(random.uniform(5, 10))  # Longer and more randomized delays

    def _scroll_page(self):
        """Scroll the page to mimic human behavior."""
        scroll_height = self.driver.execute_script("return document.body.scrollHeight")
        for _ in range(random.randint(1, 3)):
            scroll_position = random.randint(0, scroll_height)
            self.driver.execute_script(f"window.scrollTo(0, {scroll_position});")
            self._random_delay()

    def _simulate_mouse_movement(self):
        """Simulate human-like mouse movements."""
        action = ActionChains(self.driver)
        for _ in range(random.randint(2, 5)):
            x_offset = random.randint(-100, 100)
            y_offset = random.randint(-100, 100)
            action.move_by_offset(x_offset, y_offset).perform()
            self._random_delay()

    def scrape_careerbuilder(self, location="new-york", job_title="software-engineer", pages=10):
        """Scrape job data from CareerBuilder."""
        print(f"Scraping CareerBuilder for {job_title} in {location}...")

        for page in range(1, pages + 1):
            url = f"https://www.careerbuilder.com/jobs-{job_title}-in-{location}?page_number={page}"
            print(f"Scraping page {page}...")

            try:
                self.driver.get(url)
                self._random_delay()
                self._scroll_page()
                self._simulate_mouse_movement()

                # Wait for job cards to load
                WebDriverWait(self.driver, 10).until(
                    EC.presence_of_element_located((By.CSS_SELECTOR, 'div[class*="job-row"]'))
                )

                # Find all job cards
                job_cards = self.driver.find_elements(By.CSS_SELECTOR, 'div[class*="job-row"]')

                if not job_cards:
                    print(f"No job cards found on page {page}. Check the HTML structure.")
                    break

                for card in job_cards:
                    try:
                        # Extract job details
                        job_title_elem = card.find_element(By.CSS_SELECTOR, 'h2[class*="job-title"]')
                        company_elem = card.find_element(By.CSS_SELECTOR, 'div[class*="company-name"]')
                        location_elem = card.find_element(By.CSS_SELECTOR, 'div[class*="location"]')
                        salary_elem = card.find_element(By.CSS_SELECTOR, 'div[class*="pay"]')
                        job_desc_elem = card.find_element(By.CSS_SELECTOR, 'div[class*="job-description"]')

                        job_title_text = job_title_elem.text.strip() if job_title_elem else None
                        company_text = company_elem.text.strip() if company_elem else None
                        location_text = location_elem.text.strip() if location_elem else None
                        salary_text = salary_elem.text.strip() if salary_elem else None
                        job_desc_text = job_desc_elem.text.strip() if job_desc_elem else None

                        # Extract skills from job description
                        skills = self._extract_skills(job_desc_text) if job_desc_text else []

                        job_data = {
                            "title": job_title_text,
                            "company": company_text,
                            "location": location_text,
                            "salary": salary_text,
                            "description": job_desc_text,
                            "skills": skills,
                            "source": "CareerBuilder",
                        }

                        self.job_data.append(job_data)

                    except NoSuchElementException as e:
                        print(f"Error parsing job card: {e}")

            except WebDriverException as e:
                print(f"Failed to scrape page {page}. Error: {e}")
                continue

        return self.job_data

    def _extract_skills(self, text):
        """Extract skills from job description text."""
        if not text:
            return []

        # Common skills in tech/CS/data science/AI
        common_skills = [
            "Python", "Java", "JavaScript", "C++", "C#", "SQL", "NoSQL", "MongoDB",
            "React", "Angular", "Vue", "Node.js", "AWS", "Azure", "GCP",
            "Docker", "Kubernetes", "Git", "CI/CD", "Jenkins", "TensorFlow",
            "PyTorch", "Scikit-learn", "Pandas", "NumPy", "R", "Tableau",
            "Power BI", "Excel", "Machine Learning", "Deep Learning", "NLP",
            "Computer Vision", "Data Analysis", "Data Visualization", "Statistics",
            "Agile", "Scrum", "Project Management", "REST API", "GraphQL",
            "Django", "Flask", "Spring", "Hibernate", "TypeScript", "Go", "Rust",
            "Swift", "Kotlin", "PHP", "Ruby", "Rails", "Linux", "Unix", "Windows Server",
        ]

        found_skills = []
        for skill in common_skills:
            # Use regex to find whole words that match the skill
            pattern = r"\b" + skill + r"\b"
            if re.search(pattern, text, re.IGNORECASE):
                found_skills.append(skill)

        return found_skills

    def save_to_csv(self, filename="job_data.csv"):
        """Save scraped data to CSV."""
        df = pd.DataFrame(self.job_data)
        if not df.empty:
            df.to_csv(filename, index=False)
            print(f"Saved {len(df)} job listings to {filename}")
        else:
            print("No data to save. Scraping failed or no jobs found.")
        return df

    def close(self):
        """Close the Selenium browser."""
        self.driver.quit()