from selenium import webdriver
import requests
import os


def selenium_cookies_to_requests(driver):
    session = requests.Session()
    for cookie in driver.get_cookies():
        session.cookies.set(cookie["name"], cookie["value"])
    return session


def download_files_with_auth(url_list: "list[str]", login_url: str):
    options = webdriver.ChromeOptions()
    # Optional: add user profile for persistent login
    # options.add_argument("user-data-dir=/path/to/your/custom/profile")

    driver = webdriver.Chrome(options=options)

    # Navigate to login page and login manually if needed
    driver.get(login_url)
    input("Please log in and press Enter to continue...")

    # Transfer cookies
    session = selenium_cookies_to_requests(driver)

    for url, destination in url_list:
        try:
            print(f"Downloading {url}...")
            response = session.get(url)
            if response.ok:
                os.makedirs(os.path.dirname(destination), exist_ok=True)
                with open(destination, "wb") as f:
                    f.write(response.content)
                print(f"Saved to {destination}")
            else:
                print(f"Failed to download {url} - Status code: {response.status_code}")
        except Exception as e:
            print(f"Error downloading {url}: {e}")

    driver.quit()
