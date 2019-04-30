from selenium import webdriver
from selenium import webdriver
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
from selenium.webdriver.common.by import By
from selenium.webdriver.chrome.options import Options
from selenium.common.exceptions import (WebDriverException,
                                        NoSuchElementException,
                                        TimeoutException)
import pickle
import time


class Translator:
    def __init__(self, headless_browser=False, bulk=False):
        self.bulk = bulk
        self.headless_browser = headless_browser
        self.driver = self._start_driver()
        self._set_url()
        self._url = self.driver.current_url
        self.wait = WebDriverWait(self.driver, 4)

    @property
    def url(self):
        return self._url

    def _start_driver(self):
        chrome_options = Options()
        if self.headless_browser:
            chrome_options.add_argument('--headless')
            chrome_options.add_argument('--no-sandbox')

            # replaces browser User Agent from "HeadlessChrome".
            user_agent = "Chrome"
            chrome_options.add_argument('user-agent={user_agent}'
                                        .format(user_agent=user_agent))
        return webdriver.Chrome(chrome_options=chrome_options)

    def get_translation(self, sentence):
        source = '//textarea[@id="source"]'
        result = '//*[@class="tlid-translation translation"]'

        element = self.wait.until(EC.presence_of_element_located(
                                    (By.XPATH, source)))
        if self.bulk:
            self.sendKeys(element, sentence)
        else:
            element.send_keys(sentence)
        translation = self.wait.until(EC.presence_of_element_located(
                                    (By.XPATH, result)))
        return translation.text

    def sendKeys(self, elem, text):
        JS_ADD_TEXT_TO_INPUT = """
          var elm = arguments[0], txt = arguments[1];
          elm.value += txt;
          elm.dispatchEvent(new Event('change'));
          """
        self.driver.execute_script(JS_ADD_TEXT_TO_INPUT, elem, text)
        time.sleep(5)

    def translate_to(self, sentence, lang="en"):
        self._set_url(from_="es", to_=lang)

        translation_en = self.get_translation(sentence)

        self._set_url(from_=lang, to_="es")

        return self.get_translation(translation_en)

    def _set_url(self, from_="es", to_="en"):
        url = "https://translate.google.com.ar/#view=home&op=translate&sl={}&tl={}".format(from_, to_)
        self.driver.get(url)

    def quit(self):
        self.driver.quit()
