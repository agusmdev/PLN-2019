from selenium import webdriver
import time


class Translator:

    def __init__(self):
        self.driver = self._start_driver()

    def _start_driver(self):
        return webdriver.Chrome('./chromedriver')

    def translate(self, sentence, base_lang, target_lang):
        # Translate base_lang --> target_lang
        self.driver.get('https://translate.google.com/#view=home&op=translate&sl={}&tl={}'.format(base_lang, target_lang))

        self.driver.find_element_by_xpath('//textarea').send_keys(sentence)
        time.sleep(1)
        text = self.driver.find_element_by_xpath('//span[@class="tlid-translation translation"]').text
        time.sleep(2)
    return text.text


if __name__ == '__main__':
    tr = Translator()
    print(tr.translate_to('Ayer fui a la casa del perro, todo piola bro?'))