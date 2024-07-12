
from selenium import webdriver
from selenium.webdriver.common.keys import Keys
import datetime
import time
import pandas as pd

big_query_list = ["Acid",
                    "Alum and Water Treatment",
                    "Aluminum",
                    "Chemtrade",
                    "Commodity Industry",
                    "Commodities",
                    "Copper",
                    "Ethanol",
                    "Fertilizer",
                    "Glencore",
                    "High Value Metals",
                    "Industrial Chemical",
                    "Kennecott",
                    "Lead Acid Batteries",
                    "Lithium",
                    "Mining",
                    "Nickel",
                    "Paper",
                    "Petroleum",
                    "Potable Water",
                    "Pulp",
                    "Rail Cars",
                    "Rio Tinto",
                    "Rail",
                    "Ship",
                    "Shrieve Chemical",
                    "Steel",
                    "Sulfur",
                    "Tampa Sulfur",
                    "Vale",
                    "Vanadium",
                    "Vessel Charges",
                    "Vessel",
                    "Water Treatment",
                    "Zinc"
                  ]

#Getting the Training Data 
key_word_list = ["Stock", "Price", "Growth", "Production",
                 "Demand", "Supply", "Outlook", "Market", "Freight"]

def fix_query(query):
    '''Function to add the "%20" tag to indicate space between multiworded phrases
    
    Parameters:
    query (str): Word to be corrected
    
    Returns:
    _ (str): Keyword corrected'''
    return "%20".join(query.split(" "))


def get_data(big_query_list, last_year=True, bs4_call=False, word_list=key_word_list):
    '''Function to get the news articles from Google news
    
    Parameters:
    big_query_list (list[str]): List of string search words
    last_year (bool): Bool that indicates that the function should only focus on news articles posted in the last year
    bs4_call (bool): Bool to indicate that Beautiful Soup 4 shold be used to scrape the news articles (default is PhantomJS)
    words_list (list[str]): List of string additional words that provide additional search queries, added with each of the big_query_words

    Returns:
    df (pd.DataFrame): Pandas dataframe of the articles the function has scraped. Columns are, article_headlines, subject (big query word) & the link of the article
    articles_so_far (int): Number of articles that the function has scraped from google news 
    '''

    #Initialize the number of articles
    articles_so_far = 0

    #Google Search Base
    article_link_pre = "https://news.google.com/"
    
    article_headlines = []
    article_link = []
    subject_list = []

   
    
    
    #For word in key_word_list that enhances the big query words
    for key_word in key_word_list:

        #For the root word that is in the big_query_list
        for root_query in big_query_list:

            #Fixing the query so it can be searched 
            query = root_query + " " + key_word
            query_fixed = fix_query(query)

            #Query manipulation to decide on exact search term, based on duration of search 
            if last_year:
                search_term = "https://news.google.com/search?q=" + \
                    query_fixed + "%20when%3A21d&hl=en-US&gl=US&ceid=US%3Aen"
            else:
                search_term = "https://news.google.com/search?q=" + query_fixed + "&hl=en-CA&gl=CA&ceid=CA%3Aen"

            #Scraping headlines elements based on chosen method 
            if bs4_call:
                r = requests.get(search_term)
                soup = bs4(r.text, "html.parser")
                headline_list = soup.findAll("h3", "ipQwMb ekueJc RD0gLb")
            else:
                driver = webdriver.PhantomJS()
                try:
                    #driver.set_window_size(1920,1080)
                    print(search_term)
                    driver.get(search_term)
                    #lastCount = driver.find_elements_by_xpath("//a[@class='JtKRv']")
                    # while True:
                    #     #driver.execute_script("window.scrollTo(0, document.body.scrollHeight);")
                    #     #time.sleep(0.5)
                    #     newCount = driver.find_elements_by_xpath("//div[@class='m5k28']")
                    #     if lastCount == newCount:
                    #         break
                    #     lastCount = newCount
                    headline_list = driver.find_elements_by_xpath("//div[@class='m5k28']")
                except:
                    driver.save_screenshot('screenshot.png')
                    driver.close()
                    return None, None

            #Identifying the headline text, link and subject 
            for item in headline_list:
                article_headline = item.find_element_by_css_selector("a[class='JtKRv']").text
                if article_headline in article_headlines:
                    continue
                if root_query.lower() not in article_headline.lower():
                    continue
                article_headlines.append(article_headline)
                if bs4_call:
                    article_link.append(article_link_pre + item.find("a")["href"][2:])
                else:
                    article_link.append(item.find_element_by_tag_name("a").get_attribute("href"))
                subject_list.append(root_query)
                articles_so_far += 1

            time.sleep(1)

    if articles_so_far==0:
         driver.save_screenshot('screenshot.png')
    print("Total news headlines analyzed: {}".format(articles_so_far))
    df = pd.DataFrame(list(zip(article_headlines, subject_list, article_link)),
                      columns=["article_headline", "subject", "article_link"])
    now=datetime.datetime.now()
    code='Data/'+str(now.year)+'_'+str(now.month)+'_'+str(now.day)+'_'+str(now.hour)+'.csv'
    df.drop_duplicates(inplace=True)
    df.to_csv(code)
    return df, articles_so_far

get_data(big_query_list)

