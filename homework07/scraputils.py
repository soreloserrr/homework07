import requests
from bs4 import BeautifulSoup


def extract_news(parser):
    """ Extract news from a given web page """
    news_list = []
    for new_div, stat_div in zip(parser.select("tr.athing"), parser.select("td.subtext")):
        author = new_div.select_one("span.sitestr").text.strip() if new_div.select_one("span.sitestr") else ""
        comments = stat_div.select_one("a:last-child").text.split()[0] if stat_div.select_one("a:last-child") else 0
        points = int(stat_div.select_one("span.score").text.split()[0])
        title = new_div.select_one("a.storylink").text
        url = new_div.select_one("a.storylink")["href"]
        url = f"https://news.ycombinator.com/{url}" if "." not in url else url
        news_list.append({"author": author, "comments": comments, "points": points, "title": title, "url": url})
    return news_list


def extract_next_page(parser):
    """ Extract next page URL """
    return parser.find('a', {"class": "morelink"})["href"]


def get_news(url, n_pages=1):
    """ Collect news from a given web page """
    news = []
    while n_pages:
        print("Collecting data from page: {}".format(url))
        response = requests.get(url)
        soup = BeautifulSoup(response.text, "html.parser")
        news_list = extract_news(soup)
        next_page = extract_next_page(soup)
        url = "https://news.ycombinator.com/" + next_page
        news.extend(news_list)
        n_pages -= 1
    return news
