from bottle import (
    route, run, template, request, redirect
)

from scraputils import get_news
from db import News, session
from bayes import NaiveBayesClassifier

s = session()

@route("/")
@route("/news")
def news_list():
    rows = s.query(News).filter(News.label == None).all()
    return template('news_template', rows=rows)


@route("/add_label")
def add_label():
    new_id = request.query.get('id')
    label = request.query['label']
    new = s.query(News).get(new_id)
    new.label = label
    s.commit()
    redirect("/news")


@route("/update")
def update_news():
    news_list = get_news('https://news.ycombinator.com/newest', 1)
    for item in news_list:
        if s.query(News).filter(News.title == item['title'], News.author == item['author']).first():
            continue
        s.add(News(**item))
    s.commit()
    redirect("/news")


@route("/classify")
def classify_news():
    train_news = s.query(News).filter(News.label != None).all()
    X_train = [new.title for new in train_news]
    y_train = [new.label for new in train_news]

    clf = NaiveBayesClassifier()
    clf.fit(X_train, y_train)

    classified_news = s.query(News).filter(News.label == None)
    labels = clf.predict([new.title for new in classified_news])
    labels = [(labels[i], new) for i, new in enumerate(classified_news)]
    labels.sort(key=lambda x: x[0])
    rows = [new[1] for new in labels]
    labels = [label[0] for label in labels]

    return template('news_template', rows=rows, labels=labels)


if __name__ == "__main__":
    run(host="localhost", port=8080)
