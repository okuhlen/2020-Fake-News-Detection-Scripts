class Article:

    def __init__(self, articleId, title, text, subject, date, label):
        self.articleId = articleId
        self.title = title
        self.text = text
        self.subject = subject
        self.date = date
        self.label = label

    def get_article_id(self):
        return self.articleId

    def get_title(self):
        return self.title

    def get_text(self):
        return self.text

    def get_subject(self):
        return self.subject

    def get_date(self):
        return self.date

    def get_label(self):
        return self.label

