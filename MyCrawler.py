import scrapy
from bs4 import BeautifulSoup
import json
from scrapy.crawler import CrawlerProcess

queue = []
all = []
name = 1
b_name = 0


class QuotesSpider(scrapy.Spider):
    global all
    name = "MyCrawler"

    def __init__(self, start_url, degree, newN):
        self.n = newN
        self.start_urls = start_url
        self.in_degree = degree
        self.n -= self.start_urls.__len__()
        for i in range(0, self.start_urls.__len__()):
            all.append(self.start_urls[i])
        super(QuotesSpider, self).__init__()

    def parse(self, response):
        global queue, b_name

        dic = {'type': 'blog'}
        blog_name = response.css('title::text').extract_first()
        dic['blog_name'] = blog_name
        dic['blog_url'] = response.css('link::text').extract_first()

        post_urls = response.css('comments::text').extract()
        items = response.css('item').extract()

        if post_urls.__len__() >= self.in_degree:
            p = 5
        else:
            p = post_urls.__len__()

        for i in range(0, p):
            # dic['post_url_' + str(i + 1)] = post_urls[i]
            dic['post_url_' + str(i + 1)] = post_urls[i][0:post_urls[i].rfind('/')]
            post_title = items[i].split('title')[1].replace('</', '')
            post_title = post_title.replace('>', '')
            dic['post_title_' + str(i + 1)] = post_title
            post_content = items[i].split('description')[1]
            soup = BeautifulSoup(post_content)
            soup = BeautifulSoup(str(soup.text))
            dic['post_content_' + str(i + 1)] = str(soup.text)

        b_name += 1
        file = open('/Users/hanie/Desktop/output/' + 'blog_' + str(b_name) + '.json', 'w')
        json.dump(dic, file, ensure_ascii=False)

        for i in range(0, p):
            request = response.follow(post_urls[i], self.parse_post)
            request.meta['name'] = 'blog_' + str(b_name)
            request.meta['number'] = i
            yield request

    def parse_post(self, response):
        global queue, all, name

        filename = 'post.html'
        with open(filename, 'wb') as f:
            f.write(response.body)

        with open("post.html") as fp:
            soup = BeautifulSoup(fp)

        if not (str(soup.find('div', class_="post-content")) == ''):
            content = str(soup.find('div', class_="post-content"))
        if not (str(soup.find('div', class_="post-matn")) == ''):
            content = str(soup.find('div', class_="post-matn"))
        soup = BeautifulSoup(content)
        soup = BeautifulSoup(str(soup.text))
        tent = ''
        if not (soup.text == 'None'):
            tent += str(soup.text)

        blog_name = response.meta['name']
        f = open('/Users/hanie/Desktop/output/' + blog_name + '.json')
        dic = json.loads(f.read())
        dic['post_full_content_' + str(response.meta['number'])] = tent
        file = open('/Users/hanie/Desktop/output/' + blog_name + '.json', 'w')
        json.dump(dic, file, ensure_ascii=False)

        comments_urls = []
        c1 = False
        with open('post.html') as f:
            for line in f:
                if line.__contains__('<a name="comments"></a>'):
                    c1 = True
                if c1:
                    if line.__contains__('href="//') & line.__contains__('blog.ir/"'):
                        comments_urls.append(line.split('href="')[1].split('"')[0])
                if comments_urls.__len__() == self.in_degree:
                    break

        post_url = response.request.url
        dicti = {'type': 'post'}
        dicti['blog_url'] = post_url[0:post_url.rfind('blog.ir') + 7]
        dicti['post_url'] = post_url[0:post_url.rfind('/')]

        urls = []
        if self.n >= 1:
            for i in range(0, comments_urls.__len__()):
                if self.n >= 1:
                    if not (('http:' + comments_urls[i] + 'rss/') in all):
                        all.append('http:' + comments_urls[i] + 'rss/')
                        queue.append('http:' + comments_urls[i] + 'rss/')
                        self.n -= 1
                        comments_urls[i] = comments_urls[i]
                urls.append('http:' + comments_urls[i])

        dicti['comment_urls'] = urls

        file = open('/Users/hanie/Desktop/output/post' + str(name) + '.json', 'w')
        json.dump(dicti, file, ensure_ascii=False)
        name += 1

        while queue.__len__() > 0:
            yield response.follow(queue[0], self.parse, errback=self.erback, dont_filter=True)
            queue.remove(queue[0])

    def erback(self, failure):
        self.n += 1


urls = [str(x) for x in input().split(',')]
in_degree = int(input('Enter in_degree\n'))
n = int(input('Enter n:\n'))

process = CrawlerProcess({})
process.crawl(QuotesSpider, start_url=urls, degree=in_degree, newN=n)
process.start()
