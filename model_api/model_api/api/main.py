# ##################################################################################################
#  Copyright (c) 2023 Nikolaos Panagiotou                                                          #
#                                                                                                  #
#  Permission is hereby granted, free of charge, to any person obtaining a copy                    #
#  of this software and associated documentation files (the "Software"), to deal                   #
#  in the Software without restriction, including without limitation the rights                    #
#  to use, copy, modify, merge, publish, distribute, sublicense, and/or sell                       #
#  copies of the Software, and to permit persons to whom the Software is                           #
#  furnished to do so, subject to the following conditions:                                        #
#                                                                                                  #
#  The above copyright notice and this permission notice shall be included in all                  #
#  copies or substantial portions of the Software.                                                 #
#                                                                                                  #
#  THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR                      #
#  IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,                        #
#  FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE                     #
#  AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER                          #
#  LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,                   #
#  OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE                   #
#  SOFTWARE.                                                                                       #
# ##################################################################################################

from typing import List

from fastapi import FastAPI

from newsmonitor_api.data.article import ArticleFetcher
from newsmonitor_api.data.clusters_summary import ClustersSummaryFetcher
from newsmonitor_api.data.count_statistics import StatisticsFetcher
from newsmonitor_api.data.simple_news_aggregated_data import AggregatedNews
from newsmonitor_api.data.trends import TrendsFetcher
from newsmonitor_api.models.article import Article, ArticleQuery
from newsmonitor_api.models.clusters_summary import ClustersSummaryQuery, ClustersSummary
from newsmonitor_api.models.count_statistics import CountStatisticsQuery, CountStatistics
from newsmonitor_api.models.simple_news_aggregated_data import SimpleNewsAggregatedData
from newsmonitor_api.models.trends import TrendsQuery, Trends

app = FastAPI()
article_fetcher = ArticleFetcher()
trends_fetcher = TrendsFetcher()
statistics_fetcher = StatisticsFetcher()
clusters_summary_fetcher = ClustersSummaryFetcher()
window_aggregated_news = AggregatedNews()


@app.post("/post_article/")
async def create_item(article: Article):
    return article


@app.post('/get_article/')
async def get_article(article_query: ArticleQuery) -> Article:
    return article_fetcher.get_article(article_query)


@app.post('/recent_articles/')
async def recent_articles() -> List[Article]:
    return article_fetcher.get_recent_articles()


@app.post('/recent_articles_snippets/')
async def recent_articles_snippets() -> List[Article]:
    return article_fetcher.get_recent_articles_snippets()


@app.post('/get_trends/')
async def get_trends(trends_query: TrendsQuery) -> Trends:
    return trends_fetcher.get_trends(trends_query)


@app.post('/recent_trends/')
async def recent_trends(trends_query: TrendsQuery) -> List[Trends]:
    return trends_fetcher.get_recent_trends(trends_query)


@app.post('/get_statistics/')
async def get_statistics(statistics_query: CountStatisticsQuery) -> CountStatistics:
    return statistics_fetcher.get_statistics(statistics_query)


@app.post('/recent_statistics/')
async def recent_statistics(statistics_query: CountStatisticsQuery) -> List[CountStatistics]:
    return statistics_fetcher.get_recent_statistics(statistics_query)


@app.post('/get_clusters_summary/')
async def get_trends(clusters_summary_query: ClustersSummaryQuery) -> ClustersSummary:
    return clusters_summary_fetcher.clusters_summary(clusters_summary_query)


@app.post('/recent_summaries/')
async def recent_summaries() -> ClustersSummary:
    return clusters_summary_fetcher.get_recent_clusters_summary()


@app.post('/aggregated_news/')
async def aggregated_news() -> SimpleNewsAggregatedData:
    return window_aggregated_news.window_aggregated_news()
