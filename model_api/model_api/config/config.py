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

import os

MONGO_URI = os.environ.get('MONGO_URI', 'mongodb://mongoadmin:secret@localhost:27018')
MONGO_DATABASE = os.environ.get('MONGO_DATABASE', 'NewsMonitor')
MONGO_UNPROCESSED_ARTICLES_COLLECTION = os.environ.get('MONGO_UNPROCESSED_ARTICLES_COLLECTION', 'UnprocessedHtmlDocs')
MONGO_NOUNS_TRENDS_COLLECTION = os.environ.get('MONGO_NOUNS_TRENDS_COLLECTION', 'NounsTrends')
MONGO_ENTITIES_TRENDS_COLLECTION = os.environ.get('MONGO_ENTITIES_TRENDS_COLLECTION', 'EntitiesTrends')
MONGO_CLUSTERS_SUMMARIES_COLLECTION = os.environ.get('MONGO_CLUSTERS_SUMMARIES_COLLECTION', 'ClustersSummaries')
MONGO_PERSON_STATISTICS_COLLECTION = os.environ.get('MONGO_PERSON_STATISTICS_COLLECTION', 'PersonStatistics')
MONGO_GPE_STATISTICS_COLLECTION = os.environ.get('MONGO_GPE_STATISTICS_COLLECTION', 'GPEStatistics')
MONGO_PROCESSED_DOCUMENTS_COLLECTION = os.environ.get('MONGO_PROCESSED_DOCUMENTS_COLLECTION', 'TextProcessedDocuments')

SIMPLE_STATISTICS_WINDOW_SEC = int(os.environ.get('SIMPLE_STATISTICS_WINDOW_SEC', '900'))
