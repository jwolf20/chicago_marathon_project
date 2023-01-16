# -*- coding: utf-8 -*-
"""
@author: Jonathan

To use this code must specify a year and a sex.

Example use call:
scrapy crawl old_chicago -o data_files/2016_M_chicago.csv -t csv -a year=2016 -a sex=M
"""

import scrapy
from scrapy.spiders import CrawlSpider, Rule
from scrapy.linkextractors import LinkExtractor


class ChicagoSpider(CrawlSpider):
    name = "old_chicago"

    custom_settings = {
        "DOWNLOAD_DELAY": 0.1,
    }

    # start_urls = ['http://chicago-history.r.mikatiming.de/2016/?page=1&event=MAR_999999107FA309000000008D&lang=EN_CAP&num_results=50&pid=list&search%5Bage_class%5D=%25&search%5Bsex%5D={}'.format(self.sex)]
    def start_requests(self):
        events = {
            "2016": "MAR_999999107FA309000000008D",
            "2015": "MAR_999999107FA3090000000079",
            "2014": "MAR_999999107FA3090000000065",
            "2013": "MAR_9999990E9A92360000000079",
            "2012": "MAR_9999990E9A9236000000003D",
            "2011": "MAR_9999990E9A92360000000029",
            "2010": "MAR_9999990E9A92360000000015",
            "2009": "MAR_9999990E9A92360000000002",
            "2008": "MAR_9999990E9A92360000000051",
            "2007": "MAR_9999990E9A92360000000052",
            "2006": "MAR_9999990E9A92360000000065",
        }

        # testing version
        # yield scrapy.Request('http://chicago-history.r.mikatiming.de/2016/?page=1&event={0}&lang=EN_CAP&num_results=50&pid=list&search%5Bage_class%5D=%25&search%5Bsex%5D={1}'.format(events[self.year], self.sex))

        # final version
        yield scrapy.Request(
            "http://chicago-history.r.mikatiming.de/2016/?page=1&event={0}&lang=EN_CAP&num_results=1000&pid=list&search%5Bage_class%5D=%25&search%5Bsex%5D={1}".format(
                events[self.year], self.sex
            )
        )

    rules = (
        # Find all the runners on the page and get their data
        # Rule(LinkExtractor(deny = ('favorite_add', ), restrict_xpaths=('.//*[@id="cbox-left"]/div[5]/div[1]/table/tbody', ), allow_domains = ('http://chicago-history.r.mikatiming.de',)), callback = 'parse_runner'),
        Rule(
            LinkExtractor(
                restrict_xpaths=('.//*[@id="cbox-left"]/div[5]/div[1]/table/tbody',)
            ),
            callback="parse_runner",
        ),
        # Go to the next page
        Rule(LinkExtractor(restrict_xpaths=('.//a[@class="pages-nav-button"]',))),
    )

    def parse_runner(self, response):

        # select all the shit from the page, then yield
        name = response.xpath('//*[@class="f-__fullname last"]/text()').extract_first()
        age_group = response.xpath(
            '//*[@class="f-age_class last"]/text()'
        ).extract_first()
        bib_no = response.xpath(
            '//*[@class="f-start_no_text last"]/text()'
        ).extract_first()
        age = response.xpath('//*[@class="f-age last"]/text()').extract_first()
        city = response.xpath(
            '//*[@class="f-__city_state last"]/text()'
        ).extract_first()
        initials = response.xpath(
            '//*[@class="f-display_name_short last"]/text()'
        ).extract_first()

        start_time = response.xpath(
            '//*[@class="f-starttime_net last"]/text()'
        ).extract_first()

        finish_time = response.xpath(
            '//*[@class="f-time_finish_netto last"]/text()'
        ).extract_first()
        overall_place = response.xpath(
            '//*[@class="f-place_nosex last"]/text()'
        ).extract_first()
        gender_place = response.xpath(
            '//*[@class="f-place_all last"]/text()'
        ).extract_first()
        age_group_place = response.xpath(
            '//*[@class="f-place_age last"]/text()'
        ).extract_first()

        race_table = []
        for split in [
            '" f-time_05 split"',
            '"list-highlight f-time_10 split"',
            '" f-time_15 split"',
            '"list-highlight f-time_20 split"',
            '" f-time_52 split"',
            '"list-highlight f-time_25 split"',
            '" f-time_30 split"',
            '"list-highlight f-time_35 split"',
            '" f-time_40 split"',
            '"list-highlight f-time_finish_netto"',
        ]:
            for param in ['"time_day"', '"time"', '"diff"', '"min_km"', '"kmh last"']:
                race_table.append(
                    response.xpath(
                        "//*[@class={0}]//*[@class={1}]/text()".format(split, param)
                    ).extract_first()
                )

        yield {
            "year": self.year,
            "name": name,
            "sex": ["female", "male"][self.sex == "M"],
            "age group": age_group,
            "age": age,
            "bib": bib_no,
            "city": city,
            "initials": initials,
            "starting time": start_time,
            "overall time": finish_time,
            "overall place": overall_place,
            "gender place": gender_place,
            "age group place": age_group_place,
            "5k split time of day": race_table[0],
            "5k split time": race_table[1],
            "5k split diff": race_table[2],
            "5k split min/mile": race_table[3],
            "5k split miles/h": race_table[4],
            "10k split time of day": race_table[5],
            "10k split time": race_table[6],
            "10k split diff": race_table[7],
            "10k split min/mile": race_table[8],
            "10k split miles/h": race_table[9],
            "15k split time of day": race_table[10],
            "15k split time": race_table[11],
            "15k split diff": race_table[12],
            "15k split min/mile": race_table[13],
            "15k split miles/h": race_table[14],
            "20k split time of day": race_table[15],
            "20k split time": race_table[16],
            "20k split diff": race_table[17],
            "20k split min/mile": race_table[18],
            "20k split miles/h": race_table[19],
            "Half split time of day": race_table[20],
            "Half split time": race_table[21],
            "Half split diff": race_table[22],
            "Half split min/mile": race_table[23],
            "Half split miles/h": race_table[24],
            "25k split time of day": race_table[25],
            "25k split time": race_table[26],
            "25k split diff": race_table[27],
            "25k split min/mile": race_table[28],
            "25k split miles/h": race_table[29],
            "30k split time of day": race_table[30],
            "30k split time": race_table[31],
            "30k split diff": race_table[32],
            "30k split min/mile": race_table[33],
            "30k split miles/h": race_table[34],
            "35k split time of day": race_table[35],
            "35k split time": race_table[36],
            "35k split diff": race_table[37],
            "35k split min/mile": race_table[38],
            "35k split miles/h": race_table[39],
            "40k split time of day": race_table[40],
            "40k split time": race_table[41],
            "40k split diff": race_table[42],
            "40k split min/mile": race_table[43],
            "40k split miles/h": race_table[44],
            "Finish split time of day": race_table[45],
            "Finish split time": race_table[46],
            "Finish split diff": race_table[47],
            "Finish split min/mile": race_table[48],
            "Finish split miles/h": race_table[49],
        }
