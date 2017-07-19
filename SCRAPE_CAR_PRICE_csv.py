#execfile('SCRAPE_CAR_PRICE_csv.py')
import requests
import re
from bs4 import BeautifulSoup
from urlparse import urljoin
from datetime import datetime
import math
import csv
import random
import time
from retrying import retry

    
def get_search_page_urls(response): # get total page info from the first search result page
    base_url = 'https://'+ city + '.craigslist.org'
    soup = BeautifulSoup(response.text, "lxml")
    totalcountlist = soup.select('span.button.pagenum span.totalcount')

    if len(totalcountlist) == 0:
        errlist = soup.select('p') #like 'This IP has been automatically blocked...'
        print errlist[0].text
        return None

    num_results = totalcountlist[0]
    total = int(num_results.text)
    print '    Total # of results = ', total
    num_pages = total/120
    print "    Total # of Pages = ", num_pages+1
    nextpage = soup.select('span.buttons a.button.next')[0]
    nexthref = nextpage['href']
    print nexthref
    urls = [response.url,]
    for s in range(1, num_pages+1):
        start = s * 120
        nhref = re.sub(r'(\?s=\d+&)', '?s=%d&' % start , nexthref)
        urls.append(urljoin(base_url, nhref)) 
    return urls
    


def get_search_titles(response): # get data for each search result page
    soup = BeautifulSoup(response.text, "lxml")
    parent = soup.select('p.result-info')
    base_url = 'https://'+ city + '.craigslist.org'
    titlel = [ a.select('a.result-title.hdrlnk')[0] for a in parent ]
    urls = [ urljoin(base_url, t['href']) for t in titlel ]
    pids = [ t['data-id'] for t in titlel ]
    titles = [ t.text for t in titlel ]
    timel = [ a.select('time.result-date')[0] for a in parent ]
    times = [ t['datetime'] for t in timel ]
    prices = []
    for firsta in parent:
        p = firsta.select('span.result-meta span.result-price')
        if len(p) == 0:
            prices.append('')
        else:
            dollar = p[0].text
            dollar = int(dollar.strip('$'))
            prices.append(dollar)
    prices, pids, urls, titles, times = map(lambda x: list(x), \
            zip(*filter(lambda z: z[0] != '', zip(prices, pids, urls, titles, times))))
    return zip(prices, pids, urls, times, titles)
    
    

def filter_irregular_char(mystring):  # delete irregular chars from string, for title & message
    ss = re.sub(r'\n+', '\n', mystring).strip()
    ss = ss.replace('"', '')
    ss = ss.replace("'", '')
    ss = ss.replace(',', ';')
    ss = re.sub(r'[^A-Za-z0-9 \n\.\/\-;:@\?$!&%\(\)]+', '', ss)
    ss = '#'.join([m.strip() for m in ss.split('\n')])
    #'\n'.join(ss.split('\n'))
    ss =  ss.lower()
    return ss


@retry(stop_max_attempt_number=1000,wait_fixed=600)
def analyze_page(page_link):
    title=''
    imglink=''
    message='' 
    attr_model='' 
    attributes=''

    page = requests.get(page_link, headers=head)
    soup = BeautifulSoup(page.text, "lxml")

    #remov = soup.select('div.removed')
    #if len(removed)>0:
    #    return '','','','',''

    img = soup.select('img')
    if len(img)>0:
        imglink = img[0]['src']
    else:
        imglink = ''

    body = soup.find('section', attrs={'id': 'postingbody'})
    if body is not None:
        message = filter_irregular_char(body.text)
        message = '#'.join(message.split('#')[1:]) #remove 'QR code ...' line
    else:
        message = ''

    titleonly = soup.find('span', attrs={'id': 'titletextonly'})
    if titleonly is not None:
        title = filter_irregular_char(titleonly.text)
    else:
        title = ''

    attr = soup.select('p.attrgroup')
    if attr is not None:
        if len(attr) > 1:
            attr_model = attr[0].select('span')[0].text

            attrlist = attr[1].select('span')
            #print attr[1].select('span')
            attributes = '#'.join([ at.text for at in attrlist ])
    else :
        attr_model = ''
        attributes = ''
    return title, imglink, message, attr_model, attributes




def get_year(title, attr_model):
    PATTERN_YEAR = r'(?:^| )(199\d{1}|20[0-1][0-9]) '
    mm = re.search(PATTERN_YEAR , title)
    if mm :
        year = mm.group(1)
    else :
        mm = re.search(PATTERN_YEAR , attr_model)
        if mm :
            year = mm.group(1)
        else:
            print 'Year not found'
            return -1
    print '         Year =', year,
    return int(year)

   
    
def get_miles(attributes, title, message):
    PATTERN_ODOMETER = r'odometer: (\d{4,6})'
    PATTERN_MILES = r'(\d{4,6}|\d{1,3}\;*\w{3}|\d{1,3}\s*k)\s*[orignal]*\s*mil+es*'
    PATTERN_MILEAGE = r'mileage\s*\:*\s*(\d{4,6}|\d{1,3}\,*\w{3}|\d{1,3}\s*k)'
    
    mm = re.search(PATTERN_ODOMETER , attributes)
    if mm :
        mile = mm.group(1)
    else:
        mm = re.search(PATTERN_MILES , title)
        if mm :
            mile = mm.group(1)
        else:
            mm = re.search(PATTERN_MILES , message)
            if mm :
                mile = mm.group(1)
            else:
                mm = re.search(PATTERN_MILEAGE , title)
                if mm :
                    mile = mm.group(1)
                else:
                    mm = re.search(PATTERN_MILEAGE , message)
                    if mm :
                        mile = mm.group(1)   
                    else :
                        return -1
                
    mile = mile.replace(' ' , '')
    mile = mile.replace(';' , '')
    mile = mile.replace('x' , '0')
    mile = mile.replace('o' , '0')
    mile = mile.replace('k' , '000')
    mile = re.sub(r'[^0-9]+', '', mile)
    print '  Miles =', mile,
    return int(mile)



def get_title_status(attributes):
    PATTERN = r'title status:\s*(\w+)\s*'
    mm = re.search(PATTERN, attributes)
    if mm :
        title = mm.group(1)
        return title
    else :
        return None

                

def newer_than_date(date_string, cutoff_string='2017-01-01 00:00'):
    cutoff = datetime.strptime(cutoff_string, '%Y-%m-%d %H:%M')
    dt     = datetime.strptime(date_string,   '%Y-%m-%d %H:%M') 
    return dt > cutoff








## -------- START GETTING SEARCH RESULTS ----------
#from retrying import retry

#@retry(stop_max_attempt_number=5,stop_max_delay=30000)
def get_all_pages(csvfilename, URLs):
    
    wrongcount = 0
    maxwrong = 10
    pagecount = startpage
    skipcount = 0
    maxskipwait = 30
    
    for myurl in URLs: 
    
        pidl =[]
        posttimel = []
        cityl = []
        statel = []
        makel = []
        modell = []
        pricel = []
        yearl = []
        milesl = []
        urll = []
        titlel = []
        attrl = []
        imglinkl = []
        messagel = []

        print '============= >>>  Now analyzing search result page #', pagecount 
        print myurl
        
        response = requests.get(myurl, headers=head)

        results_page = get_search_titles(response) #list of (prices, pids, urls, times, titles)

        random.shuffle(results_page) ### randomize order of open post pages
    
        for i, result in enumerate(results_page) :

            y = random.uniform(0.5,2)
            time.sleep(y)

            if skipcount > maxskipwait:
                y = random.uniform(10,20)
                print 'Skiped too many, sleep for %d sec...' % y
                skipcount = 0
                time.sleep(y)

            print 'PG #', i, ':', result[3],

            if newer_than_date(result[3],filterdate) :
                pass
            else:
                wrongcount += 1
                if wrongcount > maxwrong:
                    print 'Max wrong reached, break'
                    break
                
            page_link = result[2] #urls
            print ' ', page_link
            
            #page = requests.get(page_link, headers=head)
            #soup = BeautifulSoup(page.text, "lxml")
            
            mytitle, myimglink, mymessage, myattr_model, myattributes = analyze_page(page_link)
            if mytitle == '' :
                print '        ^^^^^^^^^^^^^^ Skip this entry ^^^^^^^^^^^^  [ no title ] '
                skipcount += 1
                continue
                
            myyear = get_year(mytitle, myattr_model) 
            mymiles = get_miles(myattributes, mytitle, mymessage)
            title_status = get_title_status(myattributes)

            if myyear == -1 or mymiles == -1 or title_status == 'salvage' \
                     or model not in mytitle:  #or make not in mytitle
                #skip this entry
                print '  Title =', mytitle
                print '        ^^^^^^^^^^^^^^ Skip this entry ^^^^^^^^^^^^  [ title status ]:', title_status
                skipcount += 1
            else: 
                print '  Title =', mytitle
                print '        === >> [[ Saved ]]'
                # append data
                pidl.append(result[1].encode('ascii', 'ignore'))
                posttimel.append(result[3].encode('ascii', 'ignore'))
                cityl.append(city.encode('ascii', 'ignore'))
                statel.append(state.encode('ascii', 'ignore'))
                makel.append(make.encode('ascii', 'ignore'))
                modell.append(model.encode('ascii', 'ignore'))
                pricel.append(result[0])
                yearl.append(myyear)          
                milesl.append(mymiles)
                urll.append(result[2].encode('ascii', 'ignore'))
                titlel.append(mytitle.encode('ascii', 'ignore'))
                attrl.append(myattributes.encode('ascii', 'ignore'))
                imglinkl.append(myimglink.encode('ascii', 'ignore'))
                messagel.append(mymessage.encode('ascii', 'ignore'))
            
            
                
        # save data of this page
        rows = len(pidl)
        print "==================== >>> Added # of rows = ", rows

        alist = []

        for i in range(rows):
            alist.append([pidl[i], posttimel[i], cityl[i], statel[i], makel[i], modell[i], pricel[i], \
                          yearl[i], milesl[i], urll[i], titlel[i], attrl[i], imglinkl[i], messagel[i] ])

        with open(csvfilename, "a") as csvfile:
            writer = csv.writer(csvfile)
            writer.writerows(alist)

        print 'Page #', pagecount, 'of', len(URLs)+startpage, ':', make, model, '@', city ,'finished !'
        pagecount += 1
        print '==================== >>> rest a bit ...'
        if pagecount >= pagemax:
            print '==================== >>> Pagemax reached. Done.'
            break
        y = random.uniform(10,45)
        time.sleep(y)
    return 0
              




mymake='ford'
mymodellist=['fiesta' ,'focus','fusion', 'taurus', 'escape', 'edge', 'explorer', 'mustang']

mymake='chevrolet'
mymodellist=[ 'cruze' ,'malibu','impala', 'equinox', 'traverse', 'tahoe', 'camaro', 'corvette']

mymake='chrysler'
mymodellist=['200', '300' ]

mymake='jeep'
mymodellist=['wrangler', 'compass', 'cherokee', 'renegade', 'patriot' ]

mymake='toyota'
mymodellist=['yaris' ,'corolla', 'camry', 'avalon', 'rav4', 'highlander', 'prius']

mymake='honda'
mymodellist=['fit' ,'civic','accord', 'crv', 'pilot', 'hrv', 'odssey']

mymake='nissan'
mymodellist=['versa', 'sentra' ,'altima','maxima', 'joke', 'rogue', 'murano', 'pathfinder']

mymake='mazda'
mymodellist=['3' ,'6','cx-3', 'cx-5', 'cx-9', 'mx-5']

mymake='mercedes'
mymodellist=['c300' ,'c350','e320', 'e350', 'e500', 'e550', 'e400', 's500','s550','cla250', 'cls550', 'ml320','ml350'  ]

mymake='bmw'
mymodellist=['x1','x2','x3', 'x5', 'x6','328i' ,'328xi','335i', '335xi', '525i', '528i', '525xi', '528xi','545i','545xi', '550i', '550xi','750i', '750xi','750li']

mymake='audi'
mymodellist=['a3' ,'a4','a5', 'a6', 'a7', 'a8', 's3', 'rs3','s7','rs7', 's8', 'q3','q5', 'q7','tt' ]

mymake='vw'
mymodellist=['jetta' ,'passat','cc', 'golf', 'tiguan', 'touareg', 'beetle' ]

mymake='hyundai'
mymodellist=['elantra' ,'sonata','azera', 'tucson', 'santafe', 'accent', 'veloster','genesis' ]

mymake='kia'
mymodellist=['optima' ,'soul','niro', 'sportage', 'sorento', 'cadenza', 'k900','rio','forte', 'sedona' ]



mymakelist = ['ford', 'chevrolet', 'chrysler', 'jeep', 'toyota', 'honda', 'nissan',  'mazda', 'mercedes', 'bmw', 'audi', 'vw', 'hyundai', 'kia']


mymodellistlist = [ ['fiesta' ,'focus','fusion', 'taurus', 'escape', 'edge', 'explorer', 'mustang', 'expedition'], ['spark', 'cruze' ,'malibu','impala', 'taurus', 'trax', 'equinox', 'traverse', 'tahoe', 'suburban', 'camaro', 'corvette'], ['200', '300'], ['wrangler', 'compass', 'cherokee', 'renegade', 'patriot'], ['yaris' ,'corolla','camry', 'avalon', 'rav4', 'highlander', 'prius', 'sienna'], ['fit' ,'civic','accord', 'crv', 'pilot', 'hrv', 'prius', 'odssey'], ['versa', 'sentra' ,'altima','maxima', 'joke', 'rogue', 'murano', 'pathfinder'], ['3' ,'6','cx-3', 'cx-5', 'cx-9', 'mx-5'], ['c300' ,'c350','e320', 'e350', 'e500', 'e550', 'e400', 's500','s550','cla250', 'cls550', 'ml320','ml350' ], ['x1','x2','x3', 'x5', 'x6','328i' ,'328xi','335i', '335xi', '525i', '528i', '525xi', '528xi','545i','545xi', '550i', '550xi','750i', '750xi','750li'], ['a3' ,'a4','a5', 'a6', 'a7', 'a8', 's3', 'rs3','s7','rs7', 's8', 'q3','q5', 'q7','tt' ], ['jetta' ,'passat','cc', 'golf', 'tiguan', 'touareg', 'beetle' ], ['elantra' ,'sonata','azera', 'tucson', 'santafe', 'accent', 'veloster','genesis' ], ['optima' ,'soul','niro', 'sportage', 'sorento', 'cadenza', 'k900','rio','forte', 'sedona' ]]


citystatelist = [ ('sfbay', 'CA'), ('losangeles', 'CA'),('newyork', 'NY'), ('seattle', 'WA'),('orangecounty', 'CA'), ('sandiego','CA'),  ('chicago', 'IL'), ('sacramento','CA'), ('portland', 'OR'), ('phoenix', 'AZ'), ('washingtondc', 'DC'), ('atlanta', 'GA'), ('miami', 'FL'), ('boston', 'MA'),('dallas', 'TX'), ('inlandempire', 'CA'), ('denver', 'CO'), ('minneapolis', 'MN'), ('austin', 'TX'),('houston', 'TX'), ('tampa', 'FL'), ('orlando', 'FL'), ('newjersey', 'NJ'), ('philadelphia', 'PA'), ('lasvegas', 'NV'), ('detroit','MI') , ('stlouis', 'MO') ] 



########################################################################################################



filterdate = '2017-01-01 00:00'
pagemax = 30
#startpage = 0 #1  #default =0


city = 'sfbay'
state = 'CA'

make = 'nissan'
model = 'sentra'

#citystatepagelist = [('sfbay', 'CA', 0), ('losangeles', 'CA', 0),('newyork', 'NY', 0), ('seattle', 'WA', 0),('orangecounty', 'CA', 0), ('sandiego','CA', 0), ('chicago', 'IL', 0), ('sacramento','CA', 0), ('portland', 'OR', 0), ('phoenix', 'AZ', 0), ('washingtondc', 'DC', 0), ('atlanta', 'GA', 0), ('miami', 'FL', 0), ('boston', 'MA', 0),('dallas', 'TX', 0) ,('inlandempire', 'CA', 0), ('denver', 'CO', 0), ('minneapolis', 'MN', 0), ('austin', 'TX', 0), ('houston', 'TX', 0), ('tampa', 'FL', 0), ('orlando', 'FL', 0), ('newjersey', 'NJ', 0), ('philadelphia', 'PA', 0), ('lasvegas', 'NV', 0), ('detroit','MI', 0) , ('stlouis', 'MO', 0) ]


citystatepagelist = [('sfbay', 'CA', 0), ('losangeles', 'CA', 0),('newyork', 'NY', 0), ('seattle', 'WA', 0),('orangecounty', 'CA', 0), ('sandiego','CA', 0), ('chicago', 'IL', 0), ('sacramento','CA', 0), ('portland', 'OR', 0), ('phoenix', 'AZ', 0), ('washingtondc', 'DC', 0), ('atlanta', 'GA', 0), ('miami', 'FL', 0), ('boston', 'MA', 0),('dallas', 'TX', 0) ,('inlandempire', 'CA', 0), ('denver', 'CO', 0), ('minneapolis', 'MN', 0), ('austin', 'TX', 0), ('houston', 'TX', 0), ('tampa', 'FL', 0), ('orlando', 'FL', 0), ('newjersey', 'NJ', 0), ('philadelphia', 'PA', 0), ('lasvegas', 'NV', 0), ('detroit','MI', 0) , ('stlouis', 'MO', 0) ]

# 


for city, state, startpage in citystatepagelist :

    url = 'https://'+ city + '.craigslist.org/search/cto'
    base_url = 'https://'+ city + '.craigslist.org'
    min_year ='1995'
    max_year ='2018'

    user_agent = 'Mozilla/5.0 (Macintosh; Intel Mac OS X 10.12; rv:53.0) Gecko/20100101 Firefox/53.0'
    user_agent = 'Mozilla/5.0 (Macintosh; Intel Mac OS X 10_12_4) AppleWebKit/603.1.30 (KHTML, like Gecko) Version/10.1 Safari/603.1.30'
    user_agent = 'Mozilla/5.0 (Macintosh; Intel Mac OS X 10_12_4) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/58.0.3029.81 Safari/537.36'

    head = {'User-Agent': user_agent} 

    # "auto_make_model": make+'%20'+model

    params={"query": make+'+'+model, "hasPic": 1, "min_auto_year": min_year, "max_auto_year": max_year}

    response = requests.get(url, params, headers=head)  # first search result page
    #print response.url
    #response.text[:1000] + "..."

    print '-----------------------------------------------------'
    print '  Getting data for ', make.title() , model.title(), 'at', city.title()
    print '-----------------------------------------------------'


    urls = get_search_page_urls(response)

    print 'We start from search page #', startpage, 'out of', len(urls)

    get_all_pages('CAR_PRICE_DATA.csv',urls[startpage:])

    print '==================================== FINISHED =========================================='







