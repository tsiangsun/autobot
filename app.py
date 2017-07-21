from flask import Flask, redirect, render_template, url_for, request
import requests
import simplejson as json
import numpy as np 
import math
import pandas as pd
import matplotlib
matplotlib.use('Agg')
# Force matplotlib to not use any Xwindows backend
import matplotlib.pyplot as plt
from matplotlib.backends.backend_agg import FigureCanvasAgg as FigureCanvas
from matplotlib.figure import Figure
import urllib
from io import BytesIO
import seaborn as sns
from bokeh.plotting import figure,ColumnDataSource
from bokeh.layouts import gridplot
from bokeh.embed import components 
from bokeh.models import HoverTool, TapTool, OpenURL, Range1d, FixedTicker, FuncTickFormatter
from sklearn import base
from sklearn import datasets, linear_model, utils, preprocessing
from sklearn.base import BaseEstimator, RegressorMixin, TransformerMixin
from sklearn import neighbors
from sklearn import ensemble
from sklearn.pipeline import Pipeline, FeatureUnion
from sklearn.feature_extraction import DictVectorizer
from datetime import datetime
import sys
import logging
import dill
import os

matplotlib.rcParams['savefig.dpi'] = 200


app = Flask(__name__)

app.logger.addHandler(logging.StreamHandler(sys.stdout))
app.logger.setLevel(logging.ERROR)

app.vars={}
app.vars['gmail']="Bumblebee#$Megatron53379801<tak>"

#========================================================================================
@app.route('/')
def main():
  return redirect('/index')

#========================================================================================
@app.route('/index', methods=['GET'])
def index():
    return render_template('index.html')
 

#========================================================================================   
@app.route('/value', methods=['POST'])
def carvalue():
    import numpy as np
    app.vars['make'] = request.form['make'].lower()
    app.vars['model'] = request.form['model'].lower()
    app.vars['year'] = int(request.form['year'])
    app.vars['miles'] = int(request.form['miles'])
    app.vars['city'] = request.form['city']
    make = app.vars['make']
    model = app.vars['model']
    city = app.vars['city']
    year = app.vars['year']
    miles = app.vars['miles']

    state_dict = {'sfbay': 'CA', 'losangeles': 'CA', 'newyork': 'NY', 'seattle': 'WA', 'orangecounty': 'CA', 'sandiego':'CA',
              'chicago': 'IL', 'sacramento':'CA', 'portland': 'OR', 'phoenix': 'AZ', 'washingtondc': 'DC', 'atlanta': 'GA',
              'miami': 'FL', 'boston': 'MA', 'dallas': 'TX', 'inlandempire': 'CA', 'denver': 'CO', 'minneapolis': 'MN', 
              'austin': 'TX', 'houston': 'TX', 'tampa': 'FL', 'orlando': 'FL', 'newjersey': 'NJ', 'philadelphia': 'PA', 
              'lasvegas': 'NV', 'detroit': 'MI', 'stlouis': 'MO'}

    if make =='' or model=='' or request.form['year']=='' or request.form['miles']=='':
        return redirect('/error')
    
    ensemble_pipeline = Pipeline([ 
        ('dictflat', DictFlattener() ),
        ('vector', DictVectorizer(sparse=False)),
        ('ensemble', EnsembleTransformer(
                linear_model.LinearRegression(),
                (neighbors.KNeighborsRegressor(n_neighbors=10),
                 ensemble.RandomForestRegressor(min_samples_leaf=5)))),
        ('blend', linear_model.LinearRegression())
       ])


    X_mycar = []
    state = state_dict[city]
    dic = dict()
    md = datetime.today()
    dic['POSTDAY'] = md.timetuple().tm_yday
    dic['CITY'] = city
    dic['STATE'] = state
    dic['MAKE'] = make
    dic['MODEL'] = model
    dic['YEAR'] = int(year)
    dic['MILES'] = int(miles)
    X_mycar.append(dic)


    myfile='CAR_PRICE_DATA_1.csv'
    app.df = pd.read_csv(myfile) 


    if not os.path.isfile('ensemble_pipeline_%s_%s.dill' %(make, model)) :
        return redirect('/error')
    
    ensemble_pipeline = dill.load(open('ensemble_pipeline_%s_%s.dill' %(make, model), 'r'))
    
    y_mycar = ensemble_pipeline.predict(X_mycar)
    predprice = y_mycar[0]
    predprice = int(round(predprice, -1))

    p = plot_price_distr()
    script, div = components(p)
    
    return render_template('value.html', script=script, div=div, year=year, make=make.title(), model=model.title(), predprice=predprice)




#========================================================================================   
@app.route('/graph', methods=['POST'])
def graph():
    
    make = app.vars['make']
    model = app.vars['model']
    city = app.vars['city']
    budget = app.vars['budget']
    if budget < 100 or budget > 40000:
        return redirect('/error')
    '''
    if request.form.get('year_pri'):
        app.vars['year_pri'] = 1
    elif request.form.get('miles_pri'):
        app.vars['miles_pri'] = 1
    elif request.form.get('budget_pri'):
        app.vars['budget_pri'] = 1
    
    '''
    
    #--------------------------------------------
    myfile='CAR_PRICE_DATA_1.csv'
    app.df = pd.read_csv(myfile) #, low_memory=False

    #p1 = figure(title='Post traffic %s' % city, x_axis_label='Posts', y_axis_label='Weekday', tools= TOOLS)
    #p1.vbar(x=range(7), top=count, bottom = 0,  width=0.5, color="purple", legend='# of Posts')
    #show(p)

    #p4.line(x=range(7), y=count, line_width=3, line_color="purple", legend='# of Posts')
    p1,p2,p3,p4 = plot_traffic()
    grid = gridplot([p1,p2,p3,p4], ncols=2, plot_width=300, plot_height=300)
    script, div = components(grid)

    p = plot_price_distr()
    #grid = gridplot([p3,p4], ncols=2, plot_width=400, plot_height=400)
    script1, div1 = components(p)

    violin = violin_mpl(model)
    #pd.set_option('display.max_colwidth',54)
    #pd.set_option('display.max_seq_items',200)
    #dfresult = app.df.head()[['POSTTIME','CITY','STATE','PRICE','YEAR','MILES','TITLE','URL']]
    #df_html = dfresult.to_html(index=False)#line_width=60, col_space=70

    return render_template('graph.html', script=script, div=div, script1=script1, div1=div1, img_data=violin)
    #datatable=df_html)



#========================================================================================   
@app.route('/result', methods=['POST'])
def showresult():
    app.vars['make'] = request.form['make'].lower()
    app.vars['model'] = request.form['model'].lower()
    app.vars['budget'] = int(request.form['budget'])
    app.vars['city'] = request.form['city']
    if app.vars['make'] =='' or app.vars['model']=='' or request.form['budget']=='':
        return redirect('/error')

    make = app.vars['make'] 
    model = app.vars['model'] 
    budget = app.vars['budget'] 
    city = app.vars['city']

    #--- loading Machine learned data for this model ---
    #myfile='/Users/xiangs/github/cardeal/CAR_PRICE_DATA_1.csv'
    app.df2 = pd.read_csv('CAR_PRICE_DATA_%s.csv' % model)
    overpricefactor = 1.1
    lower = 0.8
    upper = 1.2
    dfr = app.df2[ app.df2.PRICE < app.df2.PRICEPRED*overpricefactor]
    df3 = dfr[ (dfr.PRICE > budget*lower) & (dfr.PRICE < budget*upper) & (dfr.CITY==city)]
    dfi = df3.sort_values(by=['YEAR','POSTTIME'],ascending=[False, False])[
      ['POSTTIME','YEAR','MILES','TITLE','CITY', 'STATE','PRICE','PRICEPRED','IMGLINK','URL']]

    #dfi['IMAGE'] = dfi['IMGLINK'].apply(lambda x: '<img src="{}" />'.format(x) if x else '') #
    dfresult = dfi[['POSTTIME', 'CITY', 'STATE','YEAR','MILES','TITLE','PRICE','PRICEPRED','URL','IMGLINK']]  #'IMAGE',
    dfresult['PRICEPRED'] = dfresult.PRICEPRED.astype(int)

    #p = plot_price_distr()
    p = plot_result_price_distr(dfresult)
    #grid = gridplot([p3,p4], ncols=2, plot_width=400, plot_height=400)
    script, div = components(p)

    pd.set_option('display.max_colwidth',-1)
    pd.set_option('display.max_seq_items',200)

    dfresult = dfresult[['POSTTIME', 'CITY', 'STATE','YEAR','MILES','TITLE','PRICE','PRICEPRED','URL']]  #'IMAGE',
    #dfresult = app.df.head()[['POSTTIME','CITY','STATE','PRICE','YEAR','MILES','TITLE','URL']]
    app.df_html = dfresult.to_html(index=False)#line_width=60, col_space=70
    #from IPython.display import HTML
    #df_html = HTML(df_html)

    return render_template('result.html', script=script, div=div, datatable=app.df_html, make=make.title(), model=model.title())



@app.route('/method', methods=['GET','POST'])
def algorithm():

    return render_template('method.html')


@app.route('/contact', methods=['POST'])
def emailadmin():
    import smtplib
    from email.mime.text import MIMEText
    from email.mime.multipart import MIMEMultipart

    email = request.form['email']
    name = request.form['name']
    comments = request.form['comments']

    sender = 'myautobotapp@gmail.com'
    receivers = [ 'myautobotapp@gmail.com', email ] #'xiangscode@gmail.com'

    msg = MIMEMultipart('alternative')
    msg['Subject'] = 'Contact Autobot from %s' % name
    msg['From'] = sender
    msg['To'] = ','.join(receivers)

    # Create the body of the message (a plain-text and an HTML version).
    html = "something"

    # Record the MIME types of both parts - text/plain and text/html.
    part1 = MIMEText(comments, 'plain')
    #part2 = MIMEText(html, 'html')

    # Attach parts into message container.
    msg.attach(part1)
    #msg.attach(part2)
    server = smtplib.SMTP('smtp.gmail.com', 587)
    server.starttls()
    server.login("myautobotapp@gmail.com", app.vars['gmail'])

    server.sendmail(sender, receivers, msg.as_string())
    server.quit()

    return render_template('contact.html', name=name)



#========================================================================================
@app.route('/alert', methods=['POST'])
def emailsent():

    import smtplib
    from email.mime.text import MIMEText
    from email.mime.multipart import MIMEMultipart

    email = request.form['email']
    name = request.form['name']
    #comments = request.form['comments']

    sender = 'myautobotapp@gmail.com'
    receivers = [ 'myautobotapp@gmail.com', email ] #'xiangscode@gmail.com'

    msg = MIMEMultipart('alternative')
    msg['Subject'] = 'Autobot: Car Deals for %s' % name
    msg['From'] = sender
    msg['To'] = ','.join(receivers)

    # Create the body of the message (a plain-text and an HTML version).
    html = "something"

    # Record the MIME types of both parts - text/plain and text/html.
    #part1 = MIMEText(comments, 'plain')
    part2 = MIMEText(app.df_html, 'html')

    # Attach parts into message container.
    #msg.attach(part1)
    msg.attach(part2)
    server = smtplib.SMTP('smtp.gmail.com', 587)
    server.starttls()
    server.login("myautobotapp@gmail.com", app.vars['gmail'])

    server.sendmail(sender, receivers, msg.as_string())
    server.quit()


    return render_template('alert.html', name=name, email=email)


#========================================================================================
def plot_traffic():
    rows, cols = app.df.shape
    TOOLS="pan,wheel_zoom,box_zoom,tap,box_select,lasso_select,reset,save"
    #------------------- figure 1 ---------------
    hour = []
    for i in range(rows):
        mytime = app.df.ix[i, 'POSTTIME']
        #print mytime
        mt = datetime.strptime( mytime,'%Y-%m-%d %H:%M' ) 
        #datetime.datetime.strptime(mytime, '%H:%M').time()
        hour.append(mt.time().hour)
    count = map(float, np.zeros(24)) 
    for i in range(rows):
        count[hour[i]] += 1.0

    for n in range(24):
        count[n] /= rows

    p1 = figure(title='Post traffic (day)', x_axis_label='Hour', y_axis_label='# of Posts', tools= TOOLS)
    ind = range(0,24)
    p1.vbar(x=ind, top=count, bottom = 0,  width=0.5, color="blue", alpha=0.5)
    p1.xaxis.ticker = FixedTicker(ticks=range(0,24,3))

    xtickl = ['0AM','3AM','6AM', '9AM', '12PM', '3PM', '6PM', '9PM']
    label_dict = {}
    for i, s in enumerate(xtickl):
        label_dict[i*3] = s
    p1.xaxis.formatter = FuncTickFormatter(code="""
        var labels = %s;
        return labels[tick];
    """ % label_dict)

    #------------------- figure 2 ---------------
    weekday = []
    for i in range(rows):
        mydate = app.df.ix[i,'POSTTIME']
        md = datetime.strptime( mydate,'%Y-%m-%d %H:%M' ) 
        weekday.append(md.isoweekday())  #Monday 1 ...
    count = [0,0,0,0,0,0,0] #1-7 Mon-Sun
    for i in range(rows):
        count[weekday[i]-1] += 1
    for n in range(7):
        count[n] = float(count[n])/rows

    p2 = figure(title='Post traffic (week)', x_axis_label='Day', y_axis_label='# of Posts', tools= TOOLS)
    p2.vbar(x=range(7), top=count, bottom = 0,  width=0.5, color="blue", alpha=0.5)
    p2.xaxis.ticker = FixedTicker(ticks=range(7))

    xtickl = ['Mon', 'Tue', 'Wed', 'Thu', 'Fri', 'Sat', 'Sun']
    label_dict = {}
    for i, s in enumerate(xtickl):
        label_dict[i] = s
    p2.xaxis.formatter = FuncTickFormatter(code="""
        var labels = %s;
        return labels[tick];
    """ % label_dict)

    #------------------- figure 3 ---------------
    accum = np.zeros(24) #0-23 hours
    count = np.zeros(24)
    for i in range(rows):
        mytime = app.df.ix[i, 'POSTTIME']
        mt = datetime.strptime( mytime,'%Y-%m-%d %H:%M' ) 
        accum[mt.hour] += app.df.ix[i, 'PRICE']
        count[mt.hour] += 1
    for n in range(24):
        accum[n] /= count[n]

    p3 = figure(title='Average posted price (day)', x_axis_label='Hour', y_axis_label='Price ($)', tools= TOOLS)
    ind = range(0,24)
    p3.vbar(x=ind, top=accum, bottom = 0,  width=0.5, color="red", alpha=0.5)
    p3.xaxis.ticker = FixedTicker(ticks=range(0,24,3))

    xtickl = ['0AM','3AM','6AM', '9AM', '12PM', '3PM', '6PM', '9PM']
    label_dict = {}
    for i, s in enumerate(xtickl):
        label_dict[i*3] = s
    p3.xaxis.formatter = FuncTickFormatter(code="""
        var labels = %s;
        return labels[tick];
    """ % label_dict)

    #------------------- figure 4 ---------------
    accum = np.zeros(7) #1-7 Mon-Sun
    count = np.zeros(7)
    for i in range(rows):
        mydate = app.df.ix[i,'POSTTIME']
        md = datetime.strptime( mydate,'%Y-%m-%d %H:%M' ) 
        accum[md.weekday()] += app.df.ix[i,'PRICE']
        count[md.weekday()] += 1
        
    for n in range(7):
        accum[n] /= count[n]

    p4 = figure(title='Average posted price (week)', x_axis_label='Day', y_axis_label='Price ($)', tools= TOOLS)
    p4.vbar(x=range(7), top=accum, bottom = 0,  width=0.5, color="red", alpha=0.5)
    p4.xaxis.ticker = FixedTicker(ticks=range(7))

    xtickl = ['Mon', 'Tue', 'Wed', 'Thu', 'Fri', 'Sat', 'Sun']
    label_dict = {}
    for i, s in enumerate(xtickl):
        label_dict[i] = s
    p4.xaxis.formatter = FuncTickFormatter(code="""
        var labels = %s;
        return labels[tick];
    """ % label_dict)
    #---------------------------

    return p1,p2,p3,p4



#========================================================================================
def violin_mpl(model):

    df2 = pd.read_csv('CAR_PRICE_DATA_%s.csv' % model)

    sns.set_style('darkgrid')
    fig, ax = plt.subplots()
    fig.set_size_inches(10, 5)
    sns.violinplot(x='CITY',y='PRICE', data=df2, ax=ax,  bw=0.2, width=0.98, linewidth=1)   
    plt.xticks(rotation=70)
    plt.tight_layout()

    #--------
    canvas=FigureCanvas(fig)
    #png_output = StringIO.StringIO()
    png_output=BytesIO()
    canvas.print_png(png_output)
    png_output = png_output.getvalue().encode("base64")

    return urllib.quote(png_output.rstrip('\n'))
    #return render_template("test.html", img_data=urllib.quote(png_output.rstrip('\n')))





#========================================================================================
def plot_price_distr():
    from bokeh.plotting import ColumnDataSource
    from bokeh.models import (HoverTool, PanTool, ResetTool, SaveTool, CrosshairTool,
                TapTool, WheelZoomTool, BoxSelectTool, BoxZoomTool, LassoSelectTool)

    app.df1 = app.df[app.df.MODEL == app.vars['model']] 
    rows, cols = app.df1.shape
    price = app.df1.ix[:,'PRICE'].values
    milek = app.df1.ix[:,'MILES'].values/1000.0 # k miles
    year  = app.df1.ix[:,'YEAR'].values
    imglinks = app.df1.ix[:,'IMGLINK'].values
    urls = app.df1.ix[:,'URL'].values

    myyear  = year.tolist()
    mymile  = milek.tolist()
    myprice = price.tolist()
    myimg   = imglinks.tolist()
    myurls = urls.tolist()

    #radii = map(math.sqrt, [a/160000.0 for a in price]) #price proportion to area
    #myrad = radii
    #mycolor = ["#%02x%02x%02x" % (int(r), 0, 200) for r in price/ 70.0]

    radii = price / 40000.0  #price proportion to radius
    myrad   = radii.tolist()
    mycolor = [ "#%02x%02x%02x" % (int(r), 0, 200) for r in radii*550 ]
    ####(int(r), int(g), 150) for r, g in zip(50+5*(x-1995), 30+2*y)
    source = ColumnDataSource(data=dict( x=myyear, y=mymile, desc=myprice,
                imgs = myimg, rad = myrad, c = mycolor, url=myurls))

    hover = HoverTool(
            tooltips="""
            <div>
                <div>
                    <img
                        src="@imgs" height="150" width="180" alt="@imgs" 
                        style="float: left; margin: 0px 15px 15px 0px;"
                        border="2"
                    ></img>
                </div>
                <div>
                    <span style="font-size: 17px; font-weight: bold;">$@desc</span>
                    <span style="font-size: 15px; color: #966;">[$index]</span>
                </div>
                <div>
                    <span style="font-size: 15px;">(year,miles)=</span>
                    <span style="font-size: 15px; color: #696;">(@x{int}, @y{1.1}k)</span>
                </div>
            </div>
            """
        )

    p = figure(plot_width=600, plot_height=600, tools=[hover,PanTool(),ResetTool(), SaveTool(), 
        TapTool(),WheelZoomTool(), BoxSelectTool(), BoxZoomTool(), LassoSelectTool(),CrosshairTool()],
        title="Price is proportional to symbol size")
    url = "@url"
    taptool = p.select(type=TapTool)
    taptool.callback = OpenURL(url=url)

    p.scatter(x='x', y='y', radius='rad', fill_color='c', fill_alpha=0.6, line_color=None, source=source) 
    p.xaxis[0].axis_label = 'Year'
    p.yaxis[0].axis_label = 'Mileage (k)'

    return p

    '''
    #========================================================================================
    price = app.df1.ix[:,'PRICE'].values
    mile = app.df1.ix[:,'MILES'].values
    year = app.df1.ix[:,'YEAR'].values
    x = year
    y = mile/1000.0
    z = price
    radii = z/40000.0
    colors = ["#%02x%02x%02x" % (int(r), 0, 200) for r in radii*500]
    #(int(r), int(g), 150) for r, g in zip(50+5*(x-1995), 30+2*y)
    TOOLS="pan,box_select,poly_select,lasso_select,wheel_zoom,box_zoom,reset,tap,hover,save"  #crosshair,redo,undo
    p = figure(title="Price distribution of %s %s" % (app.vars['make'].title(), app.vars['model'].title()), plot_width=800, plot_height=800, tools=TOOLS)
    p.scatter(x, y, radius=radii, fill_color=colors, fill_alpha=0.4, line_color=None)
    #output_file("color_scatter.html", title="color_scatter.py example")
    p.xaxis[0].axis_label = 'Year'
    p.yaxis[0].axis_label = 'Mileage (k)'
    #========================================================================================
    '''

    

#========================================================================================
def plot_result_price_distr(df):
    from bokeh.plotting import ColumnDataSource
    from bokeh.models import (HoverTool, PanTool, ResetTool, SaveTool, CrosshairTool,
                TapTool, WheelZoomTool, BoxSelectTool, BoxZoomTool, LassoSelectTool)

    df1 = df #df[df.MODEL == app.vars['model']] 
    rows, cols = df1.shape
    price = df1.ix[:,'PRICE'].values
    milek = df1.ix[:,'MILES'].values/1000.0 # k miles
    year  = df1.ix[:,'YEAR'].values
    imglinks = df1.ix[:,'IMGLINK'].values
    urls = df1.ix[:,'URL'].values

    myyear  = year.tolist()
    mymile  = milek.tolist()
    myprice = price.tolist()
    myimg   = imglinks.tolist()
    myurls = urls.tolist()

    #radii = map(math.sqrt, [a/160000.0 for a in price]) #price proportion to area
    #myrad = radii
    #mycolor = ["#%02x%02x%02x" % (int(r), 0, 200) for r in price/ 70.0]

    radii = price / 40000.0  #price proportion to radius
    myrad   = radii.tolist()
    mycolor = [ "#%02x%02x%02x" % (int(r), 0, 200) for r in radii*550 ]
    ####(int(r), int(g), 150) for r, g in zip(50+5*(x-1995), 30+2*y)
    source = ColumnDataSource(data=dict( x=myyear, y=mymile, desc=myprice,
                imgs = myimg, rad = myrad, c = mycolor, url=myurls ))

    hover = HoverTool(
            tooltips="""
            <div>
                <div>
                    <img
                        src="@imgs" height="150" width="180" alt="@imgs" 
                        style="float: left; margin: 0px 15px 15px 0px;"
                        border="2"
                    ></img>
                </div>
                <div>
                    <span style="font-size: 17px; font-weight: bold;">$@desc</span>
                    <span style="font-size: 15px; color: #966;">[$index]</span>
                </div>
                <div>
                    <span style="font-size: 15px;">(year,miles)=</span>
                    <span style="font-size: 15px; color: #696;">(@x{int}, @y{1.1}k)</span>
                </div>
            </div>
            """
        )

    p = figure(plot_width=600, plot_height=600, tools=[hover,PanTool(),ResetTool(), SaveTool(), 
        TapTool(),WheelZoomTool(), BoxSelectTool(), BoxZoomTool(), LassoSelectTool(),CrosshairTool()],
        title="Price is proportional to symbol size")
    url = "@url"
    taptool = p.select(type=TapTool)
    taptool.callback = OpenURL(url=url)

    p.scatter(x='x', y='y', radius='rad', fill_color='c', fill_alpha=0.6, line_color=None, source=source) 
    p.xaxis[0].axis_label = 'Year'
    p.yaxis[0].axis_label = 'Mileage (k)'

    return p




#========================================================================================
class DictFlattener(base.BaseEstimator, base.TransformerMixin):
    
    def fit(self, X, y=None):
        return self
    
    def transform(self, X):
        import numpy as np
        l = []
        for x in X:
            dic = dict()
            for a in x: 
                #x: dict like {key1:value1, key2: {kkey3: value3, kkey4: value4}}
                #a: entry like key1:value1 or key2: {kkey3: value3, kkey4: value4}
                #dic[a] = 1
                value1 = x[a]
                #print type(value1)

                if isinstance(value1, bool):
                        if value1 == True:
                            dic[a] = int(1)
                        else:
                            dic[a] = int(0)
                        continue

                if isinstance(value1, dict): 
                    for b in value1:
                        value2 = value1[b]
                        #print type(value2)
                        if isinstance(value2, bool):
                            if value2 == True:
                                dic[a+'_'+b] = int(1)
                            else:
                                dic[a+'_'+b] = int(0)
                            continue

                        if isinstance(value2, dict): 
                            print 'Error: more than 2 layers of dict !'
                        else:
                            if isinstance(value2, str): #type(value2)==type(''):
                                dic[a+'_'+b+'_'+value2] = 1
                            if isinstance(value2, (int, float)):
                                dic[a+'_'+b] = value2

                else:
                    if isinstance(value1, str): #(value1)==type(''):
                        dic[a+'_'+value1] = 1
                    if isinstance(value1, (int, float)):
                        dic[a] = value1
            l.append(dic)
        return l



class EnsembleTransformer(BaseEstimator, TransformerMixin):
    
    def __init__(self, base_estimator, residual_estimators):
        self.base_estimator = base_estimator
        self.residual_estimators = residual_estimators
    
    def fit(self, X, y):
        import numpy as np 
        self.base_estimator.fit(X, y)
        y_err = y - self.base_estimator.predict(X)
        for est in self.residual_estimators:
            est.fit(X, y_err)
        return self
    
    def transform(self, X):
        import numpy as np 
        all_ests = [self.base_estimator] + list(self.residual_estimators)
        return np.array([est.predict(X) for est in all_ests]).T


def df_Dict_Transform(df1):
    l = []
    rows_df , cols_df = df1.shape
    for i in range(rows_df):
        row = df1.ix[i]
        dic = dict()
        posttime = row['POSTTIME']
        md = datetime.strptime( posttime,'%Y-%m-%d %H:%M' ) 
        dic['POSTDAY'] = md.timetuple().tm_yday
        dic['CITY'] = row['CITY']
        dic['STATE'] = row['STATE']
        dic['MAKE'] = row['MAKE']
        dic['MODEL'] = row['MODEL']
        dic['YEAR'] = int(row['YEAR'])
        dic['MILES'] = int(row['MILES'])
        dic['ATTR'] = get_attr_dict(row['ATTR'],row['TITLE'],row['MESSAGE'])
        #print dic
        l.append(dic)    
    return l



def get_attr_dict(attr_str, title_str, message_str):
    dic = dict()
    text = title_str + '\n'+ message_str
    
    m = re.search(r'cylinders: (\d{1,2}) cylinders[#|$]' , attr_str)
    if m :  dic['CYLINDERS'] = m.group(1)
        
    m = re.search(r'fuel: (\w+)[#|$]' , attr_str)
    if m :  dic['FUEL'] = m.group(1)
        
    m = re.search(r'paint color: (\w+)[#|$]' , attr_str)
    if m :  dic['COLOR'] = m.group(1)
        
    m = re.search(r'condition: (\w+)[#|$]' , attr_str)
    if m :  dic['CONDITION'] = m.group(1)
        
    m = re.search(r'title status: (\w+)[#|$]' , attr_str)
    if m :  dic['TITLE STATUS'] = m.group(1)
        
    m = re.search(r'transmission: (\w+)[#|$]' , attr_str)
    if m :  dic['TRANSMISSION'] = m.group(1)
        
    m = re.search(r'type: (\w+)[#|$]' , attr_str)
    if m :  dic['TYPE'] = m.group(1)
        
    m = re.search(r'size: (\w+)[#|$]' , attr_str)
    if m :  dic['SIZE'] = m.group(1)
        
    m = re.search(r'drive: (\w+)[#|$]' , attr_str)
    if m :  dic['DRIVE'] = m.group(1)
        
    return dic



#========================================================================================
@app.route("/test")
def simple_mpl():
    import urllib
    import datetime
    #import StringIO
    from io import BytesIO
    import random

    from matplotlib.backends.backend_agg import FigureCanvasAgg as FigureCanvas
    from matplotlib.figure import Figure
    from matplotlib.dates import DateFormatter

    
    '''
    # working example : random
    fig=Figure(figsize=(6,4),dpi=218)
    ax=fig.add_subplot(111)
    x=[]
    y=[]
    now=datetime.datetime.now()
    delta=datetime.timedelta(days=1)
    for i in range(10):
        x.append(now)
        now+=delta
        y.append(random.randint(0, 1000))
    ax.plot_date(x, y, '-')
    ax.xaxis.set_major_formatter(DateFormatter('%Y-%m-%d'))
    fig.autofmt_xdate()
    '''
    df2 = pd.read_csv('CAR_PRICE_DATA_camry.csv')

    sns.set_style('darkgrid')
    fig, ax = plt.subplots()
    fig.set_size_inches(10, 5)
    sns.violinplot(x='CITY',y='PRICE', data=df2, ax=ax,  bw=0.2, width=0.98, linewidth=1)   
    plt.xticks(rotation=70)
    plt.tight_layout()

    #--------
    canvas=FigureCanvas(fig)
    #png_output = StringIO.StringIO()
    png_output=BytesIO()
    canvas.print_png(png_output)
    png_output = png_output.getvalue().encode("base64")

    return render_template("test.html", img_data=urllib.quote(png_output.rstrip('\n')))




#========================================================================================
@app.route('/error')
def error():
    return render_template('error.html', model=app.vars['model'])

#========================================================================================
if __name__ == '__main__':
    #app.debug = True
    app.run(port=33507)



