from django.shortcuts import render, redirect
from django.urls import reverse
import yfinance as yf
import datetime
from django.contrib import messages
from django.http import JsonResponse
from django.contrib.auth import authenticate, login, logout
import random
from django.contrib.auth.decorators import login_required
from .decorators import unauthenticated_user
from .forms import UserForm


#tickers to randomize the index page
tickers = ['AAPL','AMZN','MSFT','GOOGL','AMD','META','NFLX','IBM','NVDA','INTC']

@login_required(login_url= 'login')
def index(request):
    #redirects user to a stock page corresponding to one of the tickers above
    return redirect(reverse("stock",kwargs={'pk':random.choice(tickers)}))

@login_required(login_url= 'login')
def stock(request,pk):
    #gets object Ticker from the ticker given in the url and gets the info and last close for it.
    stock = yf.Ticker(pk)
    close = "{:.2f}".format(stock.history(period='5d').Close.values.tolist()[0])
    #sends the current day and day corresponding to a week ago to load the plot (could be done in js)
    today = datetime.date.today().strftime('%m-%d-%Y')
    week_ago = (datetime.date.today() - datetime.timedelta(days=21)).strftime('%m-%d-%Y')
    context = { 'ticker': pk,'today': today, 'week_ago': week_ago, 'stock':stock.info, 'close':close}
    return render(request,'stocks.html',context)

@login_required(login_url= 'login')
def loadstock(request):
    #function to handle ajax request for the plot of price history
    pk = request.GET.get("ticker", None)
    start = datetime.datetime.strptime(request.GET.get("start", None), '%m-%d-%Y')
    finish = datetime.datetime.strptime(request.GET.get("finish", None), '%m-%d-%Y')
    if(start > finish):
        return JsonResponse({'error':'Finish date is sooner than start date'},status=400) 
    stock = yf.Ticker(pk)
    if stock is None:
        return JsonResponse({'error':'Ticker not found'},status=404)
    data = stock.history(start = start.strftime('%Y-%m-%d'), end = finish.strftime('%Y-%m-%d'))
    index = list(data.index.strftime('%Y-%m-%d %H'))
    print(index)
    close = data.Close.values.tolist()
    print(close)
    open = data.Open.values.tolist()
    high = data.High.values.tolist()
    low = data.Low.values.tolist()
    return JsonResponse( {pk :{'index': index, 'close': close, 'open': open,'high': high, 'low': low}}, status = 200)

@login_required(login_url= 'login')
def search(request):
    #redirects user to a stock page corresponding to the query given
    query = request.GET.get("query")
    return redirect(reverse("stock",kwargs={'pk':str(query)}))

@unauthenticated_user
def loginPage(request):
    if request.method == 'POST':
        username= request.POST.get('username')
        password= request.POST.get('password')
        user= authenticate(request, username=username, password=password)
        #checks if user and password match exist in db
        if user == None:
             messages.error(request, 'Login information incorrect')
             return redirect('login')
        login(request,user)
        return redirect('index')
    context = {}
    return render(request, 'login.html',context)

@login_required(login_url='login_view')
def logoutUser(request):
    logout(request)
    return redirect('login')

@unauthenticated_user
def registerPage(request):
    form = UserForm()
    if request.method == 'POST':
        form = UserForm(request.POST)
        if form.is_valid():
            user = form.save()
            messages.success(request, 'Your account was created')
            return redirect('login')
        messages.error(request, 'Your password or user is invalid')
        return redirect('register')
    context = {'form' : form}
    return render(request, 'register.html', context)







from .forms import FeedbackForm
from .models import Feedback  # Ensure you import your Feedback model

def feedback_view(request):
    if request.method == 'POST':
        form = FeedbackForm(request.POST)
        if form.is_valid():
            # Create Feedback instance from form data manually
            Feedback.objects.create(
                name=form.cleaned_data['name'],
                email=form.cleaned_data['email'],
                satisfaction=form.cleaned_data['satisfaction'],
                accuracy=form.cleaned_data['accuracy'],
                improvements=form.cleaned_data['improvements'],
                additional_feedback=form.cleaned_data['additional_feedback']
            )
            return redirect('index')
    else:
        form = FeedbackForm()
    
    return render(request, 'feedback_form.html', {'form': form})



from django.shortcuts import render
import yfinance as yf
import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
import tensorflow as tf
from keras import Sequential
from keras.layers import Dense, LSTM


from plotly.offline import plot
import plotly.graph_objs as go

from django.shortcuts import render
import yfinance as yf
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from keras.models import Sequential
from keras.layers import Dense, LSTM
from plotly.offline import plot
import plotly.graph_objs as go
import pandas as pd
from .forms import StockPredictionForm
from django.shortcuts import render
import yfinance as yf
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from keras.models import Sequential
from keras.layers import Dense, LSTM
from plotly.offline import plot
import plotly.graph_objs as go
import pandas as pd
from .forms import StockPredictionForm
from django.shortcuts import render
import yfinance as yf
import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from keras.models import Sequential
from keras.layers import LSTM, Dense
from plotly.offline import plot
import plotly.graph_objs as go

def predict_stock(request):
    if request.method == "POST":
        form = StockPredictionForm(request.POST)
        if form.is_valid():
            ticker_value = form.cleaned_data['ticker']
            number_of_days = int(form.cleaned_data['number_of_days'])

            # Download one year of stock data
            df = yf.download(ticker_value, period='1y', interval='1d')
            
            # Check if DataFrame is empty or if not enough data
            if df.empty or len(df['Adj Close']) < 100:  # Check for at least 100 days of data for training
                return render(request, 'error.html', {'error': 'Not enough historical data to make a prediction.'})
            
            # Preprocess data
            scaler = MinMaxScaler(feature_range=(0, 1))
            scaled_data = scaler.fit_transform(df['Adj Close'].values.reshape(-1, 1))

            # Prepare training data
            prediction_days = 60
            x_train, y_train = [], []
            for i in range(prediction_days, len(scaled_data)):
                x_train.append(scaled_data[i-prediction_days:i, 0])
                y_train.append(scaled_data[i, 0])

            x_train, y_train = np.array(x_train), np.array(y_train)
            x_train = np.reshape(x_train, (x_train.shape[0], x_train.shape[1], 1))

            # Build and train the model
            model = Sequential([
                LSTM(units=50, return_sequences=True, input_shape=(x_train.shape[1], 1)),
                LSTM(units=50),
                Dense(units=25),
                Dense(units=1)
            ])
            model.compile(optimizer='adam', loss='mean_squared_error')
            model.fit(x_train, y_train, epochs=25, batch_size=32)

            # Predict future prices
            test_data = scaled_data[-prediction_days:].tolist()
            predicted_prices = []
            for _ in range(number_of_days):
                x_test = np.array([test_data[-prediction_days:]])
                x_test = x_test.reshape((1, prediction_days, 1))
                output = model.predict(x_test)[0][0]
                predicted_prices.append(output)
                test_data.append([output])

            predicted_prices = scaler.inverse_transform(np.array(predicted_prices).reshape(-1, 1))

            # Create dates for plotting
            last_date = df.index[-1]
            prediction_dates = pd.date_range(start=last_date + pd.Timedelta(days=1), periods=number_of_days, freq='D')
            historical_dates = df.index[-60:]  # Last 60 days of actual data

            # Plotting historical and predicted data as candlestick
            fig = go.Figure(data=[go.Candlestick(x=historical_dates,
                                                 open=df['Open'][-60:], high=df['High'][-60:],
                                                 low=df['Low'][-60:], close=df['Adj Close'][-60:],
                                                 name='Actual', increasing_line_color='green', decreasing_line_color='red'),
                                  go.Candlestick(x=prediction_dates,
                                                 open=predicted_prices.flatten(), high=predicted_prices.flatten(),
                                                 low=predicted_prices.flatten(), close=predicted_prices.flatten(),
                                                 name='Predicted', increasing_line_color='blue', decreasing_line_color='orange')])
            plot_div = plot(fig, output_type='div', include_plotlyjs=False)

            return render(request, 'result.html', {'plot_div': plot_div, 'ticker': ticker_value})
    else:
        form = StockPredictionForm()

    return render(request, 'predict_stock.html', {'form': form})




import yfinance as yf
from django.shortcuts import render
from .forms import FundamentalDataForm

def get_fundamental_data(request):
    if request.method == 'POST':
        form = FundamentalDataForm(request.POST)
        if form.is_valid():
            ticker = form.cleaned_data['ticker']
            stock = yf.Ticker(ticker)
            
            # Expanding the set of fundamental data retrieved
            fundamentals = {
                'market_cap': stock.info.get('marketCap'),
                'pe_ratio': stock.info.get('trailingPE'),
                'dividend_yield': stock.info.get('dividendYield'),
                'sector': stock.info.get('sector'),
                'earnings_growth': stock.info.get('earningsQuarterlyGrowth'),
                'book_value': stock.info.get('bookValue'),
                'profit_margins': stock.info.get('profitMargins'),
                'return_on_equity': stock.info.get('returnOnEquity'),
                'revenue_growth': stock.info.get('revenueGrowth'),
                'operating_margins': stock.info.get('operatingMargins'),
                'enterprise_to_revenue': stock.info.get('enterpriseToRevenue'),
                'enterprise_to_ebitda': stock.info.get('enterpriseToEbitda'),
                '52_week_change': stock.info.get('52WeekChange'),
                'current_ratio': stock.info.get('currentRatio'),
                'debt_to_equity': stock.info.get('debtToEquity'),
                'forward_eps': stock.info.get('forwardEps'),
                'forward_pe': stock.info.get('forwardPE'),
                'beta': stock.info.get('beta')
            }
            return render(request, 'get_fundamental_data.html', {'fundamentals': fundamentals, 'ticker': ticker})
    else:
        form = FundamentalDataForm()
    return render(request, 'get_fundamental_data.html', {'form': form})
