# importing libraries
import pandas
import matplotlib.pyplot as plt
from fbprophet import Prophet
import numpy
import plotly.express as px
from statsmodels.tsa import stattools
from sklearn.model_selection import train_test_split as tts
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.metrics import r2_score, mean_absolute_error
from sklearn.pipeline import Pipeline
from sklearn.tree import DecisionTreeClassifier
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import StandardScaler
plt.style.use("ggplot")

'''Read csv'''
# In this model we will only consider global covid cases
def fun():
    covid_death_cases_data = pandas.read_csv(r'dataset/CONVENIENT_global_deaths.csv')
    covid_global_cases_data = pandas.read_csv(r'dataset/CONVENIENT_global_confirmed_cases.csv')


    print(covid_death_cases_data.tail())
    print(covid_global_cases_data.tail())

    '''preprocessing check'''
    print(covid_global_cases_data.isna().sum())

    # '''Data'''
    count = []
    for i in range(1,len(covid_global_cases_data)):
        count.append(sum(pandas.to_numeric(covid_global_cases_data.iloc[i,1:].values)))

    df = pandas.DataFrame()
    df["Date"] = covid_global_cases_data["Country/Region"][1:]
    df["Cases"] = count
    df=df.set_index("Date")

    count = []
    for i in range(1,len(covid_death_cases_data)):
        count.append(sum(pandas.to_numeric(covid_death_cases_data.iloc[i,1:].values)))

    df["Deaths"] = count
    print("Dataframe\n", df.head())
    # histogram
    df.hist(bins=100)
    plt.show()

    #plot
    df.plot()
    plt.show()

    # Cases with rolling mean
    df.Cases.plot(title="Daily Covid19 Cases",marker=".",figsize=(10,5), label="Daily cases")
    df.Cases.rolling(window=5).mean().plot(label="MA5")
    plt.ylabel("Cases")
    plt.legend()
    plt.show()

    # deaths with rolling mean
    df.Deaths.plot(title="Daily Covid19 Deaths", marker=".", figsize=(10, 5), label="Daily Deaths")
    df.Deaths.rolling(window=5).mean().plot(label="MA5")
    plt.ylabel("Deaths")
    plt.legend()
    plt.show()

    # Finding stationarity of data using Dickey Fuller Test

    def stationary_data(data):
        # Mean and Std
        mean = data.rolling(window = 10).mean()
        std = data.rolling(window = 10).std()

        # stat plot
        plt.plot(data, color='green', label="Data")
        plt.plot(mean, color='blue', label='Mean')
        plt.plot(std, color='black', label="STD")
        plt.legend(loc='best')
        plt.title("Mean and STD")
        plt.show()

        # Fuller
        stat_test = stattools.adfuller(data.Cases, autolag='AIC')
        stat_out = pandas.Series(stat_test[0:4], index=['Test Statistic','p-value','#Lags Used','Number of Observations Used'])
        for key,value in stat_test[4].items():
            stat_out['Critical Value (%s)'%key] = value
        print(stat_out)

    stationary_data(df)

    # Making stat data to check if it would be perfect for the processing
    df_log = numpy.log(df)
    df_avg = df_log.rolling(window=10).mean()
    df_avg_diff = df_log - df_avg
    df_avg_diff.dropna(inplace=True)
    stationary_data(df_avg_diff)

    # using exponential weighted moving average for further processing
    df_ewma_avg = df_log.ewm(halflife=12).mean()
    df_ewma_diff = df_log - df_ewma_avg
    stationary_data(df_ewma_diff)

    # df_cases
    df_cases = pandas.DataFrame({"ds": [], "y": []})
    df_cases["ds"] = pandas.to_datetime(df.index)
    df_cases["y"] = df.iloc[:, 0].values
    print("Cases\n", df_cases.head())

    # df deaths
    df_deaths = pandas.DataFrame({"ds": [], "y": []})
    df_deaths["ds"] = pandas.to_datetime(df.index)
    df_deaths["y"] = df.iloc[:, -1].values
    print(df_deaths.head())
    # '''convert into dym'''
    days = df_cases.ds.dt.day
    month = df_cases.ds.dt.month
    year = df_cases['ds'].dt.year

    dmy = pandas.DataFrame({"date": [], "month": [], "year": [], "cases": []})
    dmy['date'] = days
    dmy['month'] = month
    dmy['year'] = year
    dmy['cases'] = df_cases['y']



    '''Machine Learning Model'''

    y = dmy.iloc[:,-1]
    x = dmy.iloc[:,:-1]

    x_train, x_test, y_train, y_test = tts(x, y, test_size=0.3)


    model = GradientBoostingRegressor()
    model.fit(x_train,y_train)
    y_pred = model.predict(x_test)
    plt.plot(y_pred)
    plt.title("Regressor")
    plt.show()
    print("R2:", r2_score(y_test, y_pred))

    pip_model = Pipeline([('ss', StandardScaler()), ('tree', DecisionTreeClassifier())])
    m = pip_model.fit(x_train, y_train)
    y_pred2 = m.predict(x_test)
    plt.plot(y_pred2)
    plt.title("Tree")
    plt.show()
    print("Decision Tree Classifier R2:", r2_score(y_test, y_pred2))

    # Linear Regression
    linear_model =  LinearRegression()
    lin_model_fit = linear_model.fit(x_train, y_train)
    y_pred_regressor = lin_model_fit.predict(x_test)
    plt.plot(y_pred_regressor)
    plt.title("Linear Regression")
    plt.show()
    print("Linear Regression Model R2:", r2_score(y_test, y_pred_regressor))


    '''Fbprophet'''
    class Fbprophet_model(object):
        def fit(self, data):
            self.data = data
            self.prophet_model = Prophet(daily_seasonality=False, weekly_seasonality=True, yearly_seasonality=False)
            self.prophet_model.fit(self.data)

        def data_forecast(self, periods, frequency):
            self.future_data = self.prophet_model.make_future_dataframe(periods=periods, freq=frequency)
            self.forecast = self.prophet_model.predict(self.future_data)
            print("Forecast",self.forecast[:5])
            self.prophet_model.plot_components(self.forecast)
            plt.show()

        def plot_data(self, xlabel="Years", ylabel="Values"):
            self.prophet_model.plot(self.forecast, xlabel=xlabel, ylabel=ylabel, figsize=(9, 4))
            self.prophet_model.plot_components(self.forecast, figsize=(9, 6))

        def mae(self):
            return mean_absolute_error(self.data.y, self.forecast.yhat[:len(df)])


        def R2_score(self):
            return r2_score(self.data.y, self.forecast.yhat[:len(df)])


    # Cases Forecast
    model = Fbprophet_model()
    model.fit(df_cases)
    model.data_forecast(45, "D")
    print(model.mae())
    print(model.R2_score())

    model_forecast = model.forecast[["ds", "yhat_lower", "yhat_upper", "yhat"]].tail(45).reset_index().set_index("ds").drop(
        "index", axis=1)
    model_forecast["yhat"].plot(marker=".", figsize=(10, 5))
    plt.fill_between(x=model_forecast.index, y1=model_forecast["yhat_lower"], y2=model_forecast["yhat_upper"], color="gray")
    plt.legend(["forecast", "Bound"], loc="upper left")
    plt.title("Forecasting of Next 45 Days Cases")
    plt.show()

    # Deaths Forecast
    model = Fbprophet_model()
    model.fit(df_deaths)
    model.data_forecast(45, "D")
    print(model.mae())
    print(model.R2_score())

    model_forecast = model.forecast[["ds", "yhat_lower", "yhat_upper", "yhat"]].tail(45).reset_index().set_index("ds").drop(
        "index", axis=1)
    model_forecast["yhat"].plot(marker=".", figsize=(10, 5))
    plt.fill_between(x=model_forecast.index, y1=model_forecast["yhat_lower"], y2=model_forecast["yhat_upper"], color="gray")
    plt.legend(["forecast", "Bound"], loc="upper left")
    plt.title("Forecasting of Next 45 Days Deaths")
    plt.show()

if __name__ == "__main__":
    fun()