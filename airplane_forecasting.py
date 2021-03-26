#################################
# Airline Passenger Forecasting
#################################


import itertools
import warnings
import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
import statsmodels.api as sm
import statsmodels.tsa.api as smt
from statsmodels.tsa.arima_model import ARIMA
from statsmodels.tsa.seasonal import seasonal_decompose
from statsmodels.tsa.holtwinters import SimpleExpSmoothing
from statsmodels.tsa.holtwinters import ExponentialSmoothing
from statsmodels.tsa.statespace.sarimax import SARIMAX
from sklearn.metrics import mean_absolute_error
warnings.filterwarnings('ignore')



#################################
# Verinin Görselleştirilmesi
#################################

#veriyi df'e atama:
df = pd.read_csv('C:/Users/Lenovo/Masaüstü/VBObootcamp/projects/Airplane_Passenger_Forecasting/airline_passengers.csv', index_col='month', parse_dates=True)
df.shape
df.head()

#veriyi görselleştirme:
df[['total_passengers']].plot(title='Passengers Data')
plt.show()

# index'in aylık olacağını ifade edelim
df.index.freq = "MS"

# 120 gözlemi train 24 gözlemi de test olarak seçiyorum.
train = df[:120]
test = df[120:]


#################################
# Zaman Serisin Yapısal Analizi
#################################

#Zaman serisini trend, mevsimsellik, artıklar anlamında analiz etmek için görselleştirelim:

def ts_decompose(y, model="additive", stationary=False):
    from statsmodels.tsa.seasonal import seasonal_decompose
    result = seasonal_decompose(y, model=model)
    fig, axes = plt.subplots(4, 1, sharex=True, sharey=False)
    fig.set_figheight(10)
    fig.set_figwidth(15)

    axes[0].set_title("Decomposition for " + model + " model")
    axes[0].plot(y, 'k', label='Original ' + model)
    axes[0].legend(loc='upper left')

    axes[1].plot(result.trend, label='Trend')
    axes[1].legend(loc='upper left')

    axes[2].plot(result.seasonal, 'g', label='Seasonality & Mean: ' + str(round(result.seasonal.mean(), 4)))
    axes[2].legend(loc='upper left')

    axes[3].plot(result.resid, 'r', label='Residuals & Mean: ' + str(round(result.resid.mean(), 4)))
    axes[3].legend(loc='upper left')
    plt.show()

    if stationary:
        print("HO: Seri Durağan değildir.")
        print("H1: Seri Durağandır.")
        p_value = sm.tsa.stattools.adfuller(y)[1]
        if p_value < 0.05:
            print(F"Sonuç: Seri Durağandır ({p_value}).")
        else:
            print(F"Sonuç: Seri Durağan Değildir ({p_value}).")

for model in ["additive", "multiplicative"]:
    ts_decompose(df[['total_passengers']], model, True)

# Trend, Mevsimsellik var.
# Toplamsal formda.

#################################
# Single Exponential Smoothing
#################################

def optimize_ses(train, alphas, step=48):
    """
    Belirlenen alfa parametrelerine göre Ses modeli kurma ve mae hesaplama

    :param train: train seti
    :param alphas: alfa parametresi
    :param step: forecast etme sayısı
    """
    for alpha in alphas:
        ses_model = SimpleExpSmoothing(train).fit(smoothing_level=alpha)
        y_pred = ses_model.forecast(step)
        mae = mean_absolute_error(test, y_pred)
        print("alpha:", round(alpha, 2), "mae:", round(mae, 4))

#alfa parametrelerini belirleyelim:
alphas = np.arange(0.01, 1, 0.10)

#fonksiyonu çağıralım:
optimize_ses(train, alphas, step=24)
# alpha: 0.11 mae: 82.528

################
#Final SES Model
################

#Final modeli oluşturalım:

ses_model = SimpleExpSmoothing(train).fit(smoothing_level=0.11)

#test seti 24 gözlemden oluştuğu için:
y_pred = ses_model.forecast(24)

def plot_prediction(y_pred, label):
    #train setini görselleştirme
    train["total_passengers"].plot(legend=True, label="TRAIN")
    #test setini görselleştirme
    test["total_passengers"].plot(legend=True, label="TEST")
    #tahminleri görselleştirme
    y_pred.plot(legend=True, label="PREDICTION")
    #grafiği oluşturma
    plt.title("Train, Test and Predicted Test Using "+label)
    plt.show()

plot_prediction(y_pred, "Single Exponential Smoothing")

#hatayı hesaplayalım:
mean_absolute_error(test, y_pred)


#################################
# Double Exponential Smoothing
#################################

def optimize_des(train, alphas, betas, step=48):
    """
    Belirlenen alfa, beta parametrelerine göre des modeli kurma ve mae hesaplama

    :param train: train seti
    :param alphas: alfa parametresi
    :param betas: beta parametresi
    :param step: forecast etme sayısı
    """
    print("Optimizing parameters...")
    results = []
    for alpha in alphas:
        for beta in betas:
            des_model = ExponentialSmoothing(train, trend="add").fit(smoothing_level=alpha, smoothing_slope=beta)
            y_pred = des_model.forecast(step)
            mae = mean_absolute_error(test, y_pred)
            results.append([round(alpha, 2), round(beta, 2), round(mae, 2)])
    results = pd.DataFrame(results, columns=["alpha", "beta", "mae"]).sort_values("mae")
    print(results)


alphas = np.arange(0.01, 1, 0.10)
betas = np.arange(0.01, 1, 0.10)

optimize_des(train, alphas, betas, step=24)
# alpha: 0.01 beta: 0.11 mae: 82.528

################
#Final DES Model
################

#Final modeli oluşturalım:
des_model = ExponentialSmoothing(train, trend="add").fit(smoothing_level=0.01,
                                                         smoothing_slope=0.21)

#test seti 24 gözlemden oluştuğu için:
y_pred = des_model.forecast(24)

plot_prediction(y_pred, "Double Exponential Smoothing")

#hatayı hesaplayalım:
mean_absolute_error(test, y_pred)


#################################
# Triple Exponential Smoothing (Holt-Winters)
#################################

def optimize_tes(train, abg, step=48):
    """
        Belirlenen alfa, beta parametrelerine göre des modeli kurma ve mae hesaplama

        :param train: train seti
        :param alphas: alfa parametresi
        :param betas: beta parametresi
        :param gammas: gamma parametresi
        :param abg: alfa-beta-gamma kombinasyonları
        :param step: forecast etme sayısı
        """
    print("Optimizing parameters...")
    results = []
    for comb in abg:
        tes_model = ExponentialSmoothing(train, trend="add", seasonal="add", seasonal_periods=12).\
            fit(smoothing_level=comb[0], smoothing_slope=comb[1], smoothing_seasonal=comb[2])
        y_pred = tes_model.forecast(step)
        mae = mean_absolute_error(test, y_pred)
        print([round(comb[0], 2), round(comb[1], 2), round(comb[2], 2), round(mae, 2)])
        results.append([round(comb[0], 2), round(comb[1], 2), round(comb[2], 2), round(mae, 2)])
    results = pd.DataFrame(results, columns=["alpha", "beta", "gamma", "mae"]).sort_values("mae")
    print(results)

alphas = betas = gammas = np.arange(0.10, 1, 0.20)

#itertools kullanarak olası kombinasyonları elde etmek:
abg = list(itertools.product(alphas, betas, gammas))

#test seti 24 gözlemden oluştuğu için:
optimize_tes(train, abg, step=24)

################
#Final TES Model
################

#Final modeli oluşturalım:

tes_model = ExponentialSmoothing(train, trend="add", seasonal="add", seasonal_periods=12).\
            fit(smoothing_level=0.3, smoothing_slope=0.3, smoothing_seasonal=0.5)

#test seti 24 gözlemden oluştuğu için:
y_pred = tes_model.forecast(24)

plot_prediction(y_pred, "Triple Exponential Smoothing ADD")

#hatayı hesaplayalım:
mean_absolute_error(test, y_pred)
#11.99


##################################################
# ARIMA(p, d, q): (Autoregressive Integrated Moving Average)
##################################################

# p, d, q kombinasyonlarının üretilmesi
p = d = q = range(0, 4)
pdq = list(itertools.product(p, d, q))

def arima_optimizer_aic(train, orders):
    """
     Belirlenen p,d, q kombinasyonlarına modeli kurma ve en kaliteli modeli seçme

        :param train: train seti
        :param orders: p,d,q kombinasyonları
        :return: best_params

    """
    best_aic, best_params = float("inf"), None
    for order in orders:
        try:
            arma_model_result = ARIMA(train, order).fit(disp=0)
            aic = arma_model_result.aic
            if aic < best_aic:
                best_aic, best_params = aic, order
            print('ARIMA%s AIC=%.2f' % (order, aic))
        except:
            continue
    print('Best ARIMA%s AIC=%.2f' % (best_params, best_aic))
    return best_params


best_params_aic = arima_optimizer_aic(train, pdq)


# Tune Edilmiş Model
arima_model = ARIMA(train, best_params_aic).fit(disp=0)

#test seti 24 gözlemden oluştuğu için:
y_pred = arima_model.forecast(24)[0]
mean_absolute_error(test, y_pred)
# 51.18

#Tahminleri Görselleştirme
plot_prediction(pd.Series(y_pred, index=test.index), "ARIMA")


##################################################
# SARIMA
##################################################

# p, d ve q kombinasyonlarının üretilmesi
p = d = q = range(0, 3)
pdq = list(itertools.product(p, d, q))
seasonal_pdq = [(x[0], x[1], x[2], 12) for x in list(itertools.product(p, d, q))]


def sarima_optimizer_aic(train, pdq, seasonal_pdq):
    """
    Trend ve mevsimsel birleşenlere göre model kurma ve en kaliteli olan modeli seçme

        :param train: train seti
        :param pdq: p,d,q kombinasyonları
        :param seasonal_pdq: mevsimsel p,d,q kombinasyonları
        :return: best_order, best_seasonal_order

    """
    best_aic, best_order, best_seasonal_order = float("inf"), float("inf"), None
    for param in pdq:
        for param_seasonal in seasonal_pdq:
            try:
                sarimax_model = SARIMAX(train, order=param, seasonal_order=param_seasonal)
                results = sarimax_model.fit(disp=0)
                aic = results.aic
                if aic < best_aic:
                    best_aic, best_order, best_seasonal_order = aic, param, param_seasonal
                print('SARIMA{}x{}12 - AIC:{}'.format(param, param_seasonal, aic))
            except:
                continue
    print('SARIMA{}x{}12 - AIC:{}'.format(best_order, best_seasonal_order, best_aic))
    return best_order, best_seasonal_order


best_order, best_seasonal_order = sarima_optimizer_aic(train, pdq, seasonal_pdq)

# Tune Edilmiş Model
model = SARIMAX(train, order=best_order, seasonal_order=best_seasonal_order)
sarima_final_model = model.fit(disp=0)

#Final model test hatası
#test seti 24 gözlemden oluştuğu için:

y_pred_test = sarima_final_model.get_forecast(steps=24)
pred_ci = y_pred_test.conf_int()

y_pred = y_pred_test.predicted_mean
mean_absolute_error(test, y_pred)
# 54.66

#Tahminleri Görselleştirme
plot_prediction(pd.Series(y_pred, index=test.index), "SARIMA")
