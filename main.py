import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from statsmodels.tsa.stattools import adfuller
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
from statsmodels.tsa.arima.model import ARIMA
from arch import arch_model


np.random.seed(42)
dates = pd.date_range(start="2025-01-01", periods=500, freq="B")

price_path = 1000 * np.exp(np.cumsum(np.random.normal(0.0005, 0.02, 500)))
df = pd.DataFrame({'price': price_path}, index=dates)



df['log_return'] = np.log(df['price'] / df['price'].shift(1))
returns = df['log_return'].dropna() * 100  


def check_stationarity(series, name):
    result = adfuller(series)
    print(f"--- آزمون ایستایی (ADF) برای {name} ---")
    print(f"آماره آزمون (Test Statistic): {result[0]:.4f}")
    print(f"p-value: {result[1]:.4f}")
    if result[1] < 0.05:
        print("نتیجه: داده ایستا/مانا است (p-value < 0.05).\n")
    else:
        print("نتیجه: داده غیرایستا است (p-value >= 0.05).\n")


check_stationarity(df['price'], "قیمت خام")
check_stationarity(returns, "بازدهی لوگاریتمی")



fig, axes = plt.subplots(1, 2, figsize=(12, 4))


plot_acf(returns, lags=20, ax=axes[0], title="نمودار ACF (تعیین درجه MA)")


plot_pacf(returns, lags=20, ax=axes[1], title="نمودار PACF (تعیین درجه AR)")

plt.tight_layout()
plt.show()



arima_order = (1, 0, 1)
arima_model = ARIMA(returns, order=arima_order)
arima_results = arima_model.fit()


residuals = arima_results.resid

print("--- خلاصه پیش‌بینی میانگین (ARIMA) ---")
print(f"ضریب AR(1): {arima_results.params.get('ar.L1', 0):.4f}")
print(f"ضریب MA(1): {arima_results.params.get('ma.L1', 0):.4f}\n")


garch_model = arch_model(residuals, mean='Zero', vol='GARCH', p=1, q=1)
garch_results = garch_model.fit(disp='off')


df.loc[returns.index, 'volatility'] = garch_results.conditional_volatility


next_return_pred = arima_results.forecast(steps=1).iloc[0]
next_vol_pred = np.sqrt(garch_results.forecast(horizon=1).variance.iloc[-1, 0])

print("--- خروجی نهایی مدل ترکیبی ARIMA-GARCH ---")
print(f"پیش‌بینی بازدهی فردا (میانگین ARIMA): {next_return_pred:.4f}%")
print(f"پیش‌بینی ریسک/نوسان فردا (واریانس GARCH): {next_vol_pred:.4f}%\n")


plt.figure(figsize=(10, 4))
plt.plot(df['volatility'], color='red', label='GARCH Conditional Volatility (Risk)')
plt.title('تخمین نوسان‌پذیری و ریسک لغزان با مدل GARCH')
plt.ylabel('نوسان‌پذیری (%)')
plt.grid(True)
plt.legend()
plt.show()