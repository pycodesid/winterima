#library
import pandas as pd
import numpy as np 
import streamlit as st

import plotly.express as px
from matplotlib import pyplot as plt
from PIL import Image

from statsmodels.tsa.stattools import acf, pacf, adfuller
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
from statsmodels.tsa.seasonal import seasonal_decompose
from statsmodels.tsa.arima_model import ARIMA
from statsmodels.tsa.holtwinters import ExponentialSmoothing
from sklearn.metrics import mean_absolute_percentage_error, mean_squared_error
import statsmodels.api as sm
import pmdarima as pmd
from pmdarima.arima.utils import ndiffs

import math
import warnings



def main():
    # Session Handling
    if 'data' not in st.session_state:
        st.session_state.data = pd.DataFrame()
    
    if 'location' not in st.session_state:
         st.session_state.location = []

    if 'layanan' not in st.session_state:
         st.session_state.layanan = []

    if 'link' not in st.session_state:
         st.session_state.link = []

    if 'bandwidth' not in st.session_state:
         st.session_state.bandwidth = []

    if 'cols' not in st.session_state:
         st.session_state.cols = []

    if 'col_name' not in st.session_state:
         st.session_state.col_name = []

    if 'x' not in st.session_state:
         st.session_state.x = []

    if 'y' not in st.session_state:
         st.session_state.y = []
    
    if 'title' not in st.session_state:
        st.session_state.title = ''

    # Main Interface
    img = Image.open('Seal_of_the_Ministry_of_Finance_of_the_Republic_of_Indonesia.svg.png')
    st.image(img, width=200)
    st.markdown(f'<h1 style="color:#E32236;font-size:56px;">{"WINTERIMA"}</h1>', unsafe_allow_html=True)
    st.header(':alpine[Aplikasi Peramalan Kebutuhan Kapasitas Bandwidth Kantor Daerah Menggunakan Metode Arima Dan Hold-Winters]')
    
    # Sidebar Initiation
    st.sidebar.title("Pilih Menu")
    menu0 = st.sidebar.selectbox("Data:", ["Choose", "Unggah Dataset", "Analisis Timeseries"])
    menu1 = st.sidebar.selectbox("Latih Data:", ["Choose", "Latih", "Prediksi"])
    menu2 = st.sidebar.selectbox("Evaluasi", ["Choose"])


    if menu0 == "Unggah Dataset" and menu1 == "Choose" and menu2 == "Choose":
        uploaded_file = st.file_uploader("Pilih Data Excel")
        if uploaded_file is not None:
            if st.button("Unggah"):
                #read excel
                xls = pd.ExcelFile(uploaded_file, engine='openpyxl')
                df = pd.read_excel(xls, 'Sheet1', header = None)
                st.success("Unggah Berhasil! Mohon Tunggu...")

                st.session_state.location = df[0].unique()
                st.session_state.layanan = df[1].unique()
                st.session_state.link = df[2].unique()
                st.session_state.bandwidth = df[3].unique()

                for i in range(len(df)):
                    st.session_state.cols.append(df[0][i] + '__' + df[1][i] + '__' + df[2][i] + '__' + str(df[3][i]))

                df2 = pd.read_excel(xls, 'Sheet2', header = None)
                df2 = df2[4:]
                df3 = df2
                df3.columns = st.session_state.cols
                df3 = df3.reset_index().drop(['index'], axis=1)
                st.session_state.data = df3.copy(deep=True)

                st.success("Proses Selesai")

        else:
            st.warning("Unggah Data Terlebih Dahulu atau Ulangi Unggah Data")

    if menu0 == "Analisis Timeseries" and menu1 == "Choose" and menu2 == "Choose":
            
            loc = st.selectbox("Pilih Lokasi", sorted(st.session_state.location))
            split_cols = [item.split('__') for item in st.session_state.cols]

            lay_ = [item[1] for item in split_cols if item[0] == loc]
            lay_ = np.array(lay_)
            lay_ = np.unique(lay_)
            lay = st.selectbox("Pilih Layanan", sorted(lay_))

            lin_ = [item[2] for item in split_cols if (item[0] == loc and item[1] == lay)]
            lin_ = np.array(lin_)
            lin_ = np.unique(lin_)
            lin = st.selectbox("Pilih Link", lin_)

            ban_ = [item[3] for item in split_cols if (item[0] == loc and item[1] == lay and item[2] == lin)]
            ban_ = np.array(ban_)
            ban_ = np.unique(ban_)
            ban_ = sorted(ban_)
            ban = st.selectbox("Pilih Bandwidth", ban_)

            col_ = loc + '__' + lay + '__' + lin + '__' + str(ban)

            type_ = st.selectbox("Pilih Transmiter atau Receiver", ["Transmiter", "Receiver"])
            type_n = st.selectbox("Pilih Peak atau Average", ["Peak", "Average"])
            num = 0
            if st.button("Import Timeseries"):
                if type_ + ' ' + type_n == 'Peak Transmit':
                     num = 0
                if type_ + ' ' + type_n == 'Average Transmit':
                     num = 1
                if type_ + ' ' + type_n == 'Peak Receive':
                     num = 2
                if type_ + ' ' + type_n == 'Average Receive':
                     num = 3

                df = st.session_state.data[st.session_state.data.index.isin(list(range(num,len(st.session_state.data), 4)))]
                df = df.reset_index()
                df = df.drop(['index'], axis=1)
                df.index = pd.date_range(start="2023-05-01",end="2023-11-30")
                df = df.fillna(method='bfill')
                df = df[col_]
                st.dataframe(df)

                fig = px.line(df,
                x=df.index,
                y=df.values,
                title = type_ + ' ' + type_n + col_,
                template = 'plotly_dark').update_layout(
                    xaxis_title="Date",
                    yaxis_title="Mbps",
                    showlegend = False
                )

                fig.update_xaxes(
                    rangeslider_visible=True,
                    rangeselector=dict(
                        buttons=list([
                            dict(count=1, label="1m", step="month", stepmode="backward"),
                            dict(count=6, label="6m", step="month", stepmode="backward"),
                            dict(count=1, label="YTD", step="year", stepmode="todate"),
                            dict(count=1, label="1y", step="year", stepmode="backward"),
                            dict(step="all")
                        ])
                    )
                )

                # Plot
                st.plotly_chart(fig, use_container_width=True)


    if menu0 == "Choose" and menu1 == "Latih" and menu2 == "Choose":
            
            def rmse(y_true, y_pred):
                y_true, y_pred = np.array(y_true), np.array(y_pred)
                mse = mean_squared_error(y_true, y_pred)
                rmse = math.sqrt(mse)

                return rmse
            
            loc = st.selectbox("Pilih Lokasi", sorted(st.session_state.location))
            split_cols = [item.split('__') for item in st.session_state.cols]

            lay_ = [item[1] for item in split_cols if item[0] == loc]
            lay_ = np.array(lay_)
            lay_ = np.unique(lay_)
            lay = st.selectbox("Pilih Layanan", sorted(lay_))

            lin_ = [item[2] for item in split_cols if (item[0] == loc and item[1] == lay)]
            lin_ = np.array(lin_)
            lin_ = np.unique(lin_)
            lin = st.selectbox("Pilih Link", lin_)

            ban_ = [item[3] for item in split_cols if (item[0] == loc and item[1] == lay and item[2] == lin)]
            ban_ = np.array(ban_)
            ban_ = np.unique(ban_)
            ban_ = sorted(ban_)
            ban = st.selectbox("Pilih Bandwidth", ban_)

            col_ = loc + '__' + lay + '__' + lin + '__' + str(ban)

            type_ = st.selectbox("Pilih Transmiter atau Receiver", ["Transmiter", "Receiver"])
            type_n = st.selectbox("Pilih Peak atau Average", ["Peak", "Average"])
            period = st.selectbox("Pilih Periode", ["Daily", "Monthly", "Annual", "Quarterly"])
            metode = st.selectbox("Pilih Metode", ["ARIMA", "HOLT-WINTERS"])
            num = 0
            if st.button("Mulai Latih"):
                if type_ + ' ' + type_n == 'Peak Transmit':
                     num = 0
                if type_ + ' ' + type_n == 'Average Transmit':
                     num = 1
                if type_ + ' ' + type_n == 'Peak Receive':
                     num = 2
                if type_ + ' ' + type_n == 'Average Receive':
                     num = 3

                df = st.session_state.data[st.session_state.data.index.isin(list(range(num,len(st.session_state.data), 4)))]
                df = df.reset_index()
                df = df.drop(['index'], axis=1)
                df.index = pd.date_range(start="2023-05-01",end="2023-11-30")
                df = df.fillna(method='bfill')
                df = df[col_]

                df = df.resample('D').mean()
                df = df.fillna(df.mean())

                if period == "Monthly":
                     df = df.resample('M').mean()
                     df.fillna(df.mean())
                if period == "Annual":
                     df = df.resample('A-DEC').mean()
                     df.fillna(df.mean())
                if period == "Quarterly":
                     df = df.resample('Q-DEC').mean()
                     df.fillna(df.mean())

                st.session_state.x = df.index
                st.session_state.y = df

                if metode == "ARIMA":
                    if 'model_arima_' not in st.session_state:
                        arima_model = sm.tsa.arima.ARIMA(st.session_state.y.values, order = (2,1,2))
                        st.session_state.model_arima_ = arima_model.fit()
                        st.write(st.session_state.model_arima_.summary())

                    arima_model = sm.tsa.arima.ARIMA(st.session_state.y, order = (2,1,2))
                    st.session_state.model_arima_ = arima_model.fit()
                    st.write(st.session_state.model_arima_.summary())
                    pred = st.session_state.model_arima_.predict(dynamic = False)

                    df_plot = pd.DataFrame({
                        "real": st.session_state.y,
                        "prediction": pred
                    })

                    st.session_state.title = type_ + ' ' + type_n + ' ' + col_

                    fig = px.line(df_plot,
                                x=df_plot.index,
                                y=df_plot.columns,
                                title = st.session_state.title,
                                template = 'plotly_dark').update_layout(
                                    xaxis_title="Date",
                                    yaxis_title="Mbps",
                                )

                    st.plotly_chart(fig, use_container_width=True)

                    model_rmse = rmse(st.session_state.y, pred)
                    st.write('RMSE ARIMA: ', model_rmse)

                if metode == "HOLT-WINTERS":
                    # Split into train and test set
                    train = st.session_state.y[:150] 
                    test = st.session_state.y[150:]

                    if 'model_holt_winter_' not in st.session_state:
                        model_holt_winter = ExponentialSmoothing(train,trend='additive', seasonal=None)
                        st.session_state.model_holt_winter_ = model_holt_winter.fit(smoothing_level=0.5, smoothing_slope=0.01, optimized=False)
                    
                    model_holt_winter = ExponentialSmoothing(train,trend='additive', seasonal=None)
                    st.session_state.model_holt_winter_ = model_holt_winter.fit(smoothing_level=0.5, smoothing_slope=0.01, optimized=False)

                    d = st.session_state.model_holt_winter_.params
                    st.dataframe(pd.DataFrame(d.items()))

                    pred = st.session_state.model_holt_winter_.forecast(len(test))

                    df_plot = pd.concat([pd.DataFrame({"train": train}), pd.DataFrame({"test": test}), pd.DataFrame({"pred": pred})])

                    title = type_ + ' ' + type_n + ' ' + col_

                    fig = px.line(df_plot,
                                x=df_plot.index,
                                y=df_plot.columns,
                                title = title,
                                template = 'plotly_dark').update_layout(
                                    xaxis_title="Date",
                                    yaxis_title="Mbps",
                                )

                    st.plotly_chart(fig, use_container_width=True)

                    model_rmse = rmse(test, pred)
                    st.write('RMSE HOLT WINTERS: ', model_rmse)

    if menu0 == "Choose" and menu1 == "Prediksi" and menu2 == "Choose":

        metode = st.selectbox("Pilih Metode", ["ARIMA", "HOLT-WINTERS"])
        if metode == "ARIMA":
            duration = st.number_input("Masukkan Jumlah Hari ke Depan", min_value = 1, step = 1)
            start = 214
            end = start + int(duration)

            pred = st.session_state.model_arima_.predict(start = start, end = end, dynamic = True)

            df_plot = pd.concat([pd.DataFrame({"real": st.session_state.y}), pd.DataFrame({"future prediction": pred})])

            st.dataframe(df_plot)

            fig = px.line(df_plot,
                        x=df_plot.index,
                        y=df_plot.columns,
                        title = st.session_state.title,
                        template = 'plotly_dark').update_layout(
                            xaxis_title="Date",
                            yaxis_title="Mbps",
                        )

            st.plotly_chart(fig, use_container_width=True)
        if metode == "HOLT-WINTERS":
            duration = st.number_input("Masukkan Jumlah Hari ke Depan", min_value = 1, step = 1)

            model_holt_winter = ExponentialSmoothing(st.session_state.y,trend='additive', seasonal=None)
            st.session_state.model_holt_winter_ = model_holt_winter.fit(smoothing_level=0.5, smoothing_slope=0.01, optimized=False)

            pred = st.session_state.model_holt_winter_.forecast(duration)

            df_plot = pd.concat([pd.DataFrame({"real": st.session_state.y}), pd.DataFrame({"future prediction": pred})])

            st.dataframe(df_plot)

            fig = px.line(df_plot,
                        x=df_plot.index,
                        y=df_plot.columns,
                        title = st.session_state.title,
                        template = 'plotly_dark').update_layout(
                            xaxis_title="Date",
                            yaxis_title="Mbps",
                        )

            st.plotly_chart(fig, use_container_width=True)


# Running program
if __name__ == "__main__":
    main()