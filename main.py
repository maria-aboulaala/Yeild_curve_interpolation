import streamlit as st
import requests
from bs4 import BeautifulSoup
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.interpolate import CubicSpline
from scipy.optimize import least_squares
import statsmodels.api as sm

st.title('Modélisation de la courbe du taux:')

# ---------------------------------- ikhan d utilisateur---------------------
# st.subheader('Entrez les parametres de votre portefeuille: :key: ')
with st.form(key="my_form"):
    country = st.text_input("Choisir le pays")
    modele = st.selectbox("Le modele", ["spline cubic", "CIR", "Nelson Siegel", "Vasicek "])
    st.form_submit_button("Modeliser")


def convert_to_months(years):
    if 'month' in years:
        return int(years.split()[0])
    elif 'year' in years:
        return int(years.split()[0]) * 12
    else:
        return None


def convert_to_int(rate):
    return float(rate.strip('%'))
# ---------------------------------- ikhan d DATA--------------------- done 

url = "https://www.worldgovernmentbonds.com/country/"
url = url + country + '/'
response = requests.get(url)
soup = BeautifulSoup(response.text, 'html.parser')

table = soup.find('table', {'class': 'w3-table money pd22 -f14'})

table_data = []

for row in table.find_all('tr'):
    cells = row.find_all('td')

    if len(cells) >= 2:
        maturity = convert_to_months(cells[1].text.strip())
        rate = convert_to_int(cells[2].text.strip())
        row_data = [maturity, rate]
        table_data.append(row_data)

df = pd.DataFrame(table_data, columns=['maturité(mois)', 'taux'])
df = df.set_index('maturité(mois)')
# st.dataframe(df)

# plt.figure(figsize=(10, 6))
plt.scatter(df.index, df['taux'], label='courbe de taux 10 ans', color='blue', marker='o')
plt.title('10-year Constant Maturity Treasury Rates')
plt.xlabel('time to maturity')
plt.ylabel('Taux d interet')
plt.plot(df.index, df['taux'], 'o', label = 'yield data points')
plt.legend()
plt.grid(True)
fig1 = plt.show()
col1, col2 = st.columns([1, 3])
col1.dataframe(df)
col2.pyplot(fig1) 




# ---------------------------------- ikhan d spline---------------------

if modele == "spline cubic" : 
    cs = CubicSpline(df.index, df['taux'])
    curve_points = np.linspace(min(df.index), max(df.index), 1000)
    
    col1, col2 = st.columns(2)
    with col1:
        plt.title('Courbe de taux pour un spline cubic', fontsize = 15)
        plt.xlabel('Maturité (en mois)')
        plt.ylabel('Taux')

        plt.plot(df.index, df['taux'], 'o', label = 'Taux')
        plt.plot(curve_points, cs(curve_points), label="spline cubic")
        plt.grid(linestyle = '--', linewidth = 1)
        plt.legend()
        st.pyplot()
    # Plot the second spline in the second column
    with col2:
        plt.figure()
        plt.title('Les derivées du spline cubic', fontsize = 15)
        plt.xlabel('Maturité (en mois)')
        plt.ylabel('Qth Derivative of 3rd Degree Spline')
        plt.plot(curve_points, cs(curve_points, 1), label="derivée premiere")
        plt.plot(curve_points, cs(curve_points, 2), label="derivée seconde")
        plt.plot(curve_points, cs(curve_points, 3), label="derivée 3")
        plt.grid(linestyle = '--', linewidth = 1)
        plt.legend()
        st.pyplot()


# ---------------------------------- ikhan d CIR---------------------
elif modele == "CIR":
    st.write('not yet')

# ---------------------------------- ikhan d nelson---------------------
elif modele == "Nelson Siegel" :
# parametres
    st.sidebar.subheader('Paramétres du modele:')
    b0 = st.sidebar.slider('Beta 0', min_value=0.0, max_value=5.0, value=1.0)
    b1 = st.sidebar.slider('Beta 1', min_value=-5.0, max_value=5.0, value=1.0)
    b2 = st.sidebar.slider('Beta 2', min_value=-5.0, max_value=5.0, value=-1.0)
    lamda = st.sidebar.slider('lamda', min_value=0.01, max_value=1.00, value=0.06)

    beta = [b0, b1, b2]

    curve_points = np.linspace(min(df.index), max(df.index), 1000)

    def NS_yield(time_, beta_):
        level_factor =  1
        slope_factor = (1 - np.exp((-1)*lamda*time_))/(lamda*time_)
        curvature_factor = (1 - np.exp((-1)*lamda*time_))/(lamda*time_) - np.exp((-1)*lamda*time_)
        return (beta_[0]*level_factor + beta_[1]*slope_factor + beta_[2]*curvature_factor)

    def factor_loading(time_):
        level_factor_loading = []
        slope_factor_loading = []
        curvature_factor_loading = []
        for i in time_:
            level_factor_loading.append(1)
            slope_factor_loading.append((1 - np.exp((-1)*lamda*i))/(lamda*i))
            curvature_factor_loading.append((1 - np.exp((-1)*lamda*i))/(lamda*i) - np.exp((-1)*lamda*i))
        
        return level_factor_loading, slope_factor_loading, curvature_factor_loading
    
    b0_loading, b1_loading, b2_loading = factor_loading(curve_points)

    plt.figure()
    plt.title(' Les chargements de facteurs', fontsize = 15)
    plt.xlabel('Maturité (en mois)')
    plt.ylabel('facteurs')
    plt.plot(curve_points, b0_loading, '--', color = 'b', label = 'B0 le facteur de niveau')
    plt.plot(curve_points, b1_loading, '-', color = 'c', label="B1  le facteur de pente")
    plt.plot(curve_points, b2_loading, '-.', color = 'y', label="B2 le facteur de courbure")
    plt.grid(linestyle = '--', linewidth = 1)
    plt.legend()
    st.pyplot()


    st.markdown("""
        _____________________________________________________________________________________________________________________________________

                """)
# -----------
    col1 , col2 , col3 = st.columns(3)
    with col1:
        def residual(beta_, time_array, yields_):
            errors = []
            c_index = 0
            for i in yields_:
                errors.append(NS_yield(time_array[c_index], beta_) - i)
                c_index += 1
            return np.array(errors)
        st.success('Erreurs pour beta choisis')
        st.write(residual(beta, df.index, df['taux']))

    

# ----- calibration 
    with col2 : 
        res = least_squares(residual, beta, args = (df.index, df['taux']))
        optimum_beta = res.x
        st.success('Beta optimum apres calibration')
        st.write(optimum_beta)
        # st.write('cost: ', res.cost)

    with col3:
        st.success('Erreurs pour beta optimim')
        st.write(residual(optimum_beta, df.index, df['taux']))

    st.markdown("""
        _____________________________________________________________________________________________________________________________________

            """)


    # beta_sm = [1, 1, 1]
    # level_factor_sm =  np.ones(df.index.size)
    # slope_factor_sm = (1 - np.exp((-1)*lamda*df.index))/(lamda*df.index)
    # curvature_factor_sm = (1 - np.exp((-1)*lamda*df.index))/(lamda*df.index) - np.exp((-1)*lamda*df.index)
    # X = np.column_stack((level_factor_sm, slope_factor_sm, curvature_factor_sm))
    # df['taux']

    # model = sm.OLS(df['taux'], X)
    # results = model.fit()
    # st.write('beta values: ', results.params)
    # st.write()
    # st.write(results.summary())

    np.round(NS_yield(df.index, optimum_beta), 3)

    plt.figure()
    plt.title('courbe de taux NelsonSiegel', fontsize = 15)
    plt.xlabel('maturité (en mois)')
    plt.ylabel('taux')
    plt.plot(df.index, df['taux'], 'o', label = 'yield data points')
    plt.plot(curve_points, NS_yield(curve_points, optimum_beta), label="Nelson Siegel")
    plt.grid(linestyle = '--', linewidth = 1)
    plt.legend()
    st.pyplot()


# ---------------------------------- ikhan d Vasicek---------------------
elif modele == "Vasicek":
    st.write('grrr')










