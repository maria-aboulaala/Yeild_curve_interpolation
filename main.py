import streamlit as st
import requests
from bs4 import BeautifulSoup
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.interpolate import interp1d, CubicSpline
from scipy.optimize import least_squares
import statsmodels.api as sm
from scipy.optimize import minimize


st.title('Modélisation de la courbe du taux:')

# ---------------------------------- ikhan d utilisateur---------------------
# st.subheader('Entrez les parametres de votre portefeuille: :key: ')
with st.form(key="my_form"):
    country = st.text_input("Choisir le pays")
    modele = st.selectbox("Le modele", ["spline cubic", "Nelson Siegel", "vasiccek ", "Linear"])
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
# elif modele == "CIR":
#     st.write('not yet')

#     def ajuster_modele_cir(beta, temps, taux):
#         r0, a, b, sigma = beta
#         erreur = cir_residu(beta, temps, taux)
#         return erreur
#     def cir_residu(beta, temps, taux):
#         r0, a, b, sigma = beta
#         dt = np.diff(temps)
#         dr = np.diff(taux)
#         esp_dt = np.exp(-a * temps[:-1] * dt)
#         erreur = (r0 - a * b) * (1 - esp_dt) - a * np.cumsum((dr - sigma * np.sqrt(esp_dt) * np.random.normal(size=len(dr))) / esp_dt)
#         return erreur

    # Fonction pour interpoler linéairement entre les points de données
    # def interpoler_lineaire(temps_connus, taux_connus, temps_interpolation):
    #     f = interp1d(temps_connus, taux_connus, kind='linear', fill_value='extrapolate')
    #     taux_interpolated = f(temps_interpolation)
    #     return taux_interpolated
    
    # Modèle CIR
    # r0_guess = df['taux'].iloc[0]
    # a_guess = 0.1
    # b_guess = df['taux'].mean()
    # sigma_guess = df['taux'].std()

    # # Initialisation des paramètres
    # beta_initial = [r0_guess, a_guess, b_guess, sigma_guess]
    # bounds = ([0, 0, 0, 0], [np.inf, np.inf, np.inf, np.inf])
    # # Ajustement des paramètres du modèle CIR aux données
    # # res_cir = least_squares(ajuster_modele_cir, beta_initial, args=(df.index, df['taux']))
    # res_cir = least_squares(ajuster_modele_cir, beta_initial, args=(df.index, df['taux']), bounds=bounds)

    # # Paramètres ajustés
    # r0_adj, a_adj, b_adj, sigma_adj = res_cir.x

    # # Temps pour l'interpolation
    # temps_interpolation = np.linspace(df.index.min(), df.index.max(), 1000)

    # # Interpolation linéaire des taux connus
    # taux_interpolated_linear = interpoler_lineaire(df.index, df['taux'], temps_interpolation)

    # # Modélisation CIR des taux
    # taux_cir_modele = np.zeros_like(temps_interpolation)
    # taux_cir_modele[0] = r0_adj

    # for i in range(1, len(temps_interpolation)):
    #     dt_cir = temps_interpolation[i] - temps_interpolation[i - 1]
    #     esp_dt_cir = np.exp(-a_adj * temps_interpolation[i - 1] * dt_cir)
    #     taux_cir_modele[i] = r0_adj + a_adj * (b_adj - taux_cir_modele[i - 1]) * (1 - esp_dt_cir) + \
    #                         sigma_adj * np.sqrt(esp_dt_cir) * np.random.normal()

    # # Affichage avec Streamlit
    # # st.write('Données de la courbe des taux:')
    # # st.dataframe(df)

    # # st.write('Paramètres ajustés du modèle CIR:')
    # # st.write('r0 = ', r0_adj)
    # # st.write('a = ', a_adj)
    # # st.write('b = ', b_adj)
    # # st.write('sigma = ', sigma_adj)

    # st.write('Modélisation de la courbe des taux avec CIR:')
    # plt.plot(df.index, df['taux'], 'o', label='Données observées')
    # # plt.plot(temps_interpolation, taux_interpolated_linear, '--', label='Interpolation linéaire')
    # plt.plot(temps_interpolation, taux_cir_modele, label='Modèle CIR')
    # plt.title('Modélisation de la courbe des taux avec CIR')
    # plt.xlabel('Maturité (mois)')
    # plt.ylabel('Taux')
    # plt.legend()
    # st.pyplot()


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

elif modele == "vasiccek " :

    # r0=0.03
    # k=0.1
    # theta=0.05
    # sigma=0.01
    # T=10
    r0 = df['taux'].iloc[0]  # Taux d'intérêt initial (première ligne du DataFrame)
    k = 0.1  # Paramètre k du modèle Vasicek (à ajuster)
    theta = df['taux'].mean()  # Moyenne à long terme (moyenne des taux du DataFrame)
    sigma = df['taux'].std()  # Volatilité (écart type des taux du DataFrame)
    T = df.index.max()  # Échéance maximale
    num_points = len(df.index) + 1
    # num_points= 100
    dt = T / num_points
    time_points = np.linspace(0, T, num_points + 1)
        
        # Calcul de la structure par terme des taux avec le modèle de Vasicek
    rates = theta + (r0 - theta) * np.exp(-k * time_points) + sigma * np.sqrt(dt) * np.random.normal(size=num_points + 1).cumsum()

    #------------------------------- Tracé de la courbe
# ... (votre code précédent)

# Affichage des paramètres dans un tableau
    parametres_table = pd.DataFrame({
        'Paramètre': ['r0', 'k', 'theta', 'sigma', 'T', 'dt', 'time_points'],
        'Valeur': [r0, k, theta, sigma, T, dt, str(time_points)]
    })

    st.table(parametres_table)

    
    plt.plot(time_points, rates, label='Structure par terme des taux (Vasicek)')
    plt.title('Structure par terme des taux avec le modèle de Vasicek')
    plt.xlabel('Temps')
    plt.ylabel('Taux d\'intérêt')
    plt.legend()
    plt.grid(True)
    figg = plt.show()
    st.pyplot(figg)
        # Paramètres du modèle


    r0 = df['taux'].iloc[0]  # Taux d'intérêt initial (première ligne du DataFrame)
    a = 0.1  # Paramètre k du modèle Vasicek (à ajuster)
    
    b = df['taux'].mean()  # Moyenne à long terme (moyenne des taux du DataFrame)
    sigma = df['taux'].std()  # Volatilité (écart type des taux du DataFrame)
    T = df.index.max()  # Échéance maximale

    # Fonctions d'espérance et de variance
    def esperance(t):
        return r0 * np.exp(-a * t) + b * (1 - np.exp(-a * t))

    def variance(t):
        return (sigma**2) / (2 * a) * (1 - np.exp(-2 * a * t))

    # Création des données pour la visualisation
    temps = np.linspace(0, 10, 1000)  # vous pouvez ajuster la plage de temps
    esperance_resultats = esperance(temps)
    variance_resultats = variance(temps)

    col1 , col2 = st.columns(2)
    with col1:        

        # Tracer l'espérance
        plt.figure(figsize=(10, 5))
        plt.subplot(1, 2, 1)
        plt.plot(temps, esperance_resultats, label='Espérance')
        plt.title('Espérance de r(t) en fonction du temps')
        plt.xlabel('Temps')
        plt.ylabel('Espérance')
        plt.legend()
        st.pyplot()

        st.write('La limite de la moyenne à long terme est : ', b)
    with col2:
        # Tracer la variance
        plt.subplot(1, 2, 2)
        plt.plot(temps, variance_resultats, label='Variance')
        plt.title('Variance de r(t) en fonction du temps')
        plt.xlabel('Temps')
        plt.ylabel('Variance')
        plt.legend()
        st.pyplot()

        st.write('La limite de la variance à long terme est : ', (sigma**2) / (2 * a))

        plt.tight_layout()
        plt.show()

    
# ---------------------------------- ikhan d linear---------------------
elif modele == "Linear" :
   
    def interpolate_linear(known_time, known_rates, interpolation_time):
        f = interp1d(known_time, known_rates, kind='linear', fill_value='extrapolate')
        curves_yield = f(interpolation_time)
        return curves_yield

    interpolation_time = [2.5, 3.5, 4.5]

    # Use your interpolation function to get interpolated yields
    interpolated_yields = interpolate_linear(df.index, df['taux'], interpolation_time)

    # Plotting the original and interpolated yield curve
    plt.plot(df.index, df['taux'], 'o-', label='Original Yield Curve')
    #plt.plot(interpolation_time, interpolated_yields, 's--', label='Interpolated Yields')
    plt.xlabel('Maturity')
    plt.ylabel('Yield')
    plt.legend()
    plt.title('Yield Curve with Linear Interpolation')
    st.pyplot()
    # plt.show()



elif modele == "polynomial" :

    def interpolate_polynomial( known_time, known_rates, interpolation_time, degree=2):
        coefficients = np.polyfit(known_time, known_rates, degree)
        curves_yield = np.polyval(coefficients, interpolation_time)
        return curves_yield
    

    interpolation_time = [2.5, 3.5, 4.5]

    # Use your interpolation function to get interpolated yields
    interpolated_yields = interpolate_polynomial(df.index, df['taux'], interpolation_time)

    # Plotting the original and interpolated yield curve
    plt.plot(df.index, df['taux'], 'o-', label='Original Yield Curve')
    #plt.plot(interpolation_time, interpolated_yields, 's--', label='Interpolated Yields')
    plt.xlabel('Maturity')
    plt.ylabel('Yield')
    plt.legend()
    plt.title('Yield Curve with polynomial Interpolation')
    st.pyplot()
    # plt.show()
