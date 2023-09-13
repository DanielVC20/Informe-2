import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import scipy.optimize

def funcion1(x, m):
    return m*x

def funcion2(x, m, b):
    return m*x + b

def get_data(nombre_archivo):
    datos = pd.read_csv(nombre_archivo, delimiter=";", decimal=",")
    keys = datos.keys()

    V = np.array(datos[keys[0]])
    I = np.array(datos[keys[1]])*1E-3

    return V, I

def ajuste_funciones(V, I, valor_V):
    lista_datos = []

    popt, pcov = scipy.optimize.curve_fit(funcion1, V, I)
    lista_datos.append([V, I, popt, pcov])

    ii = (V >= valor_V)

    V1 = V[ii]
    I1 = I[ii]

    popt1, pcov1 = scipy.optimize.curve_fit(funcion1, V1, I1)
    lista_datos.append([V1, I1, popt1, pcov1])

    V2 = V[~ii]
    I2 = I[~ii]
    
    popt2, pcov2 = scipy.optimize.curve_fit(funcion2, V2, I2)
    lista_datos.append([V2, I2, popt2, pcov2])

    return lista_datos

def generador_etiquetas(datos, nombres):
    etiquetas = []

    for i in range(len(datos)):    
        etiquetas.append(nombres[i])

        popt = datos[i][2]
        pcov = datos[i][3]

        sigmas = np.sqrt(np.diag(pcov))

        m = popt[0]/1E-2
        sigma_m = sigmas[0]/1E-2

        if i == 0:
            ajuste = "$I = mV$\n"
            ajuste += "$m$ = ({:.4f} $\pm$ {:.4f})".format(m, sigma_m)
            ajuste += " x 10$^{-2}$ $\Omega$"
        elif i == 1:
            ajuste = "$I = G_aV$\n"
            ajuste += "$G_a$ = ({:.4f} $\pm$ {:.4f})".format(m, sigma_m)
            ajuste += " x 10$^{-2}$ $\Omega$"
        elif i == 2:
            b = popt[1]/1E-3
            sigma_b = sigmas[1]/1E-3

            ajuste = "$I = G_bV + b$\n"
            ajuste += "$G_b$ = ({:.3f} $\pm$ {:.3f})".format(m, sigma_m)
            ajuste += " x 10$^{-2}$ $\Omega$\n"
            ajuste += "$b$ = ({:.1f} $\pm$ {:.1f}) mA".format(b, sigma_b)
            
        etiquetas.append(ajuste)

    return etiquetas

def graficas(datos, colores, etiquetas, funciones):
    def mediciones(sub_datos, sub_colores, sub_etiquetas, sub_funciones, num):
        plt.figure()

        for i in range(len(sub_datos)):
            V, I, popt, pcov = sub_datos[i]
            
            nombre = sub_etiquetas[2*i]
            ajuste = sub_etiquetas[2*i + 1]

            plt.scatter(V, I/1E-3, c=sub_colores[i], label=nombre, s=25)
            plt.plot(V, sub_funciones[i](V, *popt)/1E-3, c=sub_colores[i], label=ajuste)

            plt.errorbar(V, I/1E-3, xerr=0.01, yerr=0.1, fmt=".", c=sub_colores[i])
        
        plt.title("Corriente ($I_R$) vs Voltaje ($V_R$)")
        plt.legend()
        plt.xlabel("$V_R$ (V)")
        plt.ylabel("$I_R$ (mA)")
        plt.savefig("Grafica-VvsI-{}.png".format(num))
        return None
    
    def residuales(sub_datos, sub_colores, sub_funciones, num):
        plt.figure()

        for i in range(len(sub_datos)):
            V, I, popt, pcov = sub_datos[i]
            plt.scatter(V, (I - sub_funciones[i](V, *popt))/1E-3, c=sub_colores[i], s=25)

        plt.title("Residuales ajuste lineal")
        plt.xlabel("$V_R$ (V)")
        plt.ylabel("$I_R$ (mA)")
        plt.savefig("Residuales_{}.png".format(num))

        return None
    
    mediciones(datos[0:1], colores[0:1], etiquetas[0:2], funciones[0:1], 1)
    mediciones(datos[1:3], colores[1:3], etiquetas[2:], funciones[1:3], 2)

    residuales(datos[0:1], colores[0:1], funciones[0:1], 1)
    residuales(datos[1:3], colores[1:3], funciones[1:3], 2)

    return None

V, I = get_data("Datos_Chua.csv")
valor_V = -10

datos_chua = ajuste_funciones(V, I, valor_V)

colores = ["black", "red", "blue"]
nombres = ["Mediciones", "Mediciones V > -10V", "Mediciones V < -10V"]
funciones = [funcion1, funcion1, funcion2]

etiquetas = generador_etiquetas(datos_chua, nombres)

graficas(datos_chua, colores, etiquetas, funciones)

plt.show()