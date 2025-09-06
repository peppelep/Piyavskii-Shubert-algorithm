import numpy as np
import matplotlib.pyplot as plt 
import functions as funz

#Per calcolare la x̂ applichiamo la seguente formula
# x̂ = (x1+x2)/2 - (f(x2)-f(x1)/2L)
# x̂ = 0.5 * (x1+x2 - (f(x2) - f(x1))) / L
def XCappuccio(x1, x2, z1, z2, L):
    return ((x1 + x2) - (z2 - z1) / (L)) / 2 

#Per calcolare la R applichiamo la seguente formula
# R = (f(x1)+f(x2))/2 - L * (x2-x1)/2
# R = ((f(x1) + f(x2)) - (L * (x2 - x1 ))) / 2
def cR(x1, x2, z1, z2, L):
    return ((z1 + z2) - (L * (x2 - x1))) / 2

#Sappiamo che al passo 1 dell'algoritmo si devono riordinare i punti di valutazione di f(x)
# a = x0 < x1 < .. < xn = b
# ordina la lista [(x1,f1),(x2,f2),...] in base al valore di xi
def ordina(lista):  
    for i in range(len(lista) - 1):
        for j in range(len(lista) - i - 1):
            if lista[j][0] > lista[j + 1][0]:
                tmp = lista[j]
                lista[j] = lista[j + 1]
                lista[j + 1] = tmp

#Come criterio di arresto abbiamo la seguente condizione :
# se x_t - (x_t_meno_1) <= Epsilon
# l'algoritmo si ferma e restituisce le approssimazioni , altrimenti si avanza
def condizioneDiUscita(indiciX, intervallo, epsilon , estremi):
    t = indiciX[1]
    t_meno_1 = indiciX[0]

    x_t = intervallo[t][0]
    x_t_meno_1 = intervallo[t_meno_1][0]

    #if x_t - x_t_meno_1 < epsilon: 
    #    return True

    a = estremi[0]
    b = estremi[1]
    if x_t - x_t_meno_1 < epsilon * (b-a):  # x_i-x_(i-1)<Epsilon
        return True
    return False

def Piyavskii_Shubert(estremi,L,epsilon,funzione):
    #Valuto la funzione nei punti estremi iniziali
    fx0 = funzione(estremi[0]) #f(a)
    fx1 = funzione(estremi[1]) #f(b)

    #Ad ogni iterazione k del metodo si utilizzano due sistemi di indici per la successione
    #di punti di valutazione di f(x)
    #Abbiamo i punti
    # x_0 , x_1 , x_2 ... punti di valutazione in modo crescente
    x_f_ordinati = [[estremi[0],fx0],[estremi[1],fx1]]
    # x^0 , x^1 , x^2 ... punti di valutazione
    x_f_non_ordinati = [[estremi[0],fx0],[estremi[1],fx1]]

    #Setto i valori iniziali
    minimoCorrente = fx0 #f(a)
    xDelMinimo = estremi[0]  #a
    indiceDelMinimo = 0

    #calcolo R e xCappuccio
    xCappuccio = XCappuccio(estremi[0],estremi[1],fx0,fx1,L)
    R = cR(estremi[0],estremi[1],fx0,fx1,L)

    xCappuccio_R = []  #contiene le coppie [x̂,R]
    xCappuccio_R.append([xCappuccio,R])

    indiciX = (0,1) #inizio con x0 e x1 
    numeroIterazioni = 0

    #and len(xCappuccio_R) != 0

    while not condizioneDiUscita(indiciX, x_f_ordinati, epsilon, estremi) and numeroIterazioni < 20000:
        numeroIterazioni += 1

        #Scelta del sottointervallo t con la minima caratteristica R_t per la suddivisione
        # t = arg min Ri
        coppiaRMinima = xCappuccio_R[0]  #setto il minimo valore al primo presente nella lista
        for i in range(len(xCappuccio_R)) :
            if(xCappuccio_R[i][1] < coppiaRMinima[1]):   #il confronto viene fatto sul valore R ovviamente 
                coppiaRMinima = xCappuccio_R[i]

        xMinima = coppiaRMinima[0]
        RMinima = coppiaRMinima[1]

        xCappuccio_R.remove(coppiaRMinima)  # rimuoviamo la caratteristica appena estratta dalla lista

        #Valuto la funzione
        fxR = funzione(xMinima)
        x_f_ordinati.append([xMinima,fxR])
        ordina(x_f_ordinati)

        x_f_non_ordinati.append([xMinima,fxR])

        #Trovo l indice del sottointervallo associato alla caratteristica minore
        indice_aggiornamento = x_f_ordinati.index([xMinima,fxR])

        if(fxR < minimoCorrente):
            minimoCorrente = fxR
            xDelMinimo = xMinima
            indiceDelMinimo = indice_aggiornamento

        xCappuccio1 = XCappuccio(x_f_ordinati[indice_aggiornamento-1][0],x_f_ordinati[indice_aggiornamento][0],   #x_t__meno_1 , x_t
                                     x_f_ordinati[indice_aggiornamento-1][1],x_f_ordinati[indice_aggiornamento][1],L) # f(x_t_meno_1) , # f(x_t)

        xCappuccio2 = XCappuccio(x_f_ordinati[indice_aggiornamento][0], x_f_ordinati[indice_aggiornamento+1][0], #x_t , x_t_piu_1
                                     x_f_ordinati[indice_aggiornamento][1], x_f_ordinati[indice_aggiornamento+1][1],L) # f(x_t) , # f(x_t_piu_1)

        R1 = cR(x_f_ordinati[indice_aggiornamento-1][0],x_f_ordinati[indice_aggiornamento][0],
                                     x_f_ordinati[indice_aggiornamento-1][1],x_f_ordinati[indice_aggiornamento][1],L)

        R2 = cR(x_f_ordinati[indice_aggiornamento][0],x_f_ordinati[indice_aggiornamento+1][0],
                                     x_f_ordinati[indice_aggiornamento][1],x_f_ordinati[indice_aggiornamento+1][1],L)


        xCappuccio_R.append([xCappuccio1,R1])
        xCappuccio_R.append([xCappuccio2, R2])

        indiciX = (indice_aggiornamento - 1, indice_aggiornamento)
        print(f"Numero iterazioni: {numeroIterazioni}")
        #qua termina il while

    for y in x_f_ordinati:
        print(y)
    print(f"Numero iterazioni: {numeroIterazioni}")
    return (minimoCorrente, xDelMinimo, indice_aggiornamento, x_f_ordinati)

def grafico(lista, f, intervallo, minimo, minimo_x):
    # Generare una serie di valori x
    x_values = np.linspace(intervallo[0], intervallo[1], 100)  # Crea 100 punti tra -5 e 5

    # Calcolare i corrispondenti valori y utilizzando la funzione
    y_values = f(x_values)
    if np.isscalar(y_values):  # Controlla se y_values è uno scalare (funzione costante)
        y_values = np.full_like(x_values, y_values)

    plt.plot(x_values, y_values, label='f(x)')

    # Aggiungi tutti i punti in blu tranne l'ultimo che sarà rosso
    for i, elem in enumerate(lista[:-1]):
        plt.scatter(elem[0], elem[1], color='blue', label=f'x_{i}', linestyle='--')

    # Aggiungi l'ultimo punto in rosso
    plt.scatter(minimo_x, minimo, color='red', label='Punto di minimo', linestyle='--')

    # Aggiungi una linea verticale tratteggiata lungo il punto rosso
    plt.plot([minimo_x, minimo_x], [0, minimo], color='red', linestyle='--')

    # Aggiungi una linea tratteggiata sotto l'asse delle y negative
    plt.axhline(y=0, color='gray', linestyle='--')

    # Aggiungi etichette agli assi
    plt.xlabel('x')
    plt.ylabel('f(x)')

    # Aggiungi numeri sull'asse x con passo di 0.5
    plt.xticks(np.arange(intervallo[0], intervallo[1] + 0.5, 0.5), fontsize=8, rotation=45)

    # Aggiungi un titolo
    plt.title('Grafico della funzione f(x)')

    # Aggiungi un riquadro in alto a destra con le coordinate del minimo
    plt.text(0.95, 0.95, f'Coordinate Minimo:\nx = {minimo_x}\ny = {minimo}', fontsize=8, horizontalalignment='right', verticalalignment='top', transform=plt.gca().transAxes, bbox=dict(facecolor='white', edgecolor='black', boxstyle='round,pad=0.5'))

    # Mostrare il grafico
    plt.show()


'''
def funzione_di_test(x):
    return (1/6)*x**6 - (52/25)*x**5 + (39/80)*x**4 + (71/10)*x**3 - (79/20)*x**2 - x + (1/10)

# Funzione 1: sin(x) + sin(10x/3)
def funzione_1(x):
    return math.sin(x) + math.sin(10*x/3)

# Funzione 2: Sommatoria con k che va da 1 a 5 di k*sin[(k+1)x + k]
def funzione_2(x):
    result = 0
    for k in range(1, 6):
        result += k * math.sin((k+1)*x + k)
    return result

# Funzione 3: -(16x^2-24x+5)*e^(-x)
def funzione_3(x):
    return -(16*x**2 - 24*x + 5) * math.exp(-x)

# Funzione 4: (3x - 1.4) * sin(18x)
def funzione_4(x):
    return (3*x - 1.4) * math.sin(18*x)

# Funzione 5: -(x+sinx)*e^(-x^2)
def funzione_5(x):
    return -(x + math.sin(x)) * math.exp(-x**2)

# Funzione 6: sin(x) + sin(10x/3) + ln(x) -0.84x + 3
def funzione_6(x):
    return math.sin(x) + math.sin(10*x/3) + math.log(x) - 0.84*x + 3

# Funzione 7: Sommatoria con k che va da 1 a 5 di k*cos[(k+1)x + k]
def funzione_7(x):
    result = 0
    for k in range(1, 6):
        result += k * math.cos((k+1)*x + k)
    return result

# Funzione 8: sin(x) + sin(2x/3)
def funzione_8(x):
    return math.sin(x) + math.sin(2*x/3)

# Funzione 9: -x * sin(x)
def funzione_9(x):
    return -x * math.sin(x)

# Funzione 10: 2 cos(x) + cos(2x)
def funzione_10(x):
    return 2 * math.cos(x) + math.cos(2*x)

# Funzione 11: sin^3(x) + cos^3(x)
def funzione_11(x):
    return math.sin(x)**3 + math.cos(x)**3

# Funzione 12: -x^(2/3) + (x^2-1)^(1/3)
def funzione_12(x):
    return -x**(2/3) + (x**2 - 1)**(1/3)

# Funzione 13: -e^(-x) * sin(2 * pi * x)
def funzione_13(x):
    return -math.exp(-x) * math.sin(2 * math.pi * x)

# Funzione 14: (x^2-5x+6)/(x^2+1)
def funzione_14(x):
    return (x**2 - 5*x + 6) / (x**2 + 1)

# Funzione 15: 2(x-3)^2 + e^(0.5x^2)
def funzione_15(x):
    return 2 * (x - 3)**2 + math.exp(0.5 * x**2)

# Funzione 16: x^6-15x^4+27x^2+250
def funzione_16(x):
    return x**6 - 15*x**4 + 27*x**2 + 250

# Funzione 17: { (x=2)^2 se x<= 3; 2*ln(x-2) + 1 se x > 3 }
def funzione_17(x):
    if x <= 3:
        return (x - 2)**2
    else:
        return 2 * math.log(x - 2) + 1

# Funzione 18: -x + sin(3x) - 1
def funzione_18(x):
    return -x + math.sin(3*x) - 1

# Funzione 19: (sin(x)-x)*e^(-x^2)
def funzione_19(x):
    return (math.sin(x) - x) * math.exp(-x**2)
'''

def funzione_di_test(x):
    return 3
    #return funz.f9(x)

intervallo = (3.1,20.4)  # Intervallo su cui lavorare
#intervallo = (1,2)
L = 1.7  # Parametro L per il calcolo di x_cappuccio e Ri
epsilon = 0.0001  # Valore di epsilon 10^-5
funzione = funzione_di_test  # Funzione da testare

# Invocazione dell'algoritmo metodo_numerico
minimo, minimo_x, indice_aggiornamento, indici_ordinati = Piyavskii_Shubert(intervallo, L, epsilon, funzione)

# Stampa dei risultati

print(f"Coordinate del punto di minimo :  ")
print(f"x = {minimo_x}")
print(f"y = {minimo}")
#print(f"Indice di aggiornamento: {indice_aggiornamento}")

# Generazione e visualizzazione del grafico
grafico(indici_ordinati, funzione, intervallo,minimo,minimo_x)