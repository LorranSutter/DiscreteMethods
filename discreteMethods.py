from numpy import *

# ---------- Escrito em Python 3 ---------- #

"""
Biblioteca criada com o intuito de armazenar
procedimentos uteis na disciplina de 
Introducao aos Metodos Discretos
"""

def D3(f,x,h):
  """
  D3(f,x,h) -> float
  
  Determina a derivada primeira de ordem 3 da funcao
  f no ponto x com passo h
  
  Parametros
  ----------
      f   -> funcao a ser derivada
      x   -> float ponto de interesse da derivacao
      h   -> float tamanho do passo
  
  Retorno
  -------
      res -> (float) valor da derivada de f no ponto x
  """
  return 1/6/h*(2*f(x+h) + 3*f(x) - 6*f(x-h) + f(x-2*h))

def norma_n(vet,n):
  """
  norma_n(vet,n) -> float
  
  Determina a enesima norma do vetor passado como parametro.
  O vetor deve ter uma ou duas dimensoes.
  
  Parametros
  ----------
      vet   -> vetor
      n     -> numero da norma
  
  Retorno
  -------
      res   -> (float) valor da norma n
  """
  
  res = 0
  if size(shape(vet)) == 1:
    for k in vet:
      res += abs(k)**n
  else:
    for k in vet:
      for w in k:
        res += abs(w)**n
  return res**(1/n)

def norma_inf(vet):
  """
  norma_inf(vet) -> float
  
  Determina norma infinita do vetor passado como parametro.
  O vetor deve ter uma ou duas dimensoes.
  
  Parametros
  ----------
      vet   -> vetor
  
  Retorno
  -------
      res   -> (float) valor da norma infinita
  """
  
  if size(shape(vet)) == 1:
    return max(abs(vet))
  else:
    return max(max(abs(vet)))

def secante(f,x0,x1,n,tol):
  """
  secante(f,x0,x1,n,tol) -> float
  
  Metodo da secante para encontrar a raiz da equacao f mais
  proxima do intervalo [x0,x1]
  
  Parametros
  ----------
      f     -> funcao que se encontrar a raiz
      x0    -> primeiro valor do intervalo passado como parametro
      x1    -> segundo valor do intervalo passado como parametro
      n     -> numero maximo de iteracoes permitidas
      tol   -> tolerancia utilizada como criterio de parada
  
  Retorno
  -------
      x1    -> (float) raiz da equacao
  """

  for i in range(n):
    if abs(f(x1)-f(x0)) <= tol:
      return x1
    x_temp = x1 - f(x1)*(x1-x0)/(f(x1)-f(x0))
    x0 = x1
    x1 = x_temp
  return x1

def euler_explicito_n(fn,a,b,y_0,m):
  """
  euler_explicito_n(fn,a,b,y_0,m) -> [list,list[...]]
  
  Metodo de Euler Explicito para resolucao de um PVI na forma:
       / dx1/dt = f1(x1,...,xn)
       | ...
       | dxn/dt = fn(x1,...,xn)
       | x1(a) = x1_0
       | ...
       \ xn(a) = xn_0
  
  Parametros
  ----------
      fn    -> lista de funcoes a seres resolvidas
      a     -> valor inicial do intervalo
      b     -> valor final do intervalo
      y_0   -> lista de valores das funcoes no ponto a
      m     -> numero de passos desejado
  
  Retorno
  -------
      vett  -> (Array NumPy) da variavel independente 
      vety  -> (Arrays de Array NumPy) valores das solucoes de cada funcao
  """
  
  n = len(fn)
  x = [0 for k in y_0]
  
  h = (b-a)/m
  t = a
    
  vett = zeros(m)
  vety = [zeros(m) for k in range(n)]
  
  vett[0] = t
  for k in range(n):
    vety[k][0] = x[k]
  
  for k in range(1,m):
    t = a + k*h
    for w in range(n):
      x[w] += h*fn[w](x)
    
    vett[k] = t
    for w in range(n):
      vety[w][k] = x[w]
    
  return vett,vety

def euler_implicito(f,a,b,y0,m):
  """
  euler_implicito(f,a,b,y0,m) -> [list,list]
  
  Metodo de Euler Implicito para resolucao de um PVI na forma:
       / y'(x) = f(x,y)
       \ y(a) = y0
  
  Parametros
  ----------
      f  -> funcao a ser resolvida
      a  -> valor inicial do intervalo
      b  -> valor final do intervalo
      y0 -> valor da funcao no ponto a
      m  -> numero de passos desejado
  
  Retorno
  -------
      x  -> (Array NumPy) da variavel independente 
      y  -> (Array NumPy) de valores da solucao da funcao
  """

  h = (b-a)/m
  x = zeros(m+1)
  y = zeros(m+1)
  x[0] = a
  y[0] = y0
  
  for k in range(m):
    x[k+1] = x[k] + h
    f_temp = lambda Y: Y - y[k] - h*f(x[k+1],Y)
    y[k+1] = secante(f_temp,y[k],y[k] + 2*h*f(x[k],y[k]),50,0.0001)

  return x,y

def rk4(f,a,b,x0,m):
    """
    rk4(f,a,b,x0,m) -> [list,list]
  
    Metodo de Runge-Kutta de quarta ordem para resolucao de um PVI na forma:
         / x'(t) = f(t,x)
         \ x(a) = t0
    
    Parametros
    ----------
        f     -> funcao a ser resolvida
        a     -> valor inicial do intervalo
        b     -> valor final do intervalo
        x0    -> valor da funcao no ponto a
        m     -> numero de passos desejado
    
    Retorno
    -------
        t     -> (Array NumPy) da variavel independente 
        x     -> (Array NumPy) de valores da solucao da funcao
    """ 

    h = (b-a)/m

    t = zeros(m)
    x = zeros(m)
    t[0] = a
    x[0] = x0
    for i in range(m-1):
        k1 = h * f( t[i], x[i] )
        k2 = h * f( t[i] + 0.5 * h, x[i] + 0.5 * k1 )
        k3 = h * f( t[i] + 0.5 * h, x[i] + 0.5 * k2 )
        k4 = h * f( t[i] + h, x[i] + k3 )
        
        x[i+1] = x[i] + ( k1 + 2.0 * ( k2 + k3 ) + k4 ) / 6.0
        
        t[i+1] = a + i*h + h

    return t,x

def rkf(f,a,b,y0,tol,hmax,hmin,insup,ininf):
    """
    rkf(f,a,b,y0,tol,hmax,hmin) -> [array(),array()]
    
    Metodo de Runge-Kutta-Fehlberg que combina RK de 4a e RK
    de 5a ordem para resolucao de um PVI na forma:
         / y'(x) = f(x,y)
         \ y(a) = y0
    
    RK5 para erro de truncameto:
         y5_i+1 = y5_i + d1*k1 + d3*k3 + d4*k4 + d5*k5 + d6*k6
         
    RK4 para estimativa local de erro:
         y4_i+1 = y4_i + c1*k1 + c3*k3 + c4*k4 + c5*k5
         
    Avaliacoes:
         k1 = h * f( x, y )
         k2 = h * f( x + a2 * h , y + b21 * k1)
         k3 = h * f( x + a3 * h , y + b31 * k1 + b32 * k2)
         k4 = h * f( x + a4 * h , y + b41 * k1 + b42 * k2 + b43 * k3 )
         k5 = h * f( x + a5 * h , y + b51 * k1 + b52 * k2 + b53 * k3 + b54 * k4 )
         k6 = h * f( x + a6 * h , y + b61 * k1 + b62 * k2 + b63 * k3 + b64 * k4 + b65 * k5 )
    

    Parametros
    ----------
        f     -> funcao a ser resolvida
        a     -> valor inicial do intervalo
        b     -> valor final do intervalo
        y0    -> valor da funcao no ponto a
        tol   -> tolerancia para estimativa do erro de truncamento
        hmax  -> tamanho maximo limite do passo
        hmin  -> tamanho minimo limite do passo
        insup -> incremento superior para variacao do passo
        ininf -> incremento inferior para variacao do passo

    Retorno
    -------
        X     -> (Array NumPy) da variavel independente
        Y     -> (Array NumPy) de valores da solucao da funcao
        P     -> (Array NumPy) de passos utilizados de acordo com X
    """

    # Coeficientes relacionados a variavel independente das avaliacoes
    a2  =   2.500000000000000e-01  #  1/4
    a3  =   3.750000000000000e-01  #  3/8
    a4  =   9.230769230769231e-01  #  12/13
    a5  =   1.000000000000000e+00  #  1
    a6  =   5.000000000000000e-01  #  1/2

    # Coeficientes relacionados a variavel dependente das avaliacoes
    b21 =   2.500000000000000e-01  #  1/4
    b31 =   9.375000000000000e-02  #  3/32
    b32 =   2.812500000000000e-01  #  9/32
    b41 =   8.793809740555303e-01  #  1932/2197
    b42 =  -3.277196176604461e+00  # -7200/2197
    b43 =   3.320892125625853e+00  #  7296/2197
    b51 =   2.032407407407407e+00  #  439/216
    b52 =  -8.000000000000000e+00  # -8
    b53 =   7.173489278752436e+00  #  3680/513
    b54 =  -2.058966861598441e-01  # -845/4104
    b61 =  -2.962962962962963e-01  # -8/27
    b62 =   2.000000000000000e+00  #  2
    b63 =  -1.381676413255361e+00  # -3544/2565
    b64 =   4.529727095516569e-01  #  1859/4104
    b65 =  -2.750000000000000e-01  # -11/40
 
    # Coeficientes relacionados a estimativa do erro de truncamento.
    # Obtido atraves da diferenca dos metodos de RK de 5a e 4a ordem:
    #     R = (1/h)|y5_i+1 - y4_i+1|
    r1  =   2.777777777777778e-03  #  1/360
    r3  =  -2.994152046783626e-02  # -128/4275
    r4  =  -2.919989367357789e-02  # -2197/75240
    r5  =   2.000000000000000e-02  #  1/50
    r6  =   3.636363636363636e-02  #  2/55

    # Coeficientes relacionados ao metodo de Rk de 4a ordem
    c1  =   1.157407407407407e-01  #  25/216
    c3  =   5.489278752436647e-01  #  1408/2565
    c4  =   5.353313840155945e-01  #  2197/4104
    c5  =  -2.000000000000000e-01  # -1/5

    # Inicializa x e y com os valores iniciais a e y0.
    # Inicializa o passo h com hmax, assumindo maior passo possivel
    x = a
    y = y0
    h = hmax

    # Inicializa os vetores a serem retornados
    X = array( [x] )
    Y = array( [y] )
    P = array( [h] )

    while x < b - h:

        # Armazena os valores das avaliacoes
        k1 = h * f( x, y )
        k2 = h * f( x + a2 * h, y + b21 * k1 )
        k3 = h * f( x + a3 * h, y + b31 * k1 + b32 * k2 )
        k4 = h * f( x + a4 * h, y + b41 * k1 + b42 * k2 + b43 * k3 )
        k5 = h * f( x + a5 * h, y + b51 * k1 + b52 * k2 + b53 * k3 + b54 * k4 )
        k6 = h * f( x + a6 * h, y + b61 * k1 + b62 * k2 + b63 * k3 + b64 * k4 + b65 * k5 )

        # Calcula o erro local de truncamento
        r = abs( r1 * k1 + r3 * k3 + r4 * k4 + r5 * k5 + r6 * k6 ) / h
        # Se for menor que a tolerancia, o passo eh aceito e eh armazenado o valor de RK4
        if r <= tol:
            x = x + h
            y = y + c1 * k1 + c3 * k3 + c4 * k4 + c5 * k5
            X = append( X, x )
            Y = append( Y, y )
            P = append( P, h )

        # Impede divisao por zero
        if r == 0: r = P[-1]
        
        # Calcula o proximo tamanho de passo
        h = h * min( max( 0.84 * ( tol / r )**0.25, ininf ), insup )

        # Limita superiormente com hmax e inferiormente com hmin
        h = hmax if h > hmax else hmin if h < hmin else h

    # endwhile

    return X,Y,P














