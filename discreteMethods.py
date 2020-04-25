import numpy as np

def norm_n(vec,n):
  """
  norm_n(vec,n) -> float
  
  Determines the nth norm of the vector.
  Vector must have one or two dimensions.
  
  Params
  ----------
      vec   -> vector
      n     -> (int) norm degree
  
  Return
  -------
      res   -> (float) value of norm n
  """
  
  res = 0
  if np.size(np.shape(vec)) == 1:
    for k in vec:
      res += abs(k)**n
  else:
    for k in vec:
      for w in k:
        res += abs(w)**n
  return res**(1/n)

def norm_inf(vet):
  """
  norm_inf(vet) -> float
  
  Determines the inifity norm of the vector.
  Vector must have one or two dimensions.
  
  Params
  ----------
      vet   -> vector
  
  Return
  -------
      res   -> (float) volue of the infinity norm
  """
  
  if np.size(np.shape(vet)) == 1:
    return max(abs(vet))
  else:
    return max(max(abs(vet)))

def D3(f,x,h):
  """
  D3(f,x,h) -> float

  Central finite difference method
  Determines the first derivate of order 3 of the function
  f at point x with step h  
  
  Params
  ----------
      f   -> (function) to be derivate
      x   -> (float) derivate point of interest
      h   -> (float) step size
  
  Return
  -------
      res -> (float) value of the derivate of f at point x
  """
  return 1/6/h*(2*f(x+h) + 3*f(x) - 6*f(x-h) + f(x-2*h))


def secant(f,x0,x1,n,tol):
  """
  secant(f,x0,x1,n,tol) -> float
  
  Secant method to find the root of the equation f
  closest to the interval [x0,x1]
  
  Params
  ----------
      f     -> (function) of interest
      x0    -> (float) first interval value
      x1    -> (float) second interval value
      n     -> (int) max number of iterations
      tol   -> (float) stop criterion tolerance
  
  Return
  -------
      x1    -> (float) root of the equation
  """

  for _ in range(n):
    if abs(f(x1)-f(x0)) <= tol:
      return x1
    x_temp = x1 - f(x1)*(x1-x0)/(f(x1)-f(x0))
    x0 = x1
    x1 = x_temp
  return x1

def forward_euler_n(fn,a,b,y_0,m):
  """
  forward_euler_n(fn,a,b,y_0,m) -> [list,list[...]]
  
  Forward Euler method for solving an IVP in the form:
       / dx1/dt = f1(x1,...,xn)
       | ...
       | dxn/dt = fn(x1,...,xn)
       | x1(a) = x1_0
       | ...
       \ xn(a) = xn_0
  
  Params
  ----------
      fn    -> (list) of functions to be solved
      a     -> (float) initial interval value
      b     -> (float) end interval value
      y_0   -> (list) of values of the functions at point a
      m     -> (int) number of steps
  
  Return
  -------
      vett  -> (Array NumPy) of independent variables
      vety  -> (Arrays of Array NumPy) solution values of each function
  """
  
  n = len(fn)
  x = [0 for k in y_0]
  
  h = (b-a)/m
  t = a
    
  vett = np.zeros(m)
  vety = [np.zeros(m) for k in range(n)]
  
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

def backward_euler(f,a,b,y0,m):
  """
  backward_euler(f,a,b,y0,m) -> [list,list]
  
  Backward Euler method for solving an IVP in the form:
       / y'(x) = f(x,y)
       \ y(a) = y0
  
  Params
  ----------
      f  -> (function) to be solved
      a  -> (float) initial interval value
      b  -> (float) end interval value
      y0 -> (float) value of the function at point a
      m  -> (int) number of steps
  
  Return
  -------
      x  -> (Array NumPy) independent variable
      y  -> (Array NumPy) solution value of the function
  """

  h = (b-a)/m
  x = np.zeros(m+1)
  y = np.zeros(m+1)
  x[0] = a
  y[0] = y0
  
  for k in range(m):
    x[k+1] = x[k] + h
    f_temp = lambda Y: Y - y[k] - h*f(x[k+1],Y)
    y[k+1] = secant(f_temp,y[k],y[k] + 2*h*f(x[k],y[k]),50,0.0001)

  return x,y

def rk4(f,a,b,x0,m):
  """
  rk4(f,a,b,x0,m) -> [list,list]
  
  Runge-Kutta 4th order method for solving an IVP in the form:
        / x'(t) = f(t,x)
        \ x(a) = t0
  
  Params
  ----------
      f  -> (function) to be solved
      a  -> (float) initial interval value
      b  -> (float) end interval value
      x0 -> (float) value of the function at point a
      m  -> (int) number of steps
  
  Return
  -------
      t     -> (Array NumPy) independent variable
      x     -> (Array NumPy) solution value of the function
  """ 

  h = (b-a)/m

  t = np.zeros(m)
  x = np.zeros(m)
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

def rkf(f,a,b,y0,tol,hmax,hmin,inHigher,inLower):
  """
  rkf(f,a,b,y0,tol,hmax,hmin,inHigher,inLower) -> [array(),array()]
  
  Runge-Kutta-Fehlberg which combines RK 4th and RK 5h
  order for solving an IVP in the form:
        / y'(x) = f(x,y)
        \ y(a) = y0
  
  RK5 for truncation error:
        y5_i+1 = y5_i + d1*k1 + d3*k3 + d4*k4 + d5*k5 + d6*k6
        
  RK4 for local error estimation:
        y4_i+1 = y4_i + c1*k1 + c3*k3 + c4*k4 + c5*k5
        
  Evaluations:
        k1 = h * f( x, y )
        k2 = h * f( x + a2 * h , y + b21 * k1)
        k3 = h * f( x + a3 * h , y + b31 * k1 + b32 * k2)
        k4 = h * f( x + a4 * h , y + b41 * k1 + b42 * k2 + b43 * k3 )
        k5 = h * f( x + a5 * h , y + b51 * k1 + b52 * k2 + b53 * k3 + b54 * k4 )
        k6 = h * f( x + a6 * h , y + b61 * k1 + b62 * k2 + b63 * k3 + b64 * k4 + b65 * k5 )
  

  Params
  ----------
      f        -> (function) to be solved
      a        -> (float) initial interval value
      b        -> (float) end interval value
      y0       -> (float) value of the function at point a
      tol      -> (float) tolerance for truncation error
      hmax     -> (float) max step size
      hmin     -> (float) min step size
      inHigher -> (float) higher increment for step variation
      inLower  -> (float) lower increment for step variation

  Return
  -------
      X     -> (Array NumPy) independent variable
      Y     -> (Array NumPy) solution value of the function
      P     -> (Array NumPy) of steps used according to X
  """

  # Coefficients related to the independent variable of the evaluations
  a2  =   2.500000000000000e-01  #  1/4
  a3  =   3.750000000000000e-01  #  3/8
  a4  =   9.230769230769231e-01  #  12/13
  a5  =   1.000000000000000e+00  #  1
  a6  =   5.000000000000000e-01  #  1/2

  # Coefficients related to the dependent variable of the evaluations
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

  # Coefficients related to the truncation error
  # Obtained through the difference of the 5th and 4th order RK methods:
  #     R = (1/h)|y5_i+1 - y4_i+1|
  r1  =   2.777777777777778e-03  #  1/360
  r3  =  -2.994152046783626e-02  # -128/4275
  r4  =  -2.919989367357789e-02  # -2197/75240
  r5  =   2.000000000000000e-02  #  1/50
  r6  =   3.636363636363636e-02  #  2/55

  # Coefficients related to RK 4th order method
  c1  =   1.157407407407407e-01  #  25/216
  c3  =   5.489278752436647e-01  #  1408/2565
  c4  =   5.353313840155945e-01  #  2197/4104
  c5  =  -2.000000000000000e-01  # -1/5

  # Init x and y with initial values a and y0
  # Init step h with hmax, taking the biggest step possible
  x = a
  y = y0
  h = hmax

  # Init vectors to be returned
  X = np.array( [x] )
  Y = np.array( [y] )
  P = np.array( [h] )

  while x < b - h:

      # Store evaluation values
      k1 = h * f( x, y )
      k2 = h * f( x + a2 * h, y + b21 * k1 )
      k3 = h * f( x + a3 * h, y + b31 * k1 + b32 * k2 )
      k4 = h * f( x + a4 * h, y + b41 * k1 + b42 * k2 + b43 * k3 )
      k5 = h * f( x + a5 * h, y + b51 * k1 + b52 * k2 + b53 * k3 + b54 * k4 )
      k6 = h * f( x + a6 * h, y + b61 * k1 + b62 * k2 + b63 * k3 + b64 * k4 + b65 * k5 )

      # Calulate local truncation error
      r = abs( r1 * k1 + r3 * k3 + r4 * k4 + r5 * k5 + r6 * k6 ) / h
      # If it is less than the tolerance, the step is accepted and RK4 value is stored
      if r <= tol:
          x = x + h
          y = y + c1 * k1 + c3 * k3 + c4 * k4 + c5 * k5
          X = np.append( X, x )
          Y = np.append( Y, y )
          P = np.append( P, h )

      # Prevent zero division
      if r == 0: r = P[-1]
      
      # Calculate next step size
      h = h * min( max( 0.84 * ( tol / r )**0.25, inLower ), inHigher )

      # Upper limit with hmax and lower with hmin
      h = hmax if h > hmax else hmin if h < hmin else h

  return X,Y,P