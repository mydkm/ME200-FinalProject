import numpy as np
import matplotlib.pyplot as plt

#Defining Variables
m_0= 1000  #Rocket's intitial mass in kg
v_0= 343 #Rocket's initial speed in m/s
g=-9.81 #gravity m/s^2
c= 7
#t= function for finding time 
t=((((2)**(1/2))-1)/(g))*v_0
dm = -1500 #kg/s
m=np.linspace(0,5300000, 1000000)
m_t= c*t+m_0 #Check Scipy
F_g= m*g
#dv=
#v #T_eff(t,h)-Effective Thrust
T_eff=(m*(dv))+((dm)*v)+F_g

#Unresolved Problems so far: 
#1. Read Scipy docs to figure out how code will solve for: m_t(mass as a function of time), v_t(velocity as a function of time)
#and finally how it'll solve for thrust
#2. After thrust is solved for, using stuff solved above, it'll be plugged into the kinematic equation to find suicde burn height

