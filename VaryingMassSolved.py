#We're solving a boundary problem
#We're going backwards from thrust to find min Height
#Need to plot Thrust vs time, vs mass
#Mass vs time
#Thrust vs Height
#Etc, we should have lots of plots
#v_entry=should be falcon 9 data


#1. Modelling Mass is a start:
# dm/dt=m_dot
#So integrating leaves us with  m(t)=-m_0 - m_dot*t
import matplotlib.pyplot as plt
import numpy as np
from scipy.integrate import solve_ivp


#But what is Falcon 9's m_o=?
m_dry= 23000 #dry mass=mass without fuel(kg)
n=0.06 #estimated percentage of fuel burn for descent based on questionable sources(kg)
Prop= 395700 #Total Propellant used  https://en.wikipedia.org/wiki/Falcon_9?utm_source

m_0= m_dry + 0.06*Prop
print(m_0, "kg")
#Fuel burn rate for each engine =300kg/s, assuming it's constant . 
#We're assuming 2 engines:
m_dot= 600 #kg/s
t_burn= 162 #Burn time according to Falcon9 Wiki 
m_t= m_0 - m_dot*t_burn 

# Define ODE: dm/dt = -m_dot
def mass_ode(t, m_t):
    if m_t[0] <= m_dry:
        return [0]          # stop burning 
    else:
        return [-m_dot]

# Solve ODE
t_span = (0, t_burn)
y0 = [m_0]
t_eval = np.linspace(0, t_burn, 500)

sol = solve_ivp(mass_ode, t_span, y0, t_eval=t_eval)

# Plot mass vs time
plt.figure(figsize=(8,5))
plt.plot(sol.t, sol.y[0])
plt.xlabel('Time [s]')
plt.ylabel('Mass [kg]')
plt.title('Rocket Mass vs Time During Landing Burn')
plt.grid(True)
plt.show()

#Okay great, we now have m(t), which can be used for thrust. 
#2. Need v(t) and dv/dt
#3. Plotting Thrust vs burn time
#4. Using Kinematics to find min. hieght from v(t) 









