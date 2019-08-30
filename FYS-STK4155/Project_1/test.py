import ast2000tools.utils as utils
seed = utils.get_seed('andrjm')
from ast2000tools.solar_system import SolarSystem
system = SolarSystem(seed)
import ast2000tools.constants as const

"""
print('My system has a {:g} solar mass star with a radius of {:g} kilometers.'
      .format(system.star_mass, system.star_radius))

for planet_idx in range(system.number_of_planets):
    print('Planet {:d} is a {} planet with a semi-major axis of {:g} AU.'
          .format(planet_idx, system.types[planet_idx], system.semi_major_axes[planet_idx]))

#times, planet_positions = ... # Your own orbit simulation code
#system.generate_orbit_video(times, planet_positions, filename='orbit_video.xml')

One astronomical unit is defined as 149597870700.0 meters.
My system has a 0.397599 solar mass star with a radius of 386600 kilometers.
Planet 0 is a rock planet with a semi-major axis of 0.122164 AU.
Planet 1 is a rock planet with a semi-major axis of 0.181375 AU.
Planet 2 is a gas planet with a semi-major axis of 1.07903 AU.
Planet 3 is a rock planet with a semi-major axis of 0.0511407 AU.
Planet 4 is a rock planet with a semi-major axis of 0.757424 AU.
Planet 5 is a rock planet with a semi-major axis of 0.092799 AU.
Planet 6 is a gas planet with a semi-major axis of 0.549964 AU.
"""


# OPPGAVE 1
import numpy as np
masses = system.masses
radius = system.radii
solarmass = const.m_sun

#Escape velocity
m_hp = masses[0]*solarmass; r_hp = radius[0]; gamma = const.G
v_esc = np.sqrt(2*gamma*m_hp/r_hp)
#print('The escape velocity for my home planet is %g km/s.' %v_esc)
"""Terminal> The escape velocity for my home planet is 289681 km/s."""


# OPPGAVE 2
n = 100 # Number of particles
L = 1e-6 # Meters
m_p = 3.3474472e-27 # kg
k = 1.38064852e-23 # Boltzmanns constant
T = 10000 # Temperature in Kelvin
sigma = np.sqrt(k*T/m_p) # Standar deviation
mean_vel = 0 # Mean per velocity component

p =  np.random.uniform(0, L, size = (int(n), 3)) # Position array
v =  np.random.normal(mean_vel, sigma, size = (int(n), 3)) # Velocity array


# OPPGAVE 3
# a)
E_k = np.zeros(n)

i = 0
sum_E = 0
for vel in v:
    E_k[i] = m_p*np.linalg.norm(vel)**2/2
    sum_E += E_k[i]
    i = i+1

mean_E_k = sum_E/n
rel_E_k = (mean_E_k - (3*k*T/2))/ (3*k*T/2)
if np.abs(rel_E_k) < 1e-2:
    print('The mean value of the kinetic energi is equal to (3/2)kT.')
else:
    print('The mean value of the kinetic energi is not equal to (3/2)kT.')
#print(rel_E_k)

# b)
sum_v = 0
for vel in v:
    sum_v += np.linalg.norm(vel)

mean_absvel = sum_v/n
rel_absvel = (mean_absvel - np.sqrt(8*k*T/(np.pi*m_p)))/np.sqrt(8*k*T/(np.pi*m_p))
if np.abs(rel_absvel) < 1e-2:
    print('The mean value of the absolute velocity is equal to $\sqrt{8kT/(\pi m)}$.')
else:
    print('The mean value of the absolute velocity is not equal to $\sqrt{8kT/(\pi m)}$.')
#print(rel_absvel)


# OPPGAVE 4 OG OPPGAVE 5 a) OG OPPGAVE 6
dt = 1e-12
tot_dt = 1e-9
time = np.linspace(0, tot_dt, 1000)
i = 0
col = 0 # Number of collisions
momentum = 0 # Momentum left on the wall
count = 0 # Number of particles escaping
mom_x = 0 # Momentum for the particles that escape in the x-direction

"""
#This loop gets the same results as other people has gotten (for exercise 4 and 5a)


for t in time:
    p += v*dt
    for j in range(len(p)):
        if p[j][0] > L and v[j][0] > 0:
            col += 1
            momentum += np.abs(m_p*v[j][0])
        for k in range(len(p[j])):
            if p[j][k] > L and v[j][k] > 0 or p[j][k] < 0 and v[j][k] < 0:
                v[j][k] = -v[j][k]
"""


for t in time:
    p += v*dt
    for j in range(len(p)):
        if p[j][0] > L and v[j][0] > 0:
            col += 1
            momentum += np.abs(m_p*v[j][0])
            if p[j][1] < L/2 and p[j][1] > 0 and p[j][2] < L/2 and p[j][2] > 0:
                count += 1
                mom_x += np.abs(m_p*v[j][0])
                p[j] = [1E-21, L - 1E-21, L - 1E-21]
            else:
                v[j][0] = -v[j][0]
        elif p[j][0] < 0 and v[j][0] < 0:
            v[j][0] = -v[j][0]
        for k in range(1,3):
            if p[j][k] > L and v[j][k] > 0 or p[j][k] < 0 and v[j][k] < 0:
                v[j][k] = -v[j][k]


print('col = %g, mom = %g, count = %g, mom_x = %g' %(col, momentum, count, mom_x))
F = 2*momentum/tot_dt
P_1 = F/L**2 # Gas pressure on one wall


# OPPGAVE 5 b)
k = 1.38064852e-23
P_2 = n*k*T/L**3
diff_P = P_1 - P_2
print('The lack of agreement between the analytical pressure and the numerical pressure is %g, which is relatively small.' %np.abs(diff_P))
print('pressure = %g, %g'%(P_1, P_2))


# OPPGAVE 7
m_r = 1000 # rocket mass in kg
dv = mom_x/m_r
print('dv = %g' %dv)


# OPPGAVE 8
tot_time = 1200 # seconds
"""
tot_dt*multiplum = tot_time

p*multiplum*boxes = m*v_esc,
"""
multiplum = tot_time/tot_dt
boxes = v_esc*m_r/(mom_x*multiplum)
print('Number of boxes needed for acceleration to escape velocity within 20 minutes is %g.' %boxes)


# OPPGAVE 9
tot_mass_loss = count*multiplum*boxes*m_p
print('Total fuel needed is %g kg.' %tot_mass_loss)
