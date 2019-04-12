#Erin Aadland
#AST 520
#Week 4

import math as ma
import matplotlib.pyplot as plt
import numpy as np
from astropy.io import ascii
import time

#Gravitational wave signature as a function of time
#Calculates gravity from the input parameters
# t - time
# m - black hole mass in solar masses
# ti - characteristic infall time
# tr - characteristic ringdown time
# t0 - time of the merger
# P - period of the wave
def grav(t, m, ti, tr, t0, P):
    if t < t0:
        g = m/30 * ma.exp((t-t0)/ti) * ma.cos(2 * ma.pi * (t-t0) / P)
    elif t >= t0:
        g = m/30 * ma.exp((t0-t)/tr) * ma.cos(2 * ma.pi * (t-t0) / P)
    return g
    
#Start run time
start_time=time.time()
start_clock = time.clock()
################ Test Plot ####################################
#Initialize variables
m = 30       #black hole mass in solar masses
ti = 0.05    #characteristic infall time
tr = 0.02    #characteristic ringdown time
t0 = 1      #time of the merger
P = 0.03     #period

#Initialize a list for the times and gravity
g = []
t = []
#Calculate the gravity for times between 0 and 2 seconds
for i in np.linspace(0,2,2000):
    g.append(grav(i,m,ti,tr,t0,P))
    t.append(i)

#Plot up the results
plt.plot(t,g)
plt.xlabel('time [s]')
plt.ylabel('g(t)')
plt.show()
plt.close()


################ Actual Data #####################################
#Read in data
data = ascii.read("gravwave.out", data_start=0, delimiter=' ')

#Define the time and signal from catalog
times = np.array(data['col1'])
signal = np.array(data['col2'])

#Create list of Periods in range of 2ms to 122ms in 4ms increments
Periods = list(range(2,122,4))

#Initialize result arrays for maximum value and time for each Period
max_time = np.zeros(len(Periods))
max_value = np.zeros(len(Periods))

#Initialize a count
ct = 0
#Going through Periods 
for Pi in Periods:
    #Convert P to seconds
    P = Pi / 1000
    
    #Make an array for the gravity calculations
    gravity = np.zeros(len(times))
    for i in range(len(times)):
        gravity[i] = grav(times[i], m, ti, tr, t0, P)
    
    #Calculate the Fourier Transform for the gravity and signal
    ft_gravity = np.fft.fft(gravity)
    ft_signal = np.fft.fft(signal)
    #Multiply the gravity and signal Fourier Transforms together
    mult = ft_gravity * ft_signal
    #Take the inverse Fourier Transform of the product
    ift_mult = np.fft.ifft(mult)
    #Find the absolute values of the inverse Fourier Transform
    abs_mult = abs(ift_mult)
    #Find the maximum value and time
    max_ab = np.max(abs_mult)
    index = np.where(abs_mult == max_ab)
    max_t = times[index]
    #Append maximum values to result arrays
    max_value[ct] = max_ab
    max_time[ct] = max_t
    
    #Increase the count
    ct += 1
    
    print('Period: ', Pi)
    #Plot the time vs. absolute values
    plt.plot(times, abs_mult, '.')
    plt.xlabel('time [s]')
    plt.ylabel('convolution of g(t)')
    plt.show()
    plt.close()


#Print the time of the actual maximum convolution
max_v = np.max(max_value)
max_i = np.where(max_value == max_v)
max_ti = max_time[max_i]
best_P = Periods[max_i[0][0]]

print('Best Period: ', best_P)
print('Time of Merger: ', max_ti)

#PLot up all the maximum values and periods
plt.plot(Periods, max_value, '.')
plt.xlabel('period [ms]')
plt.ylabel('convolution of g(t)')
plt.show()
plt.close()


#PLot up all the maximum values and times
plt.plot(max_time, max_value, '.')
plt.xlabel('time [s]')
plt.ylabel('convolution of g(t)')
plt.show()
plt.close()

#PLot up all the maximum values and times
plt.plot(max_time, max_value, '.')
plt.xlabel('time [s]')
plt.ylabel('convolution of g(t)')
plt.xlim(12000, 14000)
plt.show()
plt.close()

#Print out the time it took for the code to run
print('time: ',time.time()-start_time)
print('CPU time: ',time.clock()-start_clock)
