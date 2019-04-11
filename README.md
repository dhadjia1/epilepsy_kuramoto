# epilepsy_kuramoto

Goal: Use coupled kuramoto oscillators to investigate emergence of network phenomenon in izhikevich neurons.

Kuromoto model is of the form:<br><br>
<img src="https://github.com/dhadjia1/epilepsy_kuramoto/blob/master/izhi/pics/CodeCogsEqn.gif">
<br><br>where $\phi$ is phase, $\w_{i}$ is the natural frequency of oscillator $i$, and $K_{ij}$ is the coupling between oscillator $i$ and $j$.

Figure 1: N = 10 oscillators. Sim A: weak coupling with heterogeneous frequencies. Sim B: less weak coupling, more homogeneous frequencies. (Top) Sum of oscillator amplitudes vs time. (Bottom) Population synchronization over time.

<img src="https://github.com/dhadjia1/epilepsy_kuramoto/blob/master/izhi/pics/kurosim-synch.png">

Figure 2: Instantaneous oscillator frequency vs time. Synchronization from Fig 1 plotted in light black. (Top) Sim A; (Bottom) Sim B

<img src="https://github.com/dhadjia1/epilepsy_kuramoto/blob/master/izhi/pics/kurosim-fslide.png">

Figure 3: The amplitude of oscillators acquired from the Kuromoto model were used as input into Izikevich (Regular spiking) neurons via current clamp and simulated for 5000 m.s. (Top) Instantaneous firing rate calculated from kernel convolution for neurons in simulation A (red) and B (blue). (Middle) Raster plot of neurons in simulation A along with sliding histogram of spikes/s. (Bottom) Same as MIDDLE but for neurons in simulation B.

<img src="https://github.com/dhadjia1/epilepsy_kuramoto/blob/master/izhi/pics/spikes.png">


Figure 4: (Top) Coefficient of variation (CV) and Inter Spike Interval (ISI) acquired from raster plots of Figure 3.

<img src="https://github.com/dhadjia1/epilepsy_kuramoto/blob/master/izhi/pics/isi-cv.png">
