# epilepsy_kuramoto

uses coupled kuramoto oscillators to investigate emergence of network phenomenon in izhikevich neurons.

Kuromoto model is of the form: $\frac{d\phi}{dt} = w_{i} - \sum_{j=1}{N}K_{ij}sin(\phi_{i} - \phi_{j} $, where $\phi$ is phase, $\w_{i}$ is the natural frequency of oscillator $i$, and $K_{ij}$ is the coupling between oscillator $i$ and $j$.

Figure 1: N = 10 oscillators. Sim A: weak coupling with heterogeneous frequencies. Sim B: less weak coupling, more homogeneous frequencies. (Top) Sum of oscillator amplitudes vs time. (Bottom) Population synchronization over time.

<img src="https://github.com/dhadjia1/epilepsy_kuramoto/blob/master/izhi/pics/kurosim-synch.png">

Figure 2: Instantaneous oscillator frequency vs time. Synchronization from Fig 1 plotted in light black. (Top) Sim A; (Bottom) Sim B

<img src="https://github.com/dhadjia1/epilepsy_kuramoto/blob/master/izhi/pics/kurosim-fslide.png">

Figure 3: Oscillators acquired from Kuromoto were input into Izikevich (Regular spiking) neurons via current clamp and simulated for 5000 m.s.

<img src="https://github.com/dhadjia1/epilepsy_kuramoto/blob/master/izhi/pics/spikes.png">


Figure 3: Coefficient of variation of Inter Spik

<img src="https://github.com/dhadjia1/epilepsy_kuramoto/blob/master/izhi/pics/kurosim-fslide.png">
