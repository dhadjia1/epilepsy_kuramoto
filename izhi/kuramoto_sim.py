import matplotlib.pyplot as plt
import numpy as np
import cmath, sys
from scipy.integrate import odeint
from scipy.signal import hilbert



class kuramoto_network(object):

    def __init__(self, noscillators, frequencies, coupling, seed=0):
        self.__noscillators = noscillators
        self.__frequency = frequencies

        if isinstance(coupling, float):
            self.__coupling  = (np.ones((noscillators, noscillators)) - np.identity(noscillators)) * coupling
        else: self.__coupling = coupling
        self.__random = np.random.RandomState(seed)
        self.__phase  = self.__random.uniform(-np.pi, np.pi, size=(self.__noscillators, ))

    def simulate(self, steps, time):
        dynamic_phase = [self.__phase]
        dynamic_time  = [0.0]
  
        init_r, init_mp    = self.__calculate_synchrony(self.__phase)
        dynamic_synchrony  = [init_r]
        dynamic_mean_phase = [init_mp % 2.*np.pi]
        step     = time / steps
        int_step = step / 10.

        for t in np.arange(step, time + step, step):
            self.__phase = self.__calculate_kuramoto(t, step, int_step)
            curr_r, curr_mp = self.__calculate_synchrony(self.__phase)
            dynamic_synchrony.append(curr_r)
            dynamic_mean_phase.append(curr_mp % 2.*np.pi)
            dynamic_phase.append(self.__phase)
            dynamic_time.append(t)
        dynamic_time = np.asarray(dynamic_time)
        dynamic_phase = np.asarray(dynamic_phase)
        return dynamic_phase, dynamic_mean_phase, dynamic_synchrony, dynamic_time

    def __calculate_synchrony(self,phases):
        mean_phase = np.mean(phases)
        exp_mean_phase = cmath.exp(1j*mean_phase)
        summ = 1./len(phases) * np.sum([cmath.exp(1j*phi) for phi in phases])
        return np.absolute(np.divide(summ, exp_mean_phase)), mean_phase

    def __calculate_kuramoto(self, t, step, int_step):
        next_phases = np.zeros((self.__noscillators,))
        for index in range(0, self.__noscillators):
            result = odeint(self.__calculate_dphidt, [self.__phase[index]], np.arange(t - step, t, int_step), (index,))
            next_phases[index] = result[len(result)-1]
        return next_phases

    def __calculate_dphidt(self, phase, t, index):
        coupling_contributions = []
        for i in range(self.__noscillators):
            index_phase, neighbor_phase = self.__phase[index], self.__phase[i]
            pairwise_coupling = self.__coupling[index, i]
            coupling_contributions.append(pairwise_coupling * np.sin(index_phase - neighbor_phase))
        dphidt = self.__frequency[index] - np.sum(coupling_contributions)
        return dphidt

if __name__ == '__main__':

    random = np.random.RandomState(seed=100)


    noscillators = 3
    coupling_A     = random.uniform(0.01, 0.05, size=(noscillators, noscillators))
    for i in range(noscillators):
        coupling_A[i,i] = 0.0
    frequencies_A  = random.normal(20.0, 1.0, size=(noscillators,))
    net_A = kuramoto_network(noscillators, frequencies_A, coupling_A, seed=10)
    os_phases_A, synchrony_A, mean_phase_A, times_A = net_A.simulate(100, 1.)
    os_phases_A = np.asarray(os_phases_A)
    os_angles_A = phase_to_angle(os_phases_A.T, times_A[1] - times_A[0]) # T x N

    amplitudes_A = phase_to_amplitude(os_phases_A.T, frequencies_A, times_A)
    sum_amp_A = np.sum(amplitudes_A, axis=1)
    sum_env_amp_A = np.absolute(hilbert(sum_amp_A))
   

    soma_v_vecs, t_vecs = [], []
    sim_A_cells = [PyramidalCell() for _ in xrange(noscillators)] 
    #sim_A_cells = [hh_neuron.HH_Cell() for _ in xrange(noscillators)]
    for i in xrange(noscillators):
        soma_v_vec, t_vec = run_neuron_simulation(sim_A_cells[i], amplitudes_A[:,i], times_A*1000.)
        soma_v_vecs.append(soma_v_vec)
        t_vecs.append(t_vec)

    np.savetxt('TESTsoma_voltages-20-1.0.txt', soma_v_vecs)
    np.savetxt('TESTtimes-20-1.0.txt', t_vecs)

    plt.figure()
    for i in xrange(len(t_vecs)):
        time = np.asarray(t_vecs[i])
        plt.plot(time, soma_v_vecs[i])

    plt.figure()
    plt.plot(os_angles_A.T)
    plt.figure()
    plt.plot(times_A, synchrony_A)
    plt.show()

    sys.exit(1)

    noscillators = 5
    scaling = 1.5
    coupling_B     = random.uniform(scaling*0.01, scaling*0.05, size=(noscillators, noscillators))
    for i in range(noscillators):
        coupling_B[i,i] = 0.0
    frequencies_B  = random.normal(3.0, 0.1, size=(noscillators,))
    net_B = kuramoto_network(noscillators, frequencies_B, coupling_B, seed=10)
    os_phases_B, synchrony_B, mean_phase_B, times_B = net_B.simulate(100, 10.)
    os_phases_B = np.asarray(os_phases_B)
    os_angles_B = phase_to_angle(os_phases_B.T, times_B[1] - times_B[0]) # T x N

    amplitudes_B  = phase_to_amplitude(os_phases_B.T, frequencies_B, times_B)
    sum_amp_B     = np.sum(amplitudes_B, axis=1)
    sum_env_amp_B = np.absolute(hilbert(sum_amp_B))

    fig, (ax, ax2, ax3, ax4)= plt.subplots(4,1)
    ax.plot(times_A, sum_env_amp_A, label='heterogeneous envelope', c='b', alpha=0.5)
    ax.plot(times_B, sum_env_amp_B, label='homogeneous envelope', c='r', alpha=0.5)
    ax.set_xlabel('time (sec)')
    ax.legend()

    ax2.plot(times_A, sum_amp_A, label='hetergeneous amplitude', c='b', alpha=0.5)
    ax2.plot(times_B, sum_amp_B, label='homogeneous amplitude', c='r', alpha=0.5)
    ax2.legend()

    mean_sync_A = np.mean(synchrony_A)
    mean_sync_B = np.mean(synchrony_B)
    ax3.plot(times_A, synchrony_A, label='heterogeneous synchrony:%0.3f'%mean_sync_A, c='b', alpha=1.0)
    ax3.plot(times_B, synchrony_B, label='homogeneous synchrony:%0.3f'%mean_sync_B, c='r', alpha=1.0)
    ax3.set_xlabel('time (sec)')
    ax3.set_ylabel('synchrony')
    ax3.legend()

    ca1 = lfp_to_ca(sum_env_amp_A, times_A)
    mca1, stdca1 = np.mean(ca1), np.std(ca1)
    ca1z = (ca1 - mca1) / stdca1
    ca2 = lfp_to_ca(sum_env_amp_B, times_B)
    ca2z = (ca2 - mca1) / stdca1

    ax4.plot(times_A, ca1z , 'b', label='ca1')
    ax4.plot(times_B, ca2z , 'r', label='ca2')
    ax4.legend()

    fig, (ax1, ax2) = plt.subplots(2,1)
    ax1.plot(os_angles_A.T)
    ax2.plot(os_angles_B.T)
    plt.show()

    
