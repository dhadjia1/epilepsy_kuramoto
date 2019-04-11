
import numpy as np
from neuron import h, gui
from izhi2019_wrapper import IzhiCell
from utils import *


# Simple network of Izhi neurons that recives spike train input from 'seizure source'
class IzhiNetwork(object):

    def __init__(self, spike_trains, nneurons, delay=0., rate=20., cell_type='RS', p_connection=0.25, stim_weight=0.04, stim_delay=1., dt=0.025):

        self.delay = delay
        self.__dt  = dt
        self.__pconnection = p_connection
        self.__nneurons = nneurons
        self.__stim_weight = stim_weight
        self.__stim_delay  = stim_delay
   
        self.stims, self.ncstims = [ [] for _ in range(self.__nneurons)], [ [] for _ in range(self.__nneurons)]
        self.connections = []
    
        id_start = 0
        self.neurons = self.__instantiate_neurons(self.__nneurons, cell_type, id_start)
        if spike_trains is not None:
            self.__generate_connections(len(spike_trains))
        self.__create_synapses()
        if self.delay > 0.0: 
            self.__generate_random_inputs(rate=rate)
        if spike_trains is not None:
            self.__insert_spiketrains(spike_trains)

    def __instantiate_neurons(self, n, cell_type, id_start):
        return [IzhiCell(cell_type, i + id_start) for i in range(n)]

    def __generate_random_inputs(self, tstart=0., rate=20.):
        for i in range(self.__nneurons):
            synapse = self.neurons[i].synapse
            spikes = generate_poisson_spike_train(rate, self.delay, tstart=tstart, as_array=True)
            for spike in spikes:
                stim = h.NetStim()
                stim.number = 1
                stim.start = spike
                ncstim = h.NetCon(stim, synapse)
                ncstim.delay = self.__stim_delay
                ncstim.weight[0] = self.__stim_weight
         
                self.ncstims[i].append(ncstim)
                self.stims[i].append(stim)
        
    
    def __generate_connections(self, N):
        for i in range(N):
            random  = np.random.RandomState(seed=i)
            choices = random.choice([0,1], p=[1.-self.__pconnection, self.__pconnection], size=(self.__nneurons,))
            self.connections.append(choices)
            
    def __create_synapses(self):
        for cell in self.neurons:
            syn = h.ExpSyn(cell.sec(0.5))
            syn.tau = 2.0
            cell.synapse = syn

    def __insert_spiketrains(self, spike_trains):
        for i in range(len(spike_trains)): # number neurons in 'seizure' layer
            spike_train = spike_trains[i]
            for (j,conn) in enumerate(self.connections[i]): # each neuron will receive spikes from 'seizure layer'
                if conn == 1:
                    synapse = self.neurons[j].synapse
                    for time in spike_train:
                        stim = h.NetStim()
                        stim.number = 1
                        stim.start = time + self.delay
                        ncstim = h.NetCon(stim, synapse)
                        ncstim.delay = self.__stim_delay
                        ncstim.weight[0] = self.__stim_weight
                        
                        self.ncstims[j].append(ncstim)
                        self.stims[j].append(stim)
 
    
    def simulate_izhi_network(self, tstop=1000., verbose=False):
        from copy import deepcopy
        tstop += self.delay

        t_vec = h.Vector()
        t_vec.record(h._ref_t)

        v_vecs = [h.Vector() for _ in range(len(self.neurons))]
        for curr_v, neuron in zip(v_vecs, self.neurons):
            curr_v.record(neuron.sec(0.5)._ref_v)
       
        h.dt = self.__dt     
        h.tstop = tstop
        h.run()

        v_vecs = [deepcopy(list(v_vec)) for v_vec in v_vecs]
        t_vec = deepcopy(list(t_vec))

        return np.asarray(v_vecs), np.asarray(t_vec)
        
        
        
       
        
        
