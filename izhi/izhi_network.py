
import numpy as np
from neuron import h, gui
from izhi2019_wrapper import IzhiCell
from kuramoto_sim import *
from utils import *


# Simple network of Izhi neurons that recives spike train input from 'seizure source'
class SimpleIzhiNetwork(object):

    def __init__(self, spike_trains, nneurons, delay=0., rate=20., cell_type='RS', p_connection=0.25, stim_weight=0.04, stim_delay=1., dt=0.025):

        self.delay = delay
        self.__dt  = dt
        self.__pconnection = p_connection
        self.__nneurons = nneurons
        self.__stim_weight = stim_weight
        self.__stim_delay  = stim_delay
   
        self.stims, self.ncstims = [ [] for _ in range(self.__nneurons)], [ [] for _ in range(self.__nneurons)]
        self.connections = None
        self.prestim_spikes  = None
    
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
        self.prestim_spikes = []
        for i in range(self.__nneurons):
            synapse = self.neurons[i].synapse
            spikes = generate_poisson_spike_train(rate, self.delay, tstart=tstart, as_array=True)
            self.prestim_spikes.append(spikes)
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
        self.connections = []
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

class IzhiModularizedNetwork(object):

    def __init__(self, nlayer1, nneurons1, nlayer2, nneurons2, nid_start=1, feedback_layer1=False, feedback_layer2=True, prestim_delay=0., rate=20., p_connection=0.25, syn_weight=0.01, syn_delay=1., stim_weight=0.04, stim_delay=1., dt=0.025):

        self.spiketrains = None
        self.cell_attributes = {}
        self.kuramoto_network = {}

        self.__nlayer1 = nlayer1
        self.__nlayer2 = nlayer2
        self.__nneurons1 = nneurons1
        self.__nneurons2 = nneurons2
        self.__feedback_layer1 = feedback_layer1
        self.__feedback_layer2 = feedback_layer2
        self.__prestim_delay = prestim_delay
        self.__rate = rate
        self.__p_connection = p_connection
        self.__syn_weight = syn_weight
        self.__syn_delay  = syn_delay
        self.__stim_weight = stim_weight
        self.__stim_delay = stim_delay
        self.__dt = dt

        nid_start = self.__generate_microcircuit(layer=1, nid_start=nid_start, feedback=self.__feedback_layer1)
        nid_start = self.__generate_microcircuit(layer=2, nid_start=nid_start, feedback=self.__feedback_layer2)
        self.__generate_connections()


    def __generate_microcircuit(self, layer=1, nid_start=1, feedback=True):
        nodes = []
        nmodules, nneurons = None, None
        if layer == 1:
            nmodules, nneurons = self.__nlayer1, self.__nneurons1
        elif layer == 2:
            nmodules, nneurons = self.__nlayer2, self.__nneurons2
          
        for n in xrange(nmodules):
            mod = MicrocircuitModule(layer,nid_start,nneurons)
            e_neurons = mod.excitatory_neurons
            for neuron in e_neurons:
                gid = neuron.cell_obj.cellid
                cell = {}
                cell['layer']  = layer
                cell['module'] = n
                cell['type']  = 'e'
                cell['object'] = neuron
                cell['connections'] = []
                self.cell_attributes[gid] = cell
            i_neurons = mod.inhibitory_neurons
            for neuron in i_neurons:
                gid = neuron.cell_obj.cellid
                cell = {}
                cell['layer'] = layer
                cell['module'] = n
                cell['type'] = 'i'
                cell['object'] = neuron
                cell['connections'] = []
                self.cell_attributes[gid] = cell

            nid_start += (nneurons * 2)
            if feedback:
                for (e,i) in zip(e_neurons, i_neurons):
                    nc = generate_nc_connection(e, i, self.__syn_weight, self.__syn_delay)
                    self.cell_attributes[e.cell_obj.cellid]['connections'].append((i.cell_obj.cellid, nc))
                
        return nid_start

    def __generate_connections(self, init_seed=0, scale=2.5):
        from scipy.stats import norm
            
        random = np.random.RandomState(seed=init_seed)
        l1_neurons = self.__get_layer(layer=1)
        l2_neurons = self.__get_layer(layer=2)
       
        for l1_gid in sorted(l1_neurons.keys()):
            l1_neuron = l1_neurons[l1_gid]
            l1_module = l1_neuron['module']
            
            for l2_gid in sorted(l2_neurons.keys()):
                l2_neuron = l2_neurons[l2_gid]
                l2_module = l2_neuron['module']
                
                p_mod2mod = 1. - norm.cdf(abs(l1_module - l2_module), loc=0., scale=scale)
                is_connected = random.choice([0, 1], p=[1.-p_mod2mod, p_mod2mod], size=(1,))
                if is_connected:
                    nc = generate_nc_connection(l2_neuron['object'], l1_neuron['object'], self.__syn_weight, self.__syn_delay)
                    self.cell_attributes[l2_gid]['connections'].append((l1_gid, nc))


    def __get_layer(self, layer=1):
        if self.cell_attributes == {}:
            raise Exception('Cell attributes has not been populated..')

        neurons = {}
        for gid in sorted(self.cell_attributes.keys()):
            this_neuron = self.cell_attributes[gid]
            if this_neuron['layer'] == layer:
                neurons[gid] = this_neuron
        return neurons

    def __instantiate_kuromoto_network(self, frequencies, coupling, steps, time, init_phases=None, seed=0):
        frequencies = frequencies[0:self.__nlayer1]
        coupling = coupling[0:self.__nlayer1, 0:self.__nlayer1]
        knet = kuramoto_network(self.__nlayer1, frequencies, coupling, init_phases=init_phases, seed=seed)
        dynamic_phase, dynamic_mean_phase, dynamic_synchrony, dynamic_time = knet.simulate(steps, time)
        self.kuromoto_network['network'] = knet
        self.kuromoto_network['dynamic phase'] = dynamic_phase
        self.kuromoto_network['dynamic mean phase'] = dynamic_mean_phase
        self.kuromoto_network['dynamic synchrony'] = dynamic_synchrony
        self.kuromoto_network['dynamic time'] = dynamic_time

        angles     = phase_to_angle(dynamic_phase.T, dynamic_time[1] - dynamic_time[0])
        amplitudes = phase_to_amplitude(dynamic_phase.T, frequencies, dynamic_time)
        self.kuromoto_network['angles'] = angles
        self.kuromoto_network['amplitudes'] = amplitudes

    def __simulate_kuromoto_layer1(self, T):
        if self.kuromoto_network is None:
            raise Exception('Need to instantiate and run kuromoto network first')
        times = self.kuromoto_network['dynamic time']
        dur = times[1] - times[0]
        valid_time_idxs = np.where(times <= T)[0]
        times = times[valid_time_idxs]
        amplitude = self.kuromoto_network['amplitudes'][valid_time_idxs,:] # T x N

        l1_neurons = self.__get_layer(layer=1)
        for l1_gid in sorted(l1_neurons.keys()):
            l1_neuron = l1_neurons[gid]
            l1_module = l1_neuron['module']
            this_amplitude = amplitude[:,l1_module]

            stims = []
            for (t, time) in times:
                stim = h.IClamp(l1_neuron.cell_obj.sec(0.5))
                stim.delay = time
                stim.dur = dur
                stim.amp = 0.30*this_amplitude[t]
                stims.append(stim)
                
                
            v_vector = h.Vector()
            v_vector.record(l1_neuron['object'].cell_obj.sec(0.5)._ref_v)
            self.cell_attributes[l1_gid]['voltage'] = v_vector
            self.cell_attributes[l1_gid]['iclamp']  = stims
        t_vector = h.Vector()
        t_vector.record(h._ref_t)

        h.dt = self.__dt
        h.tstop = time[-1]
        h.run()

        self.kuramoto_sim_time = np.asarray(list(t_vector), dtype='float32')

    def __extract_spike_times(self):
        gid_spikes = {}
        l1_neurons = self.__get_layer(layer=1)
        for l1_gid in sorted(l1_neurons.keys()):
            l1_neuron = l1_neurons[l1_gid]
            voltage = l1_neuron['voltage']
            time    = self.kuramoto_sim_time
            spike_times, spike_lst = extract_spikes(voltage, time)
            gid_spikes[l1_gid]['times'] = spike_times
            gid_spikes[l1_gid]['list']  = spike_lst
        self.l1_spikes = gid_spikes
            

class MicrocircuitModule(object):

    def __init__(self, region_id, nid_start, npairs, excitatory_cell_type='RS', inhibitory_cell_type='FS'):
        self.__region_id = region_id
        self.__nid_start = nid_start
        self.__npairs = npairs
        self.__excitatory_cell_type = excitatory_cell_type
        self.__inhibitory_cell_type = inhibitory_cell_type

        self.excitatory_neurons = None
        self.inhibitory_neurons = None
        self.nc_lst = None

        self.excitatory_neurons = generate_neurons(self.__npairs, self.__nid_start, self.__excitatory_cell_type)
        self.inhibitory_neurons = generate_neurons(self.__npairs, self.__nid_start+self.__npairs, self.__inhibitory_cell_type)

    def generate_feedback_connection(self, syn_w, syn_delay):
        self.nc_lst = generate_feedback_connection(self.excitatory_neurons, self.inhibitory_neurons, syn_w, syn_delay)
        
class Cell(object):

    def __init__(self, cell_obj):
        self.cell_obj = cell_obj
        self.synapse  = None

    def create_synapse(self):
        syn = h.Exp2Syn(self.cell_obj.sec(0.5))
        #syn.tau = 2.
        self.synapse = syn
        
def generate_neurons(N, id_start, cell_type):
    return [Cell(IzhiCell(cell_type, i + id_start)) for i in range(N)]

def generate_nc_connection(tgt, src, syn_w, syn_delay):
    if tgt.synapse is None:
        src.create_synapse()
    nc = connect_cells(tgt, src, syn_w, syn_delay)
    return nc

def connect_cells(post, pre, syn_w, syn_delay):
    tgt_syn = post.synapse
    nc = h.NetCon(pre.cell_obj.sec(0.5)._ref_v, tgt_syn, sec=pre.cell_obj.sec)
    nc.weight[0] = syn_w
    nc.delay     = syn_delay
    return nc
        
        
