import numpy as np
import cmath

def phase_to_amplitude(os_phases, frequencies, times):
    amplitudes = np.zeros_like(os_phases)
    for i in range(len(frequencies)):
        current_phases = os_phases[i,:]
        current_frequency = frequencies[i]
        for (t_idx,t) in enumerate(times):
            cv = cmath.exp(1j*(current_frequency*t + current_phases[t_idx]))
            amplitude = np.real(cv)
            amplitudes[i][t_idx] = amplitude
    return amplitudes.T

def phase_to_angle(phases, dt):
    from scipy.signal import medfilt
    
    angles = []
    for n in xrange(len(phases)):
        current_angles = np.diff(phases[n,:] % (2. * np.pi)) / dt
        med_filt_angles = medfilt(current_angles, kernel_size=11)
        angles.append(med_filt_angles)
    return np.asarray(angles, dtype='float32')


def signal_to_ca(trace, times, tau_rise=0.400, tau_fall=1.5, toff=1.0, tstop=10.0):
    from scipy.signal import fftconvolve

    times = np.asarray(times, dtype='float32')

    trise_idxs = np.where(times <= toff)[0]
    tdrop_idxs = np.where(times > toff)[0]
    tstop_idx  = np.where(times > tstop)[0][0]

    f1 = lambda t: 1. - np.exp(-t/tau_rise)
    f2 = lambda t: np.exp(-(t-toff)/tau_fall)

    yf1, yf2 = np.asarray(f1(times[trise_idxs])), np.asarray(f2(times[tdrop_idxs[0]:tstop_idx]))
    ykernel  = np.concatenate((yf1, yf2))
    conv     = fftconvolve(trace, ykernel, mode='same')
    return conv

def run_neuron_simulation(cell, inputs, times):
    from neuron import h, gui
    from copy import deepcopy
    
    dur   = times[1] - times[0]
    stims = []
    for (t,time) in enumerate(times):
        stim = h.IClamp(cell.sec(0.5))
        stim.delay = time
        stim.dur   = dur
        stim.amp   = 0.30*inputs[t]
        stims.append(stim)

    soma_v_vec = h.Vector()
    t_vec      = h.Vector()
    soma_v_vec.record(cell.sec(0.0)._ref_v)
    t_vec.record(h._ref_t)

    h.dt = 1.0
    h.tstop= times[-1]
    h.run()

    return deepcopy(list(soma_v_vec)), deepcopy(list(t_vec))


def get_layer(cells, layer):
    if cells == {}:
        raise Exception('cell attributes is not populated..')
    neurons = {}
    for gid in sorted(cells.keys()):
        this_neuron = cells[gid]
        if this_neuron['layer'] == layer:
            neurons[gid] = this_neuron
    return neurons

def get_spikes_from_cell_type(attr_dict, spike_dict, cell_type='e'):
    return_dict = {}
    for gid in sorted(spike_dict.keys()):
        if attr_dict[gid]['type'] == cell_type:
            return_dict[gid] = spike_dict[gid]
    return return_dict

def spike_detection(trace, times, threshold=-25.):
    from peakutils.peak import indexes

    idxs = indexes(np.asarray(trace), thres=0.5, min_dist=2)
    valid_idxs = []
    for idx in idxs:
        if trace[idx] >= threshold:
            valid_idxs.append(idx)
    idxs = valid_idxs
    spike_times = times[idxs]
    return np.asarray(spike_times)
    
    
def extract_spikes(voltage, neuron_time):
    return spike_detection(voltage, neuron_time)

def bin_activity(times, spike_times, nneurons, dt=0.025, chunk=25):
    chunk = int(chunk / dt)
    binned_spikes, binned_times = [], []
    for i in xrange(nneurons):
        current_spike_times = spike_times[i]
        binary_spike_lst = []
        for time in times:
            if time in current_spike_times:
                binary_spike_lst.append(1)
            else:
                binary_spike_lst.append(0)

        count = 0
        curr_spikes, curr_times = [], []
        for (j,t) in enumerate(times):
            if j % chunk == 0 and j > 0:
                curr_times.append(t)
                curr_spikes.append(count)
                count = 0
            else:
                if binary_spike_lst[j] == 1: count += 1
        binned_spikes.append(curr_spikes)
        binned_times = curr_times
    return binned_spikes, binned_times

def get_PSTH(X, N, dt, filt=False):
    histogram = np.sum(X,axis=0) / float(N * dt)
    if filt:
        from scipy.signal import medfilt
        histogram = medfilt(histogram)
    return histogram

def get_ISI(spike_train):
    from elephant.statistics import isi
    return isi(spike_train)

def get_CV(spike_train):
    from elephant.statistics import cv
    return cv(get_ISI(spike_train))

def get_instantaneous_FR(spiketrain, sampling_period, t_start=0., t_stop=5000.):
    from elephant.statistics import instantaneous_rate as ir_fnc
    from neo import SpikeTrain
    from quantities import ms

    spiketrain = SpikeTrain(spiketrain, t_stop, units='ms', t_start=t_start)
    return ir_fnc(spiketrain, sampling_period*ms, kernel='auto')

def generate_poisson_spike_train(rate, tstop, tstart=0., as_array=True):
    from elephant.spike_train_generation import homogeneous_poisson_process
    from quantities import ms, Hz

    return homogeneous_poisson_process(rate*Hz, t_start=tstart * ms, t_stop=tstop*ms, as_array=as_array)

