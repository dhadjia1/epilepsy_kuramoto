COMMENT

A "simple" implementation of the Izhikevich neuron.
Equations and parameter values are taken from
    Izhikevich EM (2007).
    "Dynamical systems in neuroscience"
    MIT Press

2019:
    Aaron D. Milstein neurosutras@gmail.com
    Modified previous implementation by:
        Salvador Dura-Bernal salvadordura@gmail.com
        Cliff Kerr cliffk@neurosim.downstate.edu
        Bill Lytton billl@neurosim.downstate.edu
        http://modeldb.yale.edu/39948

Implemented as a point process, rather than a density mechanism. The point process is meant
to be inserted into a single compartment NEURON section (nseg=1). The total surface area
and specific capacitance of the attached section will influence the membrane time constant,
though the apparent input resistance of the cell is completely controlled by the point
process. The original implementation assumed a compartment with a total membrane capacitance
of 100 pF. Each tuned celltype described in the Izhikevich, 2007 book included a factor "C"
that effectively scaled the membrane time constant. In order for compatibility with currents
generated in the section by other point processes (synaptic mechanisms and current clamp),
this factor "C" is directly applied as a multiplier to the specific capacitance of the
section, and is no longer included in the calculation of the current generated by the point
process.


Cell types available are based on Izhikevich, 2007 book:
    1. RS - Layer 5 regular spiking pyramidal cell (fig 8.12 from 2007 book)
    2. IB - Layer 5 intrinsically bursting cell (fig 8.19 from 2007 book)
    3. CH - Cat primary visual cortex chattering cell (fig 8.23 from 2007 book)
    4. LTS - Rat barrel cortex Low-threshold  spiking interneuron (fig 8.25 from 2007 book)
    5. FS - Rat visual cortex layer 5 fast-spiking interneuron (fig 8.27 from 2007 book)
    6. TC - Cat dorsal LGN thalamocortical (TC) cell (fig 8.31 from 2007 book)
    7. RTN - Rat reticular thalamic nucleus (RTN) cell  (fig 8.32 from 2007 book)

ENDCOMMENT

NEURON {
  POINT_PROCESS Izhi2019
  RANGE k, vr, vt, vpeak, a, b, c, d, celltype, derivtype, C
  NONSPECIFIC_CURRENT i
}

UNITS {
  (nA) = (nanoamp)
  (pA) = (picoamp)
  (mV) = (millivolt)
  (S) = (siemens)
  (uS) = (microS)
  (um) = (micron)
}

: Parameters from Izhikevich 2007, MIT Press for regular spiking pyramidal cell
PARAMETER {
  C = 1.            : A multiplier on the total section capacitance, used to set cm of the hoc section
  k = 0.7 (pA/mV2)
  vr = -60 (mV)     : Resting membrane potential
  vt = -40 (mV)     : Membrane threshold
  vpeak = 35 (mV)   : Peak voltage
  a = 0.03 (1)
  b = -2 (1)
  b_nl = 0.025 (1)  : For non-linear voltage sensitivity of FS cells (celltype == 5)
  c = -50 (mV)      : Reset voltage
  d = 100 (1)
  celltype = 1      : Each cell type has different dynamics (see list of cell types in initial comment).
}

ASSIGNED {
  v (mV)
  i (nA)
  derivtype
}

STATE {
  u (pA) : Slow current/recovery variable
}


INITIAL {
  u = 0.
  derivtype=2
  net_send(0,1) : Required for the WATCH statement to be active; v=vr initialization done there
}

BREAKPOINT {
  SOLVE states METHOD derivimplicit
  i = -(k*(v-vr)*(v-vt) - u) / 1000.
}

FUNCTION derivfunc () {
  : For FS neurons, include nonlinear u(v):
  : When v < d : u(v) = 0
  UNITSOFF
  if (celltype==5 && derivtype==2) {
    derivfunc = a*(0-u)
  : When v >= d: u(v) = b_nl * (v - d) ^ 3
  } else if (celltype==5 && derivtype==1) {
    derivfunc = a*((b_nl*(v-d)*(v-d)*(v-d))-u)
  } else {
    derivfunc = a*(b*(v-vr)-u)
  }
  UNITSON
}

DERIVATIVE states {
  LOCAL f
  f = derivfunc()
  UNITSOFF
  u' = f
  UNITSON
}
: Input received
NET_RECEIVE (w) {
  : Check if spike occurred
  if (flag == 1) { : Fake event from INITIAL block
    if (celltype == 4) { : LTS cell
      WATCH (v>(vpeak-0.1*u)) 2 : Check if threshold has been crossed, and if so, set flag=2     
    } else if (celltype == 6) { : TC cell
      WATCH (v>(vpeak+0.1*u)) 2 
    } else { : default for all other types
      WATCH (v>vpeak) 2 
    }
    : additional WATCHfulness
    if (celltype==6 || celltype==7) {
      WATCH (v> -65) 3 : change b param
      WATCH (v< -65) 4 : change b param
    }
    if (celltype==5) {
      WATCH (v> d) 3  : going up
      WATCH (v< d) 4  : coming down
    }
    v = vr  : initialization can be done here
  : FLAG 2 Event created by WATCH statement -- threshold crossed for spiking
  } else if (flag == 2) { 
    net_event(t) : Send spike event
    : For LTS neurons
    if (celltype == 4) {
      v = c+0.04*u : Reset voltage
      if ((u+d)<670) {u=u+d} : Reset recovery variable
      else {u=670} 
     }  
    : For FS neurons (only update v)
    else if (celltype == 5) {
      v = c : Reset voltage
     }  
    : For TC neurons (only update v)
    else if (celltype == 6) {
      v = c-0.1*u : Reset voltage
      u = u+d : Reset recovery variable
     }  else {: For RS, IB and CH neurons, and RTN
      v = c : Reset voltage
      u = u+d : Reset recovery variable
     }
  : FLAG 3 Event created by WATCH statement -- v exceeding set point for param reset
  } else if (flag == 3) { 
    : For TC neurons 
    if (celltype == 5)        { derivtype = 1
    } else if (celltype == 6) { b=0
    } else if (celltype == 7) { b=2 
    }
  : FLAG 4 Event created by WATCH statement -- v dropping below a setpoint for param reset
  } else if (flag == 4) { 
    if (celltype == 5)        { derivtype = 2
    } else if (celltype == 6) { b=15
    } else if (celltype == 7) { b=10
    }
  }
}