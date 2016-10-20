from phototaxis_network import *
from my_helper import *

start_scope()
neur=NeuronGroup(3, model_normal, threshold='v>30*mV', reset=reset, name='group')
s=Synapses(neur, neur, model=model_chem_syn, on_pre='g = 1')
s.connect(i=[0,1],j=[2,2])
s.w=[2,6]/ms
s.vReversal=[0,-70]*mV
# s.vReversal=
g=Synapses(neur, neur, model=model_gap_junction)
g.connect(i=0,j=1)
g.w=2/ms
neur.v=-70*mV
neur.u=b*neur.v

statemon=StateMonitor(neur, ['v','u'], record=True)
spikemon=SpikeMonitor(neur, record=[1,0])
statemonSyn=StateMonitor(s, ['g'], record=True)


N = Network(neur,s,statemon,spikemon, statemonSyn, g)

inputs= {'Iinput':[[0,0,0],[10,0,0],[0,0,0]]*mV/ms}
durations= [0.5,1,0.5]*second

runWithInput(N, 'group', inputs, durations)


spiketrains=spikemon.spike_trains()

print(assesQuality(spiketrains, 1*second, 4*second))

hist,edges = np.histogram(spiketrains[2], bins=len(statemon.t)//300)

centers = edges[1:]+(edges[1]-edges[0])/2.



ion()
figure()
plot(statemon.t/ms, statemon.v[0]*100, label='0')
plot(statemon.t/ms, statemon.v[1]*100, label='1')
plot(statemon.t/ms, statemon.v[2]*100, label='2')
legend()
# plot(statemon.t/ms, smoothedSpikeRate(statemon.t, spiketrains[0], scal=100*ms))
# plot(statemon.t/ms, smoothedSpikeRate(statemon.t, spiketrains[2], scal=100*ms))
# plotTracesFromStatemon(statemonSyn, 'g')
