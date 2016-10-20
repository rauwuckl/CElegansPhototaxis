# from __future__ import division

# import networkx as nx

from brian2 import *
from my_helper import *
from phototaxis_network import *
import matplotlib.pyplot as plt

#set_noiseTerm('')


# intensity eig (10**(-1.73))*20 = 0.37241742733257355
inputLight = {'lam': [750, 350, 750], 'intensity': [0, 0.372417, 0]}
inputTiming= [0.5,2,8]*second

# runWithInput(network, 'sensory_neurons', inputLight, [1, 2, 1]*second)
startResponse = 2.23
endResponse = 9.23
duration = endResponse - startResponse


class Phototaxis_evaluator:
	def __init__(self, ID=None):
		self.ID = ID
		self.results = None
		self.network = buildNetwork()
		print('build finished')


		# statemonSens = StateMonitor(network['sensory_neurons'], ['v','u'], record=True)
		# synmon = StateMonitor(network['inter_connections'], ['g', 'Iinput_post'], record=True)
		# statemon=StateMonitor(network['other_neurons'], ['v', 'u'], record=True)
		other_spikeMon = SpikeMonitor(self.network['other_neurons'], name='spikemon_other')
		self.network.add(other_spikeMon)

		self.network.store()
		print('storing complete')
		###Make a graph
		# DG = nx.DiGraph()

		# DG.add_weighted_edges_from([(neuron_names['other'][edge[0]], neuron_names['other'][edge[1]], int(edge[2])) for edge in table])
		# nx.write_graphml(DG, 'other.graphml')

	### RUN
	def assesPopulationFitness(self, pop, sender=None, numberAssesments=1):
		# update all the fitness
		for individual in pop:
			fit = self.assesNetworkFitnessNtimes(individual['content'], numberAssesments)
			# Maybe put  a lock here
			individual['fitness'] = fit

		if(sender is None):
			# not multiprocess mode
			return pop
		
		sender.send(pop)


	def setNetworkFitness(self, individual):
		individual['fitness']=self.assesNetworkFitness(individual['content'])

	def assesNetworkFitness(self, parameter):
		self.network.restore()
		print('resoring complete')
		set_parameter(self.network, parameter)
		runWithInput(self.network, 'sensory_neurons', inputLight, inputTiming)
		print('running complete')
		spiketrains = self.network['spikemon_other'].spike_trains()
		penalty = assesQuality4(spiketrains, startResponse, endResponse) #startResponse, endResponse)
		print('quality asessment complete')
		return -penalty


#	def assesNetworkVisual(self, individual):
#		print('asses Network visual:')
#		parameter=individual['content']
#		dispNet = buildNetwork()
#		dispNet.add(SpikeMonitor(dispNet['other_neurons'], name='spikemon_other'))
#		set_parameter(dispNet, parameter)
#
#		sensmon	= StateMonitor(dispNet['sensory_neurons'], 'v', record=True)
#		othermon=StateMonitor(dispNet['other_neurons'],'v', record=True)
#
#		dispNet.add(sensmon, othermon)
#
#		runWithInput(dispNet, 'sensory_neurons', inputLight, inputTiming)
#
#		spiketrains = dispNet['spikemon_other'].spike_trains()
#		penalty = assesQuality4(spiketrains, startResponse, endResponse) #startResponse, endResponse)
#		ion()
#		figure()
#		for i,trace in enumerate(sensmon.v):
#			subplot(2,2,1+i)
#			plot(sensmon.t/second, trace, label='v')
#			axvline(startResponse, ls='-', c='g', lw=1)
#			axvline(endResponse, ls='-', c='r', lw=1)
#			title('Sensory neuron: {}'.format(i))
#
#		figure()
#		for i,trace in enumerate(othermon.v):
#			subplot(2,3,1+i)
#			plot(othermon.t/second, trace, label='v')
#			axvline(startResponse, ls='-', c='g', lw=1)
#			axvline(endResponse, ls='-', c='r', lw=1)
#			for spike in spiketrains[i]:
#				axvline(spike/second, ls=':', c='y', lw=0.5)
#			title('Other neuron: {}'.format(i))
#		print(-penalty)
	def assesBestVisual(self, name, N, j=0):
		pop= np.load(name)
		best = pop[j]
		for i in range(N):
			ot,st,pen= self.simulateTraces(best)
			plotResult(ot, st)


	def simulateTraces(self, individual):
		print('plotResult: asses Network visual:')
		parameter=individual['content']
		dispNet = buildNetwork()
		dispNet.add(SpikeMonitor(dispNet['other_neurons'], name='spikemon_other'))
		set_parameter(dispNet, parameter)

		othermon=StateMonitor(dispNet['other_neurons'],'v', record=[4,5])

		dispNet.add(othermon)

		runWithInput(dispNet, 'sensory_neurons', inputLight, inputTiming)

		spiketrains = dispNet['spikemon_other'].spike_trains()
		penalty = assesQuality4(spiketrains, startResponse, endResponse) #startResponse, endResponse)
		return (othermon, spiketrains, penalty)



	def assesNetworkFitnessNtimes(self, content, N):
		'''asses fitness parameter vector N times and take the minimum'''
		tempFitnessList = []
		for i in range(N):
			tempFitnessList.append(self.assesNetworkFitness(content))
		print('network had fitness: {}'.format(tempFitnessList))
		return np.mean(tempFitnessList)

def plotResult(othermon, spiketrains):
	ion()
	fig = plt.figure()
	ax=fig.add_subplot(1,2,1)
	ax.set_title('DA Motor Neuron')
	ax.plot(othermon.t/second, othermon.v[0]/mV)
	ax.axvline(startResponse, ls='-', c='g', lw=1)
	ax.axvline(endResponse, ls='-', c='r', lw=1)
	for spike in spiketrains[4]:
		ax.axvline(spike/second, ls=':', c='y', lw=0.9)
	firstSpike=spiketrains[4][0]
	lastSpike=spiketrains[4][-1]
	ax.plot([firstSpike, lastSpike],[50, 50], ls='-', c='black', lw=3)
	ax.text(2.3, 51, 'duration: {}'.format(lastSpike-firstSpike))
	ax.plot([0.5, 2.5],[-75, -75], ls='-', c='g', lw=3)
	ax.text(1.3, -74, 'stimulus')
	ax.set_xlabel('time, seconds')
	ax.set_ylabel('membrane voltage, mV')
	
	ax=fig.add_subplot(1,2,2)
	ax.set_title('VA Motor Neuron')
	trace, = ax.plot(othermon.t/second, othermon.v[1]/mV)
	startR = ax.axvline(startResponse, ls='-', c='g', lw=1)
	endR = ax.axvline(endResponse, ls='-', c='r', lw=1)
	for spike in spiketrains[5]:
		spikesLine = ax.axvline(spike/second, ls=':', c='y', lw=0.9)
	firstSpike=spiketrains[5][0]
	lastSpike=spiketrains[5][-1]
	ax.plot([firstSpike, lastSpike],[50, 50], ls='-', c='black', lw=3)
	ax.text(2.3, 51, 'duration: {}'.format(lastSpike-firstSpike))
	ax.plot([0.5, 2.5],[-75, -75], ls='-', c='g', lw=3)
	ax.text(1.3, -74, 'stimulus')
	ax.set_xlabel('time, seconds')
	ax.set_ylabel('membrane voltage, mV')

	fig.subplots_adjust(bottom=0.3)
	fig.legend([trace, startR, endR, spikesLine], ['membrane voltage', 'start of observed behaviour', 'end of observed behaviour', 'spike'], loc='lower center')
	plt.show()

if __name__ == '__main__':
	plt.close('all')
	eva = Phototaxis_evaluator()

### PLOTTING

# plotTracesFromStatemon(synmon, 'g', 'Iinput_post')
# plotTracesFromStatemon(statemonSens,'v')
# plotTracesFromStatemon(statemon, 'v')


# def test():
# 	start_scope()
# 	neur=NeuronGroup(3, model_normal, threshold='v>30*mV', reset=reset)
# 	s=Synapses(neur, neur, model=model_chem_syn, on_pre='g = 1')
# 	s.connect(i=0,j=2)
# 	s.w=1/ms
# 	statemon=StateMonitor(neur, ['v','u'], record=True)

# 	N = Network(neur,s)
# 	N.run(500*ms)
# 	plotTracesFromStatemon(statemon, 'v')
