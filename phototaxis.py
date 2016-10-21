# from __future__ import division

# import networkx as nx

from brian2 import *
from my_helper import *
from phototaxis_network import *
import matplotlib.pyplot as plt
import sys

# uncomment for deterministic behaiviour:
#set_noiseTerm('')


# intensity eig (10**(-1.73))*20 = 0.37241742733257355
inputLight = {'lam': [750, 350, 750], 'intensity': [0, 0.372417, 0]}
inputTiming= [0.5,2,8]*second

# runWithInput(network, 'sensory_neurons', inputLight, [1, 2, 1]*second)
startResponse = 2.23
endResponse = 9.23
duration = endResponse - startResponse


class Phototaxis_evaluator:
# class that can evaluate the fitness of a parameterset 
	def __init__(self, ID=None):
		self.ID = ID
		self.results = None
		self.network = buildNetwork()
		print('build finished')


		other_spikeMon = SpikeMonitor(self.network['other_neurons'], name='spikemon_other')
		self.network.add(other_spikeMon)

		self.network.store()
		print('storing complete')


	def assesPopulationFitness(self, pop, sender=None, numberAssesments=1):
		# update all the fitness
		# send results back threw the pipe
		for individual in pop:
			fit = self.assesNetworkFitnessNtimes(individual['content'], numberAssesments)
			# Maybe put  a lock here
			individual['fitness'] = fit

		if(sender is None):
			# not multiprocess mode
			return pop
		
		sender.send(pop)



	def assesNetworkFitness(self, parameter):
		'''asseses network fitness for the given parameter'''
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


	def assesBestVisual(self, name, N, saveFileAs=None, j=0):
		'''call with .npy file name and number of runs. then the fittest individual from that file will be simulated and PLOTTED N times, with j you can specify to plot the jth fittest individual'''
		pop= np.load(name)
		best = pop[j]
		for i in range(N):
			ot,st,pen= self.simulateTraces(best)
			
			if(not(saveFileAs is None)):
				plotResult(ot, st, '{}{}.png'.format(saveFileAs,i))
			else:
				plotResult(ot,st)


	def simulateTraces(self, individual):
		'''simulate and save the traces for the motorneurons. so that they can be plotted'''
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


def plotResult(othermon, spiketrains, saveFileAs=None):
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
	if saveFileAs is None:
		plt.show()
		print('only show')
	else:
		fig.savefig(saveFileAs)

if __name__ == '__main__':
	plt.close('all')
	eva = Phototaxis_evaluator()
	pictureName = None
	if len(sys.argv)!=1:
		try:
			N = int(sys.argv[2])
			fileName = str(sys.argv[1])
			if(len(sys.argv)==4):
				pictureName = str(sys.argv[3])
		except ValueError:
			print('could not pars stuff: {}'.format(sys.argv))

		eva.assesBestVisual(fileName, N, pictureName)
