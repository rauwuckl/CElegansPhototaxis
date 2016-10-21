# import pylab
# import networkx as nx
from brian2 import *
from scipy.stats import norm as normDistribution
def plotTracesFromStatemon(statemon, *args):
	N = len(getattr(statemon, args[0]))
	N_lines = ceil(N/4)

	plt.ion()
	plt.figure()

	plots=[]

	for i in range(N):
		plt.subplot(N_lines, 4, i+1)
		for attribute in args:
			plt.plot(statemon.t/ms, getattr(statemon, attribute)[i], label=attribute)
			plt.pause(0.0001)
		plt.legend()

	plt.ioff()


def smoothedSpikeRate(x, spikes, scal=100*ms): # gausian kernel density function
	if(len(spikes)==0):
		return zeros(len(x))
	return np.sum([normDistribution.pdf(x, loc=m, scale=scal) for m in spikes], axis=0)

def combineSpikeTrains(*args):
	"""each argument should be a list of spiketimes (as given from spiketrains). returns tupel: 
	(timeOfFirst, timeOfLast)... spike"""
	# should check if all have the same unit, but the conntatenate method seems to 
	# convert everything to seconds
	return sort(concatenate(args))

def assesQuality(spiketrains, start, end):

	penalty   = 0	
	allSpikes = combineSpikeTrains(spiketrains[4],spiketrains[5])#only the two motorneurons
	if (len(allSpikes) == 0):
		return float('inf')
	# print('spike trains: {}'.format(allSpikes))
	diffStart =  20*((allSpikes[0] -start))**2
	diffEnd	  =  20*((allSpikes[-1]-  end))**2
	x = np.linspace(allSpikes[0]+150*ms, allSpikes[-1]-150*ms, 100)#so the upstroke of the firing rate in the begining doesn't increase the variance
	spikeRate = smoothedSpikeRate(x, allSpikes) 
	variance  = np.std(spikeRate)
	penalty   = diffStart + diffEnd + variance
	print('diffStart: {:2.3f}, diffEnd: {:2.3f}, variance: {:2.3f}, total: {:2.3f}'.format(diffStart, diffEnd, variance, penalty))
	return penalty


def assesQuality2(spiketrains, targetDuration):
	penalty   = 0	
	allSpikes = combineSpikeTrains(spiketrains[4],spiketrains[5])#only the two motorneurons
	if (len(allSpikes) == 0):
		return float('inf')
	# print('spike trains: {}'.format(allSpikes))
	duration =  allSpikes[-1]-allSpikes[0]
	difference = 20* ((duration-targetDuration))**2
	x = np.linspace(allSpikes[0]+150*ms, allSpikes[-1]-150*ms, 100)#so the upstroke of the firing rate in the begining doesn't increase the variance
	spikeRate = smoothedSpikeRate(x, allSpikes) 
	variance  = np.std(spikeRate)
	penalty   = difference + variance
	print('difference: {:2.3f}, variance: {:2.3f}, total: {:2.3f}'.format(difference, variance, penalty))
	return penalty

def assesQuality3(spiketrains, start, end):
	'''duration, smotheness and must not start spiking after response starts must not stop spiking after response ends'''

	penalty   = 0	
	allSpikes = combineSpikeTrains(spiketrains[4],spiketrains[5])#only the two motorneurons
	if (len(allSpikes) == 0):
		return float('inf')
	# timing
	# print('spike trains: {}'.format(allSpikes))
	diffStart =  20*(np.clip((allSpikes[0]- start),0,100))**2#only penalty for first spike AFTER the worm should start moving
	diffEnd	  =  20*(np.clip((allSpikes[-1]-  end),0,100))**2#only penalty for last spike AFTER the worm should stop moving

	# duration
	targetDuration = end-start
	duration =  allSpikes[-1]-allSpikes[0]
	difference = 20* ((duration-targetDuration))**2

	# smotheness
	#x = np.linspace(allSpikes[0]+150*ms, allSpikes[-1]-150*ms, 100)#so the upstroke of the firing rate in the begining doesn't increase the variance
	#spikeRate = smoothedSpikeRate(x, allSpikes) 
	variance  = 0  # 0*np.std(spikeRate)
	penalty   = diffStart + diffEnd + difference + variance
	print('diffStart: {:2.3f}, diffEnd: {:2.3f}, difference: {:2.3f}, variance: {:2.3f}, total: {:2.3f}'.format(diffStart, diffEnd, difference, variance, penalty))
	return penalty
# mse between start/end of spiking and behaiviour duration
# minimized variance of smoothed within spiking set

def assesQuality4(spiketrains, start, end):
	return assesSpikeTrain(spiketrains[4], start, end) + assesSpikeTrain(spiketrains[5], start, end)


def assesSpikeTrain(spiketrain, start, end):
	if(len(spiketrain)<=1):
		return 1000 #arbitrary a little higher then all normal penalties
	spiketrain = np.array(spiketrain)# to be shure
	penalty   = 0	
	diffStart =  20*(np.clip((spiketrain[0]- start),0,100))**2#only penalty for first spike AFTER the worm should start moving
	diffEnd	  =  20*(np.clip((spiketrain[-1]-  end),0,100))**2#only penalty for last spike AFTER the worm should stop moving

	# duration
	targetDuration = end-start
	duration =  spiketrain[-1]-spiketrain[0]
	difference = 20* ((duration-targetDuration))**2
	
	# variance
	# get all the intervalls between consecutive spikes
	spikeIntervalls = spiketrain[1:] - spiketrain[:-1]
	variance = 25*np.std(spikeIntervalls)

	penalty   = diffStart + diffEnd + difference + variance
	print('diffStart: {:2.3f}, diffEnd: {:2.3f}, difference: {:2.3f}, variance: {:2.3f}, total: {:2.3f}'.format(diffStart, diffEnd, difference, variance, penalty))
	return penalty

