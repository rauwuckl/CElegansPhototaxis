import numpy as np
popsize = 13 
numberThreads = 15
evaluators = range(numberThreads)
def multiProcessPopulationFitness(pop=range(popsize)):
	individualsPerProcess = popsize//numberThreads
	print('individualsPerProcess: {}, popsize: {}, numberThreads: {}'.format(individualsPerProcess, popsize, numberThreads))
	lowerBound = 0 

	processes = []
	receivers = []
	for i,evaluator in enumerate(evaluators):
	#for i,supPopulation in enumerate(np.array_split(pop)):
		# lowerBound = i*individualsPerProcess
		# upperBound = np.clip((i+1)*individualsPerProcess, 0, len(pop))
		upperBound = lowerBound+((popsize-lowerBound)//(numberThreads-i))
		print('[{}:{}]'.format(lowerBound,upperBound))
		# newprocess = Process(target=evaluators[i].assesPopulationFitness, args=(subPopulation,sender))
		lowerBound = upperBound
