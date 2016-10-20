# chemical synapses: below 0.5 inhibitory. (0.5-value)*2 == strength
# above 0.5 exitatory. (valute-0.5)*2 == strength
# line recombination
import numpy as np
import timeit
import threading
import sys
import random as randomPack
import os.path
from os import remove as os_remove
from multiprocessing import Pipe, Pool,Process,Queue,TimeoutError
from phototaxis import *

# all the kids have terrible fitness
# even some with reasonable fitness seem to have no spike in fact
# everything is close to 0.08 already in the begining
# population not diverse enough. mutate more maybe
# set_noiseTerm('')
mutation_rate = 0.00001
popsize = 300
elite_size = 20
n_generations = 200 
length_vector = 33
p = 0.02
numberThreads = 14

if (popsize-elite_size)%2 != 0:
	raise ValueError('popsize-elite_size not even')

# initialisation
best = {'fitness': -float('inf'), 'content': None} # (fitness, individual)

# np.save('popAfterCreation', population)

	# def assesFitness(param):
	# x    = (0.5-np.random.rand(55))*20
	# yt 	 = 0.3*(x**4)+0.4*(x**3)+0.8*(x**2)+0.2*(x)+0.5
	# yHat = param[4]*(x**4)+param[3]*(x**3)+param[2]*(x**2)+param[1]*(x)+param[0]
	# mse  = np.sum((yHat-yt)**2)/length_vector
	# return -mse

evaluators = [Phototaxis_evaluator(i) for i in range(numberThreads)] 

def multiProcessPopulationFitness(pop, minOfNumber=4):
	if(popsize != len(pop)):
		raise ValueError('lenghth of population doesnt fit popsize')
	lowerBound = 0 
	processes = []
	receivers = []
	for i,evaluator in enumerate(evaluators):
	#for i,supPopulation in enumerate(np.array_split(pop)):
		# the first blocks will be one smaller (implicitly rounding down with //) until we can equally fit the rest
		# stepsize = N individuals left / N jobs left
		upperBound = lowerBound+((popsize-lowerBound)//(numberThreads-i))
		print('[{}:{}]'.format(lowerBound,upperBound))
		receiver, sender = Pipe(duplex=False)
		newprocess = Process(target=evaluator.assesPopulationFitness, args=(pop[lowerBound:upperBound],sender,minOfNumber))
		# newprocess = Process(target=evaluators[i].assesPopulationFitness, args=(subPopulation,sender))
		newprocess.start()
		receivers.append(receiver)
		processes.append(newprocess)
		lowerBound = upperBound

	evaluatedPop = []
	for rec in receivers:
		evaluatedPop.extend(rec.recv())
	for job in processes:
		job.join()

	print('finished threaded Population assesment')
	return evaluatedPop


def sortPopulation(populationToSort):
	global best
	# sort the list of dictonarys by the fitness
	populationSorted = sorted(populationToSort, key=lambda individual:individual['fitness'], reverse=True)
	# if the first one is fitter then the current best put it there 
	# probably not necessary since the best view survive all the time anyways
	if (populationSorted[0]['fitness'] > best['fitness']):
			best = populationSorted[0]
	return populationSorted

def assesBestVisual(name):
	pop = np.load(name)
	evaluators[0].assesNetworkVisual(pop[0])

def evalPopNtimes(name, N):
	pop = np.load(name)
	pop = multiProcessPopulationFitness(pop, N)
	pop = sortPopulation(pop)
	np.save('properEvalPop', pop)
	evaluators[0].assesNetworkVisual(pop[0])
	

def selectParent(pop):
	canidateA = randomPack.choice(pop)
	canidateB = randomPack.choice(pop)
	if( (canidateA['fitness'] > canidateB['fitness'])):
		return canidateA['content']
	else:
		return canidateB['content']

def crossover(ParentA, ParentB):
	ChildA = np.zeros(length_vector)
	ChildB = np.zeros(length_vector)
	for i in range(length_vector):
		while True:
			alpha = randomPack.uniform(-p, 1+p)
			beta  = randomPack.uniform(-p, 1+p)
			t     = alpha*ParentA[i]+ (1-alpha)*ParentB[i]
			s     = beta *ParentB[i]+ (1-beta )*ParentA[i]

			if((0<t<1)and(0<s<1)):
				break # think about the ranges maybe
		ChildA[i] = t
		ChildB[i] = s
	return (ChildA, ChildB)

def mutate(child):
	return np.clip(child + np.random.normal(loc=0.0, scale=mutation_rate, size=length_vector),0,1)



def do_evolution(population, startGen):
	#assesPopulationFitness(population)
	#np.save('initialPop', population)
	#population = sortPopulation(population)
	#np.save('initialPopSorted', population)
	global elite_size, popsize, n_generations
	if(len(population) != popsize):
		raise ValueError('popsize not equal length of population')

	for t in range(startGen, n_generations):	
		next_population = population[0:elite_size] # keep the best
		for i in range(int((popsize-elite_size)/2)):
			ParentA = selectParent(population)
			ParentB = selectParent(population)
			ChildA, ChildB = crossover(ParentA, ParentB)
			ChildA = mutate(ChildA)
			ChildB = mutate(ChildB)

			next_population.append({'fitness': None, 'content':ChildA})
			next_population.append({'fitness': None, 'content':ChildB})
			print('{} of {} children born'.format(2*(i+1), (popsize-elite_size)))
		
		next_population = multiProcessPopulationFitness(next_population)
		population=sortPopulation(next_population)
		print('Fitest Individual: {} with fitness: {}'.format(best['content'], best['fitness']))
		np.save('population{}'.format(t), population)
		# print('finished saving')
		if os.path.exists('stop'):
			print('stop file was detected')
			os_remove('stop')
			return population
			
	return population

def checkAssesment(population):
	Eval = Phototaxis_evaluator(42)
	print('starting pop single')
	startSingle = timeit.default_timer()
	popSingle = Eval.assesPopulationFitness(population)
	popSingle = sortPopulation(popSingle)
	durationSingle = timeit.default_timer() - startSingle
	print('starting pop multi')
	startMulti = timeit.default_timer()
	popMulti = multiProcessPopulationFitness(population)
	popMulti = sortPopulation(popMulti)
	durationMulti = timeit.default_timer() - startMulti
	print('single eval took {}, multi core eval took {}, single/multi {} '.format(durationSingle, durationMulti, (durationSingle/durationMulti)))
	same = ([ind['fitness'] for ind in popSingle] == [ind['fitness'] for ind in popMulti])
	print('they are the same: {}'.format(same))
	return popSingle, popMulti

if __name__ == '__main__':
	print('bla: {}'.format(sys.argv))
	if len(sys.argv)==1:
		population = [{'fitness': -float('inf'), 'content':np.random.rand(length_vector)} for i in range(popsize)]
		population = multiProcessPopulationFitness(population)
		population = sortPopulation(population)
		do_evolution(population, 0)
	else:
		try:
			startGeneration = int(sys.argv[2])
			fileName = str(sys.argv[1])
		except ValueError:
			print('could not pars stuff: {}'.format(sys.argv))

		startPopulation = list(np.load('{}{}.npy'.format(fileName, startGeneration)))
		do_evolution(startPopulation, (startGeneration+1))

population = [{'fitness': -float('inf'), 'content':np.random.rand(length_vector)} for i in range(popsize)]

