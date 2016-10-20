import numpy as np
import timeit
import threading
import sys
import random as randomPack
import os.path
from os import remove as os_remove
from multiprocessing import Pipe, Pool,Process,Queue,TimeoutError
from phototaxis import *

# set_noiseTerm('')
mutation_rate = 0.00001
popsize = 300
elite_size = 20
n_generations = 200 
p = 0.02

# set this to 1 for a machine with only one or two cores:
numberThreads = 14

# should not be changed:
length_vector = 33

if (popsize-elite_size)%2 != 0:
	raise ValueError('popsize-elite_size not even')

# each thread uses a different instance of the network. different object
evaluators = [Phototaxis_evaluator(i) for i in range(numberThreads)] 

def multiProcessPopulationFitness(pop, NumberOfAssesment=4):
	'''Asses fitness for all individuals in the network, NumberOfAssesment times'''
	if(popsize != len(pop)):
		raise ValueError('lenghth of population doesnt fit popsize')

	
	lowerBound = 0 
	processes = []
	receivers = []
	#split the population into numberThreads parts and give each part to a different job
	for i,evaluator in enumerate(evaluators):
		# the first blocks will be one smaller (implicitly rounding down with //) until we can equally fit the rest
		# stepsize = N individuals left / N jobs left
		upperBound = lowerBound+((popsize-lowerBound)//(numberThreads-i))
		print('[{}:{}]'.format(lowerBound,upperBound))
		
		# to get the data back from the process we use a unidirectional pipe
		receiver, sender = Pipe(duplex=False)
		newprocess = Process(target=evaluator.assesPopulationFitness, args=(pop[lowerBound:upperBound],sender,NumberOfAssesment))
		# newprocess = Process(target=evaluators[i].assesPopulationFitness, args=(subPopulation,sender))
		newprocess.start()
		receivers.append(receiver)
		processes.append(newprocess)
		lowerBound = upperBound

	evaluatedPop = []
	# collect the data
	for rec in receivers:
		evaluatedPop.extend(rec.recv())
	# terminate the processes
	for job in processes:
		job.join()

	print('finished threaded Population assesment')
	return evaluatedPop


def sortPopulation(populationToSort):
	# sort the list of dictonarys by the fitness
	populationSorted = sorted(populationToSort, key=lambda individual:individual['fitness'], reverse=True)
	return populationSorted

def evalPopNtimes(name, N):
	'''asses population N times to give a better picture
	Args:
		name: name of the .npy file in which the poulation is stored
	'''
		
	pop = np.load(name)
	pop = multiProcessPopulationFitness(pop, N)
	pop = sortPopulation(pop)
	np.save('properEvalPop', pop)
	

def selectParent(pop):
	'''selects a parent from the population randomly, biased on the fitness'''
	canidateA = randomPack.choice(pop)
	canidateB = randomPack.choice(pop)
	if( (canidateA['fitness'] > canidateB['fitness'])):
		return canidateA['content']
	else:
		return canidateB['content']

def crossover(ParentA, ParentB):
	# Intermediate Recombination (similar to Line Recombination)
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
	# just add indebendent gaussian noise elementwise
	# mutation_rate is a global parameter specified above
	return np.clip(child + np.random.normal(loc=0.0, scale=mutation_rate, size=length_vector),0,1)



def do_evolution(population, startGen):
	global elite_size, popsize, n_generations
	if(len(population) != popsize):
		raise ValueError('popsize not equal length of population')

	for t in range(startGen, n_generations):	
		next_population = population[0:elite_size] # keep the best individuals straight away
		for i in range(int((popsize-elite_size)/2)):
			# select parents 
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

