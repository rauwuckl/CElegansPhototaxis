import numpy as np
from matplotlib import pyplot as plt
plt.close('all')
def read_history(filename):
	'''reads all the populations with the specified filename and puts them into two tensors
	contents has dimension (n_generations, n_individuals, n_parameters)
	fitness (n_generation, n_individuals)'''
	i = 0
	total_contents = []
	total_fitness = []
	while True:
		contents = []
		fitness = []
		try:
			population = np.load('{}{}.npy'.format(filename, i))
		except IOError:
			print('no file found with {}{}.npy'.format(filename, i))
			break
		
		for individual in population:
			contents.append(individual['content'])
			fitness.append(individual['fitness']) 
		total_contents.append(contents)
		total_fitness.append(fitness)
		i += 1
	return np.array(total_contents), np.array(total_fitness)

def get_fitness_development(fitness):
	max_fitness = np.amax(fitness, 1)
	average_fitness = np.array([np.mean(pop[~np.isinf(pop)]) for pop in fitness])
	return max_fitness, average_fitness

def get_volume_development(content):
	'''will calculate the volume of a minimal bounding box for all the individuals'''
	# volume of covered space within generation
	mins = np.amin(content, 1)
	maxs = np.amax(content, 1)
	lengths = maxs-mins
	volume = np.prod(lengths, 1)

	# volume of covered space by all individualls of all generations up to now put together
	up_to_now_min = mins[0]
	up_to_now_max = maxs[0]
	up_to_now_total_covered_volume = []
	for pop in content[:]:
		pop = np.append(pop, [up_to_now_min], 0)
		pop = np.append(pop, [up_to_now_max], 0)
		up_to_now_min = np.amin(pop, 0)
		up_to_now_max = np.amax(pop, 0)
		up_to_now_total_covered_volume.append(np.prod(up_to_now_max - up_to_now_min))
		
	return volume, up_to_now_total_covered_volume

def get_average(content):
	'''calculates how the centroids of each population move around. To give an idea of the explored parameter space'''
	#means of all individuals for each generation:
	means = np.mean(content, 1)
	first_mean = means[0]
	#change of centroid to first centroid of all generations
	distance_to_first = [np.linalg.norm(first_mean- current) for current in means]
	#change of centroid from last generation to current one
	distance_to_privious = [0]
	for i,current in enumerate(means[1:]):
		distance_to_privious.append(np.linalg.norm(means[i] - current))
	return means, distance_to_first, distance_to_privious

def plot_stuff(content, fitness):
	n_generations = np.shape(content)[0]
	max_fitness, average_fitness = get_fitness_development(fitness)
	volume, total_covered_volume = get_volume_development(content)
	means, dist_to_first, dist_to_last = get_average(content)

	plt.figure()
	plt.subplot(2,2,1)
	plt.plot(max_fitness, label='max fitness')
	plt.plot(average_fitness, label='average fitness')
	plt.legend(loc=4)
	plt.subplot(2,2,2)
	plt.plot(volume, label='volume of bounding box')
	plt.plot(total_covered_volume, label='total covered volume up to current generation')
	plt.legend(loc=5)
	plt.subplot(2,2,3)
	plt.plot(dist_to_first, label='distance to initial centroid')
	plt.plot(dist_to_last, label='distance to last centroid')
	plt.legend(loc=5)
	plt.show()
	print('plotting complete')
	

if __name__ == '__main__':
	# ipython3 --matplotlib
	# %run examine_evolution.py (while beeing in the folder with all the .npy files)
	plt.close('all')
	b,c=read_history('population')
	plot_stuff(b,c)
	
