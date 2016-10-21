for t in range(n_generations):	
	next_population <- population[0:elite_size] # keep the best individuals directly
	for i in range(int((popsize-elite_size)/2)): # always 2 individuals have 2 children
		ParentA = selectParent(population) 
		ParentB = selectParent(population)
		ChildA, ChildB = crossover(ParentA, ParentB)
		ChildA = mutate(ChildA)
		ChildB = mutate(ChildB)
		next_population <- ChildA, ChildB
	population <- sort(next_population)




	