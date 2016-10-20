from brian2 import *
from my_helper import *

prefs.codegen.target = 'cython'#'cython'

# plt.close('all')

neuron_names = {'sen':['ash', 'asj', 'ask', 'awb'], 'other':['ava', 'avb', 'avd', 'pvc', 'da', 'va']}
a=0.02/ms
b=0.2/ms
c=-65*mV
d=6*mV/ms
tauChemSyn = 7*ms  # randomly choosen possibly/probaly not trained
noiseTerm = '+ 0.3*xi*mV*(ms**(-0.5))'

def set_noiseTerm(string):
	global noiseTerm
	noiseTerm = string

### CONNECTIVITY ###

#> Sensory neurons

connections_sens=[\
[(2,1)],#connections from neuron 0 to (targetNeuron, numberConnections)
[(2,8)],#1
[(1,1)],#2
[]]#3




sensory_gap_table=np.array([\
[0,2,1]])#each line is one gab junction

#> Other


other_chemical=[\
[(1,2), (5, 47), (4,66), (3,28),(2,2)],# originating at 0
[(0,27), (2,3), (4,1),(5,3)],#originating at neuron 1
[(0,70), (1, 1),(3,1),(4,20),(5,8)],
[(0,14), (1,27),(2,8),(4,2)],
[],
[(4,8),(3,5)],
]



other_gap_table=np.array([\
[0,3,10],
[0,4,27],
[0,5,40],
[4,5,2],
	])

#connections from sensory to other neurons
sens_other_connections_table=np.array([\
[0,0,7],
[0,1,10],
[0,2,6],
[3,1,3]
	])

#since the gap junctions are symetric these functions are used to repeat the paramterlist with the same values
get_gap_source = lambda table: append(table[:,0],table[:,1])
get_gap_target = lambda table: append(table[:,1],table[:,0])
get_gap_weight = lambda w, table: multiply(append(w,w), append(table[:,2],table[:,2]))


get_connectivity_table = lambda pairs: np.array([[k,synapse[0],synapse[1]] for k,neuron in enumerate(pairs) for synapse in neuron])
#this is now a table. in each line [sourceIndex, targetIndex, Number of connections]

other_chem_table=get_connectivity_table(other_chemical)
sensory_chem_table=get_connectivity_table(connections_sens)
### emd pf connections

def buildNetwork():

	### MODEL EQUATIONS
	model_sensory = '''
	dv/dt = (0.04/ms/mV)*v**2 + (5/ms)*v + 140*mV/ms - u + Igap	+ Isyn + (((750-lam)*(30+intensity))/lam)*mV/ms {} 		: volt
	du/dt = a*(b*v-u)														: volt/second
	lam 																: 1
	intensity 															: 1
	Igap	: volt/second
	Isyn 	: volt/second
	'''.format(noiseTerm)

	model_normal = '''
	dv/dt = (0.04/ms/mV)*v**2 + (5/ms)*v + 140*mV/ms - u + Igap	+ Isyn + Iinput  {}					: volt
	du/dt = a*(b*v-u)													: volt/second
	Igap	: volt/second
	Isyn 	: volt/second
	Iinput  : volt/second
	'''.format(noiseTerm)



	reset='''
	v= c
	u= u+d
	'''

	model_gap_junction = '''w : 1/second
	Igap_post = w*(v_pre-v_post): volt/second (summed)'''





	# model_chem_syn= '''w : volt/second
	# g=exp(-(t-lastupdate)/tauChemSyn) : 1
	# Isyn_post = w*g : volt/second (summed)
	# '''

	model_chem_syn= '''w : 1/second
	vReversal :volt
	diff=(vReversal-v_post) :volt
	dg/dt = -(1/tauChemSyn)*g  : 1 (clock-driven)
	Isyn_post = w*g*diff : volt/second (summed)
	'''
	# same as above g+=1 (all spikes in the past still relevant), g=1 (only last spike relevant)

	model_chem_syn_connection= '''w : 1/second
	vReversal : volt
	diff=(vReversal-v_post) : volt
	dg/dt = -(1/tauChemSyn)*g : 1 (clock-driven)
	Iinput_post = w*g*diff : volt/second (summed)'''




	### actually build the NETWORK ###



	#> Sensory neurons
	sensory_neurons = NeuronGroup(4, model_sensory, threshold='v>30*mV', reset=reset, name='sensory_neurons', method='euler')

	#chemical synapses
	sensory_synapses=Synapses(sensory_neurons, sensory_neurons, model=model_chem_syn,  on_pre='g = 1', name='sensory_synapses', method='linear') 



	sensory_synapses.connect(i=sensory_chem_table[:,0], j=sensory_chem_table[:,1]) 

	#weights proportional to the synamptic strength


	#gap junctions
	sensory_gap_junctions=Synapses(sensory_neurons,sensory_neurons, model=model_gap_junction, name='sensory_gap_junctions')

	sensory_gap_junctions.connect(i=get_gap_source(sensory_gap_table), j=get_gap_target(sensory_gap_table))

	#weights



	#>Motor and interneurons
	other_neurons = NeuronGroup(6, model_normal, threshold='v>30*mV', reset=reset, name='other_neurons', method='euler')
	#synapses:
	other_synapses=Synapses(other_neurons, other_neurons, model=model_chem_syn, on_pre='g = 1', name='other_synapses', method='linear')

	other_synapses.connect(i=other_chem_table[:,0], j=other_chem_table[:,1])

	#gap junctions
	other_gap_junctions=Synapses(other_neurons, other_neurons, model=model_gap_junction, name='other_gap_junctions')
	other_gap_junctions.connect(i=get_gap_source(other_gap_table),j=get_gap_target(other_gap_table))


	#> sensory neurons to other neurons connection

	chemical_connections=Synapses(sensory_neurons, other_neurons, model=model_chem_syn_connection, on_pre='g = 1', name='inter_connections', method='linear')
	chemical_connections.connect(i=sens_other_connections_table[:,0],j=sens_other_connections_table[:,1])
	


	### INITIAL VALUES
	sensory_neurons.v = other_neurons.v = -70*mV#c
	sensory_neurons.u = other_neurons.u = b*(-70*mV)#b*c



	network = Network([sensory_neurons, sensory_gap_junctions, sensory_synapses, other_neurons, other_synapses, other_gap_junctions, chemical_connections])
	return network

def reset_internal_states(network):

	network['sensory_neurons'].v = network['other_neurons'].v = -70*mV#c
	network['sensory_neurons'].u = network['other_neurons'].u = b*(-70*mV)#b*c

	network['sensory_synapses'].g = network['other_synapses'].g = network['inter_connections'].g = 0





def set_parameter(network, original_parameter):
	if(len(original_parameter)!=33):
		raise ValueError('not the right number of parameter')
	parameter = [param for param in original_parameter if 0<=param<=1]
	#gives us a copy of the array and lets us check if any parameter is not in the range
	if(len(parameter)!=33):
		raise ValueError('there was a parameter not 0<=param<=1')
	# if not all(logical_or(binaryParameter==1, binaryParameter==0)):
	# 	raise ValueError('binary parameter are not all binary')


	binaryParameter=[]
	for i in range(27): # the first 27 are chemical synampses and need to be translated in to exit/inhibit
		if parameter[i]>=0.5:
			parameter[i] = (parameter[i]-0.5)*0.5
			binaryParameter.append(0) # excitatory reversal potential
		else:
			parameter[i] = (0.5 - parameter[i])*0.5
			binaryParameter.append(1) # inhibitory reversal potential

	# print(parameter)
	# print(binaryParameter)

	network['sensory_synapses'].w 		=multiply(parameter[0:3],sensory_chem_table[:,2])*1/ms
	network['other_synapses'].w 		=multiply(parameter[3:23],other_chem_table[:,2])*1/ms
	network['inter_connections'].w 		=multiply(parameter[23:27],sens_other_connections_table[:,2])*1/ms

	#gap junctions are fine between 0 and 1
	network['sensory_gap_junctions'].w 	=clip(get_gap_weight(parameter[27:28], sensory_gap_table),0,1)/ms
	network['other_gap_junctions'].w 	=clip(get_gap_weight(parameter[28:32], other_gap_table),0,1)/ms

	#directionality of synapses here only values 0 or 1
	#0 is excitatory, 1 is inhibitory
	network['sensory_synapses'].vReversal  = binaryParameter[0:3]  *(-70*mV)
	network['other_synapses'].vReversal    = binaryParameter[3:23] *(-70*mV)
	network['inter_connections'].vReversal = binaryParameter[23:27]*(-70*mV)


	#set the tau for chemical synapses globally
	global tauChemSyn
	tauChemSyn= (parameter[-1]*19 + 0.1)*ms

def runWithInput(network, groupName, inputs, timeWindows):
	"""inputs is a dictonary with parameter:[list of values]. all lists of values 
	must have same length N. timewindows is a list of the length for wich each of 
	the inputs will be presented"""
	for i,duration in enumerate(timeWindows):
		for key in inputs:
			setattr(network[groupName], key, inputs[key][i])
		network.run(duration)#, report_period=60*second)
		print('network just ran for {}'.format(duration))
