# Simiulating the Phototaxis Response in C Elegans 

A paper describing the general aproach can be found in [doc/doc.pdf](doc/doc.pdf)

## Instalation
You need to install the [brian2 simulator](https://brian2.readthedocs.io/en/stable/introduction/install.html) (including Cython) and matplotlib/pyplot for plotting. 

## Descrition of Files
### evolution.py
This file contains the evolutionary algorithm. Run 'python3 evolution.py' to start the evolution from scratch, or run 'python3 evolution.py population N' two load the initial population from the file named 'populationN.npy'. In the first lines of the file all the parameter can be specified. After each generation the current population is saved in 'populationN.npy'. If you put an empty file named 'stop', the evolution will terminate after the current generation is finished. 

### examine\_evolution.py
This file lets you plot the development of the fitness, the volume in which the individuals are and how much the centroid of all individuals moves from generation to generation. In the folder with all the .npy population files run 'python3 examine\_evolution.py'

### photoaxis.py
This file is used to asses the quality of a set of parameters. It is ussually called within evolution.py. Call with 'python3 phototaxis.py populationName.npy N pictureFileName' and it will simulate the best individual from the population in the .npy file for N times and save plots of it into pictureFileName.png files. (numbered). If you ommit the pictureFileName it will just show the plots. 

### phototaxis\_network.py
This is were the actuall simulation of the network happens. There is no point in running it direclty

### my\_helper.py
This one contains different versions of the fitness function. assesQuality4() is the current one. 

### populations/.
In this folder all the generations for one run can be found. Plus plots for 10 runs with the fittest individual of the last generation. 

## Contact
Please do not hesitate to contact me with any questions: [chutter@uos.de](mailto:chutter@uos.de)
