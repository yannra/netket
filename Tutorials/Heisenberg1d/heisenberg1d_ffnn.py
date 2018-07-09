#Copyright 2018 The Simons Foundation, Inc. - All Rights Reserved.

#Licensed under the Apache License, Version 2.0 (the "License");
#you may not use this file except in compliance with the License.
#You may obtain a copy of the License at

#    http://www.apache.org/licenses/LICENSE-2.0

#Unless required by applicable law or agreed to in writing, software
#distributed under the License is distributed on an "AS IS" BASIS,
#WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
#See the License for the specific language governing permissions and
#limitations under the License.


from __future__ import print_function
import json

pars={}

#defining the lattice
pars['Graph']={
    'Name'           : 'Hypercube',
    'L'              : 4,
    'Dimension'      : 2,
    'Pbc'            : True,
}

#defining the hamiltonian
pars['Hamiltonian']={
    'Name'           : 'Heisenberg',
    'TotalSz'        : 0,
}

#defining the wave function
pars['Machine']={
    'Name'           : 'FFNN',
    'Alpha'          : 1,
    'Layers'         : [{'Name':'Convolutional', 'InputChannels': 1, 'OutputChannels':2, "Distance":1, "UseBias":True, 'Activation':'Lncosh'},
    {'Name':'Convolutional', 'InputChannels': 2, 'OutputChannels':2, "Distance":1, "UseBias":True, 'Activation':'Lncosh'}],
    # 'Layers'         : [{'Name':'FullyConnected', 'Inputs': 20, 'Outputs':20,"UseBias":True,'Activation':'Lncosh'}],
    # 'SigmaRand'      : 0.01,

}

#defining the sampler
#here we use Metropolis sampling
#using moves from the matrix elements of the hamiltonian
pars['Sampler']={
    'Name'           : 'MetropolisHamiltonian',
}

#defining the learning method
#here we use the Stochastic Reconfiguration Method
pars['Learning']={
    'Method'         : 'Sr',
    'Nsamples'       : 1.0e3,
    'NiterOpt'       : 1000,
    'Diagshift'      : 0.01,
    'UseIterative'   : True,
    'OutputFile'     : 'test',
    'StepperType'    : 'Sgd',
    'LearningRate'   : 0.01,
}

json_file="heisenberg1d.json"
with open(json_file, 'w') as outfile:
    json.dump(pars, outfile)

print("\nGenerated Json input file: ", json_file)
print("\nNow you have two options to run NetKet: ")
print("\n1) Serial mode: netket " + json_file)
print("\n2) Parallel mode: mpirun -n N_proc netket " + json_file)
