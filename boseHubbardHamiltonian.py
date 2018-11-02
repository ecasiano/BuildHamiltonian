#Build Hamiltonian Matrix representing the Bose-Hubbard Model

from sympy.utilities.iterables import multiset_permutations,ordered_partitions
import numpy as np

def bosonicConfigurations(L,N):
    '''Input: 1D Lattice Size and Number of Bosons
    Output: All possible configurations of bosons'''
    
    #List that will store all configurations
    configurations = []
      
    #Store ordered partitions of N as a list
    partitions = list(ordered_partitions(N))
    
    for p in partitions:
        #BH Lattice containing a partition of N followed by zeros
        auxConfig = [0]*L
        auxConfig[0:len(p)] = p

        #Generate permutations based on current partition of N
        partitionConfigs = list(multiset_permutations(auxConfig))

        #Append permutations of current partition to list containing all configurations
        configurations += partitionConfigs
    
    #Promote configurations list to numpy array
    configurations = np.array(configurations)
      
    return configurations

def boseHubbardKinetic(bra,ket,t=1):
    '''Give a state and apply the kinetic operator of the BH-Model
    to determine its contribution to the total kinetic energy'''

    m = np.copy(ket)
    L = np.shape(ket)[0] #Number of total sites in the configuration
    kineticSum = 0 #Initialize total kinetic energy contribution
    
    #Loop for bdag_i*b_{i+1}
    for i in range(L):
        #Create boson on site i.
        m[i] += 1

        #Annihilate boson on site i+1 (do nothing if no bosons)
        if m[(i+1)%(L)] != 0: m[(i+1)%(L)] -= 1 
        else: m[(i+1)%(L)] = 0
        
        if np.array_equal(bra,m):
            kineticSum += np.sqrt(ket[i]+1)*np.sqrt(ket[(i+1)%(L)])
            print("EQUAL")
            print(ket,m)
        else:
            print("DIFFERENT")
            print(ket,m)
        
        #Make m a copy of the original state n again.
        m = np.copy(ket)
    
    #Loop for b_i*bdag_{i+1}
    for i in range(L):
        #Create boson on site i+1.
        m[(i+1)%(L)] += 1

        #Annihilate boson on site i (do nothing if no bosons)
        if m[i] != 0: m[i] -= 1 
        else: m[i] = 0

        if np.array_equal(bra,m):
            kineticSum += np.sqrt(ket[i])*np.sqrt(ket[(i+1)%(L)]+1)
            print("EQUAL")
            print(ket,m)
        else:
            print("DIFFERENT")
            print(ket,m)

        #Make m a copy of the original state n again.
        m = np.copy(ket)
        
    return -kineticSum

def boseHubbardHamiltonian(configurations):
    '''Input: Set of all possible configurations of bosons on a 1D Lattice'''
    
    #Store HilbertSpace Size
    hilbertSize = np.shape(configurations)[0]
    print(hilbertSize)
    
    #Store Lattice Size
    L = np.shape(configurations)[1]
    
    #Initialize Hamiltonian Matrix
    H = np.zeros((hilbertSize,hilbertSize))
    
    #Fill in upper diagonal of the Hamiltonian
    for i in range(hilbertSize):
        bra = configurations[i]
        for j in range(i,hilbertSize):
            ket = configurations[j]
            H[i,j] = boseHubbardKinetic(bra,ket)
            H[j,i] = H[i,j] #Use Hermiticity to fill up lower diagonal
            
    return H

def main():
    
    #BUG REPORT: ONLY WORKS FOR N <= L at the moment
    #Parameters
    L = 6
    N = 3
    
    #Store all possible configurations of N bosons on L lattice sites
    configurations = bosonicConfigurations(L,N)
        
    #Hamiltonian
    H = (boseHubbardHamiltonian(configurations))
    
    print("")
    print("Hamiltonian:")
    print("")
    print(H)
    
    print("")
    print("Configurations: ",np.shape(H)[0])
    
    #Find ground state energy and state of the Hamiltonian
    eigs = np.linalg.eigh(H)
    egs = eigs[0][0]
    psi = eigs[1][0]
    print("")
    print("Ground State Energy: ",egs)
    print("Ground State: ",psi)
    print("norm(psi): ",np.linalg.norm(psi))

if __name__ == "__main__":
    main()   