import sys
import numpy as np

from src.read_input import *
from src.out import *
from src.utils import *

if __name__=='__main__':

    verbose = 2

    if len(sys.argv) == 1:
        print("Please provide the input file")
        sys.exit()
    elif len(sys.argv) == 3 :
        if sys.argv[2] == 's':
            verbose = 0
        elif sys.argv[2] == 'v':
            verbose = 1
        elif sys.argv[2] == 'vv':
            verbose = 2
        elif sys.argv[2] == 'vvv':
            verbose = 3
        else:
            print("Unknown argument -%s-"%sys.argv[2])
            sys.exit()
    elif len(sys.argv) > 3:
        print("Too much arguments, please provide an input file and a verbose level (v, vv, vvv)")
        sys.exit()

    inputFile = sys.argv[1]

    # Reads all the parameters from the input file
    rB , rPP, center, xOy, xOz, yOz, X, Y, Z, symmetry, outputFile, pattern, npattern , atoms, dist, a, b, c, alpha, beta, gamma, showBath, evjen, showFrag, notInPseudo, notInFrag, symGenerator, generator, translation = read_input(inputFile)

    if verbose > 0:
        out_input_param(rB , rPP, center, X, Y, Z, symmetry, outputFile, pattern, npattern , atoms, dist, a, b, c, alpha, beta, gamma, showBath, evjen, showFrag, notInPseudo, notInFrag, symGenerator, generator, translation)

    # Converting the angles to radian
    alpha = alpha * np.pi / 180.0
    beta = beta * np.pi / 180.0
    gamma = gamma * np.pi / 180.0


    # Computing the number of replications needed in each directions using the interreticular distance
    # for the planes 100, 010 and 001.
    # 
    # The condition is : d_{hkl} >= 2*bath_radius + |translation_vector|

    fac = np.sqrt(1-np.cos(alpha)**2-np.cos(beta)**2-np.cos(gamma)**2+2*np.cos(alpha)*np.cos(beta)*np.cos(gamma))

    nA = int(np.ceil(np.sin(alpha)*(2*rB+np.linalg.norm(translation))/(a*fac)))+1
    nB = int(np.ceil(np.sin(beta)*(2*rB+np.linalg.norm(translation))/(b*fac)))+1
    nC = int(np.ceil(np.sin(gamma)*(2*rB+np.linalg.norm(translation))/(c*fac)))+1

    if verbose > 1:
        print("The big cell will be of dimensions %2ix%2ix%2i\n"%(nA,nB,nC))

    coordinates = big_cell(generator,symGenerator,a,b,c,alpha,beta,gamma,nA,nB,nC)

    # Computing the translation vector by addition of the user translation vector and a translation 
    # vector that puts the origin at the center of the big cell
    t = [-0.5,-0.5,-0.5]
    M = get_cell_matrix(nA*a,nB*b,nC*c,alpha,beta,gamma)
    t = np.matmul(M,t)
    t = [t[0]+translation[0],t[1]+translation[1],t[2]+translation[2]]

    # Translating the coordinates
    coordinates = translate(t, coordinates)

    # Finding the center and translating the coordinates
    # If this vector creates a displacment bigger than a, b or c
    # in any of the abc directions, this might result in an incomplete
    # sphere later, the user should provide a translation vector
    # to correct this
    if center != []:
        c = find_center(center,coordinates)
        coordinates = translate(-c,coordinates)

    
    # Orienting the big cell
    if xOy != []:
        a = find_center(xOy[0], coordinates)
        b = a
        w = [a]
        while np.absolute(np.absolute(np.dot(a/np.linalg.norm(a),b/np.linalg.norm(b)))-1) < 1e-6:
            b = find_center(xOy[1], coordinates, without=w)
            w.append(b)
        c = np.cross(a,b)
        M = rotation_matrix(c, [0,0,1])
        coordinates = rotate(M, coordinates)
    if xOz != []:
        a = find_center(xOz[0], coordinates)
        b = a
        w = [a]
        while np.absolute(np.absolute(np.dot(a/np.linalg.norm(a),b/np.linalg.norm(b)))-1) < 1e-6:
            b = find_center(xOz[1], coordinates, without=w)
            w.append(b)
        c = np.cross(a,b)
        M = rotation_matrix(c, [0,1,0])
        coordinates = rotate(M, coordinates)
    if yOz != []:
        a = find_center(yOz[0], coordinates)
        b = a
        w = [a]
        while np.absolute(np.absolute(np.dot(a/np.linalg.norm(a),b/np.linalg.norm(b)))-1) < 1e-6:
            b = find_center(yOz[1], coordinates, without=w)
            w.append(b)
        c = np.cross(a,b)
        M = rotation_matrix(c, [1,0,0])
        coordinates = rotate(M, coordinates)
    if X != []:
        k = [1,0,0]

        xVec = find_center(X,coordinates)
        M = rotation_matrix(xVec, k)

        coordinates = rotate(M, coordinates)
    if Y != []:
        k = [0,1,0]

        yVec = find_center(Y,coordinates)
        M = rotation_matrix(yVec, k)

        coordinates = rotate(M, coordinates)
    if Z != []:
        k = [0,0,1]

        zVec = find_center(Z,coordinates)
        M = rotation_matrix(zVec, k)

        coordinates = rotate(M, coordinates)

    if verbose > 2:
        print("The big cell contains %5i atoms and will be printed in the file big_cell.xyz\n"%len(coordinates))
        write_coordinates(coordinates,'big_cell.xyz',3)

    # Cutting the sphere in the big cell
    coordinates = cut_sphere(coordinates,rB)

    if verbose > 2:
        print("The sphere contains %5i atoms and will be printed in the file sphere.xyz\n"%len(coordinates))
        write_coordinates(coordinates,'sphere.xyz',3)

    # Finding the fragment

    coordinates = sorted(coordinates, key=lambda x:distance(x,[0,0,0]))

    nAt, coordinates = find_fragment(coordinates,pattern,npattern,notInFrag)

    if verbose > 2 or showFrag:
        print("The fragment contains %3i atoms and will be printed in the file fragment.xyz\n"%nAt)
        write_coordinates(coordinates,'fragment.xyz',4,'O')

    coordinates = find_pseudo(coordinates,rPP,notInPseudo)

    if verbose > 2 or showBath:
        print("The bath will be printed in the file bath.xyz\n")
        write_coordinates(coordinates,'bath.xyz',3)
        print("The bath sorted with the fragment/pseudo/charge will be printed in the file bath_coloured.xyz\n")
        write_coordinates(coordinates,'bath_coloured.xyz',3,color='yes')

    if evjen:
        charges = evjen_charges(coordinates,atoms)
    else:
        charges = []
        atoms = np.array(atoms).flatten()
        for i in range(len(coordinates)):
            li = coordinates[i][3]
            ii = np.where(atoms==li)[0]
            charges.append(float(atoms[ii+1]))

    if verbose > 1:
        print("The total charge fragment+pseudopotential+bath is : % 8.6f\n"%np.sum(charges))

    if symmetry != []:
        nuc1 = nuclear_repulsion(coordinates,charges)
        if verbose > 1:
            print("Nuclear repulsion before the symmetry : % 8.6f\n"%nuc1)

        coordinates,charges,indexList = compute_symmetry(coordinates,charges,symmetry)

        nuc2 = nuclear_repulsion(coordinates,charges)
        if verbose > 1:
            print("Nuclear repulsion after the symmetry  : % 8.6f\n"%nuc2)
            print("The total charge fragment+pseudopotential+bath after symmetry is : % 8.6f\n"%np.sum(charges))
            if verbose > 2:
                print("The symmetrized coordinates contain %5i atoms \n"%len(indexList))

    else:
        indexList = [i for i in range(len(coordinates))]

    write_output(outputFile,coordinates,charges,indexList)
    if verbose > 2:
        print("The output has been written to %s \n"%outputFile)
        out_interatomic_distances(coordinates)

    with open("output.tcl",'w') as f:
        tp = [atoms[i] for i in range(0, len(atoms), 4)]
        f.write("mol new fragment.xyz\nmol delrep 0 0\nmol representation CPK\n")
        for i in tp:
            if i in ["Sc", "Ti", "V" , "Cr", "Mn", "Fe", "Co", "Ni", "Cu", "Zn",
                     "Y" , "Zr", "Nb", "Mo", "Tc", "Ru", "Rh", "Pd", "Ag", "Cd",
                     "La", "Hf", "Ta", "W" , "Re", "Os", "Ir", "Pt", "Au", "Hg"]:
                f.write('mol selection "type {:s}"\nmol color ColorID 3\nmol addrep 0\n'.format(i))
            elif i in ["F", "Cl", "Br", "I"]:
                f.write('mol selection "type {:s}"\nmol color ColorID 7\nmol addrep 0\n'.format(i))
            else:
                f.write('mol selection "type {:s}"\nmol color Name\nmol addrep 0\n'.format(i))

        f.write("mol new bath_coloured.xyz\nmol delrep 0 1\nmol representation Points\n")
        f.write('mol selection "type Cl"\nmol color colorID 17\nmol addrep 1\n')
        f.write('mol selection "type C"\nmol color colorID 0\nmol addrep 1\n')
                

