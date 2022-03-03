import numpy as np
import operator
import sys

def distance(a,b):
    # Returns the 3D distance between a and b where 
    # a and b are array where x, y and z are at the
    # position 0, 1 and 2

    x = a[0]-b[0]
    y = a[1]-b[1]
    z = a[2]-b[2]

    return np.sqrt(x**2+y**2+z**2)

def get_cell_matrix(a,b,c,alpha,beta,gamma):
    # Computing the volume of the primitive cell
    omega = a*b*c*np.sqrt(1-np.cos(alpha)**2-np.cos(beta)**2-np.cos(gamma)**2+2*np.cos(alpha)*np.cos(beta)*np.cos(gamma))

    # Computing the matrix 
    M = [
            [a,b*np.cos(gamma),c*np.cos(beta)],
            [0,b*np.sin(gamma),c*(np.cos(alpha)-np.cos(beta)*np.cos(gamma))/(np.sin(gamma))],
            [0,0,omega/(a*b*np.sin(gamma))]
        ]
    return M

def big_cell(generator,symGenerator,a,b,c,alpha,beta,gamma,nA,nB,nC):
    coords = []
    
    # Computing the matrix converting fractional to cartesian
    fracToCart = get_cell_matrix(a,b,c,alpha,beta,gamma)

    for gen in generator:
        x = gen[1]
        y = gen[2]
        z = gen[3]

        for sym in symGenerator:
            u = eval(sym[0])
            v = eval(sym[1])
            w = eval(sym[2])

            # Making sure the value is within the range [0,1]
            u = u + 1*(u<0) - 1*(u>1)
            v = v + 1*(v<0) - 1*(v>1)
            w = w + 1*(w<0) - 1*(w>1)
            
            coords.append([u,v,w,gen[0]])

    # Deleting the redundant atoms
    toDel = []
    for i in range(len(coords)-1):
            for j in range(i+1,len(coords)):
                # Computing the distance using the minimum image convention
                # as described in Appendix B equation 9 of 
                # "Statistical Mechanics : Theory and Molecular Simulations
                # Mark E. Tuckerman"

                r1 = np.array(coords[i][:3])
                r2 = np.array(coords[j][:3])
                r12 = r1-r2
                da = np.sqrt(r12[0]**2+r12[1]**2+r12[2]**2)
                r12 = r12 - np.round(r12)
                db = da - np.sqrt(r12[0]**2+r12[1]**2+r12[2]**2)
                r12 = np.matmul(fracToCart,r12)
                d = np.sqrt(r12[0]**2+r12[1]**2+r12[2]**2)

                if(d<1e-2):
                    # We check if we don't already want to delete this atom
                    if j not in toDel:
                        toDel.append(j)

    toDel = sorted(toDel)

    # We delete the atoms in the list
    for i in range(len(toDel)):
        coords.pop(toDel[i]-i)

    newCoords = []

    # We replicate the cell nA, nB, nC times 
    for at in coords:
        newCoords.append([at[0],at[1],at[2],at[3]])
        for a in range(1,nA):
            newCoords.append([at[0]+a,at[1],at[2],at[3]])
            for b in range(1,nB):
                newCoords.append([at[0]+a,at[1]+b,at[2],at[3]])
                for c in range(1,nC):
                    newCoords.append([at[0]+a,at[1]+b,at[2]+c,at[3]])
            for c in range(1,nC):
                newCoords.append([at[0]+a,at[1],at[2]+c,at[3]])
        for b in range(1,nB):
            newCoords.append([at[0],at[1]+b,at[2],at[3]])
            for c in range(1,nC):
                newCoords.append([at[0],at[1]+b,at[2]+c,at[3]])
        for c in range(1,nC):
            newCoords.append([at[0],at[1],at[2]+c,at[3]])

    # Now we convert the fractionnal coordinates to cartesian coordinates
    coords = []

    for at in newCoords:
        r = [at[0],at[1],at[2]]
        rxyz = np.matmul(fracToCart,r)
        coords.append([rxyz[0],rxyz[1],rxyz[2],at[3],'C'])


    # Returns the list of the atoms [x,y,z,label,second_label]
    return coords

# Translates all the coordinates with the vector v
def translate(v,coordinates):
    for c in coordinates:
        c[0] += v[0]
        c[1] += v[1]
        c[2] += v[2]
    return coordinates

# Finds the point at the center of the given atoms that are the 
# closest to the origin
def find_center(centerList, coordinates):

    centers = []
    for i in range(len(centerList)):
        centers.append([100,100,100])  # Setting a large value for each center

    for c in centers:
        c.append(distance(c,[0,0,0]))  # Computing the distance to the origin

    for at in coordinates:
        if at[3] in centerList:
            centers = sorted(centers, key=operator.itemgetter(3)) # Sorting the list with respect to the distance to the origin
            d = distance(at,[0,0,0])
            if d <= centers[-1][-1] and d > 0.0:
                centers[-1] = [at[0],at[1],at[2],d]

    center = np.mean(centers,axis=0)[:3] # Computing the barycenter

    return center

# Defines a rotation matrix that will put r1 at the position r2
def rotation_matrix(r1,r2):

    r1 = np.array(r1)/np.linalg.norm(r1)
    r2 = np.array(r2)/np.linalg.norm(r2)

    # Computing the cross product which is the vector around which
    # the rotation is done
    crossProduct = np.cross(r1,r2)
    crossProduct = crossProduct/np.linalg.norm(crossProduct)

    # Computing the angle of the rotation
    a = np.arccos(np.dot(r1,r2))

    c = np.cos(a)
    s = np.sin(a)
    x = crossProduct[0]
    y = crossProduct[1]
    z = crossProduct[2]

    M = [
            [x**2*(1-c)+c,x*y*(1-c)-z*s,x*z*(1-c)+y*s],
            [x*y*(1-c)+z*s,y**2*(1-c)+c,y*z*(1-c)-x*s],
            [x*z*(1-c)-y*s,y*z*(1-c)+x*s,z**2*(1-c)+c]
            ]

    return M

# Rotates all the coordinates using the rotation matric M
def rotate(M,coordinates):
    for i in range(len(coordinates)):
        r = [coordinates[i][0],coordinates[i][1],coordinates[i][2]]
        rV = np.matmul(M,r)
        coordinates[i][0] = rV[0]
        coordinates[i][1] = rV[1]
        coordinates[i][2] = rV[2]

    return coordinates

# Cuts a sphere centered on the origin in the coordinates
def cut_sphere(coordinates,r):
    sphere = []
    for i in range(len(coordinates)):
        if distance(coordinates[i],[0,0,0]) <= r:
            sphere.append(coordinates[i])

    return sphere

# Finds the fragment in the coordinates
def find_fragment(coordinates, patterns, npatterns,notInFrag):

    inFrag = []

    for n in range(len(patterns)):
        pattern = patterns[n]
        npattern = npatterns[n]
        for i in range(npattern):
            c = [100,100,100]
            dc = distance([0,0,0],c)

            inPattern = []
            # Finding the closest atom of the first type in the pattern
            for at in coordinates:
                if at[3] == pattern[1]:
                    d = distance([0,0,0],at)
                    if d > 10:
                        break
                    if d < dc :
                        accept = True
                        for exc in notInFrag:
                            d = distance(exc,at)
                            if d < 1e-5:
                                accept = False
                        if accept and coordinates.index(at) not in inFrag:
                            c = [at[0],at[1],at[2],0.0, coordinates.index(at)]
                            dc = distance([0,0,0],c)
            # Finding the rest of the pattern around the atom previously found
            atIn = []
            for j in range(0,len(pattern),2):
                d = distance(c,[100,100,100])
                # Initializing the atoms 
                for k in range(pattern[j]):
                    atIn.append([100,100,100,d])

                for at in coordinates:
                    if distance(at,[0,0,0]) > 10:
                        break
                    if at[3] == pattern[j+1]:
                        atIn = sorted(atIn,key=operator.itemgetter(3))
                        d = distance(at,c)
                        trial = [at[0],at[1],at[2],d,coordinates.index(at)]
                        if d < atIn[-1][3] and trial not in atIn:
                            accept = True
                            for exc in notInFrag:
                                d = distance(exc,trial)
                                if d < 1e-5:
                                    accept = False
                            if accept:
                                atIn[-1] = trial
            for at in atIn:
                inPattern.append(at[4])

            for at in inPattern:
                if at not in inFrag:
                    inFrag.append(at)

    for at in inFrag:
        coordinates[at][4] = 'O'

    return len(inFrag), coordinates


# Finds the pseudopotential layer around
# the fragment
def find_pseudo(coordinates, rPP, notInPseudo):

    for at in coordinates:
        if at[4] != 'O':
            continue
        for i in range(len(coordinates)):
            if coordinates[i][4] != 'C':
                continue
            d = distance(at,coordinates[i])
            if d < rPP:
                coordinates[i][4] = 'Cl'

    return coordinates

# Creates lists containing the neighbours of each
# atom
def find_neighbours(coordinates, atoms):
    neighbourList = [[] for i in range(len(coordinates))]

    atoms = np.array(atoms).flatten()

    for i in range(len(coordinates)-1):
        for j in range(i+1,len(coordinates)):
            li = coordinates[i][3] # Label of the atom i
            lj = coordinates[j][3] # Label of the atom j

            ii = np.where(atoms==li)[0]
            jj = np.where(atoms==lj)[0]

            ci = float(atoms[ii+1]) # Charge of the atom i
            cj = float(atoms[jj+1]) # Charge of the atom j

            if ci*cj < 0: # Checking if the charges have opposite signs
                d = distance(coordinates[i],coordinates[j])

                if d < float(atoms[ii+3]) and d < float(atoms[jj+3]):
                    neighbourList[i].append(j)
                    neighbourList[j].append(i)
    return neighbourList

# For each atom, finds if it has the correct number of neighbours,
# if not, modify its charge 
def evjen_charges(coordinates,atoms):
    neighbourList = find_neighbours(coordinates,atoms)

    atoms = np.array(atoms).flatten()

    charges = []

    for i in range(len(coordinates)):
        li = coordinates[i][3]
        ii = np.where(atoms==li)[0]

        nr = len(neighbourList[i])
        nt = int(atoms[ii+2])
        ci = float(atoms[ii+1])
        
        if nr > nt:
            print("Error : too much neighbours for atom n°%i, count %i neighbours where it should have a maximum of %i"%(i,nr,nt))
            sys.exit()
        charges.append(ci*nr/nt)

    return charges

# Computes the nuclear repulsion 
def nuclear_repulsion(coordinates,charges):

    rep = 0.0
    
    for i in range(len(coordinates)-1):
        for j in range(i+1,len(coordinates)):
            rij = distance(coordinates[i],coordinates[j])
            ci = charges[i]
            cj = charges[j]

            if(rij < 1):
                print(i,j,"\n",coordinates[i],"\n",coordinates[j],"\n",rij)

            rep += (ci*cj)/rij
    return rep

# Computes the symmetry in the whole system
def compute_symmetry(coordinates,charges,symmetry):
    symmetrizedCoordinates = []
    symmetrizedCharges = []
    uniqueIndexList = [] # The list containing the indexes of the unique atoms

    treated = [] # Will store the index of the atoms already treated

    symOp = []

    # Storing all the symmetry operations
    for s in symmetry:
        if s == 'C2x':
            symOp.append(np.array([1,-1,-1]))
        elif s == 'C2y':
            symOp.append(np.array([-1,1,-1]))
        elif s == 'C2z':
            symOp.append(np.array([-1,-1,1]))
        elif s == 'xOy':
            symOp.append(np.array([1,1,-1]))
        elif s == 'xOz':
            symOp.append(np.array([1,-1,1]))
        elif s == 'yOz':
            symOp.append(np.array([-1,1,1]))
        elif s == 'i':
            symOp.append(np.array([-1,-1,-1]))

    for i in range(len(coordinates)):
        if i in treated:
            continue
        
        treated.append(i)
        at1 = np.array(coordinates[i][:3])
        symmetrizedCoordinates.append(coordinates[i])
        symmetrizedCharges.append(charges[i])
        uniqueIndexList.append(len(symmetrizedCoordinates)-1)

        for j in range(len(coordinates)):
            if j in treated or coordinates[i][3] != coordinates[j][3]:
                continue

            at2 = np.array(coordinates[j][:3])

            for s in symOp:
                if distance(at1,at1*s) > 1e-4 and distance(at2,at1*s) < 1e-4: # Checking if op.at1 != at1 and that op.at2 = at1
                    p = at1*s
                    treated.append(j)
                    symmetrizedCoordinates.append([p[0],p[1],p[2],coordinates[i][3],coordinates[i][4]])
                    symmetrizedCharges.append(charges[i])
                    break

    return symmetrizedCoordinates,symmetrizedCharges,uniqueIndexList
