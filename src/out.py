import numpy as np
import operator

from src.utils import distance

def out_input_param(rB , rPP, center, X, Y, Z, symmetry, outputFile, pattern, npattern , atoms, dist, a, b, c, alpha, beta, gamma, showBath, evjen, showFrag, notInPseudo, notInFrag, symGenerator, generator, translation):
        print("Output file is            : %s\n"%outputFile)
        print("Bath radius               : % 16.10f\n"%rB)
        print("Pseudo-potential width    : % 16.10f\n"%rPP)
        print("Lattice parameters        :")
        print("      a  = % 8.5f"%a)
        print("      b  = % 8.5f"%b)
        print("      c  = % 8.5f"%c)
        print("  alpha  = % 6.5f"%alpha)
        print("  beta   = % 6.5f"%beta)
        print("  gamma  = % 6.5f\n"%gamma)
        print("Group symmetry operations :")
        for i in symGenerator:
            print("    %10s    %10s    %10s"%(i[0],i[1],i[2]))
        print("\nAtomic positions          :")
        for i in generator:
            print("     %2s    % 8.5f     % 8.5f     % 8.5f"%(i[0],i[1],i[2],i[3]))
        print("\nThere %3s %2i %8s :"%('is'*(len(pattern)==1)+'are'*(len(pattern)>1),len(pattern),'pattern'+'s'*(len(pattern)>1)))
        for i in range(len(pattern)):
            print("The following pattern will appear %2i %5s"%(npattern[i],'time'+'s'*(npattern[i]>1)))
            for j in range(0,len(pattern[i]),2):
                print("  %2i  %2s\n"%(pattern[i][j],pattern[i][j+1]),end='')
        print("\nThere %3s %2i %5s"%('is'*(len(atoms)==1)+'are'*(len(atoms)>1), len(atoms), 'atom'+'s'*(len(atoms)>1)))
        print("Label    Charge    Number of neighbours   Maximum bond length")
        for i in atoms:
            print("  %2s     % 5.2f              %2i                 % 6.4f\n"%(i[0],i[1],i[2],i[3]))
        if(len(center)>0):
            print("Centering between the following atom(s) :")
            for i in center:
                print("   %2s   \n"%i,end='')
            print()
        if(len(X) > 0):
            print("Aligning the X axis with the following atom(s) :")
            for i in X:
                print("   %2s   "%i,end='')
            print()
        if(len(Y) > 0):
            print("Aligning the Y axis with the following atom(s) :")
            for i in Y:
                print("   %2s   "%i,end='')
            print()
        if(len(Z) > 0):
            print("Aligning the Z axis with the following atom(s) :")
            for i in Z:
                print("   %2s   "%i,end='')
            print()
        if translation != [0.0,0.0,0.0]:
            print("The following translation will be applied : % 5.3f  % 5.3f   % 5.3f"%(translation[0],translation[1],translation[2]))
        if(len(symmetry) > 0):
            print("Treating the following symmetry operations :")
            for i in symmetry:
                print("   %3s   "%i,end='')
            print()
        if(evjen):
            print("The program will reequilibrate the charges at the limits of the spheres using Evjen method\n")
        if(showFrag):
            print("The program will print the fragment coordinates in the file fragment.xyz\n")
        if(showFrag):
            print("The program will print the bath coordinates in the files bath.xyz and bath_coloured.xyz\n")
        if len(notInPseudo) > 0:
            print("The following %5s will not be considered in the pseudopotential shell :"%('atom'+'s'*(len(notInPseudo)>1)))
            for i in notInPseudo:
                print("  %2s  "%i)
            print()
        if len(notInFrag) > 0:
            print("The following %5s will be excluded from the fragment :"%('atom'+'s'*(len(notInFrag)>1)))
            for i in notInFrag:
                print(" % 8.6f   % 8.6f    % 8.6f"%(i[0],i[1],i[2]))           
            print()

# Write the coordinates to a file using the string
# at the position 'label' as atom name
# if which is specified, it must be an array of labels
# color = 'no' means that the label will be printed
# color = 'yes' means that the color in the bath will be printed
def write_coordinates(coordinates, fileName, label, which='all',color='no'):

    f = open(fileName,'w')
    l = [i[label] for i in coordinates]

    if color=='no':
        lab = 3
    else:
        lab = 4

    unique, count = np.unique(l,return_counts=True)
    labelDic = dict(zip(unique,count))

    if which=='all':
        f.write("%i \n\n"%len(coordinates))
        for i in coordinates:
            f.write("%2s     % 10.6f      % 10.6f      % 10.6f\n"%(i[lab],i[0],i[1],i[2]))
    else:
        count = 0
        for la in which:
            count += labelDic[la]
        f.write("%i \n\n"%count)
        for i in coordinates:
            if i[label] in which:
                f.write("%2s     % 10.6f      % 10.6f      % 10.6f\n"%(i[lab],i[0],i[1],i[2]))

    f.close()

# Writes the output file
def write_output(outputFile, coordinates, charges, indexList):

    f = open(outputFile,'w')

    d = {'O':'FRAGMENT','Cl':'PSEUDOPOTENTIAL','C':'BATH'}

    count = 0

    for t in d:
        f.write("%s\n"%d[t])

        f.write(" Label             x                    y                   z         Charge\n")
        for i in indexList:
            if coordinates[i][4] == t:
                count += 1
                l = coordinates[i][3]+str(count)
                f.write("%6s    % 16.10f    % 16.10f    % 16.10f   % 8.5f\n"%(l,coordinates[i][0],coordinates[i][1],coordinates[i][2],charges[i]))
        f.write("\n")


# Writes the interatomic distances inferior to 4 A
def out_interatomic_distances(coordinates):
    distanceList = []

    for i in range(len(coordinates)-1):
        for j in range(i+1,len(coordinates)):
            a = coordinates[i]
            b = coordinates[j]
            d = distance(a,b)

            accept = True
            if d > 4:
                accept = False
            else:
                for k in distanceList:
                    if ((a[3] == k[0] and b[3] == k[1]) or (a[3] == k[1] and b[3] == k[0])) and np.abs(d-k[2]) < 1e-5:
                        accept = False

            if accept:
                distanceList.append([a[3],b[3],d])

    distanceList = sorted(distanceList,key=operator.itemgetter(2))

    print("All the interatomic distances < 5 A :")
    print("Atom1       Atom2       Distance")
    for i in distanceList:
        print(" %2s          %2s          %6.4f"%(i[0],i[1],i[2]))

