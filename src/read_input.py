import sys

# Parses the input file 
def read_input(inputFile):

    rB = 0.0
    rPP = 0.0
    center = []
    X = []
    Y = []
    Z = []
    xOy = []
    xOz = []
    yOz = []
    symmetry = []
    outputFile = ""
    pattern = []
    npattern = []
    atoms = []
    dist = []
    a = 0.0
    b = 0.0
    c = 0.0
    alpha = 90.0
    beta = 90.0
    gamma = 90.0
    showBath = False
    evjen = False
    showFrag = False
    notInPseudo = []
    notInFrag = []
    symGenerator = []
    generator = []
    translate = [0.0,0.0,0.0]

    checkInput = {'bath':False,'pseudo':False,'output':False,'pattern':False,'npattern':False,'a':False,'b':False,'c':False,'atoms':False,'symmetry_generator':False,'generator':False}

    f = open(inputFile,'r')

    line = 'x'

    while line.casefold() != 'end_of_input':
        line = f.readline().strip()
        ls = line.split()
        if ls == []:
            continue
        if ls[0].casefold() in checkInput:
            checkInput[ls[0].casefold()] = True
        if ls[0].casefold() == 'bath':
            try:
                rB = float(ls[1])
            except ValueError:
                print("Error while parsing the input file : %s is not a valid bath radius value"%(ls[1]))
                sys.exit()
        elif ls[0].casefold() == 'pseudo':
            try:
                rPP = float(ls[1])
            except ValueError:
                print("Error while parsing the input file : %s is not a valid pseudopotential radius value"%(ls[1]))
                sys.exit()
        elif ls[0].casefold() == 'center':
            ls.pop(0)
            center = [i for i in ls]
        elif ls[0].casefold() == 'x_axis':
            ls.pop(0)
            X = [i for i in ls]
        elif ls[0].casefold() == 'y_axis':
            ls.pop(0)
            Y = [i for i in ls]
        elif ls[0].casefold() == 'z_axis':
            ls.pop(0)
            Z = [i for i in ls]
        elif ls[0].casefold() == "xoy":
            xOy.append( f.readline().split() )
            xOy.append( f.readline().split() )
        elif ls[0].casefold() == "xoz":
            xOz.append( f.readline().split() )
            xOz.append( f.readline().split() )
        elif ls[0].casefold() == "yoz":
            yOz.append( f.readline().split() )
            yOz.append( f.readline().split() )
        elif ls[0].casefold() == 'symmetry':
            ls.pop(0)
            symmetry = [i for i in ls]
        elif ls[0].casefold() ==  'output':
            outputFile = ls[1]
        elif ls[0].casefold() == 'pattern':
            line = f.readline()
            while line.strip().casefold() != 'end':
                pattern.append([])
                ls = line.split()
                for i in range(len(ls)):
                    if i%2 == 0:
                        pattern[-1].append(int(ls[i]))
                    else:
                        pattern[-1].append(ls[i])
                line = f.readline()
        elif ls[0].casefold() == 'npattern':
            ls.pop(0)
            try:
                npattern = [int(i) for i in ls]
            except ValueError:
                print("Error while parsing the input file : the number of patterns is not valid %s"%(line))
                sys.exit()
        elif ls[0].casefold() == 'lattice':
            line = f.readline()
            while line.strip().casefold() != 'end':
                ls = line.split()
                if ls[0].casefold() in checkInput:
                    checkInput[ls[0].casefold()] = True
                if ls[0].casefold() == 'a':
                    try:
                        a = float(ls[1])
                    except ValueError:
                        print("Error while parsing the input file : bad value for the lattice parameter %s"%ls[1])
                        sys.exit()
                elif ls[0].casefold() == 'b':
                    try:
                        b = float(ls[1])
                    except ValueError:
                        print("Error while parsing the input file : bad value for the lattice parameter %s"%ls[1])
                        sys.exit()
                elif ls[0].casefold() == 'c':
                    try:
                        c = float(ls[1])
                    except ValueError:
                        print("Error while parsing the input file : bad value for the lattice parameter %s"%ls[1])
                        sys.exit()
                elif ls[0].casefold() == 'alpha':
                    try:
                        alpha = float(ls[1])
                    except ValueError:
                        print("Error while parsing the input file : bad value for the lattice parameter %s"%ls[1])
                        sys.exit()
                elif ls[0].casefold() == 'beta':
                    try:
                        beta = float(ls[1])
                    except ValueError:
                        print("Error while parsing the input file : bad value for the lattice parameter %s"%ls[1])
                        sys.exit()
                elif ls[0].casefold() == 'gamma':
                    try:
                        gamma = float(ls[1])
                    except ValueError:
                        print("Error while parsing the input file : bad value for the lattice parameter %s"%ls[1])
                        sys.exit()
                line = f.readline()
        elif ls[0].casefold() == 'atoms':
            line = f.readline()
            while line.strip().casefold() != 'end':
                ls = line.split()
                if(len(ls)) != 4:
                        print("Error while parsing the input file : not enough values given for the atom in line %s"%line)
                        sys.exit()
                try:
                    atoms.append([ls[0], float(ls[1]), int(ls[2]), float(ls[3])])
                except ValueError:
                    print("Error while parsing the input file : bad value for the atom %s"%line)
                line = f.readline()
        elif ls[0].casefold() == 'show_bath':
            showBath = True
        elif ls[0].casefold() == 'translate':
            ls.pop(0)
            try:
                translate = [float(ls[0]),float(ls[1]),float(ls[2])]
            except ValueError:
                print("Error while parsing the input file : the translation vector is not valid %s"%line)
                sys.exit()
        elif ls[0].casefold() == 'not_in_pseudo':
            ls.pop(0)
            notInPseudo = [i for i in ls]
        elif ls[0].casefold() == 'show_frag':
            showFrag = True
        elif ls[0].casefold() == 'evjen':
            evjen = True
        elif ls[0].casefold() == 'symmetry_generator':
            line = f.readline().replace("'","")
            while line.strip().casefold() != 'end':
                symGenerator.append(line.split(','))
                line = f.readline().replace("'","")
        elif ls[0].casefold() == 'generator':
            line = f.readline()
            while line.strip().casefold() != 'end':
                ls = line.split()
                try:
                    generator.append([ls[0],float(ls[1]),float(ls[2]),float(ls[3])])
                except ValueError:
                    print("Error while parsing the input file : bad value for the generator atom %s"%line)
                    sys.exit()
                line = f.readline()
        elif ls[0].casefold() == 'not_in_frag':
            line = f.readline()
            while line.strip().casefold() != 'end':
                ls = line.split()
                try:
                    notInFrag.append([float(ls[0]),float(ls[1]),float(ls[2])])
                except ValueError:
                    print("Error while parsing the input file : bad value for the atom %s"%line)
                    sys.exit()
                line = f.readline()
    f.close()
    for t in checkInput:
        if checkInput[t] == False:
            print("Bad input : missing the keyword -- %s --"%t)
            sys.exit()

    return rB , rPP, center, xOy, xOz, yOz, X, Y, Z, symmetry, outputFile, pattern, npattern , atoms, dist, a, b, c, alpha, beta, gamma, showBath, evjen, showFrag, notInPseudo, notInFrag, symGenerator, generator, translate
