import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

# Class Beam with:
# 6 attributes: width, height, length, Young's Modulus, number of elements along the beam,
# and length of each element
# 2 methods:(a)Calculate the deflection of each point on the beam based on the beam theory where y(x) = -P/6EI (3Lx2-x3)
# Calculate the deflection of each point on the beam by assembling the stiffness matrix for all elements
class beam(object):
    def __init__(self,b_beam = .5, h_beam = .5, L_beam = 10, E_beam = 2.9e7, num_elements = 10 ):
        self.b_beam = b_beam
        self.h_beam = h_beam
        self.L_beam = L_beam
        self.E_beam = E_beam
        self.num_elements = num_elements
        self.elem_length = L_beam/num_elements

    def def_calc_beam_theory(self,P):
        self.I_beam = (1 / 12) * self.b_beam * self.h_beam ** 3  # I = 1/12 bh3
        self.loc_beam = np.linspace(0, self.L_beam, self.num_elements+1)
        self.defl_beam_theory = -P/(6*self.E_beam*self.I_beam)*(3*self.L_beam*self.loc_beam**2-self.loc_beam**3)
        return self.loc_beam, self.defl_beam_theory

    # I grabbed the formulation of stiffness matrix of each element from literature
    # Shifted the element matrices based on the element number so that they are at the correct location in the assem mat
    # Added the element matrices to yield stiffness matrix of the system
    # Created Force Vector. Two forces are assigned to each node, one is the force and the other is the moment.
    # F = Ku --> u = inv(k)F
    # Created the inverse of the stiffness matrix
    # The first two arrays correspond to the boundary condition of y(0)=ydot(0)=0. So there is no need to include them
    # in the inverse matrix. Including them would result in singularity and therefore, the solution does not work.
    # Set up the deflection matrix. The first two arrays are 0 to suffice the y(0) = ydot(0) = 0 boundary condition.
    # Two DOFs are assigned to each node. One is translation deflection and the other is rotational. I am only
    # interested in translation deflections which are the even arrays of the deflection vector.
    def def_calc_stiffness_matrix(self,P):
        stiffness_matrix_elem = (self.E_beam * self.I_beam) / (self.elem_length ** 3) * np.array(
            [[12, 6 * self.elem_length, -12, 6 * self.elem_length],
             [6 * self.elem_length, 4 * self.elem_length ** 2, -6 * self.elem_length, 2 * self.elem_length ** 2],
             [-12, -6 * self.elem_length, 12, -6 * self.elem_length],
             [6 * self.elem_length, 2 * self.elem_length ** 2, -6 * self.elem_length, 4 * self.elem_length ** 2]])

        # Create Full Stiffness Matrix for all elements
        stiffness_matrix_assem = np.zeros((2 * (self.num_elements + 1), 2 * (self.num_elements + 1)))
        for i in range(1, num_elements + 1):
            stiffness_matrix_elem_expanded = np.zeros((2 * (self.num_elements + 1), 2 * (self.num_elements + 1)))
            for x in range(4):
                for y in range(4):
                    stiffness_matrix_elem_expanded[2 * (i - 1) + x, 2 * (i - 1) + y] = stiffness_matrix_elem[x, y]
            stiffness_matrix_assem += stiffness_matrix_elem_expanded

        F = np.zeros(2 * (self.num_elements + 1))
        F[2 * self.num_elements] = -P

        stiffness_matrix_inv = np.linalg.inv(stiffness_matrix_assem[2:2 * (self.num_elements + 1), 2:2 * (self.num_elements + 1)])

        deflection = np.zeros((2 * (self.num_elements + 1)))
        deflection[2:2 * (self.num_elements + 1)] = np.dot(stiffness_matrix_inv, F[2:2 * (self.num_elements + 1)])
        self.defl = np.zeros(num_elements + 1)

        r = 0
        for i in range(len(deflection)):
            if i % 2 == 0:
                self.defl[r] = deflection[i]
                r += 1

        return self.loc_beam, self.defl


# Reads an ANSYS results file and checks if the file exists. The function does not check if the dimensions of the beam
# match what stated in the code.
def ansys_reader(fname):
    while True:
        try:
            df = pd.read_csv(str(fname), delimiter='\t', usecols=[1, 2])
            print("ANSYS File Found!")
            break
        except FileNotFoundError:
            print("Invalid File Name.\n")
            fname = str(input("Specify the FULL file name again:\t"))

    loc_ansys = df['Length [in]']
    defl_ansys = df['Value [in]']
    return loc_ansys, defl_ansys

# Beginning of the code
# Get the required information from the user
print("This snippet of code gets the required properties of a cantilvered rectangular beam going under a force at the "
      "tip and plots the deflection. \n It also plots ANSYS results as well, if provided by the user. \n "
      "All units are imperial.")
b_beam = float(input("Width of the beam (default = 0.5): \t") or "0.5")
h_beam = float(input("Height of the beam (default = 0.5): \t") or "0.5")
E_beam = float(input("Young Modulus of the beam (default = 2.9e7): \t") or "2.9e7")
L_beam = float(input("Length of the beam (default = 10): \t") or "10")
num_elements = int(input("Number of Elements (default = 10): \t") or "10")
P = float(input("Force (default = 100): \t") or "100")
ansys_exist = int(input("Do you have an ANSYS results file? (0/1): \t"))
# Import results from ANSYS
if ansys_exist:
    filename = str(input("Indicate the FULL name of the ANSYS file: \t"))
    [loc_ansys,defl_ansys] = ansys_reader(filename)


I_beam = (1/12)*b_beam*h_beam**3 # second moment of area of the beam
elem_length = L_beam/num_elements   # Length of each element on the beam

# Create The Beam Object
bent_beam = beam(b_beam,h_beam,L_beam,E_beam,num_elements)

# Calculate the beam deflection using beam theory method defined under the beam class
[loc_beam_theory, defl_beam_theory] = bent_beam.def_calc_beam_theory(P)

# Calculate the beam deflection using stiffness matrix method defined under the beam class
[loc_stiffness_matrix, defl_stiffness_matrix] = bent_beam.def_calc_stiffness_matrix(P)


# Plot the results
plt.plot(loc_stiffness_matrix,defl_stiffness_matrix,'b', label = 'Stiffness Matrix')
plt.plot(loc_beam_theory,defl_beam_theory, 'r', label = 'Beam Theory')
if ansys_exist:
    plt.plot(loc_ansys,defl_ansys,'--g',label = 'ANSYS results')

plt.grid()
plt.title('Width = {a} in; Height = {b} in; Length = {c} in \n'
          'Young Modulus = {d} psi; Force = {e}'.format(a = b_beam, b = h_beam, c = L_beam, d = E_beam, e = P))
plt.xlabel("Length (in)")
plt.ticklabel_format(axis="y", style="sci", scilimits=(0,0))
plt.ylabel("Deflection (in)")
plt.legend()
plt.show()