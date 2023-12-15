import numpy as np
from numba import jit
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import vtk

width = 1.      # domain width
height = 1.     # domain height
nx = 30         # number of grid points in x-direction
ny = 30         # number of grid points in y-direction
dx = width/nx   # grid size in x-direction
dy = height/ny  # grid size in y-direction


c = 0.03          # thermal diffusivity
dt = (dx**2)/(4*c)  # time step size
t_end = 100.        # simulation length
tolerance = 1e-5 # toleration for convergence

uo = np.zeros((nx+2,ny+2)) # temperature at current time step
un = np.zeros((nx+2,ny+2)) # temperature at next time step

def create_vtk_grid(un, nx, ny, dx, dy):
    vtk_grid = vtk.vtkStructuredGrid()
    vtk_grid.SetDimensions(nx, ny, 1)

    # Create vtk points
    points = vtk.vtkPoints()
    for j in range(ny):
        for i in range(nx):
            points.InsertNextPoint(i * dx, j * dy, 0)

    vtk_grid.SetPoints(points)

    # Add temperature data to the grid
    temperature_array = vtk.vtkDoubleArray()
    temperature_array.SetNumberOfComponents(1)
    temperature_array.SetName("Temperature")

    for j in range(ny):
        for i in range(nx):
            temperature_array.InsertNextValue(un[i+1, j+1])

    vtk_grid.GetPointData().AddArray(temperature_array)

    return vtk_grid

def write_vtk_file(vtk_grid, filename):
    writer = vtk.vtkXMLStructuredGridWriter()
    writer.SetFileName(filename)
    writer.SetInputData(vtk_grid)
    writer.Write()

# update scheme
@jit(nopython=True, nogil=True, fastmath=True)
def comp_temperature(un, uo, c, dx, dy, dt, nx, ny):
  for i in range(1, nx+1):
    for j in range(1, ny+1):
      un[i,j] = uo[i,j] + c*dt*( (uo[i+1,j] - 2*uo[i,j] + uo[i-1,j])/(dx**2) + (uo[i,j+1] - 2*uo[i,j] + uo[i,j-1])/(dy**2) )

# set boundary conditions
def update_boundary(u, nx, ny):
  # warm point in the middle
  u[int(nx/2),int(ny/2)] = 1.

  # neumann boundary conditions all 4 sides
  u[0,:] = u[1,:] # left
  u[-1,:] = u[-2,:] # right
  u[:,0] = u[:,1] # bottom
  u[:,-1] = u[:,-2] # top

# simulation main loop
t = 0.
while t < t_end:
  update_boundary(uo, nx, ny)
  comp_temperature(un, uo, c, dx, dy, dt, nx, ny)

  if abs(t - round(t)) < tolerance:
    vtk_grid = create_vtk_grid(un, nx, ny, dx, dy)
    write_vtk_file(vtk_grid, f"vtk_files/output_{int(t)}.vts")

  for i in range(1, nx+1):
    for j in range(1, ny+1):
      uo[i,j] = un[i,j]
  print(f't: {t:06.2f}')
  t = t + dt

# plot results
x = np.linspace(0, 1, nx+2)
y = np.linspace(0, 1, ny+2)
X, Y = np.meshgrid(x, y)
plt.xlabel(r'$x$')
plt.ylabel(r'$y$')
plt.contourf(X, Y, np.rot90(un,3), cmap=plt.cm.plasma)
plt.colorbar()
ax = plt.gca()
ax.set_aspect(1)
plt.show()