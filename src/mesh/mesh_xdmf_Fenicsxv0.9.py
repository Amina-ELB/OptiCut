from dolfinx.io import XDMFFile
import gmsh
from dolfinx.io.gmshio import model_to_mesh
from mpi4py import MPI
import math
mesh_comm = MPI.COMM_WORLD
gdim = 2
model_rank = 0

gmsh.initialize()

case = "L_shape" # "cruciform_Chen2022_simplify", "dogbone_1holl", "cruciform_Chen2022", "dobgone", "dogbone_immersed","carre"
if case == "L_shape":
    
    L = 1
    H = 0.4

    # Récupérer la factory
    factory = gmsh.model.geo

    # Points:
    lc = 0.009
    p1 = factory.add_point(0, 0, 0, lc,1)
    p2 = factory.add_point(L, 0, 0, lc,2)
    p3 = factory.add_point(L, H, 0, lc,3)
    p4 = factory.add_point(H, H, 0, lc,4)
    p5 = factory.add_point(H, L, 0, lc,5)
    p6 = factory.add_point(0, L, 0, lc,6)

    # Définir les lignes du rectangle
    l1 = factory.addLine(p1, p2)
    l2 = factory.addLine(p2, p3)
    l3 = factory.addLine(p3, p4)
    l4 = factory.addLine(p4, p5)
    l5 = factory.addLine(p5, p6)
    l6 = factory.addLine(p6, p1)

    # Créer la surface
    loop = factory.addCurveLoop([l1, l2, l3, l4, l5, l6])
    surface = factory.addPlaneSurface([loop])

# Synchroniser la géométrie
gmsh.model.geo.synchronize()
    
# Ajouter une entité physique (nécessaire pour FEniCSx)
gmsh.model.addPhysicalGroup(2, [surface], 1)
gmsh.model.setPhysicalName(2, 1, "Domain")

# Générer le maillage 2D
gmsh.model.mesh.generate(2)

# Sauvegarder le maillage en format Gmsh 4
gmsh.option.setNumber("Mesh.MshFileVersion", 4.1)
gmsh.write("rectangle.msh")

# Visualiser le maillage dans Gmsh
gmsh.fltk.run()

# Finaliser Gmsh
gmsh.finalize()
