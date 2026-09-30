from dolfinx.io import XDMFFile
import gmsh
from dolfinx.io.gmsh import model_to_mesh
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
    
# Ajouter une entité physique (le domaine principal)
gmsh.model.addPhysicalGroup(2, [surface], 1)
gmsh.model.setPhysicalName(2, 1, "Domain")

# Ajouter les tags physiques pour les frontières
boundaries = gmsh.model.getBoundary([(2, surface)], combined=False, oriented=False, recursive=False)
for i, boundary in enumerate(boundaries):
    gmsh.model.addPhysicalGroup(boundary[0], [boundary[1]], i + 2)

gmsh.option.setNumber("Mesh.SurfaceFaces", 1)
gmsh.option.setNumber("Mesh.VolumeEdges", 0)

# Générer le maillage 2D
gmsh.model.mesh.generate(2)

# Conversion vers FEniCSx
mesh_data = model_to_mesh(gmsh.model, mesh_comm, model_rank, gdim=gdim)
domain = mesh_data.mesh
cell_tags = mesh_data.cell_tags
facet_tags = mesh_data.facet_tags

# Finaliser Gmsh avant d'écrire le XDMF
gmsh.finalize()

# Écrire le fichier XDMF
with XDMFFile(domain.comm, f"{case}.xdmf", "w") as xdmf:
    xdmf.write_mesh(domain)
    xdmf.write_meshtags(cell_tags, domain.geometry)
    xdmf.write_meshtags(facet_tags, domain.geometry)

print(f"Maillage exporté avec succès : {case}.xdmf")
