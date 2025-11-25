from dolfinx.io import XDMFFile
import gmsh
from dolfinx.io.gmshio import model_to_mesh
from mpi4py import MPI
import math
mesh_comm = MPI.COMM_WORLD
gdim = 2
model_rank = 0

gmsh.initialize()

case = "dogbone" # "cruciform_Chen2022_simplify", "dogbone_1holl", "cruciform_Chen2022", "dobgone", "dogbone_immersed","carre"
if case == "cruciform_Chen2022_simplify":
    l = 20/70
    L = 50/70
    H = 70/70
    # cube points:
    lc = 0.01
    H_t = H - 2*lc
    point1 = gmsh.model.occ.add_point(0, 0, 0, lc,1)
    point2 = gmsh.model.occ.add_point(H, 0, 0, lc,2)
    point3 = gmsh.model.occ.add_point(H, H, 0, lc,3)
    point4 = gmsh.model.occ.add_point(L, H, 0, lc,4)
    point5 = gmsh.model.occ.add_point(L, H_t, 0, lc,5)
    point6 = gmsh.model.occ.add_point(2*lc, l, 0, lc,6)
    point7 = gmsh.model.occ.add_point(0, l, 0, lc,7)

    lines = []
    lines.append(gmsh.model.occ.addLine(1, 2, 1))
    lines.append(gmsh.model.occ.addLine(2, 3, 2))
    lines.append(gmsh.model.occ.addLine(3, 4, 3))
    lines.append(gmsh.model.occ.addLine(4, 5, 4))
    lines.append(gmsh.model.occ.addLine(5, 6, 5))
    lines.append(gmsh.model.occ.addLine(6, 7, 6))
    lines.append(gmsh.model.occ.addLine(7, 1, 7))
    loop = gmsh.model.occ.addCurveLoop(lines, 8)
    surface = gmsh.model.occ.addPlaneSurface([8], 9)
elif case == "carre":
    N = 120
    lc = 1/N
    point1 = gmsh.model.occ.add_point(0, 0, 0, lc,1)
    point2 = gmsh.model.occ.add_point(1, 0, 0, lc,2)
    point3 = gmsh.model.occ.add_point(1, 1, 0, lc,3)
    point4 = gmsh.model.occ.add_point(0, 1, 0, lc,4)
    lines = []
    lines.append(gmsh.model.occ.addLine(1, 2, 1))
    lines.append(gmsh.model.occ.addLine(2, 3, 2))
    lines.append(gmsh.model.occ.addLine(3, 4, 3))
    lines.append(gmsh.model.occ.addLine(4, 1, 4))

    loop = gmsh.model.occ.addCurveLoop(lines, 8)
    surface = gmsh.model.occ.addPlaneSurface([8], 9)
elif case == "cruciform_Chen2022":
    l = 20/70
    L = 50/70
    H = 70/70
    r = L/10
    # cube points:
    lc = 0.01
    H_t = H - 2*lc
    point1 = gmsh.model.occ.add_point(0, 0, 0, lc,1)
    point2 = gmsh.model.occ.add_point(H, 0, 0, lc,2)
    point3 = gmsh.model.occ.add_point(H, H, 0, lc,3)
    point4 = gmsh.model.occ.add_point(L, H, 0, lc,4)
    point5 = gmsh.model.occ.add_point(L, H-(L-L/10), 0, lc,5)
    point6 = gmsh.model.occ.add_point(L-L/10, l, 0, lc,6)
    middlepoint = gmsh.model.occ.add_point(L-L/10, H-(L-L/10), 0, lc,7)

    point8 = gmsh.model.occ.add_point(0, l, 0, lc,)

    lines = []
    lines.append(gmsh.model.occ.addLine(1, 2, 1))
    lines.append(gmsh.model.occ.addLine(2, 3, 2))
    lines.append(gmsh.model.occ.addLine(3, 4, 3))
    lines.append(gmsh.model.occ.addLine(4, 5, 4))
    lines.append(gmsh.model.occ.addCircleArc(5,7,6,center=True))
    lines.append(gmsh.model.occ.addLine(6, 8, 6))
    lines.append(gmsh.model.occ.addLine(8, 1, 7))
    loop = gmsh.model.occ.addCurveLoop(lines, 8)
    surface = gmsh.model.occ.addPlaneSurface([8], 9)

elif case == "dogbone_1holl":
    N = 60
    lc = 0.0625
    w = 6 # width
    l = 10 # length
    r = 0.5 # radius
    tag_circle = 5
    point1 = gmsh.model.occ.add_point(0, 0, 0, lc,1)
    point2 = gmsh.model.occ.add_point(l, 0, 0, lc,2)
    point3 = gmsh.model.occ.add_point(l, w, 0, lc,3)
    point4 = gmsh.model.occ.add_point(0, w, 0, lc,4)
    # creation of the central hole
    circle = gmsh.model.occ.addCircle(l/2,w/2,0,r,10)
    circle_holl = gmsh.model.occ.addCurveLoop([10],10)

    # creation of the 2 circular surface to define the dogbone geometry:
    c_1 = [l/2,w+2]
    c_2 = [l/2,-2]
    point5 = gmsh.model.occ.add_point(1.5, w, 0, lc,11)
    point6 = gmsh.model.occ.add_point(8.5, w, 0, lc,12)
    middlepoint_1 = gmsh.model.occ.add_point(c_1[0], c_1[1], 0, lc,7)
    semicircle1 = gmsh.model.occ.addCircleArc(11,7,12,center=True)


    point8 = gmsh.model.occ.add_point(1.5, 0, 0, lc,8)
    point9 = gmsh.model.occ.add_point(8.5, 0, 0, lc,9)
    middlepoint_2 = gmsh.model.occ.add_point(c_2[0], c_2[1], 0, lc,10)
    semicircle2 = gmsh.model.occ.addCircleArc(8,10,9,center=True)
    
    lines = []
    lines.append(gmsh.model.occ.addLine(1, 8, 1))
    lines.append(gmsh.model.occ.addCircleArc(8,10,9,center=True))
    lines.append(gmsh.model.occ.addLine(9, 2, 3))
    lines.append(gmsh.model.occ.addLine(2, 3, 4))
    lines.append(gmsh.model.occ.addLine(3, 12, 5))
    lines.append(gmsh.model.occ.addCircleArc(12,7,11,center=True))
    lines.append(gmsh.model.occ.addLine(11, 4, 7))
    lines.append(gmsh.model.occ.addLine(4, 1, 8))

    loop = gmsh.model.occ.addCurveLoop(lines, 13)

    #plane_surface = gmsh.model.geo.addPlaneSurface([8,10])
    surface = gmsh.model.occ.addPlaneSurface([13,10], 9)
elif case == "dogbone":
    N = 60
    lc = 0.5
    w = 6 # width
    l = 10 # length
    r = 5 # radius
    tag_circle = 5
    point1 = gmsh.model.occ.add_point(0, 0, 0, lc,1)
    point2 = gmsh.model.occ.add_point(l, 0, 0, lc,2)
    point3 = gmsh.model.occ.add_point(l, w, 0, lc,3)
    point4 = gmsh.model.occ.add_point(0, w, 0, lc,4)
    # creation of the central hole
    circle = gmsh.model.occ.addCircle(l/2,w/2,0,r,10)
    circle_holl = gmsh.model.occ.addCurveLoop([10],10)

    # creation of the 2 circular surface to define the dogbone geometry:
    c_1 = [l/2,w+2]
    c_2 = [l/2,-2]
    point5 = gmsh.model.occ.add_point(1.5, w, 0, lc,11)
    point6 = gmsh.model.occ.add_point(8.5, w, 0, lc,12)
    middlepoint_1 = gmsh.model.occ.add_point(c_1[0], c_1[1], 0, lc,7)
    semicircle1 = gmsh.model.occ.addCircleArc(11,7,12,center=True)


    point8 = gmsh.model.occ.add_point(1.5, 0, 0, lc,8)
    point9 = gmsh.model.occ.add_point(8.5, 0, 0, lc,9)
    middlepoint_2 = gmsh.model.occ.add_point(c_2[0], c_2[1], 0, lc,10)
    semicircle2 = gmsh.model.occ.addCircleArc(8,10,9,center=True)
    
    lines = []
    lines.append(gmsh.model.occ.addLine(1, 8, 1))
    lines.append(gmsh.model.occ.addCircleArc(8,10,9,center=True))
    lines.append(gmsh.model.occ.addLine(9, 2, 3))
    lines.append(gmsh.model.occ.addLine(2, 3, 4))
    lines.append(gmsh.model.occ.addLine(3, 12, 5))
    lines.append(gmsh.model.occ.addCircleArc(12,7,11,center=True))
    lines.append(gmsh.model.occ.addLine(11, 4, 7))
    lines.append(gmsh.model.occ.addLine(4, 1, 8))

    loop = gmsh.model.occ.addCurveLoop(lines, 13)

    #plane_surface = gmsh.model.geo.addPlaneSurface([8,10])
    surface = gmsh.model.occ.addPlaneSurface([13], 9)
elif case == "dogbone_topo":
    N = 100
    lc = 0.01
    w = 1 # width
    l = 2 # length
    r = 0.3 # radius
    tag_circle = 5
    point1 = gmsh.model.occ.add_point(0, 0, 0, lc,1)
    point2 = gmsh.model.occ.add_point(l, 0, 0, lc,2)
    point3 = gmsh.model.occ.add_point(l, w, 0, lc,3)
    point4 = gmsh.model.occ.add_point(0, w, 0, lc,4)
    # creation of the central hole
    circle = gmsh.model.occ.addCircle(l/2,w/2,0,r,10)
    circle_holl = gmsh.model.occ.addCurveLoop([10],10)

    # creation of the 2 circular surface to define the dogbone geometry:
    c_1 = [l/2,w+2]
    c_2 = [l/2,-2]
    point5 = gmsh.model.occ.add_point(1.5, w, 0, lc,11)
    point6 = gmsh.model.occ.add_point(8.5, w, 0, lc,12)
    middlepoint_1 = gmsh.model.occ.add_point(c_1[0], c_1[1], 0, lc,7)
    semicircle1 = gmsh.model.occ.addCircleArc(11,7,12,center=True)


    point8 = gmsh.model.occ.add_point(1.5, 0, 0, lc,8)
    point9 = gmsh.model.occ.add_point(8.5, 0, 0, lc,9)
    middlepoint_2 = gmsh.model.occ.add_point(c_2[0], c_2[1], 0, lc,10)
    semicircle2 = gmsh.model.occ.addCircleArc(8,10,9,center=True)
    
    lines = []
    lines.append(gmsh.model.occ.addLine(1, 8, 1))
    lines.append(gmsh.model.occ.addCircleArc(8,10,9,center=True))
    lines.append(gmsh.model.occ.addLine(9, 2, 3))
    lines.append(gmsh.model.occ.addLine(2, 3, 4))
    lines.append(gmsh.model.occ.addLine(3, 12, 5))
    lines.append(gmsh.model.occ.addCircleArc(12,7,11,center=True))
    lines.append(gmsh.model.occ.addLine(11, 4, 7))
    lines.append(gmsh.model.occ.addLine(4, 1, 8))

    loop = gmsh.model.occ.addCurveLoop(lines, 13)

    #plane_surface = gmsh.model.geo.addPlaneSurface([8,10])
    surface = gmsh.model.occ.addPlaneSurface([13], 9)
elif case == "dogbone_immersed":
    N = 60
    lc = 0.03125
    w = 6 # width
    l = 10 # length
    h = 1
    r = math.sqrt((5-h)**2+6**2) # radius
    tag_circle = 5
    point1 = gmsh.model.occ.add_point(0, 0, 0, lc,1)
    point2 = gmsh.model.occ.add_point(l, 0, 0, lc,2)
    point3 = gmsh.model.occ.add_point(l, w, 0, lc,3)
    point4 = gmsh.model.occ.add_point(0, w, 0, lc,4)
    
    point5 = gmsh.model.occ.add_point(0+h, 0, 0, lc,5)
    point6 = gmsh.model.occ.add_point(l-h, 0, 0, lc,6)
    point7 = gmsh.model.occ.add_point(l-h, w, 0, lc,7)
    point8 = gmsh.model.occ.add_point(0+h, w, 0, lc,8)
    

    # creation of the 2 circular surface to define the dogbone geometry:
    c_1 = [l/2,w+2]
    c_2 = [l/2,-2]
    middlepoint_1 = gmsh.model.occ.add_point(5,0, 0, lc,9)
    semicircle1 = gmsh.model.occ.addCircleArc(7,9,8,center=True)


    middlepoint_2 = gmsh.model.occ.add_point(5,w, 0, lc,10)
    semicircle2 = gmsh.model.occ.addCircleArc(5,10,6,center=True)
    
    lines = []
    lines.append(gmsh.model.occ.addLine(1, 5, 11))
    lines.append(gmsh.model.occ.addCircleArc(5,10,6,center=True))
    lines.append(gmsh.model.occ.addLine(6, 2, 20))
    lines.append(gmsh.model.occ.addLine(2, 3, 13))
    lines.append(gmsh.model.occ.addLine(3, 7, 14))
    lines.append(gmsh.model.occ.addCircleArc(7,9,8,center=True))
    lines.append(gmsh.model.occ.addLine(8, 4, 16))
    lines.append(gmsh.model.occ.addLine(4, 1, 17))

    loop = gmsh.model.occ.addCurveLoop(lines, 18)

    #plane_surface = gmsh.model.geo.addPlaneSurface([8,10])
    surface = gmsh.model.occ.addPlaneSurface([18], 19)

elif case == "circle":
    N = 640
    lc = 0.00625
    w = 6 # width
    l = 10 # length
    r_cirlce_1 = 0.25+0.1 # radius
    r_cirlce_2 = 0.25-0.1 
    tag_circle = 5
    # creation of the central hole
    circle_1 = gmsh.model.occ.addCircle(0.5,0.5,0,r_cirlce_1,1)
    circle_1 = gmsh.model.occ.addCurveLoop([1],1)

    # creation of the 2 circular surface to define the dogbone geometry:
    circle_2 = gmsh.model.occ.addCircle(0.5, 0.5, 0,r_cirlce_2,2)
    circle_2 = gmsh.model.occ.addCurveLoop([2],2)

    lines = []
    
    # lines.append(gmsh.model.occ.addCircleArc(1,2,3,center=True))
    # loop = gmsh.model.occ.addCurveLoop(lines, 4)
    #plane_surface = gmsh.model.geo.addPlaneSurface([8,10])
    surface = gmsh.model.occ.addPlaneSurface([1,2], 3)


    
# Finally, specify a global mesh size and mesh the imported model
gmsh.option.setNumber("Mesh.MeshSizeMin", lc)
gmsh.option.setNumber("Mesh.MeshSizeMax", lc)

gmsh.model.occ.synchronize()

# Add physical tag for bulk
gmsh.model.addPhysicalGroup(2, [surface], 1)
gmsh.model.setPhysicalName(2, 1, "surface")

# Add physical tag for boundaries
boundary = gmsh.model.getBoundary(
    [(2, surface)], combined=False, oriented=False, recursive=False)
for i, (boundary) in enumerate(boundary):
    gmsh.model.addPhysicalGroup(boundary[0], [boundary[1]], i)
gmsh.option.setNumber("Mesh.SurfaceFaces", 1)
gmsh.option.setNumber("Mesh.VolumeEdges", 0)
gmsh.model.mesh.generate(gdim)
domain, cell_tags, facet_tags = model_to_mesh(
    gmsh.model, mesh_comm, model_rank, gdim=gdim)
gmsh.finalize()

with XDMFFile(domain.comm, str(case) + "_h" + str(lc) + ".xdmf", "w") as xdmf:
    xdmf.write_mesh(domain)
    xdmf.write_meshtags(cell_tags, domain.geometry)
    xdmf.write_meshtags(facet_tags, domain.geometry)

