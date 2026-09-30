import numpy as np
import math
import ufl


###################################
####  Level set's definition   ####
###################################
def level_set(x):
    r_cercle = 0.7
    d0 = -(ufl.sqrt((x[0]) ** 2 + (x[1]) ** 2) - r_cercle)
    # d0 = ufl.max_value(-(ufl.sqrt((x[0]-1)**2 + (x[1] - 0)**2) - r_cercle),-(ufl.sqrt((x[0]+ 1)**2 + (x[1] - 0)**2) - r_cercle))
    d1 = ufl.max_value(-(ufl.sqrt((x[0] + 1.2) ** 2 + (x[1] - 0) ** 2) - r_cercle), d0)
    d2 = ufl.max_value(d1, -(ufl.sqrt((x[0] - 4) ** 2 + (x[1] + 2) ** 2) - r_cercle))
    d3 = ufl.max_value(-(ufl.sqrt((x[0] + 4) ** 2 + (x[1] - 2) ** 2) - r_cercle), d2)
    d4 = ufl.max_value(d3, -(ufl.sqrt((x[0] + 4) ** 2 + (x[1] + 2) ** 2) - r_cercle))
    # d5 = ufl.max_value(d4,-(ufl.sqrt((x[0])**2 + (x[1] )**2) - r_cercle))
    d6 = ufl.max_value(d4, -(ufl.sqrt((x[0] - 5) ** 2 + (x[1]) ** 2) - r_cercle))
    d7 = ufl.max_value(d6, -(ufl.sqrt((x[0] + 5) ** 2 + (x[1]) ** 2) - r_cercle))
    return d0


def test(x, parameters):

    camera = 1
    f = parameters.lx / 10  # scale factor
    H = parameters.ly  # y dimension of the rectangle
    L = parameters.lx  # x dimension of the rectangle
    r_cercle = f * (math.pi - 0.15)
    if camera == 1:
        c_1 = [L / 2, H + 2.85 * f * 1]
        c_2 = [L / 2, -2.85 * f * 1]
    else:
        c_1 = [L / 2, H + f * 1]
        c_2 = [L / 2, -f * 1]
    # r_cercle = 4.5*f*(math.pi-0.15)
    # c_1 = [L/2,H+6.75*1.5]
    # c_2 = [L/2,-6.75*1.5]
    # 0 trou
    d0 = ufl.max_value(
        -(ufl.sqrt((x[1] - c_2[1]) ** 2 + (x[0] - c_2[0]) ** 2) - r_cercle),
        -(ufl.sqrt((x[1] - c_1[1]) ** 2 + (x[0] - c_1[0]) ** 2) - r_cercle),
    )
    d1 = ufl.max_value(
        d0, -(ufl.sqrt((x[0] - 5) ** 2 + (x[1] - 1) ** 2) - 0.5)
    )  # for 1 hole
    # d1 =   ufl.max_value(d0,-(ufl.sqrt((x[0]- 4.)*2 + (x[1] - 1)*2) - 0.5))
    d2 = ufl.max_value(d1, -(ufl.sqrt((x[0] - 6) ** 2 + (x[1] - 1) ** 2) - 0.5))
    res = -2 * ufl.cos(3.0 / 5 * math.pi * x[0]) * ufl.cos(2.0 * math.pi * x[1]) - 0.6
    camera = ufl.conditional(ufl.le(x[0], 3), -1, res)
    camera = ufl.conditional(ufl.ge(x[0], parameters.lx - 3), -1, camera)
    holes = ufl.max_value(
        -(ufl.sqrt((x[0] - 6) ** 2 + (x[1] - 1) ** 2) - 0.5),
        -(ufl.sqrt((x[0] - 4.0) ** 2 + (x[1] - 1) ** 2) - 0.5),
    )
    # return camera
    return d0


def level_set_DIP(x, parameters):

    camera = 1
    f = parameters.lx / 10  # scale factor
    H = parameters.ly  # y dimension of the rectangle
    L = parameters.lx  # x dimension of the rectangle
    r_cercle = 0.6
    if camera == 1:
        c_1 = [L / 2 + 0.5, H]
        c_2 = [L / 2 - 0.5, 0]
    else:
        c_1 = [L / 2, H + f * 1]
        c_2 = [L / 2, -f * 1]
    # r_cercle = 4.5*f*(math.pi-0.15)
    # c_1 = [L/2,H+6.75*1.5]
    # c_2 = [L/2,-6.75*1.5]
    # 0 trou
    d0 = ufl.max_value(
        -(ufl.sqrt((x[1] - c_2[1]) ** 2 + (x[0] - c_2[0]) ** 2) - r_cercle),
        -(ufl.sqrt((x[1] - c_1[1]) ** 2 + (x[0] - c_1[0]) ** 2) - r_cercle),
    )
    return d0


def level_set_DIP3D(x, parameters):

    camera = 1
    f = parameters.lx / 10  # scale factor
    H = parameters.ly  # y dimension of the rectangle
    L = parameters.lx  # x dimension of the rectangle
    h = parameters.lz  # z dimension of the rectangle
    r_cercle = 0.5
    if camera == 1:
        c_1 = [L / 2 + 0.5, H, h]
        c_2 = [L / 2 - 0.5, 0, 0]
        c_3 = [L / 2, 0, h / 2]
    else:
        c_1 = [L / 2, H + f * 1]
        c_2 = [L / 2, -f * 1]
    # r_cercle = 4.5*f*(math.pi-0.15)
    # c_1 = [L/2,H+6.75*1.5]
    # c_2 = [L/2,-6.75*1.5]
    # 0 trou
    # d0 = ufl.max_value(-(ufl.sqrt((x[1]- c_2[1])**2 + (x[0] - c_2[0])**2 +(x[2] - c_2[2])**2) - r_cercle),-(ufl.sqrt((x[1] - c_1[1])**2 + (x[0] - c_1[0])**2+(x[2] - c_1[2])**2) - r_cercle))
    d0 = -(
        ufl.sqrt((x[1] - c_2[1]) ** 2 + (x[0] - c_2[0]) ** 2 + (x[2] - c_2[2]) ** 2)
        - r_cercle
    )

    return d0


def level_set_DI1P(x, parameters):

    camera = 1
    f = parameters.lx / 10  # scale factor
    H = parameters.ly  # y dimension of the rectangle
    L = parameters.lx  # x dimension of the rectangle
    r_cercle = 0.6
    if camera == 1:
        c_1 = [L / 2 + 0.5, H]
        c_2 = [L / 2 - 0.5, 0]
    else:
        c_1 = [L / 2, H + f * 1]
        c_2 = [L / 2, -f * 1]
    # r_cercle = 4.5*f*(math.pi-0.15)
    # c_1 = [L/2,H+6.75*1.5]
    # c_2 = [L/2,-6.75*1.5]
    # 0 trou
    d0 = ufl.max_value(
        -(ufl.sqrt((x[1] - c_2[1]) ** 2 + (x[0] - c_2[0]) ** 2) - r_cercle),
        -(ufl.sqrt((x[1] - c_1[1]) ** 2 + (x[0] - c_1[0]) ** 2) - r_cercle),
    )
    return d0


def level_set_RI(x):

    f = 1  # scale factor
    H = f * 6  # y dimension of the rectangle
    L = f * 10  # x dimension of the rectangle
    c_1 = [L / 2, H + 1]
    c_2 = [L / 2, -1]
    h = 0.1
    r_cercle = 4.5 * f * (math.pi - 0.15)
    c_1 = [L / 2, H + 12.85]
    c_2 = [L / 2, -12.85]
    # 0 trou
    # r_cercle = 13*f*(math.pi-0.15)
    # c_1 = [L/2,H+38.7*1]
    # c_2 = [L/2,-38.7*1]
    # 0 trou
    d3 = ufl.max_value(
        -(ufl.sqrt((x[1] - c_2[1]) ** 2 + (x[0] - c_2[0]) ** 2) - r_cercle),
        -(ufl.sqrt((x[1] - c_1[1]) ** 2 + (x[0] - c_1[0]) ** 2) - r_cercle),
    )
    d0 = ufl.conditional(ufl.le(x[1], H), -1, 1)
    d1 = ufl.conditional(ufl.le(x[1], 0 - 0.01), 1, d0)
    return d1


def circle(x):
    r_cercle = 0.305
    d0 = -(ufl.sqrt((x[0] - 0.5) ** 2 + (x[1] - 0.5) ** 2) - r_cercle) - 0.051
    return d0


def level_set_DH(x):
    r_cercle = 0.305
    d0 = -(ufl.sqrt((x[0] - 5) ** 2 + (x[1] - 3) ** 2) - r_cercle)
    return d0


def level_set_immersed_dogbone(x):
    # r_cercle = 1.
    # d0 = ufl.max_value(-(ufl.sqrt((x[0]-6.3)**2 + (x[1] - 1.8)**2) - r_cercle),-(ufl.sqrt((x[0]-8.7)**2 + (x[1] - 3.2)**2) - r_cercle))
    # d0 = -(ufl.sqrt((x[0]-7.5)**2 + (x[1] - 2.5)**2) - r_cercle)
    # d1 = ufl.max_value(d0,-(ufl.sqrt((x[0]-4)**2 + (x[1]-2.5)**2) - r_cercle))
    # d2 = ufl.max_value(d1,-(ufl.sqrt((x[0]-10)**2 + (x[1]-2.5)**2) - r_cercle))
    f = 1  # scale factor
    H = f * 5  # y dimension of the rectangle
    L = f * 15  # x dimension of the rectangle
    r_cercle = 3.5 * f * (math.pi - 0.15)
    c_1 = [L / 2, H + 6.75 * 1.5]
    c_2 = [L / 2, -6.75 * 1.5]
    # 0 trou
    d0 = ufl.max_value(
        -(ufl.sqrt((x[1] - c_2[1]) ** 2 + (x[0] - c_2[0]) ** 2) - r_cercle),
        -(ufl.sqrt((x[1] - c_1[1]) ** 2 + (x[0] - c_1[0]) ** 2) - r_cercle),
    )
    # 1 trou
    d3 = ufl.max_value(d0, -(ufl.sqrt((x[0] - 5) ** 2 + (x[1] - 3) ** 2) - 0.7))
    # 2 trous
    d1 = ufl.max_value(d0, -(ufl.sqrt((x[0] - 4) ** 2 + (x[1] - 3) ** 2) - 0.7))
    d2 = ufl.max_value(d1, -(ufl.sqrt((x[0] - 6) ** 2 + (x[1] - 3) ** 2) - 0.7))
    return d0


def level_set_immersed_dogbone_1(x):
    # r_cercle = 1.
    # d0 = ufl.max_value(-(ufl.sqrt((x[0]-6.3)**2 + (x[1] - 1.8)**2) - r_cercle),-(ufl.sqrt((x[0]-8.7)**2 + (x[1] - 3.2)**2) - r_cercle))
    # d0 = -(ufl.sqrt((x[0]-7.5)**2 + (x[1] - 2.5)**2) - r_cercle)
    # d1 = ufl.max_value(d0,-(ufl.sqrt((x[0]-4)**2 + (x[1]-2.5)**2) - r_cercle))
    # d2 = ufl.max_value(d1,-(ufl.sqrt((x[0]-10)**2 + (x[1]-2.5)**2) - r_cercle))
    res = ufl.conditional(ufl.le(x[1], 0.05), -1, 1)
    res = ufl.conditional(ufl.ge(x[1], 4.95), -1, res)
    return -res


def level_set_rectangle(x):
    r_cercle = 0.8
    d0 = ufl.max_value(
        -(ufl.sqrt((x[0] - 6.3) ** 2 + (x[1] - 1.8) ** 2) - r_cercle),
        -(ufl.sqrt((x[0] - 8.7) ** 2 + (x[1] - 3.2) ** 2) - r_cercle),
    )
    # d0 = -(ufl.sqrt((x[0]-7.5)**2 + (x[1] - 2.5)**2) - r_cercle)
    # d1 = ufl.max_value(d0,-(ufl.sqrt((x[0]-4)**2 + (x[1]-2.5)**2) - r_cercle))
    # d2 = ufl.max_value(d1,-(ufl.sqrt((x[0]-10)**2 + (x[1]-2.5)**2) - r_cercle))
    return d0


def level_set_RH(x):
    r_cercle = 1.5
    d0 = -(ufl.sqrt((x[0] - 6) ** 2 + (x[1] - 3) ** 2) - r_cercle)
    # d0 = -(ufl.sqrt((x[0]-7.5)**2 + (x[1] - 2.5)**2) - r_cercle)
    # d1 = ufl.max_value(d0,-(ufl.sqrt((x[0]-4)**2 + (x[1]-2.5)**2) - r_cercle))
    # d2 = ufl.max_value(d1,-(ufl.sqrt((x[0]-10)**2 + (x[1]-2.5)**2) - r_cercle))
    return d0


def level_set_test(x):
    r_cercle = 0.421
    d0 = -(ufl.sqrt((x[0] - 2) ** 2 + (x[1] - 0.5) ** 2) - r_cercle)
    return d0


def omega_dogbone(x):
    return ufl.ge(x[1], 0)


def omega(x, parameters):
    return ufl.ge(x[1], parameters.ly / 2)


def omega_test(x, parameters):
    return ufl.ge(x[1], parameters.ly / 2)


def level_set_immersed_dogbone(x, parameters):

    camera = 1
    f = parameters.lx / 10  # scale factor
    H = parameters.ly  # y dimension of the rectangle
    L = parameters.lx  # x dimension of the rectangle
    r_cercle = f * (math.pi - 0.15)
    if camera == 1:
        c_1 = [L / 2, H + 2.85 * f * 1]
        c_2 = [L / 2, -2.85 * f * 1]
    else:
        c_1 = [L / 2, H + f * 1]
        c_2 = [L / 2, -f * 1]
    # r_cercle = 4.5*f*(math.pi-0.15)
    # c_1 = [L/2,H+6.75*1.5]
    # c_2 = [L/2,-6.75*1.5]
    # 0 trou
    d0 = ufl.max_value(
        -(ufl.sqrt((x[1] - c_2[1]) ** 2 + (x[0] - c_2[0]) ** 2) - r_cercle),
        -(ufl.sqrt((x[1] - c_1[1]) ** 2 + (x[0] - c_1[0]) ** 2) - r_cercle),
    )

    return d0


def level_set_immersed_dogbone_hoolev2(x, parameters):

    camera = 1
    f = parameters.lx / 10  # scale factor
    H = parameters.ly  # y dimension of the rectangle
    L = parameters.lx  # x dimension of the rectangle
    r_cercle = f * (math.pi - 0.15)
    if camera == 1:
        c_1 = [L / 2, H + 2.85 * f * 1]
        c_2 = [L / 2, -2.85 * f * 1]
    else:
        c_1 = [L / 2, H + f * 1]
        c_2 = [L / 2, -f * 1]
    # r_cercle = 4.5*f*(math.pi-0.15)
    # c_1 = [L/2,H+6.75*1.5]
    # c_2 = [L/2,-6.75*1.5]
    # 0 trou
    d0 = ufl.max_value(
        -(ufl.sqrt((x[1] - c_2[1]) ** 2 + (x[0] - c_2[0]) ** 2) - r_cercle),
        -(ufl.sqrt((x[1] - c_1[1]) ** 2 + (x[0] - c_1[0]) ** 2) - r_cercle),
    )
    d1 = ufl.max_value(d0, -(ufl.sqrt((x[0] - 5) ** 2 + (x[1] - 1) ** 2) - 0.25))
    return d1


# def test_dogbone_cv(x):
#     return -(np.logical_and(np.logical_and(x[1]<2,np.logical_and(x[0]>3,x[0]<7)),x[0]<3))
#     # return ufl.conditional(ufl.ge(res,0.5),-0.5,0.5)
#     # res = ufl.conditional(ufl.le(x[0],3),-1,1)
#     # res = ufl.conditional(ufl.ge(x[0],7),-1,res)
#     # res = ufl.conditional(np.logical_and(ufl.le(x[1],2),np.logical_and(ufl.ge(x[0],3),ufl.le(x[0],7))),-1,res)
#     # res = ufl.conditional(np.logical_and(ufl.ge(x[1],1),np.logical_and(ufl.ge(x[0],3),ufl.le(x[0],7))),-1,res)
#     # return res
def level_set_2(x, parameters):
    # res = -ufl.cos(6.0/parameters.lx*math.pi*x[0]) * ufl.cos(4
    # .0*math.pi*x[1]) - 0.6
    res = (
        -ufl.cos(10 / parameters.lx * math.pi * x[0])
        * ufl.cos(6.0 / parameters.ly * math.pi * x[1])
        - 0.6
    )
    return res / 2


def level_set_immersed_dogbone_hoole(x):
    f = 1  # scale factor
    H = f * 6  # y dimension of the rectangle
    L = f * 10  # x dimension of the rectangle
    r_cercle = f * (math.pi - 0.15)
    c_1 = [L / 2, H + 1]
    c_2 = [L / 2, -1]
    # 0 trou
    d0 = ufl.max_value(
        -(ufl.sqrt((x[1] - c_2[1]) ** 2 + (x[0] - c_2[0]) ** 2) - r_cercle),
        -(ufl.sqrt((x[1] - c_1[1]) ** 2 + (x[0] - c_1[0]) ** 2) - r_cercle),
    )
    # 1 trou
    d1 = ufl.max_value(d0, -(ufl.sqrt((x[0] - 5) ** 2 + (x[1] - 3) ** 2) - 0.7))
    return d1


def level_set_DIH(x, parameters):
    camera = 1
    f = parameters.lx / 10  # scale factor
    H = parameters.ly  # y dimension of the rectangle
    L = parameters.lx  # x dimension of the rectangle
    r_cercle = f * (math.pi - 0.15)
    if camera == 1:
        c_1 = [L / 2, H + 2.85 * f * 1]
        c_2 = [L / 2, -2.85 * f * 1]
    else:
        c_1 = [L / 2, H + f * 1]
        c_2 = [L / 2, -f * 1]
    # r_cercle = 4.5*f*(math.pi-0.15)
    # c_1 = [L/2,H+6.75*1.5]
    # c_2 = [L/2,-6.75*1.5]
    # 0 trou
    d0 = ufl.max_value(
        -(ufl.sqrt((x[1] - c_2[1]) ** 2 + (x[0] - c_2[0]) ** 2) - r_cercle),
        -(ufl.sqrt((x[1] - c_1[1]) ** 2 + (x[0] - c_1[0]) ** 2) - r_cercle),
    )
    d1_hole = ufl.max_value(d0, -(ufl.sqrt((x[0] - 5) ** 2 + (x[1] - 1) ** 2) - 0.5))
    d2_holes = ufl.max_value(d0, -(ufl.sqrt((x[0] - 4.0) ** 2 + (x[1] - 1) ** 2) - 0.5))
    d3_holes = ufl.max_value(
        d2_holes, -(ufl.sqrt((x[0] - 6) ** 2 + (x[1] - 1) ** 2) - 0.5)
    )
    res = -2 * ufl.cos(3.0 / 5 * math.pi * x[0]) * ufl.cos(2.0 * math.pi * x[1]) - 0.6
    camera = ufl.conditional(ufl.le(x[0], 3), -1, res)
    camera = ufl.conditional(ufl.ge(x[0], parameters.lx - 3), -1, camera)
    holes = ufl.max_value(
        -(ufl.sqrt((x[0] - 6) ** 2 + (x[1] - 1) ** 2) - 0.5),
        -(ufl.sqrt((x[0] - 4.0) ** 2 + (x[1] - 1) ** 2) - 0.5),
    )
    # return camera
    return d3_holes


def level_set_DI2H(x, parameters):
    camera = 1
    f = parameters.lx / 10  # scale factor
    H = parameters.ly  # y dimension of the rectangle
    L = parameters.lx  # x dimension of the rectangle
    r_cercle = 2.3 * f * (math.pi)
    if camera == 1:
        c_1 = [L / 2, H + 7 * f * 1]
        c_2 = [L / 2, -7 * f * 1]
    else:
        c_1 = [L / 2, H + f * 1]
        c_2 = [L / 2, -f * 1]
    # r_cercle = 4.5*f*(math.pi-0.15)
    # c_1 = [L/2,H+6.75*1.5]
    # c_2 = [L/2,-6.75*1.5]
    # 0 trou
    d0 = ufl.max_value(
        -(ufl.sqrt((x[1] - c_2[1]) ** 2 + (x[0] - c_2[0]) ** 2) - r_cercle),
        -(ufl.sqrt((x[1] - c_1[1]) ** 2 + (x[0] - c_1[0]) ** 2) - r_cercle),
    )
    d1_hole = ufl.max_value(d0, -(ufl.sqrt((x[0] - 5) ** 2 + (x[1] - 1) ** 2) - 0.5))
    d2_holes = ufl.max_value(d0, -(ufl.sqrt((x[0] - 4.0) ** 2 + (x[1] - 1) ** 2) - 0.3))
    d3_holes = ufl.max_value(
        d2_holes, -(ufl.sqrt((x[0] - 6) ** 2 + (x[1] - 1) ** 2) - 0.3)
    )
    res = -2 * ufl.cos(3.0 / 5 * math.pi * x[0]) * ufl.cos(2.0 * math.pi * x[1]) - 0.6
    camera = ufl.conditional(ufl.le(x[0], 3), -1, res)
    camera = ufl.conditional(ufl.ge(x[0], parameters.lx - 3), -1, camera)
    holes = ufl.max_value(
        -(ufl.sqrt((x[0] - 6) ** 2 + (x[1] - 1) ** 2) - 0.5),
        -(ufl.sqrt((x[0] - 4.0) ** 2 + (x[1] - 1) ** 2) - 0.5),
    )
    # return camera
    return d3_holes


def level_set_DI1H(x, parameters):
    camera = 1
    f = parameters.lx / 10  # scale factor
    H = parameters.ly  # y dimension of the rectangle
    L = parameters.lx  # x dimension of the rectangle
    r_cercle = f * (math.pi - 0.15)
    if camera == 1:
        c_1 = [L / 2, H + 2.85 * f * 1]
        c_2 = [L / 2, -2.85 * f * 1]
    else:
        c_1 = [L / 2, H + f * 1]
        c_2 = [L / 2, -f * 1]
    # r_cercle = 4.5*f*(math.pi-0.15)
    # c_1 = [L/2,H+6.75*1.5]
    # c_2 = [L/2,-6.75*1.5]
    # 0 trou
    d0 = ufl.max_value(
        -(ufl.sqrt((x[1] - c_2[1]) ** 2 + (x[0] - c_2[0]) ** 2) - r_cercle),
        -(ufl.sqrt((x[1] - c_1[1]) ** 2 + (x[0] - c_1[0]) ** 2) - r_cercle),
    )
    d1_hole = ufl.max_value(d0, -(ufl.sqrt((x[0] - 5) ** 2 + (x[1] - 1) ** 2) - 0.5))
    d2_holes = ufl.max_value(d0, -(ufl.sqrt((x[0] - 4.0) ** 2 + (x[1] - 1) ** 2) - 0.5))
    d3_holes = ufl.max_value(
        d2_holes, -(ufl.sqrt((x[0] - 6) ** 2 + (x[1] - 1) ** 2) - 0.5)
    )
    res = -2 * ufl.cos(3.0 / 5 * math.pi * x[0]) * ufl.cos(2.0 * math.pi * x[1]) - 0.6
    camera = ufl.conditional(ufl.le(x[0], 3), -1, res)
    camera = ufl.conditional(ufl.ge(x[0], parameters.lx - 3), -1, camera)
    holes = ufl.max_value(
        -(ufl.sqrt((x[0] - 6) ** 2 + (x[1] - 1) ** 2) - 0.5),
        -(ufl.sqrt((x[0] - 4.0) ** 2 + (x[1] - 1) ** 2) - 0.5),
    )
    # return camera
    return d1_hole


def level_set_RH(x):
    r_cercle = 0.501
    d2 = ufl.max_value(
        -(ufl.sqrt((x[0] - 6.0013) ** 2 + (x[1] - 1.0007) ** 2) - r_cercle),
        -(ufl.sqrt((x[0] - 4.0013) ** 2 + (x[1] - 1.0007) ** 2) - r_cercle),
    )
    return d2


def level_set_RHASYM(x):
    d2 = ufl.max_value(
        -(ufl.sqrt((x[0] - 4) ** 2 + (x[1] - 3) ** 2) - 0.7),
        -(ufl.sqrt((x[0] - 6) ** 2 + (x[1] - 3) ** 2) - 0.7),
    )
    return d2


def level_set_3D(x, parameters):
    """
    Initial level set for 3D topology optimization.
    ϕ0(x, y, z) = - cos(2πx) cos(2πy) cos(2πz) - 0.3
    """
    res = (
        -ufl.cos(2.0 * math.pi * x[0])
        * ufl.cos(2.0 * math.pi * x[1])
        * ufl.cos(2.0 * math.pi * x[2])
        - 0.2
    )
    return res + 1e-9
