import numpy as np

def edge_edge_dist(p, a, q, b):
    """
    https://github.com/NVIDIA-Omniverse/PhysX/blob/main/physx/source/geomutils/src/sweep/GuSweepCapsuleCapsule.cpp
    p: origin of edge1, [x, y, z]
    a: vector of edge1 from p, [x, y, z]
    q: origin of edge2, [x, y, z]
    b: vector of edge2 from q, [x, y, z]
    cp: closest point on (p, a), [x, y, z]
    cq: closest point on (q, b), [x, y, z]
    """

    assert type(p) == np.ndarray
    assert type(a) == np.ndarray
    assert type(q) == np.ndarray
    assert type(b) == np.ndarray

    T = q - p
    ADotA = np.dot(a, a)
    BDotB = np.dot(b, b)
    ADotB = np.dot(a, b)
    ADotT = np.dot(a, T)
    BDotT = np.dot(b, T)

    ### t parameterizes ray (p, a)
    ### u parameterizes ray (q, b)

    ### Compute denominator
    Denom = ADotA * BDotB - ADotB * ADotB

    ### Compute t for the closest point on segment (p, a) to segment (q, b)
    if Denom != 0:
        t = np.clip((ADotT * BDotB - BDotT * ADotB) / Denom, 0, 1)
    else:
        t = 0

    ### Find u for point on ray (q, b) closest to point at t
    if BDotB != 0:
        u = (t * ADotB - BDotT) / BDotB

        ### Clamp u if it's out of segment bounds, then recompute t if needed
        if u < 0:
            u = 0
            if ADotA != 0:
                t = np.clip(ADotT / ADotA, 0, 1)
            else:
                t = 0
        elif u > 1:
            u = 1
            if ADotA != 0:
                t = np.clip((ADotB + ADotT) / ADotA, 0, 1)
            else:
                t = 0
    else:
        u = 0
        if ADotA != 0:
            t = np.clip(ADotT / ADotA, 0, 1)
        else:
            t = 0

    ### Compute closest points
    cp = p + a * t
    cq = q + b * u

    return cp, cq


def distance_triangle_triangle(p, q):
    """
    https://github.com/NVIDIA-Omniverse/PhysX/blob/main/physx/source/geomutils/src/distance/GuDistanceTriangleTriangle.cpp
    p: triangle1 vertices, [[x0, y0, z0], [x1, y1, z1], [x2, y2, z2]]
    q: triangle2 vertices, [[x0, y0, z0], [x1, y1, z1], [x2, y2, z2]]
    cp: closest point in triangle1, [x, y, z]
    cq: closest point in triangle2, [x, y, z]
    """

    assert type(p) == np.ndarray
    assert type(q) == np.ndarray

    Sv = [np.subtract(p[1], p[0]), np.subtract(p[2], p[1]), np.subtract(p[0], p[2])] # edges of p
    Tv = [np.subtract(q[1], q[0]), np.subtract(q[2], q[1]), np.subtract(q[0], q[2])] # edges of q

    minP, minQ = np.zeros(3), np.zeros(3)
    shown_disjoint = False
    mindd = float('inf')

    for i in range(3):
        for j in range(3):
            cp, cq = edge_edge_dist(p[i], Sv[i], q[j], Tv[j])
            V = cq - cp
            dd = np.dot(V, V)

            if dd <= mindd:
                minP, minQ = cp.copy(), cq.copy()
                mindd = dd

                id = (i + 2) % 3
                Z = p[id] - cp
                a = np.dot(Z, V)

                id = (j + 2) % 3
                Z = q[id] - cq
                b = np.dot(Z, V)

                # print(a, b)

                if a <= 0.0 and b >= 0.0:
                    # print("return 1")
                    return cp, cq

                if a <= 0.0:
                    a = 0.0
                elif b > 0.0:
                    b = 0.0

                if mindd - a + b > 0.0:
                    shown_disjoint = True

    ### Face normal of p
    Sn = np.cross(Sv[0], Sv[1])
    Snl = np.dot(Sn, Sn)

    if Snl > 1e-15:
        Tp = np.array([np.dot(p[0] - q[0], Sn), np.dot(p[0] - q[1], Sn), np.dot(p[0] - q[2], Sn)])
        index = -1

        if (Tp > 0).all():
            if Tp[0] < Tp[1]:
                index = 0
            else: 
                index = 1
            if Tp[2] < Tp[index]:
                index = 2
        elif (Tp < 0).all():
            if Tp[0] > Tp[1]:
                index = 0
            else: 
                index = 1
            if Tp[2] > Tp[index]:
                index = 2

        if index >= 0:
            shown_disjoint = True
            qIndex = q[index]
            V = qIndex - p[0]
            Z = np.cross(Sn, Sv[0])
            if np.dot(V, Z) > 0.0:
                V = qIndex - p[1]
                Z = np.cross(Sn, Sv[1])
                if np.dot(V, Z) > 0.0:
                    V = qIndex - p[2]
                    Z = np.cross(Sn, Sv[2])
                    if np.dot(V, Z) > 0.0:
                        cp = qIndex + Sn * Tp[index]/Snl
                        cq = qIndex
                        # print("return 2")
                        return cp, cq

    ### Face normal of q
    Tn = np.cross(Tv[0], Tv[1])
    Tnl = np.dot(Tn, Tn)

    if Tnl > 1e-15:
        Sp = np.array([np.dot(q[0] - p[0], Tn), np.dot(q[0] - p[1], Tn), np.dot(q[0] - p[2], Tn)])
        index = -1

        if (Sp > 0).all():
            if(Sp[0]<Sp[1]):
                index = 0 
            else: 
                index = 1
            if Sp[2] < Sp[index]:
                index = 2
            # print("return 3: pos")
        elif (Sp < 0).all():
            if Sp[0] > Sp[1]:
                index = 0
            else: 
                index = 1
            if Sp[2] > Sp[index]:
                index = 2
            # print("return 3: neg")

        if index >= 0:
            shown_disjoint = True
            pIndex = p[index]
            V = pIndex - q[0]
            Z = np.cross(Tn, Tv[0])
            if np.dot(V, Z) > 0.0:
                V = pIndex - q[1]
                Z = np.cross(Tn, Tv[1])
                if np.dot(V, Z) > 0.0:
                    V = pIndex - q[2]
                    Z = np.cross(Tn, Tv[2])
                    if np.dot(V, Z) > 0.0:
                        cp = pIndex
                        cq = pIndex + Tn * Sp[index]/Tnl
                        # print("return 3")
                        return cp, cq

    if shown_disjoint:
        cp, cq = minP, minQ
        # print("return 4")
        return cp, cq
    else:
        # print("return 5")
        return np.array([0.0, 0.0, 0.0]), np.array([0.0, 0.0, 0.0]) # In-collision

