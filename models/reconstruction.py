import numpy as np
from scipy.optimize import fsolve, minimize


def CommonRegion(walls, pwalls, width, ratio=0.25, band=10, threshold=10):
    '''
    define the common region of two neighborhood plane
    :param walls: numpy array [[x y x y score cls],...]
    :return: [[lc, rc, lb, rb], ...], the lc, rc is the left/right boundary of potential intersection region. The lb, rb is the detection boxes boundary.
    '''
    # sort wall plane by x coordinate
    xyxy = walls[:, :4]
    centerx = np.mean(xyxy[:, [0, 2]], axis=1)
    centery = np.mean(xyxy[:, [1, 3]], axis=1)
    index = np.argsort(centerx)
    plane = walls[index]
    param = pwalls[index]

    # common region
    common = []
    num = len(plane)
    for i in range(num-1):
        p0 = plane[i]
        p1 = plane[i+1]
        w0 = p0[2] - p0[0]
        w1 = p1[2] - p1[0]
        if p0[2] < p1[0]:  # two boxes have no intersection
            dis = p1[0] - p0[2]
            n0 = param[i][:3]
            n1 = param[i+1][:3]
            s = 1 + np.sum(n0 * n1)
            if dis > threshold and s < 0.003:  # the distance is larger than threshold and is almost parallel (180'), their intersection is infinite.
                common.append([1e5, 1e5, p0[2], p1[0]])
            else:
                left = max(p0[2] - min(ratio * w0, band), 0)
                right = min(p1[0] + min(ratio * w1, band), width)
                common.append([left, right, p0[2], p1[0]])
        else:  # two boxes have intersection
            left = max(p1[0] - min(ratio * w1, band), p0[0]+w0/2, 0)
            right = min(p0[2] + min(ratio * w0, band), p1[2]-w1/2, width)
            common.append([left, right, p1[0], p0[2]])
    common = np.array(common)

    return common, param


def FilterLine(common, lines, height=512):
    if len(common) == 0 or len(lines) == 0:
        return lines
    else:
        vaild = np.zeros(len(lines))
        for c in common:
            bound = np.array(
                [[c[0], 0], [c[1], 0], [c[0], height], [c[1], height]])  # 4*2
            bound = np.reshape(bound, (4, 1, 2))
            offset = bound[:, :, 0] - lines[:, 0] * \
                bound[:, :, 1] - lines[:, 1]  # 4 * N
            maxv = np.max(offset, axis=0)
            minv = np.min(offset, axis=0)
            vaild[maxv*minv < 0] = 1
        return lines[vaild == 1]


def PreProcess(planes, params, lines, threshold=(0.5, 0.1, 0.1, 0.5)):
    '''
    only process one img per time
    :param planes: a list [[x y x y score cls], ...]
    :param params: a list [[*n d], ...]
    :param lines: a list [[m b score], ...]
    :param threshold: the threshold for wall floor ceiling line
    :return:
    '''
    planes = np.array(planes)
    params = np.array(params)
    lines = np.array(lines)
    
    # select valid detection after nms output
    params = params[planes[:, -1] == 1]
    planes = planes[planes[:, -1] == 1, :-1]
    lines = lines[lines[:, -1] == 1, :-1]
    # split category wall floor ceiling
    walls = planes[planes[:, 5] == 0]
    floor = planes[planes[:, 5] == 1]
    ceiling = planes[planes[:, 5] == 2]
    # split plane params into wall/floor/ceiling param
    pwalls = params[planes[:, 5] == 0]
    pfloor = params[planes[:, 5] == 1]
    pceiling = params[planes[:, 5] == 2]
    # select highest output, at least one plane should be in an image
    hparam = params[planes[:, 4] == np.max(planes[:, 4])][0]
    hplane = planes[planes[:, 4] == np.max(planes[:, 4])][0]
    # select higher score output than threshold
    pwalls = pwalls[walls[:, 4] > threshold[0]]
    pfloor = pfloor[floor[:, 4] > threshold[1]]
    pceiling = pceiling[ceiling[:, 4] > threshold[2]]
    walls = walls[walls[:, 4] > threshold[0]]
    floor = floor[floor[:, 4] > threshold[1]]
    ceiling = ceiling[ceiling[:, 4] > threshold[2]]
    lines = lines[lines[:, 2] > threshold[3]]
    # supposed only one floor and ceiling and floor and ceiling cann't intersection
    if len(floor) > 1:  # at most one floor
        pfloor = pfloor[floor[:, 4] == np.max(floor[:, 4])]
    if len(ceiling) > 1:  # at most one ceiling
        pceiling = pceiling[ceiling[:, 4] == np.max(ceiling[:, 4])]
    if len(pfloor) + len(pceiling) == 2 and len(pwalls) == 0:  # if there are both floor and ceiling, and no walls exist. we select higher score plane for simplify. 
        pfloor = [] if np.max(floor[:, 4]) < np.max(
            ceiling[:, 4]) else pfloor
        pceiling = [] if np.max(floor[:, 4]) >= np.max(
            ceiling[:, 4]) else pceiling
    if len(pfloor) + len(pceiling) + len(pwalls) == 0:  # at least one plane
        if hplane[5] == 0:
            pwalls = np.array([hparam])
            walls = np.array([hplane])
        elif hplane[5] == 1:
            pfloor = np.array([hparam])
        else:
            pceiling = np.array([hparam])

    return walls, pwalls, pfloor, pceiling, lines


def MergeNeighborWithSameParam(walls, pwalls):
    # sort wall plane by x coordinate of center
    xyxy = walls[:, :4]
    centerx = np.mean(xyxy[:, [0, 2]], axis=1)
    centery = np.mean(xyxy[:, [1, 3]], axis=1)
    index = np.argsort(centerx)
    walls = walls[index]
    pwalls = pwalls[index]
    num = len(index)
    # merge neighborhood walls with similar params
    if num <= 1:
        return walls, pwalls
    else:
        valid = np.ones((num,))
        start = -1
        merge = []
        for i in range(num):
            if i < num-1:
                p1 = pwalls[i]
                p2 = pwalls[i+1]
                s1 = np.sum(p1[:3] * p2[:3])  # angle by inner product
                s2 = np.abs(p1[3] - p2[3])  # offset
                # neighbor walls that their angle is less than 5' and offset is less than 0.1
                if (1 - s1) < 0.003 and s2 < 0.1:
                    if start == -1:
                        start = i
                    merge.append(i+1)
                    continue
            if start != -1:
                merge = np.array(merge)
                valid[merge] = 0
                mpwalls = np.mean(pwalls[[start, *merge]], axis=0)
                mpwalls = mpwalls / np.linalg.norm(mpwalls[:3], ord=2)
                pwalls[start] = mpwalls
                mwalls = walls[[start, *merge]]
                xyxy = [min(mwalls[:, 0]), min(mwalls[:, 1]),
                        max(mwalls[:, 2]), max(mwalls[:, 3])]
                walls[start, :4] = np.array(xyxy)
            start = -1
            merge = []
    walls = walls[valid == 1]
    pwalls = pwalls[valid == 1]
    return walls, pwalls


def OptimizerLayout(pwalls, lines, common, K, size, downsample=8, opt=True):
    num = len(pwalls)
    h = size[0] / downsample
    K_inv = np.linalg.inv(K)
    # intersection boundary
    case = []
    dtls = []
    _ = pwalls
    if num > 1:
        for i in range(num-1):
            p0 = pwalls[i]
            p1 = pwalls[i+1]
            c = common[i]
            l = CalculateInterSectionLine(
                p0, p1, K_inv=K_inv, downsample=downsample)  # [m, b]

            # whether intersection line locates in common region
            bound = np.array(
                [[c[0], 0], [c[1], 0], [c[0], h], [c[1], h]])  # 4*2
            offset = bound[:, 0] - l[0] * bound[:, 1] - l[1]  # 4
            maxv = np.max(offset, axis=0)
            minv = np.min(offset, axis=0)
            inside = maxv * minv
            # whether common region exists detection lines
            bound = np.reshape(bound, (4, 1, 2))
            offset = bound[:, :, 0] - lines[:, 0] * \
                bound[:, :, 1] - lines[:, 1]  # 4 * N
            maxv = np.max(offset, axis=0)
            minv = np.min(offset, axis=0)
            inlines = lines[maxv * minv < 0]

            if inside < 0:
                # calculated intersection line exists in common region
                if len(inlines) > 0:
                    # detection line exists and plane intersection
                    dtl = inlines[inlines[:, 2] == np.max(inlines[:, 2])][0]
                    case.append(0)
                    pass
                else:
                    # intersection line is loss
                    dtl = np.array([*l, 1])
                    case.append(1)
                    pass
            else:
                # intersection line not exist in common region
                if len(inlines) > 0:
                    # an occulision line exists
                    dtl = inlines[inlines[:, 2] == np.max(inlines[:, 2])][0]
                    case.append(2)
                else:
                    # others case fail detection
                    if c[0] == 1e5:
                        dtl = np.array([0, c[2], c[3]])
                        case.append(3)
                    else:
                        dtl = np.array([0, (c[2]+c[3])/2, 1])
                        case.append(4)
            dtls.append(dtl)
        # optimize plane params
        if opt:
            pwalls = OptimizerParams(pwalls, case, dtls, K_inv, downsample)
    return _, pwalls, case, dtls


def OptimizerParams(pwalls, case, dtls, K_inv, downsample):
    num = len(pwalls)
    pwalls = np.copy(pwalls)
    # only optimize related variable
    variable = []
    index = []
    oriindex = []
    for i in range(num-1):
        if case[i] == 0:
            if i == 0:
                variable.append(pwalls[i])
                variable.append(pwalls[i+1])
                index.append(0)
                index.append(1)
                oriindex.append(0)
                oriindex.append(1)
            else:
                if case[i-1] == 0:
                    index.append(len(variable)-1)
                    variable.append(pwalls[i+1])
                    index.append(len(variable)-1)
                    oriindex.append(i+1)
                else:
                    variable.append(pwalls[i])
                    index.append(len(variable)-1)
                    variable.append(pwalls[i+1])
                    index.append(len(variable)-1)
                    oriindex.append(i)
                    oriindex.append(i+1)
    optnum = len(variable)
    index = np.array(index).reshape([-1, 2]).astype(np.int32)
    oriindex = np.array(oriindex).astype(np.int32)
    variable = np.array(variable).reshape(-1)
    meta = [1, 1, 1, 1, 0.01, 0.01, 0, 0, downsample]

    def func(x):
        w, downsample = meta[:-1], meta[-1]
        sum = 0
        j = -1
        for i in range(num-1):
            if case[i] == 0:
                j += 1
                p0 = pwalls[i]
                p1 = pwalls[i+1]
                dtl = dtls[i][:-1]  # x=my+b

                j1 = index[j][0]
                j2 = index[j][1]
                var0 = x[4*j1: 4*j1+4]
                var1 = x[4*j2: 4*j2+4]
                pal = CalculateInterSectionLineOptimize(
                    var0, var1, K_inv, downsample)  # ax+by+cz=0
                e0 = (dtl + pal[1:]/pal[0])**2

                e1 = np.sum((var0[:3] - p0[:3]) ** 2)
                e2 = np.sum((var1[:3] - p1[:3]) ** 2)
                e3 = np.sum((var0[3] - p0[3]) ** 2)  # div 1000 for s3d
                e4 = np.sum((var1[3] - p1[3]) ** 2)  # div 1000 for s3d

                e5 = (min(1e-5, var0[3]))**2
                e6 = (min(1e-5, var1[3]))**2

                sum += e0[0] * w[0] + e0[1] * w[1] + e1 * w[2] + e2 * \
                    w[3] + e3 * w[4] + e4 * w[5] + e5 * w[6] + e6 * w[7]
        return sum
    if optnum != 0:
        res = minimize(func, variable)
        res = res.x.reshape([optnum, -1])
        res = res[:, :3] * res[:, 3:]
        d = np.linalg.norm(res, ord=2, axis=1, keepdims=True)
        normal = res / d
        res = np.concatenate([normal, d], axis=1)
        pwalls[oriindex] = res
    return pwalls


def CalculateInterSectionLine(p0, p1, K_inv, downsample=8):
    # p0, p1 nx+d=0
    line = np.dot((p0[:3] / p0[3] - p1[:3] / p1[3]), K_inv)
    line = -1 * line[1:] / line[0]
    # downsample to output size
    line[1] /= downsample
    return line


def CalculateInterSectionLineOptimize(p0, p1, K_inv, downsample=8):
    # p0, p1 nx+d=0
    line = np.dot((p0[:3] * p1[3] - p1[:3] * p0[3]), K_inv)  # 3
    # downsample to output size
    M = np.array([[downsample, 0, 0], [0, downsample, 0], [0, 0, 1]])
    line = np.dot(line, M)  # ax + by + c=0
    return line


def CalculateIntersectionPoint(p0, p1, p2, K, UD=0, downsample=2, size=(1024, 1280)):
    # nx+d=0 plane params is for original resolution
    if len(p2) == 0:  # not exist floor or ceiling
        K_inv = np.linalg.inv(K)
        if UD == 0:  # floor
            fake_line = np.array([[0, size[0]-1], [size[1]-1, size[0]-1]])
            p2 = CalculateFakePlane(fake_line, K_inv)
        else:  # ceiling
            fake_line = np.array([[0, 0], [size[1]-1, 0]])
            p2 = CalculateFakePlane(fake_line, K_inv)
    else:
        p2 = p2[0]
    coefficient = np.array([p0, p1, p2])
    A = coefficient[:, :3]
    B = -1 * coefficient[:, 3]
    res = np.linalg.solve(A, B)
    # project 3d to 2d
    point_3d = res.reshape(3, 1)
    point_2d = np.dot(K, point_3d) / point_3d[2, 0]
    point_2d = point_2d[:2, 0] / downsample
    return point_2d


def CalculateFakePlane(line, K_inv):
    def func(variable, ray):
        x, y, z = variable
        a = ray[0, 0] * x + ray[1, 0] * y + ray[2, 0] * z
        b = ray[0, 1] * x + ray[1, 1] * y + ray[2, 1] * z
        c = x * x + y * y + z * z - 1
        return [a, b, c]
    ones = np.ones([2, 1])
    point = np.concatenate([line, ones], axis=1).T
    ray = np.dot(K_inv, point)  # 3*2
    result = fsolve(func, np.array([0, 0, 0]), args=(ray))
    result = [*result, 0]
    return result


def GenerateLayout(pwalls, case, dtls, pfloor, pceiling, K, size, downsample=2, upsample=8):
    '''
    :param pwalls, pfloor, pceiling: plane params
    :param case: neighbor walls relationship
    :param dtls: detection line and virtual line defined by box boundary
    :param upsample: the downsample ratio output size to original image
    :param downsample: the downsample ratio input size to original image
    '''
    ups = []
    downs = []
    attribution = 0
    num = len(pwalls)
    K_inv = np.linalg.inv(K)
    param_layout = []
    for i in range(num-1):
        p0 = pwalls[i]
        p1 = pwalls[i+1]
        if case[i] == 0 or case[i] == 1:  # two walls intersect
            point0 = CalculateIntersectionPoint(
                p0, p1, pfloor, K, 0, downsample=downsample, size=size)
            point1 = CalculateIntersectionPoint(
                p0, p1, pceiling, K, 1, downsample=downsample, size=size)
            downs.append(point0)
            ups.append(point1)
            param_layout.append(p0)
            pass
        elif case[i] == 2 or case[i] == 4:  # one occlusion line exist
            dtl = dtls[i]  # x=my+b
            fake_line = np.array([[dtl[1], 0], [dtl[0]+dtl[1], 1]]) * upsample
            fake_plane = CalculateFakePlane(fake_line, K_inv)
            point0 = CalculateIntersectionPoint(
                p0, fake_plane, pfloor, K, 0, downsample=downsample, size=size)
            point1 = CalculateIntersectionPoint(
                p0, fake_plane, pceiling, K, 1, downsample=downsample, size=size)
            point2 = CalculateIntersectionPoint(
                fake_plane, p1, pfloor, K, 0, downsample=downsample, size=size)
            point3 = CalculateIntersectionPoint(
                fake_plane, p1, pceiling, K, 1, downsample=downsample, size=size)
            downs.append(point0)
            downs.append(point2)
            ups.append(point1)
            ups.append(point3)
            param_layout.append(p0)
            param_layout.append(None)
            pass
        else:  # two occlusion line exist, the case maybe solve by add an infinite plane, TODO
            dtl = dtls[i]
            dtl0 = [dtl[0], dtl[1], 1]
            dtl1 = [dtl[0], dtl[2], 1]
            fake_line0 = np.array(
                [[dtl0[1], 0], [dtl0[0] + dtl0[1], 1]]) * upsample
            fake_line1 = np.array(
                [[dtl1[1], 0], [dtl1[0] + dtl1[1], 1]]) * upsample
            fake_plane0 = CalculateFakePlane(fake_line0, K_inv)
            fake_plane1 = CalculateFakePlane(fake_line1, K_inv)

            point0 = CalculateIntersectionPoint(
                p0, fake_plane0, pfloor, K, 0, downsample=downsample, size=size)
            point1 = CalculateIntersectionPoint(
                p0, fake_plane0, pceiling, K, 1, downsample=downsample, size=size)
            point2 = CalculateIntersectionPoint(
                fake_plane1, p1, pfloor, K, 0, downsample=downsample, size=size)
            point3 = CalculateIntersectionPoint(
                fake_plane1, p1, pceiling, K, 1, downsample=downsample, size=size)
            downs.append(point0)
            downs.append(point2)
            ups.append(point1)
            ups.append(point3)
            param_layout.append(p0)
            param_layout.append(None)
            pass
    if num > 0:  # determine the left and right boundary with image boundary.
        if num == 1:
            # left boundary
            fake_line = np.array([[0, 0], [0, size[0]]])
            fake_plane = CalculateFakePlane(fake_line, K_inv)
            point0 = CalculateIntersectionPoint(
                fake_plane, pwalls[0], pfloor, K, 0, downsample=downsample, size=size)
            point1 = CalculateIntersectionPoint(
                fake_plane, pwalls[0], pceiling, K, 1, downsample=downsample, size=size)
            downs = [point0, *downs]
            ups = [point1, *ups]
            # right boundary
            fake_line = np.array([[size[1], 0], [size[1], size[0]]])
            fake_plane = CalculateFakePlane(fake_line, K_inv)
            point0 = CalculateIntersectionPoint(
                fake_plane, pwalls[-1], pfloor, K, 0, downsample=downsample, size=size)
            point1 = CalculateIntersectionPoint(
                fake_plane, pwalls[-1], pceiling, K, 1, downsample=downsample, size=size)
            downs.append(point0)
            ups.append(point1)
            param_layout.append(pwalls[0])
        else:
            # left boundary
            fake_line = np.array([ups[0], downs[0]])
            m = (fake_line[0, 0] - fake_line[1, 0]) / \
                (fake_line[0, 1] - fake_line[1, 1])
            left_line = np.zeros_like(fake_line)

            if m < 0:
                left_line[0] = [0, 0]
            else:
                left_line[0] = [0, size[0]]
            b = left_line[0, 0] - m * left_line[0, 1]
            left_line[1] = [m*size[0]/2 + b, size[0]/2]

            fake_plane = CalculateFakePlane(left_line, K_inv)
            point0 = CalculateIntersectionPoint(
                fake_plane, pwalls[0], pfloor, K, 0, downsample=downsample, size=size)
            point1 = CalculateIntersectionPoint(
                fake_plane, pwalls[0], pceiling, K, 1, downsample=downsample, size=size)
            downs = [point0, *downs]
            ups = [point1, *ups]
            # right boundary
            fake_line = np.array([ups[-1], downs[-1]])
            m = (fake_line[0, 0] - fake_line[1, 0]) / \
                (fake_line[0, 1] - fake_line[1, 1])
            right_line = np.zeros_like(fake_line)

            if m < 0:
                right_line[0] = [size[1], size[0]]
            else:
                right_line[0] = [size[1], 0]
            b = right_line[0, 0] - m * right_line[0, 1]
            right_line[1] = [m*size[0]/2 + b, size[0]/2]

            fake_plane = CalculateFakePlane(right_line, K_inv)
            point0 = CalculateIntersectionPoint(
                fake_plane, pwalls[-1], pfloor, K, 0, downsample=downsample, size=size)
            point1 = CalculateIntersectionPoint(
                fake_plane, pwalls[-1], pceiling, K, 1, downsample=downsample, size=size)
            downs.append(point0)
            ups.append(point1)
            param_layout.append(pwalls[-1])

    else:
        assert len(pfloor) + len(pceiling) == 1
        if len(pceiling) == 1:
            attribution = 1
            param_layout.append(pceiling)
        else:
            param_layout.append(pfloor)
            attribution = 2
    return ups, downs, attribution, param_layout


def ConvertLayout(img, ups, downs, attribution, pwalls=None, pfloor=None, pceiling=None, K=None, ixy1map=None, oxy1map=None, valid=None, pixelwise=None):
    # display segmentation
    import cv2
    img = img.cpu().numpy().transpose([1, 2, 0])
    mean, std = np.array([0.485, 0.456, 0.406]), np.array(
        [0.229, 0.224, 0.225])
    img = ((img * std) + mean)*255
    img = img[:, :, ::-1]
    h, w = img.shape[0], img.shape[1]
    colors = np.random.uniform(10, 250, (30, 3))
    floor = (0, 255, 0)
    ceiling = (255, 0, 0)
    segmentation = -1 * np.ones([h, w])  # [0: ceiling, 1: floor, 2...:walls]
    valid_pwalls = []
    if attribution == 1:  # only ceiling
        mask = np.zeros_like(img)
        mask[:, :] = ceiling
        mask_img = (img + mask) / 2
        segmentation[:, :] = 0
    elif attribution == 2:  # only floor
        mask = np.zeros_like(img)
        mask[:, :] = floor
        mask_img = (img + mask) / 2
        segmentation[:, :] = 1
    else:
        mask = np.zeros((h, w, 3), np.uint8)

        ups = np.array(ups).astype(np.int32)
        minuy = min(np.min(ups[:, 1]) - 10, -1)

        downs = np.array(downs).astype(np.int32)
        maxdy = max(np.max(downs[:, 1]) + 10, h+1)

        # draw floor and ceiling
        if len(pceiling) > 0:
            cv2.fillPoly(img=mask, pts=np.array(
                [[[ups[0, 0], minuy], *ups, [ups[-1, 0], minuy]]]), color=ceiling)
            cv2.fillPoly(img=segmentation, pts=np.array(
                [[[ups[0, 0], minuy], *ups, [ups[-1, 0], minuy]]]), color=0)
        if len(pfloor) > 0:
            cv2.fillPoly(img=mask, pts=np.array(
                [[[downs[0, 0], maxdy], *downs, [downs[-1, 0], maxdy]]]), color=floor)
            cv2.fillPoly(img=segmentation, pts=np.array(
                [[[downs[0, 0], maxdy], *downs, [downs[-1, 0], maxdy]]]), color=1)
        # draw walls
        assert len(ups) == len(pwalls) + 1
        j = -1
        for i in range(len(ups)-1):
            u0 = ups[i]
            u1 = ups[i+1]
            d0 = downs[i]
            d1 = downs[i+1]
            if pwalls[i] is None:
                assert i > 0 and i < len(ups)-2
                continue
            else:
                valid_pwalls.append(pwalls[i])
                j = j + 1
            color = tuple(colors[j].tolist())
            cv2.fillPoly(img=mask, pts=np.array(
                [[u0, d0, d1, u1]]), color=color)
            cv2.fillPoly(img=segmentation, pts=np.array(
                [[u0, d0, d1, u1]]), color=2+j)
    # gt layout
    gtlayout_mask = np.zeros_like(img)
    labels = np.unique(valid)
    for i, label in enumerate(labels):
        gtlayout_mask[valid == label] = colors[i]
    _gtlayout_mask = (0.7 * gtlayout_mask + 0.3 * img)

    # display depth
    K_inv = np.linalg.inv(K)
    pwinverdepth = np.ones_like(segmentation) * 1e5
    # pixelwise
    if pixelwise is not None:
        n_d = pixelwise[:3] / np.clip(pixelwise[3], 1e-8, 1e8)
        n_d = np.transpose(n_d, [1, 2, 0])
        pwinverdepth = -1 * np.sum(np.dot(n_d, K_inv) * oxy1map, axis=2)  # 1/z
        pwinverdepth = cv2.resize(
            pwinverdepth, (w, h), interpolation=cv2.INTER_LINEAR)
        pwinverdepth[pwinverdepth <= 0.02] = 1e5

    # instance depth
    valid = valid != -1
    depth = np.ones_like(segmentation) * 1e5
    labels = np.unique(segmentation[valid])
    for i, label in enumerate(labels):
        label = int(label)
        if label == -1:
            assert i == 0
            continue
        mask = segmentation == label
        if label == 0:
            param = pceiling[0]
        elif label == 1:
            param = pfloor[0]
        else:
            param = valid_pwalls[label-2]
        if param is None:
            raise IOError
        else:
            n_d = param[:3] / np.clip(param[3], 1e-8, 1e8)  # meter n/d
            n_d = n_d[np.newaxis, np.newaxis, :]
            inverdepth = -1 * np.sum(np.dot(n_d, K_inv) * ixy1map, axis=2)
            depth[mask] = inverdepth[mask]

    depth[depth <= 0.02] = pwinverdepth[depth <= 0.02]
    depth[depth == 1e5] = pwinverdepth[depth == 1e5]

    # display mesh 3D layout
    try:
        _2ds, _3ds = DisplayMeshLayout(
            ups, downs, attribution, pwalls, pceiling, pfloor, ixy1map, K_inv)
    except:
        _2ds, _3ds = None, None  # TODO
        pass
    polys = [_2ds, _3ds]
    return segmentation, depth, img, polys


def Convert2DTo3D(point2d, ixy1map, param, K_inv):
    point2d = point2d.reshape([2, -1]).T.astype(np.int32)
    point2d = ixy1map[point2d[:, 1], point2d[:, 0]]  # n*3
    n_d = param[:3] / np.clip(param[3], 1e-8, 1e8)  # meter n/d
    n_d = n_d[None]
    z = -1 / np.sum(np.dot(n_d, K_inv) * point2d, axis=1)  # n
    _3d = (z[None] * np.dot(K_inv, point2d.T)).T  # n*3
    _2d_ploygon = point2d[:, :2].tolist()
    _3d_polygon = _3d.tolist()
    return _2d_ploygon, _3d_polygon


def DisplayMeshLayout(ups, downs, attribution, pwalls, pceiling, pfloor, ixy1map, K_inv):
    from shapely.geometry import Polygon
    _2ds = []
    _3ds = []
    h, w = 359, 639
    empty_polygon = Polygon([[0, 0], [w, 0], [w, h], [0, h]])
    if attribution == 1:  # only ceiling
        poly = Polygon([[0, 0], [w, 0], [w, h], [0, h]])
        contour_ = np.array(poly.exterior.coords.xy)
        _2d, _3d = Convert2DTo3D(contour_, ixy1map, pceiling[0], K_inv)
        _2ds.append(_2d)
        _3ds.append(_3d)
    elif attribution == 2:  # only floor
        poly = Polygon([[0, 0], [w, 0], [w, h], [0, h]])
        contour_ = np.array(poly.exterior.coords.xy)
        _2d, _3d = Convert2DTo3D(contour_, ixy1map, pfloor[0], K_inv)
        _2ds.append(_2d)
        _3ds.append(_3d)
    else:
        ups = np.array(ups).astype(np.int32)
        minuy = min(np.min(ups[:, 1]) - 10, -1)

        downs = np.array(downs).astype(np.int32)
        maxdy = max(np.max(downs[:, 1]) + 10, h+1)

        # draw floor and ceiling
        if len(pceiling) > 0:
            poly = Polygon([[ups[0, 0], minuy], *ups, [ups[-1, 0], minuy]])
            poly = empty_polygon.intersection(poly)
            contour_ = np.array(poly.exterior.coords.xy)[:, :-1]
            _2d, _3d = Convert2DTo3D(contour_, ixy1map, pceiling[0], K_inv)
            _2ds.append(_2d)
            _3ds.append(_3d)
        if len(pfloor) > 0:
            poly = Polygon(
                [[downs[0, 0], maxdy], *downs, [downs[-1, 0], maxdy]])
            poly = empty_polygon.intersection(poly)
            contour_ = np.array(poly.exterior.coords.xy)[:, :-1]
            _2d, _3d = Convert2DTo3D(contour_, ixy1map, pfloor[0], K_inv)
            _2ds.append(_2d)
            _3ds.append(_3d)
        # draw walls
        assert len(ups) == len(pwalls) + 1
        for i in range(len(ups) - 1):
            u0 = ups[i]
            u1 = ups[i + 1]
            d0 = downs[i]
            d1 = downs[i + 1]
            if pwalls[i] is None:
                assert i > 0 and i < len(ups) - 2
                continue
            poly = Polygon([u0, d0, d1, u1])
            poly = empty_polygon.intersection(poly)
            contour_ = np.array(poly.exterior.coords.xy)[:, :-1]
            _2d, _3d = Convert2DTo3D(contour_, ixy1map, pwalls[i], K_inv)
            _2ds.append(_2d)
            _3ds.append(_3d)
    return _2ds, _3ds


def Reconstruction(planes, params_ins, lines, K, size, threshold=(0.2, 0.05, 0.05, 0.5), downsample=8, cat='both'):
    '''
    :param: planes, params_ins, lines are detection results. 
    :param: K is camera intrinsic. 
    :param: size is the original image size that corresponds to K.
    :param: threshold is to filter wall floor ceiling and line. 
    :param: downsample is ratio that corresponds to K. (input downsample * model downsample = 2 * 4)
    :param: which mode
    '''
    # filter detection results
    walls, pwalls, pfloor, pceiling, lines = PreProcess(planes, params_ins, lines, threshold=threshold)
    # merge two neigiborhood walls with similar plane params
    walls, pwalls = MergeNeighborWithSameParam(walls, pwalls)
    # define potential intersection region between two neighbor walls
    common, pwalls = CommonRegion(walls, pwalls, width=size[1]/downsample)
    # filter texture line, it's not essential, we only consider the lines which locate in potential intersection region
    lines = FilterLine(common, lines, height=size[0]/downsample)
    # no optimizer
    _ups, _downs, _attribution, _params_layout = None, None, None, None
    # optimizer
    ups, downs, attribution, params_layout = None, None, None, None
    if cat == 'both':
        # judge space relationship between two neighbor walls and optimizer plane params if opt=True
        _, pwalls, case, dtls = OptimizerLayout(pwalls, lines, common, K, size=size, downsample=downsample, opt=True)
        # calculate opt params intersection point with floor and ceiling
        ups, downs, attribution, params_layout = GenerateLayout(
            pwalls, case, dtls, pfloor, pceiling, K, size, downsample=downsample/4, upsample=downsample)
        # calculate no opt params intersection point with floor and ceiling    
        _ups, _downs, _attribution, _params_layout = GenerateLayout(
            _, case, dtls, pfloor, pceiling, K, size, downsample=downsample/4, upsample=downsample)
    elif cat == 'opt':
        # judge space relationship between two neighbor walls and optimizer plane params if opt=True
        _, pwalls, case, dtls = OptimizerLayout(pwalls, lines, common, K, size=size, downsample=downsample, opt=True)
        # calculate opt params intersection point with floor and ceiling
        ups, downs, attribution, params_layout = GenerateLayout(
            pwalls, case, dtls, pfloor, pceiling, K, size, downsample=downsample/4, upsample=downsample)
    else:
        # judge space relationship between two neighbor walls and optimizer plane params if opt=True
        _, pwalls, case, dtls = OptimizerLayout(pwalls, lines, common, K, size=size, downsample=downsample, opt=False)
        # calculate no opt params intersection point with floor and ceiling
        _ups, _downs, _attribution, _params_layout = GenerateLayout(
            _, case, dtls, pfloor, pceiling, K, size, downsample=downsample / 4, upsample=downsample)

    return (_ups, _downs, _attribution, _params_layout), (ups, downs, attribution, params_layout), (pfloor, pceiling)
