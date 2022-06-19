# -*- coding: utf-8 -*-
import numpy as np
from PIL import Image
import colorsys
import cv2
import numba
import itertools as it

def prn_obj(obj):
    print('\n'.join(['%s:%s' % item for item in obj.__dict__.items()]))

def HSVColor(img):
    if isinstance(img,Image.Image):
        r,g,b = img.split()
        Hdat = []
        Sdat = []
        Vdat = []
        for rd,gn,bl in zip(r.getdata(),g.getdata(),b.getdata()) :
            h,s,v = colorsys.rgb_to_hsv(rd/255.,gn/255.,bl/255.)
            Hdat.append(int(h*255.))
            Sdat.append(int(s*255.))
            Vdat.append(int(v*255.))
        r.putdata(Hdat)
        g.putdata(Sdat)
        b.putdata(Vdat)
        return Image.merge('RGB',(r,g,b))
    else:
        return None

def getDistance(p1, p2):
    '''
    求欧式距离
    :param p1:
    :param p2:
    :return:
    '''
    return np.sqrt( (p1[0]-p2[0]) * (p1[0]-p2[0]) + (p1[1]-p2[1]) * (p1[1]-p2[1]))

def getPoint2Line(p, p1, p2):
    '''
    获得点p到p1p2所定直线的距离
    '''
    x0, y0 = p
    x1, y1 = p1
    x2, y2 = p2
    if x1==x2:
        return np.fabs(x0-x1)
    else:
        a = y2-y1
        b = x1-x2
        c = x2*y1-x1*y2
        return np.fabs(a*x0+b*y0+c)/np.sqrt(a**2+b**2)

def getMinPoint2Points(p, p1, p2):
    '''
    获得p到p1和p2的最短距离
    :param p:
    :param p1:
    :param p2:
    :return:
    '''
    l1 = getDistance(p, p1)
    l2 = getDistance(p, p2)
    return l1 if l1<l2 else l2

def getCosin(p1, p2, p0):
    '''
    计算中三个点的夹角，即∠p0的弧度
    :param p1: 一个端点
    :param p2: 另一个端点
    :param p0: 角的顶点
    :return: 返回吉∠p0的弧度
    ref: 已知三个点的坐标，求中间点的夹角余弦
		 ref:https://wenku.baidu.com/view/7d97f0a7b0717fd5360cdc75.html
    '''
    a = getDistance(p0, p1)
    b = getDistance(p0, p2)
    c = getDistance(p1, p2)
    return (a*a + c*c - b*b) / (2*a*c)

def StdGaussian(x):
    return np.exp( - x * x / 2) / np.sqrt(2 * np.pi)

def getGaussianMask(width,height):
    '''
    生成一个width*height的高斯蒙版
    :param width:
    :param height:
    :return:
    '''
    cx = width//2
    cy = height // 2
    R = np.sqrt(cx**2 + cy**2)
    map = np.zeros((height, width), dtype=np.float)
    for i in range(height):
        for j in range(width):
            dis = np.sqrt((i-cy)**2 + (j-cx)**2)
            map[i, j] = np.exp((-0.5 * dis / R))

    return map / (width * height)

def getGradientMap(img):
    '''
    生成一阶梯度图
    注意：由于一阶梯度和形态学梯度差不多，为了代码统一，本实现使用标准形态学createMorphGradeMap_Standard
    :return:
    '''
    src = np.array(img)
    Gx = cv2.Sobel(src,cv2.CV_64F,1,0)
    Gy = cv2.Sobel(src,cv2.CV_64F,0,1)
    G = np.sqrt(Gx * Gx + Gy * Gy)
    return G

def Normalize(data):
    mx = np.max(data)
    mn = np.min(data)
    return (data-mn) / (mx - mn)

def getMahalanobis(img, trainpixels):
    vals = []
    np_img = np.asarray(img)
    rows, cols, channel = np_img.shape

    print(np_img.shape)
    for tp in trainpixels:
        x,y = tp
        vals.append(np_img[y, x]) # y is rows and x is column

    vals = np.array(vals)
    mean = np.mean(vals, axis=0)
    cov=np.cov(vals, rowvar=False) #每列为一个维度

    # print("协方差矩阵：",cov)
    # print("cov的模：", np.linalg.norm(cov))
    if np.linalg.norm(cov) == 0:#如果涂鸦是恒定颜色
        ones = np.ones(channel)
        inv = np.diag(ones)
    else:
        inv = np.linalg.inv(cov)
        inv = Normalize(inv)

    res = np_img - mean #计算每个像素点的残差
    # mahalabonis distance = (I(x) - mean).T * cov.inv * (I(x) - mean)
    res = res.reshape(-1,channel)
    ma = np.dot(res, inv)
    ma = np.sum(ma * res, axis=1)
    ma = ma.reshape(rows, cols)
    #
    ma = Normalize(ma)

    return ma
    pass

def polynomialFitting(coords, n):
    '''
    多项式拟合
    :param coords:待拟合的坐标列表
    :param n: n次多项式拟合
    :return: 拟合后的坐标list
    '''
    points = np.array(coords)
    # space = len(points) // 10
    x = points[::10,0]
    y = points[::10,1]
    a = np.polyfit(x, y, n)  # 用n次多项式拟合x，y数组
    b = np.poly1d(a)  # 拟合完之后用这个函数来生成多项式对象
    c = b(x)  # 生成多项式对象之后，就是获取x在这个多项式处的值
    return list(zip(x,c))

@numba.jit()
def getEuclidianDistance(distance, p1):
    x1, y1 = p1
    height, width = distance.shape
    for y in range(height):
        for x in range(width):
            distance[y, x] = np.sqrt((x-x1)**2 + (y-y1)**2)

    pass

def smoothCurve(points, mode):
    '''
    曲线平滑滤波算法
    :param points: 原始曲线点列表
    :param mode: 滤波模式
    :return: 滤波后的曲线点列表
    '''
    points_2 = []
    points_2.append(points[0])
    points_2.append(points[1])

    for i in range(2, len(points)-2):
        p = np.array(points[i])
        pp_1 = np.array(points[i-1])
        pp_2 = np.array(points[i-2])
        pn_1 = np.array(points[i+1])
        pn_2 = np.array(points[i+2])
        if mode==0: # 三点线性
            p = (pp_1+p+pn_1) / 3
        elif mode == 1: #五点二次滤波
            p = (12 * (pp_1+pn_1) - 3 * (pp_2 + pn_2) + 17 * p) / 35
        elif mode == 2: # 三点钟形滤波
            p = 0.212 * pp_1 + 0.576 * p + 0.212 * pn_1
        elif mode==3: #五点钟形滤波
            p = (0.11 * (pp_2 + pn_2)+  0.24 * (pp_1 + pn_1) + 0.3 * p)
        elif mode == 4: # 三点汉明滤波
            p = 0.07 * pp_1 + 0.86 * p + 0.07 * pn_1
        elif mode == 5: #五点汉明滤波
            p = 0.04 * (pp_2 + pn_2) + 0.24 * (pp_1 + pn_1) + 0.44 * p

        p=p.astype(np.int).tolist()
        points_2.append(p)

    points_2.append(points[-2])
    points_2.append(points[-1])
    return points_2
    pass

def meanSmoothCurve(points, length):
    '''
    曲线平滑滤波算法
    :param points: 原始曲线点列表
    :param mode: 滤波模式
    :return: 滤波后的曲线点列表
    '''
    points_2 = []
    if len(points)<length:
        return points

    points = np.array(points)
    for i in range(0, len(points)):
        st = 0 if i-length//2<0 else i-length//2
        if st+length<=len(points):
            stp = st + length
        else:
            stp = len(points)
            st = stp - length

        p = np.array([0,0])
        for pp in points[st:stp]:
            p+=pp
        p = p / length
        p=p.astype(np.int).tolist()
        points_2.append(p)

    return points_2
    pass

def cubicSmoothCurve(points, iteration):
    '''
    % 五点三次平滑滤波
    :param points: 原始曲线点列表
    :param iteration: 迭代次数
    :return: 滤波后的曲线点列表
    '''

    n = len(points)
    a = np.array(points, dtype=np.float32)
    b = np.zeros_like(a)
    for i in range(iteration):
        b[0] = (69 * a[0] + 4 * (a[1] + a[3]) - 6 * a[2] -a[4]) / 70.0
        b[1] = (2 * (a[0] + a[4]) + 27 * a[1] + 12 * a[2] - 8 * a[3]) / 35.0
        for j in range(2, n-2):
            b[j] = (-3 * (a[j-2]+a[j+2]) + 12 * (a[j-1]+a[j+1]) + 17 * a[j]) / 35.0
        b[-2] = (2*(a[-1]+a[-5]) + 27 * a[-2] + 12 * a[-3] - 8 * a[-4]) / 35.0
        b[-1] = (69 * a[-1] + 4 * (a[-2]+a[-4]) - 6 * a[-3] - a[-5]) / 70.0
        a = b
        b = np.zeros_like(a)
    return a.astype(np.int32).tolist()

def getRoi(img, pt, dx, dy):
    col, row = pt
    height, width = img.shape
    left = 0 if col - dx//2 <0 else col - dx//2
    right= width if col + dx//2>width else col + dx//2
    top = 0 if row - dy // 2<0 else row + dy//2
    bottom = height if row + dy//2 > height else row + dy//2
    return img[top:bottom, left:right]


@numba.jit
def sweep(A, Cost):
    max_diff = 0.0
    for i in range(1, A.shape[0]):
        for j in range(1, A.shape[1]):
            t1, t2 = A[i, j - 1], A[i - 1, j]
            C = Cost[i, j]
            if abs(t1 - t2) > C:
                t0 = min(t1, t2) + C  # handle degenerate case
            else:
                t0 = 0.5 * (t1 + t2 + np.sqrt(2 * C * C - (
                            t1 - t2) ** 2))  # 参考《A FAST SWEEPING METHOD FOR EIKONAL EQUATION》或者：https://blog.csdn.net/lusongno1/article/details/88409735
            max_diff = max(max_diff, A[i, j] - t0)
            A[i, j] = min(A[i, j], t0)
    return max_diff


def GDT(A, C, max_iter_n=4, max_diff=0.1):
    A = A.copy()
    sweeps = [A, A[:, ::-1], A[::-1], A[::-1, ::-1]]  # A[:,::-1]水平翻转，A[::-1]垂直翻转 A[::-1,::-1]水平+垂直翻转
    costs = [C, C[:, ::-1], C[::-1], C[::-1, ::-1]]
    for i, (a, c) in enumerate(it.cycle(zip(sweeps, costs))):
        r = sweep(a, c)
        # print(i, r)
        if r < max_diff or i >= max_iter_n:
            break
    return A


@numba.jit
def traceBack(U, p1, p2):
    '''
    利用能量图，反向从终点追踪到起点
    :param U: 最小能量图
    :param p1: 起点
    :param p2: 终点
    :return:
    '''
    x1, y1 = p1
    height, width = U.shape
    roadcenters = []
    minP = p2
    roadcenters.append(minP) # 把终点加入，因为终点经过meanshift调整过了，记住后面不再把终止人工种子点加入路径
    while True:
        minU = float('inf')
        x, y = minP
        for i in range(-1, 2):
            for j in range(-1, 2):
                if i == 0 and j == 0:
                    continue
                xx = x + i
                yy = y + j
                if xx < 0 or yy < 0 or xx >= width or yy >= height:
                    continue

                v = U[yy, xx]
                if minU > v:
                    minU = v
                    minP = [xx, yy]

        if minP in roadcenters:
            raise Exception('出现回路，提取失败')

        roadcenters.append(minP)  # 把中心点加入（包括起始点），因为起点经过meanshift调整过了，记住后面不再把起始人工种子点加入路径

        if minP[0] == x1 and minP[1] == y1:  # 回到起点
            break

    roadcenters.reverse()
    return roadcenters

@numba.jit
def meanShift(W, p, d=5, step=5, th_shift=0.01, maxiter=1000):
    '''
    均值漂移，利用W（密度），将p调整到密度中心，ref:https://blog.csdn.net/qwerasdf_1_2/article/details/54577336
    :param W:密度场
    :param p:原始起点
    :param d:测试半径
    :param th_shift: 偏移阈值，当偏移量小于该值，则停止漂移
    :return:调整后的点
    '''
    # import matplotlib.pyplot as plt
    # plt.figure()
    # plt.imshow(W, "jet")
    # plt.scatter(p[1], p[0], s=20, c="yellow", marker="+")
    # plt.pause(0.1)
    height, width = W.shape
    x,y = p
    it = 0
    while(True):
        sw = 0
        sx = 0
        sy = 0
        for r in range(-d, d+1):
            for c in range(-d, d+1):
                yy = y + r
                xx = x + c
                iyy = int(np.round(yy))
                ixx = int(np.round(xx))
                if iyy<0 or iyy>=height or ixx<0 or ixx>=width:
                    continue
                if np.sqrt((yy-y)**2 + (xx-x)**2)>d:
                    continue
                sx += xx * W[iyy, ixx]
                sy += yy * W[iyy, ixx]
                sw += W[iyy, ixx]

        vx = (sx/sw-x) * step
        vy = (sy/sw-y) * step

        vx = d if vx > d else vx
        vy = d if vy > d else vy
        if np.sqrt(vx**2+vy**2)<th_shift:
            break

        x += vx
        y += vy


        # plt.scatter(x, y, s=20, c="blue", marker="+")
        # plt.pause(0.1)
        # print('meanshift:', x, y)
        it+=1
        if it>maxiter:
            break;
    return [int(x), int(y)]
    pass

def directLine(p1,p2):
    '''
    计算连个点之间的直线坐标
    :param p1: 起点
    :param p2: 终点
    :return: 两个点之间的直线坐标点
    '''
    y1, x1 = p1
    y2, x2 = p2
    pts = []
    if(np.fabs(x2-x1)>np.fabs(y2-y1)):
        step = 1 if x2>x1 else -1
        for x in range(x1+step, x2, step):
            y = (x-x1) / (x2-x1) * (y2-y1) + y1
            pts.append([int(y), int(x)])
    else:
        step = 1 if y2 > y1 else -1
        for y in range(y1+step, y2, step):
            x = (y - y1) / (y2 - y1) * (x2 - x1) + x1
            pts.append([int(y), int(x)])
    return pts
    pass

def gabor_fn(sigma, theta, Lambda, psi, gamma):
    '''
    gabor滤波器生成函数
    :param sigma: 高斯函数的标准差，控制滤波器的
    :param theta: 滤波器的方向
    :param Lambda: 滤波器的长度
    :param psi: 滤波器相位偏移
    :param gamma: 空间纵横比，决定了Gabor函数形状，当γ= 1时，形状是圆的。当γ< 1时，形状随着平行条纹方向而拉长
    :return: 滤波器蒙版
    '''
    sigma_x = sigma
    sigma_y = float(sigma) / gamma

    # Bounding box
    nstds = 3 # Number of standard deviation sigma
    # xmax = max(abs(nstds * sigma_x * np.cos(theta)), abs(nstds * sigma_y * np.sin(theta)))
    # xmax = np.ceil(max(1, xmax))
    # ymax = max(abs(nstds * sigma_x * np.sin(theta)), abs(nstds * sigma_y * np.cos(theta)))
    # ymax = np.ceil(max(1, ymax))
    # xmin = -xmax
    # ymin = -ymax
    width = max(abs(nstds * sigma_x), abs(nstds * sigma_y))
    width = np.ceil(width).astype(np.int)
    width = 20 if width > 20 else width
    # width = 31
    # theta = np.pi / 8
    xmax = width
    xmin = -xmax
    ymax = width
    ymin = -ymax
    (x, y) = np.meshgrid(np.arange(ymin, ymax + 1), np.arange(xmin, xmax + 1))
    # Rotation
    x_theta = x * np.cos(theta) + y * np.sin(theta)
    y_theta = -x * np.sin(theta) + y * np.cos(theta)

    x_theta = np.round(x_theta, 3)
    y_theta = np.round(y_theta, 3)
    gb = np.exp(-.5 * (x_theta ** 2 / sigma_x ** 2 + y_theta ** 2 / sigma_y ** 2))* np.cos(2 * np.pi / Lambda * x_theta + psi)

    gb = gb / np.sum(gb)
    return gb

def enhanceByGaborFilters(img):
    gammas = [0.5]  # 长宽比
    theta_int = 5  # 滤波器组的方向间隔
    thetas = range(0, 180, theta_int)

    filters = []
    sigma = 2
    lamda = 30
    for gamma in gammas:  # MASS
        # for gamma in [0.1, 0.2, 0.4]: # GOOGLE EARTH
        for theta in thetas:
            r_theta = theta * np.pi / 180
            gb = gabor_fn(sigma, r_theta, lamda, 0, gamma)
            filters.append(gb)
    # 再加入一个高斯滤波，其实就是gamma=1 , 只用于Google earth
    # gb = gabor_fn(sigma, r_theta, lamda, 0, 1)
    # filters.append(gb)

    resp = np.ones(img.shape, np.float32) * (-1e5)
    for gb in filters:
        fimg = cv2.filter2D(img, cv2.CV_32F, gb)
        np.maximum(resp, fimg, resp)

    resp = ((resp-np.min(resp)) / (np.max(resp) -np.min(resp))* 255).astype(np.uint8)

    return resp

def psf2otf(psf, size):
    if not (0 in psf):
        # Pad the PSF to outsize
        psf = np.double(psf)
        psfsize = np.shape(psf)
        psfsize = np.array(psfsize)
        padsize = size - psfsize
        psf = np.lib.pad(psf, ((0, padsize[0]), (0, padsize[1])), 'constant')
        # Circularly shift otf so that the "center" of the PSF is at the (1,1) element of the array.
        psf = np.roll(psf, -np.array(np.floor(psfsize / 2), 'i'), axis=(0, 1))
        # Compute the OTF
        otf = np.fft.fftn(psf, axes=(0, 1))
        # Estimate the rough number of operations involved in the computation of the FFT.
        nElem = np.prod(psfsize, axis=0)
        nOps = 0
        for k in range(0, np.ndim(psf)):
            nffts = nElem / psfsize[k]
            nOps = nOps + psfsize[k] * np.log2(psfsize[k]) * nffts
        mx1 = (abs(np.imag(otf[:])).max(0)).max(0)
        mx2 = (abs(otf[:]).max(0)).max(0)
        eps = 2.2204e-16
        if mx1 / mx2 <= nOps * eps:
            otf = np.real(otf)
    else:
        otf = np.zeros(size)
    return otf

def L0Smoothing(Im, lamda=2e-2, kappa=2.0):
    S = Im / 255
    betamax = 1e5
    fx = np.array([[1, -1]])
    fy = np.array([[1], [-1]])
    N, M, D = np.shape(Im)
    sizeI2D = np.array([N, M])
    otfFx = psf2otf(fx, sizeI2D)
    otfFy = psf2otf(fy, sizeI2D)
    Normin1 = np.fft.fft2(S, axes=(0, 1))
    Denormin2 = abs(otfFx) ** 2 + abs(otfFy) ** 2
    if D > 1:
        D2 = np.zeros((N, M, D), dtype=np.double)
        for i in range(D):
            D2[:, :, i] = Denormin2
        Denormin2 = D2
    beta = lamda * 2
    while beta < betamax:
        Denormin = 1 + beta * Denormin2
        # h-v subproblem
        h1 = np.diff(S, 1, 1)
        h2 = np.reshape(S[:, 0], (N, 1, 3)) - np.reshape(S[:, -1], (N, 1, 3))
        h = np.hstack((h1, h2))
        v1 = np.diff(S, 1, 0)
        v2 = np.reshape(S[0, :], (1, M, 3)) - np.reshape(S[-1, :], (1, M, 3))
        v = np.vstack((v1, v2))
        if D == 1:
            t = (h ** 2 + v ** 2) < lamda / beta
        else:
            t = np.sum((h ** 2 + v ** 2), 2) < lamda / beta
            t1 = np.zeros((N, M, D), dtype=np.bool)
            for i in range(D):
                t1[:, :, i] = t
            t = t1
        h[t] = 0
        v[t] = 0
        # S subproblem
        Normin2 = np.hstack((np.reshape(h[:, -1], (N, 1, 3)) - np.reshape(h[:, 0], (N, 1, 3)), -np.diff(h, 1, 1)))
        Normin2 = Normin2 + np.vstack(
            (np.reshape(v[-1, :], (1, M, 3)) - np.reshape(v[0, :], (1, M, 3)), -np.diff(v, 1, 0)))
        FS = (Normin1 + beta * np.fft.fft2(Normin2, axes=(0, 1))) / Denormin
        S = np.real(np.fft.ifft2(FS, axes=(0, 1)))
        beta *= kappa
        print('.')
    print('\n')
    return S

def direction_estimation(img):
    # 获得种子点所在切片的边缘信息(先双边平滑处理），及其离散傅里叶频谱
    # 分析频谱，获得道路的主方向
    if len(img.shape)==3:
        img = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    #双边滤波
    # d – 在滤波过程中使用的各像素邻域直径，如果这是一个非整数，则这个值由sigmaSpace决定。
    # sigmaColor – 颜色空间的标准方差。数值越大，意味着越远的的颜色会被混进邻域内，从而使更大的颜色段获得相同的颜色。
    # sigmaSpace – 坐标空间的标注方差。 数值越大，以为着越远的像素会相互影响，从而使更大的区域足够相似的颜色获取相同的颜色。
    # 当d>1999，d指定了邻域大小且与sigmaSpace无关。否则，d正比于sigmaSpace。

    # img = cv2.bilateralFilter(img, 3, 3 * 2, 3 / 2)
    edge = cv2.Canny(img,0,60,3)

    fft2 = np.fft.fft2(edge)
    shift2center = np.fft.fftshift(fft2)
    log_shift2center = np.log(1 + np.abs(shift2center))

    sumfft = np.zeros(180)
    row, col = img.shape
    R = row if row < col else col
    R //= 2
    y0 = row // 2
    x0 = col // 2
    for theta in range(180):
        sumfft[theta] = 0
        for r in range(R):
            x = int(x0 + r * np.cos(theta * np.pi / 180.0))
            y = int(y0 + r * np.sin(theta * np.pi / 180.0))
            sumfft[theta] += log_shift2center[y, x]

    angle = np.argmax(sumfft)

    if False:
        x = int(x0 + R * np.cos(angle*np.pi / 180))
        y = int(y0 + R * np.sin(angle*np.pi / 180))
        print(np.max(log_shift2center))
        print(np.min(log_shift2center))
        log_shift2center = (log_shift2center-np.min(log_shift2center)) / (np.max(log_shift2center) - np.min(log_shift2center)) * 255
        log_shift2center = log_shift2center.astype(np.uint8)
        rgb = cv2.cvtColor(log_shift2center, cv2.COLOR_GRAY2RGB)
        cv2.line(rgb, (x0,y0), (x,y),(0,0,255), 1, 8, 0)

        import matplotlib.pyplot as plt
        plt.figure()
        plt.subplot(131)
        plt.imshow(img, cmap='gray')
        plt.scatter(32, 32)
        plt.subplot(132)
        plt.imshow(edge, cmap='gray')
        plt.subplot(133)
        plt.imshow(rgb)
        plt.show()

    # 道路的主方向与能量的主方向垂直
    angle += 90
    return angle

    pass