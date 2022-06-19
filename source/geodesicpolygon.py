'''
    本程序改进自GeodesicOnce
    改进点：
        1.将geodesiconce提取的路径作为初始中心线，对初始中心线上每隔一定距离取一个关键点，对
'''
import time
import numpy as np
import cv2
import math
from PIL import Image
from util import getCosin, getDistance
from circle import Circle
from marker import Marker
from line import Line
from orderedlist import OrderList,Node
from util import StdGaussian, getGaussianMask, getGradientMap
import time
from comm import Comm
import numba
from util import getEuclidianDistance
from util import smoothCurve, meanSmoothCurve, cubicSmoothCurve
from util import GDT, traceBack
from util import directLine
from util import gabor_fn

class GeodesicPolygon(Comm):
    methodName = 'geodesciPolygon'
    def __init__(self, th_distance = 10, th_radius=8):
        super(GeodesicPolygon, self).__init__()
        self.roadcenters = []
        self.testing = False # 是否在实时跟踪
        self.th_distance = th_distance
        self.th_radius = th_radius
        self.epsilon = 1e-5

        self.a = 1
        self.b = 0
        self.c = 0

    def createMorphGradeMap_Standard(self):
        src = self.workingImg
        elem = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
        edged_img = cv2.morphologyEx(src, cv2.MORPH_GRADIENT, elem)
        # 标准化边缘图
        normlized_edged_img = edged_img / np.max(edged_img)
        # 存入画布中
        morphGradeMap = (normlized_edged_img * 255).astype(np.uint8)  # 转成灰度图（0-255）
        _, morphGradeMap = cv2.threshold(morphGradeMap,0, 255, cv2.THRESH_BINARY+cv2.THRESH_OTSU)
        print('sss', np.max(morphGradeMap))
        morphGradeMap = Image.fromarray(morphGradeMap)  # 转成Image
        self.roadcenters.morphGradeMap = morphGradeMap
        # 刷新
        # self.roadcenters.show_image()

    def createSobelMap(self):
        src = self.workingImg

        x = cv2.Sobel(src, cv2.CV_16S, 1, 0)
        y = cv2.Sobel(src, cv2.CV_16S, 0, 1)

        absX = cv2.convertScaleAbs(x)  # 转回uint8
        absY = cv2.convertScaleAbs(y)

        edged_img = cv2.addWeighted(absX, 0.5, absY, 0.5, 0)

        # import matplotlib.pyplot as plt
        # plt.figure()
        # plt.imshow(edged_img,'gray')
        # plt.show()

        # 标准化边缘图
        normlized_edged_img = edged_img / np.max(edged_img)
        # 存入画布中
        morphGradeMap = (normlized_edged_img * 255).astype(np.uint8)  # 转成灰度图（0-255）
        # _, morphGradeMap = cv2.threshold(morphGradeMap,0, 255, cv2.THRESH_BINARY+cv2.THRESH_OTSU)
        print('sss', np.max(morphGradeMap))
        self.morphGradeMap = Image.fromarray(morphGradeMap)  # 转成Image
        

    def sumGrayInsideTemplate(self, gray, center, radius):
        height, width = gray.shape
        ccx, ccy = center
        r = radius
        graysum = 0.0 # 这里很重要，不能赋值0，否则它只会保存0-255， 这里有坑
        maxv = np.max(gray)
        pixels = 0
        for i in range(-r, r + 1):
            for j in range(-r, r + 1):
                # 模板之外的像素
                if i * i + j * j > r * r:
                    continue
                # 图像之外的像素
                xx, yy = ccx + i, ccy + j
                if xx < 0 or yy < 0 or xx >= width or yy >= height:
                    graysum += maxv
                else:
                    graysum += gray[yy, xx]
                pixels += 1

        return graysum, pixels

    def shiftAndOptimizeCircleTemplate(self, seed):
        '''
        调整seed到道路中心点，同时返回最佳圆形模板的半径
        :param gradient: 形态学梯度图
        :param seed: 原始人工种子点
        :return: 最佳半径（注意，seed可能会被修改）
        '''
        if self.morphGradeMap is None:
            # self.createMorphGradeMap_Standard()
            self.createSobelMap()

        morphGradeMap = np.array(self.morphGradeMap)
        morphGradeMap = morphGradeMap / np.max(morphGradeMap)

        height, width = morphGradeMap.shape
        cx, cy = seed.point
        # 开始估算最佳圆形半径
        r = 1

        # region 调试
        # import matplotlib.pyplot as plt
        # fig = plt.figure()
        # ax = fig.add_subplot(1,1,1)
        # ax.imshow(self.roadcenters.image)
        # ax.scatter(seed.point[0], seed.point[1], c='r', marker='+', s=100, linewidths=20)
        # t = 10
        # endregion

        while True:
            # 梯度累积，每次计算当前中心点所在的3*3范围内的坐标为圆心的模板所覆盖的形态学梯度累积
            gradientSum = np.zeros(9)
            # 3*3左上角像素为圆心的模板序号为0
            for idx in range(9):
                # 局部位移
                dx, dy = idx % 3 - 1, idx // 3 - 1
                # 当前圆心
                ccx, ccy = cx + dx, cy + dy
                gradientSum[idx], _ = self.sumGrayInsideTemplate(morphGradeMap, [ccx, ccy], r)

            mind = np.argmin(gradientSum)
            dx, dy = mind % 3 - 1, mind // 3 - 1
            cx, cy = cx + dx, cy + dy

            # region 调试
            # if r % 2==1:
            #     ax.scatter(cx, cy, c='b', marker='o', alpha=0.9)
            #
            #     theta = np.linspace(0, 2 * np.pi, 200)
            #     x = cx + r * np.cos(theta)
            #     y = cy + r * np.sin(theta)
            #
            #     if len(ax.lines)>0:
            #         ax.lines.pop(0)
            #     ax.plot(x, y, color="yellow", linewidth=1)
            #
            #     plt.pause(t)
            #     if t>5:
            #         t=1
            #     plt.savefig('adacircle-'+str(r))
            # endregion


            if cx < 0 or cx >= width or cy < 0 or cy >= height:
                return -1
            # print(r, gradientSum[mind])
            if gradientSum[mind] > r:
                # ax.scatter(cx, cy, c='r', marker='o')
                # plt.savefig('adacircle-' + str(r))
                break
            else:
                r += 1

            # if r >= self.th_radius:
            #     break

        # 修正seed的坐标
        seed.point = [cx, cy]
        # 标记种子点已经被调整过，并记住最佳半径
        seed.radius = r

    def getPreCurSeed(self):
        '''
        该函数返回前后两个人工种子点:
        1. testing=True, 则 cur = allscribbles的栈顶tags=’tempseed'的Marker, pre = allscribbles的栈顶tags=’seed'的Marker
        2. testing=False, 则 cur = allscribbles的栈顶tags=’seed'的Marker, pre = allscribbles的栈顶cur前的Marker
        :return: allscribbles栈顶相邻的两个种子点
        '''
        pre, cur = None, None
        if self.testing:
            for i in range(-1, -len(self.roadcenters.allscribble) - 1, -1):
                obj = self.roadcenters.allscribble[i]
                # 要保证tempseed在seed之上
                if obj.tags == 'tempseed':
                    cur = obj
                elif obj.tags == 'seed':
                    pre = obj
                    break
        else:
            for i in range(-1, -len(self.roadcenters.allscribble)-1, -1):
                obj = self.roadcenters.allscribble[i]
                if obj.tags == 'seed':
                    if cur is None:
                        cur = obj
                    elif pre is None:
                        pre = obj
                        break
        return pre, cur

    def genRoadSaliencyMap(self, pre_seed, cur_seed):

        if pre_seed is None or cur_seed is None:
            print("至少要两个种子点——genRoadSaliencyMap")
            return

        # 使用小半径模板
        r = cur_seed.radius if pre_seed.radius > cur_seed.radius else pre_seed.radius

        # 原始图像灰度图
        src = np.array(self.workingImg.copy())

        # 获得前后两个道路种子点（调整过了）模板范围内的灰度和以及他们的平均值
        pre_gray, pre_pixels = self.sumGrayInsideTemplate(src, pre_seed.point, r)
        pre_avg_gray = pre_gray / pre_pixels
        cur_gray, cur_pixels = self.sumGrayInsideTemplate(src, cur_seed.point, r)
        cur_avg_gray = cur_gray / cur_pixels
        # 两个模板内的图像平均灰度
        avg_gray = int((pre_avg_gray + cur_avg_gray) / 2)
        # 求道路显著图
        src = src.astype(np.int)  # 这里很重要，否则原来时uint8时，下面的减法有问题，即-1的会编程255
        # src = 255-np.fabs(src - avg_gray).astype(np.uint8)
        src = 255-np.fabs(src - pre_avg_gray).astype(np.uint8)

        # region 调试
        # import matplotlib.pyplot as plt
        # tmp = src / np.max(src)
        # plt.imshow(tmp, 'rainbow')
        # plt.colorbar()
        # plt.axis('off')
        # plt.show()

        # endregion

        self.roadSaliencyMap = Image.fromarray(src)
        # self.roadcenters.show_image()

    def geodesic_distance(self, W, G, p1, p2):
        '''
        未使用有序列表，节省空间，但时间效率低
        :param W:
        :param G:
        :param p1:
        :param p2:
        :return:
        '''
        x1, y1 = p1
        height, width = W.shape
        mask = np.zeros_like(W, dtype=np.bool)
        m = np.ma.masked_array(np.ones_like(W), mask)
        mask = m.mask
        visit_mask = mask.copy()  # mask visited cells
        m = m.filled(np.inf)
        m[y1, x1]  = 0
        m[m != 0] = np.inf
        # distance_increments = np.asarray([np.sqrt(2), 1., np.sqrt(2), 1., 1., np.sqrt(2), 1., np.sqrt(2)])
        connectivity = [(i, j) for i in [-1, 0, 1] for j in [-1, 0, 1] if (not (i == j == 0))]
        cc = np.unravel_index(m.argmin(), m.shape)  # current_cell(row, cell)

        found = False #是否碰到了P2
        while not found:
            neighbors = [tuple(e) for e in np.asarray(cc) - connectivity
                         if (e[0]>=0 and e[1]>=0 and e[1]<width and e[0]<height) and not visit_mask[tuple(e)]]

            for i, e in enumerate(neighbors):
                d = W[e] * G[e] + m[cc]
                # d = W[e] + m[cc]
                if d < m[e]:
                    m[e] = d
                if e==tuple(p2[::-1]):
                    found = True

            visit_mask[cc] = True
            m_mask = np.ma.masked_array(m, visit_mask)
            cc = np.unravel_index(m_mask.argmin(), m.shape)

        # import matplotlib.pyplot as plt
        # plt.figure()
        # plt.imshow(m)
        # plt.show()

        # 开始反向追溯
        print('开始回溯')
        # 搜索到的道路中心点，但不包含首尾2个点
        roadcenters = []
        minP = tuple(p2[::-1])
        while True:
            neighbors = [tuple(e) for e in np.asarray(minP) - connectivity
                         if visit_mask[tuple(e)] and (e[0] >= 0 and e[1] >= 0 and e[1] < width and e[0] < height)]
            minU = np.inf
            for i, e in enumerate(neighbors):
                if minU > m[e]:
                    minU = m[e]
                    minP = e
            if minP == tuple(p1[::-1]): # 找到起点
                break

            if minP[::-1] in roadcenters:
                print(minP[::-1])
                print(roadcenters)
                raise Exception('出现回路，提取失败')
                return

            roadcenters.append(minP[::-1])

        roadcenters.reverse()
        return roadcenters

    @numba.jit()
    def geodesic_distance_2_inner(self, W, G, U, M, orderlist, p1, p2):
        x1, y1 = p1
        x2, y2 = p2
        height, width = W.shape
        stop = False
        while not stop:
            node = orderlist.pop_front()
            if node is None:
                break

            x, y = node.point
            minU = node.v

            M[y, x] = 2  # 标记minP为ALIVE

            for i in range(-1, 2):
                for j in range(-1, 2):
                    if i == 0 and j == 0:
                        continue
                    xx = x + i
                    yy = y + j
                    if xx < 0 or yy < 0 or xx >= width or yy >= height:
                        continue
                    if xx == x2 and yy == y2:
                        stop = True
                        break

                    if M[yy, xx] == 0:  # FARAWAY
                        M[yy, xx] = 1   # ACTIVE
                        w = W[yy, xx]
                        g = G[yy, xx]
                        f = minU + w * g

                        node = Node([xx, yy], f)
                        orderlist.add(node)
                        U[yy, xx] = f

                if stop:
                    break
        # 开始反向追溯
        print('开始回溯')
        # 搜索到的道路中心点，但不包含首尾2个点
        roadcenters = []
        minP = p2
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

            if minP[0] == x1 and minP[1] == y1:  # 回到起点
                break
            if minP in roadcenters:
                raise Exception('出现回路，提取失败')
                return


            roadcenters.append(minP)

        roadcenters.reverse()
        return roadcenters

    def geodesic_distance_2(self, W, G, p1, p2):
        '''
        获得两点之间的测地距离
        使用有序列表，占更多空间，但时间效率更高 （可以作为写论文的点）
        这个算法可能有点问题，在更新过程中没有更新OPEN状态的节点（
            解决：
            得益于梯度图的引入，使得先OPEN的节点一定会比后OPEN的节点的势能低，
            因为它都是由当前的最小值加上当前坐标下的显著图与梯度的成绩，同一个节点越往后（如果执行更新）它的势能，其值一定是越大）
        :param W: 图像场
        :param G: 梯度场
        :param p1: 起点
        :param p2: 终点
        :return: 返回俩之间的路径像素坐标队列
        '''
        x1, y1 = p1
        x2, y2 = p2

        height, width = W.shape
        U = np.ones(W.shape, np.float)  # 最小能量图
        M = np.zeros(W.shape, np.uint8)  # 每点标记类型为:ALIVE=2, ACTIVE=1, FARAWAY=0;【对应FMM的CLOSE, OPEN, FARAWAY】
        # 初始化最小能量图
        U = U * float('inf')
        # 初始化起始点p1的信息, U(p1)=0;
        U[y1, x1] = 0
        orderlist = OrderList()
        # 插入队列
        node = Node(p1, 0)
        orderlist.add(node)

        roadcenters = self.geodesic_distance_2_inner(W,G,U,M,orderlist,p1,p2)
        # import matplotlib.pyplot as plt
        # plt.figure()
        # U[M == 0] = np.max(U[M!=0])
        # print(np.max(U))
        # plt.imshow(U, cmap='hot')
        # plt.show()

        roadcenters_fit = meanSmoothCurve(roadcenters, 20)

        return roadcenters_fit
        pass

    def geodesic_distance_4(self, W, G, p1, p2):
        '''
        获得两点之间的测地距离
        采用扫描法进行能量图计算，直接获得能量场，然后从终点回溯到起点

        :param W: 图像场
        :param G: 梯度场
        :param p1: 起点
        :param p2: 终点
        :return: 返回俩之间的路径像素坐标队列
        '''
        x1, y1 = p1
        # 先计算欧式距离
        Euclidean_Distance = np.zeros_like(W)
        getEuclidianDistance(Euclidean_Distance, p1)

        # 距离场梯度图（用于线长约束项）
        GG = getGradientMap(Euclidean_Distance)
        # 距离场梯度的梯度图（用于线光滑度约束项）
        GG2 = getGradientMap(GG)
        #
        # import matplotlib.pyplot as plt
        # plt.figure()
        # plt.imshow(G, 'hot')
        # plt.show()
        # plt.figure()
        # plt.subplot(221)
        # plt.imshow(GG, 'hot')
        # plt.title("GG1")
        # plt.subplot(222)
        # plt.imshow(GG2, 'hot')
        # plt.title("GG2")
        # plt.subplot(223)
        # plt.imshow(Euclidean_Distance, 'hot')
        # plt.title("Euclidean_Distance")
        # plt.show()
        # print("g.max", G.max())
        # print("gg.max", GG.max())
        # print("gg2.max", GG2.max())

        W = np.max(W) - W + self.epsilon
        Cost = self.a * W  + self.b * GG + self.c * GG2

        # Cost = self.a * W * G # + self.b * GG + self.c * GG2
        # Cost = self.a * G #+ self.b * GG + self.c * GG2

        U = np.ones(W.shape, np.float)  # 最小能量图
        U[:] = 1e6
        U[y1, x1] = 0
        U = GDT(U, Cost, 500)

        # 查看距离场
        # region 2d
        # import matplotlib.pyplot as pl
        # fig = pl.figure()
        # hi = U.max()
        # n = 20
        # trs = np.linspace(0, hi, n + 1)
        # duration = 0.5
        # t = 1
        # t /= duration
        # tmp = np.ones_like(self.roadcenters.image, np.uint8) * 255
        # pl.imshow(U,'rainbow')
        # pl.colorbar()
        # pl.axis('off')
        # # pl.contour(U, trs + t * hi / n, origin='image', colors='black')
        # pl.show()
        # endregion

        #regin 3d
        # import matplotlib.pyplot as plt
        # from mpl_toolkits.mplot3d import Axes3D
        # fig = plt.figure()
        # # 创建3d图形的两种方式
        # # ax = Axes3D(fig)
        # ax = fig.add_subplot(111, projection='3d')
        # # X, Y value
        # height, width = U.shape
        # X = np.arange(0, width, 1)
        # Y = np.arange(0, height, 1)
        # X, Y = np.meshgrid(X, Y)  # x-y 平面的网格
        #
        # # rstride:行之间的跨度  cstride:列之间的跨度
        # # rcount:设置间隔个数，默认50个，ccount:列的间隔个数  不能与上面两个参数同时出现
        # # vmax和vmin  颜色的最大值和最小值
        # ax.plot_surface(X, Y, U, rstride=1, cstride=1, cmap=plt.get_cmap('rainbow'))
        # # zdir : 'z' | 'x' | 'y' 表示把等高线图投射到哪个面
        # # offset : 表示等高线图投射到指定页面的某个刻度
        # # ax.contourf(X, Y, U, zdir='z', offset=-2)
        # # 设置图像z轴的显示范围，x、y轴设置方式相同
        # # ax.set_zlim(-2, 2)
        # plt.show()


        #endregion


        # ax = pl.gca()
        # ax.set_facecolor('white')
        # ax.xaxis.set_ticklabels([])
        # ax.yaxis.set_ticklabels([])
        # pl.tight_layout()
        # pl.grid()
        # fig.canvas.draw()
        # img = np.array(fig.canvas.renderer._renderer)[..., :3]
        # pl.close()
        # pl.imshow(img)
        # pl.show()

        # 开始反向追溯
        # print('开始回溯')
        # 搜索到的道路中心点，但不包含首尾2个点
        return traceBack(U, p1, p2)
        pass

    def geodesic_distance_3(self, W, G, p1, p2):
        '''
        未使用有序列表，节省空间，但时间效率低
        geodesic_distance_2 逻辑有问题，没有更新OPEN节点
        :param W:
        :param G:
        :param p1:
        :param p2:
        :return:
        '''
        # constant
        ALIVE = 2; ACTIVE = 1; FARAWAY = 0

        x1, y1 = p1
        height, width = W.shape
        # step_size = 1/width if width>height else 1/height
        step_size=1.0
        mask = np.zeros_like(W, dtype=np.bool)
        m = np.ma.masked_array(np.ones_like(W), mask) #当掩码的元素为False时，关联数组的相应元素有效，并且被称为未屏蔽。 当掩码的元素为True时，相关数组的相应元素被称为被屏蔽（无效）。
        mask = m.mask
        visit_mask = mask.copy()  # mask visited cells
        m = m * np.inf # 初始化距离为无穷大
        m[y1, x1] = 0
        s = np.zeros_like(W, dtype=np.int) # 状态，初始化为FARAWAY
        father = np.zeros((height, width, 2), dtype=np.int) # 父节点，初始化为0
        s[y1, x1] = ALIVE # 设置已经关闭
        nei = [(1,0),(-1,0),(0,1),(0,-1)] # 四邻域
        cc = np.unravel_index(m.argmin(), m.shape)  # current_cell(row, cell)

        found = False #是否碰到了P2
        while not found:
            neighbors = [tuple(e) for e in np.asarray(cc) - nei
                         if (e[0]>=0 and e[1]>=0 and e[1]<width and e[0]<height) and not visit_mask[tuple(e)]]

            for i, e in enumerate(neighbors):
                yy, xx = e
                f = [0,0] # current father
                P = step_size * W[e] * G[e]
                if P==0:
                    raise Exception("势能错误, P=0")

                # neighbors values
                a1 = float('inf')
                if yy<height-1:
                    a1 = m[yy + 1, xx]
                    f[0] = (yy+1) * width + xx
                if yy>0 and m[yy -1, xx]<a1:
                    a1 = m[yy - 1, xx]
                    f[0] = (yy - 1) * width + xx

                a2 = float('inf')
                if xx < width - 1:
                    a2 = m[yy, xx+1]
                    f[1] = yy * width + xx + 1
                if xx > 0 and m[yy, xx-1] < a2:
                    a2 = m[yy, xx-1]
                    f[1] = yy * width + xx - 1

                if a1>a2:
                    tmp = a1
                    a1 = a2
                    a2 = tmp
                    f = f[::-1]

                if P > a2 - a1: # delta 大于0
                    delta = 2 * P ** 2 - (a2 - a1) ** 2
                    A1 = (a1 + a2 + math.sqrt(delta)) / 2
                else: # 否则用dijkstra方法, 沿着格子走, 公式为:max | ux, uy |= 1 / Fijk
                    A1 = a1 + P
                    f[1] = 0 # 将第2个父节点设为0

                if s[e] == ALIVE: # 闭集不用更新
                    pass
                elif s[e] == ACTIVE: # 开集才更新
                    if A1 < m[e]:
                        m[e] = A1
                        father[yy, xx, :] = f
                elif s[e] == FARAWAY:
                    m[e] = A1
                    father[yy, xx, :] = np.asarray(f)
                    s[e] = ACTIVE

                if e == tuple(p2[::-1]):
                    found = True

            visit_mask[cc] = True
            m_mask = np.ma.masked_array(m, visit_mask)
            cc = np.unravel_index(m_mask.argmin(), m.shape)

        # import matplotlib.pyplot as plt
        # plt.figure()
        # plt.imshow(m)
        # plt.show()

        # 开始反向追溯
        print('开始回溯')
        # 搜索到的道路中心点，但不包含首尾2个点
        roadcenters = []
        '''
        minP = tuple(p2[::-1])        
        connectivity = [(i, j) for i in [-1, 0, 1] for j in [-1, 0, 1] if (not (i == j == 0))]
        while True:
            neighbors = [tuple(e) for e in np.asarray(minP) - connectivity
                         if visit_mask[tuple(e)] and (e[0] >= 0 and e[1] >= 0 and e[1] < width and e[0] < height)]
            minU = np.inf
            for i, e in enumerate(neighbors):
                if minU > m[e]:
                    minU = m[e]
                    minP = e
            if minP == tuple(p1[::-1]): # 找到起点
                break

            if minP[::-1] in roadcenters:
                print(minP[::-1])
                print(roadcenters)
                print(m[e[0]-5:e[0]+6, e[1]-5:e[1]+6])
                raise Exception('出现回路，提取失败')
                return

            roadcenters.append(minP[::-1])
        '''
        p = p2[::-1] #(y,x)
        p = father[p[0], p[1], 0] #前序点的index
        p = np.unravel_index(p, m.shape) #转成坐标（y,x)
        while((p!=tuple(p1[::-1]))):# 找到起点
            roadcenters.append(p[::-1]) #canvas使用（x,y）坐标
            p = father[p[0], p[1], 0] # 下一个前序点的index
            p = np.unravel_index(p, m.shape) # 转成(y, x）坐标

        roadcenters.reverse()
        return roadcenters

    def softPDE(self, W, r):

        # print('saliency map:', W.min(), W.max())
        # 求道路高斯概率
        # W = StdGaussian(W / np.max(W))
        # print('StdGaussian:', W.min(), W.max())
        W = W / np.max(W)
        # 求soft PDE
        '''
            算法：对于每个像素x，以x为中心的r*r的矩形内每个像素y，求 Gausssian(W(y)) * sum( Gausssian(distance(x,y))) / (r*r)
            1.可以将距离高斯直接做成以距离滤波器，反复使用
        '''

        gaussianMap = getGaussianMask(2 * r + 1, 2 * r + 1)
        # 卷积操作
        W = cv2.filter2D(W, -1, gaussianMap)
        # print('softPDE:', W.min(), W.max())

        # import matplotlib.pyplot as plt
        # tmp = W / np.max(W)
        # plt.figure()
        # plt.imshow(tmp, "rainbow")
        # plt.colorbar()
        # plt.axis('off')
        # plt.show()

        # 势能取反
        # W = np.max(W) - W + self.epsilon
        # W = 1 / (W + self.epsilon)
        # todo:为什么不管使用最大中心概率还是用最小概率都能得到正确结果？答:使用了PDE的soft梯度当代价
        return W

    def PDE(self, W, r):
        # 求hard PDE
        # 直方图均衡
        # W = cv2.equalizeHist(W)
        # 转成二值图, 大津阈值
        # th, W = cv2.threshold(W,20, 1, cv2.THRESH_BINARY_INV)

        r = r if r % 2 else r + 1
        W = cv2.GaussianBlur(W, (r,r), 0)

        # import matplotlib.pyplot as plt
        # plt.figure()
        # plt.imshow(W, cmap='gray')
        # plt.show()


        th, W = cv2.threshold(W,0,1,cv2.THRESH_BINARY+cv2.THRESH_OTSU)
        # th, W = cv2.threshold(W, th // 2 , 1, cv2.THRESH_BINARY_INV)

        # import matplotlib.pyplot as plt
        # plt.figure()
        # plt.imshow(W, cmap='gray')
        # plt.show()

        '''
            算法：对于每个像素x，以x为中心的r*r的矩形内每个像素y，求 Gausssian(W(y)) * sum( Gausssian(distance(x,y))) / (r*r)
            1.可以将距离高斯直接做成以距离滤波器，反复使用
        '''
        gaussianMap = getGaussianMask(2*r+1,2*r+1)
        W = W.astype(np.float)  # 重要，否则下面的卷积会溢出而被截断
        # 卷积操作
        W = cv2.filter2D(W, -1, gaussianMap)

        # import matplotlib.pyplot as plt
        # plt.figure()
        # plt.imshow(W, cmap='jet')
        # plt.show()
        # exit()

        # 势能取反
        # W = np.max(W)-W+self.epsilon
        return W

    def detectRoadCenters(self, p1, p2, r):
        # 道路显著图
        W = np.array(self.roadcenters.roadSaliencyMap, dtype=np.uint8)
        # print(W.shape)
        # W = self.PDE(W, r) # 这个需要二值化，阈值不容易设置，稳定性更差（这里可以作为改进的点，写论文，而且本文不需要两次测地距离，效率更好）
        W = self.softPDE(W, r) # 这个直接使用显著图定义道路中心概率，鲁棒性更好, r:种子点圆形模板半径（较小的那个）

        # 转成热图，放入roadcenters，便于观察调试
        roadCenterMap = (W-np.min(W)) * 255.0 / (np.max(W)-np.min(W))

        self.roadCenterProb = roadCenterMap.copy() # 保留下来为了后面的自动种子点调精

        roadCenterMap = roadCenterMap.astype(np.uint8)
        roadCenterMap = cv2.applyColorMap(roadCenterMap, cv2.COLORMAP_JET)
        roadCenterMap = cv2.cvtColor(roadCenterMap, cv2.COLOR_BGR2RGB)
        self.roadcenters.roadCenterMap = Image.fromarray(roadCenterMap)
        # G = np.array(self.roadcenters.morphGradeMap)  # 梯度图(该图在计算自适应圆形模板时已经计算）
        # G = np.array(G) + self.epsilon

        # 重新对PDE做梯度图
        G = getGradientMap(W) + self.epsilon

        # import matplotlib.pyplot as plt
        # plt.figure()
        # plt.imshow(G, 'hot')
        # plt.show()

        # pts = self.geodesic_distance(W, G, p1, p2) # 未使用有序列表，节省空间，但时间效率低
        # pts = self.geodesic_distance_2(W, G, p1, p2) # 使用有序列表，占更多空间，但时间效率更高 （可以作为写论文的点）
        # pts = self.geodesic_distance_3(W, G, p1, p2)  # 未使用有序列表，节省空间，但时间效率低
        pts = self.geodesic_distance_4(W, G, p1, p2)  # 你用扫描法，并采用numba提速

        return pts

    def searchRoadCenterPoints(self):
        # 取最后两个种子点
        pre_seed, cur_seed = self.getPreCurSeed()

        if pre_seed is None or cur_seed is None:
            print("至少需要两个种子点--searchRoadCenterPoints")
            return

        p1, p2 = pre_seed.point, cur_seed.point

        if self.directLinking:
            # s = directLine(p1, p2)
            s = [p1, p2]
        else:
            # 根据前后两个种子点生成道路显著图（已改成只看前一个种子点）
            self.genRoadSaliencyMap(pre_seed, cur_seed)
            # 搜索两个种子点之间的其他道路种子点，放入roadcenters,
            # 以小圆为准
            r = cur_seed.radius if pre_seed.radius > cur_seed.radius else pre_seed.radius
            s = self.detectRoadCenters(p1, p2, r) # s不包含起始和终止种子点
            s = self.polyFitting(p1, p2, s) # s包含起始和终止种子点

        '''    
        将多折线拟合的控制点放入集合
        '''
        for p in s:
            if (p[0] == p1[0] and p[1]==p1[1]) or (p[0] == p2[0] and p[1] == p2[1]):
                continue
            c = Marker(point=p, color='red', thick=2, tags='temproadcenter')
            self.roadcenters.insert(-1, c)

    def tracking(self, point):
        '''
        尝试用上次seed和当前的point，提取道路中心点（用蓝色圆形显示）
        :param point:
        :return:
        '''
        # 不在有效范围内不添加点标签
        rx, ry = point
        if rx < 0 or ry < 0 or rx > self.roadcenters.width or ry > self.roadcenters.height:
            return

        # 创建一个临时种子点
        tempseed = Marker(point=[rx, ry], marker='o', color='blue', thick=4, tags='tempseed')
        self.shiftAndOptimizeCircleTemplate(tempseed)
        self.roadcenters.append(tempseed)
        # 搜索栈顶两个相邻种子点的道路圆
        self.searchRoadCenterPoints()
        self.showRoadCenterLine()

    # 控制点调精
    def refineAutoSeed(self, p1, p2):
        '''
        调精自动提取的道路种子点（折线拟合控制点），调整p2的位置，搜索方向为向量p1p2的垂直方向，方位为向量长度
        :param p1: 直线段的第一个控制点
        :param p2: 直线段的第二个控制点
        :return: 调整后的p2
        '''

        # 中点的列向量
        if p1[0] == p2[0]:
            theta = math.pi / 2
        else:
            theta = math.atan((p2[1] - p1[1]) / (p2[0] - p1[0]))

        # 坐标系转化矩阵
        transMatrix = np.array([[math.cos(theta), math.sin(theta)], [-math.sin(theta), math.cos(theta)]])
        invTransMatrix = np.linalg.pinv(transMatrix)
        # p2在转换后的新坐标中的坐标
        pp2 = np.matmul(transMatrix, np.array(p2).T)
        # print('pp2.shape', pp2.shape)

        distance =math.sqrt((p2[0] - p1[0])**2 + (p2[1] - p1[1])**2)
        # 垂线的长度
        PL = np.ceil(distance // 2).astype(np.int)

        height, width = self.roadCenterProb.shape
        maxRoadCenterProb = 0
        optimal_point = None

        # tmp=[]
        for k in range(0, 2 * PL + 1):
            # 偏移的中垂线坐标
            p2_trans_shift = pp2.copy()
            p2_trans_shift[1] = p2_trans_shift[1] + k - PL # 偏移转换后的y坐标

            # 转回原始坐标
            p2_shift = np.matmul(invTransMatrix, p2_trans_shift).astype(np.int)
            # tmp.append(p2_shift)
            if p2_shift[0] < 0 or p2_shift[1] < 0 or p2_shift[0] >= width or p2_shift[1] >= height:
                continue

            if maxRoadCenterProb < self.roadCenterProb[p2_shift[1], p2_shift[0]]:
                maxRoadCenterProb = self.roadCenterProb[p2_shift[1], p2_shift[0]]
                optimal_point = p2_shift

        # import matplotlib.pyplot as plt
        # plt.figure()
        # plt.imshow(self.roadcenters.image)
        # plt.plot(np.array([p1,p2])[:,0],np.array([p1,p2])[:,1], 'b-')
        # plt.plot(np.array(tmp)[:,0],np.array(tmp)[:,1], 'ro')
        # plt.plot(optimal_point[0], optimal_point[1], 'go')
        # plt.show()

        return optimal_point.tolist()

    def isWiggle(self, auto_seeds, index, theta1):
        if index<3:
            return True
        elif index+3>=len(auto_seeds):
            return True
        else:
            q1 = auto_seeds[index-3]
            q2 = auto_seeds[index-2]
            q3 = auto_seeds[index-1]
            v1 = [q1[0] - q3[0], q1[1] - q3[1]]
            p1 = auto_seeds[index + 1]
            p2 = auto_seeds[index + 2]
            p3 = auto_seeds[index + 3]
            v2 = [p3[0] - p1[0], p3[1] - p1[1]]
            # https://www.zhihu.com/question/410509747/answer/1368486832
            theta2 = 180 * np.arccos((v1[0] * v2[0] + v1[1] * v2[1]) / (np.sqrt(v1[0] ** 2 + v1[1] ** 2) * np.sqrt(v2[0] ** 2 + v2[1] ** 2))) / np.pi
            if theta2>theta1:
                return True
        return False

    def polyFitting(self, pre_seed, cur_seed, s):
        # 开始分段调精
        """
         自动获得道路种子点,获得方法为：
            1.按一定距离分成多个段落（去除第一段和最后一段）
            2.计算每个段中的道路中心概率最大的点，
            3.求这些点的中心概率平均值
            4.从这些点中去除中心概率低于平均值50%的种子点
            4.1 若两个相邻两点的距离过短，则删除道路中心概率较低的那个点
            5.对于每个种子点做垂直方向的偏移，获得该点垂直方向上的最大的中心概率位置，并更新该种子点的坐标
            6.直线连接这些调整后的道路种子点
        @:param s: 测地线上的每个点坐标
        @:param pre_seed: 起始种子点
        @:param cur_seed: 终止种子点
        @:return 返回所有多边形的控制点，其中第一个和最后一个分别是起始和终止种子点
        """
        # 调试
        # import matplotlib.pyplot as plt
        # ns = np.array(s)
        # plt.figure()
        # plt.imshow(self.roadcenters.image)
        # plt.plot(ns[:,0], ns[:,1], 'b-')
        # plt.savefig('polyfitting-0')
        # plt.pause(1)

        # 获得每段道路中心概率最大的点
        auto_seeds = [pre_seed, cur_seed] # s仅包含测地线上的点，不含前后人工种子点
        best_seed_in_segment = None
        maxRoadCenter_in_segment = 0
        for i in range(self.th_distance, len(s) - self.th_distance + 1, 1):  # 去除第一段和最后一段
            p = s[i]
            if i % self.th_distance == 0:
                if best_seed_in_segment is not None:
                    auto_seeds.insert(-1, best_seed_in_segment)
                    maxRoadCenter_in_segment = 0

            if maxRoadCenter_in_segment < self.roadCenterProb[p[1], p[0]]:  # 点以x,y形式存储，即col, row
                maxRoadCenter_in_segment = self.roadCenterProb[p[1], p[0]]
                best_seed_in_segment = p

        # 调试-所有控制点
        # nauto_seeds = np.array(auto_seeds)
        # plt.plot(nauto_seeds[:,0],nauto_seeds[:,1], 'ro', markersize=6)
        # plt.savefig('polyfitting-1')
        # plt.pause(1)

        # 删除中心概率低于平均值50%的种子点
        avgRoadCenterProb = 0
        for p in auto_seeds:
            avgRoadCenterProb += self.roadCenterProb[p[1], p[0]]
        avgRoadCenterProb /= len(auto_seeds)
        # print('avgRoadCenterProb=', avgRoadCenterProb)

        tmp = auto_seeds.copy()
        for p in tmp:
            if self.roadCenterProb[p[1], p[0]] < avgRoadCenterProb * 0.5:
                auto_seeds.remove(p)
        del tmp

        # 若相邻两个控制点过近，则删除道路中心概率较低的那个
        i = 0
        while (i < len(auto_seeds) - 2):
            p1 = auto_seeds[i]
            p2 = auto_seeds[i + 1]
            if np.sqrt((p1[0] - p2[0]) ** 2 + (p1[1] - p2[1]) ** 2) < self.th_distance:
                if self.roadCenterProb[p1[1], p1[0]] < self.roadCenterProb[p2[1], p2[0]]:
                    auto_seeds.remove(p1)
                else:
                    auto_seeds.remove(p2)
            else:
                i += 1

        # 调试-滤除异常控制点后
        # nauto_seeds = np.array(auto_seeds)
        # plt.plot(nauto_seeds[:, 0], nauto_seeds[:, 1], 'go')
        # plt.savefig('polyfitting-2')
        # plt.pause(1)

        # 种子点调精
        # 设P1P2是相邻两个种子点，则求向量P1P2的垂直方向d，在方向d上查找最大道路中心概率坐标，查找范围为向量P1P2的长度
        for i in range(len(auto_seeds) - 2):  # 第一个和最后一个点不需要调精，因为通过圆形模板已经调整到道路中心
            p1 = auto_seeds[i]
            p2 = auto_seeds[i + 1]
            p2 = self.refineAutoSeed(p1, p2)
            auto_seeds[i + 1] = p2

        # 调试-控制点调精后
        # nauto_seeds = np.array(auto_seeds)
        # plt.plot(nauto_seeds[:, 0], nauto_seeds[:, 1], 'g-')
        # plt.plot(nauto_seeds[:, 0], nauto_seeds[:, 1], 'b+', markersize=12)
        # plt.savefig('polyfitting-3')
        # plt.pause(1)

        # 去除角度过小的控制点
        while(True):
            chg = False
            i = 1
            while (i < len(auto_seeds) - 1):
                p0 = auto_seeds[i]
                p1 = auto_seeds[i - 1]
                p2 = auto_seeds[i + 1]
                v1 = [p1[0] - p0[0], p1[1] - p0[1]]
                v2 = [p2[0] - p0[0], p2[1] - p0[1]]
                # https://www.zhihu.com/question/410509747/answer/1368486832
                theta = 180 * np.arccos((v1[0] * v2[0] + v1[1] * v2[1]) / (
                            np.sqrt(v1[0] ** 2 + v1[1] ** 2) * np.sqrt(v2[0] ** 2 + v2[1] ** 2))) / np.pi
                if theta < 150 and self.isWiggle(auto_seeds, i, theta):
                    auto_seeds.remove(p0)
                    chg = True
                else:
                    i += 1
            if not chg:
                break

        # 调试-控制点调精后
        # nauto_seeds = np.array(auto_seeds)
        # plt.plot(nauto_seeds[:, 0], nauto_seeds[:, 1], 'r--')
        # plt.savefig('polyfitting-4')
        # plt.pause(1)

        return auto_seeds

    def showRoadCenterLine(self):
        