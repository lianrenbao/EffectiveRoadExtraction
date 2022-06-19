# -*- coding: utf-8 -*-
class Circle():
    point=None # 圆心
    radius = 1 # 半径
    color = None
    tags = None
    thick =None
    index = 0 #无实际意义，辅助绘图时撤销用

    def __init__(self, point, radius=1, color='red', thick=4, tags='seed', index=0):
        self.point = point
        self.radius = radius
        self.index = index
        self.color = color
        self.tags = tags
        self.thick = thick
        self.role = ''
        pass

