# -*- coding: utf-8 -*-
class Marker():
    MakerType = ['o', 'x', '+', 's'] # s : squire
    def __init__(self, point, marker='o', color='yellow', thick=4, tags='start', index=0):
        if marker not in self.MakerType:
            raise RuntimeError('Marker type must be one of the :' + str(self.MakerType))
        self.point = point
        self.index = index
        self.marker = marker
        self.color = color
        self.tags = tags
        self.thick = thick
        self.role = ''
        self.radius = 0
        pass

