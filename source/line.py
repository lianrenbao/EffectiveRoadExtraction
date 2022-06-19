# -*- coding: utf-8 -*-
class Line():
    LINETYPES = ['solid', 'dashed'] # s : squire
    def __init__(self, p1, p2, linetype='solid', color='red', thick=4, tags='roadline', index=0):
        if linetype not in self.LINETYPES:
            raise RuntimeError('line type must be one of the :' + str(self.LINETYPES))
        self.p1 = p1
        self.p2 = p2
        self.index = index
        self.linetype = linetype
        self.color = color
        self.tags = tags
        self.thick = thick
        self.role = ''
        pass

