class Comm:
    def __init__(self):
        self.times = []
        self.startTime = None
        self.endTime = None
        self.epsilon = 1e-5
        self.testing = False  # 是否在实时跟踪
        self.th_distance = 10
        self.th_radius = 7
        self.directLinkingCnt = 0

    def recTime(self, t):
        self.times.append(t)

    def recDirectLinking(self):
        self.directLinkingCnt +=1