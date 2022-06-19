# Node 类
class Node():
    def __init__(self, point, v):
        self.point = point
        self.v = v
        self.next = None

    def getData(self):
        return self.v

    def getNext(self):
        return self.next

    def setNext(self, newnext):
        self.next = newnext

# 有序列表类
class OrderList():
    # 这是一个从小到大排序的列表
    def __init__(self):
        self.head = None
    # 判断链表是否为空，即列表头是否指向None
    def isEmpty(self):
        return self.head == None
    # 添加add()方法
    def add(self, node):
        previous = None
        current = self.head
        stop = False
        while current != None and not stop:
            if current.getData() > node.getData():
                stop = True
            else:
                previous = current
                current = current.getNext()
        if previous == None:
            node.setNext(self.head)
            self.head = node
        else:
            node.setNext(current)
            previous.setNext(node)


    # 添加size()方法，此过程需要遍历链表
    def size(self):
        current = self.head
        count = 0
        while current != None:
            count += 1
            current = current.getNext()
        return count

    # 添加search方法
    def search(self, v):
        current = self.head
        found = False
        while current != None and not found and current.getData() <= v:
            if current.getData() == v:
                found = True
            else:
                current = current.getNext()
        return found

    # 添加remove()方法
    def remove(self, node):
        previous = None
        current = self.head
        found = False
        while not found and not current:
            if current.getNext == node:
                found = True
            else:
                previous = current
                current = current.getNext()
        # 要删除的节点为第一个节点
        if previous == None:
            self.head = current.getNext()
        else:
            previous.setNext(current.getNext())

    # 尾部弹出
    def pop(self):
        previous = None
        current = self.head
        while current != None:
            previous = current
            current = current.getNext()
        previous.setNext(None)
        return previous

    # 对头弹出
    def pop_front(self):
        if not self.isEmpty():
            node = self.head
            self.head = self.head.getNext()
            node.setNext(None)
            return node
        else:
            return None

    # traverse orderedList
    def println(self):
        current = self.head
        lists = []
        while current != None:
            lists.append(current.data)
            current = current.getNext()
        print(lists)