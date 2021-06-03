class Node(object):
    def __init__(self, name):
        self.name = name
        self.visited = False
        self.neighbors = []


class BFS(object):
    def __init__(self, start):
        self.queue = []
        self.start = start

    def traversal(self):
        self.start.visited = True
        self.queue.append(self.start)

        while self.queue:
            node = self.queue.pop(0)
            yield node

            for n in node.neighbors:
                if not n.visited:
                    n.visited = True
                    self.queue.append(n)

