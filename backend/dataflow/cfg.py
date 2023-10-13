from backend.dataflow.basicblock import BasicBlock

"""
CFG: Control Flow Graph

nodes: sequence of basic block
edges: sequence of edge(u,v), which represents after block u is executed, block v may be executed
links: links[u][0] represent the Prev of u, links[u][1] represent the Succ of u,
"""


class CFG:
    def __init__(self, nodes: list[BasicBlock], edges: list[(int, int)]) -> None:
        self.nodes = nodes
        self.edges = edges

        self.links = []

        for i in range(len(nodes)):
            self.links.append((set(), set()))

        for (u, v) in edges:
            self.links[u][1].add(v)
            self.links[v][0].add(u)

        """
        You can start from basic block 0 and do a DFS traversal of the CFG
        to find all the reachable basic blocks.
        """

        self.reachable = set()

        def dfs(node):
            self.reachable.add(self.nodes[node])
            for suc in self.links[node][1]:
                if self.nodes[suc] not in self.reachable:
                    dfs(suc)

        dfs(0)

    def isReachable(self, node: BasicBlock):
        return node in self.reachable

    def getBlock(self, index):
        return self.nodes[index]

    def getPrev(self, index):
        return self.links[index][0]

    def getSucc(self, index):
        return self.links[index][1]

    def getInDegree(self, index):
        return len(self.links[index][0])

    def getOutDegree(self, index):
        return len(self.links[index][1])

    def iterator(self):
        return iter(self.nodes)
