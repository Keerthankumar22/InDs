import random
import copy
import numpy as np

class Parameters:
    def __init__(
        self,
        lower_edge,
        upper_edge,
        lower_node,
        upper_node,
        lower_x_pos,
        upper_x_pos,
        lower_y_pos,
        upper_y_pos,
        lower_delay,
        upper_delay,
    ) -> None:
        self.lower_edge = lower_edge
        self.uppper_edge = upper_edge
        self.lower_node = lower_node
        self.upper_node = upper_node
        self.upper_x_pos = upper_x_pos
        self.upper_y_pos = upper_y_pos
        self.lower_x_pos = lower_x_pos
        self.lower_y_pos = lower_y_pos
        self.lower_delay = lower_delay
        self.upper_delay = upper_delay


class Graph:
    def __init__(self, nodes, edges, parameters) -> None:
        lower_edge = parameters.lower_edge
        upper_edge = parameters.uppper_edge
        lower_node = parameters.lower_node
        upper_node = parameters.upper_node
        self.nodes = nodes
        self.edges = list(edges)
        self.neighbours = dict()
        self.node_weights = dict() # CRB
        self.edge_weights = dict() # BandWidth
        self.node_pos = dict()
        self.node_max=dict()
        self.delay = dict()
        self.node_max=dict()
        self.band_max = dict()
        self.parameters = parameters
        # self.congestion_r = 0.9
        
        for a, b in edges:
            self.edge_weights[(a, b)] = int(np.random.uniform(lower_edge, upper_edge))
            self.edge_weights[(b, a)] = self.edge_weights[(a, b)]
            self.delay[(a, b)] = int(np.random.uniform(parameters.lower_delay, parameters.upper_delay))
            self.delay[(b, a)] = self.delay[(a, b)]

        for i in range(self.nodes):
            self.node_weights[i] = int(np.random.uniform(lower_node, upper_node))

            # Since the initial CRB is the max CRB of the node
            self.node_max[i]=self.node_weights[i]
            l = list()
            l.append(int(np.random.uniform(parameters.lower_x_pos, parameters.upper_x_pos)))
            l.append(int(np.random.uniform(parameters.lower_y_pos, parameters.upper_y_pos)))
            self.node_pos[i] = tuple(l)
        for i in range(self.nodes):
            self.neighbours[i] = set()
            for a, b in self.edges:
                if int(a) == i:
                    self.neighbours[i].add(b)
        

    #This function calculates the maximum bandwidth for each node
    # It is equal to sum of bandwidths of the neighbouring node
    def get_max_bandwidth(self):
     
        self.band_max=dict()
        for a in range(self.nodes):
            self.band_max[str(a)]=0
            for b in self.neighbours[a]:
                self.band_max[str(a)]+=self.edge_weights[(str(a),b)]
        # print("max bandwidth ",self.band_max)

    def findPathsCongestion(self, s, d, visited, path, all_paths, weight):
        visited[int(s)] = True
        path.append(s)
        if s == d:
            all_paths.append(path.copy())
        else:
            for i in self.neighbours[int(s)]:
                if visited[int(i)] == False and self.edge_weights[(s,i)] >= weight:
                    if weight/self.edge_weights[(s,i)] <= 0.9:
                        self.findPaths(i, d, visited, path, all_paths, weight)

        path.pop()
        visited[int(s)] = False

    def findPaths(self, s, d, visited, path, all_paths, weight):
        visited[int(s)] = True
        path.append(s)
        if s == d:
            all_paths.append(path.copy())
        else:
            for i in self.neighbours[int(s)]:
                if visited[int(i)] == False and self.edge_weights[(s,i)] >= weight:
                    
                    self.findPaths(i, d, visited, path, all_paths, weight)

        path.pop()
        visited[int(s)] = False


    def findPathFromSrcToDstCongestion(self, s, d, weight):
        all_paths = []
        visited = [False] * (self.nodes)
        path = []
        self.findPathsCongestion(s, d, visited, path, all_paths, weight)
        if all_paths == []:
            return []
        else:
            return sorted(all_paths, key=len)
        
    def findPathFromSrcToDst(self, s, d, weight):
        all_paths = []
        visited = [False] * (self.nodes)
        path = []
        self.findPaths(s, d, visited, path, all_paths, weight)
        if all_paths == []:
            return []
        else:
            return sorted(all_paths, key=len)

    def BFS(self, src, dest, v, pred, dist, weight):
        queue = []
        visited = [False for i in range(v)]
        for i in range(v):
            dist[i] = 1000000
            pred[i] = -1
        visited[int(src)] = True
        dist[int(src)] = 0
        queue.append(src)
        while len(queue) != 0:
            u = queue[0]
            queue.pop(0)
            for i in self.neighbours[int(u)]:
                if visited[int(i)] == False and self.edge_weights[(u, i)] >= weight:
                    visited[int(i)] = True
                    dist[int(i)] = dist[int(u)] + 1
                    pred[int(i)] = u
                    queue.append(i)
                    if i == dest:
                        return True

        return False

    def findShortestPath(self, s, dest, weight):
        v = self.nodes
        pred = [0 for i in range(v)]
        dist = [0 for i in range(v)]
        ls = []
        if self.BFS(s, dest, v, pred, dist, weight) == False:
            return ls
        path = []
        crawl = dest
        crawl = dest
        path.append(crawl)

        while pred[int(crawl)] != -1:
            path.append(pred[int(crawl)])
            crawl = pred[int(crawl)]

        for i in range(len(path) - 1, -1, -1):
            ls.append(path[i])

        return ls

   
    '''A recursive function to print all paths from 'u' to 'd'.
    visited[] keeps track of vertices in current path.
    path[] stores actual vertices and path_index is current
    index in path[]'''
    def printAllPathsUtil(self, u, d, visited, weight, path, all_path):
        # Mark the current node as visited and store in path
        visited[int(u)]= True
        path.append(u)
        # print(f"{u} {d}")
        # print(path)
        # If current vertex is same as destination, then print
        # current path[]
        
        if u==d:
            all_path.append(copy.deepcopy(path))
        else:
            # If current vertex is not destination
            # Recur for all the vertices adjacent to this vertex
            for i in self.neighbours[int(u)]:
                if visited[int(i)] == False and self.edge_weights[(u, i)] >= weight:
                # if visited[int(i)]== False:
                    self.printAllPathsUtil(i, d, visited, weight, path, all_path)
                     
        # Remove current vertex from path[] and mark it as unvisited
        path.pop()
        visited[int(u)]= False
       
  
  
    # Prints all paths from 's' to 'd'
    def printAllPaths(self, s, d, weight):
        visited =[False]*(self.nodes) # Mark all the vertices as not visited 
        path = []   # Create an array to store a path
        all_path = []   # array to store all the paths
        self.printAllPathsUtil(s, d, visited, weight, path, all_path)  # Call the recursive helper function to print all paths
        return all_path
  
if __name__ == '__main__':
    nodes = 6
    para = Parameters(50, 100, 50, 100, 0, 100, 0, 100, 1, 1)
    edges = [('0','1'), ('1','0'), ('0','2'), ('2','0'), ('0','3'),('3','0'), ['3','2'],['2','3'], ['0','4'],['4','0']
             , ['4','3'],['3','4'], ['4','2'], ['2','4']]
    graph = Graph(nodes, edges, para)
    # res = graph.printAllPaths('0', '4', 0)
    res = graph.findPathFromSrcToDst('0','4',0)
    print(graph.findShortestPath('0','4',0))
    print(f"res {res}")