"""
Name: Jeffrey Valentic
Graph ADT and A* Pathfinding
12/15/2020
CSE 331 FS20 (Onsay)
"""

import heapq
import itertools
import math
import queue
import random
import time
from typing import TypeVar, Callable, Tuple, List, Set

import matplotlib.cm as cm
import matplotlib.patches as patches
import matplotlib.pyplot as plt
import numpy as np

T = TypeVar('T')
Matrix = TypeVar('Matrix')  # Adjacency Matrix
Vertex = TypeVar('Vertex')  # Vertex Class Instance
Graph = TypeVar('Graph')    # Graph Class Instance


class Vertex:
    """ Class representing a Vertex object within a Graph """

    __slots__ = ['id', 'adj', 'visited', 'x', 'y']

    def __init__(self, idx: str, x: float = 0, y: float = 0) -> None:
        """
        DO NOT MODIFY
        Initializes a Vertex
        :param idx: A unique string identifier used for hashing the vertex
        :param x: The x coordinate of this vertex (used in a_star)
        :param y: The y coordinate of this vertex (used in a_star)
        """
        self.id = idx
        self.adj = {}             # dictionary {id : weight} of outgoing edges
        self.visited = False      # boolean flag used in search algorithms
        self.x, self.y = x, y     # coordinates for use in metric computations

    def __eq__(self, other: Vertex) -> bool:
        """
        DO NOT MODIFY
        Equality operator for Graph Vertex class
        :param other: vertex to compare
        """
        if self.id != other.id:
            return False
        elif self.visited != other.visited:
            print(f"Vertex '{self.id}' not equal")
            print(f"Vertex visited flags not equal: self.visited={self.visited},"
                  f" other.visited={other.visited}")
            return False
        elif self.x != other.x:
            print(f"Vertex '{self.id}' not equal")
            print(f"Vertex x coords not equal: self.x={self.x}, other.x={other.x}")
            return False
        elif self.y != other.y:
            print(f"Vertex '{self.id}' not equal")
            print(f"Vertex y coords not equal: self.y={self.y}, other.y={other.y}")
            return False
        elif set(self.adj.items()) != set(other.adj.items()):
            diff = set(self.adj.items()).symmetric_difference(set(other.adj.items()))
            print(f"Vertex '{self.id}' not equal")
            print(f"Vertex adj dictionaries not equal:"
                  f" symmetric diff of adjacency (k,v) pairs = {str(diff)}")
            return False
        return True

    def __repr__(self) -> str:
        """
        DO NOT MODIFY
        Represents Vertex object as string.
        :return: string representing Vertex object
        """
        lst = [f"<id: '{k}', weight: {v}>" for k, v in self.adj.items()]

        return f"<id: '{self.id}'" + ", Adjacencies: " + "".join(lst) + ">"

    def __str__(self) -> str:
        """
        DO NOT MODIFY
        Represents Vertex object as string.
        :return: string representing Vertex object
        """
        return repr(self)

    def __hash__(self) -> int:
        """
        DO NOT MODIFY
        Hashes Vertex into a set; used in unit tests
        :return: hash value of Vertex
        """
        return hash(self.id)

    def degree(self) -> int:
        """
        Provides the degree of the current vertex (number of adjacent elements)
        :return: int - length of the adj dict.
        """
        return len(self.adj)

    def get_edges(self) -> Set[Tuple[str, float]]:
        """
        Provides a set of all edges connected to the vertex.
        :return: set ( ID of connected vertex, weight of connection)
        """
        edges = {(i[0], i[1]) for i in self.adj.items()}
        return edges

    def euclidean_distance(self, other: Vertex) -> float:
        """
        Provides the euclidean distance between two vertices
        :param other: The vertex to find the distance to.
        :return: float - The euclidean distance between the two vertices
        """
        dist = math.sqrt(((other.x - self.x)**2) + ((other.y - self.y)**2))
        return dist

    def taxicab_distance(self, other: Vertex) -> float:
        """
        Provides the taxicab distance between two vertices
        :param other: The vertex to find the distance to.
        :return: float - The taxicab distance between the two vertices
        """
        dist = abs(self.x - other.x) + abs(self.y - other.y)
        return dist


class Graph:
    """ Class implementing the Graph ADT using an Adjacency Map structure """

    __slots__ = ['size', 'vertices', 'plot_show', 'plot_delay']

    def __init__(self, plt_show: bool = False, matrix: Matrix = None, csv: str = "") -> None:
        """
        DO NOT MODIFY
        Instantiates a Graph class instance
        :param: plt_show : if true, render plot when plot() is called; else, ignore calls to plot()
        :param: matrix : optional matrix parameter used for fast construction
        :param: csv : optional filepath to a csv containing a matrix
        """
        matrix = matrix if matrix else np.loadtxt(csv, delimiter=',', dtype=str).tolist() if csv else None
        self.size = 0
        self.vertices = {}

        self.plot_show = plt_show
        self.plot_delay = 0.2

        if matrix is not None:
            for i in range(1, len(matrix)):
                for j in range(1, len(matrix)):
                    if matrix[i][j] == "None" or matrix[i][j] == "":
                        matrix[i][j] = None
                    else:
                        matrix[i][j] = float(matrix[i][j])
            self.matrix2graph(matrix)

    def __eq__(self, other: Graph) -> bool:
        """
        DO NOT MODIFY
        Overloads equality operator for Graph class
        :param other: graph to compare
        """
        if self.size != other.size or len(self.vertices) != len(other.vertices):
            print(f"Graph size not equal: self.size={self.size}, other.size={other.size}")
            return False
        else:
            for vertex_id, vertex in self.vertices.items():
                other_vertex = other.get_vertex(vertex_id)
                if other_vertex is None:
                    print(f"Vertices not equal: '{vertex_id}' not in other graph")
                    return False

                adj_set = set(vertex.adj.items())
                other_adj_set = set(other_vertex.adj.items())

                if not adj_set == other_adj_set:
                    print(f"Vertices not equal: adjacencies of '{vertex_id}' not equal")
                    print(f"Adjacency symmetric difference = "
                          f"{str(adj_set.symmetric_difference(other_adj_set))}")
                    return False
        return True

    def __repr__(self) -> str:
        """
        DO NOT MODIFY
        Represents Graph object as string.
        :return: String representation of graph for debugging
        """
        return "Size: " + str(self.size) + ", Vertices: " + str(list(self.vertices.items()))

    def __str__(self) -> str:
        """
        DO NOT MODIFY
        Represents Graph object as string.
        :return: String representation of graph for debugging
        """
        return repr(self)

    def plot(self) -> None:
        """
        DO NOT MODIFY
        Creates a plot a visual representation of the graph using matplotlib
        :return: None
        """
        if self.plot_show:

            # if no x, y coords are specified, place vertices on the unit circle
            for i, vertex in enumerate(self.get_vertices()):
                if vertex.x == 0 and vertex.y == 0:
                    vertex.x = math.cos(i * 2 * math.pi / self.size)
                    vertex.y = math.sin(i * 2 * math.pi / self.size)

            # show edges
            num_edges = len(self.get_edges())
            max_weight = max([edge[2] for edge in self.get_edges()]) if num_edges > 0 else 0
            colormap = cm.get_cmap('cool')
            for i, edge in enumerate(self.get_edges()):
                origin = self.get_vertex(edge[0])
                destination = self.get_vertex(edge[1])
                weight = edge[2]

                # plot edge
                arrow = patches.FancyArrowPatch((origin.x, origin.y),
                                                (destination.x, destination.y),
                                                connectionstyle="arc3,rad=.2",
                                                color=colormap(weight / max_weight),
                                                zorder=0,
                                                **dict(arrowstyle="Simple,tail_width=0.5,"
                                                                  "head_width=8,head_length=8"))
                plt.gca().add_patch(arrow)

                # label edge
                plt.text(x=(origin.x + destination.x) / 2 - (origin.x - destination.x) / 10,
                         y=(origin.y + destination.y) / 2 - (origin.y - destination.y) / 10,
                         s=weight, color=colormap(weight / max_weight))

            # show vertices
            x = np.array([vertex.x for vertex in self.get_vertices()])
            y = np.array([vertex.y for vertex in self.get_vertices()])
            labels = np.array([vertex.id for vertex in self.get_vertices()])
            colors = np.array(
                ['yellow' if vertex.visited else 'black' for vertex in self.get_vertices()])
            plt.scatter(x, y, s=40, c=colors, zorder=1)

            # plot labels
            for j, _ in enumerate(x):
                plt.text(x[j] - 0.03*max(x), y[j] - 0.03*max(y), labels[j])

            # show plot
            plt.show()
            # delay execution to enable animation
            time.sleep(self.plot_delay)

#============== Modify Graph Methods Below ==============#

    def reset_vertices(self) -> None:
        """
        Sets the visited status to False for all vertices in graph
        :return: None
        """
        for i in self.vertices.items():
            i[1].visited = False

    def get_vertex(self, vertex_id: str) -> Vertex:
        """
        Provides the vertex that has ID vertex_id.
        :param vertex_id: The id value of the vertex to find
        :return: Vertex object or None if vertex doesnt exist
        """
        if vertex_id in self.vertices.keys():
            return self.vertices[vertex_id]
        return None

    def get_vertices(self) -> Set[Vertex]:
        """
        Provides a set of all vertices contained in the graph.
        :return: set containing Vertex objects
        """
        verts = {v[1] for v in self.vertices.items()}
        return verts

    def get_edge(self, start_id: str, dest_id: str) -> Tuple[str, str, float]:
        """
        Provides information about an edge between two vertices
        :param start_id: The ID of the vertex to start from
        :param dest_id: The ID of the vertex connected by the desired edge
        :return: Tuple[ start_id, dest_id, weight of edge]
        """
        if start_id in self.vertices.keys():
            start_adj = self.vertices[start_id].adj
            if dest_id in start_adj.keys():
                dest_edg = start_adj[dest_id]
                return start_id, dest_id, dest_edg
        return None

    def get_edges(self) -> Set[Tuple[str, str, float]]:
        """
        Provides a set of all edges contained in the graph.
        :return: Set containing Tuples[ start_id, neighbor_id, weight between them]
        """
        edge_set = set()
        for a in self.vertices.items():
            start_id = a[0]
            start_vert = a[1]
            for b in start_vert.adj.items():
                new_edg = (start_id, b[0], b[1])
                edge_set.add(new_edg)

        return edge_set

    def add_to_graph(self, start_id: str, dest_id: str = None, weight: float = 0) -> None:
        """
        Adds two vertices and a connection between them to the graph.
        :param start_id: ID if first vertex
        :param dest_id: ID of second vertex
        :param weight: Weight of the edge connecting the two vertices
        :return: None
        """
        if self.get_vertex(start_id) is None:
            self.vertices[start_id] = Vertex(start_id)
            self.size += 1
        if dest_id is not None and self.get_vertex(dest_id) is None:
            self.vertices[dest_id] = Vertex(dest_id)
            self.size += 1

        if dest_id is not None:
            start_edg = self.vertices[start_id].adj
            start_edg[dest_id] = weight

    def matrix2graph(self, matrix: Matrix) -> None:
        """
        Converts an N x N matrix to a graph data structure
        :param matrix: The matrix containing vertices and
                       connections to convert to a graph.
        :return: None
        """
        ids = matrix[0]
        num_v = len(matrix[0])
        for i in range(1, num_v):
            for j in range(1, num_v):
                if matrix[i][j] is not None:
                    self.add_to_graph(ids[i], ids[j], matrix[i][j])
                else:
                    self.add_to_graph(ids[i])

    def graph2matrix(self) -> Matrix:
        """
        Converts the current graph to a N x N matrix,
        where N is the number of vertices.
        :return: An NxN array (Matrix) representation of the graph.
        """
        if self.size == 0:
            return None

        matrx = [[0]*(self.size+1) for i in range(self.size+1)]
        matrx[0][0] = None

        keys = list(self.vertices.keys())
        for i in range(1, self.size+1):
            matrx[0][i] = keys[i-1]
            matrx[i][0] = keys[i-1]
            matrx[0][-1] = keys[-1]
            matrx[-1][0] = keys[-1]
            vert1 = self.vertices[keys[i-1]]

            for q in range(1, self.size+1):
                vert2 = self.vertices[keys[q-1]]
                id1 = vert1.id
                id2 = vert2.id
                edge = self.get_edge(id1, id2)

                if edge is None:
                    matrx[i][q] = None
                else:
                    matrx[i][q] = edge[2]

        return matrx

    def bfs(self, start_id: str, target_id: str) -> Tuple[List[str], float]:
        """
        A breadth first search of the graph, starting from vertex
        with start_id, and ending at vertex with target_id.
        :param start_id: The "root" vertex that the search will start from.
        :param target_id: The desired vertex that is being searched for.
        :return: Tuple containing a path (list) with id's of visited nodes, and the total
        weight (distance) of all nodes in the path.
        """
        if start_id not in self.vertices.keys() or target_id not in self.vertices.keys():
            return [], 0

        for i in self.vertices.items():
            i[1].visited = False

        path = {} # {cur : prev}
        queue = []
        start_v = self.get_vertex(start_id)
        start_v.visited = True
        queue.append(start_v)

        if len(start_v.adj) == 0:
            return [], 0

        if target_id in start_v.adj.keys():
            return [start_id, target_id], start_v.adj[target_id]

        notFound = True
        while notFound:
            if len(queue) == 0:
                return [], 0

            s = queue.pop(0)

            if s.id == target_id:
                notFound = False

            if target_id in s.adj.keys():
                end_v = self.get_vertex(target_id)
                end_v.visited = True
                notFound = False
                path[end_v.id] = s.id

            if notFound:
                for i in s.adj.items():
                    new_v = self.get_vertex(i[0])
                    if new_v.visited == False:
                        queue.append(new_v)
                        path[new_v.id] = s.id
                        new_v.visited = True

        cur_id = target_id
        final_path = [target_id]
        dist = 0
        while cur_id != start_id:
            prev_v = path[cur_id]
            final_path.append(prev_v)
            dist += self.vertices[prev_v].adj[cur_id]
            cur_id = prev_v

        final_path.reverse()
        return final_path, dist

    def dfs(self, start_id: str, target_id: str) -> Tuple[List[str], float]:
        """
        A depth first search of the graph, starting from vertex
        with start_id, and ending at vertex with target_id.
        :param start_id: The "root" vertex that the search will start from.
        :param target_id: The desired vertex that is being searched for.
        :return: Tuple containing a path (list) with id's of visited nodes, and the total
        weight (distance) of all nodes in the path.
        """

        if start_id not in self.vertices.keys() or target_id not in self.vertices.keys():
            return [], 0

        def dfs_inner(current_id: str, target_id: str, path: List[str] = [])\
                -> Tuple[List[str], float]:
            """
            Recursively calls itself on vertex connections to the current_id vertex
            in order to build the depth first path from start to target.
            :param current_id: the ID of the current vertex
            :param target_id: the ID of the desired vertex
            :param path: a list containing the ID's of all visited vertices
            :return: Tuple containing a path (list) with id's of visited nodes, and the total
                    weight (distance) of all nodes in the path.
            """
            cur_v = self.get_vertex(current_id)

            if cur_v.visited is False:
                path.append(current_id)

            cur_v.visited = True

            if cur_v is None:
                return None

            for n in cur_v.adj.items():
                new_v = self.get_vertex(n[0])
                if new_v.visited is False:
                    visited[n[0]] = current_id
                    dfs_inner(n[0], target_id, path)
            return path

        path = []
        visited = {start_id: None}
        path = dfs_inner(start_id, target_id, path)

        if path is None or len(path) <= 1 or target_id not in path:
            return [], 0

        cur_id = target_id
        final_path = [target_id]
        dist = 0

        while cur_id != start_id:
            prev_v = visited[cur_id]
            final_path.append(prev_v)
            dist += self.vertices[prev_v].adj[cur_id]
            cur_id = prev_v

        final_path.reverse()
        return final_path, dist

    def a_star(self, start_id: str, target_id: str, metric: Callable[[Vertex, Vertex], float])\
            -> Tuple[List[str], float]:
        """
        Finds the shortest path between vertices with start_id and target_id
        by using the A Star pathfinding algorithm. Uses AStarPriorityQueue
        to handle vertex priority. Uses a combination of shortest distance
        to the next vertex, and the distance between the next vertex and the
        target vertex to determine priority.
        :param start_id: ID of the vertex to start from
        :param target_id: ID of the desired destination vertex
        :param metric: A heuristic to determine the distance between the
                        current vertex and the destination vertex.
        :return: Tuple containing a path (list) with id's of visited nodes, and the total
        weight (distance) of all nodes in the path.
        """

        def create_path(path, id_start, id_end):
            cur_id = id_end
            final_path = [id_end]
            dist = 0

            while cur_id != id_start:
                prev_v = path[cur_id]
                final_path.append(prev_v)
                dist += self.vertices[prev_v].adj[cur_id]
                cur_id = prev_v
            final_path.reverse()

            return final_path, dist

        v_parents = {}
        v_dist = {}
        apq = AStarPriorityQueue()
        for i in self.vertices.values():
            if i.id == start_id:
                apq.push(0, i)
                v_dist[i.id] = 0
            else:
                apq.push(float('inf'), i)
                v_dist[i.id] = float('inf')

        while apq.empty() is False:
            cur_v = apq.pop()

            if cur_v[1].id == target_id:
                return create_path(v_parents, start_id, target_id)

            for i in cur_v[1].adj.items():
                n_id, n_dist = i[0], i[1]
                step_dist = n_dist + v_dist[cur_v[1].id]

                if step_dist < v_dist[n_id]:
                    v_parents[n_id] = cur_v[1].id
                    v_dist[n_id] = step_dist

                    n_vert = self.vertices[n_id]
                    dest_vert = self.vertices[target_id]

                    priority_score = step_dist + metric(dest_vert, n_vert)
                    apq.update(priority_score, n_vert)

    def make_equivalence_relation(self) -> int:
        """
        Description.
        :return:
        """
        pass


class AStarPriorityQueue:
    """
    Priority Queue built upon heapq module with support for priority key updates
    Created by Andrew McDonald
    Inspired by https://docs.python.org/3/library/heapq.html
    """

    __slots__ = ['data', 'locator', 'counter']

    def __init__(self) -> None:
        """
        Construct an AStarPriorityQueue object
        """
        self.data = []                        # underlying data list of priority queue
        self.locator = {}                     # dictionary to locate vertices within priority queue
        self.counter = itertools.count()      # used to break ties in prioritization

    def __repr__(self) -> str:
        """
        Represent AStarPriorityQueue as a string
        :return: string representation of AStarPriorityQueue object
        """
        lst = [f"[{priority}, {vertex}], " if vertex is not None else "" for
               priority, count, vertex in self.data]
        return "".join(lst)[:-1]

    def __str__(self) -> str:
        """
        Represent AStarPriorityQueue as a string
        :return: string representation of AStarPriorityQueue object
        """
        return repr(self)

    def empty(self) -> bool:
        """
        Determine whether priority queue is empty
        :return: True if queue is empty, else false
        """
        return len(self.data) == 0

    def push(self, priority: float, vertex: Vertex) -> None:
        """
        Push a vertex onto the priority queue with a given priority
        :param priority: priority key upon which to order vertex
        :param vertex: Vertex object to be stored in the priority queue
        :return: None
        """
        # list is stored by reference, so updating will update all refs
        node = [priority, next(self.counter), vertex]
        self.locator[vertex.id] = node
        heapq.heappush(self.data, node)

    def pop(self) -> Tuple[float, Vertex]:
        """
        Remove and return the (priority, vertex) tuple with lowest priority key
        :return: (priority, vertex) tuple where priority is key,
        and vertex is Vertex object stored in priority queue
        """
        vertex = None
        while vertex is None:
            # keep popping until we have valid entry
            priority, count, vertex = heapq.heappop(self.data)
        del self.locator[vertex.id]            # remove from locator dict
        vertex.visited = True                  # indicate that this vertex was visited
        while len(self.data) > 0 and self.data[0][2] is None:
            heapq.heappop(self.data)          # delete trailing Nones
        return priority, vertex

    def update(self, new_priority: float, vertex: Vertex) -> None:
        """
        Update given Vertex object in the priority queue to have new priority
        :param new_priority: new priority on which to order vertex
        :param vertex: Vertex object for which priority is to be updated
        :return: None
        """
        node = self.locator.pop(vertex.id)      # delete from dictionary
        node[-1] = None                         # invalidate old node
        self.push(new_priority, vertex)         # push new node
