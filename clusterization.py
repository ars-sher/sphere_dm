# warnings.filterwarnings('ignore')

import numpy as np
import pylab as pl
import sklearn.datasets as ds
from sklearn.cluster import KMeans, DBSCAN
import pandas as pd
import pylab as pl
import numpy as np
import scipy.spatial as ss
import sklearn.cluster as sc
import sklearn.manifold as sm
import sklearn.datasets as ds
import sklearn.metrics as smt
from sklearn.neighbors import NearestNeighbors
import scipy.spatial.distance as dist

from heapq import heapify, heappush, heappop


class PriorityDict(dict):
    """Dictionary that can be used as a priority queue.

    Keys of the dictionary are items to be put into the queue, and values
    are their respective priorities. All dictionary methods work as expected.
    The advantage over a standard heapq-based priority queue is
    that priorities of items can be efficiently updated (amortized O(1))
    using code as 'thedict[item] = new_priority.'

    The 'smallest' method can be used to return the object with lowest
    priority, and 'pop_smallest' also removes it.

    The 'sorted_iter' method provides a destructive sorted iterator.
    """

    def __init__(self, *args, **kwargs):
        super(PriorityDict, self).__init__(*args, **kwargs)
        self._rebuild_heap()

    def _rebuild_heap(self):
        self._heap = [(v, k) for k, v in self.iteritems()]
        heapify(self._heap)

    def smallest(self):
        """Return the item with the lowest priority.

        Raises IndexError if the object is empty.
        """

        heap = self._heap
        v, k = heap[0]
        while k not in self or self[k] != v:
            heappop(heap)
            v, k = heap[0]
        return k

    def pop_smallest(self):
        """Return the item with the lowest priority and remove it.

        Raises IndexError if the object is empty.
        """

        heap = self._heap
        v, k = heappop(heap)
        while k not in self or self[k] != v:
            v, k = heappop(heap)
        del self[k]
        return k

    def __setitem__(self, key, val):
        # We are not going to remove the previous value from the heap,
        # since this would have a cost O(n).

        super(PriorityDict, self).__setitem__(key, val)

        if len(self._heap) < 2 * len(self):
            heappush(self._heap, (val, key))
        else:
            # When the heap grows larger than 2 * len(self), we rebuild it
            # from scratch to avoid wasting too much memory.
            self._rebuild_heap()

    def setdefault(self, key, val):
        if key not in self:
            self[key] = val
            return val
        return self[key]

    def update(self, *args, **kwargs):
        # Reimplementing dict.update is tricky -- see e.g.
        # http://mail.python.org/pipermail/python-ideas/2007-May/000744.html
        # We just rebuild the heap from scratch after passing to super.

        super(PriorityDict, self).update(*args, **kwargs)
        self._rebuild_heap()

    def sorted_iter(self):
        """Sorted iterator of the priority dictionary items.

        Beware: this will destroy elements as they are returned.
        """

        while self:
            yield self.pop_smallest()


ordered_file_dt = np.dtype([('index', np.int32), ('core_dist', np.float64), ('reach_dist', np.float64)])


class OPTICSComputer:
    def __init__(self, eps, min_pts, metric, x):
        self.eps = eps
        self.min_pts = min_pts
        self.metric = metric
        self.x = x
        # number of points
        self.n = x.shape[0]
        # holds ordered by OPTICS points with their core-distance and reachability-distance
        self.ordered_file = np.empty(self.n, dtype=ordered_file_dt)
        # counter of ordered_file occupancy
        self.ordered_file_index = 0
        # point is processed or not
        self.processed = np.zeros(self.n, dtype=bool)
        # reachability distances. We store them twice: in ordered_file and here, to be able to retrieve them at O(1)
        self.reachability_distances = np.empty(self.n, dtype=np.float64)
        self.reachability_distances.fill(np.inf)
        # holds calculated labels
        self.labels = np.empty(self.n, dtype=np.int32)
        # indices of neighbors
        self.indices = []
        # distances of nearest neighbors
        self.distances = []

    def compute(self):
        nn = NearestNeighbors(radius=self.eps, algorithm='auto', metric=self.metric).fit(self.x)
        self.distances, self.indices = nn.radius_neighbors(self.x, self.eps)
        # print "indices:", self.indices
        # print "distances:", self.distances
        for i in xrange(self.n):
            if not self.processed[i]:
                self.expand_cluster_order(i)
        assert self.ordered_file_index == self.n
        # print self.ordered_file
        self.draw_reachability_plot()
        return self.ordered_file

    def expand_cluster_order(self, i):
        core_dist, nbr_indexes, nbr_distances = self.get_core_distance_and_neighbors(i)
        self.processed[i] = True
        self.reachability_distances[i] = np.inf # TODO
        self.ordered_file[self.ordered_file_index] = (i, core_dist, np.inf)
        self.ordered_file_index += 1
        if np.isfinite(core_dist):
            pq = PriorityDict()
            self.update_seeds(pq, nbr_indexes, nbr_distances, i, core_dist)
            while pq:
                next_point = pq.pop_smallest()
                core_dist, nbr_indexes, nbr_distances = self.get_core_distance_and_neighbors(next_point)
                self.processed[next_point] = True
                self.reachability_distances[i] = np.inf # TODO
                self.ordered_file[self.ordered_file_index] = (next_point, core_dist,
                                                              self.reachability_distances[next_point])
                self.ordered_file_index += 1
                if np.isfinite(core_dist):
                    self.update_seeds(pq, nbr_indexes, nbr_distances, next_point, core_dist)

    def get_core_distance_and_neighbors(self, i):
        neighbors_distances = self.distances[i]
        if len(neighbors_distances) < self.min_pts:
            return np.inf, None, None
        neighbors_indexes = self.indices[i]
        min_pts_nearest_distances_indexes = np.argpartition(neighbors_distances, self.min_pts - 1)[:self.min_pts]
        min_pts_nearest_distances = neighbors_distances[min_pts_nearest_distances_indexes]
        core_dist = np.amax(min_pts_nearest_distances)
        return core_dist, neighbors_indexes, neighbors_distances

    def update_seeds(self, pq, nbr_indexes, nbr_distances, center_index, center_core_distance):
        for ni, nd in np.broadcast(nbr_indexes, nbr_distances):
            if not self.processed[ni]:
                i_center_dist = dist.pdist(np.vstack((self.x[ni], self.x[center_index])), self.metric)[0]
                new_reachability_dist = max(center_core_distance, i_center_dist)
                # inf means that ni have never been in queue: we always set reachability_distance while adding
                if not np.isfinite(self.reachability_distances[ni]): # TODO: we can merge it
                    self.reachability_distances[ni] = new_reachability_dist
                    pq[ni] = new_reachability_dist
                else: # ni is already in queue
                    if new_reachability_dist < self.reachability_distances[ni]:
                        self.reachability_distances[ni] = new_reachability_dist
                        pq[ni] = new_reachability_dist

    def draw_reachability_plot(self):
        rds = np.copy(self.ordered_file['reach_dist'])
        m = np.amax(rds[rds != np.inf])
        rds[rds == np.inf] = m
        pl.plot(rds)
        pl.show()


class OPTICS:
    def __init__(self, eps=0.5, min_pts=5, metric='euclidean'):
        assert min_pts > 0
        self.eps = eps
        self.min_pts = min_pts
        self.metric = metric
        # number of points
        self.n = 0

        # holds ordered by OPTICS points with their core-distance and reachability-distance
        self.__ordered_file = np.empty(0, dtype=ordered_file_dt)
        # holds calculated labels
        self.__labels = np.empty(0, dtype=np.int32)

    # accepts features matrix. We do not validate shape of x
    def fit(self, x):
        optics_computer = OPTICSComputer(self.eps, self.min_pts, self.metric, x)
        self.__ordered_file = optics_computer.compute()
        self.__labels = optics_computer.labels # it is empty still, but thus we allocate needed size
        self.n = optics_computer.n
        return self

    def predict(self, xi=0.1, dbscan=False, dbscan_eps=0.5):
        if dbscan:
            self.__extract_dbscan(dbscan_eps)
        else:
            self.__extract(xi)
        return self.__labels

    def fit_predict(self, x, xi=0.1, dbscan=False, dbscan_eps=0.5):
        self.fit(x)
        return self.predict(xi, dbscan, dbscan_eps)

    def is_steep_downward_point(self, i, xi):
        assert i < self.n
        res = self.__ordered_file[i]['reach_dist'] * (1 - xi) >= self.__ordered_file[i + 1]['reach_dist']
        # if res:
        #     print "%s is downward point with xi %s " % (i, xi)
        return res

    def is_steep_upward_point(self, i, xi):
        assert i < self.n
        res = self.__ordered_file[i]['reach_dist'] <= self.__ordered_file[i + 1]['reach_dist'] * (1 - xi)
        # if res:
        #     print "%s is upward point with xi %s" % (i, xi)
        return res

    def is_cluster(self, sda, sua, xi):
        reach_start_index = sda[0]
        reach_start = self.__get_rd(reach_start_index)
        # points to one element righter the end of upward region, intentionally
        reach_end_index = sua[1]
        reach_end = self.__get_rd(reach_end_index)
        print "checking potential cluster %s - %s" % (sda, sua)
        # check 4 and find left&right
        left_i, left, = reach_start_index, reach_start
        # cluster borded can't be outside upward area, so -1
        right_i, right = reach_end_index - 1, self.__get_rd(reach_end_index - 1)
        if left <= right:
            while left < right * (1 - xi):
                right_i -= 1
                right = self.__get_rd(right_i)
                if right_i < left_i:
                    print "potential cluster check failed, no left&right rd intersection; right was higher"
                    return False, None, None  # I wonder whether this is possible
        else:
            while right < left * (1 - xi):
                left_i += 1
                left = self.__get_rd(left_i)
                if left_i > right_i:
                    print "potential cluster check failed, no left&right rd intersection; left was higher"
                    return False, None, None  # I wonder whether this is possible
        right_i += 1  # restore extra element for slicing
        # check 3a
        if right_i - left_i < self.min_pts:
            print "potential cluster check failed, number of elements is less that min_pts"
            return False, None, None
        # check 3b
        inside_cluster_max = np.amax(self.__ordered_file[left_i + 1: right_i - 1]['reach_dist'])
        # if inside_cluster_max > min(reach_start, reach_end) * (1 - xi):
        if inside_cluster_max > min(reach_start, reach_end) * (1 - xi):
            print "potential cluster check %s - %s failed, condition 3b" % (sda, sua)
            print "max is %s, reach_start is %s, reach_end is %s" % (inside_cluster_max, reach_start, reach_end)
            return False, None, None
        return True, left_i, right_i

    def __get_rd(self, i):
        return self.__ordered_file[i]['reach_dist']

    def __extract(self, xi):
        # the last point cannot be steep, so we will handle it separately
        print xi
        # steep down areas
        sdas = set()
        # set of tuples of cluster borders
        clusters = set()
        i = 0
        while i < self.n - 1:
            # start of downward region. On exit, i will point to first not-downward-steep point
            # TODO: consider the last point left
            if self.is_steep_downward_point(i, xi):
                # print "%s is downward point, starting searching the area" % i
                startsteep = i
                endsteep = i + 1  # not including; downward region is [startsteep, endsteep)
                i += 1
                while i < self.n - 1:
                    if not self.is_steep_downward_point(i, 0):  # oh wait, we are going upward
                        break
                    if self.is_steep_downward_point(i, xi): # downward point again, keep going
                        endsteep = i + 1
                    else:  # break, if min_pts consecutive xhi-equal point goes
                        if i - endsteep > self.min_pts:
                            break
                    i += 1
                i = endsteep
                sdas.add((startsteep, endsteep))
                print "found downward region [%s, %s)" % (startsteep, endsteep)
                continue
            # start of upward region. On exit, i will point to first not-upward-steep point
            if self.is_steep_upward_point(i, xi):
                startsteep = i
                endsteep = i + 1  # not including; upward region is [startsteep, endsteep]
                i += 1
                while i < self.n - 1:
                    if not self.is_steep_upward_point(i, 0):  # oh wait, we are going downward
                        break
                    if self.is_steep_upward_point(i, xi):  # upward point again, keep going
                        endsteep = i + 1
                    else:  # break, if min_pts consecutive xhi-equal point goes
                        if i - endsteep > self.min_pts:
                            break
                    i += 1
                i = endsteep
                print "found upward region [%s, %s)" % (startsteep, endsteep)
                cluster_found = False
                for sda in sdas:
                    is_cluster, left, right = self.is_cluster(sda, (startsteep, endsteep), xi)
                    if is_cluster:
                        clusters.add((left, right))
                        print "found cluster [%s, %s)" % (left, right)
                        cluster_found = True
                        break
                if cluster_found:
                    sdas.clear()
                continue
            i += 1
        print "Again, selected clusters:"
        for cluster in clusters:
            print cluster
        clusterid = -1
        self.__labels.fill(-1)
        for cluster in clusters:
            clusterid += 1
            indexes = self.__ordered_file['index'][cluster[0]:cluster[1]]
            self.__labels[indexes] = clusterid

    def __extract_dbscan(self, eps):
        assert eps <= self.eps
        clusterid = -1
        for point in self.__ordered_file:
            rd = point['reach_dist']
            i = point['index']
            if rd > eps:
                if point['core_dist'] <= eps:
                    clusterid += 1
                    self.__labels[i] = clusterid
                else:
                    self.__labels[i] = -1
            else:
                self.__labels[i] = clusterid

if __name__ == "__main__":
    np.random.seed(42)

    # # and this is for microdebug
    # data = np.array([[1, 1],
    #                  [3, 1],
    #                  [5, 1],
    #                  [6, 1],
    #                  [6, 2],
    #                  [6, 0],
    #                  [7, 1],
    #                  [7, 6],
    #                  [10, 10],
    #                  [10, 12],
    #                  [12, 10]])
    #
    # pred_optics = OPTICS(eps=2, min_pts=3).fit_predict(data)
    # pred_dbsan = DBSCAN(eps=2, min_samples=3).fit_predict(data)
    # pl.scatter(data[:, 0], data[:, 1], c=pred_dbsan)
    # pl.show()


    # and this is for debug now
    # iris = ds.load_iris()
    # data = iris.data[:, 2:4]  # data
    #
    # # pred_kmeans = KMeans(n_clusters=2).fit_predict(data)
    # pred_dbscan = DBSCAN(eps=0.5, min_samples=4).fit_predict(data)
    # pred_optics = OPTICS(eps=10, min_pts=4).fit_predict(data)
    # print "Adjusted Rand index for iris is: %.2f" % smt.adjusted_rand_score(pred_optics, pred_dbscan)
    #
    # pl.subplot(1, 3, 1)
    # pl.scatter(data[:, 0], data[:, 1], lw=0, s=30)
    # pl.xlabel('Sepal length')
    # pl.ylabel('Sepal width')
    # #
    # pl.subplot(1, 3, 2)
    # pl.scatter(data[:, 0], data[:, 1], c=pred_dbscan, cmap=pl.cm.RdBu, lw=0, s=30)
    # pl.xlabel('Sepal length')
    # pl.ylabel('Sepal width')
    # #
    # pl.subplot(1, 3, 3)
    # pl.scatter(data[:, 0], data[:, 1], c=pred_optics, cmap=pl.cm.RdBu, lw=0, s=30)
    # pl.xlabel('Sepal length')
    # pl.ylabel('Sepal width')
    # pl.show()

    # this stuff is for check later: we have here 4-d features, but draw two 2d plots. Not good.
    iris = ds.load_iris()
    data = iris.data[:100] # data
    y_iris = iris.target[:100]  # clusters

    # pred_optics = OPTICS(eps=10, min_pts=4).fit_predict(data, dbscan=True, dbscan_eps=0.75)
    pred_optics = OPTICS(eps=10, min_pts=10).fit_predict(data, xi=0.15)
    pl.subplot(2, 2, 1)
    pl.scatter(data[:, 0], data[:, 1], c=y_iris, cmap=pl.cm.RdBu, lw=0, s=30)
    pl.xlabel('Sepal length, reference clusters')
    pl.ylabel('Sepal width')

    pl.subplot(2, 2, 2)
    pl.scatter(data[:, 2], data[:, 3], c=y_iris, cmap=pl.cm.RdBu, lw=0, s=30)
    pl.xlabel('Petal length, reference clusters')
    pl.ylabel('Petal width')

    pl.subplot(2, 2, 3)
    pl.scatter(data[:, 0], data[:, 1], c=pred_optics, cmap=pl.cm.RdBu, lw=0, s=30)
    pl.xlabel('Sepal length, optics clusters')
    pl.ylabel('Sepal width')

    pl.subplot(2, 2, 4)
    pl.scatter(data[:, 2], data[:, 3], c=pred_optics, cmap=pl.cm.RdBu, lw=0, s=30)
    pl.xlabel('Petal length, optics clusters')
    pl.ylabel('Petal width')
    pl.show()
    print "Adjusted Rand index for iris is: %.2f" % smt.adjusted_rand_score(y_iris, pred_optics)
