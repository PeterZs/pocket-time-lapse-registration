import numpy as np
import pandas as pd
try:
    import networkx as nx
except:
    print("import of networkx failed. Perhaps try installing graphviz? With homebrew on osx, `brew install graphviz`.")



GRAPH_DISTANCE_METRIC = 'cost'
# GRAPH_DISTANCE_METRIC = 'match_penalty'



class AlignmentGraph(object):
    def __init__(self, cmdb, samples, subgraph=True, cutoff = None):
        self.samples = samples;
        self.cmdb = cmdb;
        self.graph = cmdb.get_edge_graph()
        if(subgraph):
            self.graph = self.graph.subgraph(samples.get_sample_name_list()).copy();

    def get_alignment_matrix_for_samples(self, from_sample, to_sample):
        raise NotImplementedError;


        # primary_closeness_centrality.keys()

class AllPathsAlignmentGraph(AlignmentGraph):
    def __init__(self, cmdb, samples, subgraph=True, cutoff=None):
        super(AllPathsAlignmentGraph, self).__init__(cmdb, samples, subgraph=subgraph, cutoff=cutoff);
        self._closeness_centrality = nx.closeness_centrality(self.graph,
                                                             distance=GRAPH_DISTANCE_METRIC,
                                                             wf_improved=True
                                                             )
        self._shortest_paths = dict(nx.all_pairs_dijkstra(self.graph, cutoff=cutoff, weight=GRAPH_DISTANCE_METRIC))
        closeness = pd.DataFrame.from_dict(self._closeness_centrality, orient='index', columns=['closeness_centrality'])
        closeness.sort_values(by=['closeness_centrality'], ascending=False, inplace=True)
        closeness_rows = closeness.reset_index(names='sample_name')
        self.sample_dict = {}
        for s in samples:
            self.sample_dict[s.sample_name] = s;
        self.central_sample = self.sample_dict[closeness_rows.iloc[0]['sample_name']];

    def get_alignment_matrix_for_samples(self, from_sample, to_sample):
        fname = from_sample.sample_name;
        tname = to_sample.sample_name;
        if(fname == tname):
            return np.eye(3);
        # print("FROM {}".format(fname))
        # print("TO {}".format(tname))
        path = self._shortest_paths[fname][1][tname]

        sample_path = [self.sample_dict[x] for x in path];
        matrix = np.eye(3);
        if (len(sample_path) > 1):
            for si in range(len(sample_path) - 1):
                stvg = self.cmdb.get_geometry_for_sample_pair(sample_path[si], sample_path[si + 1]);
                if(stvg is None):
                    print("No geometry for {}->{}: Did you register since adding new samples?".format(sample_path[si].sample_name, sample_path[si+1].sample_name))
                    return None
                newmat = stvg.H(sample_path[si], sample_path[si + 1]);
                if (isinstance(newmat, np.ndarray)):
                    matrix = newmat @ matrix;
                else:
                    print("BAD MATRIX for {}->{}: {}".format(sample_path[si].sample_name, sample_path[si+1].sample_name, newmat))
                    # raise ValueError(newmat);
                    return None;
            return matrix;
        else:
            if (sample_path[0] != from_sample.sample_name):
                raise ValueError("Something weird going on with finding alignment matrix!!!")
            else:
                return np.eye(3);

        # def GetAlignmentDistanceForSamples(self, from_sample, to_sample):
        #     fname = from_sample.sample_name;
        #     tname = to_sample.sample_name;
        #     if (fname == tname):
        #         return np.eye(3);
        #     path = self._shortest_paths[fname][1][tname]
        #
        #     sample_path = [self.sample_dict[x] for x in path];
        #     matrix = np.eye(3);
        #     cumdist = 0;
        #     if (len(sample_path) > 1):
        #         for si in range(len(sample_path) - 1):
        #             stvg = self.cmdb.GetGeometryForSamplePair(sample_path[si], sample_path[si + 1]);
        #             # cumdist = cumdist+stvg.
        #             print(stvg)
        #             newmat = stvg.H(sample_path[si], sample_path[si + 1]);
        #             if (isinstance(newmat, np.ndarray)):
        #                 matrix = newmat @ matrix;
        #             else:
        #                 print("BAD MATRIX for {}->{}: {}".format(sample_path[si].sample_name,
        #                                                          sample_path[si + 1].sample_name, newmat))
        #                 # raise ValueError(newmat);
        #                 return None;
        #         return matrix;
        #     else:
        #         if (sample_path[0] != from_sample.sample_name):
        #             raise ValueError("Something weird going on with finding alignment matrix!!!")
        #         else:
        #             return np.eye(3);

        # primary_closeness_centrality.keys()

    def get_alignment_path_for_samples(self, from_sample, to_sample):
        fname = from_sample.sample_name;
        tname = to_sample.sample_name;
        if (fname == tname):
            return [tname];
        return self._shortest_paths[fname][1][tname];




class CentralSampleAlignmentGraph(AlignmentGraph):
    def __init__(self, cmdb, samples, central_sample, subgraph=True, cutoff=None):
        super(CentralSampleAlignmentGraph, self).__init__(cmdb, samples, subgraph=subgraph, cutoff=cutoff);
        self.central_sample = central_sample;
        self._paths_from_center_node = nx.single_source_dijkstra(
            self.graph,
            central_sample.sample_name,
            target=None,
            cutoff=cutoff,
            # weight='cost'
            weight=GRAPH_DISTANCE_METRIC
        )
        self.sample_dict = {}
        for s in samples:
            self.sample_dict[s.sample_name] = s;

    def get_alignment_matrix_for_sample(self, from_sample, from_central_node=False):
        fname = from_sample.sample_name;
        tname = self.central_sample.sample_name;
        path = self._paths_from_center_node[1][fname]
        if(not from_central_node):
            path.reverse();

        sample_path = [self.sample_dict[x] for x in path];
        matrix = np.eye(3);
        if (len(sample_path) > 1):
            for si in range(len(sample_path) - 1):
                stvg = self.cmdb.get_geometry_for_sample_pair(sample_path[si], sample_path[si + 1]);
                newmat = stvg.H(sample_path[si], sample_path[si + 1]);
                if(isinstance(newmat, np.ndarray)):
                    matrix = newmat @ matrix;
                else:
                    print(newmat)
                    # raise ValueError(newmat);
                    return None;
            return matrix;
        else:
            if (sample_path[0] != from_sample.sample_name):
                raise ValueError("Something weird going on with finding alignment matrix!!!")
            else:
                return np.eye(3);

    def get_alignment_path_for_sample(self, from_sample, from_central_node=False):
        fname = from_sample.sample_name;
        path = self._paths_from_center_node[1][fname]
        if(not from_central_node):
            path.reverse();
        return path;