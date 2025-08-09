import os
import collections
import numpy as np
import struct

import pandas as pd

import sqlite3

from .CDBCameras import CDBCameras
from .CDBImages import CDBImages
from .CDBKeypoints import CDBKeypoints
from .CDBMatches import CDBMatches
from .CDBTVGeometries import CDBTVGeometries, TVGeometry
from .CDBTable import image_ids_to_pair_id, pair_id_to_image_ids
from .AlignmentGraph import *
from ptlreg.apydn.datanode import DataNode
from ptlreg.apydn.FPath import FPath

try:
    import networkx as nx
except:
    print("import of networkx failed. Perhaps try installing graphviz? With homebrew on osx, `brew install graphviz`.")

# from pandas import read_sql_query, read_sql_table




class COLMAPDB(DataNode):
    def __init__(self, path,
                 # cameras=None,
                 # images=None,
                 # keypoints=None,
                 # matches=None,
                 # copy=False,
                 **kwargs
                 ):
        fpath = FPath(path);
        if(fpath.is_dir()):
            fpath = FPath(os.path.join(path, 'colmap.db'));

        dataframes = COLMAPDB.dataframe_dict_from_db_file(fpath.get_absolute_path());
        # 'cameras', 'sqlite_sequence', 'images', 'keypoints', 'descriptors', 'matches', 'two_view_geometries'
        self.data_node_sets = dict(
            cameras=CDBCameras.from_dataframe_with_binary_info(dataframes['cameras']),
            images=CDBImages.from_dataframe_with_binary_info(dataframes['images']),
            keypoints=CDBKeypoints.from_dataframe_with_binary_info(dataframes['keypoints']),
            matches=CDBMatches.from_dataframe_with_binary_info(dataframes['matches']),
            two_view_geometries = CDBTVGeometries.from_dataframe_with_binary_info(dataframes['two_view_geometries']),
        )
        # self._image_id_to_filename = self.images.nodes['name'].map(lambda x: os.path.basename(x));
        self.root_path = fpath.get_directory_path()
        self._imageid_to_subpath = self.images.nodes['name'];

        def get_sample_name(file_subpath):
            name_parts = os.path.splitext(os.path.basename(file_subpath));
            return name_parts[0];

        self._imageid_to_sample_name = self.images.nodes['name'].map(lambda x: get_sample_name(x));
        self._subpath_to_imageid = pd.Series(self._imageid_to_subpath.index.values, index=self._imageid_to_subpath);

    # if(copy):
    #     self.data_node_sets = dict(
    #         cameras=CDBCameras._CloneWithNodes(cameras),
    #         images=CDBImages._CloneWithNodes(images),
    #         keypoints=CDBKeypoints._CloneWithNodes(keypoints),
    #         matches =CDBMatches._CloneWithNodes(matches),
    #     )
    # else:
    #     self.data_node_sets = dict(
    #         cameras=cameras,
    #         images=images,
    #         keypoints=keypoints,
    #         matches=matches,
    #     )

    # self._image_id_to_filename = self.images.nodes['name'].map(lambda x: os.path.basename(x));
    # self._filename_to_image_id = pd.Series(self._index_to_filename.index.values, index=self._index_to_filename);

    @classmethod
    def pair_id_from_image_ids(cls, id1, id2):
        return image_ids_to_pair_id(id1, id2);

    @classmethod
    def image_ids_from_pair_id(cls, pair_id):
        return pair_id_to_image_ids(pair_id);

    def get_image_id_for_file_name(self, file_name):
        return self._filename_to_index[file_name];


    @property
    def imageid_to_subpath(self):
        return self._imageid_to_subpath;

    @property
    def subpath_to_imageid(self):
        return self._subpath_to_imageid;


    @property
    def cameras(self):
        return self.data_node_sets['cameras'];


    @property
    def images(self):
        return self.data_node_sets['images'];


    @property
    def keypoints(self):
        return self.data_node_sets['keypoints'];


    @property
    def matches(self):
        return self.data_node_sets['matches'];

    @property
    def tvg(self):
        return self.data_node_sets['two_view_geometries'];

    @property
    def two_view_geometries(self):
        return self.data_node_sets['two_view_geometries'];


    @classmethod
    def from_dataframes(cls, cameras_dataframe, images_dataframe, keypoints_dataframe, matches_dataframe, **kwargs):
        return cls(
            cameras=CDBCameras.from_dataframe(cameras_dataframe),
            images=CDBImages.from_dataframe(images_dataframe),
            keypoints=CDBKeypoints.from_dataframe(keypoints_dataframe),
            matches=CDBMatches.from_dataframe(matches_dataframe)
        );


    @classmethod
    def dataframe_dict_from_db_file(cls, path):
        print(path)
        with sqlite3.connect(path) as dbcon:
            tables = list(pd.read_sql_query("SELECT name FROM sqlite_master WHERE type='table';", dbcon)['name'])
            dataframes = {tbl: pd.read_sql_query(f"SELECT * from {tbl}", dbcon) for tbl in tables}
        return dataframes;


    @classmethod
    def from_colmapdb_file(cls, path):
        # with sqlite3.connect(path) as dbcon:
        #     tables = list(pd.read_sql_query("SELECT name FROM sqlite_master WHERE type='table';", dbcon)['name'])
        #     dataframes = {tbl: pd.read_sql_query(f"SELECT * from {tbl}", dbcon) for tbl in tables}
        # dataframes = COLMAPDB.DataFrameDictFromDBFile(path);

        # 'cameras', 'sqlite_sequence', 'images', 'keypoints', 'descriptors', 'matches', 'two_view_geometries'
        return cls(
            path=path,
            # cameras=CDBCameras.from_dataframeWithBinaryInfo(dataframes['cameras']),
            # images=CDBImages.from_dataframeWithBinaryInfo(dataframes['images']),
            # keypoints=CDBKeypoints.from_dataframeWithBinaryInfo(dataframes['keypoints']),
            # matches=CDBMatches.from_dataframeWithBinaryInfo(dataframes['matches']),
        )


    def _get_matches_for_image_id_pair(self, first_id, second_id):
        return self.matches.get_matches_for_image_id_pair(first_id, second_id);


    def _get_keypoints_match_locations_for_image_id_pair(self, first_id, second_id):
        matches = self.matches.get_matches_for_image_id_pair(first_id, second_id)
        positions_in_first = self.keypoints[first_id].point_locations[matches[:, 0]];
        positions_in_second = self.keypoints[second_id].point_locations[matches[:, 1]];
        return positions_in_first, positions_in_second;

    def get_root_subpath_for_absolute_path(self, absolute_path):
        return os.path.relpath(absolute_path, self.root_path);

    def _get_image_id_for_subpath(self, subpath):
        if(subpath in self._subpath_to_imageid):
            return self._subpath_to_imageid[subpath];
        else:
            return None;

    def _get_image_ids_for_samples(self, samples, key_function=None):
        """

        :param samples:
        :param key_function: function that maps a sample to its id, By default is the subpath to the sample from the db root,
        :return:
        """
        rval = [];
        if(key_function is None):
            for s in samples:
                sp = self.get_root_subpath_for_absolute_path(s.absolute_file_path);
                rval.append(self._get_image_id_for_subpath(sp));
        else:
            for s in samples:
                sp = key_function(s);
                rval.append(self._get_image_id_for_subpath(sp));
        return rval;


    def get_keypoint_matches_for_sample_ids(self, first_id, second_id):
        """

        :param first_id: id of first image
        :param second_id:
        :return:
        """
        matches = self.matches.get_matches_for_image_id_pair(first_id, second_id)
        if(matches is np.nan or matches is None):
            return None, None;
        positions_in_first = self.keypoints[first_id].point_locations[matches[:, 0]];
        positions_in_second = self.keypoints[second_id].point_locations[matches[:, 1]];
        return positions_in_first, positions_in_second;

    def get_keypoint_matches_for_sample_pair(self, first_sample, second_sample, key_function = None):
        """

        :param first_sample:
        :param second_sample:
        :param key_function: function that maps a sample to its id, By default is the subpath to the sample from the db root,
        :return:
        """
        pair_ids = self._get_image_ids_for_samples([first_sample, second_sample], key_function=key_function);
        [first_id, second_id] = pair_ids;
        if(first_id is None or second_id is None):
            return None, None;
        matches = self.matches.get_matches_for_image_id_pair(first_id, second_id)
        if(matches is np.nan or matches is None):
            # print("NAN MATCHES FOR:\n{}\n{}".format(first_sample.file_path, second_sample.file_path))
            return None, None;
        # print(second_sample.file_name)
        # print(self.keypoints[first_id].point_locations)
        # print(matches);

        positions_in_first = self.keypoints[first_id].point_locations[matches[:, 0]];
        positions_in_second = self.keypoints[second_id].point_locations[matches[:, 1]];
        return positions_in_first, positions_in_second;

    def get_pair_id_for_samples(self, first_sample, second_sample):
        image_ids = self._get_image_ids_for_samples([first_sample, second_sample]);
        return image_ids_to_pair_id(image_ids[0], image_ids[1]);


    def get_geometry_for_sample_pair(self, first_sample, second_sample, key_function = None):
        """

        :param first_sample:
        :param second_sample:
        :param key_function:
        :return:
        """
        pair_ids = self._get_image_ids_for_samples([first_sample, second_sample], key_function=key_function);
        [first_id, second_id] = pair_ids;
        if (first_id is None or second_id is None):
            print("NO TVG FOR PAIR {} | {}".format(first_sample.sample_name, second_sample.sample_name))
            return None;
        pair_id = image_ids_to_pair_id(first_id, second_id);
        # print(self.tvg)
        # print(self.tvg.index)
        tvg = self.tvg._get_geometry_for_pair_id(pair_id);
        # print("PairID {}".format(pair_id));
        if (tvg is np.nan or tvg is None):
            print("tvg was {}!!!!!!".format(tvg))
            return None;
        id_order = pair_id_to_image_ids(pair_id);
        if(id_order[0] == first_id):
            return TVGeometry(from_sample=first_sample, to_sample=second_sample, tvgnode=tvg);
        else:
            return TVGeometry(from_sample=second_sample, to_sample=first_sample, tvgnode=tvg);

    def get_edge_graph_dataframe(self):
        def getMDet(x):
            if (not isinstance(x, np.ndarray)):
                return x;
            else:
                m = np.array(x).reshape(3,3);
                if(m[2,2] == 0):
                    return None;
                m = m/m[2,2];
                return np.linalg.det(m);

        def getH(x):
            if (not isinstance(x, np.ndarray)):
                return x;
            else:
                return x.tolist();
                # m = np.array(x).reshape(3,3);
                # if(m[2,2] == 0):
                #     return None;
                # m = m/m[2,2];
                # return np.linalg.det(m);

        def getHScore(x):
            if (not isinstance(x, np.ndarray)):
                return x;
            else:
                m = np.array(x).reshape(3,3);
                if(m[2,2] == 0):
                    return None;
                m = m/m[2,2];
                det = np.linalg.det(m);
                if(det<1):
                    return det;
                if(det>1):
                    return 1/det;

        def getMLinDet(x):
            if (not isinstance(x, np.ndarray)):
                return x;
            else:
                return np.linalg.det(np.array(x).reshape(3,3)[:2,:2]);

        # def datashape(x):
        #     if (isinstance(x, np.ndarray)):
        #         return len(x);
        #     else:
        #         return 0

        tvg = self.tvg.nodes;
        max_inliers = tvg.n_inliers.max();

        def match_score(row):
            if(CDBTVGeometries.CONFIG_IS_HOMOGRAPHY(row['config'])):
                rowh = row['H'];
                if (rowh is None):
                    print(rowh);
                    raise ValueError("Shouldnt get null H here!")
                else:
                    rowh = np.array(rowh);
                # if(not isinstance(rowh, np.ndarray)):
                #     print(rowh);
                #     raise ValueError("Shouldnt get null H here!")
                return np.true_divide(row['n_inliers'], max_inliers);
            else:
                return 0;

        def ellipse_area(points, sqrt=False):
            # Ensure points is a 2D numpy array
            points = np.asarray(points)

            # Compute the covariance matrix of the points
            cov_matrix = np.cov(points.T)

            # Compute the eigenvalues of the covariance matrix
            eigenvalues = np.linalg.eigvals(cov_matrix)

            # The standard deviations are the square roots of the eigenvalues
            sigma_1 = np.sqrt(np.max(eigenvalues))  # larger eigenvalue (major axis)
            sigma_2 = np.sqrt(np.min(eigenvalues))  # smaller eigenvalue (minor axis)

            # The area of the standard deviation ellipse
            area = np.pi * sigma_1 * sigma_2
            if(sqrt):
                return np.sqrt(area);
            else:
                return area;

        def match_penalty(row):
            return np.true_divide(1.0, row['match_score']);

        r_dataframe = pd.DataFrame(dict(
            source=tvg.pair_ids.map(lambda x: self._imageid_to_sample_name[x[0]]),
            target=tvg.pair_ids.map(lambda x: self._imageid_to_sample_name[x[1]]),
            id_pair=tvg.pair_ids,
            config=tvg.config,
            traslation_norm=tvg.tvec.map(lambda x: np.linalg.norm(x)),
            H=tvg.H.map(getH),
            HDet=tvg.H.map(getMDet),
            HScore=tvg.H.map(getHScore),
            n_inliers=tvg.n_inliers,
            # match_score = tvg.n_inliers.map(lambda x:x/max_inliers),
            # match_penalty=tvg.n_inliers.map(lambda x: max_inliers/x),
            # HLinDet=tvg.H.map(getMLinDet),
            FDet=tvg.F.map(getMDet),
            source_spread=0,
            target_spread=0
            # FLinDet=tvg.F.map(getMLinDet),
        ))

        r_dataframe['match_score'] = r_dataframe.apply(match_score, axis=1);
        r_dataframe['match_penalty'] = r_dataframe.apply(match_penalty, axis=1);

        # r_dataframe['match_points'] = r_dataframe.apply(match_points, axis=1);

        def get_matches(row):
            ids = row['id_pair'];
            return self.get_keypoint_matches_for_sample_ids(
                ids[0],
                ids[1],
            );

        matches = r_dataframe.apply(get_matches, axis=1)

        def source_spread(keymatches):
            if (keymatches[0] is not None):
                return ellipse_area(keymatches[0])
            else:
                return 0

        def target_spread(keymatches):
            if (keymatches[0] is not None):
                return ellipse_area(keymatches[1])
            else:
                return 0

        r_dataframe['source_spread'] = matches.apply(source_spread);
        r_dataframe['target_spread'] = matches.apply(target_spread);

        def spread_ratio(row):
            ssp = row['source_spread']
            tsp = row['target_spread']
            mx = np.max([ssp, tsp]);
            if (mx<=0):
                return 0;
            else:
                mn = np.min([ssp, tsp]);
                return mn/mx;

        r_dataframe['spread_ratio'] = r_dataframe.apply(spread_ratio, axis=1);

        def registration_score_unnormalized(row):
            ssp = row['source_spread']
            tsp = row['target_spread']
            mscore = row['match_score']
            spread_ratio = row['spread_ratio'];
            # rval = mscore
            # rval = np.sqrt(mscore * spread_ratio)
            # rval = mscore * np.sqrt(spread_ratio)
            rval = (mscore * spread_ratio)  # jan10night
            # rval = (tsp * mscore * spread_ratio)  # aslfgkjneuh
            # rval = (tsp * mscore * spread_ratio) # aslfgkjneuh
            # rval = np.power((np.sqrt(tsp) * mscore * spread_ratio), 2); # V_eoskejf
            return rval
            # if(rval<0):
            #     print(rval)
            #     raise ValueError
            # return (spread_ratio * mscore);
            # sqrt_ssptsp_sq_mscore * mscore);
            # return (np.sqrt(ssp*tsp) * mscore * spread_ratio);

        # sqrt_ssptsp_sq_mscore

        r_dataframe['registration_score_unnormalized'] = r_dataframe.apply(registration_score_unnormalized, axis=1);

        max_score = r_dataframe.registration_score_unnormalized.max()

        r_dataframe['registration_score'] = r_dataframe['registration_score_unnormalized'] / max_score

        def costfunc(row):
            return np.true_divide(1, row['registration_score'])
            # return np.true_divide( 1, np.power(row['registration_score'], 0.5))

        r_dataframe['cost'] = r_dataframe.apply(costfunc, axis=1);
        return r_dataframe

    def get_edge_graph(self):
        edf = self.get_edge_graph_dataframe();
        g = nx.from_pandas_edgelist(edf, 'source', 'target', True);
        return g;


    def get_all_paths_alignment_graph_for_samples(self, samples, cutoff=None):
        return AllPathsAlignmentGraph(cmdb=self, samples = samples, cutoff=cutoff)

    def get_central_sample_alignment_graph_for_samples(self, samples, central_sample, subgraph=True, cutoff=None):
        return CentralSampleAlignmentGraph(cmdb=self, samples = samples, central_sample=central_sample, subgraph=subgraph, cutoff=cutoff);
    # def getCentralNode(self):
    #




    # def GetNXGraph(self):
    #     G = nx.from_pandas_edgelist(self.GetEdgeGraphDataFrame());
    #     nx.set_node_attributes(G, 'name', pd.Series(nodes.name, index=nodes.node).to_dict())
    #     nx.set_node_attributes(G, pd.Series(nodes.gender, index=nodes.node).to_dict())
