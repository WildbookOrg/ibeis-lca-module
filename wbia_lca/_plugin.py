# -*- coding: utf-8 -*-
from wbia.control import controller_inject
from wbia.constants import CONTAINERIZED, PRODUCTION  # NOQA
from wbia import constants as const
from wbia.web.graph_server import GraphClient, GraphActor
from wbia.algo.graph.state import POSTV, NEGTV, INCMP, UNREV, UNKWN, NULL
from wbia.algo.graph.core import _rectify_decision

import numpy as np
import logging
import utool as ut
from functools import partial

from wbia_lca import db_interface
from wbia_lca import edge_generator

import configparser
import threading
import random
import json
import sys

from wbia_lca import ga_driver
from wbia_lca import overall_driver

import tqdm

logger = logging.getLogger('wbia_lca')


_, register_ibs_method = controller_inject.make_ibs_register_decorator(__name__)

register_api = controller_inject.get_wbia_flask_api(__name__)
register_route = controller_inject.get_wbia_flask_route(__name__)

register_preproc_image = controller_inject.register_preprocs['image']
register_preproc_annot = controller_inject.register_preprocs['annot']

AUTOREVIEW_IDENTITY = 'user:autoreview'

HUMAN_AUG_NAME = 'human'
HUMAN_IDENTITY = 'user:web'
HUMAN_IDENTITY_PREFIX = '%s:' % (HUMAN_IDENTITY.split(':')[0],)
ALGO_AUG_NAME = 'vamp'
ALGO_IDENTITY = 'algo:vamp'
ALGO_IDENTITY_PREFIX = '%s:' % (ALGO_IDENTITY.split(':')[0],)

# HUMAN_CORRECT_RATE = 0.97
HUMAN_CORRECT_RATE = 1.0

USE_COLDSTART = ut.get_argflag('--lca-coldstart')
USE_AUTOREVIEW = ut.get_argflag('--lca-autoreview') or USE_COLDSTART


LOG_DECISION_FILE = 'lca.decisions.csv'
LOG_LCS_FILE = 'lca.log'


@register_ibs_method
@register_api('/api/plugin/lca/sim/', methods=['GET'])
def wbia_plugin_lca_sim(ibs, ga_config, verifier_gt, request, db_result=None):
    r"""
    Create an LCA graph algorithm object and run a simulator

    Args:
        ibs (IBEISController): wbia controller object
        ga_config (str): graph algorithm config INI file
        verifier_gt (str): json file containing verification algorithm ground truth
        request (str): json file continain graph algorithm request info
        db_result (str, optional): file to write resulting json database

    Returns:
        object: changes_to_review

    CommandLine:
        python -m wbia_lca._plugin wbia_plugin_lca_sim
        python -m wbia_lca._plugin wbia_plugin_lca_sim:0
        python -m wbia_lca._plugin wbia_plugin_lca_sim:1
        python -m wbia_lca._plugin wbia_plugin_lca_sim:2

    RESTful:
        Method: GET
        URL:    /api/plugin/lca/sim/

    Doctest:
        >>> # ENABLE_DOCTEST
        >>> import wbia
        >>> import utool as ut
        >>> import random
        >>> random.seed(1)
        >>> from wbia.init import sysres
        >>> dbdir = sysres.ensure_testdb_identification_example()
        >>> ibs = wbia.opendb(dbdir=dbdir)
        >>> ga_config = 'examples/default/config.ini'
        >>> verifier_gt = 'examples/default/verifier_probs.json'
        >>> request = 'examples/default/request_example.json'
        >>> db_result = 'examples/default/result.json'
        >>> changes_to_review = ibs.wbia_plugin_lca_sim(ga_config, verifier_gt, request, db_result)
        >>> results = []
        >>> for cluster in changes_to_review:
        >>>     lines = []
        >>>     for change in cluster:
        >>>         line = []
        >>>         line.append('query nodes %s' % (sorted(change.query_nodes),))
        >>>         line.append('change_type %s' % (change.change_type,))
        >>>         line.append('old_clustering %s' % (sorted(change.old_clustering), ))
        >>>         line.append('len(new_clustering) %s' % (len(sorted(change.new_clustering)), ))
        >>>         line.append('removed_nodes %s' % (sorted(change.removed_nodes),))
        >>>         lines.append('\n'.join(line))
        >>>     results.append('\n-\n'.join(sorted(lines)))
        >>> result = '\n----\n'.join(sorted(results))
        >>> print('----\n%s\n----' % (result, ))
        ----
        query nodes ['c', 'e']
        change_type Merge
        old_clustering ['100', '101']
        len(new_clustering) 1
        removed_nodes []
        ----
        query nodes ['f']
        change_type Extension
        old_clustering ['102']
        len(new_clustering) 1
        removed_nodes []
        -
        query nodes ['g']
        change_type New
        old_clustering []
        len(new_clustering) 1
        removed_nodes []
        ----
        query nodes ['m']
        change_type Extension
        old_clustering ['103']
        len(new_clustering) 1
        removed_nodes []
        ----

    Doctest:
        >>> # ENABLE_DOCTEST
        >>> import wbia
        >>> import utool as ut
        >>> import random
        >>> random.seed(1)
        >>> from wbia.init import sysres
        >>> dbdir = sysres.ensure_testdb_identification_example()
        >>> ibs = wbia.opendb(dbdir=dbdir)
        >>> ga_config = 'examples/merge/config.ini'
        >>> verifier_gt = 'examples/merge/verifier_probs.json'
        >>> request = 'examples/merge/request_example.json'
        >>> db_result = 'examples/merge/result.json'
        >>> changes_to_review = ibs.wbia_plugin_lca_sim(ga_config, verifier_gt, request, db_result)
        >>> results = []
        >>> for cluster in changes_to_review:
        >>>     lines = []
        >>>     for change in cluster:
        >>>         line = []
        >>>         line.append('query nodes %s' % (sorted(change.query_nodes),))
        >>>         line.append('change_type %s' % (change.change_type,))
        >>>         line.append('old_clustering %s' % (sorted(change.old_clustering), ))
        >>>         line.append('len(new_clustering) %s' % (len(sorted(change.new_clustering)), ))
        >>>         line.append('removed_nodes %s' % (sorted(change.removed_nodes),))
        >>>         lines.append('\n'.join(line))
        >>>     results.append('\n-\n'.join(sorted(lines)))
        >>> result = '\n----\n'.join(sorted(results))
        >>> print('----\n%s\n----' % (result, ))
        ----
        query nodes []
        change_type Merge/Split
        old_clustering ['100', '101']
        len(new_clustering) 2
        removed_nodes []
        -
        query nodes []
        change_type Unchanged
        old_clustering ['102']
        len(new_clustering) 1
        removed_nodes []
        ----

    Doctest:
        >>> # ENABLE_DOCTEST
        >>> import wbia
        >>> import utool as ut
        >>> import random
        >>> random.seed(1)
        >>> from wbia.init import sysres
        >>> dbdir = sysres.ensure_testdb_identification_example()
        >>> ibs = wbia.opendb(dbdir=dbdir)
        >>> ga_config = 'examples/zero/config.ini'
        >>> verifier_gt = 'examples/zero/verifier_probs.json'
        >>> request = 'examples/zero/request_example.json'
        >>> db_result = 'examples/zero/result.json'
        >>> changes_to_review = ibs.wbia_plugin_lca_sim(ga_config, verifier_gt, request, db_result)
        >>> results = []
        >>> for cluster in changes_to_review:
        >>>     lines = []
        >>>     for change in cluster:
        >>>         line = []
        >>>         line.append('query nodes %s' % (sorted(change.query_nodes),))
        >>>         line.append('change_type %s' % (change.change_type,))
        >>>         line.append('old_clustering %s' % (sorted(change.old_clustering), ))
        >>>         line.append('len(new_clustering) %s' % (len(sorted(change.new_clustering)), ))
        >>>         line.append('removed_nodes %s' % (sorted(change.removed_nodes),))
        >>>         lines.append('\n'.join(line))
        >>>     results.append('\n-\n'.join(sorted(lines)))
        >>> result = '\n----\n'.join(sorted(results))
        >>> print('----\n%s\n----' % (result, ))
        ----
        query nodes ['a', 'b', 'c', 'd', 'e']
        change_type New
        old_clustering []
        len(new_clustering) 1
        removed_nodes []
        -
        query nodes ['f', 'h', 'i', 'j']
        change_type New
        old_clustering []
        len(new_clustering) 1
        removed_nodes []
        -
        query nodes ['g']
        change_type New
        old_clustering []
        len(new_clustering) 1
        removed_nodes []
        -
        query nodes ['k', 'l', 'm']
        change_type New
        old_clustering []
        len(new_clustering) 1
        removed_nodes []
        ----
    """
    # 1. Configuration
    config_ini = configparser.ConfigParser()
    config_ini.read(ga_config)

    # 2. Recent results from verification ground truth tests. Used to
    # establish the weighter.
    with open(verifier_gt, 'r') as fn:
        verifier_gt = json.loads(fn.read())

    # 3. Form the parameters dictionary and weight objects (one per
    # verification algorithm).
    lca_config, wgtrs = ga_driver.params_and_weighters(config_ini, verifier_gt)
    if len(wgtrs) > 1:
        logger.info('Not currently handling more than one weighter!!')
        sys.exit(1)
    wgtr = wgtrs[0]

    # 4. Get the request dictionary, which includes the database, the
    # actual request edges and clusters, and the edge generator edges
    # and ground truth (for simulation).
    with open(request, 'r') as fn:
        request = json.loads(fn.read())

    db = overall_driver.form_database(request)
    edge_gen = overall_driver.form_edge_generator(request, db, wgtr)
    verifier_req, human_req, cluster_req = overall_driver.extract_requests(request, db)

    # 5. Form the graph algorithm driver
    driver = ga_driver.ga_driver(
        verifier_req, human_req, cluster_req, db, edge_gen, lca_config
    )

    # 6. Run it. Changes are logged.
    ccPIC_gen = driver.run_all_ccPICs()
    changes_to_review = list(ccPIC_gen)
    logger.info(changes_to_review)

    # 7. Commit changes. Record them in the database and the log
    # file.
    # TBD

    return changes_to_review


def is_aug_name_human(aug_name):
    return aug_name == HUMAN_AUG_NAME


def is_aug_name_algo(aug_name):
    return aug_name == ALGO_AUG_NAME


def is_identity_human(identity):
    return identity.startswith(HUMAN_IDENTITY_PREFIX)


def is_identity_algo(identity):
    return identity.startswith(ALGO_IDENTITY_PREFIX)


def convert_aug_name_to_identity(aug_name_list):
    identity_list = []
    for aug_name in aug_name_list:
        if is_aug_name_human(aug_name):
            identity = HUMAN_IDENTITY
        elif is_aug_name_algo(aug_name):
            identity = ALGO_IDENTITY
        else:
            raise ValueError()
        identity_list.append(identity)
    return identity_list


def convert_identity_to_aug_name(identity_list):
    aug_name_list = []
    for identity in identity_list:
        if is_identity_human(identity):
            aug_name = HUMAN_AUG_NAME
        elif is_identity_algo(identity):
            aug_name = ALGO_AUG_NAME
        else:
            raise ValueError()
        aug_name_list.append(aug_name)
    return aug_name_list


def convert_lca_cluster_id_to_wbia_name_id(lca_cluster_id):
    wbia_name_id = int(lca_cluster_id)
    return wbia_name_id


def convert_wbia_name_id_to_lca_cluster_id(wbia_name_id):
    lca_cluster_id = '%05d' % (wbia_name_id,)
    return lca_cluster_id


def convert_lca_node_id_to_wbia_annot_id(lca_node_id):
    wbia_annot_id = int(lca_node_id)
    return wbia_annot_id


def convert_wbia_annot_id_to_lca_node_id(wbia_annot_id):
    lca_node_id = '%05d' % (wbia_annot_id,)
    return lca_node_id


def get_dates(ibs, gid_list, gmt_offset=3.0):
    unixtime_list = ibs.get_image_unixtime2(gid_list)
    unixtime_list = [unixtime + (gmt_offset * 60 * 60) for unixtime in unixtime_list]
    datetime_list = [
        'UNKNOWN'
        if unixtime is None or np.isnan(unixtime)
        else ut.unixtime_to_datetimestr(unixtime)
        for unixtime in unixtime_list
    ]
    date_str_list = [value[:10] for value in datetime_list]
    return date_str_list


def get_ggr_stats(ibs, valid_aids, valid_nids):
    from wbia.other.dbinfo import sight_resight_count

    valid_gids = ibs.get_annot_gids(valid_aids)
    date_str_list = get_dates(ibs, valid_gids)

    name_dates_stats = {}
    for valid_aid, valid_nid, date_str in zip(valid_aids, valid_nids, date_str_list):
        if valid_nid not in name_dates_stats:
            name_dates_stats[valid_nid] = set([])
        name_dates_stats[valid_nid].add(date_str)

    valid_date_strs = set(
        [
            '2016/01/30',
            '2016/01/31',
            '2018/01/27',
            '2018/01/28',
        ]
    )

    ggr_name_dates_stats = {
        'GGR-16 D1 OR D2': 0,
        'GGR-16 D1 AND D2': 0,
        'GGR-18 D1 OR D2': 0,
        'GGR-18 D1 AND D2': 0,
        'GGR-16 AND GGR-18': 0,
        '0 Days': 0,
        '1+ Days': 0,
        '2+ Days': 0,
        '3+ Days': 0,
        '4+ Days': 0,
    }
    for date_str in sorted(set(date_str_list) | valid_date_strs):
        if date_str not in valid_date_strs:
            continue
        ggr_name_dates_stats[date_str] = 0

    for nid in name_dates_stats:
        date_strs = name_dates_stats[nid]
        date_strs = list(set(date_strs) & valid_date_strs)
        total_days = len(date_strs)
        assert 0 <= total_days and total_days <= 4
        if total_days == 0:
            key = '0 Days'
            ggr_name_dates_stats[key] += 1

        for val in range(1, total_days + 1):
            key = '%d+ Days' % (val,)
            ggr_name_dates_stats[key] += 1
        for date_str in date_strs:
            ggr_name_dates_stats[date_str] += 1
        if '2016/01/30' in date_strs or '2016/01/31' in date_strs:
            ggr_name_dates_stats['GGR-16 D1 OR D2'] += 1
            if '2018/01/27' in date_strs or '2018/01/28' in date_strs:
                ggr_name_dates_stats['GGR-16 AND GGR-18'] += 1
        if '2018/01/27' in date_strs or '2018/01/28' in date_strs:
            ggr_name_dates_stats['GGR-18 D1 OR D2'] += 1
        if '2016/01/30' in date_strs and '2016/01/31' in date_strs:
            ggr_name_dates_stats['GGR-16 D1 AND D2'] += 1
        if '2018/01/27' in date_strs and '2018/01/28' in date_strs:
            ggr_name_dates_stats['GGR-18 D1 AND D2'] += 1

    ggr16_pl_index, ggr16_pl_error = sight_resight_count(
        ggr_name_dates_stats['2016/01/30'],
        ggr_name_dates_stats['2016/01/31'],
        ggr_name_dates_stats['GGR-16 D1 AND D2'],
    )
    ggr_name_dates_stats['GGR-16 PL INDEX'] = '%0.01f' % (ggr16_pl_index,)
    ggr_name_dates_stats['GGR-16 PL CI'] = '%0.01f' % (ggr16_pl_error,)
    ggr_name_dates_stats['GGR-16 PL INDEX STR'] = '%0.01f +/- %0.01f' % (
        ggr16_pl_index,
        ggr16_pl_error,
    )
    total = ggr_name_dates_stats['GGR-16 D1 OR D2']
    if ggr16_pl_index == 0:
        ggr_name_dates_stats['GGR-16 COVERAGE'] = 'UNDEFINED'
    else:
        ggr_name_dates_stats['GGR-16 COVERAGE'] = '%0.01f (%0.01f - %0.01f)' % (
            100.0 * total / ggr16_pl_index,
            100.0 * total / (ggr16_pl_index + ggr16_pl_error),
            100.0 * min(1.0, total / (ggr16_pl_index - ggr16_pl_error)),
        )

    ggr18_pl_index, ggr18_pl_error = sight_resight_count(
        ggr_name_dates_stats['2018/01/27'],
        ggr_name_dates_stats['2018/01/28'],
        ggr_name_dates_stats['GGR-18 D1 AND D2'],
    )
    ggr_name_dates_stats['GGR-18 PL INDEX'] = '%0.01f' % (ggr18_pl_index,)
    ggr_name_dates_stats['GGR-18 PL CI'] = '%0.01f' % (ggr18_pl_error,)
    ggr_name_dates_stats['GGR-18 PL INDEX STR'] = '%0.01f +/- %0.01f' % (
        ggr18_pl_index,
        ggr18_pl_error,
    )
    total = ggr_name_dates_stats['GGR-18 D1 OR D2']
    if ggr18_pl_index == 0:
        ggr_name_dates_stats['GGR-18 COVERAGE'] = 'UNDEFINED'
    else:
        ggr_name_dates_stats['GGR-18 COVERAGE'] = '%0.01f (%0.01f - %0.01f)' % (
            100.0 * total / ggr18_pl_index,
            100.0 * total / (ggr18_pl_index + ggr18_pl_error),
            100.0 * min(1.0, total / (ggr18_pl_index - ggr18_pl_error)),
        )

    return ggr_name_dates_stats


def progress_db(actor, gai, iter_num):
    reviews = dict(zip(gai.weight_mgr.aug_names, gai.weight_mgr.counts))
    num_reviews_auto = reviews[ALGO_AUG_NAME]
    num_reviews_user = reviews[HUMAN_AUG_NAME]
    num_names = len(gai.clustering)
    num_todo = gai.queues.num_lcas()

    node_to_cluster = {}
    for cluster in gai.clustering:
        for node in gai.clustering[cluster]:
            assert node not in node_to_cluster
            node_to_cluster[node] = cluster
    assert set(node_to_cluster.keys()) == set(ut.flatten(gai.clustering.values()))
    assert set(node_to_cluster.values()) == set(gai.clustering.keys())

    valid_aids = []
    valid_nids = []
    for valid_aid in actor.infr.aids:
        valid_node = convert_wbia_annot_id_to_lca_node_id(valid_aid)
        valid_cluster = node_to_cluster.get(valid_node, None)
        valid_nid = valid_cluster
        # valid_nid = convert_lca_cluster_id_to_wbia_name_id(valid_cluster)
        if valid_nid is not None:
            valid_aids.append(valid_aid)
            valid_nids.append(valid_nid)

    ggr_name_dates_stats = get_ggr_stats(
        actor.infr.ibs,
        valid_aids,
        valid_nids,
    )
    with open(LOG_DECISION_FILE, 'a') as logfile:
        data = (
            iter_num,
            num_names,
            ggr_name_dates_stats['GGR-16 D1 OR D2'],
            ggr_name_dates_stats['GGR-16 PL INDEX'],
            ggr_name_dates_stats['GGR-16 PL CI'],
            ggr_name_dates_stats['2016/01/30'],
            ggr_name_dates_stats['2016/01/31'],
            ggr_name_dates_stats['GGR-16 D1 AND D2'],
            ggr_name_dates_stats['GGR-16 COVERAGE'],
            ggr_name_dates_stats['GGR-18 D1 OR D2'],
            ggr_name_dates_stats['GGR-18 PL INDEX'],
            ggr_name_dates_stats['GGR-18 PL CI'],
            ggr_name_dates_stats['2018/01/27'],
            ggr_name_dates_stats['2018/01/28'],
            ggr_name_dates_stats['GGR-18 D1 AND D2'],
            ggr_name_dates_stats['GGR-18 COVERAGE'],
            ggr_name_dates_stats['GGR-16 AND GGR-18'],
            num_reviews_auto,
            num_reviews_user,
            num_todo,
        )
        line = ','.join(map(str, data))
        logger.info('Progress: %s' % (line,))
        logfile.write('%s\n' % (line,))


class db_interface_wbia(db_interface.db_interface):  # NOQA
    def __init__(self, actor):
        self.controller = actor
        self.infr = actor.infr
        self.ibs = actor.infr.ibs

        """
        CHECK ME: make sure this is handled consistently
        """
        self.max_auto_reviews = 1
        self.max_human_reviews = 10
        self.max_reviews = self.max_auto_reviews + self.max_human_reviews

        edges = []

        """
        FIX ME: Handle COLDSTART as a parameter, not a commandline thingy
        """
        if USE_COLDSTART:
            logger.info('Cold Start: ignoring existing name clustering')
            clustering = {}
        else:
            clustering = self._get_existing_clustering()

        super(db_interface_wbia, self).__init__(edges, clustering)

    def _get_existing_clustering(self, use_ibeis_database=False):
        clustering = {}

        """
        FIX ME TO USE THE WBIA DATABASE CLUSTERING AS THE DEFAULT
        and to handle properly AIDS that don't have names.
        NOTES:
        1. MAKE SURE THIS IS ONLY CALLED ONCE!!!
        """

        if use_ibeis_database:
            src_str = 'IBEIS DB'
            aids = self.infr.aids
            nids = self.ibs.get_annot_nids(aids)
            for aid, nid in zip(aids, nids):
                cluster_label_ = convert_wbia_name_id_to_lca_cluster_id(nid)
                cluster_node_ = convert_wbia_annot_id_to_lca_node_id(aid)
                if cluster_label_ not in clustering:
                    clustering[cluster_label_] = []
                clustering[cluster_label_].append(cluster_node_)
        else:
            src_str = 'INFR POS_GRAPH'
            clustering_labels = list(self.infr.pos_graph.component_labels())
            clustering_components = list(self.infr.pos_graph.connected_components())
            assert len(clustering_labels) == len(clustering_components)

            for clustering_label, clustering_component in zip(
                clustering_labels, clustering_components
            ):
                clustering_label_ = convert_wbia_name_id_to_lca_cluster_id(
                    clustering_label
                )
                clustering_component = list(
                    map(convert_wbia_annot_id_to_lca_node_id, clustering_component)
                )
                clustering[clustering_label_] = clustering_component

        args = (
            len(clustering),
            src_str,
        )
        logger.info('Retrieving clustering with %d names (source: %s)' % args)

        for nid in sorted(clustering.keys()):
            clustering[nid] = sorted(clustering[nid])
            logger.info(
                '\tGT Cluster NID %r: %r'
                % (
                    nid,
                    clustering[nid],
                )
            )

        return clustering

    def _cleanup_edges(self, max_auto=None, max_human=None):
        weight_rowid_list = self.ibs.get_edge_weight_rowids_between(self.infr.aids)
        if max_auto is None:
            max_auto = self.max_auto_reviews
        if max_human is None:
            max_human = self.max_human_reviews

        """
        CHECK ME: Ensure that this works properly
        """
        self.ibs.check_edge_weights(
            weight_rowid_list=weight_rowid_list,
            max_auto=max_auto,
            max_human=max_human,
        )

    def add_edges_db(self, quads):
        aid_1_list = list(
            map(convert_lca_node_id_to_wbia_annot_id, ut.take_column(quads, 0))
        )
        aid_2_list = list(
            map(convert_lca_node_id_to_wbia_annot_id, ut.take_column(quads, 1))
        )
        value_list = ut.take_column(quads, 2)
        aug_name_list = ut.take_column(quads, 3)
        identity_list = convert_aug_name_to_identity(aug_name_list)

        weight_rowid_list = self.ibs.add_edge_weight(
            aid_1_list, aid_2_list, value_list, identity_list
        )
        self._cleanup_edges()
        return weight_rowid_list

    def edges_from_attributes_db(self, n0, n1):
        n0_ = convert_lca_node_id_to_wbia_annot_id(n0)
        n1_ = convert_lca_node_id_to_wbia_annot_id(n1)
        edges = [(n0_, n1_)]
        weight_rowid_list = self.ibs.get_edge_weight_rowids_from_edges(edges)
        weight_rowid_list = weight_rowid_list[0]
        weight_rowid_list = sorted(weight_rowid_list)

        aid_1_list = [n0] * len(weight_rowid_list)
        aid_2_list = [n1] * len(weight_rowid_list)
        value_list = self.ibs.get_edge_weight_value(weight_rowid_list)
        identity_list = self.ibs.get_edge_weight_identity(weight_rowid_list)
        aug_name_list = convert_identity_to_aug_name(identity_list)

        quads = list(zip(aid_1_list, aid_2_list, value_list, aug_name_list))

        num_auto = aug_name_list.count(ALGO_AUG_NAME)
        num_human = aug_name_list.count(HUMAN_AUG_NAME)
        assert num_auto <= self.max_auto_reviews
        assert num_human <= self.max_human_reviews
        assert len(quads) <= self.max_reviews

        return quads

    def commit_cluster_change_db(self, cc):
        """
        Two jobs: (1) create a change dictionary with WBIA names, and
        (2) save clusters as WBIA names. The latter adds NIDs if they didn't
        exist before and removes NIDs that have changed more than just as
        the addition of query annotations. This will likely require REVISITING
        because it makes no effort to preserve metadata that is associated
        with names that have been removed.
        """
        logger.info(
            '[commit_cluster_change_db] Requested to commit cluster change: %r' % (cc,)
        )
        change = cc.serialize()

        change['type'] = change.pop('change_type').lower()

        change['added'] = sorted(
            list(
                map(
                    convert_lca_node_id_to_wbia_annot_id,
                    change.pop('query_nodes'),
                )
            )
        )

        change['removed'] = sorted(
            list(
                map(
                    convert_lca_node_id_to_wbia_annot_id,
                    change.pop('removed_nodes'),
                )
            )
        )
        change['old'] = []
        change['new'] = []

        logger.info(f"Change type {change['type']}")

        old_clustering = change.pop('old_clustering')
        nids_to_remove = []
        for old_lca_cluster_id in old_clustering:
            old_lca_node_ids = old_clustering[old_lca_cluster_id]
            wbia_aids = list(
                map(convert_lca_node_id_to_wbia_annot_id, old_lca_node_ids)
            )
            change['old'].append(wbia_aids)
            old_nids = self.ibs.get_annot_nids(wbia_aids)
            old_nids = list(set([nid for nid in old_nids if nid > 0]))

            logger.info(f'Old name id {old_nids}, annotations {wbia_aids}')

            if len(old_nids) > 1:
                logger.info(
                    'Error: annots in this old cluster are spread across multiple names'
                )
                nids_to_remove.extend(old_nids)  # Set to remove these
            elif len(old_nids) == 1 and change['type'] in [
                'removed',
                'split',
                'merge',
                'merge/split',
            ]:
                nids_to_remove.extend(old_nids)

        if len(nids_to_remove) > 0:
            logger.info(f'Deleting old nids {nids_to_remove}')
            self.ibs.delete_names(nids_to_remove)

        new_clustering = change.pop('new_clustering')
        for new_lca_cluster_id in new_clustering:
            new_lca_node_ids = new_clustering[new_lca_cluster_id]
            wbia_aids = list(
                map(convert_lca_node_id_to_wbia_annot_id, new_lca_node_ids)
            )
            change['new'].append(wbia_aids)
            prev_nids = self.ibs.get_annot_nids(wbia_aids)
            prev_nids = list(set([nid for nid in prev_nids if nid > 0]))

            logger.info(f'Previous nids {prev_nids} for cluster of aids {wbia_aids}')
            if len(prev_nids) == 0 or change['type'] in [
                'split',
                'merge',
                'merge/split',
            ]:
                self.ibs.set_annot_names_to_next_name(wbia_aids)
                new_nid = self.ibs.get_annot_nids(wbia_aids[0])
                logger.info(f'Created new name nid {new_nid} for aids {wbia_aids}')
            elif change['type'] == 'extension':
                assert len(prev_nids) == 1
                prev_aids = self.ibs.get_name_aids(prev_nids[0])
                new_aids = list(set(wbia_aids) - set(prev_aids))
                self.ibs.set_annot_name_rowids(new_aids, prev_nids * len(new_aids))
                logger.info(
                    f'Added aids {new_aids} to prev aids {prev_aids} for nid {prev_nids[0]}'
                )
            else:
                logger.info(f'Cluster {wbia_aids} with nid {prev_nids[0]} is unchanged')

        return change


class edge_generator_wbia(edge_generator.edge_generator):  # NOQA
    def _cleanup_edges(self):
        clean_edge_requests = []
        for edge in self.edge_requests:
            n0, n1, aug_name = edge
            aid1 = convert_lca_node_id_to_wbia_annot_id(n0)
            aid2 = convert_lca_node_id_to_wbia_annot_id(n1)
            if aid1 > aid2:
                aid1, aid2 = aid2, aid1
            n0 = convert_wbia_annot_id_to_lca_node_id(aid1)
            n1 = convert_wbia_annot_id_to_lca_node_id(aid2)
            clean_edge = (n0, n1, aug_name)
            clean_edge_requests.append(clean_edge)
        self.edge_requests = clean_edge_requests

    def get_edge_requests(self):
        self._cleanup_edges()
        return self.edge_requests

    def set_edge_requests(self, new_edge_requests):
        self.edge_requests = new_edge_requests
        self._cleanup_edges()
        return self.edge_requests

    def edge_request_cb_async(self):
        actor = self.controller

        requested_auto_edges = []
        keep_edge_requests = []
        for edge in self.get_edge_requests():
            n0, n1, aug_name = edge
            if is_aug_name_algo(aug_name):
                aid1 = convert_lca_node_id_to_wbia_annot_id(n0)
                aid2 = convert_lca_node_id_to_wbia_annot_id(n1)
                requested_auto_edges.append((aid1, aid2))
            else:
                keep_edge_requests.append(edge)

        request_data = actor._candidate_edge_probs(
            requested_auto_edges, update_infr=True
        )
        (
            requested_auto_probs,
            requested_auto_prob_quads,
            requested_auto_quads,
        ) = request_data
        self.edge_results += requested_auto_quads
        self.set_edge_requests(keep_edge_requests)

        args = (
            len(requested_auto_edges),
            len(requested_auto_quads),
            len(self.edge_results),
            len(keep_edge_requests),
        )
        logger.info(
            'Received %d Verifier edge requests, added %d new results for %d total, kept %d requests in queue'
            % args
        )

    def add_feedback(
        self,
        edge,
        evidence_decision=None,
        tags=None,
        user_id=None,
        meta_decision=None,
        confidence=None,
        timestamp_c1=None,
        timestamp_c2=None,
        timestamp_s1=None,
        timestamp=None,
        verbose=None,
        priority=None,
    ):
        aid1, aid2 = edge

        if evidence_decision is None:
            evidence_decision = UNREV
        if meta_decision is None:
            meta_decision = const.META_DECISION.CODE.NULL
        decision = _rectify_decision(evidence_decision, meta_decision)

        if decision == POSTV:
            flag = True
        elif decision == NEGTV:
            flag = False
        elif decision == INCMP:
            flag = None
        else:
            # UNREV, UNKWN
            return

        n0 = convert_wbia_annot_id_to_lca_node_id(aid1)
        n1 = convert_wbia_annot_id_to_lca_node_id(aid2)

        human_triples = [
            (n0, n1, flag),
        ]
        new_edge_results = self.new_edges_from_human(human_triples)
        self.edge_results += new_edge_results

        # Remove edge request for this pair now that a result has been returned
        found_edge_requests = []
        keep_edge_requests = []
        for edge in self.get_edge_requests():
            n0_, n1_, aug_name = edge
            if is_aug_name_human(aug_name):
                if n0 == n0_ and n1 == n1_:
                    found_edge_requests.append(edge)
                    continue
            keep_edge_requests.append(edge)
        args = (
            len(found_edge_requests),
            len(keep_edge_requests),
        )
        logger.info(
            'Found %d human edge requests to remove, kept %d requests in queue' % args
        )
        self.set_edge_requests(keep_edge_requests)


class LCAActor(GraphActor):
    """

    CommandLine:
        python -m wbia_lca._plugin LCAActor
        python -m wbia_lca._plugin LCAActor:0

    Doctest:
        >>> from wbia.web.graph_server import _testdata_feedback_payload
        >>> import wbia
        >>> actor = LCAActor()
        >>> # Start the process
        >>> # dbdir = wbia.sysres.db_to_dbdir('GZ_CensusAnnotation_Eval')
        >>> dbdir = wbia.sysres.db_to_dbdir('PZ_MTEST', allow_newdir=True)
        >>> payload = {'action': 'start', 'dbdir': dbdir, 'aids': 'all'}
        >>> start_resp = actor.handle(payload)
        >>> print('start_resp = {!r}'.format(start_resp))
        >>> # Respond with a user decision
        >>> user_request = actor.handle({'action': 'resume'})
        >>> # Wait for a response and the LCAActor in another proc
        >>> if len(user_request) > 0:
        >>>     edge, priority, edge_data = user_request[0]
        >>>     user_resp_payload = _testdata_feedback_payload(edge, 'match')
        >>>     content = actor.handle(user_resp_payload)
    """

    def __init__(
        actor, *args, ranker='hotspotter', verifier='vamp', num_waiting=10, **kwargs
    ):
        actor.infr = None
        actor.graph_uuid = None

        actor.warmup = True

        actor.db = None
        actor.edge_gen = None
        actor.driver = None
        actor.ga_gen = None
        actor.changes = None

        actor.resume_lock = threading.Lock()

        actor.phase = 0
        actor.loop_phase = 'init'

        # fmt: off
        actor.infr_config = {
            # 'manual.n_peek': 100,
            # 'autoreview.enabled': True,
            # 'autoreview.prioritize_nonpos': True,
            # 'inference.enabled': True,
            # 'ranking.enabled': True,
            # 'ranking.ntop': 5,
            # 'redun.enabled': True,
            # 'redun.enforce_neg': True,
            # 'redun.enforce_pos': True,
            # 'redun.neg.only_auto': False,
            # 'redun.neg': 2,
            # 'redun.pos': 2,
            # 'algo.hardcase': False,

            'autoreview.enabled': True,
            'autoreview.prioritize_nonpos': True,
            'inference.enabled': True,
            'ranking.enabled': True,
            'ranking.ntop': 10,
            'redun.enabled': True,
            'redun.enforce_neg': True,
            'redun.enforce_pos': True,
            'redun.neg.only_auto': False,
            'redun.neg': 2,
            'redun.pos': 2,
            'refresh.window': 20,
            'refresh.patience': 20,
            'refresh.thresh': np.exp(-2),

            # 'autoreview.enabled': False,
            # 'inference.enabled': True,
            # 'ranking.enabled': True,
            # 'ranking.ntop': 10,
            # 'redun.enabled': False,
            # 'algo.hardcase': False,
        }

        if ranker == 'hotspotter':
            actor.ranker_config = {}
        elif ranker == 'pie_v2':
            actor.ranker_config = {
                'pipeline_root': 'PieTwo',
                'use_knn': False,
            }
        else:
            raise ValueError('Unsupported Ranker')

        if verifier == 'vamp':
            actor.verifier_config = {
                'verifier': 'vamp',
                'load_verifier_gt_filepath': None,
                'save_verifier_gt_filepath': None,
            }
        elif verifier == 'vamp+':
            actor.verifier_config = {
                'verifier': 'vamp',
                'load_verifier_gt_filepath': '/data/db/lca.verifier.zebra_grevys.canonical.pkl',
                'save_verifier_gt_filepath': '/data/db/lca.verifier.zebra_grevys.canonical.pkl',
            }
        elif verifier == 'pie_v2':
            actor.verifier_config = {
                'verifier': 'pie_v2',
            }
        else:
            raise ValueError('Unsupported Verifier')

        actor.lca_config = {
            'aug_names': [
                ALGO_AUG_NAME,
                HUMAN_AUG_NAME,
            ],
            'prob_human_correct': HUMAN_CORRECT_RATE,

            # DEFAULT
            # 'min_delta_converge_multiplier': 0.95,
            # 'min_delta_stability_ratio': 8,
            # 'num_per_augmentation': 2,
            # 'tries_before_edge_done': 4,
            # 'ga_max_num_waiting': 1000,

            # EXTENSIVE
            'min_delta_converge_multiplier': 0.98,   # was 1.5,
            'min_delta_stability_ratio': 4,
            'num_per_augmentation': 2,
            'tries_before_edge_done': 4,
            'ga_max_num_waiting': num_waiting,

            'ga_iterations_before_return': 100,  # IS THIS USED?

            'log_level': logging.INFO,
            'log_file': LOG_LCS_FILE,

            'draw_iterations': False,
            'drawing_prefix': 'wbia_lca',
        }

        prob_human_correct = actor.lca_config.get('prob_human_correct', HUMAN_CORRECT_RATE)
        actor.config = {
            'warmup.n_peek': 500,
            'weighter_required_reviews': 50,
            'weighter_recent_reviews': 1000,
            'init_nids': [],
            'autoreview.enabled': USE_AUTOREVIEW,
            'autoreview.prob_human_correct': prob_human_correct,
        }
        # fmt: on

        from wbia_lca import formatter

        handler = logging.FileHandler(actor.lca_config['log_file'])
        handler.setLevel(actor.lca_config['log_level'])
        handler.setFormatter(formatter)
        logger.addHandler(handler)

        super(LCAActor, actor).__init__(*args, **kwargs)

    def _init_infr(actor, aids, dbdir, **kwargs):
        import wbia

        assert dbdir is not None, 'must specify dbdir'
        assert actor.infr is None, 'AnnotInference already running'
        ibs = wbia.opendb(dbdir=dbdir, use_cache=False, web=False, force_serial=True)

        # Create the reference AnnotInference
        logger.info('starting via actor with ibs = %r' % (ibs,))
        actor.infr = wbia.AnnotInference(ibs=ibs, aids=aids, autoinit=True)
        actor.infr.print('started via actor')
        actor.infr.print('config = {}'.format(ut.repr3(actor.infr_config)))

        # Configure
        for key in actor.infr_config:
            actor.infr.params[key] = actor.infr_config[key]

        # Pull reviews from staging
        actor.infr.print('Initializing infr tables')
        if not USE_COLDSTART:
            table = kwargs.get('init', 'staging')
            actor.infr.reset_feedback(table, apply=True)
        actor.infr.ensure_mst()
        actor.infr.apply_nondynamic_update()

        actor.infr.print('infr.status() = {}'.format(ut.repr4(actor.infr.status())))

        # Load Verifier models for Verifier
        actor.infr.print('loading published models')
        actor.infr.load_published()

        if USE_COLDSTART:
            actor.infr.reset(state='empty')

        assert actor.infr is not None

    def _get_edge_quads_ext_using_reviews(
        actor, delay_compute=False, desired_aug_name=None
    ):
        assert actor.infr is not None

        review_rowid_list = actor.infr.ibs.get_review_rowids_between(actor.infr.aids)
        review_edge_list = actor.infr.ibs.get_review_aid_tuple(review_rowid_list)
        review_decision_list = actor.infr.ibs.get_review_decision(review_rowid_list)
        review_identity_list = actor.infr.ibs.get_review_identity(review_rowid_list)
        review_aug_name_list = convert_identity_to_aug_name(review_identity_list)

        if delay_compute:
            review_prob_list = [None] * len(review_rowid_list)
        else:
            review_prob_list, _, _ = actor._candidate_edge_probs(review_edge_list)

        quads_ext = []
        zipped = zip(
            review_edge_list,
            review_decision_list,
            review_prob_list,
            review_aug_name_list,
        )
        for review_edge, review_decision, review_prob, review_aug_name in zipped:
            if desired_aug_name is not None:
                if review_aug_name != desired_aug_name:
                    continue
            aid1, aid2 = review_edge
            n0 = convert_wbia_annot_id_to_lca_node_id(aid1)
            n1 = convert_wbia_annot_id_to_lca_node_id(aid2)
            review_decision_code = const.EVIDENCE_DECISION.INT_TO_CODE[review_decision]
            quad_ext = (n0, n1, review_decision_code, review_prob, review_aug_name)
            quads_ext.append(quad_ext)

        return quads_ext

    def _init_edge_weights_using_reviews(actor, desired_aug_name=None):
        assert actor.edge_gen is not None
        quads_ext = actor._get_edge_quads_ext_using_reviews(
            desired_aug_name=desired_aug_name
        )
        quads_ = [
            (aid1, aid2, weight, aug_name)
            for aid1, aid2, decision, weight, aug_name in quads_ext
        ]
        quads = actor.edge_gen.new_edges_from_verifier(quads_, db_add=False)
        return quads, quads_ext

    def _init_weighter(actor):
        logger.info('Attempting to warmup (_init_weighter)')

        assert actor.infr is not None

        load_verifier_gt_filepath = actor.verifier_config.get(
            'load_verifier_gt_filepath', None
        )
        save_verifier_gt_filepath = actor.verifier_config.get(
            'save_verifier_gt_filepath', None
        )

        if load_verifier_gt_filepath is None:
            quads_ext = actor._get_edge_quads_ext_using_reviews(delay_compute=True)
            logger.info('Fetched %d reviews' % (len(quads_ext),))

            verifier_gt = {
                ALGO_AUG_NAME: {
                    'gt_positive_probs': [],
                    'gt_negative_probs': [],
                }
            }
            for n0, n1, decision, weight, aug_name in quads_ext:
                edge = (
                    convert_lca_node_id_to_wbia_annot_id(n0),
                    convert_lca_node_id_to_wbia_annot_id(n1),
                )
                if not is_aug_name_human(aug_name):
                    continue
                if decision == POSTV:
                    key = 'gt_positive_probs'
                elif decision == NEGTV:
                    key = 'gt_negative_probs'
                else:
                    key = None
                if key is not None:
                    verifier_gt[ALGO_AUG_NAME][key].append(edge)

            for algo in verifier_gt:
                for key in verifier_gt[algo]:
                    edges = verifier_gt[algo][key]
                    num_edges_ = len(edges)
                    edges = list(set(edges))
                    num_edges = len(edges)
                    min_edges = actor.config.get('weighter_required_reviews')
                    max_edges = actor.config.get('weighter_recent_reviews')
                    logger.info(
                        'Found %d de-duplicated review edges (from %d total) for %s %s'
                        % (
                            num_edges,
                            num_edges_,
                            algo,
                            key,
                        )
                    )
                    if num_edges < min_edges:
                        args = (
                            key,
                            num_edges,
                            min_edges,
                        )
                        logger.info(
                            'WARMUP failed: key %r has %d edges, needs %d' % args
                        )
                        return False

                    thresh_edges = -1 * min(num_edges, max_edges)
                    random.seed(1)
                    random.shuffle(edges)
                    edges = edges[thresh_edges:]
                    probs, _, _ = actor._candidate_edge_probs(edges)

                    # Filter out outliers
                    probs = np.array(probs)
                    mean_probs = np.mean(probs)
                    std_probs = np.std(probs)
                    min_probs = mean_probs - (std_probs * 2.0)
                    max_probs = mean_probs + (std_probs * 2.0)
                    logger.info(
                        'Discarding outlies in %d review edges with [%0.02f <- %0.02f +/- %0.02f -> %0.02f]'
                        % (
                            len(probs),
                            min_probs,
                            mean_probs,
                            std_probs,
                            max_probs,
                        )
                    )
                    probs = [
                        prob
                        for prob in probs
                        if min_probs <= prob and prob <= max_probs
                    ]
                    logger.info('Keeping %d review edges' % (len(probs),))

                    verifier_gt[algo][key] = probs
        else:
            verifier_gt = ut.load_cPkl(load_verifier_gt_filepath)

        if save_verifier_gt_filepath is not None:
            ut.save_cPkl(save_verifier_gt_filepath, verifier_gt)

        logger.info(ut.repr3(verifier_gt))

        wgtrs = ga_driver.generate_weighters(actor.lca_config, verifier_gt)
        actor.wgtr = wgtrs[0]

        # Update delta score thresholds
        multiplier = actor.lca_config['min_delta_converge_multiplier']
        ratio = actor.lca_config['min_delta_stability_ratio']

        human_gt_positive_weight = actor.wgtr.human_wgt(is_marked_correct=True)
        human_gt_negative_weight = actor.wgtr.human_wgt(is_marked_correct=False)

        human_gt_delta_weight = human_gt_positive_weight - human_gt_negative_weight
        convergence = -1.0 * multiplier * human_gt_delta_weight
        stability = convergence / ratio

        actor.lca_config['min_delta_score_converge'] = convergence
        actor.lca_config['min_delta_score_stability'] = stability

        logger.info(
            'Using provided   min_delta_converge_multiplier = %0.04f' % (multiplier,)
        )
        logger.info(
            'Using provided   min_delta_stability_ratio     = %0.04f' % (ratio,)
        )
        logger.info(
            'Using calculated min_delta_score_converge      = %0.04f' % (convergence,)
        )
        logger.info(
            'Using calculated min_delta_score_stability     = %0.04f' % (stability,)
        )

        return True

    def _init_lca(actor):
        # Initialize the weighter
        success = actor._init_weighter()
        if not success:
            return

        # Initialize the DB
        actor.db = db_interface_wbia(actor)

        # Initialize the Edge Generator
        actor.edge_gen = edge_generator_wbia(actor.db, actor.wgtr, controller=actor)

        # We have warmed up
        actor.warmup = False

    def start(actor, dbdir, aids='all', config={}, graph_uuid=None, **kwargs):
        actor.config.update(config)

        # Initialize INFR
        actor._init_infr(aids, dbdir, **kwargs)
        actor.graph_uuid = graph_uuid

        # Initialize LCA
        actor._init_lca()

        # Initialize the review iterator
        actor._gen = actor.main_gen()
        status = 'warmup' if actor.warmup else 'initialized'
        logger.info(f'Leaving start with status {status}')

        return status

    def _candidate_edge_probs_auto(actor, candidate_edges, update_infr=False):
        task_probs = actor.infr._make_task_probs(candidate_edges)
        match_probs = list(task_probs['match_state']['match'])
        nomatch_probs = list(task_probs['match_state']['nomatch'])

        if update_infr:
            match_thresh = actor.infr.task_thresh['match_state']['match']
            nomatch_thresh = actor.infr.task_thresh['match_state']['nomatch']

            zipped = zip(candidate_edges, match_probs, nomatch_probs)
            for candidate_edge, match_prob, nomatch_prob in zipped:
                if match_prob >= match_thresh:
                    evidence_decision = POSTV
                elif nomatch_prob >= nomatch_thresh:
                    evidence_decision = NEGTV
                else:
                    evidence_decision = None

                if evidence_decision is not None:
                    feedback = {
                        'edge': candidate_edge,
                        'user_id': ALGO_IDENTITY,
                        'confidence': const.CONFIDENCE.CODE.PRETTY_SURE,
                        'evidence_decision': evidence_decision,
                        'meta_decision': NULL,
                        'timestamp': None,
                        'timestamp_s1': None,
                        'timestamp_c1': None,
                        'timestamp_c2': None,
                        'tags': [],
                    }
                    actor.infr.add_feedback(**feedback)
            actor.infr.write_wbia_staging_feedback()

        candidate_probs = []
        for match_prob, nomatch_prob in zip(match_probs, nomatch_probs):
            prob_ = 0.5 + (match_prob - nomatch_prob) / 2
            candidate_probs.append(prob_)

        return candidate_probs

    def _candidate_edge_probs_pie_v2(actor, candidate_edges):
        from wbia_pie_v2._plugin import distance_to_score

        # Ensure features
        actor.infr.ibs.pie_v2_embedding(actor.infr.aids)

        candidate_probs = []
        for edge in tqdm.tqdm(candidate_edges):
            qaid, daid = edge
            pie_annot_distances = actor.infr.ibs.pie_v2_predict_light_distance(
                qaid,
                [daid],
            )
            pie_annot_distance = pie_annot_distances[0]
            score = distance_to_score(pie_annot_distance, norm=500.0)
            candidate_probs.append(score)

        return candidate_probs

    def _candidate_edge_probs(actor, candidate_edges, update_infr=False):
        if len(candidate_edges) == 0:
            return [], [], []

        verifier_algo = actor.verifier_config.get('verifier', 'vamp')
        if verifier_algo == 'vamp':
            candidate_probs = actor._candidate_edge_probs_auto(
                candidate_edges, update_infr=update_infr
            )
        elif verifier_algo == 'pie_v2':
            candidate_probs = actor._candidate_edge_probs_pie_v2(candidate_edges)
        else:
            raise ValueError(
                'Verifier algorithm %r is not supported' % (verifier_algo,)
            )

        num_probs = len(candidate_probs)
        min_probs = None if num_probs == 0 else '%0.04f' % (min(candidate_probs),)
        max_probs = None if num_probs == 0 else '%0.04f' % (max(candidate_probs),)
        mean_probs = None if num_probs == 0 else '%0.04f' % (np.mean(candidate_probs),)
        std_probs = None if num_probs == 0 else '%0.04f' % (np.std(candidate_probs),)

        args = (num_probs, min_probs, max_probs, mean_probs, std_probs)
        logger.info(
            'Verifier probabilities on %d edges (range: %s - %s, mean: %s +/- %s)'
            % args
        )
        # logger.info(ut.repr2(list(zip(candidate_edges, candidate_probs))))

        if actor.edge_gen is None:
            candidate_prob_quads = None
            candidate_quads = None
        else:
            candidate_prob_quads = [
                (
                    convert_wbia_annot_id_to_lca_node_id(aid1),
                    convert_wbia_annot_id_to_lca_node_id(aid2),
                    prob,
                    ALGO_AUG_NAME,
                )
                for (aid1, aid2), prob in zip(candidate_edges, candidate_probs)
            ]
            candidate_quads = actor.edge_gen.new_edges_from_verifier(
                candidate_prob_quads, db_add=False
            )

        return candidate_probs, candidate_prob_quads, candidate_quads

    def _refresh_data(actor, warmup=False, desired_states=None):
        if desired_states is None:
            desired_states = [[POSTV, NEGTV, INCMP, UNKWN, UNREV]]
            # desired_states = [desired_states] + desired_states

        """
        # Reset ranker_params to empty
        old_ranker_params = actor.infr.ranker_params
        actor.infr.ranker_params = {}
        logger.info('In _refresh_data')

        # Run Ranker to find matches
        if USE_COLDSTART:
            logger.info('Using cold start')
            candidate_edges = actor.infr.find_lnbnn_candidate_edges(
                cfgdict_=actor.ranker_config
            )
        else:
            logger.info('No cold start')
            candidate_edges = []
            for desired_states_ in desired_states:
                for K in [5]:  # [3, 5, 7]:
                    for Knorm in [5]:  # [3, 5, 7]:
                        for score_method in ['csum']:  # #['csum', 'nsum']:
                            candidate_edges += actor.infr.find_lnbnn_candidate_edges(
                                desired_states=desired_states_,
                                can_match_samename=True,
                                K=K,
                                Knorm=Knorm,
                                prescore_method=score_method,
                                score_method=score_method,
                                requery=False,
                                cfgdict_=actor.ranker_config,
                            )
                            candidate_edges += actor.infr.find_lnbnn_candidate_edges(
                                desired_states=desired_states_,
                                can_match_samename=False,
                                K=K,
                                Knorm=Knorm,
                                prescore_method=score_method,
                                score_method=score_method,
                                requery=False,
                                cfgdict_=actor.ranker_config,
                            )

        # Reset ranker_params to default
        actor.infr.ranker_params = old_ranker_params

        candidate_edges = list(set(candidate_edges))
        logger.info('Edges from ranking %d' % (len(candidate_edges),))

        # Run Verifier on candidates
        candidate_probs, _, candidate_quads = actor._candidate_edge_probs(candidate_edges)
        """
        candidate_edges, candidate_probs, candidate_quads = [], [], []

        # Requested warm-up, return this data immediately
        if warmup:
            warmup_data = candidate_edges, candidate_probs
            logger.info(f'Returning with {len(candidate_edges)} candidate edges')
            return warmup_data

        assert None not in [actor.infr, actor.db, actor.edge_gen]

        # Initialize edge weights from reviews
        if USE_COLDSTART:
            # Clear out all existing human edge weights, we will repopulate using reviews
            actor.db._cleanup_edges(max_human=0, max_auto=0)
            review_quads, review_quads_ext = [], []
        else:
            # Clear out all existing human edge weights, we will repopulate using reviews
            actor.db._cleanup_edges(max_human=0)
            review_quads, review_quads_ext = actor._init_edge_weights_using_reviews()

        actor.db.add_edges_db(review_quads)

        # Initialize the edge weights from Ranker
        actor.db.add_edges_db(candidate_quads)

        # Collect verifier results from Ranker matches and Verifier scores
        weight_rowid_list = actor.infr.ibs.get_edge_weight_rowids_between(
            actor.infr.aids
        )
        weight_edge_list = actor.infr.ibs.get_edge_weight_aid_tuple(weight_rowid_list)
        weight_edge_list = list(set(weight_edge_list))

        # Update all Verifier edges in database
        _, verifier_prob_quads, verifier_quads = actor._candidate_edge_probs(
            weight_edge_list
        )
        actor.db.add_edges_db(verifier_quads)

        verifier_results = verifier_prob_quads
        logger.info('Using %d Verifier edge weights' % (len(verifier_results),))

        # Collect human decisions
        human_decisions = []
        for aid1, aid2, decision, weight, aug_name in review_quads_ext:
            if not is_aug_name_human(aug_name):
                continue
            if decision == POSTV:
                flag = True
            elif decision == NEGTV:
                flag = False
            elif decision == INCMP:
                flag = None
            else:
                # UNREV, UNKWN
                continue
            human_decision = (aid1, aid2, flag)
            human_decisions.append(human_decision)
        logger.info('Using %d human decisions' % (len(human_decisions),))

        # Purge database of edges
        actor.db._cleanup_edges(max_human=0, max_auto=0)
        weight_rowid_list = actor.infr.ibs.get_edge_weight_rowids_between(
            actor.infr.aids
        )
        assert len(weight_rowid_list) == 0

        # Get the clusters to check
        cluster_ids_to_check = actor.config.get('init_nids')

        driver_data = verifier_results, human_decisions, cluster_ids_to_check
        return driver_data

    def _make_review_tuple(actor, edge, priority=1.0):
        """Makes tuple to be sent back to the user"""
        edge_data = actor.infr.get_nonvisual_edge_data(edge, on_missing='default')
        # Extra information
        edge_data['nid_edge'] = None
        if actor.edge_gen is None:
            edge_data['queue_len'] = 0
        else:
            edge_data['queue_len'] = len(actor.edge_gen.edge_requests)
        edge_data['n_ccs'] = (-1, -1)
        return (edge, priority, edge_data)

    def _attempt_autoreview(actor, user_request):
        if actor.config.get('autoreview.enabled'):
            for review_request in user_request:
                edge, priority, edge_data = review_request
                aid1, aid2 = edge

                review_rowid_list = actor.infr.ibs.get_review_rowids_from_edges([edge])[
                    0
                ]
                review_rowid_list = sorted(review_rowid_list)
                review_decision_list = actor.infr.ibs.get_review_decision(
                    review_rowid_list
                )
                review_identity_list = actor.infr.ibs.get_review_identity(
                    review_rowid_list
                )
                flag_list = [
                    review_identity.startswith(HUMAN_IDENTITY)
                    for review_identity in review_identity_list
                ]
                real_decisions = ut.compress(review_decision_list, flag_list)
                # Take the most recent review
                real_decision = None if len(real_decisions) == 0 else real_decisions[-1]

                if real_decision is None:
                    name1, name2 = actor.infr.ibs.get_annot_names([aid1, aid2])
                    oracle = random.uniform(0.0, 1.0)
                    prob_human_correct = actor.config.get(
                        'autoreview.prob_human_correct'
                    )
                    prob_human_incorrect = 1.0 - prob_human_correct
                    throw_incorrect = oracle <= prob_human_incorrect

                    if const.UNKNOWN in [name1, name2]:
                        evidence_decision = None
                    elif name1 == name2:
                        evidence_decision = NEGTV if throw_incorrect else POSTV
                    elif name1 != name2:
                        evidence_decision = POSTV if throw_incorrect else NEGTV
                    else:
                        raise ValueError()

                    if throw_incorrect:
                        message = (
                            ' (THROWING INTENTIONALLY INCORRECT DECISION, p=%0.02f)'
                            % (prob_human_correct,)
                        )
                    else:
                        message = ''

                    decision_source = 'inferred'
                else:
                    evidence_decision = const.EVIDENCE_DECISION.INT_TO_CODE[
                        real_decision
                    ]
                    message = ''
                    decision_source = 'matched'

                args = (
                    edge,
                    evidence_decision,
                    message,
                    decision_source,
                )
                logger.info('HUMAN AUTOREVIEWING EDGE %r -> %r%s [%s]' % args)

                feedback = {
                    'edge': edge,
                    'user_id': AUTOREVIEW_IDENTITY,
                    'confidence': const.CONFIDENCE.CODE.PRETTY_SURE,
                    'evidence_decision': evidence_decision,
                    'meta_decision': NULL,
                    'timestamp': None,
                    'timestamp_s1': None,
                    'timestamp_c1': None,
                    'timestamp_c2': None,
                    'tags': [],
                }
                actor.feedback(**feedback)
            return None
        else:
            return user_request

    def main_gen(actor):
        actor.phase = 0
        actor.loop_phase = 'warmup'

        while actor.warmup:
            logger.info('WARMUP: Computing warmup data')

            # We are still in warm-up, need to ask user for reviews
            warmup_data = actor._refresh_data(warmup=True, desired_states=[[UNREV]])
            candidate_edges, candidate_probs = warmup_data
            candidate_probs_ = list(
                map(int, np.around(np.array(candidate_probs) * 10.0))
            )

            # Create stratified buckets based on probabilities
            candidate_buckets = {}
            for candidate_edge, candidate_prob_ in zip(
                candidate_edges, candidate_probs_
            ):
                if candidate_prob_ not in candidate_buckets:
                    candidate_buckets[candidate_prob_] = []
                candidate_buckets[candidate_prob_].append(candidate_edge)
            buckets = list(candidate_buckets.keys())
            logger.info('WARMUP: Creating stratified buckets: %r' % (buckets,))

            num = actor.config.get('warmup.n_peek')
            user_request = []
            for index in range(num):
                if len(buckets) == 0:
                    continue
                bucket = random.choice(buckets)
                edges = candidate_buckets[bucket]
                edge = random.choice(edges)
                args = (
                    bucket,
                    edge,
                )
                # logger.info('WARMUP: bucket %r, edge %r' % args)
                # create a bunch of random edges (from stratified buckets) to the user
                user_request += [actor._make_review_tuple(edge)]

            user_request = actor._attempt_autoreview(user_request)
            if user_request is not None:
                logger.info(f'Yielding a user request of length {len(user_request)}')
                yield user_request

            # Try to re-initialize LCA
            actor._init_lca()

        # Get existing clustering of names before processing has started
        other_clustering = actor.db._get_existing_clustering(
            use_ibeis_database=USE_COLDSTART
        )

        actor.phase = 1
        actor.loop_phase = 'driver'

        if actor.driver is None:
            # Get driver data
            assert not actor.warmup
            driver_data = actor._refresh_data()
            verifier_results, human_decisions, cluster_ids_to_check = driver_data

            # Initialize the Driver
            actor.driver = ga_driver.ga_driver(
                verifier_results,
                human_decisions,
                cluster_ids_to_check,
                actor.db,
                actor.edge_gen,
                actor.lca_config,
            )

        actor.phase = 2
        actor.loop_phase = 'run_all_ccPICs'

        with open(LOG_DECISION_FILE, 'a') as logfile:
            header = (
                'ITER',
                'NAMES',
                'NAMES_16',
                'PL_INDEX_16',
                'PL_CI_16',
                'DAY1_16',
                'DAY2_16',
                'RESIGHT_16',
                'COVERAGE_16',
                'NAMES_18',
                'PL_INDEX_18',
                'PL_CI_18',
                'DAY1_18',
                'DAY2_18',
                'RESIGHT_18',
                'COVERAGE_18',
                'RESIGHT_16_18',
                'AUTO',
                'HUMAN',
                'TODO',
            )
            data = [''] * len(header)
            line = ','.join(map(str, data))
            logfile.write('%s\n' % (line,))
            line = ','.join(map(str, header))
            logfile.write('%s\n' % (line,))

        partial_progress_cb = partial(progress_db, actor)
        actor.ga_gen = actor.driver.run_all_ccPICs(
            yield_on_paused=True,
            progress_cb=partial_progress_cb,
            other_clustering=other_clustering,
        )

        changes_to_review = []
        while True:
            try:
                change_to_review = next(actor.ga_gen)
            except StopIteration:
                break

            if change_to_review is None:
                requested_human_edges = []
                for edge in actor.edge_gen.get_edge_requests():
                    n0, n1, aug_name = edge
                    if is_aug_name_human(aug_name):
                        aid1 = convert_lca_node_id_to_wbia_annot_id(n0)
                        aid2 = convert_lca_node_id_to_wbia_annot_id(n1)
                        requested_human_edges.append((aid1, aid2))

                args = (len(requested_human_edges),)
                logger.info('Received %d human edge requests' % args)

                user_request = []
                for edge in requested_human_edges:
                    user_request += [actor._make_review_tuple(edge)]

                user_request = actor._attempt_autoreview(user_request)
                if user_request is not None:
                    yield user_request
            else:
                logger.info(f'Change to review type {type(change_to_review)}')
                if isinstance(change_to_review, list):
                    logger.info(f'Length {len(change_to_review)}')

                changes_to_review.append(change_to_review)

        actor.phase = 3
        actor.loop_phase = 'commit_cluster_change'

        actor.changes = []
        c_count = 0
        for changes in changes_to_review:
            for cc in changes:
                change = actor.db.commit_cluster_change(cc)
                logger.info(f'Cluster change {c_count} for WBIA')
                if change is None:
                    logger.info('None!')
                elif isinstance(change, dict):
                    for k, v in change.items():
                        logger.info(f'   {k}: {v}')
                else:
                    logger.info(f'type is {type(change)}')
                c_count += 1
                if change is not None:
                    actor.changes.append(change)

        actor.phase = 4
        actor.loop_phase = None

        return 'finished:main'

    def resume(actor):
        with actor.resume_lock:
            if actor._gen is None:
                return 'finished:stopped'
            try:
                user_request = next(actor._gen)
            except StopIteration:
                actor._gen = None
                user_request = 'finished:stopiteration'
            return user_request

    def feedback(actor, **feedback):
        actor.infr.add_feedback(**feedback)
        actor.infr.write_wbia_staging_feedback()
        if actor.edge_gen is not None:
            actor.edge_gen.add_feedback(**feedback)

    def add_aids(actor, aids, **kwargs):
        raise NotImplementedError()

    def remove_aids(actor, aids, **kwargs):
        raise NotImplementedError()

    def logs(actor):
        return None

    def status(actor):
        actor_status = {}
        try:
            actor_status['phase'] = actor.phase
        except Exception:
            pass
        try:
            actor_status['loop_phase'] = actor.loop_phase
        except Exception:
            pass
        try:
            actor_status['is_inconsistent'] = False
        except Exception:
            pass
        try:
            actor_status['is_converged'] = actor.phase == 4
        except Exception:
            pass
        try:
            actor_status['num_meaningful'] = 0
        except Exception:
            pass
        try:
            actor_status['num_pccs'] = (
                None if actor.edge_gen is None else len(actor.edge_gen.edge_requests)
            )
        except Exception:
            pass
        try:
            actor_status['num_inconsistent_ccs'] = 0
        except Exception:
            pass
        try:
            actor_status['cc_status'] = {
                'num_names_max': len(actor.db.clustering),
                'num_inconsistent': 0,
            }
        except Exception:
            pass
        try:
            actor_status['changes'] = actor.changes
        except Exception:
            pass
        return actor_status

    def metadata(actor):
        if actor.infr.verifiers is None:
            actor.infr.verifiers = {}
        verifier = actor.infr.verifiers.get('match_state', None)
        extr = None if verifier is None else verifier.extr
        metadata = {
            'extr': extr,
        }
        return metadata


class LCAClient(GraphClient):
    actor_cls = LCAActor

    def sync(client, ibs):
        ret_dict = {
            'changes': client.actor_status.get('changes', []),
        }
        return ret_dict


if __name__ == '__main__':
    r"""
    CommandLine:
        python -m wbia_lca._plugin
    """
    import xdoctest

    xdoctest.doctest_module(__file__)
