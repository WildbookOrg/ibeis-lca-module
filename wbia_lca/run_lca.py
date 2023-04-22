import os
import utool as ut
from wbia_lca._plugin import OVERALL_CONFIG_FILE


"""
This file shows and explains functionality to run LCA via the plugin that
interfaces to it. The code depends on the assumption that the embed function
has been run and we are sitting in an ipython shell, with an active ibs
controller.
"""


""" ===================================================
1. Clean up reviews so that each pair of annotations has AT MOST one review.
During cleanup give priority to reviews marked as positive or negative
or incomparable. Second priority is given to unreviewed reviews.

I recommend that we delete all edge weights. This will cause the verifier
to regenerate edges before LCA is started, but it will also ensure that
we are in a consistent state. For similar reasons, I also recommend that
we delete all names.
"""


def make_review_rowids_unique(ibs, rowids_to_fix):
    """
    Given a list of review rowids, gather the unique edges (aid pairs),
    and then reduce down so that each edge has one review
    """
    edges_to_fix = ibs.get_review_aid_tuple(rowids_to_fix)
    edges_to_fix = sorted(list(set(edges_to_fix)))
    make_reviews_unique(ibs, edges_to_fix)


def make_reviews_unique(ibs, edges_to_fix):
    """
    Given a list of edges, find the reviews (a list of 0 or more for
    each edge), and eliminate all but one. The actual delete calls
    are commented out for the moment.
    """
    edges_rids = ibs.get_review_rowids_from_edges(edges_to_fix)
    for rids in edges_rids:
        decisions = ibs.get_review_decision(rids)
        print('====================')
        print('Rowids:', rids)
        if len(rids) == 0:
            print('No reviews of any kind')
            continue
        print(decisions)
        if len(decisions) == 1:
            print('Unique decision, so nothing to do')
            continue

        is_unreviewed = [d is None for d in decisions]
        unrev_rids = ut.compress(rids, is_unreviewed)
        is_reviewed = [d is not None for d in decisions]
        rev_rids = ut.compress(rids, is_reviewed)
        print(f'Num unreviewd: {len(unrev_rids)}, num reviewed: {len(rev_rids)}')

        if len(rev_rids) == 0:  # note that we must have len(unrev_rids) > 1
            print(f'Deleting {len(unrev_rids)-1} unreviewed. None are reviewed.')
            # ibs.delete_review(unrev_rids[1:])
        else:
            if len(unrev_rids) == 0:
                print('No unreviewed to delete')
            else:
                print(
                    f'Deleting all {len(unrev_rids)} unreviewed because there is at least one reviewed'
                )
                # ibs.delete_review(unrev_rids)
            if len(rev_rids) > 1:
                print(f'Deleting {len(rev_rids)-1} reviews, all but the last one')
                # ibs.delete_review(rev_rids[:-1])
            else:
                print('Keeping the only review')


def filter_for_uniqueness(ibs):
    aids = ibs.get_valid_aids()
    rids = ibs.get_review_rowids_between(aids)
    make_review_rowids_unique(ibs, rids)

    # Uncomment to delete all edge weights and names
    # ew_rids = ibs._get_all_edge_weight_rowids()
    # ibs.delete_edge_weight(ew_rids)
    # nids = ibs.get_annot_nids(aids)
    # ibs.delete_names(nids)


""" ===========================================================

2. Start with the defaults for the overall config dictionary, including
   the function to write it as a pkl file to a location provided by the
   plugin. This assumes that we are currently sitting within the same
   docker container as the web interface, which also builds the objects
   in the plugin.

"""

base_dir = os.path.dirname(OVERALL_CONFIG_FILE)
species = 'zebra_grevys'
verifier_gt_name = 'lca_verifier_calib_' + species + '.pkl'
clustering_result_name = 'lca_clustering_results' + species + '.pkl'

OVERALL_CONFIG = {
    'species': species,
    'ranker': 'hotspotter',
    'verifier': 'vamp',
    'verifier_gt_filepath': os.path.join(base_dir, verifier_gt_name),
    'clustering_result_filepath': os.path.join(base_dir, clustering_result_name),
    'run_on_all_names': False,
    'use_short_circuiting': False,
    'use_simulated_human_reviews': False,
}


def write_overall_config():
    ut.save_cPkl(OVERALL_CONFIG_FILE, OVERALL_CONFIG)


""" ===================================================

3. Run ranking. The results are stored as ibs DB reviews
   with decision UNREVIEWED.

   NOTE ON SATURDAY 4/22:  THIS DOES NOT YET WORK WITH
   run_qaids_again_themselves = False
   THIS SHOULDN'T BE TOO HARD TO FIND AND FIX.

"""


def run_ranking(ibs, daids, qaids=None, run_qaids_against_themselves=True):
    from wbia_lca._plugin import LCAActor

    write_overall_config()

    """
    Sometimes, as in the first time annotations are processed
    for ID, we want to run queries against themselves and against
    the database. Other times, as when merging two sets of aids
    that have previously been matched, we do not.
    """
    if run_qaids_against_themselves and qaids is not None:
        daids = list(set(daids) | set(qaids))

    actor = LCAActor()
    actor.run_ranker_lnbnn(daids, ibs.dbdir, qaids)


""" ======================================================

4. Run LCA AFTER candidate matches have been established
   through ranking.

    4a. The aids should be the combination of the query aids
        and the database daids from above. It is possible to
        run ranking multiple times with different annotation
        sets before running LCA. This will work as long as all
        qaids and daids into the aids list

    4b. By default we only run LCA in regions of the ID graph
        that could be affected by the new candidate matches and
        new human reviews. This is the normal mode of operation.
        When LCA is interrupted before completion, however, it
        is best to run_on_all_names=True and then LCA will check
        everything.

    4c. Set use_short_circuiting=True if you would like the
        short_circuiting function to use non-visual information
        to decide if a match is impossible.

    4d. If you would like to simulate human reviews, set
        use_simulated_reviews=True. More on this below.
"""


def run_lca(
    ibs,
    aids,
    run_on_all_names=False,
    use_short_circuiting=False,
    use_simulated_human_reviews=False,
):
    import requests
    import json
    from wbia_lca._plugin import LCAActor

    global OVERALL_CONFIG
    OVERALL_CONFIG['run_on_all_names'] = run_on_all_names
    OVERALL_CONFIG['use_short_circuiting'] = use_short_circuiting
    OVERALL_CONFIG['use_simulated_human_reviews'] = use_simulated_human_reviews
    write_overall_config()

    actor = LCAActor()
    _ = actor.start(
        dbdir=ibs.dbdir, aids=aids, graph_uuid='11111111-1111-1111-1111-111111111111'
    )

    data = {'aids': json.dumps(aids)}
    response = requests.post(
        'http://localhost:5000/review/identification/lca/refer/', data=data
    )
    return response


""" =============================================================

5. Simulation requires some work:

    a. Run ranking and LCA to get to a place in the experiments on
       your annotations just before you'd like to start the simulation.
       Specifically, suppose the annotations you'd like to run queries
       during a simulation are in a list called qaids.  Ten you should
       pause computation just after running ranking on the qaids against
       the daids you'd like to target.

    b. Manually create backup copies of the database files that are
       in /data/db/_ibsdb: _ibeis_staging.sqlite3 and _ibeis_database.sqlite3
       Do this in a temporary directory outside the /data/db/_ibsdb
       directory. You may need to exit and restart the ipython shell,
       but I don't think so.

    c. Back in the ipython shell, run_lca to completion in the normal
       way, with use_simulated_human_reviews set to False. Make sure
       that the file named in
              OVERALL_CONFIG['clustering_result_filepath']
       is succcessfully created by the _plugin. It will contain the
       clustering results that you will compare againts.

    d. Gather whatever statistics you need on the results for comparison
       and then exit the ipython shell.

    e. Restore the manually-copied databases and start the ipython shell,
       and change whatever you wish to change in the configuration (such
       as using short-circuitingg).

    f. Using the exact same set of aids (union of the qaids and daids
       from setp a), call run_lca, but this time with
          OVERALL_CONFIG['use_simulated_human_reviews'] set to True

    g. Gather statistics for comparison

    h. Repeat steps e through g as needed.

"""
