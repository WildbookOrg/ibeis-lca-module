# Example Zero

An example of running the graph algorithm with no verification
algorithm at all, so the only augmentation is human.  This is
appropriate for initialization, before enough data is gathered for
training verification algorithms.

## request_example.json

Looking at request_example.json the most important consideration is
the verifier edge quads. These each have the special name "zero" as
their augmentation method. (They also have the verifier probability of
0.0, but that's just a place holder.)  These will have 0 weight and
will only provide direction on what nodes to try to form into
clusters --- nodes in different connected components of the "zero"
edges will not be tested for connectivity.

In request_example.json, the database is empty.  This is not
necessary, but it may be typical.  The edge generator contains no
verifier results and a few manually-specific human decisions.  The
ground truth clusters are provided to allow the edge generator to
simulate the remaining human decisions.  There is a delay of 5
iterations of the LCA graph algorithm, but this only demonstrates that
the algorithm can handle delays before augmentation results (in this
case only from humans) are returned.

##  verifier_probs.json

This contains an empty dictionary because there is no verification
algorithm to train.

##  default_config.ini

The only change here is that only "human" is provided as the
verification.

##  Command-line

python ../overall_driver.py --ga_config default_config.ini --verifier_gt verifier_probs.json --request request_example.json
