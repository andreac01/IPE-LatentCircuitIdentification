# Backward Search Approximated

## Introduction
Starting from the backward search implemented in the baseline IPE methodology 
described in https://openreview.net/forum?id=ccaYnWj0mO&noteId=ccaYnWj0mO we 
found that a view of the model as a computational graph allows to estimate 
the effect of single paths in a large language model. Furthermore the results 
available on the [MIB benchmark leaderboard](https://huggingface.co/spaces/mib-bench/leaderboard), 
shows that this approach is a viable alternative to state-of-the-art cricuit 
discovery methods, such as Edge Attribution Patching (EAP) and Edge ACTivation 
Patching (EActP).
The main limitation of IPE lies in a limited scalability, as it requires to 
explore thousands of paths in the model. Particularly to evaluate a candidate 
path it is required to propagate the message through all its nodes, which is 
computationally expensive and difficult to cache.
Edge Attribution Patching (EAP) on the other hand is very efficient as it only 
requires two forward passes and a single backward pass to evaluate all edges in 
the circuit. The limitation are twofold:
1. It only evaluates the contributions of edges, which is often not enough to 
   understand the role of a circuit in the model.
2. The evaluation is based on a linear approximation of the model with respect 
   the edges, which is not always accurate, especially the first layer of the 
   model.

## Ideas
From qualitative studies of EAP and IPE behaviour it is possible to observe that:
1. For any specific task, in the model at least two types of relevant circuits 
   exist, one is composed of edges responsible for the correct prediction, one 
   is composed of edges responsible for lowering the confidence in the prediction.
   The relevance of the first type of circuit is trivial, while the second circuit
   is important to properly model the next token probability distribution, 
   lowering the confidente in order to avoid high losses in case of misspredictions.
2. A complete path can either contribute positively or negatively to the 
   prediction, and most of the time this contribution is negligible.
   As an example the presented circuit in the MIB benchmark leaderboard is composed
   of ~10% of the total edges in the model, and <<1e-16% of the total paths. The
   behaviour of the model was significantly reconstructed despite the low number
   of retained paths.
3. A single edge can be part of multiple paths, whose contribution can be either 
   positive or negative. This means that the contribution of a single edge may 
   overall be negligible, while being a relevant part of both circuits.
4. The approximated contribution of an edge in EAP is based on calculating the 
   derivative of the model with respect to the edge, which is equivalent to 
   calculating the derivative of all paths starting from the edge. The furthermore 
   the linear approximation is better when less non linearities are present in the
   computational graph.

From these observation we can speculate that approximating the contribution of 
adding an edge to a path can be done by calculating the derivative of the contribution
of the path with respect to its input and see how the change in the input coused by 
removing the edge affects the output of the model.
Particularly this should:
1. Provide a fast way to evaluate candidate continuations of a path. Given that 
   the number of candidate continuations is often in the order of some thousands
   an impleentation of this type allowed for a speedup in the order of 100x.
2. Provide a more accurate derivative of the contribution with respect to methods
   like EAP. This should be because the function we are approximating is simpler
   because it is limited to a single path.

## Example

Imagine that we have the path 'h8.6 -> h9.9 -> out' and we have a metric m(r) that
measures the effect of setting the output residual stream to r.