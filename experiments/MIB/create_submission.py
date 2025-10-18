import os
import pickle as pkl
from datetime import datetime
import json
from ipe.nodes import Node, EMBED_Node, ATTN_Node, MLP_Node, FINAL_Node
from transformer_lens import HookedTransformer
import argparse


def path_to_edges(score: float, path: list[Node], edges: dict) -> dict:
    """Convert a path of nodes into a set of edges with associated scores.

    Args:
        score (float): score associated with the path
        path (list[Node]): list of nodes in the path
        edges (set[Node]): set of edges to update

    Raises:
        ValueError: if an unknown node type is encountered

    Returns:
        set[Node]: updated set of edges with scores
    """
    n0s = ['input']
    for i in range(1, len(path)):
        if isinstance(path[i], EMBED_Node):
            n1s = ['input']
        elif isinstance(path[i], ATTN_Node):
            n1s = []
            if path[i].patch_key:
                n1s.append(f'a{path[i].layer}.h{path[i].head}<k>')
            if path[i].patch_value:
                n1s.append(f'a{path[i].layer}.h{path[i].head}<v>')
            if path[i].patch_query:
                n1s.append(f'a{path[i].layer}.h{path[i].head}<q>')
        elif isinstance(path[i], MLP_Node):
            n1s = [f'm{path[i].layer}']
        elif isinstance(path[i], FINAL_Node):
            n1s = ['logits']
        else:
            raise ValueError(f"Unknown node type: {type(path[i])}")

        n0 = n0s[0].split('<')[0]
        for n1 in n1s:        
            if f'{n0}->{n1}' not in edges:
                edges[f'{n0}->{n1}'] = score
            else:
                cur_score = edges[f'{n0}->{n1}']
                edges[f'{n0}->{n1}'] = cur_score + score
        n0s = n1s
    return edges


def path_to_nodes(path: list[Node]) -> set[str]:
    """Convert a path of nodes into a set of strings for serialization.

    Args:
        path (list[Node]): list of nodes in the path
    Returns:
        set[str]: set of strings representing the nodes
    """
    nodes = set()
    for node in path:
        if isinstance(node, EMBED_Node):
            nodes.add('input')
        elif isinstance(node, ATTN_Node):
            nodes.add(f'a{node.layer}.h{node.head}')
        elif isinstance(node, MLP_Node):
            nodes.add(f'm{node.layer}')
        elif isinstance(node, FINAL_Node):
            nodes.add('logits')
        else:
            raise ValueError(f"Unknown node type: {type(node)}")
    return nodes
    

def create_submission(model: str = "gpt2-small",
                      path_source_dir: str = "./detected_paths",
                      edges_output_dir: str = "./submissions"):
    """Process all detected paths and convert them to submission format.
    Args:
        path_source_dir (str): Directory containing detected paths
        edges_output_dir (str): Directory to save submission files
    """

    if not os.path.exists(edges_output_dir):
        os.makedirs(edges_output_dir)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    submission_dir = os.path.join(edges_output_dir, f"submission_{timestamp}")

    files = os.listdir(path_source_dir)
    target = [f for f in files if f.endswith('.pkl') and model in f]
    assert len(target) == 1, f"Unexpected number of files found for model '{model}' in '{path_source_dir}' (found {len(target)})"
    target = target[0]
    print(f"Processing file: {target}")
    
    paths = pkl.load(open(os.path.join(path_source_dir, target), 'rb'))
    edges = {}
    nodes = set()
    for score, path in paths:
        edges = path_to_edges(score, path, edges)
        nodes = nodes.union(path_to_nodes(path))
    print(f"Extracted {len(nodes)} nodes and {len(edges)} edges from {len(paths)} paths")
    count = 0
    for e in edges:
        if count < 10:
            print(f"Edge: {e}")
        count += 1
    print(f"Number of edges with positive score: {count} out of {len(edges)}")
    if model == "gpt2":
        model = "gpt2-small"
    elif model == "qwen":
        model = "Qwen/Qwen2.5-0.5B"
    model = HookedTransformer.from_pretrained(model)
    
    submission_dict = {}
    submission_dict['cfg'] = {
        "n_layers": model.cfg.n_layers,
        "n_heads": model.cfg.n_heads,
        "d_model": model.cfg.d_model,
        "parallel_attn_mlp": model.cfg.parallel_attn_mlp
    }

    submission_dict['nodes'] = {str(n): {"in_graph": True} for n in nodes}
    submission_dict['edges'] = {str(e[0]): {"score": e[1], "in_graph": True} for e in edges.items()}

    if not os.path.exists(submission_dir):
        os.makedirs(submission_dir)
    output_filename = os.path.join(submission_dir, f"{model.cfg.model_name.lower().replace("/","-")}.json")
    with open(output_filename, 'w') as f:
        json.dump(submission_dict, f, indent=4)
    print(f"Submission saved to {output_filename}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Create submission JSON from detected paths.")
    parser.add_argument("--model", type=str, default="gpt2-small", help="Model name to filter .pkl files and to load")
    parser.add_argument("--source_dir", type=str, default="./detected_paths", help="Directory containing detected paths (.pkl files)")
    parser.add_argument("--output_dir", type=str, default="./submissions", help="Directory to save submission JSON")
    args = parser.parse_args()

    create_submission(model=args.model,
                        path_source_dir=args.source_dir,
                        edges_output_dir=args.output_dir)