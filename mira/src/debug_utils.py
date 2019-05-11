import shutil, re
from pathlib import Path
from semantic_modeling.utilities.serializable import deserialize, deserializeJSON, serialize, serializeJSON
from data_structure import Graph
from multiprocessing.pool import ThreadPool

def draw_graph(same_dir: bool):
    finput = Path("/tmp/sm_debugging/draw_graphs.json")
    input = deserializeJSON(finput)
    new_id = -1
    for item in finput.parent.iterdir():
        match = re.match("draw_graph_(\d+)$", item.name)
        if match is not None:
            if int(match.groups()[0]) > new_id:
                new_id = int(match.groups()[0])

    if not same_dir:
        new_id += 1
    
    output = finput.parent / f"draw_graph_{new_id}"
    output.mkdir(exist_ok=True)

    n_graphs = len(list(output.iterdir()))
    graphs = [Graph.from_dict(o) for o in input["graphs"]]
    with ThreadPool() as p:
        p.map(lambda ig: ig[1].render2img(output / f"graph_{ig[0]}.png"), enumerate(graphs, start=n_graphs))


if __name__ == "__main__": 
    import sys
    if sys.argv[1] == "draw_graph":
        draw_graph(sys.argv[2].strip().lower() == "true")
    else:
        print("Invalid command: %s", sys.argv[1])
        exit(1)

