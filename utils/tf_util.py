import tensorflow as tf


def load_graph(pb_file_path):
    try:
        with tf.io.gfile.GFile(pb_file_path, "rb") as f:
            graph_def = tf.compat.v1.GraphDef()
            graph_def.ParseFromString(f.read())
    except Exception as e:
        raise ValueError(f"Error reading the protobuf file: {e}")

    with tf.Graph().as_default() as graph:
        try:
            tf.import_graph_def(graph_def, name="")
        except Exception as e:
            raise ValueError(f"Error importing the graph: {e}")
    return graph


def get_graph_io(graph):
    input_nodes = []
    output_nodes = []

    for op in graph.get_operations():
        if op.type == "Placeholder":
            input_nodes.append(op)
        if any([output.dtype for output in op.outputs]):
            output_nodes.append(op)

    return input_nodes, output_nodes


# Path to your .pb file
pb_file_path = "/Users/nikornlansa/Workspace/ML/ClearScanner/sync/DocumentCornerLocalization/train/version7/saved_model/saved_model/saved_model.pb"


# Load the graph
try:
    graph = load_graph(pb_file_path)
except ValueError as e:
    print(e)
    exit(1)

# Get inputs and outputs
input_nodes, output_nodes = get_graph_io(graph)

print("Inputs:")
for input_node in input_nodes:
    print(input_node.name)

print("\nOutputs:")
for output_node in output_nodes:
    print(output_node.name)

