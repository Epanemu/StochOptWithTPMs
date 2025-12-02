import numpy as np
import spn.algorithms.Inference as spflow_inference
from spn.algorithms.Inference import log_likelihood
from spn.algorithms.LearningWrappers import learn_cnet, learn_parametric

# from spn.structure.leaves.parametric.Bernoulli import create_bernoulli_leaf
from spn.structure.Base import Context, Product, Sum
from spn.structure.leaves.cltree.CLTree import CLTree
from spn.structure.leaves.parametric.Parametric import Bernoulli, Parametric


class CNetNode:
    """
    An easy-to-read CNet class that is self-contained.
    It stores the CNet parameters as data and performs inference
    without relying on the spflow library.
    """

    def __init__(
        self,
        node_type,
        children=None,
        weights=None,
        scope=None,
        dist_type=None,
        cltree_tree=None,
        cltree_log_factors=None,
        bernoulli_p0=None,
    ):

        self.node_type = node_type  # "SUM", "PRODUCT", or "LEAF"
        self.children = children if children else []
        self.weights = np.array(weights) if weights else np.array([])

        # --- Leaf-specific attributes ---
        self.dist_type = dist_type  # "CLTREE" or "BERNOULLI"
        self.scope = scope  # List of variable indices this leaf covers

        # For CLTree leaves
        self.cltree_tree = cltree_tree  # List of parent indices
        self.cltree_log_factors = cltree_log_factors  # 3D list of log-probs

        # For Bernoulli leaves
        self.bernoulli_p0 = bernoulli_p0  # P(X=0)

    def __repr__(self, level=0):
        """Recursively prints the CNet structure."""
        indent = "  " * level

        if self.node_type == "LEAF":
            if self.dist_type == "CLTREE":
                if len(self.scope) == 1:
                    p0 = np.exp(self.cltree_log_factors[0][0][0])
                    return f"{indent}LEAF(dist=CLTree, scope={self.scope}, P(0)={p0:.3f}, P(1)={1-p0:.3f})"
                else:
                    return f"{indent}LEAF(dist=CLTree, scope={self.scope})"
            elif self.dist_type == "BERNOULLI":
                p1 = 1.0 - self.bernoulli_p0
                return f"{indent}LEAF(dist=Bernoulli, scope={self.scope[0]}, P(0)={self.bernoulli_p0:.3f}, P(1)={p1:.3f})"
            else:
                return f"{indent}LEAF(dist=Unknown, scope={self.scope})"

        s = f"{indent}{self.node_type} ({self.scope})"
        if self.node_type == "SUM":
            for i, child in enumerate(self.children):
                s += f"\n{indent}  [weight={self.weights[i]:.3f}]"
                s += f"\n{child.__repr__(level + 2)}"
        elif self.node_type == "PRODUCT":
            for child in self.children:
                s += f"\n{child.__repr__(level + 1)}"
        return s

    def inference(self, x):
        """
        Performs inference on a single data point (numpy array)
        using only the stored parameters.

        *** This method has no dependency on spflow ***
        """
        if self.node_type == "LEAF":

            if self.dist_type == "BERNOULLI":
                # Get the value for the single variable this leaf governs
                val = x[self.scope[0]]
                return self.bernoulli_p0 if val == 0 else (1.0 - self.bernoulli_p0)

            elif self.dist_type == "CLTREE":
                # This leaf represents a Chow-Liu Tree. We must
                # re-implement its inference logic.
                # P(x) = Prod_i P(x_i | x_parent(i))
                # log P(x) = Sum_i log P(x_i | x_parent(i))

                # Get the slice of data relevant to this leaf
                x_slice = x[self.scope]

                total_log_prob = 0.0

                # Iterate over each variable *within the leaf's scope*
                for i in range(len(self.scope)):
                    # Get the value of this variable (0 or 1)
                    v_i = int(x_slice[i])

                    # Find its parent *within the CLTree*
                    parent_local_idx = self.cltree_tree[i]

                    if parent_local_idx == -1:
                        # This is the root node of the CLTree.
                        # Its factor is log P(X_i).
                        # The parent_val dimension is redundant, so we can just use 0.
                        v_p = 0
                    else:
                        # This is a child node.
                        # Its factor is log P(X_i | X_parent).
                        # Get the parent's value.
                        v_p = int(x_slice[parent_local_idx])

                    # Add the pre-computed log factor
                    total_log_prob += self.cltree_log_factors[i][v_i][v_p]

                return np.exp(total_log_prob)

            else:
                raise TypeError(f"Unknown leaf distribution type: {self.dist_type}")

        elif self.node_type == "PRODUCT":
            prob = 1.0
            for child in self.children:
                prob *= child.inference(x)
            return prob

        elif self.node_type == "SUM":
            prob = 0.0
            # Use numpy for a slightly faster weighted sum
            for i, child in enumerate(self.children):
                prob += self.weights[i] * child.inference(x)
            return prob


def build_cnet_wrapper(spflow_node):
    """
    Recursively translates an SPFlow SPN structure into our CNetNode class,
    extracting all parameters so the new object is self-contained.

    Args:
        spflow_node: The root node of the learned SPN from spflow.

    Returns:
        The root node of our CNetNode structure.
    """

    if isinstance(spflow_node, Product):
        children = [build_cnet_wrapper(c) for c in spflow_node.children]
        return CNetNode(node_type="PRODUCT", children=children, scope=spflow_node.scope)

    elif isinstance(spflow_node, Sum):
        children = [build_cnet_wrapper(c) for c in spflow_node.children]
        return CNetNode(
            node_type="SUM",
            children=children,
            weights=spflow_node.weights,
            scope=spflow_node.scope,
        )

    elif isinstance(spflow_node, CLTree):
        # Extract all necessary data from the CLTree object
        return CNetNode(
            node_type="LEAF",
            dist_type="CLTREE",
            scope=spflow_node.scope,
            cltree_tree=spflow_node.tree,
            cltree_log_factors=spflow_node.log_factors,
        )

    elif isinstance(spflow_node, Parametric):
        # Extract all necessary data from the Bernoulli leaf
        # (Assuming Bernoulli as per the learning setup)
        return CNetNode(
            node_type="LEAF",
            dist_type="BERNOULLI",
            scope=spflow_node.scope,
            bernoulli_p0=spflow_node.p[0],
        )

    else:
        raise TypeError(f"Unknown SPFlow node type: {type(spflow_node)}")


# --- 1. Generate Synthetic Data ---
# Create 100 samples of 4 binary variables
# We'll make var 0 and 1 correlated, and var 2 and 3 correlated.
np.random.seed(42)
data = np.zeros((200, 4), dtype=int)
# Group 1: X0 and X1 are often the same
same_01 = np.random.randint(0, 2, 100)
data[:100, 0] = same_01
data[:100, 1] = same_01
# Group 2: X2 and X3 are often different
val_2 = np.random.randint(0, 2, 100)
data[100:, 2] = val_2
data[100:, 3] = 1 - val_2
# Add some noise
data[100:, 0] = np.random.randint(0, 2, 100)
data[100:, 1] = np.random.randint(0, 2, 100)
data[:100, 2] = np.random.randint(0, 2, 100)
data[:100, 3] = np.random.randint(0, 2, 100)

data = data.astype(np.float32)

# --- 2. Learn the CNet using SPFlow ---
# Define the types of variables (all Bernoulli for this binary data)
# var_types = [create_bernoulli_leaf for _ in range(4)]
var_types = Context(
    parametric_types=[
        Bernoulli,
        Bernoulli,
        Bernoulli,
        Bernoulli,
    ]
).add_domains(data)
# Learn the CNet structure
# spflow_cnet_structure = learn_cnet(data, var_types)
spflow_cnet_structure = learn_cnet(
    data, var_types, cond="random", min_instances_slice=100, min_features_slice=1
)

# Learn the parameters
# learn_parametric(spflow_cnet_structure, data)

# --- 3. Convert to our easy-to-read CNet class ---
my_cnet = build_cnet_wrapper(spflow_cnet_structure)

# --- 4. Print the readable CNet structure ---
print("--- Readable CNet Structure ---")
print(my_cnet)
print("-" * 30)

# --- 5. Perform Inference ---
# Create a test data point
test_point = np.array([0, 0, 1, 0], dtype=np.float32)

# 5a. Inference using our CNetNode class
prob_my_cnet = my_cnet.inference(test_point)
print(f"Test Point: {test_point}")
print(f"Probability from CNetNode class: {prob_my_cnet:.6f}")

# 5b. Inference using SPFlow's built-in algorithm for verification
# spflow expects a 2D array (batch of data)
test_point_batch = test_point.reshape(1, -1).astype(int)
# prob_spflow = spflow_inference.prob(spflow_cnet_structure, test_point_batch)[0]
prob_spflow = log_likelihood(spflow_cnet_structure, test_point_batch)[0, 0]
print(f"Probability from SPFlow        : {np.exp(prob_spflow):.6f}")

# --- Verification ---
assert np.isclose(prob_my_cnet, np.exp(prob_spflow)), "Inference results do not match!"
print("\nSuccess: Inference results match.")
