"""
Tree Structured Partition Node class, represents a node (space partition) in the tree. 

"""

import numpy as np
import math
import numba
import matplotlib.pyplot as plt
from numpy.random import default_rng as rng

@numba.njit
def filter_in_bounds(data, idx_array, lower_bounds, upper_bounds, Xdim_indicator, is_x_partition):
    """
    Filter the samples in 'idx_array' by checking if they lie within the 
    [lower_bounds, upper_bounds] for the relevant dimensions (X or Y).
    
    Parameters
    ----------
    data : np.ndarray
        The entire dataset of shape (N, dim).
    idx_array : np.ndarray
        1D array of sample indices to filter.
    lower_bounds : np.ndarray
        1D array of length 'dim' giving the lower bound for each dimension.
    upper_bounds : np.ndarray
        1D array of length 'dim' giving the upper bound for each dimension.
    Xdim_indicator : np.ndarray (bool)
        1D boolean array of length 'dim'. True for X dimensions, False for Y.
    is_x_partition : bool
        True if we should check only dimensions indicated by Xdim_indicator == True.
        False if we should check only those indicated by Xdim_indicator == False.

    Returns
    -------
    np.ndarray
        The subset of indices that passed all bound checks on relevant dimensions.
    """
    out_idx = np.empty(len(idx_array), dtype=idx_array.dtype)
    count = 0

    for i in range(len(idx_array)):
        sample_idx = idx_array[i]
        # We'll check each dimension that belongs to X (if is_x_partition=True)
        # or belongs to Y (if is_x_partition=False).
        pass_all = True
        for d in range(len(Xdim_indicator)):
            dim_is_x = Xdim_indicator[d]
            # Decide if dimension 'd' is relevant
            if (is_x_partition and dim_is_x) or ((not is_x_partition) and (not dim_is_x)):
                val = data[sample_idx, d]
                # If out of bounds, short-circuit for this sample
                if val < lower_bounds[d] or val > upper_bounds[d]:
                    pass_all = False
                    break
        
        if pass_all:
            out_idx[count] = sample_idx
            count += 1

    return out_idx[:count]


class TSPNode:
    """
    TSPNode represents a node (space partition) in the tree structure used for 
    mutual information estimation. Each node maintains properties such as:
      - Bounds of the partition.
      - Marginal and joint distribution measures.
      - Indices of samples belonging to X or Y partitions.
      - References to left and right child nodes.
    """

    def __init__(self, parent=None):
        """
        Initialize a TSPNode with references to its parent and default 
        properties for distribution measures and partition indices.

        Parameters
        ----------
        parent : TSPNode or None
            The parent node in the tree. If None, this node may become the root.
        """
        self.left = None
        self.right = None
        self.parent = parent
        self.condJointDist = 0.0
        self.condMargProd = 0.0
        self.relativeCMIgain = float("-inf")
        self.absoluteCMIgain = float("-inf")
        self.n_samples = 0
        self.n_marginal_samples_X = 0
        self.n_marginal_samples_Y = 0
        self.idx_marginal_samples_X = []
        self.idx_marginal_samples_Y = []
        self.lowerBounds = np.array([])
        self.upperBounds = np.array([])
        self.partitions = []

    def grow(self, parent, nodeIdx, data, lowerBounds, upperBounds, 
             Xdim_indicator, dim, kn, projDim):
        """
        Grows the partitioning tree by recursively splitting the dataset. 
        Fixes the 'TypeError' by ensuring 'nodeIdx' is always a NumPy array.

        Parameters
        ----------
        parent : TSPNode or None
            The parent node in the tree. If None, this node may be the root.
        nodeIdx : array-like of int
            Indices of samples to be partitioned at the current node.
        data : np.ndarray
            The full dataset of shape (N, dim).
        lowerBounds : np.ndarray
            1D array of length 'dim', the lower bounds of the partition.
        upperBounds : np.ndarray
            1D array of length 'dim', the upper bounds of the partition.
        Xdim_indicator : list of bool
            A boolean list indicating which dimensions belong to X (True) or Y (False).
        dim : int
            Total number of dimensions in the data.
        kn : float or int
            Critical mass threshold for deciding whether to split further.
        projDim : int
            The dimension index used for the current split operation.

        Returns
        -------
        TSPNode
            The newly created node (this node) after potential recursive splitting.
        """

        # Ensure nodeIdx is a NumPy array (important for advanced indexing)
        if not isinstance(nodeIdx, np.ndarray):
            nodeIdx = np.asarray(nodeIdx, dtype=int)

        # Create a new TSPNode with the given parent
        node = TSPNode(parent)

        # If we're at the root level (projDim == 0), set the parent to itself
        if projDim == 0:
            node.parent = node
            node.idx_marginal_samples_X = nodeIdx
            node.idx_marginal_samples_Y = nodeIdx
            node.n_marginal_samples_X = len(nodeIdx)
            node.n_marginal_samples_Y = len(nodeIdx)

        # Basic node properties
        node.n_samples = len(nodeIdx)
        node.lowerBounds = lowerBounds
        node.upperBounds = upperBounds

        # Compute conditional joint distribution (node samples / parent samples)
        node.condJointDist = node.n_samples / node.parent.n_samples

        # Compute conditional marginal product
        node.condMargProd = node.conditionalMarginalProd(
            data, lowerBounds, upperBounds, Xdim_indicator, dim, projDim
        )

        # If enough samples remain to split further
        if (node.n_samples // 2) >= kn:

            # Project data onto current dimension
            proj_axis = projDim % dim
            projectedData = data[nodeIdx, proj_axis]

            # Find the median index (partial sort)
            medianIdx = node.n_samples // 2
            idx_partitioned = np.argpartition(projectedData, medianIdx)

            # Split indices into two groups around the median
            leftNodeIdx = nodeIdx[idx_partitioned[:medianIdx]]
            rightNodeIdx = nodeIdx[idx_partitioned[medianIdx:]]

            # Compute partition boundary
            left_max = np.max(projectedData[idx_partitioned[:medianIdx]])
            right_min = np.min(projectedData[idx_partitioned[medianIdx:]])
            mean = (left_max + right_min) / 2

            # Update bounds for left and right children
            leftUpper = upperBounds.copy()
            leftUpper[proj_axis] = mean
            rightLower = lowerBounds.copy()
            rightLower[proj_axis] = mean

            # Save the partition info
            node.partitions = [(proj_axis, rightLower, leftUpper)]

            # Recursively grow left and right subtrees
            node.left = self.grow(
                node, leftNodeIdx, data,
                lowerBounds, leftUpper,
                Xdim_indicator, dim, kn, projDim + 1
            )

            node.right = self.grow(
                node, rightNodeIdx, data,
                rightLower, upperBounds,
                Xdim_indicator, dim, kn, projDim + 1
            )

            # Compute mutual information gains for this node
            node.relativeCMIgain = node.getCMIgain()
            node.absoluteCMIgain = node.relativeCMIgain * (node.n_samples / data.shape[0])

        return node

    def conditionalMarginalProd(self, data, lower_bounds, upper_bounds, Xdim_indicator, dim, proj_dim):
        """
        Numba-optimized version: uses the JIT-compiled filter_in_bounds(...) 
        to quickly filter samples for X or Y partitions.

        Parameters
        ----------
        data : np.ndarray
            The full dataset of shape (N, dim).
        lower_bounds : np.ndarray
            The lower bounds for each dimension of the current partition.
        upper_bounds : np.ndarray
            The upper bounds for each dimension of the current partition.
        Xdim_indicator : array-like of bool
            Which dimensions belong to X (True) or Y (False).
        dim : int
            The total dimensionality of the data.
        proj_dim : int
            The current projection (dimension index) used for partitioning.

        Returns
        -------
        float
            The ratio of filtered samples over the parent's marginal samples, 
            serving as the conditional marginal product for this node.
        """

        if proj_dim == 0:
            return 1.0

        # Ensure Xdim_indicator is a NumPy boolean array
        Xdim_indicator = np.asarray(Xdim_indicator, dtype=np.bool_)

        # Check if the partition is on X or Y
        is_x_partition = Xdim_indicator[(proj_dim - 1) % dim]

        if is_x_partition:
            # Use parent's marginal samples for Y
            self.idx_marginal_samples_Y = self.parent.idx_marginal_samples_Y
            self.n_marginal_samples_Y = self.parent.n_marginal_samples_Y

            # Filter parent's X
            parent_idx_X = np.asarray(self.parent.idx_marginal_samples_X, dtype=np.int32).ravel()
            filtered_idx = filter_in_bounds(
                data,
                parent_idx_X,
                lower_bounds,
                upper_bounds,
                Xdim_indicator,
                True  # we are checking X dimensions
            )

            self.idx_marginal_samples_X = filtered_idx
            self.n_marginal_samples_X = len(filtered_idx)

            # Return proportion of samples
            return self.n_marginal_samples_X / self.parent.n_marginal_samples_X

        else:
            # Use parent's marginal samples for X
            self.idx_marginal_samples_X = self.parent.idx_marginal_samples_X
            self.n_marginal_samples_X = self.parent.n_marginal_samples_X

            # Filter parent's Y
            parent_idx_Y = np.asarray(self.parent.idx_marginal_samples_Y, dtype=np.int32).ravel()
            filtered_idx = filter_in_bounds(
                data,
                parent_idx_Y,
                lower_bounds,
                upper_bounds,
                Xdim_indicator,
                False  # we are checking Y dimensions
            )

            self.idx_marginal_samples_Y = filtered_idx
            self.n_marginal_samples_Y = len(filtered_idx)

            return self.n_marginal_samples_Y / self.parent.n_marginal_samples_Y

    def getCMIgain(self):
        """
        Compute the local conditional mutual information (CMI) gain at this node
        by evaluating its left and right children.

        Returns
        -------
        float
            The sum of CMI contributions from left and right child nodes.
        """
        left_CMIgain = self.left.condJointDist * math.log2(self.left.condJointDist / self.left.condMargProd)
        right_CMIgain = self.right.condJointDist * math.log2(self.right.condJointDist / self.right.condMargProd)
        return left_CMIgain + right_CMIgain

    def getPartitions(self):
        """
        Recursively gather all partition boundaries from this node downward.

        Returns
        -------
        list
            A list of partition tuples (dimension, lower_bounds, upper_bounds).
        """
        if self.left is None and self.right is None:
            return []
        else:
            return self.partitions + self.left.getPartitions() + self.right.getPartitions()

    def getEMI(self):
        """
        Recursively compute the empirical mutual information from this node
        by adding the current CMI gain and weighting children's EMI values.

        Returns
        -------
        float
            The accumulated EMI from this node's level downward.
        """
        if self.left is None and self.right is None:
            return 0
        else:
            return (
                self.relativeCMIgain
                + (self.left.condJointDist * self.left.getEMI())
                + (self.right.condJointDist * self.right.getEMI())
            )

    def getSize(self):
        """
        Recursively count the number of leaf nodes in the subtree rooted at this node.

        Returns
        -------
        int
            The count of leaves under this node, including itself if it is a leaf.
        """
        if self.left is None and self.right is None:
            return 1
        else:
            return self.left.getSize() + self.right.getSize()
        

class TSP:
    def __init__(self, l_bn, w_bn, _lambda):
        """
        Initialize the TSP instance with the necessary hyperparameters 
        for tree construction and regularization.

        Parameters
        ----------
        l_bn : float
            Exponent controlling the critical mass calculation, must lie in (0, 1/3).
        w_bn : float
            Coefficient for the critical mass calculation.
        _lambda : float
            Regularization parameter.
        
        Raises
        ------
        ValueError
            If l_bn is not in (0, 1/3).
        """
        if l_bn >= 1/3.0 or l_bn <= 0:
            raise ValueError("Parameter `l_bn` must belong to the interval (0, 1/3).")
        
        self.l_bn = l_bn
        self.w_bn = w_bn
        self.kn = 0
        self._lambda = _lambda
        self.root = None
        self.dim = 0
        self.n_samples = 0
        self.tsp_size = 0
        self.tsp_reg_size = 0
        self.tsp_emi = 0.0
        self.tsp_reg_emi = 0.0
        self.tsp_partitions = []

    def grow(self, x, y):
        """
        Builds (grows) the TSP tree from data X and Y. The data arrays are combined,
        and the initial node indexes are generated to recursively partition the space.

        Parameters
        ----------
        x : array-like
            Data samples for the first variable (X). Must be 2D: shape (n_samples, x_dim).
        y : array-like
            Data samples for the second variable (Y). Must be 2D: shape (n_samples, y_dim).
        
        Notes
        -----
        This method updates:
            - self.root with a fully grown TSPNode tree.
            - self.tsp_emi, self.tsp_size, and self.tsp_partitions based on the grown tree.
        """
        # Convert inputs to NumPy arrays if they aren't already
        x = np.asarray(x)
        y = np.asarray(y)

        # Concatenate X and Y into a single data array
        data = np.concatenate((x, y), axis=1)

        # Number of samples and dimensions
        self.n_samples = data.shape[0]
        self.dim = data.shape[1]
        
        # Create a root node
        self.root = TSPNode()

        # Determine critical mass kn = ceil(w_bn * n^(1 - l_bn))
        self.kn = np.ceil(self.w_bn * (self.n_samples ** (1 - self.l_bn)))

        # Define the global bounds for all dimensions
        lowerBounds = np.min(data, axis=0) - 0.001
        upperBounds = np.max(data, axis=0) + 0.001

        # Create a Boolean list indicating which dims belong to X (True) and which to Y (False)
        Xdim_indicator = [True] * x.shape[1] + [False] * y.shape[1]

        # Indices of every sample in the whole space
        nodeIdx = np.arange(self.n_samples, dtype=int)

        # Grow the tree from the root
        self.root = self.root.grow(
            None,                   
            nodeIdx,               
            data,
            lowerBounds,
            upperBounds,
            Xdim_indicator,
            self.dim,
            self.kn,
            projDim=0
        )

        # After full growth, collect results
        self.tsp_emi = self.root.getEMI()
        self.tsp_size = self.root.getSize()
        self.tsp_partitions = self.root.getPartitions()

    def regularize(self):
        """
        Regularizes (prunes) the grown TSP tree to balance the trade-off 
        between the empirical mutual information and model complexity.

        Raises
        ------
        ValueError
            If no tree (self.root) exists (i.e., grow() has not been called).

        Notes
        -----
        - Uses self.minimum_cost_trees to obtain the mutual information (EMI) 
          for every minimal-cost tree.
        - Employs a cost function with a regularization term determined by self._lambda.
        - Updates self.tsp_reg_emi and self.tsp_reg_size with the best result.
        """
        # A grown tree is required
        if self.root is None:
            raise ValueError("Observations not provided.")

        # The size (number of leaves) of the full-grown tree
        full_tree_size = self.tsp_size

        # Regularizer terms
        bn = self.w_bn * np.power(self.n_samples, -self.l_bn)
        inv_deltan = np.exp(self.n_samples ** (1/3.0))

        # Build array of EMI values for each minimal-cost tree size
        treesEMI = np.zeros(full_tree_size)
        self.minimum_cost_trees(treesEMI, full_tree_size)

        # We will look for the optimal cost and size
        optimal_cost = -treesEMI[0]
        optimal_size = 1

        # Precompute repeated logs and constants
        constant_term = 8.0 / self.n_samples
        dim_log = (self.dim + 1) * np.log(2) + self.dim * np.log(self.n_samples)
        log_inv = np.log(8 * inv_deltan)

        # Evaluate cost for every minimum-cost tree size from 2..full_tree_size
        for k in range(2, full_tree_size + 1):
            # Epsilon depends on k
            cost_arg = log_inv + k * dim_log
            epsilon = (12.0 / bn) * np.sqrt(constant_term * cost_arg)

            cost = -treesEMI[k - 1] + self._lambda * epsilon

            # Update optimal cost/size if improved
            if cost < optimal_cost:
                optimal_cost = cost
                optimal_size = k

        # Once pruned, store the results
        self.tsp_reg_emi = -optimal_cost
        self.tsp_reg_size = optimal_size

    def minimum_cost_trees(self, treesEMI, full_tree_size):
        """
        This method computes EMI values for all possible "minimum cost" trees 
        by iteratively splitting the leaf with the maximum absoluteCMIgain 
        and storing partial EMI sums in `treesEMI`.

        Parameters
        ----------
        treesEMI : np.ndarray
            A 1D array (length full_tree_size) where each index k will store
            the EMI value for the minimum-cost tree of size k+1.
        full_tree_size : int
            Total number of leaves (size of the fully grown tree).

        Notes
        -----
        - The first element of treesEMI (for tree size=1) remains 0.
        - The method repeatedly expands the leaf with the highest absoluteCMIgain
          and updates the EMI accordingly.
        - Calls subadditive_insert(...) to maintain leaves in ascending 
          order of absoluteCMIgain.
        """
        # Array of leaves. Each index in `leaves` will eventually point to a leaf in the tree.
        leaves = [None] * full_tree_size
        leaves[0] = self.root

        # By definition, the tree of size=1 has EMI = 0
        # So treesEMI[0] is already 0 by default

        for k in range(full_tree_size - 1):
            # Leaf to expand = the leaf with highest absoluteCMIgain among existing leaves
            leaf_maxCMI = leaves[k]

            # EMI for tree of size k+1 = EMI of tree of size k + absoluteCMIgain of chosen leaf
            treesEMI[k + 1] = treesEMI[k] + leaf_maxCMI.absoluteCMIgain

            # Insert left & right children into `leaves`, maintaining ascending absoluteCMIgain
            self.subadditive_insert(leaf_maxCMI.left, leaves, k)
            self.subadditive_insert(leaf_maxCMI.right, leaves, k + 1)

    def subadditive_insert(self, new_leaf, leaves, j):
        """
        Replaces bubble-sort with a single-pass insertion from right to left.
        This yields the same final ordering as the old bubble-sort approach 
        (ascending absoluteCMIgain) but more efficiently.

        Parameters
        ----------
        new_leaf : TSPNode
            The newly expanded child leaf to insert into the list.
        leaves : list of TSPNode
            The collection of leaves being maintained in ascending order of absoluteCMIgain.
        j : int
            The initial index at which `new_leaf` is placed before insertion sort correction.
        """
        leaves[j] = new_leaf
        new_gain = new_leaf.absoluteCMIgain

        # We'll move left from j until we find the correct insertion point
        i = j
        while i > 0 and leaves[i - 1].absoluteCMIgain > new_gain:
            # Swap adjacent leaves
            leaves[i], leaves[i - 1] = leaves[i - 1], leaves[i]
            i -= 1

    def visualize(self, x, y):
        """
        Visualizes the 2D partitioning (TSP tree boundaries) over the given data.

        Parameters
        ----------
        x : array-like
            Samples for X (2D array).
        y : array-like
            Samples for Y (2D array).

        Raises
        ------
        ValueError
            If the tree has not been grown (self.root is None) 
            or the data dimension is not 2.
        """
        if self.root is None:
            raise ValueError("Observations not provided.")
        if self.dim != 2:
            raise ValueError("The tree can only be visualized for a two dimensional problem.")

        x = np.asarray(x)
        y = np.asarray(y)

        # Partitions
        partitions = self.partitions()

        # Separate partitions by type
        horizontal_partition = [p[1:] for p in partitions if p[0] == 0]
        vertical_partition   = [p[1:] for p in partitions if p[0] == 1]

        fig, axes = plt.subplots(nrows=1, ncols=3, figsize=(16, 5))

        # Left subplot: samples only
        axes[0].scatter(x, y, marker='o', edgecolor='k', s=25, alpha=0.5)
        axes[0].set_xlabel('x')
        axes[0].set_ylabel('y')
        axes[0].set_title('Samples')
        axes[0].grid(alpha=0.3)

        # Middle subplot: TSP lines over samples
        axes[1].scatter(x, y, marker='o', edgecolor='k', s=25, alpha=0.5)

        # Plot horizontal lines
        for bound in vertical_partition:
            axes[1].hlines(
                y=bound[0][1],
                xmin=bound[0][0],
                xmax=bound[1][0],
                color='black'
            )

        # Plot vertical lines
        for bound in horizontal_partition:
            axes[1].vlines(
                x=bound[0][0],
                ymin=bound[0][1],
                ymax=bound[1][1],
                color='black'
            )

        axes[1].set_xlabel('x')
        axes[1].set_ylabel('y')
        axes[1].set_title(
            f"TSP: samples={self.n_samples}, cell_samples={int(self.kn)}, grow_size={self.size()}"
        )
        axes[1].grid(alpha=0.3)

        # Right subplot: TSP lines, samples with alpha=0.1
        axes[2].scatter(x, y, marker='o', edgecolor='k', s=25, alpha=0.1)

        for bound in vertical_partition:
            axes[2].hlines(
                y=bound[0][1],
                xmin=bound[0][0],
                xmax=bound[1][0],
                color='black'
            )
        for bound in horizontal_partition:
            axes[2].vlines(
                x=bound[0][0],
                ymin=bound[0][1],
                ymax=bound[1][1],
                color='black'
            )

        axes[2].set_xlabel('x')
        axes[2].set_ylabel('y')
        axes[2].set_title("TSP Partitions")
        axes[2].grid(alpha=0.3)

        plt.tight_layout()
        plt.show()

    def emi(self):
        """
        Returns the empirical mutual information (EMI) for the fully grown TSP tree.

        Returns
        -------
        float
            The EMI value stored in `self.tsp_emi`.

        Raises
        ------
        ValueError
            If the tree has not been grown (self.root is None).
        """
        if self.root is None:
            raise ValueError("Observations not provided.")
        return self.tsp_emi

    def reg_emi(self):
        """
        Returns the regularized empirical mutual information for the pruned TSP tree.

        Returns
        -------
        float
            The regularized EMI value stored in `self.tsp_reg_emi`.

        Raises
        ------
        ValueError
            If the tree has not been grown (self.root is None).
        """
        if self.root is None:
            raise ValueError("Observations not provided.")
        return self.tsp_reg_emi

    def size(self):
        """
        Returns the size (number of leaves) of the fully grown TSP tree.

        Returns
        -------
        int
            The tree size stored in `self.tsp_size`.

        Raises
        ------
        ValueError
            If the tree has not been grown (self.root is None).
        """
        if self.root is None:
            raise ValueError("Observations not provided.")
        return self.tsp_size

    def reg_size(self):
        """
        Returns the size (number of leaves) of the pruned TSP tree 
        after the regularization process.

        Returns
        -------
        int
            The pruned tree size stored in `self.tsp_reg_size`.

        Raises
        ------
        ValueError
            If the tree has not been grown (self.root is None).
        """
        if self.root is None:
            raise ValueError("Observations not provided.")
        return self.tsp_reg_size

    def partitions(self):
        """
        Returns a list of partitions in the form 
        [(dim, lower_bounds, upper_bounds), ...] for each split.

        Returns
        -------
        list
            The list of partitions stored in `self.tsp_partitions`.

        Raises
        ------
        ValueError
            If the tree has not been grown (self.root is None).
        """
        if self.root is None:
            raise ValueError("Observations not provided.")
        return self.tsp_partitions


def tsp(x: np.ndarray, y: np.ndarray, l_bn: float = 0.167, w_bn: float = 5e-2, __lambda = 2.3e-5, seed_1:int = 1923, seed_2:int = 1453):
    """ TSP mutual information estimation
    requires
    x, y: np.ndarray
    (l_bn, w_bn, __lambda): regularization and tree parameters
    seed_1 and seed_2: noise random states to estimate MI """
    
    # noise perturbation
    x = x + rng(seed_1).uniform(-1e-8, 1e-8, size=x.shape)
    y = y + rng(seed_2).uniform(-1e-8, 1e-8, size=y.shape)
    
    # TSP initialization
    tsp = TSP(l_bn, w_bn, __lambda)
    # TSP MI estimation and regularization process
    tsp.grow(x, y)
    tsp.regularize()

    # regularized MI
    reg_emi = tsp.reg_emi()
    return reg_emi

def mi_assessment(x: np.ndarray, y_true: np.ndarray, y_hat: np.ndarray, 
                  l_bn: float = 0.167, w_bn: float = 5e-2, __lambda = 2.3e-5,
                  seed_1:int = 1923, seed_2:int = 1453):
    """ TSP mutual information assessment
    requires
    x, y: np.ndarray
    true_mi: ground truth mutual information value
    (l_bn, w_bn, __lambda): regularization and tree parameters
    seed_1 and seed_2: noise random states to estimate MI """
    
    # estimated MI
    residual = y_true - y_hat
    est_mi = tsp(x, residual, l_bn, w_bn, __lambda, seed_1, seed_2)

    return est_mi