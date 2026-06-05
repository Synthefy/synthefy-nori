"""Hierarchical SCM synthetic data generator for LimiX training.

Generates synthetic tabular datasets using a hierarchical composition of
Local Causal Structures (LCS). Each LCS is a small DAG (3-8 nodes) with
random edge functions (MLP, decision tree, conv1d). LCS are composed
hierarchically: child LCS nodes can depend on parent LCS nodes.

synth_v4 (2026-02-19): TabICLv2-inspired diversity improvements:
  - Expanded MLP activations (8 -> 30+)
  - ExtraTrees filtering (reject unlearnable datasets)
  - Power-law feature importances
  - Remove Gaussian edge noise
  - Richer aggregation (max, logsumexp)
  - Quadratic + product edge functions
  - Random feature rescaling
  - Kumaraswamy warping
"""

import numpy as np


# ---------------------------------------------------------------------------
# Edge function factories
# ---------------------------------------------------------------------------

def _get_activations(expanded=False):
    """Return list of activation functions for MLP edge functions.

    Args:
        expanded: if True, return 30+ activations (synth_v4).
                  if False, return original 8 activations.
    """
    base = [
        lambda x: x,                          # identity
        np.tanh,                               # tanh
        lambda x: 1 / (1 + np.exp(-np.clip(x, -20, 20))),  # sigmoid
        lambda x: np.log1p(np.abs(x)),         # log(1+|x|)
        np.abs,                                # abs
        np.sin,                                # sin
        lambda x: x ** 2,                      # x^2
        lambda x: x * (1 / (1 + np.exp(-1.702 * np.clip(x, -20, 20)))),  # approx gelu
    ]
    if not expanded:
        return base
    extra = [
        # Standard activations
        lambda x: np.where(x > 0, x, 0.01 * x),  # LeakyReLU
        lambda x: np.where(x > 0, x, 1.0 * (np.exp(np.clip(x, -20, 20)) - 1)),  # ELU
        lambda x: 1.0507 * np.where(x > 0, x, 1.6733 * (np.exp(np.clip(x, -20, 20)) - 1)),  # SELU
        lambda x: x / (1 + np.exp(-np.clip(x, -20, 20))),  # SiLU / Swish
        lambda x: np.maximum(0, np.minimum(x, 6)),  # ReLU6
        lambda x: np.clip(x, -1, 1),          # HardTanh
        lambda x: np.sign(x),                  # signum
        lambda x: (x > 0).astype(float),       # Heaviside
        lambda x: np.exp(-x ** 2),             # Gaussian
        lambda x: np.log1p(np.exp(np.clip(x, -20, 20))),  # softplus
        lambda x: np.maximum(x, 0),            # ReLU (explicit)
        # Non-standard / diverse
        lambda x: np.cos(x),                   # cosine
        lambda x: np.sign(x) * np.sqrt(np.abs(x)),  # sqrt with sign
        lambda x: np.clip(x, 0, 1),            # clamp 0-1
        lambda x: np.round(np.clip(x, -10, 10)),  # round
        lambda x: np.mod(x, 1.0),              # modulo 1
        lambda x: x ** 3,                      # cube
        lambda x: np.sign(x) * (np.abs(x) ** 0.5),  # power 0.5
        lambda x: np.sign(x) * (np.abs(x) ** 1.5),  # power 1.5
        lambda x: np.where(x > 0, x, -0.5 * x),  # Leaky ReLU (0.5)
        lambda x: x * np.tanh(np.log1p(np.exp(np.clip(x, -20, 20)))),  # Mish
        lambda x: np.log(np.maximum(np.abs(x), 1e-6)),  # log(|x|)
    ]
    return base + extra


def _make_mlp_fn(in_dim, rng, expanded_activations=False):
    """Create a random MLP edge function (numpy-based)."""
    n_layers = rng.integers(1, 4)  # 1-3 layers
    width = int(rng.integers(4, 65))
    activations = _get_activations(expanded=expanded_activations)
    act_idx = rng.integers(0, len(activations))
    act = activations[act_idx]

    # Build weight matrices
    weights = []
    biases = []
    dims = [in_dim]
    for i in range(n_layers):
        out = width if i < n_layers - 1 else 1
        # Xavier init
        limit = np.sqrt(6.0 / (dims[-1] + out))
        W = rng.uniform(-limit, limit, size=(dims[-1], out))
        b = rng.uniform(-0.1, 0.1, size=(out,))
        weights.append(W)
        biases.append(b)
        dims.append(out)

    def mlp_fn(x):
        # x: [n_samples, in_dim]
        h = x
        for i, (W, b) in enumerate(zip(weights, biases)):
            h = h @ W + b
            if i < n_layers - 1:
                h = act(h)
        out = h.ravel()
        # Smooth asymptote: avoids jagged flat regions at ±50 in deep chains
        return 50.0 * np.tanh(out / 50.0)

    return mlp_fn


def _make_random_tree(rng, depth, max_depth):
    """Recursively build a random decision tree with sampled thresholds and leaf values."""
    if depth >= max_depth or rng.random() < 0.3:
        # Leaf node
        value = rng.uniform(-2, 2)
        return ('leaf', value)
    threshold = rng.uniform(-4, 4)
    left = _make_random_tree(rng, depth + 1, max_depth)
    right = _make_random_tree(rng, depth + 1, max_depth)
    return ('split', threshold, left, right)


def _eval_random_tree(tree, x):
    """Evaluate a random tree on scalar input x (vectorized)."""
    kind = tree[0]
    if kind == 'leaf':
        return np.full(x.shape, tree[1])
    # split node
    _, threshold, left, right = tree
    mask = x <= threshold
    out = np.empty_like(x)
    if mask.any():
        out[mask] = _eval_random_tree(left, x[mask])
    if (~mask).any():
        out[~mask] = _eval_random_tree(right, x[~mask])
    return out


def _make_tree_fn(rng):
    """Create a random decision tree edge function with sampled thresholds and leaf values."""
    max_depth = int(rng.integers(2, 7))  # depth 2-6
    tree = _make_random_tree(rng, 0, max_depth)

    def tree_fn(x):
        # x: [n_samples, in_dim]
        v = x.ravel()
        return _eval_random_tree(tree, v)

    return tree_fn


def _make_saturating_fn(rng):
    """Create a saturating nonlinear edge function (sigmoid, softplus, or hinge).

    These capture real-world diminishing-returns / threshold patterns and are
    ICL-compatible (each sample computed independently, no sample-order dependence).
    Replaces conv1d which convolved along the sample dimension — unusable by ICL
    since sample ordering is arbitrary.
    """
    choice = int(rng.integers(0, 3))
    a = rng.uniform(0.5, 3.0)
    b = rng.uniform(-2, 2)

    if choice == 0:
        # Sigmoid: dose-response, probability saturation
        def fn(x):
            v = np.clip(x.ravel(), -10, 10)
            return a * (1 / (1 + np.exp(-(v - b))))
    elif choice == 1:
        # Softplus: smooth ReLU, always non-negative
        def fn(x):
            v = np.clip(x.ravel(), -10, 10)
            return a * np.log1p(np.exp(v - b))
    else:
        # Hinge / clipped linear: threshold activation
        def fn(x):
            v = x.ravel()
            return a * np.maximum(v - b, 0)

    return fn


def _make_piecewise_fn(rng):
    """Create a piecewise-linear edge function with random breakpoints and slopes."""
    n_segments = int(rng.integers(2, 6))  # 2-5 segments
    breakpoints = np.sort(rng.uniform(-3, 3, n_segments - 1))
    slopes = rng.uniform(-2, 2, n_segments)
    intercepts = rng.uniform(-1, 1, n_segments)

    def piecewise_fn(x):
        # x: [n_samples, in_dim]
        v = x.ravel()
        # Use np.searchsorted for vectorized segment lookup instead of
        # Python loop over segments with masking
        seg_idx = np.searchsorted(breakpoints, v)  # 0..n_segments-1
        out = slopes[seg_idx] * v + intercepts[seg_idx]
        return 50.0 * np.tanh(out / 50.0)

    return piecewise_fn


def _make_poly_fn(rng):
    """Create a polynomial edge function (degree 2-4)."""
    degree = int(rng.integers(2, 5))  # 2-4
    coeffs = rng.standard_normal(degree + 1) * 0.5  # scale down to prevent overflow

    def poly_fn(x):
        # x: [n_samples, in_dim]
        v = np.clip(x.ravel(), -5, 5)
        # Horner's method: avoids repeated v**i, ~2x faster for degree 4
        out = coeffs[-1]
        for c in coeffs[-2::-1]:
            out = out * v + c
        return out

    return poly_fn


def _make_periodic_fn(rng):
    """Create a periodic (sinusoidal) edge function."""
    freq = rng.uniform(0.5, 5.0)
    phase = rng.uniform(0, 2 * np.pi)
    amplitude = rng.uniform(0.5, 2.0)

    def periodic_fn(x):
        # x: [n_samples, in_dim]
        v = x.ravel()
        return amplitude * np.sin(freq * v + phase)

    return periodic_fn


def _make_rbf_fn(rng):
    """Create an RBF (Gaussian kernel) edge function: amp * exp(-(x-c)^2 / (2*s^2))."""
    center = rng.uniform(-2, 2)
    sigma = rng.uniform(0.3, 2.0)
    amplitude = rng.uniform(0.5, 3.0)

    def rbf_fn(x):
        v = x.ravel()
        return amplitude * np.exp(-((v - center) ** 2) / (2 * sigma ** 2))

    return rbf_fn


def _make_logexp_fn(rng):
    """Create a log or exp edge function for heavy-tailed/exponential patterns."""
    use_log = rng.random() < 0.5
    a = rng.uniform(0.5, 2.0)

    if use_log:
        def logexp_fn(x):
            v = x.ravel()
            return a * np.log1p(np.abs(v)) * np.sign(v)
    else:
        b = rng.uniform(0.3, 1.5)
        def logexp_fn(x):
            v = np.clip(x.ravel(), -5, 5)
            return a * (np.exp(b * v) - 1)

    return logexp_fn


def _make_quadratic_fn(rng, in_dim=1):
    """Create a quadratic edge function: f(x) = x^T M x + w^T x + b.

    Inspired by TabICLv2's RandomQuadraticFunction. Captures second-order
    feature interactions that linear/MLP may miss.

    Args:
        in_dim: input dimensionality. When in_dim > 1, M is a full symmetric
                matrix enabling true multivariate quadratic interactions.
    """
    # Random symmetric matrix for quadratic term
    A = rng.standard_normal((in_dim, in_dim)) * 0.3
    M = (A + A.T) / 2  # symmetric
    w = rng.standard_normal(in_dim) * 0.5
    b = rng.uniform(-0.5, 0.5)

    if in_dim == 1:
        def quadratic_fn(x):
            v = np.clip(x.ravel(), -5, 5)
            out = M[0, 0] * v ** 2 + w[0] * v + b
            return 50.0 * np.tanh(out / 50.0)
    else:
        def quadratic_fn(x):
            # x: [n_samples, in_dim]
            x_clipped = np.clip(x, -5, 5)
            # x^T M x for each sample: sum of element-wise (x @ M) * x
            quad = np.sum((x_clipped @ M) * x_clipped, axis=1)
            linear = x_clipped @ w
            return 50.0 * np.tanh((quad + linear + b) / 50.0)

    return quadratic_fn


def _make_product_fn(rng, in_dim):
    """Create a product edge function: f(x) * g(x).

    Inspired by TabICLv2's RandomProductFunction. Creates multiplicative
    interactions between two different random functions.
    """
    # Use simpler functions for the two factors (avoid recursion/slowness)
    simple_makers = [_make_piecewise_fn, _make_poly_fn, _make_periodic_fn,
                     _make_rbf_fn, _make_logexp_fn, _make_quadratic_fn]
    fn1 = simple_makers[int(rng.integers(0, len(simple_makers)))](rng)
    fn2 = simple_makers[int(rng.integers(0, len(simple_makers)))](rng)

    def product_fn(x):
        return 50.0 * np.tanh(fn1(x) * fn2(x) / 50.0)

    return product_fn


def _make_multidim_scalar_fn(in_dim, rng, expanded=False):
    """Wrap a scalar edge function to handle multi-dim input.

    Projects [n_samples, in_dim] -> [n_samples, 1] via random weights,
    then applies a scalar edge function. Returns [n_samples].
    """
    proj = rng.standard_normal(in_dim)
    proj /= (np.linalg.norm(proj) + 1e-8)
    scalar_fn = _make_edge_fn(1, rng, expanded=expanded)

    def multidim_fn(x):
        # x: [n_samples, in_dim]
        projected = x @ proj  # [n_samples]
        return scalar_fn(projected.reshape(-1, 1))

    return multidim_fn


def _make_concat_edge_fn(in_dim, rng, expanded=False):
    """Create a multivariate edge function for concat-then-transform mode.

    Only MLP and quadratic support in_dim > 1. Randomly selects one.
    Returns a function: [n_samples, in_dim] -> [n_samples].
    """
    if rng.random() < 0.5:
        return _make_mlp_fn(in_dim, rng, expanded_activations=expanded)
    else:
        return _make_quadratic_fn(rng, in_dim=in_dim)


def _make_edge_fn(in_dim, rng, expanded=False):
    """Randomly select and create an edge function.

    Args:
        expanded: if True, include quadratic and product functions (synth_v4)
    """
    n_choices = 10 if expanded else 8
    choice = rng.integers(0, n_choices)
    if choice == 0:
        return _make_mlp_fn(in_dim, rng, expanded_activations=expanded)
    elif choice == 1:
        return _make_tree_fn(rng)
    elif choice == 2:
        return _make_saturating_fn(rng)
    elif choice == 3:
        return _make_piecewise_fn(rng)
    elif choice == 4:
        return _make_poly_fn(rng)
    elif choice == 5:
        return _make_periodic_fn(rng)
    elif choice == 6:
        return _make_rbf_fn(rng)
    elif choice == 7:
        return _make_logexp_fn(rng)
    elif choice == 8:
        return _make_quadratic_fn(rng)
    else:
        return _make_product_fn(rng, in_dim)


# ---------------------------------------------------------------------------
# Root node distribution sampling
# ---------------------------------------------------------------------------

def _sample_root(n_samples, rng):
    """Sample root node values from a random distribution."""
    choice = rng.integers(0, 7)
    if choice == 0:
        return rng.standard_normal(n_samples)
    elif choice == 1:
        return rng.uniform(-3, 3, n_samples)
    elif choice == 2:
        a = rng.uniform(0.5, 5.0)
        b = rng.uniform(0.5, 5.0)
        return rng.beta(a, b, n_samples)
    elif choice == 3:
        return rng.lognormal(0, 1, n_samples)
    elif choice == 4:
        df = rng.uniform(2, 10)
        return rng.standard_t(df, n_samples)
    elif choice == 5:
        # Pareto distribution (heavy-tailed)
        alpha = rng.uniform(1.5, 5.0)
        scale = rng.uniform(0.5, 2.0)
        return scale * (1 + rng.pareto(alpha, n_samples))
    else:
        # Mixture of Gaussians (2-3 components)
        n_components = int(rng.integers(2, 4))
        weights = rng.dirichlet(np.ones(n_components))
        means = rng.uniform(-3, 3, n_components)
        stds = rng.uniform(0.3, 2.0, n_components)
        assignments = rng.choice(n_components, size=n_samples, p=weights)
        samples = np.empty(n_samples)
        for k in range(n_components):
            mask = assignments == k
            samples[mask] = rng.normal(means[k], stds[k], mask.sum())
        return samples


# ---------------------------------------------------------------------------
# Aggregation functions
# ---------------------------------------------------------------------------

def _aggregate(parent_values, rng, expanded=False):
    """Aggregate values from multiple parents.

    parent_values: list of arrays, each [n_samples]
    expanded: if True, include max and logsumexp aggregation (synth_v4)
    Returns: [n_samples]
    """
    if len(parent_values) == 1:
        return parent_values[0]

    stacked = np.column_stack(parent_values)
    n_choices = 6 if expanded else 4
    choice = rng.integers(0, n_choices)
    if choice == 0:
        # Simple average
        return stacked.mean(axis=1)
    elif choice == 1:
        # Weighted average
        weights = rng.dirichlet(np.ones(len(parent_values)))
        return stacked @ weights
    elif choice == 2:
        # 2-layer MLP aggregation
        in_dim = len(parent_values)
        limit1 = np.sqrt(6.0 / (in_dim + in_dim * 2))
        W1 = rng.uniform(-limit1, limit1, (in_dim, in_dim * 2))
        b1 = rng.uniform(-0.1, 0.1, in_dim * 2)
        limit2 = np.sqrt(6.0 / (in_dim * 2 + 1))
        W2 = rng.uniform(-limit2, limit2, (in_dim * 2, 1))
        b2 = rng.uniform(-0.1, 0.1, 1)
        h = np.tanh(stacked @ W1 + b1)
        return (h @ W2 + b2).ravel()
    elif choice == 3:
        # Multiplicative aggregation: element-wise product (clipped)
        result = np.clip(stacked[:, 0], -10, 10)
        for i in range(1, stacked.shape[1]):
            result = np.clip(result * np.clip(stacked[:, i], -10, 10), -10, 10)
        return result
    elif choice == 4:
        # Max aggregation (synth_v4)
        return stacked.max(axis=1)
    else:
        # Logsumexp aggregation (synth_v4): smooth max
        clipped = np.clip(stacked, -20, 20)
        return np.log(np.sum(np.exp(clipped), axis=1) + 1e-8)


# ---------------------------------------------------------------------------
# Local Causal Structure (LCS)
# ---------------------------------------------------------------------------

class LocalCausalStructure:
    """A small causal graph with 3-8 nodes, topologically ordered."""

    def __init__(self, n_nodes, external_parent_indices, rng, expanded=False,
                 synth_v5=False, external_d_nodes=None, synth_v5_declone=True,
                 max_parents=4):
        """
        n_nodes: number of nodes in this LCS
        external_parent_indices: list of global indices of nodes from
            previously created LCS that can serve as parents
        expanded: if True, use synth_v4 expanded edge functions
        synth_v5: if True, enable concat-then-transform and multi-dim nodes
        external_d_nodes: dict mapping global_idx -> dimensionality for
            external parent nodes (synth_v5 only)
        synth_v5_declone: if True AND synth_v5, use per-dim scaling in
            multi-dim expansion (fixes max_abs_corr near-duplicates)
        rng: numpy random generator
        """
        self.n_nodes = n_nodes
        self.rng = rng
        self.expanded = expanded
        self.synth_v5 = synth_v5
        self.synth_v5_declone = synth_v5_declone
        self._max_parents = max_parents
        if external_d_nodes is None:
            external_d_nodes = {}
        # Build internal DAG (topological order = node index order)
        # Each node has edges from some subset of prior nodes (internal + external)
        self.parents = {}  # node_idx -> list of (source_type, source_idx)
        self.edge_fns = {}  # node_idx -> list of edge functions
        self.agg_fns = {}  # node_idx -> aggregation function
        self.concat_mode = {}  # node_idx -> bool (synth_v5 concat-then-transform)

        # Multi-dimensional node sizes (synth_v5 Phase A)
        # d_nodes[i] = dimensionality of node i's output
        self.d_nodes = {}

        for i in range(n_nodes):
            # Determine node dimensionality
            if synth_v5:
                # Pareto-distributed: most stay 1-2, rare up to 4
                # Reduced from clip(0,7) to clip(0,3) to prevent excessive
                # Bayes error that was making tasks too hard (oob_auc 0.72
                # vs real 0.81).
                d = 1 + int(np.clip(rng.pareto(1.5), 0, 3))
            else:
                d = 1
            self.d_nodes[i] = d

            # Possible parents: internal nodes with idx < i, plus external nodes
            internal_candidates = list(range(i))
            all_candidates = [(True, idx) for idx in internal_candidates] + \
                             [(False, idx) for idx in external_parent_indices]

            if len(all_candidates) == 0:
                # Root node
                self.parents[i] = []
                self.concat_mode[i] = False
                continue

            # Randomly select 1-3 parents (can be overridden by max_parents kwarg)
            max_p = getattr(self, '_max_parents', 4)
            n_parents = min(rng.integers(1, max_p), len(all_candidates))
            chosen = rng.choice(len(all_candidates), size=n_parents, replace=False)
            parents = [all_candidates[c] for c in chosen]
            self.parents[i] = parents

            # Helper to get parent dimensionality
            def _parent_dim(is_internal, idx):
                if is_internal:
                    return self.d_nodes[idx]
                return external_d_nodes.get(idx, 1)

            # Concat-then-transform mode: synth_v5, 2+ parents, 40% probability
            use_concat = (synth_v5 and len(parents) >= 2
                          and rng.random() < 0.4)
            self.concat_mode[i] = use_concat

            if use_concat:
                # Single multivariate function for all parents concatenated
                total_in_dim = sum(
                    _parent_dim(is_internal, idx)
                    for is_internal, idx in parents
                )
                self.edge_fns[i] = [_make_concat_edge_fn(
                    total_in_dim, rng, expanded=expanded)]
            else:
                # Per-parent edge functions (original behavior)
                fns = []
                for is_internal, idx in parents:
                    parent_d = _parent_dim(is_internal, idx)
                    if synth_v5 and parent_d > 1:
                        # Multi-dim parent: project d->1 then scalar fn
                        fns.append(_make_multidim_scalar_fn(parent_d, rng,
                                                            expanded=expanded))
                    else:
                        fns.append(_make_edge_fn(1, rng, expanded=expanded))
                self.edge_fns[i] = fns

    @staticmethod
    def _robust_standardize(v):
        """Standardize node output using robust statistics (median/MAD).

        Prevents cascading magnitude explosion through deep SCM chains
        while preserving distribution shape. Applied after each non-root
        node computation.

        Uses np.sort + indexing instead of np.median for ~4x speedup
        (np.median calls partition twice internally; sort + index is faster
        for the small arrays typical in SCM nodes).
        """
        if v.ndim == 1:
            s = np.sort(v)
            n = len(s)
            med = (s[n // 2] + s[(n - 1) // 2]) * 0.5
            abs_dev = np.abs(s - med)
            abs_dev.sort()
            mad = (abs_dev[n // 2] + abs_dev[(n - 1) // 2]) * 0.5
            scale = mad * 1.4826  # MAD to std conversion
            if scale < 1e-8:
                scale = max(float(np.std(v)), 1e-8)
            return (v - med) / scale
        else:
            # Per-column standardization for multi-dim nodes
            s = np.sort(v, axis=0)
            n = v.shape[0]
            med = (s[n // 2] + s[(n - 1) // 2]) * 0.5
            med = med[np.newaxis, :]
            abs_dev = np.sort(np.abs(v - med), axis=0)
            mad = (abs_dev[n // 2] + abs_dev[(n - 1) // 2]) * 0.5
            mad = mad[np.newaxis, :]
            scale = mad * 1.4826
            # Fallback per-column to std if MAD is 0
            fallback = np.maximum(np.std(v, axis=0, keepdims=True), 1e-8)
            scale = np.where(scale < 1e-8, fallback, scale)
            return (v - med) / scale

    def _expand_to_multidim(self, result, d, n_samples):
        """Expand scalar result to multi-dim node, optionally with per-dim
        scaling and noise for de-cloning."""
        if self.synth_v5_declone:
            base = result.reshape(-1, 1)
            # Clip scales to [0.3, 1.7] to prevent sign flips and
            # extreme amplification that caused cascading overflow
            scales = np.clip(
                self.rng.normal(loc=1.0, scale=0.35, size=d), 0.3, 1.7)
            base = base * scales[None, :]
            sig = max(float(np.std(result)), 1e-6)
            noise_rel = sig * self.rng.uniform(0.20, 0.80)
            noise = self.rng.standard_normal((n_samples, d)) * noise_rel
            out = base + noise
            if self.rng.random() < 0.25:
                Q, _ = np.linalg.qr(self.rng.standard_normal((d, d)))
                out = out @ Q
            return out
        else:
            noise = self.rng.standard_normal((n_samples, d)) * 0.1
            return result.reshape(-1, 1) + noise

    def generate(self, n_samples, external_values):
        """Generate node values.

        external_values: dict mapping global index -> values array [n_samples]
                         or [n_samples, d] for multi-dim nodes
        Returns: list of arrays, each [n_samples] or [n_samples, d_node]
        """
        values = []
        for i in range(self.n_nodes):
            d = self.d_nodes[i]
            if len(self.parents[i]) == 0:
                # Root node — standardize to prevent Pareto/lognormal extremes
                # from cascading through downstream edge functions
                if d == 1:
                    root = _sample_root(n_samples, self.rng)
                    values.append(self._robust_standardize(root))
                else:
                    cols = [_sample_root(n_samples, self.rng) for _ in range(d)]
                    root_val = np.column_stack(cols)
                    if self.rng.random() < 0.5:
                        Q, _ = np.linalg.qr(self.rng.standard_normal((d, d)))
                        root_val = root_val @ Q
                    values.append(self._robust_standardize(root_val))
            elif self.concat_mode[i]:
                # Concat-then-transform: stack all parents, apply one function
                parent_arrays = []
                for is_internal, idx in self.parents[i]:
                    if is_internal:
                        pv = values[idx]
                    else:
                        pv = external_values[idx]
                    if pv.ndim == 1:
                        pv = pv.reshape(-1, 1)
                    parent_arrays.append(pv)
                concat_input = np.column_stack(parent_arrays)
                # Clip concatenated input to prevent edge fn amplification
                concat_input = np.clip(concat_input, -10, 10)
                fn = self.edge_fns[i][0]
                result = fn(concat_input)  # [n_samples]
                # Standardize after edge function to prevent cascading growth
                result = self._robust_standardize(result)
                if d > 1:
                    result = self._expand_to_multidim(result, d, n_samples)
                values.append(result)
            else:
                # Per-parent transform + aggregate (original behavior)
                parent_contributions = []
                for (is_internal, idx), fn in zip(self.parents[i], self.edge_fns[i]):
                    if is_internal:
                        parent_val = values[idx]
                    else:
                        parent_val = external_values[idx]
                    if parent_val.ndim == 1:
                        parent_val = parent_val.reshape(-1, 1)
                    parent_contributions.append(fn(parent_val))
                agg_result = _aggregate(parent_contributions, self.rng,
                                        expanded=self.expanded)
                # Standardize after aggregation
                agg_result = self._robust_standardize(agg_result)
                if d > 1:
                    agg_result = self._expand_to_multidim(
                        agg_result, d, n_samples)
                values.append(agg_result)
        return values


# ---------------------------------------------------------------------------
# Hierarchical SCM
# ---------------------------------------------------------------------------

def _build_hierarchical_scm(n_features, rng, expanded=False, synth_v5=False,
                            synth_v5_declone=True, max_parents=4):
    """Build a hierarchical SCM with multiple LCS.

    Returns:
        lcs_list: list of (LCS, offset) tuples
        global_nodes: list of all global node indices
        global_parents: dict mapping global_idx -> list of global parent indices
    """
    # Determine number and sizes of LCS
    target_nodes = n_features + 1  # +1 for target variable
    lcs_list = []
    node_offset = 0
    global_nodes = []  # list of global indices for all nodes
    global_d_nodes = {}  # global_idx -> dimensionality (for synth_v5)
    global_parents = {}  # global_idx -> list of global parent indices

    while node_offset < target_nodes:
        remaining = target_nodes - node_offset
        n_nodes = min(int(rng.integers(3, 9)), remaining)
        if remaining - n_nodes < 3 and remaining <= 8:
            n_nodes = remaining

        # External parents from previous LCS nodes
        external_parents = list(range(node_offset))

        lcs = LocalCausalStructure(n_nodes, external_parents, rng,
                                   expanded=expanded, synth_v5=synth_v5,
                                   external_d_nodes=global_d_nodes,
                                   synth_v5_declone=synth_v5_declone,
                                   max_parents=max_parents)
        lcs_list.append((lcs, node_offset))
        for j in range(n_nodes):
            global_idx = node_offset + j
            global_nodes.append(global_idx)
            global_d_nodes[global_idx] = lcs.d_nodes[j]
            # Map local parents to global indices
            gparents = []
            for is_internal, idx in lcs.parents[j]:
                if is_internal:
                    gparents.append(node_offset + idx)
                else:
                    gparents.append(idx)
            global_parents[global_idx] = gparents
        node_offset += n_nodes

    return lcs_list, global_nodes, global_parents


def _check_learnability(
    X,
    y,
    task_type,
    *,
    cls_min_score=0.60,
    cls_margin=0.10,
    reg_min_score=0.10,
):
    """Quick ExtraTrees check: can a simple model beat a constant baseline?

    Returns True if the dataset is learnable, False if it should be filtered.
    Inspired by TabICLv2's data filtering (rejects ~25-35% of datasets).
    """
    import warnings
    from sklearn.ensemble import ExtraTreesRegressor, ExtraTreesClassifier

    n, f = X.shape
    if n < 20:
        return True  # too small to test

    # Subsample large tables to keep ExtraTrees fast in prefetch workers.
    # 1000 rows is enough for a reliable OOB signal; fitting on 4096×384
    # is ~16x slower and causes CPU starvation with multiple DDP workers.
    max_rows = 1000
    if n > max_rows:
        idx = np.random.default_rng(0).choice(n, size=max_rows, replace=False)
        X_fit, y_fit = X[idx], y[idx]
    else:
        X_fit, y_fit = X, y

    # Use a fast model with limited depth
    if task_type == 'cls':
        model = ExtraTreesClassifier(
            n_estimators=15, max_depth=6, bootstrap=True,
            oob_score=True, random_state=0, n_jobs=1)
    else:
        model = ExtraTreesRegressor(
            n_estimators=15, max_depth=6, bootstrap=True,
            oob_score=True, random_state=0, n_jobs=1)

    # Replace NaN with 0 for sklearn
    X_clean = np.nan_to_num(X_fit, nan=0.0)
    y_fit_clean = y_fit

    try:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            model.fit(X_clean, y_fit_clean)
        if task_type == 'cls':
            # OOB AUC proxy: accuracy must meaningfully exceed chance
            n_classes = len(np.unique(y_fit_clean))
            return model.oob_score_ > max(1.0 / n_classes + cls_margin, cls_min_score)
        else:
            # OOB R2 must show real signal, not just noise fitting
            return model.oob_score_ > reg_min_score
    except Exception:
        return True  # if fitting fails, don't filter


def _check_icl_scaling(
    X,
    y,
    task_type,
    *,
    reg_min_score=0.10,
    min_improvement=0.03,
    n_context_sizes=4,
):
    """ICL scaling filter: does more context actually help?

    Fits ExtraTrees at logarithmically spaced context sizes and checks that:
    1. Final (largest context) R² > reg_min_score
    2. R² improves by at least min_improvement from smallest to largest

    This catches:
    - Pure noise (R² near 0 at all sizes)
    - Memorization/lookup (R² high at tiny context, no improvement)
    - Constant targets (R² 0 everywhere)

    And keeps datasets where more context genuinely helps — the hallmark of
    ICL-learnable data.
    """
    import warnings
    from sklearn.ensemble import ExtraTreesRegressor, ExtraTreesClassifier

    n, f = X.shape
    if n < 60:
        return True  # too small for meaningful scaling analysis

    # Subsample features and rows for speed
    max_rows = 1000
    rng = np.random.default_rng(1)
    if n > max_rows:
        idx = rng.choice(n, size=max_rows, replace=False)
        X_sub, y_sub = X[idx], y[idx]
        n = max_rows
    else:
        X_sub, y_sub = X.copy(), y.copy()

    X_clean = np.nan_to_num(X_sub, nan=0.0)

    # Shuffle to avoid ordering artifacts
    perm = rng.permutation(n)
    X_clean = X_clean[perm]
    y_sub = y_sub[perm]

    # Hold out a fixed test set (last 20%), vary context from the rest
    n_test = max(20, int(n * 0.2))
    n_pool = n - n_test
    X_test, y_test = X_clean[n_pool:], y_sub[n_pool:]
    X_pool, y_pool = X_clean[:n_pool], y_sub[:n_pool]

    # Logarithmically spaced context sizes from ~25% to 100% of pool
    min_ctx = max(20, int(n_pool * 0.25))
    max_ctx = n_pool
    if max_ctx <= min_ctx:
        return True

    context_sizes = np.unique(np.geomspace(min_ctx, max_ctx, n_context_sizes).astype(int))
    if len(context_sizes) < 2:
        return True

    scores = []
    for ctx_size in context_sizes:
        X_train, y_train = X_pool[:ctx_size], y_pool[:ctx_size]

        if len(X_test) < 10:
            continue

        try:
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                if task_type == 'cls':
                    model = ExtraTreesClassifier(
                        n_estimators=10, max_depth=5,
                        random_state=0, n_jobs=1)
                    model.fit(X_train, y_train)
                    score = model.score(X_test, y_test)
                else:
                    model = ExtraTreesRegressor(
                        n_estimators=10, max_depth=5,
                        random_state=0, n_jobs=1)
                    model.fit(X_train, y_train)
                    score = model.score(X_test, y_test)
                scores.append(float(score))
        except Exception:
            scores.append(0.0)

    if len(scores) < 2:
        return True

    final_score = scores[-1]
    improvement = scores[-1] - scores[0]

    # Both conditions must hold
    if task_type == 'reg':
        return final_score > reg_min_score and improvement > min_improvement
    else:
        n_classes = len(np.unique(y_sub))
        chance = 1.0 / max(n_classes, 2)
        return final_score > chance + 0.05 and improvement > min_improvement


_icl_filter_model = None
_icl_filter_model_path = None


def _get_icl_filter_model(model_path):
    """Lazily load a frozen LimiX model for ICL-based learnability filtering.

    The model is cached in a module-level variable so it's loaded once per
    worker process, not once per dataset.  CPU thread count is pinned to 2
    to avoid thread-contention when multiple prefetch workers run in parallel.
    """
    global _icl_filter_model, _icl_filter_model_path
    if _icl_filter_model is not None and _icl_filter_model_path == model_path:
        return _icl_filter_model

    import torch
    torch.set_num_threads(2)
    torch.set_num_interop_threads(1)

    from synthefy_tabular.utils.loading import load_model
    model = load_model(model_path, mask_prediction=True)
    model.eval()
    for p in model.parameters():
        p.requires_grad_(False)
    _icl_filter_model = model
    _icl_filter_model_path = model_path
    return model


def _check_learnability_icl(
    X,
    y,
    task_type,
    model_path,
    *,
    cls_min_auc=0.55,
    reg_min_r2=0.05,
    max_context=64,
    max_query=32,
    max_features=16,
):
    """ICL-based learnability check: can a frozen LimiX model beat chance?

    Runs a single forward pass on CPU with a context/query split.
    Returns True if the dataset is learnable, False if it should be filtered.

    Subsample sizes are kept small (64+32 rows, 16 features) because this
    is a binary keep/reject decision, not a precise score — and CPU inference
    on the 2M-param model is expensive.  At these sizes: ~60-70ms per call
    with 2 CPU threads (vs ~1.2s at 500+200 rows with 64 threads).
    """
    import torch

    n, f = X.shape
    if n < 20:
        return True

    model = _get_icl_filter_model(model_path)

    rng = np.random.default_rng(0)

    if f > max_features:
        feat_idx = rng.choice(f, size=max_features, replace=False)
        X = X[:, feat_idx]
        f = max_features

    total = min(n, max_context + max_query)
    idx = rng.choice(n, size=total, replace=False) if n > total else np.arange(n)
    n_ctx = min(max_context, total - 10)
    n_qry = total - n_ctx

    X_sub = X[idx]
    y_sub = y[idx].copy()

    X_sub = np.nan_to_num(X_sub, nan=0.0).astype(np.float32)

    if task_type == 'cls':
        classes = np.unique(y_sub[np.isfinite(y_sub)])
        n_classes = len(classes)
        if n_classes < 2:
            return True
        label_map = {c: i for i, c in enumerate(classes)}
        y_sub = np.array([label_map.get(v, 0) for v in y_sub], dtype=np.float32)
        n_classes = min(n_classes, 10)
    else:
        y_sub = y_sub.astype(np.float32)
        ctx_y = y_sub[:n_ctx]
        ctx_mean = np.nanmean(ctx_y)
        ctx_std = np.nanstd(ctx_y)
        if ctx_std < 1e-8:
            return True
        y_sub = (y_sub - ctx_mean) / ctx_std
        n_classes = None

    x_t = torch.from_numpy(X_sub).unsqueeze(0)
    y_t = torch.from_numpy(y_sub).unsqueeze(0)

    try:
        with torch.no_grad():
            out = model(x_t, y_t, eval_pos=n_ctx,
                        task_type=task_type,
                        y_type='cls' if task_type == 'cls' else 'reg')
    except Exception:
        return True

    if task_type == 'cls':
        logits = out['cls_output'][:, :n_qry, :n_classes]
        preds = logits.argmax(dim=-1).squeeze(0).numpy()
        true = y_sub[n_ctx:n_ctx + n_qry]
        acc = np.mean(preds == true)
        chance = 1.0 / n_classes
        return acc > chance + (cls_min_auc - 0.5)
    else:
        preds = out['reg_output'][:, :n_qry, 0].squeeze(0).numpy()
        true = y_sub[n_ctx:n_ctx + n_qry]
        ss_res = np.sum((true - preds) ** 2)
        ss_tot = np.sum((true - np.mean(true)) ** 2)
        r2 = 1.0 - ss_res / max(ss_tot, 1e-8)
        return r2 > reg_min_r2


def _extract_quality_task_rules(quality_rules, task_type):
    """Return task-specific rules from a quality rules dict.

    Supports either a top-level dict with `task_rules` or a direct
    task-keyed dict.
    """
    if not quality_rules or not isinstance(quality_rules, dict):
        return None
    task_map = quality_rules.get('task_rules', quality_rules)
    if task_type == 'cls':
        return task_map.get('classification') or task_map.get('cls')
    return task_map.get('regression') or task_map.get('reg')


def _compute_health_stats_fast(X, y, task_type):
    """Compute fast, training-time dataset health stats for filtering."""
    X = np.asarray(X)
    y = np.asarray(y)
    n_samples, n_features = X.shape

    if X.size == 0:
        nan_frac = 0.0
        nonconst = 0
        const_frac = 1.0
    else:
        nan_frac = float(np.isnan(X).mean())
        col_std = np.nanstd(X, axis=0)
        col_std = np.nan_to_num(col_std, nan=0.0, posinf=0.0, neginf=0.0)
        nonconst = int(np.sum(col_std > 1e-8))
        const_frac = float(1.0 - (nonconst / max(n_features, 1)))

    stats = {
        'nan_frac': nan_frac,
        'const_feature_frac': const_frac,
        'nonconstant_features': nonconst,
        'n_samples': int(n_samples),
        'n_features': int(n_features),
    }

    if task_type == 'cls':
        y_int = y.astype(np.int64, copy=False)
        vals, counts = np.unique(y_int, return_counts=True)
        stats['y_unique'] = int(len(vals))
        stats['minority_frac'] = float(counts.min() / max(len(y_int), 1)) if len(counts) > 0 else 0.0
    else:
        y_f = y.astype(np.float64, copy=False)
        y_f = np.nan_to_num(y_f, nan=0.0, posinf=0.0, neginf=0.0)
        stats['y_std'] = float(np.std(y_f))

    return stats


def _passes_quality_rules(X, y, task_type, quality_rules):
    """Return True if dataset passes mined quality rules."""
    task_rules = _extract_quality_task_rules(quality_rules, task_type)
    if task_rules is None:
        return True

    stats = _compute_health_stats_fast(X, y, task_type)

    max_nan_frac = task_rules.get('max_nan_frac')
    if max_nan_frac is not None and stats['nan_frac'] > float(max_nan_frac):
        return False

    max_const_feature_frac = task_rules.get('max_const_feature_frac')
    if max_const_feature_frac is not None and stats['const_feature_frac'] > float(max_const_feature_frac):
        return False

    min_nonconstant_features = task_rules.get('min_nonconstant_features')
    if min_nonconstant_features is not None and stats['nonconstant_features'] < int(min_nonconstant_features):
        return False

    if task_type == 'cls':
        min_y_unique = task_rules.get('min_y_unique')
        if min_y_unique is not None and stats['y_unique'] < int(min_y_unique):
            return False
        min_minority_frac = task_rules.get('min_minority_frac')
        if min_minority_frac is not None and stats['minority_frac'] < float(min_minority_frac):
            return False
    else:
        min_y_std = task_rules.get('min_y_std')
        if min_y_std is not None and stats['y_std'] < float(min_y_std):
            return False

    return True


def _apply_feature_importances(X, rng, mild=False):
    """Apply power-law random feature importances to columns.

    Inspired by TabICLv2's random weights: w_m = m^{-q} * exp(N(0, sigma^2))
    This makes some features much more important than others, matching real data.

    Args:
        mild: if True, use weaker power-law (for CLS where MI is easily overshoot)
    """
    n_features = X.shape[1]
    if n_features <= 1:
        return X

    if mild:
        # Milder: q in [0.3, 2.0], sigma in [0.001, 1.0]
        q = np.exp(rng.uniform(np.log(0.3), np.log(2.0)))
        sigma = np.exp(rng.uniform(np.log(1e-3), np.log(1.0)))
    else:
        q = np.exp(rng.uniform(np.log(0.5), np.log(5.0)))
        sigma = np.exp(rng.uniform(np.log(1e-3), np.log(2.0)))

    ranks = np.arange(1, n_features + 1, dtype=float)
    weights = ranks ** (-q) * np.exp(rng.normal(0, sigma, n_features))
    weights = np.abs(weights)
    # Floor: no feature gets fully zeroed (prevents constant-feature artifacts)
    weights = np.maximum(weights, 0.05 * weights.max())
    weights /= (weights.sum() + 1e-8)
    weights *= n_features  # normalize so mean weight = 1

    # Shuffle so importance isn't correlated with column order
    rng.shuffle(weights)

    return X * weights[np.newaxis, :]


def _create_informative_categorical(X, col, rng, cardinality):
    """Replace a continuous column with informative categorical values.

    Uses the existing continuous value as a latent variable z, then assigns
    categories via one of three methods (randomly chosen):
      1. Nearest-prototype: sample K prototypes, assign to nearest
      2. Quantile-binning: equal-frequency bins
      3. Softmax: random linear projections with temperature

    Optional 2-10% assignment noise for soft boundaries.
    """
    z = X[:, col].copy()
    n = len(z)
    method = int(rng.integers(0, 3))

    if method == 0:
        # Nearest-prototype
        zmin, zmax = z.min(), z.max()
        if zmax - zmin < 1e-8:
            X[:, col] = rng.integers(0, cardinality, size=n).astype(np.float64)
            return
        prototypes = rng.uniform(zmin, zmax, size=cardinality)
        dists = np.abs(z[:, None] - prototypes[None, :])  # [n, K]
        cats = dists.argmin(axis=1)
    elif method == 1:
        # Quantile-binning
        percentiles = np.linspace(0, 100, cardinality + 1)[1:-1]
        thresholds = np.nanpercentile(z, percentiles)
        thresholds = np.unique(thresholds)
        cats = np.digitize(z, thresholds)
    else:
        # Softmax: random projection + temperature
        zmin, zmax = z.min(), z.max()
        if zmax - zmin < 1e-8:
            X[:, col] = rng.integers(0, cardinality, size=n).astype(np.float64)
            return
        z_norm = (z - zmin) / (zmax - zmin)  # [0, 1]
        # Random weights and biases for each category
        w = rng.standard_normal(cardinality)
        b = rng.standard_normal(cardinality)
        temp = rng.uniform(0.3, 2.0)
        logits = (z_norm[:, None] * w[None, :] + b[None, :]) / temp  # [n, K]
        # Stable softmax
        logits -= logits.max(axis=1, keepdims=True)
        exp_logits = np.exp(logits)
        probs = exp_logits / (exp_logits.sum(axis=1, keepdims=True) + 1e-8)
        # Sample from categorical distribution
        cats = np.array([rng.choice(cardinality, p=p) for p in probs])

    # Optional assignment noise: flip 2-10% of assignments
    if rng.random() < 0.5:
        noise_rate = rng.uniform(0.02, 0.10)
        n_flip = max(1, int(n * noise_rate))
        flip_idx = rng.choice(n, size=n_flip, replace=False)
        cats[flip_idx] = rng.integers(0, cardinality, size=n_flip)

    X[:, col] = cats.astype(np.float64)


def _create_repeated_entity_categorical(X, col, rng):
    """Replace a column with a repeated-entity high-cardinality categorical.

    This targets Amazon-like lookup tasks: repeated IDs with skewed frequency
    and a simple shared entity effect that context rows can teach. The goal is
    not giant mostly-unique IDs; it is head-tail categorical support with
    enough repeats for in-context generalization.

    Returns:
        row_effect: per-row lookup effect induced by the sampled entity IDs
        entity_meta: summary stats for the sampled entity IDs
    """
    n = X.shape[0]
    latent = X[:, col].copy()

    # Keep the average number of repeats comfortably above 1 so the model can
    # infer useful lookup behavior from context rows.
    max_cardinality = int(min(512, max(16, n // 4)))
    cardinality = int(np.exp(rng.uniform(np.log(8.0), np.log(float(max_cardinality)))))
    cardinality = int(np.clip(cardinality, 8, max_cardinality))

    # Zipf-like head-tail frequency: a few common entities, many rare ones.
    alpha = rng.uniform(0.8, 1.6)
    ranks = np.arange(1, cardinality + 1, dtype=np.float64)
    probs = ranks ** (-alpha)
    probs *= np.exp(rng.normal(0.0, 0.15, size=cardinality))
    probs = np.maximum(probs, 1e-12)
    probs /= probs.sum()
    entity_ids = rng.choice(cardinality, size=n, replace=True, p=probs)

    # Shared lookup-table effects with coarse group structure so different
    # entities can have related behavior without being identical.
    n_groups = max(2, min(32, int(np.sqrt(cardinality))))
    entity_group = rng.integers(0, n_groups, size=cardinality)
    group_effect = rng.standard_normal(n_groups)
    entity_effect = (
        0.75 * group_effect[entity_group]
        + 0.35 * rng.standard_normal(cardinality)
    )
    entity_effect = entity_effect - entity_effect.mean()
    entity_effect /= (np.std(entity_effect) + 1e-8)
    row_effect = entity_effect[entity_ids]
    counts = np.bincount(entity_ids, minlength=cardinality)
    observed = counts[counts > 0]

    # Blend a little of the original latent signal back in so the categorical
    # column is not a pure synthetic ID with no local structure at all.
    latent_std = np.std(latent)
    if latent_std > 1e-8 and rng.random() < 0.5:
        latent_norm = (latent - np.mean(latent)) / latent_std
        row_effect = row_effect + 0.25 * latent_norm

    X[:, col] = entity_ids.astype(np.float64)
    entity_meta = {
        'cardinality': int(cardinality),
        'observed_entities': int(len(observed)),
        'repeated_entity_fraction': float(
            np.mean(observed >= 2) if len(observed) > 0 else 0.0
        ),
        'max_entity_fraction': float(counts.max() / max(n, 1)),
        'mean_count_per_observed_entity': float(
            observed.mean() if len(observed) > 0 else 0.0
        ),
    }
    return row_effect.astype(np.float64), entity_meta


def _create_nominal_categorical(X, col, n_samples, rng, cardinality,
                                task_type, n_features, discrete_cols, cat_cols_iter):
    """Create nominal (non-ordinal) categorical features with random effects.

    Two types:
    1. Independent nominal: categories with random effects, no inherent ordering
    2. Crossed nominal: two columns interact (city × product_type)

    Args:
        X: feature matrix [n_samples, n_features] — modified in-place
        col: primary column index to overwrite
        n_samples: number of samples
        rng: numpy random generator
        cardinality: target cardinality for the primary column
        task_type: 'cls' or 'reg'
        n_features: total features
        discrete_cols: set of already-discrete column indices
        cat_cols_iter: set of columns being categorified this iteration

    Returns:
        row_effect: per-row additive target effect [n_samples]
        n_cols_used: number of columns consumed (1 for independent, 2 for crossed)
    """
    use_crossed = rng.random() < 0.4 and n_features >= 4

    # --- Generate primary nominal column ---
    # Zipf-like frequency: some categories common, others rare
    alpha = rng.uniform(0.6, 1.4)
    ranks = np.arange(1, cardinality + 1, dtype=np.float64)
    probs = ranks ** (-alpha)
    probs /= probs.sum()
    cat_ids = rng.choice(cardinality, size=n_samples, replace=True, p=probs)

    # Random effect per category (NOT ordered — this is the key difference
    # from ordinal categoricals derived from continuous values)
    cat_effects = rng.standard_normal(cardinality)
    cat_effects -= cat_effects.mean()
    cat_effects /= (np.std(cat_effects) + 1e-8)
    row_effect = cat_effects[cat_ids].astype(np.float64)

    X[:, col] = cat_ids.astype(np.float64)

    if use_crossed:
        # --- Crossed nominal: two columns interact ---
        # Find a second column that's not already categorified
        available = [c for c in range(n_features)
                     if c != col and c not in discrete_cols and c not in cat_cols_iter]
        if not available:
            return row_effect, 1

        col2 = rng.choice(available)
        K2 = int(rng.integers(3, min(16, max(4, n_samples // 8))))
        alpha2 = rng.uniform(0.6, 1.4)
        ranks2 = np.arange(1, K2 + 1, dtype=np.float64)
        probs2 = ranks2 ** (-alpha2)
        probs2 /= probs2.sum()
        cat_ids2 = rng.choice(K2, size=n_samples, replace=True, p=probs2)

        # Second column's main effect
        cat_effects2 = rng.standard_normal(K2)
        cat_effects2 -= cat_effects2.mean()
        cat_effects2 /= (np.std(cat_effects2) + 1e-8)

        # Sparse interaction: only ~30% of (K1 × K2) cells are nonzero
        interaction = np.zeros((cardinality, K2))
        n_nonzero = max(1, int(0.3 * cardinality * K2))
        nonzero_idx = rng.choice(cardinality * K2, size=n_nonzero, replace=False)
        interaction.ravel()[nonzero_idx] = rng.standard_normal(n_nonzero) * 0.5

        # Combined effect: main1 + main2 + interaction
        row_effect2 = cat_effects2[cat_ids2].astype(np.float64)
        row_interaction = interaction[cat_ids, cat_ids2].astype(np.float64)
        row_effect = row_effect + row_effect2 + row_interaction

        X[:, col2] = cat_ids2.astype(np.float64)

        return row_effect, 2

    return row_effect, 1


def _apply_kumaraswamy_warping(X, col, rng):
    """Apply Kumaraswamy warping to a column: 1 - (1 - x^a)^b.

    Inspired by TabICLv2's numerical converter. Creates non-linear monotonic
    transformations of feature values.
    """
    a = np.exp(rng.uniform(np.log(0.2), np.log(5.0)))
    b = np.exp(rng.uniform(np.log(0.2), np.log(5.0)))

    v = X[:, col]
    # Min-max scale to [0, 1]
    vmin, vmax = v.min(), v.max()
    if vmax - vmin < 1e-8:
        return  # skip constant columns
    v_scaled = (v - vmin) / (vmax - vmin)
    v_scaled = np.clip(v_scaled, 1e-8, 1 - 1e-8)

    X[:, col] = 1 - (1 - v_scaled ** a) ** b


def _apply_feature_rescaling(X, rng):
    """Apply random per-column rescaling: multiply each column by LogUniform(0.1, 10).

    Inspired by TabICLv2's "random rescale" step. Real features have different scales.
    """
    n_features = X.shape[1]
    scales = np.exp(rng.uniform(np.log(0.1), np.log(10.0), n_features))
    return X * scales[np.newaxis, :]


def _inject_heavy_tails(X, rng, causal_cols=None, exclude_cols=None):
    """Inject heavy-tailed outliers into random features.

    Real data has median feature kurtosis ~1.5, synthetic has ~-0.9.
    Adds: (1) per-feature Student-t spikes, (2) per-row corruption.

    Args:
        causal_cols: optional set/list of column indices that are causal
            ancestors. When provided, preferentially target non-causal columns
            to boost kurtosis realism with minimal learnability cost.
        exclude_cols: optional set of column indices to exclude entirely
            (e.g., discrete columns that should stay integer-valued).
    """
    n_samples, n_features = X.shape
    if exclude_cols is None:
        exclude_cols = set()

    # Per-feature: replace small fraction with heavy-tailed draws
    eligible_features = [c for c in range(n_features) if c not in exclude_cols]
    if not eligible_features:
        return X
    n_cols_affected = max(1, int(len(eligible_features) * rng.uniform(0.1, 0.5)))
    if causal_cols is not None and len(causal_cols) < n_features:
        # Prefer non-causal eligible columns (80% non-causal, 20% causal)
        causal_set = set(causal_cols)
        non_causal = [c for c in eligible_features if c not in causal_set]
        causal_elig = [c for c in eligible_features if c in causal_set]
        n_from_non_causal = min(len(non_causal),
                                max(1, int(n_cols_affected * 0.8)))
        n_from_causal = min(len(causal_elig),
                            n_cols_affected - n_from_non_causal)
        affected_cols = np.concatenate([
            rng.choice(non_causal, size=n_from_non_causal, replace=False)
            if non_causal else np.array([], dtype=int),
            rng.choice(causal_elig, size=n_from_causal, replace=False)
            if causal_elig and n_from_causal > 0 else np.array([], dtype=int),
        ]).astype(int)
    else:
        affected_cols = rng.choice(eligible_features,
                                   size=min(n_cols_affected, len(eligible_features)),
                                   replace=False)
    for col in affected_cols:
        col_std = np.std(X[:, col])
        if col_std < 1e-8:
            continue
        mu = np.mean(X[:, col])

        if rng.random() < 0.5:
            # Spike+slab mixture: produces realistic high kurtosis
            # Real kurtosis ~82 often comes from zero-inflation + rare extremes
            p_zero = rng.uniform(0.6, 0.95)   # big spike at mode
            p_out = rng.uniform(0.005, 0.03)   # rare extreme outliers

            mask_zero = rng.random(n_samples) < p_zero
            X[mask_zero, col] = 0.0

            mask_out = (~mask_zero) & (rng.random(n_samples) < p_out)
            if np.any(mask_out):
                df = rng.uniform(1.5, 3.0)     # heavier tails than original
                amp = rng.uniform(8.0, 30.0)   # bigger amplification
                X[mask_out, col] = mu + amp * col_std * rng.standard_t(
                    df, size=int(mask_out.sum()))
        else:
            # Original: Student-t outlier injection
            outlier_frac = rng.uniform(0.01, 0.05)
            n_outliers = max(1, int(n_samples * outlier_frac))
            outlier_idx = rng.choice(n_samples, size=n_outliers, replace=False)
            df = rng.uniform(2.0, 5.0)
            outlier_vals = rng.standard_t(df, size=n_outliers)
            amp = rng.uniform(3.0, 10.0)
            X[outlier_idx, col] = mu + amp * col_std * outlier_vals

    # Per-row corruption: small fraction of rows get multiple columns corrupted
    if rng.random() < 0.3:
        corrupt_frac = rng.uniform(0.005, 0.02)
        n_corrupt = max(1, int(n_samples * corrupt_frac))
        corrupt_rows = rng.choice(n_samples, size=n_corrupt, replace=False)
        # Only corrupt eligible (continuous) columns
        corrupt_pool = eligible_features if eligible_features else list(range(n_features))
        n_cols_corrupt = max(1, int(len(corrupt_pool) * rng.uniform(0.2, 0.6)))
        corrupt_cols = rng.choice(corrupt_pool,
                                  size=min(n_cols_corrupt, len(corrupt_pool)),
                                  replace=False)
        for col in corrupt_cols:
            col_std = np.std(X[:, col])
            if col_std < 1e-8:
                continue
            shift = rng.uniform(5, 20) * col_std * rng.choice([-1, 1])
            X[corrupt_rows, col] += shift

    return X


def _add_latent_bayes_error(X, y_raw, n_features, rng, task_type):
    """Add latent dimensions and hard negatives to create realistic Bayes error.

    Three mechanisms:
    1. Latent influence: unobserved variables affect y, creating irreducible noise
    2. Hard negatives: features correlated with causal features but NOT with y
    3. Feature-dependent label noise (for classification)
    """
    n_samples = X.shape[0]

    # 1. Latent influence: generate hidden features that affect y but aren't in X
    n_latent = max(1, int(n_features * rng.uniform(0.05, 0.2)))
    for _ in range(n_latent):
        latent = _sample_root(n_samples, rng)
        fn = _make_edge_fn(1, rng, expanded=True)
        weight = rng.normal(0, 0.5)
        y_raw = y_raw + weight * fn(latent.reshape(-1, 1)).ravel()

    # 2. Hard negatives: features correlated with important features but not y
    if n_features >= 4 and rng.random() < 0.4:
        n_hard = max(1, int(n_features * rng.uniform(0.05, 0.15)))
        # Pick source columns (likely important ones)
        src_cols = rng.choice(n_features, size=min(n_hard, n_features),
                              replace=False)
        # Pick target columns to overwrite
        available = [c for c in range(n_features) if c not in src_cols]
        if available:
            tgt_cols = rng.choice(available,
                                  size=min(n_hard, len(available)),
                                  replace=False)
            for src, tgt in zip(src_cols, tgt_cols):
                # Correlated with src but add enough noise to break y-correlation
                noise_level = rng.uniform(0.5, 2.0)
                col_std = np.std(X[:, src])
                X[:, tgt] = X[:, src] + rng.normal(0, col_std * noise_level,
                                                     n_samples)

    return X, y_raw


def _probabilistic_label(y_raw, X, n_classes, rng):
    """Generate classification labels using non-percentile strategies.

    Instead of quantile bucketing (which is a memorizable fingerprint), use
    one of 4 probabilistic labeling strategies that create diverse decision
    boundaries in feature space.

    Args:
        y_raw: continuous SCM target [n_samples]
        X: feature matrix [n_samples, n_features] (may contain NaN)
        n_classes: number of classes
        rng: numpy random generator

    Returns:
        y: integer class labels [n_samples] as float32
    """
    n_samples, n_features = X.shape
    strategy = int(rng.integers(0, 4))

    # Work with a clean copy of X for computing labels (replace NaN with 0)
    X_clean = X.copy()
    nan_mask = ~np.isfinite(X_clean)
    X_clean[nan_mask] = 0.0

    if strategy == 0:
        # --- Logistic/Softmax labeler ---
        # Pick 1-3 feature columns, compute logits via random projection
        n_proj = min(int(rng.integers(1, 4)), n_features)
        proj_cols = rng.choice(n_features, size=n_proj, replace=False)
        X_proj = X_clean[:, proj_cols]
        # Standardize
        mu = X_proj.mean(axis=0, keepdims=True)
        std = X_proj.std(axis=0, keepdims=True)
        std = np.where(std < 1e-8, 1.0, std)
        X_proj = (X_proj - mu) / std
        # Random weights and bias
        W = rng.standard_normal((n_proj, n_classes)) * 0.8
        b = rng.standard_normal(n_classes) * 0.3
        temp = rng.uniform(0.5, 2.0)
        logits = (X_proj @ W + b) / temp  # [n_samples, n_classes]
        # Also mix in the SCM target for some signal continuity
        y_std = np.std(y_raw)
        if y_std > 1e-8:
            y_norm = (y_raw - np.mean(y_raw)) / y_std
            logits[:, 0] += y_norm * rng.uniform(0.3, 1.0)
        # Stable softmax
        logits -= logits.max(axis=1, keepdims=True)
        exp_logits = np.exp(logits)
        probs = exp_logits / (exp_logits.sum(axis=1, keepdims=True) + 1e-8)
        # Sample from distribution
        y = np.array([rng.choice(n_classes, p=p) for p in probs],
                     dtype=np.float32)

    elif strategy == 1:
        # --- Random tree labeler ---
        # Axis-aligned splits on 2-4 features, depth 2-4
        n_split_feats = min(int(rng.integers(2, 5)), n_features)
        split_cols = rng.choice(n_features, size=n_split_feats, replace=False)
        depth = int(rng.integers(2, 5))
        # Build random splits: at each level, pick a feature and threshold
        n_leaves = 2 ** depth
        leaf_labels = rng.integers(0, n_classes, size=n_leaves).astype(np.float32)
        # Assign each sample to a leaf via recursive splitting
        leaf_idx = np.zeros(n_samples, dtype=np.int64)
        for d in range(depth):
            feat = split_cols[d % len(split_cols)]
            col_vals = X_clean[:, feat]
            finite_vals = col_vals[np.isfinite(col_vals)]
            if len(finite_vals) > 1:
                pct = rng.uniform(25, 75)
                threshold = np.nanpercentile(finite_vals, pct)
            else:
                threshold = 0.0
            goes_right = col_vals > threshold
            leaf_idx = leaf_idx * 2 + goes_right.astype(np.int64)
        leaf_idx = leaf_idx % n_leaves
        y = leaf_labels[leaf_idx]

    elif strategy == 2:
        # --- Class-conditional Gaussian labeler ---
        # Each class has a random center in 1-3 feature dimensions
        # Assign to nearest class center
        n_dims = min(int(rng.integers(1, 4)), n_features)
        dim_cols = rng.choice(n_features, size=n_dims, replace=False)
        X_sub = X_clean[:, dim_cols]
        # Standardize
        mu = X_sub.mean(axis=0, keepdims=True)
        std = X_sub.std(axis=0, keepdims=True)
        std = np.where(std < 1e-8, 1.0, std)
        X_sub = (X_sub - mu) / std
        # Random class centers
        centers = rng.standard_normal((n_classes, n_dims)) * 1.5
        # Compute distances and assign
        dists = np.sum((X_sub[:, None, :] - centers[None, :, :]) ** 2,
                       axis=2)  # [n_samples, n_classes]
        y = dists.argmin(axis=1).astype(np.float32)

    else:
        # --- Threshold-based labeler ---
        # Pick 1-2 features, define class regions via thresholds
        n_thresh_feats = min(int(rng.integers(1, 3)), n_features)
        thresh_cols = rng.choice(n_features, size=n_thresh_feats, replace=False)
        if n_thresh_feats == 1:
            # Single feature: n_classes-1 thresholds with random ordering
            col_vals = X_clean[:, thresh_cols[0]]
            # Random (non-percentile) thresholds
            vmin, vmax = np.min(col_vals), np.max(col_vals)
            if vmax - vmin < 1e-8:
                y = rng.integers(0, n_classes, size=n_samples).astype(np.float32)
            else:
                thresholds = np.sort(rng.uniform(vmin, vmax,
                                                 size=n_classes - 1))
                region = np.digitize(col_vals, thresholds)
                # Random permutation of region-to-class mapping
                perm = rng.permutation(n_classes)
                y = perm[region].astype(np.float32)
        else:
            # Two features: grid of regions
            col0 = X_clean[:, thresh_cols[0]]
            col1 = X_clean[:, thresh_cols[1]]
            n_splits = max(2, int(np.ceil(np.sqrt(n_classes))))
            vmin0, vmax0 = np.min(col0), np.max(col0)
            vmin1, vmax1 = np.min(col1), np.max(col1)
            if vmax0 - vmin0 < 1e-8 or vmax1 - vmin1 < 1e-8:
                y = rng.integers(0, n_classes, size=n_samples).astype(np.float32)
            else:
                t0 = np.sort(rng.uniform(vmin0, vmax0, size=n_splits - 1))
                t1 = np.sort(rng.uniform(vmin1, vmax1, size=n_splits - 1))
                r0 = np.digitize(col0, t0)
                r1 = np.digitize(col1, t1)
                grid_idx = r0 * n_splits + r1
                # Map grid cells to classes (random assignment)
                n_cells = n_splits * n_splits
                cell_to_class = rng.integers(0, n_classes, size=n_cells)
                y = cell_to_class[grid_idx % n_cells].astype(np.float32)

    # Safety: ensure labels are in valid range
    y = np.clip(y, 0, n_classes - 1).astype(np.float32)
    return y


def _apply_heteroskedastic_label_noise(y, X, n_classes, rng):
    """Apply feature-dependent label noise: flip rate depends on feature values.

    More realistic than uniform random flips. Real data has annotation noise
    that's higher for ambiguous/borderline samples.
    """
    n_samples = len(y)
    n_features = X.shape[1]

    # Pick a feature to condition noise on
    cond_col = rng.integers(0, n_features)
    cond_vals = X[:, cond_col]

    # Samples near the median of the conditioning feature get more noise
    median_val = np.median(cond_vals)
    distance = np.abs(cond_vals - median_val)
    # Normalize distance to [0, 1]
    max_dist = np.max(distance) + 1e-8
    normalized_dist = distance / max_dist

    # Flip probability: high near median, low at extremes
    base_rate = rng.uniform(0.02, 0.10)
    max_rate = rng.uniform(0.10, 0.25)
    flip_prob = base_rate + (max_rate - base_rate) * (1 - normalized_dist)

    # Apply flips
    flip_mask = rng.random(n_samples) < flip_prob
    n_flips = flip_mask.sum()
    if n_flips > 0:
        y[flip_mask] = rng.integers(0, n_classes, size=n_flips).astype(np.float32)

    return y


def _generate_regression_prior(n_samples, n_features, rng, reg_denoise=False,
                               reg_dense=False, reg_deterministic_prob=0.20,
                               pareto_importance_prob=0.0,
                               latent_factor_prob=0.0,
                               X_scm=None, X_scm_target=None):
    """Generate regression dataset matching common real-world patterns.

    Covers regimes the SCM generator under-represents: real regression is
    often near-additive with many weak features, correlated predictors,
    realistic noise levels, and occasional outliers.

    Randomly selects one of ten target types:
      0. Dense linear: y = X @ beta, many small t-distributed coefficients
      1. Sparse linear: y = X_active @ beta, few strong effects
      2. GAM: y = sum of smooth univariate functions on active features
      3. Additive + pairwise: y = X @ beta + sum(x_i * x_j interactions)
      4. Random MLP: deep nonlinear target
      5. Random tree: piecewise-constant target
      6. Radial / RBF network: sum_j w_j * exp(-||x-c_j||^2 / 2s_j^2)
      7. Fourier features: sum_j a_j * cos(omega_j · x + phi_j)
      8. Chained trig / kinematics-like: recursive sin/cos composition
      9. Full polynomial surface: x^T M x + w^T x + b over many features

    Args:
        X_scm: optional pre-built feature matrix from SCM. When provided,
               skips Gaussian X generation and uses this instead. This gives
               rich/correlated/discrete features paired with clean targets.
    """
    use_scm_features = X_scm is not None
    prior_meta = {
        'generator_family': 'regression_prior',
        'use_scm_features': bool(use_scm_features),
    }
    if use_scm_features:
        # Hybrid mode: use SCM-generated features with regression prior targets.
        # The SCM provides rich structure (correlations, discretization, multi-modal)
        # while the prior provides clean, interpretable target functions.
        X_model = np.array(X_scm, dtype=np.float32, copy=True)
        target_source = X_scm_target if X_scm_target is not None else X_scm
        X = np.array(target_source, dtype=np.float64, copy=True)
        nan_mask = np.isnan(X)
        if np.any(nan_mask):
            col_means = np.nanmean(X, axis=0)
            col_means = np.where(np.isfinite(col_means), col_means, 0.0)
            X = np.where(nan_mask, col_means[np.newaxis, :], X)
        X = np.where(np.isfinite(X), X, 0.0)
        # Standardize columns so target functions see unit-scale inputs
        for col in range(n_features):
            v = X[:, col]
            std = np.std(v)
            if std > 1e-8:
                X[:, col] = (v - np.mean(v)) / std
            else:
                X[:, col] = v - np.mean(v)
    else:
        X = rng.standard_normal((n_samples, n_features)).astype(np.float64)
        X_model = None

    # --- Correlated features (70%) ---
    if not use_scm_features and n_features >= 4 and rng.random() < 0.7:
        n_blocks = max(1, min(int(rng.integers(2, 8)), n_features // 2))
        block_size = n_features // n_blocks
        for b in range(n_blocks):
            start = b * block_size
            end = min(start + block_size, n_features)
            k = end - start
            if k < 2:
                continue
            A = rng.standard_normal((k, k)) * 0.3
            cov = A @ A.T + np.eye(k) * 0.1
            try:
                L = np.linalg.cholesky(cov)
                X[:, start:end] = X[:, start:end] @ L.T
            except np.linalg.LinAlgError:
                pass

    # --- Heavy-tailed features (30%) ---
    if not use_scm_features and rng.random() < 0.3:
        n_heavy = max(1, int(rng.integers(1, max(2, n_features // 4 + 1))))
        heavy_cols = rng.choice(n_features, size=n_heavy, replace=False)
        for col in heavy_cols:
            df = rng.uniform(3, 8)
            X[:, col] = rng.standard_t(df=df, size=n_samples)

    def _sample_active_count(
        min_active: int,
        max_active: int,
        *,
        dense_floor: int | None = None,
    ) -> int:
        hi = max(1, min(max_active, n_features))
        lo = min(min_active, hi)
        if reg_dense and dense_floor is not None:
            lo = min(max(dense_floor, min_active), hi)
        return int(rng.integers(lo, hi + 1))

    # --- Choose target type ---
    # 10 types: original linear/additive families plus stronger smooth
    # multivariate priors for spatial, kinematic, and process-style targets.
    if reg_dense:
        # Dense mode favors target types where many features matter jointly.
        reg_type = int(rng.choice(
            [0, 2, 3, 4, 6, 7, 8, 9],
            p=[0.06, 0.08, 0.10, 0.16, 0.22, 0.15, 0.13, 0.10],
        ))
    else:
        # Upweight smooth multivariate priors (RBF, Fourier, chained trig,
        # polynomial surface) to address spatial/kinematic/process gaps.
        reg_type = int(rng.choice(
            10,
            p=[0.08, 0.07, 0.10, 0.08, 0.10, 0.05, 0.18, 0.15, 0.11, 0.08],
        ))
    # Latent-factor override: shared low-dim structure across many features.
    # Replaces the chosen reg_type with reg_type=10 so the branches below
    # remain unchanged.
    if latent_factor_prob > 0 and rng.random() < latent_factor_prob and n_features >= 4:
        reg_type = 10
    smooth_joint_prior = reg_type in (6, 7, 8, 9)
    reg_type_names = {
        0: 'dense_linear',
        1: 'sparse_linear',
        2: 'gam',
        3: 'additive_pairwise',
        4: 'random_mlp',
        5: 'random_tree',
        6: 'radial_distance',
        7: 'fourier_features',
        8: 'chained_trig',
        9: 'polynomial_surface',
        10: 'latent_factor',
    }
    prior_meta['reg_type_id'] = int(reg_type)
    prior_meta['reg_type_name'] = reg_type_names[int(reg_type)]
    prior_meta['smooth_joint_prior'] = bool(smooth_joint_prior)

    # Pareto feature-importance gate. Build a per-feature multiplier that
    # concentrates total importance on a few features (a few × 10–100 the
    # rest) — mirrors real datasets where many columns are weak distractors
    # and a few are decisive (drug fingerprints, demographic indicators,
    # high-cardinality molecular features). Applied below to the linear
    # families (dense/sparse linear, GAM, additive_pairwise) where there is
    # an explicit per-feature weight; ignored for nonlinear branches whose
    # importance structure is implicit (MLP, tree, Fourier, RBF).
    pareto_w = None
    if pareto_importance_prob > 0 and rng.random() < pareto_importance_prob and n_features >= 2:
        alpha_pareto = float(rng.uniform(1.5, 3.0))
        raw = rng.pareto(alpha_pareto, size=n_features) + 1.0
        # Normalize so the average importance is 1.0 (preserves overall y scale).
        pareto_w = raw / raw.mean()
        prior_meta['pareto_importance'] = True
        prior_meta['pareto_alpha'] = alpha_pareto

    if reg_type == 0:
        # Dense linear: many small effects (ridge-like)
        beta = rng.standard_t(df=3, size=n_features)
        beta /= max(np.sqrt(n_features), 1)
        if pareto_w is not None:
            beta = beta * pareto_w
        y = X @ beta
        prior_meta['active_dims'] = int(n_features)

    elif reg_type == 1:
        # Sparse linear: few strong effects
        k = max(1, int(rng.integers(1, max(2, n_features // 3 + 1))))
        active = rng.choice(n_features, size=k, replace=False)
        beta = np.zeros(n_features)
        beta[active] = rng.standard_normal(k) * 2.0 / max(np.sqrt(k), 1)
        if pareto_w is not None:
            # Pareto on top of sparsity: among the k active features, importance
            # is even more skewed (1 dominates among the active).
            beta = beta * pareto_w
        y = X @ beta
        prior_meta['active_dims'] = int(k)

    elif reg_type == 2:
        # GAM: sum of smooth univariate functions
        n_active = _sample_active_count(2, 20, dense_floor=4)
        active = rng.choice(n_features, size=n_active, replace=False)
        y = np.zeros(n_samples)
        for col in active:
            fn_type = int(rng.integers(0, 8))
            x_col = X[:, col]
            weight = rng.standard_normal() / max(np.sqrt(n_active), 1)
            if pareto_w is not None:
                weight = float(weight * pareto_w[col])
            if fn_type == 0:
                y += weight * x_col
            elif fn_type == 1:
                y += weight * (x_col ** 2 - 1)
            elif fn_type == 2:
                y += weight * np.clip(x_col ** 3, -50, 50) * 0.1
            elif fn_type == 3:
                y += weight * np.sign(x_col) * np.log1p(np.abs(x_col))
            elif fn_type == 4:
                y += weight * np.tanh(x_col)
            elif fn_type == 5:
                # sin: periodic patterns (physics, seasonal, etc.)
                freq = rng.uniform(0.5, 3.0)
                y += weight * np.sin(freq * x_col)
            elif fn_type == 6:
                # cos: phase-shifted periodic
                freq = rng.uniform(0.5, 3.0)
                y += weight * np.cos(freq * x_col)
            else:
                # exp decay: exponential proximity patterns
                y += weight * np.exp(-0.5 * x_col ** 2)
        prior_meta['active_dims'] = int(n_active)

    elif reg_type == 3:
        # Additive + pairwise interactions
        beta = rng.standard_normal(n_features) / max(np.sqrt(n_features), 1)
        if pareto_w is not None:
            beta = beta * pareto_w
        y = X @ beta
        max_pairs = n_features * (n_features - 1) // 2
        n_interactions = max(1, int(rng.integers(1, min(10, max_pairs + 1))))
        for _ in range(n_interactions):
            if n_features < 2:
                break
            i, j = rng.choice(n_features, size=2, replace=False)
            weight = rng.standard_normal() * 0.3 / max(np.sqrt(n_interactions), 1)
            y += weight * X[:, i] * X[:, j]
        prior_meta['active_dims'] = int(n_features)
        prior_meta['interaction_terms'] = int(n_interactions)

    elif reg_type == 4:
        # Random MLP: smooth nonlinear target (forward kinematics, spatial, etc.)
        # Use wider/deeper networks with smooth activations to produce
        # gradually-varying response surfaces (not piecewise-linear from ReLU).
        _mlp_hi = max(4, min(n_features + 1, 30))
        n_active = min(n_features, _sample_active_count(2, _mlp_hi - 1, dense_floor=4))
        active = rng.choice(n_features, size=n_active, replace=False)
        X_active = X[:, active]

        # Wider hidden layers (64-512) produce smoother manifolds
        hid = int(rng.integers(64, 513))
        n_layers = int(rng.integers(2, 5))  # 2-4 layers

        # Smooth activations dominate (GELU/SiLU/tanh/softplus >> ReLU)
        act_roll = rng.random()
        if act_roll < 0.30:
            act_fn = lambda h: np.tanh(h)       # smooth, bounded
        elif act_roll < 0.55:
            # GELU approximation: x * sigmoid(1.702 * x)
            act_fn = lambda h: h * (1.0 / (1.0 + np.exp(-1.702 * h)))
        elif act_roll < 0.75:
            # SiLU / Swish: x * sigmoid(x)
            act_fn = lambda h: h * (1.0 / (1.0 + np.exp(-h)))
        elif act_roll < 0.85:
            # Softplus: log(1 + exp(x))
            act_fn = lambda h: np.log1p(np.exp(np.clip(h, -20, 20)))
        else:
            act_fn = lambda h: np.maximum(h, 0)  # ReLU (still useful for some patterns)

        # Multi-layer forward pass
        h = X_active
        in_dim = n_active
        for layer_i in range(n_layers - 1):
            out_dim = hid if layer_i < n_layers - 2 else hid
            W = rng.standard_normal((in_dim, out_dim)) / np.sqrt(in_dim)
            b = rng.standard_normal(out_dim) * 0.1
            h = act_fn(h @ W + b)
            in_dim = out_dim
        # Final projection
        W_out = rng.standard_normal((in_dim, 1)) / np.sqrt(in_dim)
        y = (h @ W_out).ravel()
        prior_meta['active_dims'] = int(n_active)
        prior_meta['hidden_width'] = int(hid)
        prior_meta['n_mlp_layers'] = int(n_layers)

    elif reg_type == 6:
        # Radial / distance prior: smoother and more structured than a plain
        # scalar RBF sum. Project to a low-dimensional metric space, then mix
        # Gaussian, Cauchy, and log-distance kernels.
        n_active = _sample_active_count(3, min(12, n_features), dense_floor=5)
        active = rng.choice(n_features, size=n_active, replace=False)
        X_active = X[:, active]
        metric_dim = min(n_active, int(rng.integers(2, min(6, n_active) + 1)))
        metric_proj = rng.standard_normal((n_active, metric_dim)) / np.sqrt(max(n_active, 1))
        Z = X_active @ metric_proj
        Z = Z - Z.mean(axis=0, keepdims=True)
        Z_std = Z.std(axis=0, keepdims=True)
        Z = Z / np.where(Z_std > 1e-8, Z_std, 1.0)
        n_basis = int(rng.integers(3, min(12, max(4, n_samples // 8 + 1)) + 1))
        if n_samples >= n_basis:
            center_idx = rng.choice(n_samples, size=n_basis, replace=False)
            centers = Z[center_idx]
        else:
            centers = rng.standard_normal((n_basis, metric_dim))
        y = np.zeros(n_samples)
        for j in range(n_basis):
            amp = rng.standard_normal() / np.sqrt(n_basis)
            lengthscales = rng.uniform(0.6, 2.2, size=metric_dim)
            diff = (Z - centers[j]) / lengthscales[np.newaxis, :]
            dist = np.sqrt(np.sum(diff ** 2, axis=1) + 1e-6)
            kernel_roll = rng.random()
            if kernel_roll < 0.55:
                basis = np.exp(-0.5 * dist ** 2)
            elif kernel_roll < 0.85:
                basis = 1.0 / (1.0 + dist ** 2)
            else:
                basis = -np.log(dist + 0.15)
            y += amp * basis
        if rng.random() < 0.5:
            linear = rng.standard_normal(metric_dim) / np.sqrt(metric_dim)
            y += 0.2 * (Z @ linear)
        prior_meta['active_dims'] = int(n_active)
        prior_meta['latent_dim'] = int(metric_dim)
        prior_meta['basis_count'] = int(n_basis)

    elif reg_type == 7:
        # Fourier features: low-dimensional smooth periodic structure with
        # paired sin/cos bases and harmonics.
        n_active = _sample_active_count(3, min(18, n_features), dense_floor=6)
        active = rng.choice(n_features, size=n_active, replace=False)
        X_active = X[:, active]
        latent_dim = min(n_active, int(rng.integers(2, min(6, n_active) + 1)))
        latent_proj = rng.standard_normal((n_active, latent_dim)) / np.sqrt(max(n_active, 1))
        Z = X_active @ latent_proj
        Z = Z - Z.mean(axis=0, keepdims=True)
        Z_std = Z.std(axis=0, keepdims=True)
        Z = Z / np.where(Z_std > 1e-8, Z_std, 1.0)
        n_terms = int(rng.integers(4, min(20, 3 * latent_dim + 6) + 1))
        y = np.zeros(n_samples)
        for _ in range(n_terms):
            omega = rng.standard_normal(latent_dim)
            omega /= (np.linalg.norm(omega) + 1e-8)
            omega *= rng.uniform(0.25, 2.0)
            harmonic = int(rng.integers(1, 4))
            phase = rng.uniform(-np.pi, np.pi)
            proj = harmonic * (Z @ omega) + phase
            a = rng.standard_normal() / np.sqrt(n_terms)
            b = rng.standard_normal() / np.sqrt(n_terms)
            y += a * np.cos(proj) + b * np.sin(proj)
        prior_meta['active_dims'] = int(n_active)
        prior_meta['latent_dim'] = int(latent_dim)
        prior_meta['term_count'] = int(n_terms)

    elif reg_type == 8:
        # Kinematics-like chained trigonometric system. Unlike the original
        # version, each joint angle depends on a mixed projection of many
        # active features rather than a single column.
        n_active = _sample_active_count(6, min(16, n_features), dense_floor=8)
        active = rng.choice(n_features, size=n_active, replace=False)
        X_active = X[:, active]
        # Small-feature episodes can legitimately reach this prior via the shared
        # generator path. In that case, shrink the lower bound instead of asking
        # rng.integers(4, 3), which raises `ValueError: low >= high`.
        joint_hi = min(10, n_active)
        joint_lo = min(4, joint_hi)
        n_joints = int(rng.integers(joint_lo, joint_hi + 1))
        joint_proj = rng.standard_normal((n_active, n_joints)) / np.sqrt(max(n_active, 1))
        joint_inputs = X_active @ joint_proj
        theta = np.zeros(n_samples)
        x_pos = np.zeros(n_samples)
        y_pos = np.zeros(n_samples)
        aux = np.zeros(n_samples)
        for i in range(n_joints):
            freq = rng.uniform(0.6, 2.0)
            phase = rng.uniform(-np.pi, np.pi)
            coupling = rng.uniform(0.05, 0.30)
            theta = theta + freq * joint_inputs[:, i] + phase + coupling * np.sin(
                theta + 0.5 * joint_inputs[:, i]
            )
            link = rng.uniform(0.6, 1.6) / np.sqrt(n_joints)
            x_pos += link * np.cos(theta)
            y_pos += link * np.sin(theta)
            aux += link * np.sin(rng.uniform(0.5, 1.5) * theta + phase)
        coeffs = rng.standard_normal(4)
        y = (
            coeffs[0] * x_pos
            + coeffs[1] * y_pos
            + coeffs[2] * aux
            + coeffs[3] * np.sin(theta)
        )
        prior_meta['active_dims'] = int(n_active)
        prior_meta['joint_count'] = int(n_joints)

    elif reg_type == 9:
        # Full polynomial surface: smooth quadratic response over many features,
        # with an occasional weak cubic component for process-style curvature.
        n_active = _sample_active_count(3, min(14, n_features), dense_floor=6)
        active = rng.choice(n_features, size=n_active, replace=False)
        X_active = X[:, active]
        A = rng.standard_normal((n_active, n_active)) * (0.25 / np.sqrt(max(n_active, 1)))
        M = (A + A.T) / 2
        w = rng.standard_normal(n_active) / np.sqrt(max(n_active, 1))
        b = rng.normal(0.0, 0.1)
        quad = np.sum((X_active @ M) * X_active, axis=1)
        y = quad + X_active @ w + b
        if rng.random() < 0.5:
            u = rng.standard_normal(n_active)
            u /= (np.linalg.norm(u) + 1e-8)
            y += 0.15 * np.clip(X_active @ u, -4, 4) ** 3
        prior_meta['active_dims'] = int(n_active)

    elif reg_type == 10:
        # Latent-factor target: y = sum_j h_j(X @ V_j) where V is d×k random
        # projection and h_j is a smooth nonlinearity. Models datasets where
        # many features share a low-dimensional structure (drug binding
        # fingerprints, molecular embeddings, demographic indices, multi-task
        # representations). Distinct from sparse_linear (which selects a few
        # features); here ALL features contribute through a shared k-dim
        # latent space.
        k = int(rng.integers(2, min(9, max(3, n_features // 4 + 1))))
        V = rng.standard_normal((n_features, k))
        # Normalize each column so per-latent magnitude is comparable.
        col_norm = np.linalg.norm(V, axis=0) + 1e-8
        V = V / col_norm
        Z = X @ V  # [n, k]
        y = np.zeros(n_samples)
        for j in range(k):
            fn_type = int(rng.integers(0, 5))
            z = Z[:, j]
            wj = rng.standard_normal() / max(np.sqrt(k), 1)
            if fn_type == 0:
                y += wj * z
            elif fn_type == 1:
                y += wj * np.tanh(z)
            elif fn_type == 2:
                y += wj * (z ** 2 - 1)
            elif fn_type == 3:
                freq = rng.uniform(0.5, 2.0)
                y += wj * np.sin(freq * z)
            else:
                y += wj * np.sign(z) * np.log1p(np.abs(z))
        prior_meta['active_dims'] = int(n_features)
        prior_meta['latent_dim'] = int(k)

    else:
        # Random tree-like: piecewise-constant target (insurance, pricing, etc.)
        n_active = max(1, min(n_features, int(rng.integers(2, min(n_features + 1, 15)))))
        active = rng.choice(n_features, size=n_active, replace=False)
        y = np.zeros(n_samples)
        n_rules = int(rng.integers(3, 16))  # 3-15 additive threshold rules
        for _ in range(n_rules):
            col = rng.choice(active)
            threshold = rng.standard_normal()
            weight = rng.standard_normal() / np.sqrt(n_rules)
            y += weight * (X[:, col] > threshold).astype(np.float64)
        prior_meta['active_dims'] = int(n_active)
        prior_meta['rule_count'] = int(n_rules)

    # --- Smooth clipping for extreme y values (before adding noise) ---
    # Use tanh soft-clip instead of hard clip to avoid gradient cliffs
    # at boundaries. Scale: clip at ±50 with soft transition zone.
    y_absmax = np.max(np.abs(y))
    if y_absmax > 50:
        y = 50.0 * np.tanh(y / 50.0)

    # --- Add noise for realistic R² ---
    # Zero-noise mode is controlled by reg_deterministic_prob and teaches the
    # model to produce exact predictions on clean datasets (physics,
    # forward kinematics, pure math functions).
    y_signal_std = np.std(y)
    if y_signal_std < 1e-8:
        y_signal_std = 1.0

    # Decide if this is a zero-noise episode.
    _zero_noise_prob = float(np.clip(reg_deterministic_prob, 0.0, 1.0))
    if rng.random() < _zero_noise_prob:
        target_r2 = 1.0  # deterministic — no noise added
    elif reg_dense:
        target_r2 = 0.7 + 0.28 * rng.beta(4.0, 1.5)  # [0.7, 0.98], mean ~0.89
    elif reg_denoise:
        target_r2 = 0.5 + 0.48 * rng.beta(3.0, 1.5)  # [0.5, 0.98], mean ~0.81
    else:
        target_r2 = 0.3 + 0.65 * rng.beta(3.0, 1.5)
    if smooth_joint_prior and target_r2 < 1.0:
        # Smooth multivariate problems are typically cleaner than generic
        # synthetic regression. Keep them moderately high-SNR without making
        # them fully deterministic.
        target_r2 = min(0.985, target_r2 + rng.uniform(0.04, 0.10))
    prior_meta['target_r2'] = float(target_r2)
    prior_meta['deterministic'] = bool(target_r2 >= 1.0)

    if target_r2 < 1.0:
        noise_var = y_signal_std ** 2 * (1 - target_r2) / max(target_r2, 0.01)
        y += rng.standard_normal(n_samples) * np.sqrt(noise_var)

        # --- Heteroskedastic noise ---
        _het_prob = 0.15 if reg_denoise else 0.3
        if smooth_joint_prior:
            _het_prob *= 0.5
        if n_features >= 2 and rng.random() < _het_prob:
            driver_col = int(rng.choice(n_features))
            het_scale = np.abs(X[:, driver_col])
            het_scale /= (np.mean(het_scale) + 1e-8)
            y += rng.standard_normal(n_samples) * het_scale * y_signal_std * 0.3

        # --- Occasional y outliers ---
        _outlier_prob = 0.15 if reg_denoise else 0.3
        if smooth_joint_prior:
            _outlier_prob *= 0.5
        if rng.random() < _outlier_prob:
            n_outliers = max(1, int(n_samples * rng.uniform(0.01, 0.05)))
            outlier_idx = rng.choice(n_samples, size=n_outliers, replace=False)
            y[outlier_idx] += rng.standard_normal(n_outliers) * y_signal_std * rng.uniform(3, 8)

    # --- Discretize some features (30%) ---
    _disc_prob = 0.15 if smooth_joint_prior else 0.3
    if not use_scm_features and rng.random() < _disc_prob and n_features >= 2:
        n_disc = max(1, int(rng.integers(1, max(2, n_features // 3 + 1))))
        disc_cols = rng.choice(n_features, size=n_disc, replace=False)
        for col in disc_cols:
            n_bins = int(rng.integers(3, 15))
            col_vals = X[:, col]
            edges = np.quantile(col_vals, np.linspace(0, 1, n_bins + 1))
            X[:, col] = np.digitize(col_vals, edges[1:-1]).astype(np.float64)

    # --- Replace some features with noise ---
    if smooth_joint_prior:
        _noise_prob = 0.04 if reg_dense else 0.08
    else:
        _noise_prob = 0.08 if reg_dense else 0.2
    if not use_scm_features and rng.random() < _noise_prob and n_features >= 4:
        n_noise = max(1, int(rng.integers(1, max(2, n_features // 4 + 1))))
        noise_cols = rng.choice(n_features, size=n_noise, replace=False)
        for col in noise_cols:
            X[:, col] = rng.standard_normal(n_samples)

    # --- Convert and safety ---
    y = y.astype(np.float32)

    # Guard against nan/inf in y (e.g. from extreme outlier injection or
    # float32 overflow). Replace with 0 so normalization stays finite.
    if not np.all(np.isfinite(y)):
        y = np.where(np.isfinite(y), y, 0.0)

    if use_scm_features:
        X_out = X_model
    else:
        X = X.astype(np.float32)
        # Winsorize X per-column
        for col in range(n_features):
            v = X[:, col]
            finite = v[np.isfinite(v)]
            if len(finite) < 5:
                continue
            med = np.median(finite)
            mad = np.median(np.abs(finite - med)) * 1.4826
            if mad < 1e-8:
                mad = max(np.std(finite), 1e-8)
            lo, hi = med - 6 * mad, med + 6 * mad
            X[:, col] = np.clip(v, lo, hi)
        X = np.clip(X, -1e4, 1e4)
        X = np.where(np.isfinite(X), X, 0.0)
        X_out = X

    # Global normalization stabilizes the generator output scale, preventing
    # extreme y values from blowing up the loss before the trainer's context-only
    # normalization gets to it.
    y = y.astype(np.float32)
    if not np.all(np.isfinite(y)):
        y = np.where(np.isfinite(y), y, 0.0)
    y_mean = np.mean(y)
    y_std = np.std(y)
    if y_std > 1e-8:
        y = (y - y_mean) / y_std
    else:
        y = y - y_mean

    return {
        'X': X_out,
        'y': y,
        'task_type': 'reg',
        'n_classes': None,
        'filtered': False,
        'meta': prior_meta,
    }


def generate_dataset(n_samples, n_features, task_type, n_classes=None,
                     rng=None, augment=False, augment_v3=False,
                     rich_reg_targets=True, scale_variation=True,
                     augment_v4=False, v4_filter=True, v4_no_edge_noise=True,
                     synth_v5=False, synth_v5_denoise=True,
                     synth_v5_declone=True, synth_v5_mixture=False,
                     reg_denoise=False, reg_dense=False,
                     probabilistic_labels=False, nominal_categoricals=False,
                     enhanced_missingness=False):
    """Generate a synthetic dataset using hierarchical SCM.

    Args:
        n_samples: number of samples to generate
        n_features: number of features (will be the actual feature count)
        task_type: 'cls' or 'reg'
        n_classes: number of classes for classification (ignored for regression)
        rng: numpy random generator
        augment: if True, apply synth_v2 augmentations (discretization, noise
                 features, extreme imbalance, label noise, missingness,
                 correlated blocks)
        augment_v3: if True, apply synth_v3 augmentations (true categoricals,
                    feature interactions, skewed regression targets)
        rich_reg_targets: if True AND augment_v3, add multi-feature deps +
                          interaction terms to regression targets
        scale_variation: if True AND augment_v3, apply random target scale
        augment_v4: if True, apply synth_v4 improvements (expanded activations,
                    quadratic/product edge fns, richer aggregation, feature
                    importances, feature rescaling, Kumaraswamy warping)
        v4_filter: if True AND augment_v4, apply ExtraTrees filtering
        v4_no_edge_noise: if True AND augment_v4, skip Gaussian edge noise
        synth_v5: if True, apply synth_v5 SCM improvements (informative
                  categoricals, concat-then-transform aggregation,
                  multi-dimensional SCM nodes)
        synth_v5_denoise: if True AND synth_v5, reduce noise-feature replacement
                  and Gaussian noise scale (v5 learnability fix)
        synth_v5_declone: if True AND synth_v5, use per-dim scaling + signal-relative
                  noise in multi-dim expansion (fixes max_abs_corr)
        synth_v5_mixture: if True, randomly select per-dataset mode from a
                  mixture of v4/v5a/v5c-like settings. Overrides synth_v5,
                  synth_v5_denoise, synth_v5_declone.
        reg_denoise: if True AND task_type=='reg', reduce noise levels:
                  noise features 0-50%→0-20%, structural missingness 30%→15%,
                  Gaussian noise scale capped at 0.15, discretization 0-80%→0-40%,
                  target transforms 40%→20%

    Returns:
        dict with keys:
            'X': np.ndarray [n_samples, n_features]
            'y': np.ndarray [n_samples]
            'task_type': str
    """
    if rng is None:
        rng = np.random.default_rng()
    meta = {
        'generator_family': 'scm',
        'task_type': task_type,
        'repeated_entity_cols': [],
        'repeated_entity_stats': [],
        'entity_lookup_signal_std': 0.0,
        'entity_lookup_target_ratio': 0.0,
    }

    # Mixture prior: randomly select v4/v5a/v5c-like mode per dataset
    if synth_v5_mixture:
        mode_roll = rng.random()
        if mode_roll < 0.25:
            # v4-ish: importance + missingness heavy, no multi-dim
            synth_v5 = False
            synth_v5_denoise = False
            synth_v5_declone = False
        elif mode_roll < 0.60:
            # v5c-ish: full structural + declone + denoise
            synth_v5 = True
            synth_v5_denoise = True
            synth_v5_declone = True
        else:
            # v5a-ish: structural v5, no denoise/declone
            synth_v5 = True
            synth_v5_denoise = False
            synth_v5_declone = False

    # Build SCM (expanded edge fns/aggregation if synth_v4)
    _max_parents = 7 if (reg_dense and task_type == 'reg') else 4
    lcs_list, global_nodes, global_parents = _build_hierarchical_scm(
        n_features, rng, expanded=augment_v4, synth_v5=synth_v5,
        synth_v5_declone=synth_v5_declone, max_parents=_max_parents)

    # Track causal column indices (synth_v5: protected from noise overwrite)
    _causal_col_indices = []

    # Generate all node values
    all_values = {}
    for lcs, offset in lcs_list:
        external_values = all_values.copy()
        node_values = lcs.generate(n_samples, external_values)
        for j, vals in enumerate(node_values):
            # Safety: clamp and sanitize. Node outputs are already
            # standardized inside generate(), so values should be O(1).
            # The ±100 clamp is a defense-in-depth backstop.
            vals = np.clip(vals, -100, 100)
            vals = np.where(np.isfinite(vals), vals, 0.0)
            all_values[offset + j] = vals

    # --- 1. Assemble X from SCM nodes ---
    # When synth_v5=True, nodes can be multi-dimensional [n_samples, d].
    # Feature extraction picks random column subsets from each node;
    # unobserved columns create natural Bayes error.
    total_nodes = len(all_values)
    all_indices = sorted(all_values.keys())

    if synth_v5:
        # Multi-dim feature extraction with target-aware causal coverage.
        # Ensures ancestors of the target node are observed, then fills
        # remaining slots randomly. Unobserved dims create Bayes error.
        target_idx = all_indices[-1]

        # Compute ancestors of target (BFS backward through global_parents)
        ancestors = set()
        frontier = list(global_parents.get(target_idx, []))
        while frontier:
            node = frontier.pop()
            if node not in ancestors and node != target_idx:
                ancestors.add(node)
                frontier.extend(global_parents.get(node, []))

        # Priority order: direct parents first, then other ancestors,
        # then remaining non-ancestor nodes
        direct_parents = set(global_parents.get(target_idx, []))
        other_ancestors = ancestors - direct_parents
        non_ancestors = [i for i in all_indices
                         if i != target_idx and i not in ancestors]
        rng.shuffle(non_ancestors)

        priority_order = (sorted(direct_parents)
                          + sorted(other_ancestors)
                          + list(non_ancestors))

        # Extract feature columns, respecting priority
        feature_cols = []
        # Track which node indices contribute features (for noise protection)
        _causal_col_indices = []  # column indices that are causal ancestors
        n_cols_so_far = 0
        for idx in priority_order:
            if n_cols_so_far >= n_features:
                break
            v = all_values[idx]
            is_causal = idx in ancestors
            if v.ndim == 1:
                feature_cols.append(v.reshape(-1, 1))
                if is_causal:
                    _causal_col_indices.append(n_cols_so_far)
                n_cols_so_far += 1
            else:
                d = v.shape[1]
                if d > 1:
                    # Observe at least 75% of dims — keeps some hidden for
                    # natural Bayes error but avoids making tasks too hard
                    min_observe = max(1, int(d * 0.75))
                    n_observe = int(rng.integers(min_observe, d + 1))
                    observed_dims = rng.choice(d, size=n_observe, replace=False)
                    chunk = v[:, observed_dims]
                else:
                    chunk = v
                    n_observe = 1
                feature_cols.append(chunk)
                if is_causal:
                    for k in range(n_observe):
                        _causal_col_indices.append(n_cols_so_far + k)
                n_cols_so_far += n_observe

        X = np.column_stack(feature_cols) if feature_cols else \
            rng.standard_normal((n_samples, 1))

        # Trim or pad to exactly n_features
        if X.shape[1] > n_features:
            # Keep causal columns, randomly trim others
            n_keep = n_features
            causal_set = set(c for c in _causal_col_indices if c < X.shape[1])
            other_cols = [c for c in range(X.shape[1]) if c not in causal_set]
            # Keep all causal cols that fit, fill rest with random others
            keep = sorted(causal_set)[:n_keep]
            remaining_budget = n_keep - len(keep)
            if remaining_budget > 0 and other_cols:
                keep.extend(rng.choice(
                    other_cols,
                    size=min(remaining_budget, len(other_cols)),
                    replace=False).tolist())
            keep = sorted(keep)[:n_features]
            X = X[:, keep]
            # Remap causal indices
            keep_set = set(keep)
            _causal_col_indices = [i for i, k in enumerate(keep)
                                   if k in causal_set]
        elif X.shape[1] < n_features:
            extra = n_features - X.shape[1]
            X = np.column_stack([X, rng.standard_normal((n_samples, extra))])

        # Target: reduce multi-dim node to scalar
        target_val = all_values[target_idx]
        if target_val.ndim > 1:
            w = rng.standard_normal(target_val.shape[1])
            w /= (np.linalg.norm(w) + 1e-8)
            y_raw = target_val @ w
        else:
            y_raw = target_val
    else:
        # Original scalar feature extraction
        if total_nodes > n_features + 1:
            chosen = rng.choice(all_indices, size=n_features + 1, replace=False)
            chosen = sorted(chosen)
            target_idx = chosen[-1]
            feature_indices = chosen[:-1]
        else:
            target_idx = all_indices[-1]
            feature_indices = all_indices[:n_features]

        X = np.column_stack([all_values[i] for i in feature_indices])

        # Pad if we don't have enough features
        if X.shape[1] < n_features:
            extra = n_features - X.shape[1]
            X = np.column_stack([X, rng.standard_normal((n_samples, extra))])

        y_raw = all_values[target_idx]

    # Safety: ensure y_raw is 1D
    if y_raw.ndim > 1:
        w = rng.standard_normal(y_raw.shape[1])
        w /= (np.linalg.norm(w) + 1e-8)
        y_raw = y_raw @ w
    y_raw = y_raw.ravel()
    entity_lookup_signal = np.zeros(n_samples, dtype=np.float64)

    # --- Track discrete/categorical columns for protection ---
    # Columns marked discrete will be excluded from continuous transforms
    # (Kumaraswamy warping, feature rescaling, heavy-tail injection) and
    # snapped back to integers before final output.
    discrete_cols = set()
    cat_cardinality = {}  # col -> K for true categoricals

    # --- 2. Replace noise features (Change 2) ---
    # CLS v4 gets more noise features to dilute MI (real CLS MI median=0.021)
    # synth_v5+denoise: reduced range — multi-dim hidden columns already suppress learnability
    if augment and n_features >= 2:
        if augment_v4 and task_type == 'cls':
            if synth_v5 and synth_v5_denoise:
                noise_frac = rng.uniform(0.05, 0.25)
            else:
                noise_frac = rng.uniform(0.15, 0.55)
        else:
            if reg_dense and task_type == 'reg':
                noise_frac = rng.uniform(0, 0.15)
            elif reg_denoise and task_type == 'reg':
                noise_frac = rng.uniform(0, 0.2)
            else:
                noise_frac = rng.uniform(0, 0.5)
        n_noise = int(n_features * noise_frac)
        if n_noise > 0:
            if synth_v5 and _causal_col_indices:
                # Only replace non-causal columns with noise
                causal_set = set(_causal_col_indices)
                eligible = [c for c in range(n_features) if c not in causal_set]
                if eligible:
                    n_noise = min(n_noise, len(eligible))
                    noise_cols = rng.choice(eligible, size=n_noise, replace=False)
                else:
                    noise_cols = []
            else:
                noise_cols = rng.choice(n_features, size=n_noise, replace=False)
            for col in noise_cols:
                X[:, col] = _sample_root(n_samples, rng)

    # --- 3. Correlated feature blocks (Change 7) ---
    # Generate correlated copies with explicit target correlation rho.
    # Old approach (copy + tiny noise) created near-duplicates that spiked
    # max_abs_corr. rho-mixing gives controllable correlation.
    if augment and n_features >= 4 and rng.random() < 0.25:
        n_anchors = int(rng.integers(1, min(4, n_features // 2) + 1))
        anchor_cols = rng.choice(n_features, size=n_anchors, replace=False)
        available = [c for c in range(n_features) if c not in anchor_cols]
        rng.shuffle(available)
        idx = 0
        for anchor in anchor_cols:
            if idx >= len(available):
                break
            n_copies = int(rng.integers(1, min(5, len(available) - idx) + 1))
            for _ in range(n_copies):
                if idx >= len(available):
                    break
                col = available[idx]
                # Target correlation: 0.4-0.95 (cap avoids near-duplicates)
                rho = rng.uniform(0.4, 0.95)
                a = X[:, anchor].copy()
                a_std = np.std(a)
                a_mean = np.mean(a)
                if a_std > 1e-8:
                    a_normed = (a - a_mean) / a_std
                else:
                    a_normed = a - a_mean
                eps = rng.standard_normal(n_samples)
                x_new = rho * a_normed + np.sqrt(max(1 - rho**2, 1e-6)) * eps
                # Re-scale to anchor-like scale
                X[:, col] = x_new * (a_std + 1e-6) + a_mean
                idx += 1

    # --- 3.5 Feature interaction terms [synth_v3] ---
    if augment_v3 and n_features >= 4 and rng.random() < 0.3:
        n_interactions = int(rng.integers(1, min(6, n_features // 2) + 1))
        for _ in range(n_interactions):
            i, j = rng.choice(n_features, size=2, replace=False)
            target_col = int(rng.integers(0, n_features))
            interaction_type = rng.integers(0, 3)
            if interaction_type == 0:
                # Product: x_i * x_j (common in revenue = price * qty)
                X[:, target_col] = X[:, i] * X[:, j]
            elif interaction_type == 1:
                # Ratio: x_i / (|x_j| + 1) (common in rates, per-capita)
                X[:, target_col] = X[:, i] / (np.abs(X[:, j]) + 1)
            else:
                # Absolute difference: |x_i - x_j| (common in distances)
                X[:, target_col] = np.abs(X[:, i] - X[:, j])

    # --- 3.7 Feature importances [synth_v4] ---
    # CLS uses milder importances to avoid MI overshoot (real CLS MI median=0.021)
    # reg_dense: skip power-law importances 60% of the time (flat importance profile)
    _imp_prob = 0.35 if (reg_dense and task_type == 'reg') else 0.85
    if augment_v4 and n_features >= 2 and rng.random() < _imp_prob:
        X = _apply_feature_importances(X, rng, mild=(task_type == 'cls'))

    # --- 4. Add Gaussian noise to features (existing) ---
    # synth_v4: reduced noise (TabICLv2 found no benefit from full noise,
    # but removing it entirely makes MI too high vs real data)
    # CLS gets more noise since real CLS has very low MI (median 0.021)
    if augment_v4 and v4_no_edge_noise:
        if task_type == 'cls':
            # synth_v5+denoise: less noise needed (multi-dim hidden dims already suppress MI)
            if synth_v5 and synth_v5_denoise:
                noise_scale = rng.uniform(0.05, 0.25)
            else:
                noise_scale = rng.uniform(0.10, 0.40)
        else:
            if synth_v5 and synth_v5_denoise:
                noise_scale = rng.uniform(0.01, 0.10)
            else:
                noise_scale = rng.uniform(0.02, 0.15)
    else:
        if reg_dense and task_type == 'reg':
            noise_scale = rng.uniform(0.005, 0.08)
        elif reg_denoise and task_type == 'reg':
            noise_scale = rng.uniform(0.01, 0.15)
        else:
            noise_scale = rng.uniform(0.01, 0.3)
    for col in range(X.shape[1]):
        col_std = np.std(X[:, col])
        if col_std > 0:
            X[:, col] += rng.normal(0, col_std * noise_scale, n_samples)

    # --- 5. Discretize random features (Change 1) ---
    if augment and n_features >= 2:
        # Per-episode "binary fingerprint" mode: a fraction of episodes get
        # mostly-binary columns with varying per-column density. Models real
        # molecular fingerprints (QSAR-TID-11: 1024 binary bits), one-hot
        # indicator stacks, and similar binary-bag feature regimes that the
        # default 2-20 bucket discretization rarely produces (its uniform
        # bin-count sampling makes strict 2-bucket cols only ~5% of all
        # discretizations). This is a single-line behavior toggle inside the
        # existing flag — no new config knob.
        force_binary_episode = rng.random() < 0.25

        # synth_v4: higher discretization to match real data (~70%+ discrete)
        if force_binary_episode:
            disc_frac = float(rng.uniform(0.7, 0.98))
        elif augment_v4:
            disc_frac = rng.uniform(0.4, 0.98)
        else:
            if reg_denoise and task_type == 'reg':
                disc_frac = rng.uniform(0, 0.4)
            else:
                disc_frac = rng.uniform(0, 0.8)
        n_disc = int(n_features * disc_frac)
        if n_disc > 0:
            disc_cols = rng.choice(n_features, size=n_disc, replace=False)
            for col in disc_cols:
                if force_binary_episode:
                    # Strict 2-bucket binarization with random per-column
                    # density. density ∈ [0.10, 0.60] mimics fingerprint bit
                    # frequencies — some bits common (in many molecules),
                    # some rare. Threshold at the (1-density) quantile so
                    # `density` fraction of rows are 1s.
                    density = float(rng.uniform(0.10, 0.60))
                    threshold = float(np.nanquantile(X[:, col], 1.0 - density))
                    X[:, col] = (X[:, col] > threshold).astype(np.float64)
                elif rng.random() < 0.5:
                    # Ordinal binning: quantile-based, 2-20 bins
                    n_bins = int(rng.integers(2, 21))
                    percentiles = np.linspace(0, 100, n_bins + 1)[1:-1]
                    thresholds = np.nanpercentile(X[:, col], percentiles)
                    # Remove duplicate thresholds
                    thresholds = np.unique(thresholds)
                    X[:, col] = np.digitize(X[:, col], thresholds).astype(
                        np.float64)
                else:
                    # Round to 0-2 decimal places, then rank to integer
                    decimals = int(rng.integers(0, 3))
                    rounded = np.round(X[:, col], decimals)
                    # Map unique values to integer ranks
                    unique_vals = np.unique(rounded)
                    rank_map = {v: i for i, v in enumerate(unique_vals)}
                    X[:, col] = np.array(
                        [rank_map[v] for v in rounded], dtype=np.float64)
                discrete_cols.add(int(col))

    # --- 5.5 True categorical features [synth_v3] ---
    # Unlike discretization (which bins continuous values preserving order),
    # these are inherently unordered integers with random cardinality.
    # synth_v4: higher probability (50%) to match real data's categorical dominance
    # synth_v5: 70% informative (derived from continuous SCM values), 30% random
    # v3 repeated-entity path: skewed high-cardinality IDs with shared lookup
    # effects, meant to cover Amazon-like repeated-entity tasks.
    cat_prob = 0.5 if augment_v4 else 0.3
    if augment_v3 and n_features >= 2 and rng.random() < cat_prob:
        cat_frac = rng.uniform(0.1, 0.5)
        n_cat = max(1, int(n_features * cat_frac))
        cat_cols = rng.choice(n_features, size=n_cat, replace=False)
        cat_cols_set = set(int(c) for c in cat_cols)
        for col in cat_cols:
            # --- Nominal categorical (non-ordinal, random effects) ---
            if nominal_categoricals and rng.random() < 0.4:
                cardinality = int(rng.integers(3, min(31, max(4, n_samples // 8))))
                row_effect, n_used = _create_nominal_categorical(
                    X, col, n_samples, rng, cardinality,
                    task_type, n_features, discrete_cols, cat_cols_set)
                entity_lookup_signal += row_effect
                discrete_cols.add(int(col))
                cat_cardinality[int(col)] = cardinality
                if n_used == 2:
                    # Crossed nominal consumed a second column
                    meta.setdefault('nominal_crossed_cols', []).append(int(col))
                else:
                    meta.setdefault('nominal_independent_cols', []).append(int(col))
                continue

            entity_prob = 0.55 if (n_samples >= 1024 and n_features <= 64) else 0.35
            use_repeated_entity = (
                task_type == 'cls'
                and n_samples >= 64
                and rng.random() < entity_prob
            )
            if use_repeated_entity:
                row_effect, entity_meta = _create_repeated_entity_categorical(
                    X, col, rng
                )
                entity_lookup_signal += row_effect
                cardinality = int(entity_meta['cardinality'])
                meta['repeated_entity_cols'].append(int(col))
                meta['repeated_entity_stats'].append(entity_meta)
            else:
                cardinality = int(rng.integers(2, 31))
                if synth_v5 and rng.random() < 0.7:
                    # Informative categorical: derived from continuous SCM value
                    _create_informative_categorical(X, col, rng, cardinality)
                else:
                    # Random noise categorical (original behavior)
                    X[:, col] = rng.integers(0, cardinality,
                                             size=n_samples).astype(np.float64)
            discrete_cols.add(int(col))
            cat_cardinality[int(col)] = cardinality

    # --- 5.7 Kumaraswamy warping [synth_v4] ---
    # Only warp continuous columns — discrete cols must stay integer-valued
    if augment_v4 and n_features >= 2 and rng.random() < 0.3:
        continuous_cols = [c for c in range(n_features) if c not in discrete_cols]
        if continuous_cols:
            warp_frac = rng.uniform(0.05, 0.3)
            n_warp = max(1, int(len(continuous_cols) * warp_frac))
            warp_cols = rng.choice(continuous_cols, size=min(n_warp, len(continuous_cols)),
                                   replace=False)
            for col in warp_cols:
                if not np.any(np.isnan(X[:, col])):
                    _apply_kumaraswamy_warping(X, col, rng)

    # --- 5.8 Random feature rescaling [synth_v4] ---
    # Only rescale continuous columns — discrete cols stay integer-valued
    if augment_v4 and n_features >= 2 and rng.random() < 0.5:
        continuous_cols = [c for c in range(n_features) if c not in discrete_cols]
        if continuous_cols:
            cont_idx = np.array(continuous_cols)
            scales = np.exp(rng.uniform(np.log(0.1), np.log(10.0), len(cont_idx)))
            X[:, cont_idx] = X[:, cont_idx] * scales[np.newaxis, :]

    # --- 5.9 Heavy-tail outlier injection [synth_v4] ---
    # Real data has median feature kurtosis ~82.7, synthetic ~0.
    # This is the #1 discriminator feature across all variants.
    # synth_v5: target non-causal columns to boost kurtosis without hurting learnability
    # Only inject into continuous columns — discrete cols get kurtosis from spike distributions
    if augment_v4 and n_features >= 2 and rng.random() < 0.6:
        _ht_causal = _causal_col_indices if synth_v5 else None
        X = _inject_heavy_tails(X, rng, causal_cols=_ht_causal,
                                exclude_cols=discrete_cols)

    # Regression targets should be generated from the clean feature values
    # before structural missingness is applied to the observed table.
    X_target = X.copy() if task_type == 'reg' else None

    # --- 6. Structural missingness (Change 5) ---
    _miss_prob = 0.15 if (reg_denoise and task_type == 'reg') else 0.3
    if augment and n_features >= 2 and rng.random() < _miss_prob:
        # Enhanced missingness: row-level and block patterns applied first
        # (these affect multiple columns at once, not per-column)
        _did_enhanced = False
        if enhanced_missingness and rng.random() < 0.5:
            _enh_type = int(rng.integers(0, 4))
            if _enh_type == 0:
                # Row-level dropout: 5-20% of rows lose 50-90% of features
                row_frac = rng.uniform(0.05, 0.20)
                col_frac = rng.uniform(0.50, 0.90)
                n_drop_rows = max(1, int(n_samples * row_frac))
                n_drop_cols = max(1, int(n_features * col_frac))
                drop_rows = rng.choice(n_samples, size=n_drop_rows, replace=False)
                for row in drop_rows:
                    drop_cols_r = rng.choice(n_features, size=n_drop_cols,
                                             replace=False)
                    X[row, drop_cols_r] = np.nan
                _did_enhanced = True
            elif _enh_type == 1:
                # Block missingness: contiguous block of 2-5 cols × 10-30% rows
                block_n_cols = min(int(rng.integers(2, 6)), n_features)
                block_start = int(rng.integers(0, max(1, n_features - block_n_cols + 1)))
                block_cols = list(range(block_start, block_start + block_n_cols))
                row_frac = rng.uniform(0.10, 0.30)
                n_block_rows = max(1, int(n_samples * row_frac))
                block_rows = rng.choice(n_samples, size=n_block_rows, replace=False)
                X[np.ix_(block_rows, block_cols)] = np.nan
                _did_enhanced = True
            elif _enh_type == 2:
                # Target-dependent: features missing when y_raw is extreme
                # (outcome-dependent censoring)
                n_td_cols = min(int(rng.integers(1, 4)), n_features)
                td_cols = rng.choice(n_features, size=n_td_cols, replace=False)
                for col in td_cols:
                    if rng.random() < 0.5:
                        pct = rng.uniform(75, 95)
                        threshold = np.nanpercentile(y_raw, pct)
                        X[y_raw > threshold, col] = np.nan
                    else:
                        pct = rng.uniform(5, 25)
                        threshold = np.nanpercentile(y_raw, pct)
                        X[y_raw < threshold, col] = np.nan
                _did_enhanced = True
            elif _enh_type == 3 and discrete_cols:
                # Categorical missing-as-category: replace NaN with new level
                # for discrete columns. The model learns that "missing" is a
                # meaningful category, not just absence of data.
                mac_cols = [c for c in discrete_cols
                            if c in cat_cardinality]
                if mac_cols:
                    n_mac = min(len(mac_cols), int(rng.integers(1, 4)))
                    mac_selected = rng.choice(mac_cols, size=n_mac, replace=False)
                    miss_frac = rng.uniform(0.05, 0.25)
                    for col in mac_selected:
                        n_miss = max(1, int(n_samples * miss_frac))
                        miss_rows = rng.choice(n_samples, size=n_miss, replace=False)
                        new_cat = float(cat_cardinality[int(col)])
                        X[miss_rows, col] = new_cat
                    _did_enhanced = True

        # Per-column missingness (original patterns, also applied when enhanced
        # didn't fire or alongside enhanced patterns)
        if not _did_enhanced or rng.random() < 0.5:
            n_miss_features = int(rng.integers(1, min(6, n_features) + 1))
            miss_cols = rng.choice(n_features, size=n_miss_features, replace=False)
            for col in miss_cols:
                miss_type = rng.integers(0, 3)
                if miss_type == 0:
                    # MNAR: high values missing (above 60-95th percentile)
                    pct = rng.uniform(60, 95)
                    threshold = np.nanpercentile(X[:, col], pct)
                    X[X[:, col] > threshold, col] = np.nan
                elif miss_type == 1:
                    # MNAR: low values missing (below 5-40th percentile)
                    pct = rng.uniform(5, 40)
                    threshold = np.nanpercentile(X[:, col], pct)
                    X[X[:, col] < threshold, col] = np.nan
                else:
                    # MAR: missing depends on another feature
                    other_col = rng.integers(0, n_features)
                    while other_col == col and n_features > 1:
                        other_col = rng.integers(0, n_features)
                    pct = rng.uniform(30, 70)
                    threshold = np.nanpercentile(X[:, other_col], pct)
                    if rng.random() < 0.5:
                        X[X[:, other_col] > threshold, col] = np.nan
                    else:
                        X[X[:, other_col] < threshold, col] = np.nan

    # --- 6.5 Latent Bayes error + hard negatives [synth_v4] ---
    # Skip when synth_v5: multi-dim nodes already provide natural Bayes error
    # via hidden columns. Stacking both causes double-dipping and makes tasks
    # too hard (oob_auc/oob_r2 far below real data).
    if augment_v4 and not synth_v5 and n_features >= 4 and rng.random() < 0.4:
        X, y_raw = _add_latent_bayes_error(X, y_raw, n_features, rng, task_type)

    # --- 7. Create target y ---
    if task_type == 'cls':
        if np.std(entity_lookup_signal) > 1e-8:
            # Make the repeated-entity columns genuinely predictive without
            # letting them swamp the original SCM target.
            entity_std = np.std(entity_lookup_signal)
            y_std_orig = np.std(y_raw)
            if y_std_orig < 1e-8:
                y_std_orig = 1.0
            target_ratio = rng.uniform(0.25, 0.55)
            entity_scale = (
                np.sqrt(target_ratio / max(1 - target_ratio, 1e-6))
                * y_std_orig / entity_std
            )
            y_raw = y_raw + entity_lookup_signal * entity_scale
            y_raw = np.where(np.isfinite(y_raw), y_raw, 0.0)
            meta['entity_lookup_signal_std'] = float(entity_std)
            meta['entity_lookup_target_ratio'] = float(target_ratio)
        if n_classes is None:
            n_classes = int(rng.integers(2, 11))

        if augment:
            imbalance_roll = rng.random()
            if augment_v4:
                # synth_v4: rebalanced to match real data (median ratio ~4)
                # 25% balanced, 55% moderate Dirichlet (milder alpha), 20% extreme
                if imbalance_roll < 0.25:
                    percentiles = np.linspace(0, 100, n_classes + 1)
                elif imbalance_roll < 0.80:
                    # Milder Dirichlet: alpha 0.5-3.0 → median ratios ~2-10
                    alpha = rng.uniform(0.5, 3.0)
                    proportions = rng.dirichlet(np.full(n_classes, alpha))
                    cumulative = np.cumsum(proportions)
                    percentiles = np.concatenate(
                        [[0.0], cumulative[:-1] * 100, [100.0]])
                else:
                    # Extreme: one dominant class at 80-95% (toned down)
                    dominant = int(rng.integers(0, n_classes))
                    dominant_pct = rng.uniform(0.80, 0.95)
                    remaining = 1.0 - dominant_pct
                    proportions = np.full(n_classes, remaining / max(n_classes - 1, 1))
                    proportions[dominant] = dominant_pct
                    cumulative = np.cumsum(proportions)
                    percentiles = np.concatenate(
                        [[0.0], cumulative[:-1] * 100, [100.0]])
            else:
                # 30% balanced, 30% moderate Dirichlet, 40% extreme imbalance
                if imbalance_roll < 0.3:
                    percentiles = np.linspace(0, 100, n_classes + 1)
                elif imbalance_roll < 0.6:
                    # Moderate: Dirichlet-sampled class proportions
                    alpha = rng.uniform(0.3, 1.0)
                    proportions = rng.dirichlet(np.full(n_classes, alpha))
                    cumulative = np.cumsum(proportions)
                    percentiles = np.concatenate(
                        [[0.0], cumulative[:-1] * 100, [100.0]])
                else:
                    # Extreme: one dominant class at 85-99%
                    dominant = int(rng.integers(0, n_classes))
                    dominant_pct = rng.uniform(0.85, 0.99)
                    remaining = 1.0 - dominant_pct
                    proportions = np.full(n_classes, remaining / max(n_classes - 1, 1))
                    proportions[dominant] = dominant_pct
                    cumulative = np.cumsum(proportions)
                    percentiles = np.concatenate(
                        [[0.0], cumulative[:-1] * 100, [100.0]])
        else:
            # Original: 50% balanced, 50% moderate Dirichlet
            if rng.random() < 0.5:
                percentiles = np.linspace(0, 100, n_classes + 1)
            else:
                alpha = rng.uniform(0.3, 1.0)
                proportions = rng.dirichlet(np.full(n_classes, alpha))
                cumulative = np.cumsum(proportions)
                percentiles = np.concatenate(
                    [[0.0], cumulative[:-1] * 100, [100.0]])

        # --- Probabilistic labelers: non-percentile classification targets ---
        # When enabled, ~50% of episodes use a feature-based labeler instead
        # of quantile bucketing. This removes a known memorization fingerprint.
        if probabilistic_labels and rng.random() < 0.5:
            y = _probabilistic_label(y_raw, X, n_classes, rng)
        else:
            thresholds = np.nanpercentile(y_raw, percentiles[1:-1])
            y = np.digitize(y_raw, thresholds).astype(np.float32)

        # Label noise (Change 4): random label flips
        # synth_v4: 50% of cls datasets get noise (real CLS has very low MI)
        # synth_v5: 50% chance of heteroskedastic noise (feature-dependent flips)
        label_noise_prob = 0.5 if augment_v4 else 0.3
        if augment and rng.random() < label_noise_prob:
            if synth_v5 and rng.random() < 0.5:
                # Heteroskedastic: flip rate depends on feature values
                y = _apply_heteroskedastic_label_noise(y, X, n_classes, rng)
            else:
                # Uniform random flips (original)
                flip_rate = rng.uniform(0.01, 0.15)
                n_flip = int(n_samples * flip_rate)
                if n_flip > 0:
                    flip_idx = rng.choice(n_samples, size=n_flip, replace=False)
                    y[flip_idx] = rng.integers(0, n_classes, size=n_flip).astype(
                        np.float32)

    else:
        # --- Rich regression target [synth_v3]: more feature dependencies ---
        # SCM targets depend on only 1-3 parent features. Real regression
        # datasets have targets that depend on many features through complex
        # interactions. Add direct dependencies and interaction terms to y.
        target_features = X_target if X_target is not None else X
        if augment_v3 and rich_reg_targets and n_features >= 4:
            # Accumulate extra terms separately so we can control their
            # variance contribution relative to the original SCM target.
            extra = np.zeros(n_samples, dtype=np.float64)
            n_extra = int(rng.integers(2, min(10, n_features // 2) + 1))
            dep_cols = rng.choice(n_features, size=n_extra, replace=False)
            for col in dep_cols:
                edge_fn = _make_edge_fn(1, rng)
                weight = rng.normal(0, 1.0)
                extra = extra + weight * edge_fn(target_features[:, col:col+1]).ravel()

            # Add feature interaction terms to target (x_i * x_j, ratios, etc.)
            if rng.random() < 0.5:
                n_interactions = int(rng.integers(1, min(5, n_features // 2) + 1))
                for _ in range(n_interactions):
                    i, j = rng.choice(n_features, size=2, replace=False)
                    interaction_type = rng.integers(0, 3)
                    weight = rng.normal(0, 0.5)
                    if interaction_type == 0:
                        extra = extra + weight * target_features[:, i] * target_features[:, j]
                    elif interaction_type == 1:
                        extra = extra + weight * target_features[:, i] / (np.abs(target_features[:, j]) + 1)
                    else:
                        extra = extra + weight * np.abs(target_features[:, i] - target_features[:, j])

            # Scale extra terms to contribute 30-70% of total y variance.
            # Without this, n_extra terms with weight~N(0,1) can have
            # variance ~50x the original SCM target, drowning it out.
            extra_std = np.std(extra)
            y_std_orig = np.std(y_raw)
            if extra_std > 1e-8 and y_std_orig > 1e-8:
                target_ratio = rng.uniform(0.3, 0.7)
                extra_scale = np.sqrt(target_ratio / (1 - target_ratio)) * y_std_orig / extra_std
                extra = extra * extra_scale
            y_raw = y_raw + extra

            # Safety clip after adding dependencies
            y_raw = np.clip(y_raw, -1e6, 1e6)
            y_raw = np.where(np.isfinite(y_raw), y_raw, 0.0)

        # Normalize regression targets to zero-mean, unit-variance
        y = y_raw.astype(np.float32)
        y_mean = np.mean(y)
        y_std = np.std(y)
        if y_std > 1e-8:
            y = (y - y_mean) / y_std
        else:
            y = y - y_mean

        # --- Target transform for regression [synth_v3] ---
        # Distort target distribution to non-Gaussian shapes (log-normal,
        # heavy-tailed, compressed) then re-normalize. Teaches the model
        # that real regression targets are rarely Gaussian.
        _transform_prob = 0.2 if reg_denoise else 0.4
        if augment_v3 and rng.random() < _transform_prob:
            transform_type = rng.integers(0, 3)
            if transform_type == 0:
                # Exponential: log-normal-like (prices, salaries)
                scale = rng.uniform(0.3, 0.8)
                y = np.sign(y) * np.expm1(np.abs(y) * scale)
            elif transform_type == 1:
                # Power: heavy-tailed (counts, sizes)
                power = rng.uniform(1.5, 3.0)
                y = np.sign(y) * (np.abs(y) ** power)
            else:
                # Sqrt: compressed (rates, probabilities)
                y = np.sign(y) * np.sqrt(np.abs(y))
            # Re-normalize after transform
            y = y.astype(np.float32)
            y_mean = np.mean(y)
            y_std = np.std(y)
            if y_std > 1e-8:
                y = (y - y_mean) / y_std

        # --- Target scale variation [synth_v3] ---
        # During inference, y_train is raw (unnormalized). Training always
        # with N(0,1) means the model never learns to handle different
        # scales. Moderate variation teaches in-context scale adaptation.
        if augment_v3 and scale_variation and rng.random() < 0.3:
            log_scale = rng.uniform(-1, 1)  # ~0.37x to ~2.72x
            y = y * float(np.exp(log_scale))

    # --- 7.5 Constant-column repair ---
    # Detect and replace constant/near-constant columns with noise.
    # These arise from saturating activations (Heaviside/sign/clip/round/mod),
    # aggressive discretization, and chained transforms. They're an easy
    # fingerprint (real mean=0.28, synth_v4 mean=1.74).
    if augment_v4 and n_features >= 2:
        for col in range(n_features):
            col_vals = X[:, col]
            non_nan = col_vals[np.isfinite(col_vals)]
            if len(non_nan) < 2:
                continue
            if np.std(non_nan) < 1e-8 or len(np.unique(non_nan)) <= 1:
                # Add tiny perturbation instead of replacing with pure noise
                # (pure noise breaks the causal chain and injects uninformative features)
                X[:, col] = non_nan[0] + rng.standard_normal(n_samples).astype(X.dtype) * 1e-3

    # --- 7.9 Snap discrete columns back to integers ---
    # Gaussian noise, feature interactions, and other continuous transforms may
    # have perturbed discrete columns slightly. Round them back.
    if discrete_cols:
        for col in discrete_cols:
            if col >= n_features:
                continue
            mask = np.isfinite(X[:, col])
            X[mask, col] = np.round(X[mask, col])
            if col in cat_cardinality:
                K = cat_cardinality[col]
                X[mask, col] = np.clip(X[mask, col], 0, K - 1)

    # --- 8. Final safety: winsorize + clip + nan check ---
    # Per-column winsorization: clamp extreme outliers to ±6 MAD from median.
    # This prevents the amplification chain (importance × rescaling × heavy-tails)
    # from creating 1e4+ values that dominate the model's internal normalization
    # and cause loss spikes / training collapse.
    for col in range(X.shape[1]):
        v = X[:, col]
        finite = v[np.isfinite(v)]
        if len(finite) < 5:
            continue
        med = np.median(finite)
        mad = np.median(np.abs(finite - med)) * 1.4826
        if mad < 1e-8:
            mad = max(np.std(finite), 1e-8)
        lo, hi = med - 6 * mad, med + 6 * mad
        # Preserve NaN; only clip finite values
        finite_mask = np.isfinite(v)
        v_clipped = np.clip(v, lo, hi)
        X[:, col] = np.where(finite_mask, v_clipped, v)
    # Hard backstop — no feature value should ever exceed ±1e4
    X = np.clip(X, -1e4, 1e4)
    # Preserve NaN from structural missingness; only replace inf
    finite_or_nan = np.isfinite(X) | np.isnan(X)
    X = np.where(finite_or_nan, X, 0.0)
    X = X.astype(np.float32)

    # --- 8.5 Duplicate row injection [synth_v4] ---
    # Real data has ~5% mean duplicate rows (heavy right tail: some up to 49%).
    # This is NOT correlated with discrete features — it arises from repeated
    # measurements, rounding, small datasets.
    # Tuned: 70% of datasets (was 40%), cap 50% (was 30%), scale 0.07 (was 0.05)
    # to match real mean ~0.05 (was producing ~0.02).
    if augment_v4 and n_samples >= 20 and rng.random() < 0.7:
        # Right-skewed with heavier tail: 85% get 2-7%, 15% get larger fractions
        if rng.random() < 0.85:
            dup_frac = min(rng.exponential(0.06), 0.25)
        else:
            dup_frac = min(rng.exponential(0.20), 0.50)
        n_dup = max(1, int(n_samples * dup_frac))
        # Sample rows to duplicate (with replacement — same row can appear 3+x)
        src_idx = rng.choice(n_samples, size=n_dup, replace=True)
        dst_idx = rng.choice(n_samples, size=n_dup, replace=False)
        X[dst_idx] = X[src_idx]
        y[dst_idx] = y[src_idx]
        if X_target is not None:
            X_target[dst_idx] = X_target[src_idx]

    return {
        'X': X,
        'y': y,
        'task_type': task_type,
        'n_classes': n_classes if task_type == 'cls' else None,
        'filtered': False,
        'meta': meta,
        'X_target': None if X_target is None else np.where(
            np.isfinite(X_target),
            X_target,
            0.0,
        ).astype(np.float32),
    }


def _random_decision_tree_predict(X, feature_cols, rng, depth=4, n_classes_leaf=0):
    """Generate predictions from a random (unfitted) decision tree.

    Procedurally generates random axis-aligned splits and random leaf values.
    No sklearn fitting — pure random construction, O(n_samples * depth).

    Args:
        X: feature matrix [n_samples, n_features]
        feature_cols: indices of features this tree can split on
        rng: numpy random generator
        depth: tree depth (2-6)
        n_classes_leaf: if > 0, leaf values are class labels [0, n_classes_leaf)
                        if 0, leaf values are continuous N(0, 1)

    Returns:
        predictions: [n_samples] (float)
    """
    n_samples = X.shape[0]
    n_leaves = 2 ** depth
    if n_classes_leaf > 0:
        leaf_values = rng.integers(0, n_classes_leaf, size=n_leaves).astype(np.float64)
    else:
        leaf_values = rng.standard_normal(n_leaves)

    leaf_idx = np.zeros(n_samples, dtype=np.int64)
    for d in range(depth):
        feat = feature_cols[d % len(feature_cols)]
        col_vals = X[:, feat].copy()
        col_vals = np.where(np.isfinite(col_vals), col_vals, 0.0)
        finite = col_vals[np.isfinite(col_vals)]
        if len(finite) > 1:
            pct = rng.uniform(20, 80)
            threshold = np.nanpercentile(finite, pct)
        else:
            threshold = 0.0
        goes_right = (col_vals > threshold).astype(np.int64)
        leaf_idx = leaf_idx * 2 + goes_right

    leaf_idx = leaf_idx % n_leaves
    return leaf_values[leaf_idx]


def _generate_tree_prior_episode(n_samples, n_features, task_type,
                                 n_classes, rng,
                                 probabilistic_labels=False):
    """Generate a dataset with tree-ensemble targets.

    Produces piecewise-constant targets from random decision tree ensembles.
    Covers XGBoost/LightGBM/CatBoost-like data manifolds that the SCM
    generator doesn't produce.

    Features are a mix of continuous and categorical. The target is the
    averaged output of 3-10 random trees, each splitting on a random
    subset of features.
    """
    # --- Generate features: mix continuous + categorical ---
    X = np.zeros((n_samples, n_features), dtype=np.float64)
    cat_frac = rng.uniform(0.2, 0.6)
    n_cat = max(0, int(n_features * cat_frac))
    n_cont = n_features - n_cat

    # Continuous features
    for j in range(n_cont):
        dist = int(rng.integers(0, 4))
        if dist == 0:
            X[:, j] = rng.standard_normal(n_samples)
        elif dist == 1:
            X[:, j] = rng.uniform(-3, 3, size=n_samples)
        elif dist == 2:
            a, b = rng.uniform(0.5, 5.0), rng.uniform(0.5, 5.0)
            X[:, j] = rng.beta(a, b, size=n_samples) * 6 - 3
        else:
            df = rng.uniform(3, 8)
            X[:, j] = rng.standard_t(df, size=n_samples)

    # Categorical features
    for j in range(n_cont, n_features):
        K = int(rng.integers(2, min(21, max(3, n_samples // 10))))
        X[:, j] = rng.integers(0, K, size=n_samples).astype(np.float64)

    # --- Build random tree ensemble ---
    n_trees = int(rng.integers(3, 11))
    tree_depth = int(rng.integers(2, 7))
    n_active = min(int(rng.integers(2, min(8, n_features) + 1)), n_features)
    y = np.zeros(n_samples, dtype=np.float64)

    for _ in range(n_trees):
        # Each tree uses a random feature subset
        tree_feats = rng.choice(n_features, size=n_active, replace=False)
        y += _random_decision_tree_predict(X, tree_feats, rng, depth=tree_depth)

    y /= n_trees  # Average

    # --- Add noise + create final target ---
    if task_type == 'reg':
        # Calibrated noise
        y_std = np.std(y)
        if y_std > 1e-8:
            target_r2 = rng.uniform(0.4, 0.95)
            noise_std = y_std * np.sqrt((1 - target_r2) / max(target_r2, 1e-6))
            y += rng.standard_normal(n_samples) * noise_std
        # Normalize
        mu, std = np.mean(y), np.std(y)
        if std > 1e-8:
            y = (y - mu) / std
        y = y.astype(np.float32)
        actual_n_classes = None
    else:
        if n_classes is None:
            n_classes = int(rng.integers(2, 11))
        if probabilistic_labels and rng.random() < 0.5:
            y_cls = _probabilistic_label(y, X, n_classes, rng)
        else:
            percentiles = np.linspace(0, 100, n_classes + 1)
            thresholds = np.nanpercentile(y, percentiles[1:-1])
            y_cls = np.digitize(y, thresholds).astype(np.float32)
        y = y_cls
        actual_n_classes = n_classes

    # Add noise features (0-20%)
    noise_frac = rng.uniform(0.0, 0.2)
    n_noise = int(n_features * noise_frac)
    if n_noise > 0:
        noise_cols = rng.choice(n_features, size=n_noise, replace=False)
        for col in noise_cols:
            X[:, col] = rng.standard_normal(n_samples)

    return {
        'X': X,
        'y': y.astype(np.float32),
        'task_type': task_type,
        'n_classes': actual_n_classes,
        'filtered': False,
        'meta': {'tree_prior': True, 'n_trees': n_trees, 'tree_depth': tree_depth},
    }


def _generate_gp_prior_episode(n_samples, n_features, task_type,
                               n_classes, rng,
                               reg_denoise=False,
                               probabilistic_labels=False):
    """Generate a dataset with a Gaussian Process smooth function target.

    Uses Random Fourier Features (RFF) to approximate GP samples:
        y ≈ Σ_i w_i * cos(ω_i · x + φ_i)

    Kernel spectral densities:
        RBF:      ω ~ N(0, 1/l²)         → infinitely smooth
        Matern32: ω ~ Student-t(df=3)/l   → smooth with some roughness
        Matern52: ω ~ Student-t(df=5)/l   → very smooth (real-data sweet spot)

    Directly produces smooth joint multivariate functions — the data shape
    of sulfur, debutanizer, space_ga, kin8nm, houses, physiochemical_protein.
    """
    # --- Generate features ---
    X = np.zeros((n_samples, n_features), dtype=np.float64)

    # Mix of distributions for realism
    for j in range(n_features):
        dist = int(rng.integers(0, 4))
        if dist == 0:
            X[:, j] = rng.standard_normal(n_samples)
        elif dist == 1:
            X[:, j] = rng.uniform(-2, 2, size=n_samples)
        elif dist == 2:
            a, b = rng.uniform(0.5, 5.0), rng.uniform(0.5, 5.0)
            X[:, j] = rng.beta(a, b, size=n_samples) * 4 - 2
        else:
            df = rng.uniform(3, 8)
            X[:, j] = rng.standard_t(df, size=n_samples)

    # Standardize
    for j in range(n_features):
        mu, s = np.mean(X[:, j]), np.std(X[:, j])
        if s > 1e-8:
            X[:, j] = (X[:, j] - mu) / s

    # --- Select active features for the GP target ---
    max_active = min(20, n_features)
    min_active = min(4, n_features)
    n_active = int(rng.integers(min_active, max_active + 1))
    active = rng.choice(n_features, size=n_active, replace=False)
    X_active = X[:, active]

    # --- Add feature correlations (80% chance) ---
    # Real gap datasets have mean_corr 0.2-0.55 (sulfur=0.23, houses=0.28,
    # physiochemical_protein=0.55). Generate correlated features using a
    # random covariance matrix via Cholesky decomposition.
    if rng.random() < 0.8 and n_active >= 3:
        # Target mean absolute correlation: 0.15-0.50
        target_corr = rng.uniform(0.15, 0.50)
        # Build correlation matrix: off-diag = target_corr * random sign
        C = np.eye(n_active)
        for i in range(n_active):
            for j in range(i + 1, n_active):
                c = target_corr * rng.uniform(0.5, 1.5)
                c = min(c, 0.95)  # keep positive definite
                C[i, j] = C[j, i] = c
        # Ensure positive definite
        eigvals = np.linalg.eigvalsh(C)
        if eigvals.min() < 0.01:
            C += (0.02 - eigvals.min()) * np.eye(n_active)
        L = np.linalg.cholesky(C)
        X_active = (L @ X_active.T).T
        # Re-standardize per column
        for j in range(n_active):
            mu, s = np.mean(X_active[:, j]), np.std(X_active[:, j])
            if s > 1e-8:
                X_active[:, j] = (X_active[:, j] - mu) / s
        # Write correlated features back to the full X matrix
        X[:, active] = X_active

    # --- GP kernel parameters ---
    kernel_type = rng.choice(['rbf', 'matern32', 'matern52', 'rbf', 'matern52'])

    # Per-feature lengthscales — use LONGER lengthscales for smoother functions
    # Real gap datasets (sulfur, debutanizer) have very smooth targets.
    # Longer lengthscale = smoother GP = higher R2 = matches real data better.
    lengthscales = rng.uniform(1.0, 5.0, size=n_active)

    # Amplitude
    amplitude = rng.uniform(0.5, 2.0)

    # Number of Random Fourier Features — fewer = smoother GP sample.
    # Real gap datasets have smooth targets (R2=0.4-0.8 with ExtraTrees).
    # Too many RFF creates noisy, oscillatory functions.
    n_rff = int(rng.integers(30, 150))

    # --- Sample spectral frequencies from kernel's spectral density ---
    if kernel_type == 'rbf':
        # RBF kernel: spectral density is Gaussian
        omega = rng.standard_normal((n_rff, n_active)) / lengthscales
    elif kernel_type == 'matern32':
        # Matern-3/2: spectral density is Student-t(df=2*nu) = Student-t(3)
        omega = rng.standard_t(3, size=(n_rff, n_active)) / lengthscales
    else:  # matern52
        # Matern-5/2: spectral density is Student-t(5)
        omega = rng.standard_t(5, size=(n_rff, n_active)) / lengthscales

    # Random phases
    phi = rng.uniform(0, 2 * np.pi, size=n_rff)

    # Random weights (Gaussian → GP sample)
    w = rng.standard_normal(n_rff) * amplitude / np.sqrt(n_rff)

    # --- Compute GP sample: y = Σ w_i cos(ω_i · x + φ_i) ---
    projections = X_active @ omega.T + phi  # [n_samples, n_rff]
    y = np.cos(projections) @ w

    # Optional: add a smooth trend (30% chance)
    if rng.random() < 0.3:
        trend_w = rng.standard_normal(n_active) * 0.3
        y += X_active @ trend_w

    # Soft clip
    y = 50.0 * np.tanh(y / 50.0)

    # --- Add correlated features to non-active columns ---
    remaining = n_features - n_active
    if remaining >= 2 and rng.random() < 0.5:
        n_corr = min(int(rng.integers(2, min(8, remaining) + 1)), remaining)
        corr_targets = rng.choice(
            [i for i in range(n_features) if i not in active],
            size=n_corr, replace=False)
        for ct in corr_targets:
            src = rng.choice(active)
            rho = rng.uniform(0.3, 0.8)
            X[:, ct] = rho * X[:, src] + np.sqrt(1 - rho**2) * rng.standard_normal(n_samples)

    # --- Noise + normalization ---
    if task_type == 'reg':
        y_std = np.std(y)
        if y_std > 1e-8:
            # GP targets tend to be cleaner — higher R²
            target_r2 = rng.uniform(0.75, 0.995) if reg_denoise else rng.uniform(0.55, 0.98)
            noise_std = y_std * np.sqrt((1 - target_r2) / max(target_r2, 1e-6))
            y += rng.standard_normal(n_samples) * noise_std
        # Heteroskedastic noise (15% chance)
        if rng.random() < 0.15:
            het_feat = rng.choice(active)
            het_scale = 0.3 * np.abs(X[:, het_feat])
            y += rng.standard_normal(n_samples) * het_scale * 0.1
        mu, std = np.mean(y), np.std(y)
        if std > 1e-8:
            y = (y - mu) / std
        y = y.astype(np.float32)
        actual_n_classes = None
    else:
        if n_classes is None:
            n_classes = int(rng.integers(2, 11))
        if probabilistic_labels and rng.random() < 0.5:
            y_cls = _probabilistic_label(y, X, n_classes, rng)
        else:
            percentiles = np.linspace(0, 100, n_classes + 1)
            thresholds = np.nanpercentile(y, percentiles[1:-1])
            y_cls = np.digitize(y, thresholds).astype(np.float32)
        y = y_cls
        actual_n_classes = n_classes

    # Add noise features (0-10%) — only if there are non-active columns available
    non_active = [i for i in range(n_features) if i not in active]
    if non_active:
        noise_frac = rng.uniform(0.0, 0.10)
        n_noise = min(int(n_features * noise_frac), len(non_active))
        if n_noise > 0:
            noise_cols = rng.choice(non_active, size=n_noise, replace=False)
            for col in noise_cols:
                X[:, col] = rng.standard_normal(n_samples)

    return {
        'X': X,
        'y': y.astype(np.float32),
        'task_type': task_type,
        'n_classes': actual_n_classes,
        'filtered': False,
        'meta': {'gp_prior': True, 'kernel': kernel_type,
                 'n_active': n_active, 'n_rff': n_rff},
    }


def _generate_quadratic_surface_episode(n_samples, n_features, task_type,
                                        n_classes, rng,
                                        reg_denoise=False,
                                        probabilistic_labels=False):
    """Generate a dataset with a quadratic response surface target.

    y = x^T M x + w^T x + b over a random subset of 4-20 features.
    Covers smooth coupled multivariate interactions (industrial process
    control, kinematics, response surfaces).
    """
    X = np.zeros((n_samples, n_features), dtype=np.float64)
    for j in range(n_features):
        dist = int(rng.integers(0, 4))
        if dist == 0:
            X[:, j] = rng.standard_normal(n_samples)
        elif dist == 1:
            X[:, j] = rng.uniform(-2, 2, size=n_samples)
        elif dist == 2:
            a, b = rng.uniform(0.5, 5.0), rng.uniform(0.5, 5.0)
            X[:, j] = rng.beta(a, b, size=n_samples) * 4 - 2
        else:
            df = rng.uniform(3, 8)
            X[:, j] = rng.standard_t(df, size=n_samples)

    for j in range(n_features):
        col = X[:, j]
        mu, s = np.mean(col), np.std(col)
        if s > 1e-8:
            X[:, j] = (col - mu) / s

    max_active = min(20, n_features)
    min_active = min(4, n_features)
    n_active = int(rng.integers(min_active, max_active + 1))
    active = rng.choice(n_features, size=n_active, replace=False)
    X_active = X[:, active]

    A = rng.standard_normal((n_active, n_active))
    M = (A + A.T) / 2.0
    eigvals, eigvecs = np.linalg.eigh(M)
    max_abs = max(np.abs(eigvals).max(), 1e-8)
    eigvals = eigvals / max_abs * rng.uniform(0.3, 1.0, size=n_active)
    M = (eigvecs * eigvals[None, :]) @ eigvecs.T

    w = rng.standard_normal(n_active) * 0.5
    y_quad = np.sum((X_active @ M) * X_active, axis=1)
    y_linear = X_active @ w
    b = rng.standard_normal() * 0.1
    y = y_quad + y_linear + b

    if rng.random() < 0.3 and n_active >= 3:
        n_cubic = int(rng.integers(1, min(4, n_active)))
        cubic_feats = rng.choice(n_active, size=n_cubic, replace=False)
        cubic_coefs = rng.standard_normal(n_cubic) * 0.2
        for k, feat_idx in enumerate(cubic_feats):
            y += cubic_coefs[k] * X_active[:, feat_idx] ** 3

    y = 50.0 * np.tanh(y / 50.0)

    remaining = n_features - n_active
    if rng.random() < 0.5 and remaining >= 2:
        n_corr = min(int(rng.integers(2, min(6, remaining) + 1)), remaining)
        if n_corr > 0:
            corr_targets = rng.choice(
                [i for i in range(n_features) if i not in active],
                size=n_corr, replace=False)
            for ct in corr_targets:
                src = rng.choice(active)
                rho = rng.uniform(0.5, 0.95)
                X[:, ct] = rho * X[:, src] + np.sqrt(1 - rho**2) * rng.standard_normal(n_samples)

    if task_type == 'reg':
        y_std = np.std(y)
        if y_std > 1e-8:
            target_r2 = rng.uniform(0.6, 0.98) if reg_denoise else rng.uniform(0.4, 0.95)
            noise_std = y_std * np.sqrt((1 - target_r2) / max(target_r2, 1e-6))
            y += rng.standard_normal(n_samples) * noise_std
        if rng.random() < 0.15:
            het_feat = rng.choice(active)
            het_scale = 0.5 * np.abs(X[:, het_feat])
            y += rng.standard_normal(n_samples) * het_scale * 0.1
        mu, std = np.mean(y), np.std(y)
        if std > 1e-8:
            y = (y - mu) / std
        y = y.astype(np.float32)
        actual_n_classes = None
    else:
        if n_classes is None:
            n_classes = int(rng.integers(2, 11))
        if probabilistic_labels and rng.random() < 0.5:
            y_cls = _probabilistic_label(y, X, n_classes, rng)
        else:
            percentiles = np.linspace(0, 100, n_classes + 1)
            thresholds = np.nanpercentile(y, percentiles[1:-1])
            y_cls = np.digitize(y, thresholds).astype(np.float32)
        y = y_cls
        actual_n_classes = n_classes

    noise_frac = rng.uniform(0.0, 0.15)
    n_noise = int(n_features * noise_frac)
    if n_noise > 0:
        noise_cols = rng.choice(
            [i for i in range(n_features) if i not in active],
            size=min(n_noise, n_features - n_active), replace=False)
        for col in noise_cols:
            X[:, col] = rng.standard_normal(n_samples)

    return {
        'X': X,
        'y': y.astype(np.float32),
        'task_type': task_type,
        'n_classes': actual_n_classes,
        'filtered': False,
        'meta': {'quadratic_surface': True, 'n_active': n_active},
    }


def _generate_sparse_nonlinear_episode(n_samples, n_features, task_type,
                                       n_classes, rng,
                                       reg_denoise=False,
                                       probabilistic_labels=False):
    """QSAR/MIP-inspired high-dim sparse nonlinear with realistic correlation structure.

    Real chemistry/genomics datasets have:
    - Feature blocks with within-block correlation 0.3-0.95 (multicollinearity)
    - A few strong predictors + many weakly-correlated features (not binary active/noise)
    - Feature transforms: discretized, squared, log-abs (molecular descriptor diversity)
    - Heteroskedastic noise (uncertainty scales with feature magnitude)
    - Occasional threshold effects (reaction regimes)

    Structure:
    - n_blocks feature groups, each tied to a latent factor z_b
    - Within block: x_j = ρ_j · z_b + sqrt(1-ρ_j²) · ε,  ρ_j ∈ [0.3, 0.95]
    - 15% of features get post-hoc nonlinear transforms (square, log-abs, discretize, tanh)
    - n_strong features drive y via nonlinear function (MLP/GAM/quad/RBF/interactions)
    - n_weak features add small linear background signal
    - Remaining features are redundant proxies (correlated via blocks) or distractors
    - 30% of episodes add a threshold effect; 50% use heteroskedastic noise
    """
    # === 1. Feature blocks with multicollinearity ===
    # n_blocks: roughly n_features / avg_block_size, clipped to [max(1, n/10), min(80, n/2)]
    min_blocks = max(1, min(10, n_features // 10))
    max_blocks = min(80, max(min_blocks, n_features // 2))
    if max_blocks <= min_blocks:
        n_blocks = min_blocks
    else:
        n_blocks = int(rng.integers(min_blocks, max_blocks + 1))
    n_blocks = max(1, n_blocks)

    # Random block assignment per feature
    block_ids = rng.integers(0, n_blocks, size=n_features)
    block_latents = rng.standard_normal((n_samples, n_blocks))

    # Feature values: partial copy of block latent + independent noise
    X = np.zeros((n_samples, n_features), dtype=np.float64)
    for j in range(n_features):
        b = int(block_ids[j])
        rho = rng.uniform(0.3, 0.95)
        X[:, j] = rho * block_latents[:, b] + np.sqrt(1.0 - rho * rho) * rng.standard_normal(n_samples)

        # 15% chance: apply nonlinear transform (chemistry/genomics descriptor flavor)
        if rng.random() < 0.15:
            t = int(rng.integers(0, 4))
            col = X[:, j]
            if t == 0:
                X[:, j] = col * col - 1.0  # centered square
            elif t == 1:
                X[:, j] = np.sign(col) * np.log1p(np.abs(col))
            elif t == 2:
                n_bins = int(rng.integers(3, 15))
                bin_edges = np.linspace(-2.5, 2.5, n_bins + 1)[1:-1]
                X[:, j] = np.digitize(col, bin_edges).astype(np.float64)
            else:
                X[:, j] = np.tanh(col)

    # Standardize each feature column
    for j in range(n_features):
        col = X[:, j]
        mu = np.mean(col)
        s = np.std(col)
        if s > 1e-8:
            X[:, j] = (col - mu) / s

    # === 2. Pick strong and weak features ===
    # Strong features drive the nonlinear target; prefer one per block for
    # independent signal contributions.
    max_strong = min(20, max(3, n_features // 50), n_features)
    min_strong = min(3, max_strong)
    if max_strong <= min_strong:
        n_strong = min_strong
    else:
        n_strong = int(rng.integers(min_strong, max_strong + 1))
    n_strong = max(1, min(n_strong, n_features))

    block_perm = rng.permutation(n_blocks)
    strong_indices = []
    for b in block_perm:
        if len(strong_indices) >= n_strong:
            break
        in_block = np.where(block_ids == b)[0]
        if len(in_block) > 0:
            strong_indices.append(int(rng.choice(in_block)))
    # If fewer unique blocks than n_strong, pad with random non-duplicates.
    while len(strong_indices) < n_strong:
        candidates = [i for i in range(n_features) if i not in strong_indices]
        if not candidates:
            break
        strong_indices.append(int(rng.choice(candidates)))
    strong_indices = sorted(strong_indices)
    n_strong = len(strong_indices)

    non_strong = [i for i in range(n_features) if i not in strong_indices]
    # Weak features: small linear contribution. Target 20-200, scaled down for small n_features.
    if len(non_strong) >= 20:
        max_weak = min(200, max(20, n_features // 5), len(non_strong))
        n_weak = int(rng.integers(20, max_weak + 1))
    else:
        n_weak = max(0, min(len(non_strong) // 2, 10))
    weak_indices = sorted(rng.choice(non_strong, size=n_weak, replace=False).tolist()) if n_weak > 0 else []

    X_strong = X[:, strong_indices]

    # === 3. Strong-signal target (nonlinear over strong features) ===
    fn_type = int(rng.integers(0, 5))

    if fn_type == 0:
        hidden = int(rng.integers(16, 65))
        W1 = rng.standard_normal((n_strong, hidden)) / np.sqrt(n_strong)
        b1 = rng.standard_normal(hidden) * 0.1
        W2 = rng.standard_normal(hidden) / np.sqrt(hidden)
        act = int(rng.integers(0, 3))
        h = X_strong @ W1 + b1
        if act == 0:
            h = np.maximum(h, 0)
        elif act == 1:
            h = np.tanh(h)
        else:
            h = np.sin(h)
        y = h @ W2
        fn_name = 'mlp'
    elif fn_type == 1:
        fn_bank = [
            lambda x, r=rng: x,
            lambda x, r=rng: x ** 2,
            lambda x, r=rng: np.log1p(np.abs(x)) * np.sign(x),
            lambda x, r=rng: np.tanh(x * r.uniform(0.5, 2.0)),
            lambda x, r=rng: np.sin(x * r.uniform(1.0, 4.0)),
            lambda x, r=rng: np.cos(x * r.uniform(1.0, 4.0)),
            lambda x, r=rng: 1.0 / (1.0 + np.exp(-x * r.uniform(0.5, 3.0))),
        ]
        y = np.zeros(n_samples, dtype=np.float64)
        coefs = rng.standard_normal(n_strong)
        for k in range(n_strong):
            fn = fn_bank[int(rng.integers(0, len(fn_bank)))]
            y += coefs[k] * fn(X_strong[:, k])
        fn_name = 'gam'
    elif fn_type == 2:
        A = rng.standard_normal((n_strong, n_strong))
        M = (A + A.T) / 2.0
        eigvals, eigvecs = np.linalg.eigh(M)
        max_abs = max(np.abs(eigvals).max(), 1e-8)
        eigvals = eigvals / max_abs * rng.uniform(0.3, 1.0, size=n_strong)
        M = (eigvecs * eigvals[None, :]) @ eigvecs.T
        w = rng.standard_normal(n_strong) * 0.5
        y = np.sum((X_strong @ M) * X_strong, axis=1) + X_strong @ w
        fn_name = 'quadratic'
    elif fn_type == 3:
        n_centers = int(rng.integers(3, min(12, n_samples // 10 + 1)))
        centers = rng.standard_normal((n_centers, n_strong))
        sigmas = rng.uniform(0.5, 3.0, size=n_centers)
        weights = rng.standard_normal(n_centers)
        y = np.zeros(n_samples, dtype=np.float64)
        for c_idx in range(n_centers):
            diffs = X_strong - centers[c_idx]
            dists_sq = np.sum(diffs ** 2, axis=1)
            y += weights[c_idx] * np.exp(-dists_sq / (2 * sigmas[c_idx] ** 2))
        fn_name = 'rbf'
    else:
        w = rng.standard_normal(n_strong)
        y = X_strong @ w
        max_inter = max(2, min(8, n_strong * (n_strong - 1) // 2))
        n_interactions = int(rng.integers(1, max_inter + 1)) if max_inter >= 1 else 0
        for _ in range(n_interactions):
            i, j = rng.choice(n_strong, size=2, replace=False)
            coef = rng.standard_normal() * 0.5
            y += coef * X_strong[:, i] * X_strong[:, j]
        fn_name = 'interactions'

    # Smooth cap on strong signal
    y = 50.0 * np.tanh(y / 50.0)

    # === 4. Weak additive background (linear over weak features) ===
    if n_weak > 0:
        y_strong_std = max(float(np.std(y)), 1e-8)
        weak_scale_frac = rng.uniform(0.10, 0.30)
        X_weak = X[:, weak_indices]
        weak_coefs = rng.standard_normal(n_weak)
        y_weak_raw = X_weak @ weak_coefs
        y_weak_std = max(float(np.std(y_weak_raw)), 1e-8)
        y = y + y_weak_raw * (weak_scale_frac * y_strong_std / y_weak_std)

    # === 5. Occasional threshold effect (30%) ===
    if rng.random() < 0.30 and n_strong > 0:
        idx = int(rng.integers(0, n_strong))
        threshold = rng.uniform(-1.0, 1.0)
        effect_scale = max(float(np.std(y)), 1e-8) * rng.uniform(0.2, 0.6)
        above = (X_strong[:, idx] > threshold).astype(np.float64) * 2.0 - 1.0
        y = y + above * effect_scale

    # === 6. Noise (heteroskedastic 50% / homoscedastic 50%) ===
    if task_type == 'reg':
        y_std = float(np.std(y))
        if y_std > 1e-8:
            target_r2 = rng.uniform(0.5, 0.95) if reg_denoise else rng.uniform(0.3, 0.90)
            base_noise_std = y_std * np.sqrt((1.0 - target_r2) / max(target_r2, 1e-6))

            if rng.random() < 0.5 and n_strong > 0:
                driver_idx = int(rng.integers(0, n_strong))
                driver = X_strong[:, driver_idx]
                noise_scale = np.abs(driver) + 0.5
                noise_scale = noise_scale / max(float(np.mean(noise_scale)), 1e-8)
                y = y + rng.standard_normal(n_samples) * base_noise_std * noise_scale
            else:
                y = y + rng.standard_normal(n_samples) * base_noise_std

        mu = float(np.mean(y))
        sd = float(np.std(y))
        if sd > 1e-8:
            y = (y - mu) / sd
        y = y.astype(np.float32)
        actual_n_classes = None
    else:
        if n_classes is None:
            n_classes = int(rng.integers(2, 11))
        if probabilistic_labels and rng.random() < 0.5:
            y_cls = _probabilistic_label(y, X, n_classes, rng)
        else:
            percentiles = np.linspace(0, 100, n_classes + 1)
            thresholds = np.nanpercentile(y, percentiles[1:-1])
            y_cls = np.digitize(y, thresholds).astype(np.float32)
        y = y_cls
        actual_n_classes = n_classes

    return {
        'X': X.astype(np.float32),
        'y': y.astype(np.float32),
        'task_type': task_type,
        'n_classes': actual_n_classes,
        'filtered': False,
        'meta': {
            'sparse_nonlinear_v2': True,
            'n_strong': n_strong,
            'n_weak': n_weak,
            'n_blocks': n_blocks,
            'fn_type': fn_name,
        },
    }


def _generate_lookup_prior_episode(n_samples, n_features, task_type,
                                   n_classes, rng,
                                   probabilistic_labels=False):
    """Generate a dataset with categorical lookup targets.

    Produces data where y = f(entity_ids) + continuous_signal + noise.
    Covers repeated-entity tasks (Amazon, hotel booking, customer churn)
    where entity IDs (user, product, store) drive the target.

    1-3 categorical ID columns with Zipf frequency + 2-10 continuous features.
    Target depends primarily on entity-specific effects.
    """
    X = np.zeros((n_samples, n_features), dtype=np.float64)

    # --- Categorical ID columns (1-3) ---
    n_id_cols = min(int(rng.integers(1, 4)), max(1, n_features // 3))
    n_cont = max(2, n_features - n_id_cols)
    n_id_cols = n_features - n_cont  # Adjust if needed

    entity_signal = np.zeros(n_samples, dtype=np.float64)
    for j in range(n_id_cols):
        # Cardinality: 10-300, often higher than n_samples/4 to force
        # genuine train/test partial overlap (audit: chscase_foot has 236
        # train levels vs 125 test, only 64 overlap; CookbookReviews and
        # Goodreads have similar high-cardinality patterns).
        # Mix of "moderate" and "high-cardinality near-unique" regimes.
        if rng.random() < 0.40:
            # High-cardinality (near-unique IDs): K close to n_samples
            max_K = int(min(300, max(20, n_samples // 2)))
        else:
            # Moderate cardinality (counts more useful)
            max_K = int(min(200, max(10, n_samples // 4)))
        K = int(np.exp(rng.uniform(np.log(10.0), np.log(float(max_K)))))
        K = int(np.clip(K, 10, max_K))

        # Zipf-like frequency. Stronger long-tail (alpha 1.0-3.0 vs old
        # 0.6-1.5) — most levels appear ≤2 times, matching real-world
        # high-card data.
        alpha = rng.uniform(1.0, 3.0)
        ranks = np.arange(1, K + 1, dtype=np.float64)
        probs = ranks ** (-alpha)
        probs /= probs.sum()
        cat_ids = rng.choice(K, size=n_samples, replace=True, p=probs)

        # Force some IDs to appear EXACTLY ONCE (rare singleton pattern).
        # Real high-card datasets have ~5-25% singleton rate. Pick random
        # rows to overwrite with rare-IDs (drawn from the tail of K).
        if rng.random() < 0.50 and K > 30:
            singleton_frac = rng.uniform(0.05, 0.25)
            n_singletons = max(1, int(n_samples * singleton_frac))
            # IDs in the tail (less common already) — promote to singletons
            tail_start = max(K - n_singletons, K // 2)
            singleton_ids = rng.choice(
                np.arange(tail_start, K), size=n_singletons,
                replace=(K - tail_start < n_singletons)
            )
            singleton_rows = rng.choice(n_samples, size=n_singletons, replace=False)
            cat_ids[singleton_rows] = singleton_ids

        # Per-entity random effect with group structure
        n_groups = max(2, min(20, int(np.sqrt(K))))
        entity_group = rng.integers(0, n_groups, size=K)
        group_effect = rng.standard_normal(n_groups) * 1.5
        entity_effect = (
            0.7 * group_effect[entity_group]
            + 0.4 * rng.standard_normal(K)
        )
        entity_effect -= entity_effect.mean()
        entity_effect /= (np.std(entity_effect) + 1e-8)

        # Weight decreases for secondary ID columns
        weight = 1.0 if j == 0 else rng.uniform(0.3, 0.7)
        entity_signal += weight * entity_effect[cat_ids]

        X[:, j] = cat_ids.astype(np.float64)

    # --- Continuous features ---
    for j in range(n_id_cols, n_features):
        dist = int(rng.integers(0, 3))
        if dist == 0:
            X[:, j] = rng.standard_normal(n_samples)
        elif dist == 1:
            X[:, j] = rng.uniform(-3, 3, size=n_samples)
        else:
            a, b = rng.uniform(0.5, 5.0), rng.uniform(0.5, 5.0)
            X[:, j] = rng.beta(a, b, size=n_samples) * 6 - 3

    # Weak continuous signal (1-3 features)
    n_active_cont = min(int(rng.integers(1, 4)), n_cont)
    active_cols = rng.choice(range(n_id_cols, n_features),
                             size=n_active_cont, replace=False)
    beta = rng.standard_normal(n_active_cont) * 0.5
    cont_signal = X[:, active_cols] @ beta

    # Combined signal: entity dominant, continuous secondary
    entity_weight = rng.uniform(0.6, 0.9)
    y_raw = entity_weight * entity_signal + (1 - entity_weight) * cont_signal

    # --- Create final target ---
    if task_type == 'reg':
        y_std = np.std(y_raw)
        if y_std > 1e-8:
            target_r2 = rng.uniform(0.4, 0.90)
            noise_std = y_std * np.sqrt((1 - target_r2) / max(target_r2, 1e-6))
            y_raw += rng.standard_normal(n_samples) * noise_std
        mu, std = np.mean(y_raw), np.std(y_raw)
        if std > 1e-8:
            y_raw = (y_raw - mu) / std
        y = y_raw.astype(np.float32)
        actual_n_classes = None
    else:
        if n_classes is None:
            n_classes = int(rng.integers(2, 11))
        if probabilistic_labels and rng.random() < 0.5:
            y = _probabilistic_label(y_raw, X, n_classes, rng)
        else:
            percentiles = np.linspace(0, 100, n_classes + 1)
            thresholds = np.nanpercentile(y_raw, percentiles[1:-1])
            y = np.digitize(y_raw, thresholds).astype(np.float32)
        actual_n_classes = n_classes

    # Pad remaining columns with noise features
    for j in range(n_id_cols + n_active_cont, n_features):
        if j >= n_id_cols and j not in active_cols:
            X[:, j] = rng.standard_normal(n_samples)

    return {
        'X': X,
        'y': y.astype(np.float32),
        'task_type': task_type,
        'n_classes': actual_n_classes,
        'filtered': False,
        'meta': {'lookup_prior': True, 'n_id_cols': n_id_cols},
    }


def _generate_clean_lowdim_episode(n_samples, n_features_batch, task_type,
                                   n_classes, rng,
                                   probabilistic_labels=False,
                                   nominal_categoricals=False):
    """Generate a clean, low-dimensional episode with high categorical fraction.

    Covers real-world tabular datasets with 5-30 features, mostly categorical,
    simple rules, and low noise (Amazon, polish_companies, hotel booking, etc.).

    The generated X has n_features_batch columns, with the first n_features_actual
    being real features and the rest padded with NaN.
    """
    # Override n_features to low-dim range, clamped to batch width
    n_features = int(rng.integers(5, max(6, min(31, n_features_batch + 1))))
    n_features = min(n_features, n_features_batch)

    # High categorical fraction: 30-80%
    cat_frac = rng.uniform(0.30, 0.80)
    n_cat = max(1, int(n_features * cat_frac))
    n_cont = n_features - n_cat

    X = np.zeros((n_samples, n_features), dtype=np.float64)

    # --- Generate continuous features ---
    for j in range(n_cont):
        root_type = int(rng.integers(0, 4))
        if root_type == 0:
            X[:, j] = rng.standard_normal(n_samples)
        elif root_type == 1:
            X[:, j] = rng.uniform(-3, 3, size=n_samples)
        elif root_type == 2:
            a, b = rng.uniform(0.5, 5.0), rng.uniform(0.5, 5.0)
            X[:, j] = rng.beta(a, b, size=n_samples) * 6 - 3
        else:
            X[:, j] = rng.standard_normal(n_samples) * rng.uniform(0.5, 3.0)

    # --- Generate categorical features ---
    cat_effects_list = []
    for j in range(n_cont, n_features):
        cardinality = int(rng.integers(2, min(21, max(3, n_samples // 10))))
        # Zipf-like frequency
        alpha = rng.uniform(0.5, 1.2)
        ranks = np.arange(1, cardinality + 1, dtype=np.float64)
        probs = ranks ** (-alpha)
        probs /= probs.sum()
        cat_ids = rng.choice(cardinality, size=n_samples, replace=True, p=probs)

        if nominal_categoricals and rng.random() < 0.6:
            # Nominal: random effects per category
            effects = rng.standard_normal(cardinality)
            effects -= effects.mean()
            effects /= (np.std(effects) + 1e-8)
            cat_effects_list.append(effects[cat_ids])
        else:
            # Ordinal: category ID is the value (preserves ordering)
            cat_effects_list.append(cat_ids.astype(np.float64) / max(cardinality - 1, 1))

        X[:, j] = cat_ids.astype(np.float64)

    # --- Generate target y ---
    if task_type == 'reg':
        # Simple target: sparse linear on continuous + categorical effects
        y = np.zeros(n_samples, dtype=np.float64)
        n_active = min(int(rng.integers(1, 6)), n_features)
        active_cols = rng.choice(n_cont, size=min(n_active, n_cont), replace=False)
        if len(active_cols) > 0:
            beta = rng.standard_normal(len(active_cols))
            y += X[:, active_cols] @ beta
        # Add categorical effects
        for eff in cat_effects_list:
            w = rng.uniform(0.3, 1.5)
            y += w * eff
        # Low noise
        target_r2 = rng.uniform(0.6, 0.95)
        y_std = np.std(y)
        if y_std > 1e-8:
            noise_std = y_std * np.sqrt((1 - target_r2) / max(target_r2, 1e-6))
            y += rng.standard_normal(n_samples) * noise_std
        # Normalize
        mu, std = np.mean(y), np.std(y)
        if std > 1e-8:
            y = (y - mu) / std
        y = y.astype(np.float32)
        actual_n_classes = None
    else:
        # Classification target
        if n_classes is None:
            n_classes = int(rng.integers(2, 11))
        # Build continuous signal from features
        y_raw = np.zeros(n_samples, dtype=np.float64)
        n_active = min(int(rng.integers(1, 4)), n_cont)
        if n_active > 0:
            active_cols = rng.choice(n_cont, size=n_active, replace=False)
            beta = rng.standard_normal(n_active)
            y_raw += X[:, active_cols] @ beta
        for eff in cat_effects_list:
            w = rng.uniform(0.5, 2.0)
            y_raw += w * eff
        # Add small noise
        y_raw += rng.standard_normal(n_samples) * 0.1

        if probabilistic_labels and rng.random() < 0.6:
            y = _probabilistic_label(y_raw, X, n_classes, rng)
        else:
            percentiles = np.linspace(0, 100, n_classes + 1)
            thresholds = np.nanpercentile(y_raw, percentiles[1:-1])
            y = np.digitize(y_raw, thresholds).astype(np.float32)
        actual_n_classes = n_classes

    # Ensure output matches batch n_features: pad with NaN or truncate
    if n_features < n_features_batch:
        X_padded = np.full((n_samples, n_features_batch), np.nan, dtype=np.float64)
        X_padded[:, :n_features] = X
        X = X_padded
    elif n_features > n_features_batch:
        X = X[:, :n_features_batch]
        n_features = n_features_batch

    # Minimal winsorization
    for col in range(n_features):
        col_vals = X[:, col]
        finite = np.isfinite(col_vals)
        if finite.sum() > 4:
            sorted_vals = np.sort(col_vals[finite])
            n_fin = len(sorted_vals)
            median = sorted_vals[n_fin // 2]
            mad = np.median(np.abs(sorted_vals - median))
            if mad > 1e-12:
                lo = median - 6 * mad
                hi = median + 6 * mad
                X[finite, col] = np.clip(col_vals[finite], lo, hi)

    return {
        'X': X.astype(np.float64),
        'y': y.astype(np.float32),
        'task_type': task_type,
        'n_classes': actual_n_classes,
        'filtered': False,
        'meta': {'clean_lowdim': True, 'actual_n_features': n_features},
    }


def _generate_cat_dominant_episode(n_samples, n_features, task_type,
                                    n_classes, rng,
                                    probabilistic_labels=False):
    """Generate a categorical-dominant regression dataset.

    Fills the gap between SCM (mostly continuous, only 1-3 cat cols via
    synth_v3 cat branch) and lookup_prior (limited to 1-3 ID columns):
    real benchmarks like Ailerons, Buzzinsocialmedia, Food_Delivery_Time,
    MIP-2016 have 10-50 categorical features that drive most of the signal.

    Generation:
      - 60-95% of columns are categorical with mixed cardinalities (2-50,
        biased toward low-K which is most common in real data)
      - y = Σ per-column category-level effects (importance-weighted)
            + 0-4 pairwise cat-cat interactions
            + small continuous-feature signal (10-25% of variance)
            + calibrated Gaussian noise targeting R² ∈ [0.45, 0.92]
      - Mix of nominal (random per-level effects) and ordinal (monotone)
        cats per column
    """
    X = np.zeros((n_samples, n_features), dtype=np.float64)

    cat_frac = float(rng.uniform(0.60, 0.95))
    n_cat = max(min(n_features, 4), int(round(n_features * cat_frac)))
    n_cont = n_features - n_cat

    # Cardinality distribution: 40% binary, 35% small (3-7), 20% medium
    # (8-20), 5% large (21-50). Real datasets are dominated by binary and
    # small-cardinality cats; this is the empirical mix from a survey of
    # OpenML reg datasets.
    cat_cards = []
    for j in range(n_cat):
        u = rng.random()
        if u < 0.40:
            K = 2
        elif u < 0.75:
            K = int(rng.integers(3, 8))
        elif u < 0.95:
            K = int(rng.integers(8, 21))
        else:
            K = int(rng.integers(21, 51))
        # Cap so each level has at least ~5 samples on average
        K = max(2, min(K, max(2, n_samples // 5)))
        cat_cards.append(K)

        # Frequency: 60% uniform (nominal), 40% Zipf-skewed
        if rng.random() < 0.60:
            cat_ids = rng.integers(0, K, size=n_samples)
        else:
            alpha = rng.uniform(0.5, 2.0)
            ranks = np.arange(1, K + 1, dtype=np.float64)
            probs = ranks ** (-alpha)
            probs /= probs.sum()
            cat_ids = rng.choice(K, size=n_samples, replace=True, p=probs)
        X[:, j] = cat_ids.astype(np.float64)

    # Continuous features (minority)
    for j in range(n_cat, n_features):
        dist = int(rng.integers(0, 3))
        if dist == 0:
            X[:, j] = rng.standard_normal(n_samples)
        elif dist == 1:
            X[:, j] = rng.uniform(-3, 3, size=n_samples)
        else:
            X[:, j] = rng.standard_normal(n_samples) ** 2 - 1.0

    # --- Build y: per-column main effects + sparse pairwise interactions ---
    y_raw = np.zeros(n_samples, dtype=np.float64)

    # Active subset: 50-90% of cat cols are informative; the rest are
    # noise distractors (matches real-data sparsity in cat-heavy datasets).
    active_frac = float(rng.uniform(0.50, 0.90))
    n_active_cat = max(2, int(round(n_cat * active_frac)))
    active_cat_cols = rng.choice(n_cat, size=n_active_cat, replace=False)

    # Power-law importance: a few cat cols dominate
    importance = rng.exponential(1.0, size=n_active_cat)
    importance /= max(importance.mean(), 1e-8)

    for k, col in enumerate(active_cat_cols):
        K = cat_cards[col]
        cat_ids = X[:, col].astype(int)
        # 70% nominal (random per-level effects), 30% ordinal (monotone +
        # small noise). Binary cats always treated as nominal.
        if rng.random() < 0.70 or K <= 2:
            level_effects = rng.standard_normal(K)
        else:
            slope = rng.standard_normal()
            level_effects = (np.arange(K, dtype=np.float64) - K / 2.0) * slope / max(K, 2)
            level_effects += rng.standard_normal(K) * 0.2
        level_effects -= level_effects.mean()
        y_raw += importance[k] * level_effects[cat_ids]

    # Pairwise cat-cat interactions: 0-4 pairs (only when joint table fits)
    n_pairs = int(rng.integers(0, min(5, n_active_cat // 2 + 1)))
    for _ in range(n_pairs):
        if n_active_cat < 2:
            break
        a, b = rng.choice(active_cat_cols, size=2, replace=False)
        Ka, Kb = cat_cards[a], cat_cards[b]
        # Skip if joint table too large vs n_samples (would make every
        # cell ~unique, killing signal).
        if Ka * Kb > max(8, n_samples // 3):
            continue
        ids_a = X[:, a].astype(int)
        ids_b = X[:, b].astype(int)
        joint_effects = rng.standard_normal((Ka, Kb)) * rng.uniform(0.3, 0.8)
        joint_effects -= joint_effects.mean()
        y_raw += joint_effects[ids_a, ids_b]

    # Small continuous-feature signal (cat dominant)
    if n_cont > 0:
        n_active_cont = min(n_cont, int(rng.integers(1, max(2, n_cont // 2 + 1))))
        active_cont_cols = rng.choice(range(n_cat, n_features),
                                        size=n_active_cont, replace=False)
        beta = rng.standard_normal(n_active_cont) * 0.4
        cont_signal = X[:, active_cont_cols] @ beta
        if np.std(y_raw) > 1e-8 and np.std(cont_signal) > 1e-8:
            y_raw = (y_raw / np.std(y_raw)) + 0.3 * (cont_signal / np.std(cont_signal))

    # Calibrated noise + standardize
    if task_type == 'reg':
        y_std = float(np.std(y_raw))
        if y_std > 1e-8:
            target_r2 = float(rng.uniform(0.45, 0.92))
            noise_std = y_std * np.sqrt((1 - target_r2) / max(target_r2, 1e-6))
            y_raw += rng.standard_normal(n_samples) * noise_std
        mu, std = float(np.mean(y_raw)), float(np.std(y_raw))
        if std > 1e-8:
            y_raw = (y_raw - mu) / std
        y = y_raw.astype(np.float32)
        actual_n_classes = None
    else:
        if n_classes is None:
            n_classes = int(rng.integers(2, 11))
        if probabilistic_labels and rng.random() < 0.5:
            y = _probabilistic_label(y_raw, X, n_classes, rng)
        else:
            percentiles = np.linspace(0, 100, n_classes + 1)
            thresholds = np.nanpercentile(y_raw, percentiles[1:-1])
            y = np.digitize(y_raw, thresholds).astype(np.float32)
        actual_n_classes = n_classes

    # Winsorize continuous-only columns (cat ids stay as integer levels).
    for col in range(n_cat, n_features):
        col_vals = X[:, col]
        finite = np.isfinite(col_vals)
        if finite.sum() > 4:
            sorted_vals = np.sort(col_vals[finite])
            n_fin = len(sorted_vals)
            median = sorted_vals[n_fin // 2]
            mad = np.median(np.abs(sorted_vals - median))
            if mad > 1e-12:
                X[finite, col] = np.clip(col_vals[finite],
                                           median - 6 * mad,
                                           median + 6 * mad)

    return {
        'X': X.astype(np.float64),
        'y': y.astype(np.float32),
        'task_type': task_type,
        'n_classes': actual_n_classes,
        'filtered': False,
        'meta': {'cat_dominant': True, 'n_cat': n_cat,
                 'cat_frac': float(n_cat) / max(n_features, 1)},
    }


def _generate_binary_fingerprint_episode(n_samples, n_features, task_type,
                                          n_classes, rng,
                                          probabilistic_labels=False):
    """Generate a binary-fingerprint regression episode.

    Targets the QSAR-TID-11 archetype: a high-dimensional binary feature
    matrix (chemical or molecular fingerprints) where only 5-30 of the
    bits actually drive y; the rest are noise distractors.

    Generation:
      - n_features columns, all binary (0/1)
      - per-column on-rate drawn from Beta(0.5, 4) — heavy mass near 0,
        rare-bit pattern that matches real fingerprints
      - y is a sparse function (linear / shallow MLP / GAM) over 5-30
        active bits, importance-weighted with a power-law tail
      - calibrated Gaussian noise targeting R² ∈ [0.40, 0.90]
    """
    # Per-column on-rate: heavy-tailed near 0 (most bits rare).
    on_rates = rng.beta(0.5, 4.0, size=n_features)
    on_rates = np.clip(on_rates, 1.0 / max(n_samples, 1), 0.5)

    X = (rng.random((n_samples, n_features)) < on_rates).astype(np.float64)

    # Degenerate-column repair: flip ~1% of cells in all-0 / all-1 columns
    # so the encoder sees per-column variance.
    for j in range(n_features):
        s = int(X[:, j].sum())
        if s == 0:
            flip_idx = rng.choice(n_samples, size=max(1, n_samples // 100),
                                    replace=False)
            X[flip_idx, j] = 1.0
        elif s == n_samples:
            flip_idx = rng.choice(n_samples, size=max(1, n_samples // 100),
                                    replace=False)
            X[flip_idx, j] = 0.0

    # --- Active subset (signal-bearing bits) ---
    # Sample 2..30 active bits, clamped by n_features so the choice() call
    # never asks for more columns than exist (small-bucket safety).
    hi = min(30, max(2, n_features // 2 + 1))
    lo = min(5, hi)
    n_active = int(rng.integers(lo, hi + 1))
    n_active = max(2, min(n_active, n_features))
    active_cols = rng.choice(n_features, size=n_active, replace=False)

    # Power-law importance: a few critical bits matter most
    imp_alpha = rng.uniform(1.0, 3.0)
    importances = np.sort(rng.pareto(imp_alpha, size=n_active))[::-1]
    importances = importances / max(importances.mean(), 1e-8)
    signs = rng.choice([-1.0, 1.0], size=n_active)

    X_active = X[:, active_cols]

    # --- Target type: 50% sparse linear, 30% shallow MLP, 20% GAM ---
    target_t = rng.random()

    if target_t < 0.50:
        beta = importances * signs
        y_raw = X_active @ beta
        # 60% chance to add 1-3 pairwise AND interactions (substructure
        # co-occurrence, common in cheminformatics)
        if rng.random() < 0.6 and n_active >= 4:
            n_int = int(rng.integers(1, 4))
            for _ in range(n_int):
                i, j = rng.choice(n_active, size=2, replace=False)
                w = rng.standard_normal() * 0.5
                y_raw += w * X_active[:, i] * X_active[:, j]

    elif target_t < 0.80:
        # Shallow MLP, 1 hidden layer, tanh activation
        h_dim = int(rng.integers(4, 16))
        W1 = rng.standard_normal((n_active, h_dim)) / np.sqrt(max(n_active, 1))
        b1 = rng.standard_normal(h_dim) * 0.3
        W2 = rng.standard_normal(h_dim) / np.sqrt(max(h_dim, 1))
        X_weighted = X_active * importances[None, :]
        h = np.tanh(X_weighted @ W1 + b1)
        y_raw = h @ W2

    else:
        # GAM-style: per-feature transforms summed
        y_raw = np.zeros(n_samples, dtype=np.float64)
        for k in range(n_active):
            x = X_active[:, k]
            transform = int(rng.integers(0, 3))
            if transform == 0:
                fx = x  # linear
            elif transform == 1:
                fx = 2.0 * x - 1.0  # ±1 encoded
            else:
                fx = x * (x - 0.5) * 4.0  # quadratic on binary == 0 (so noisy)
            y_raw += importances[k] * signs[k] * fx

    # Smooth clip before noise
    y_raw = 50.0 * np.tanh(y_raw / 50.0)

    if task_type == 'reg':
        y_std = float(np.std(y_raw))
        if y_std > 1e-8:
            target_r2 = float(rng.uniform(0.40, 0.90))
            noise_std = y_std * np.sqrt((1 - target_r2) / max(target_r2, 1e-6))
            y_raw += rng.standard_normal(n_samples) * noise_std
        mu, std = float(np.mean(y_raw)), float(np.std(y_raw))
        if std > 1e-8:
            y_raw = (y_raw - mu) / std
        y = y_raw.astype(np.float32)
        actual_n_classes = None
    else:
        if n_classes is None:
            n_classes = int(rng.integers(2, 11))
        if probabilistic_labels and rng.random() < 0.5:
            y = _probabilistic_label(y_raw, X, n_classes, rng)
        else:
            percentiles = np.linspace(0, 100, n_classes + 1)
            thresholds = np.nanpercentile(y_raw, percentiles[1:-1])
            y = np.digitize(y_raw, thresholds).astype(np.float32)
        actual_n_classes = n_classes

    return {
        'X': X.astype(np.float64),
        'y': y.astype(np.float32),
        'task_type': task_type,
        'n_classes': actual_n_classes,
        'filtered': False,
        'meta': {'binary_fingerprint': True, 'n_active': n_active,
                 'mean_on_rate': float(on_rates.mean())},
    }


def _generate_temporal_prior_episode(n_samples, n_features, task_type,
                                      n_classes, rng,
                                      probabilistic_labels=False):
    """Generate a temporally-structured regression episode.

    The generated rows are in TEMPORAL ORDER. Since the trainer's eval_pos
    split slices [0:eval_pos] as context and [eval_pos:] as query, this
    automatically produces a time-ordered train/test split — matching real-
    world tabular benchmarks where train comes before test in time
    (NASA_PHM, Food_Delivery_Time, Allstate_Claims_Severity, dataset_sales,
    BNG, time-of-day-keyed insurance).

    Five randomly-selected sub-modes:
      1. AR(1) features: x_t = ρ * x_{t-1} + noise (autocorrelated)
      2. Lagged y: y_t = f(x_{t-k}) — y depends on PAST features
      3. Concept drift: regression coefficients change linearly over t
      4. Trend + seasonal: features include sin/cos of time, target has trend
      5. Pure trend on y: y_t = α*t + β*x + noise (smooth target drift)

    Rationale: addresses the "no temporal prior" gap. Currently every prior
    we have generates i.i.d. rows. Real tabular benchmarks frequently have
    autocorrelation or temporal dependencies we don't model.

    Args:
        n_samples: total rows (in temporal order)
        n_features: feature count
        task_type: 'reg' or 'cls'
        n_classes: if cls
        rng: numpy random generator
        probabilistic_labels: if cls, use probabilistic labeling
    """
    # Sub-mode selection — 5 distinct temporal patterns
    mode = rng.integers(0, 5)
    # Time index in [0, 1]
    t = np.arange(n_samples, dtype=np.float64) / max(n_samples - 1, 1)

    # ---- Generate features ----
    if mode == 0:
        # AR(1) features: x_t = ρ_j * x_{t-1} + sqrt(1 - ρ²) * noise
        # Different ρ per feature column. Auto-correlated time series.
        rho = rng.uniform(0.3, 0.95, size=n_features)
        X = np.zeros((n_samples, n_features), dtype=np.float64)
        X[0] = rng.standard_normal(n_features)
        noise_scale = np.sqrt(1.0 - rho ** 2)
        for ti in range(1, n_samples):
            X[ti] = rho * X[ti - 1] + noise_scale * rng.standard_normal(n_features)
    elif mode == 1:
        # i.i.d. features, but y depends on LAGGED features
        X = rng.standard_normal((n_samples, n_features))
    elif mode == 2:
        # Mix of i.i.d. + slow-drift features
        X = rng.standard_normal((n_samples, n_features))
        # 30% of features get slow drift baseline
        n_drift_feat = max(1, int(n_features * 0.30))
        drift_cols = rng.choice(n_features, size=n_drift_feat, replace=False)
        for c in drift_cols:
            # Random walk with normalization
            drift = np.cumsum(rng.standard_normal(n_samples)) * 0.1
            drift = (drift - drift.mean()) / (drift.std() + 1e-8)
            X[:, c] = 0.5 * X[:, c] + 0.5 * drift
    elif mode == 3:
        # Trend + seasonal: dedicate first few features to time signals
        X = rng.standard_normal((n_samples, n_features))
        period = float(rng.uniform(5, 50))  # cycles within n_samples
        if n_features >= 1:
            X[:, 0] = t  # explicit time
        if n_features >= 2:
            X[:, 1] = np.sin(2 * np.pi * t * period)
        if n_features >= 3:
            X[:, 2] = np.cos(2 * np.pi * t * period)
        if n_features >= 4:
            # Slower secondary cycle
            X[:, 3] = np.sin(2 * np.pi * t * period * 0.3)
    else:  # mode == 4
        # Pure trend on y: features mostly i.i.d., y has explicit time component
        X = rng.standard_normal((n_samples, n_features))

    # ---- Generate target ----
    if mode == 0:
        # AR(1) features → linear/MLP target
        if rng.random() < 0.5:
            beta = rng.standard_normal(n_features)
            y_raw = X @ beta
        else:
            # Shallow MLP
            h_dim = int(rng.integers(4, 16))
            W1 = rng.standard_normal((n_features, h_dim)) / np.sqrt(n_features)
            b1 = rng.standard_normal(h_dim) * 0.3
            W2 = rng.standard_normal(h_dim) / np.sqrt(h_dim)
            y_raw = np.tanh(X @ W1 + b1) @ W2
    elif mode == 1:
        # Lagged dependency: y_t = f(x_{t-k})
        max_lag = int(min(5, max(1, n_samples // 50)))
        lag = int(rng.integers(1, max_lag + 1))
        # Active subset
        n_active = int(rng.integers(2, min(8, n_features) + 1))
        active = rng.choice(n_features, size=n_active, replace=False)
        beta = rng.standard_normal(n_active)
        # First `lag` rows: cold-start (use current features as proxy)
        y_raw = np.zeros(n_samples)
        y_raw[:lag] = X[:lag, active] @ beta
        y_raw[lag:] = X[:-lag, active] @ beta
        # Optional second-lag term (further history)
        if lag + 1 < n_samples // 4 and rng.random() < 0.4:
            lag2 = int(rng.integers(lag + 1, max(lag + 2, n_samples // 4) + 1))
            beta2 = rng.standard_normal(n_active) * 0.5
            y_raw[lag2:] = y_raw[lag2:] + X[:-lag2, active] @ beta2
    elif mode == 2:
        # Concept drift: regression coefficients change over t
        n_active = int(rng.integers(2, min(10, n_features) + 1))
        active = rng.choice(n_features, size=n_active, replace=False)
        beta_start = rng.standard_normal(n_active)
        beta_end = rng.standard_normal(n_active)
        y_raw = np.zeros(n_samples)
        for ti in range(n_samples):
            alpha_t = t[ti]
            beta_t = (1.0 - alpha_t) * beta_start + alpha_t * beta_end
            y_raw[ti] = X[ti, active] @ beta_t
    elif mode == 3:
        # Trend + seasonal: y has explicit time components + feature signal
        trend_amp = float(rng.uniform(0.3, 1.5))
        seasonal_amp = float(rng.uniform(0.3, 1.5))
        period_y = float(rng.uniform(5, 50))
        y_time = trend_amp * t + seasonal_amp * np.sin(2 * np.pi * t * period_y)
        # Optionally add second seasonal harmonic
        if rng.random() < 0.4:
            y_time = y_time + 0.3 * seasonal_amp * np.sin(2 * np.pi * t * period_y * 2)
        # Plus feature contribution
        if n_features > 4:
            n_active = int(rng.integers(2, min(8, n_features - 4) + 1))
            active = rng.choice(np.arange(4, n_features), size=n_active, replace=False)
            beta = rng.standard_normal(n_active) * 0.5
            y_feat = X[:, active] @ beta
        else:
            y_feat = np.zeros(n_samples)
        y_raw = y_time + y_feat
    else:  # mode == 4
        # Pure trend on y: y = α*t + β*x + small periodic
        trend_slope = float(rng.uniform(-2.0, 2.0))
        n_active = int(rng.integers(2, min(8, n_features) + 1))
        active = rng.choice(n_features, size=n_active, replace=False)
        beta = rng.standard_normal(n_active)
        y_raw = trend_slope * t + X[:, active] @ beta
        if rng.random() < 0.5:
            # Smooth periodic component
            period_y = float(rng.uniform(5, 30))
            y_raw = y_raw + 0.3 * np.sin(2 * np.pi * t * period_y)

    # Smooth y clipping before noise
    y_raw = 50.0 * np.tanh(y_raw / 50.0)

    # Calibrated noise + standardize
    if task_type == 'reg':
        y_std = float(np.std(y_raw))
        if y_std > 1e-8:
            target_r2 = float(rng.uniform(0.45, 0.92))
            noise_std = y_std * np.sqrt((1 - target_r2) / max(target_r2, 1e-6))
            y_raw = y_raw + rng.standard_normal(n_samples) * noise_std
        mu, std = float(np.mean(y_raw)), float(np.std(y_raw))
        if std > 1e-8:
            y_raw = (y_raw - mu) / std
        y = y_raw.astype(np.float32)
        actual_n_classes = None
    else:
        if n_classes is None:
            n_classes = int(rng.integers(2, 11))
        if probabilistic_labels and rng.random() < 0.5:
            y = _probabilistic_label(y_raw, X, n_classes, rng)
        else:
            percentiles = np.linspace(0, 100, n_classes + 1)
            thresholds = np.nanpercentile(y_raw, percentiles[1:-1])
            y = np.digitize(y_raw, thresholds).astype(np.float32)
        actual_n_classes = n_classes

    # Per-column winsorization for safety (light — temporal features can have
    # legitimate large values from trends, but extreme outliers from noise
    # propagation should still be clipped)
    for col in range(n_features):
        col_vals = X[:, col]
        finite = np.isfinite(col_vals)
        if finite.sum() > 4:
            sorted_vals = np.sort(col_vals[finite])
            n_fin = len(sorted_vals)
            median = sorted_vals[n_fin // 2]
            mad = np.median(np.abs(sorted_vals - median))
            if mad > 1e-12:
                X[finite, col] = np.clip(col_vals[finite],
                                          median - 8 * mad,
                                          median + 8 * mad)

    return {
        'X': X.astype(np.float64),
        'y': y.astype(np.float32),
        'task_type': task_type,
        'n_classes': actual_n_classes,
        'filtered': False,
        'meta': {'temporal_prior': True, 'mode': int(mode)},
    }


def generate_dataset_filtered(n_samples, n_features, task_type, n_classes=None,
                              rng=None, max_retries=3, quality_rules=None, **kwargs):
    """Generate a synthetic dataset with optional learnability filtering.

    Supports two filter backends:
      - ExtraTrees (default): fits a small ExtraTrees model, checks OOB score.
      - ICL (--icl-filter-model): runs a frozen LimiX forward pass on CPU,
        checks if the model can predict better than chance.  Preferred because
        it tests in-context learnability directly.

    If the dataset is unlearnable, regenerate up to max_retries times.
    """
    v4_filter = kwargs.get('augment_v4', False) and kwargs.get('v4_filter', True)
    learnability_filter = kwargs.pop('learnability_filter', False) or v4_filter
    learnability_filter_cls_min_score = kwargs.pop(
        'learnability_filter_cls_min_score', 0.60)
    learnability_filter_cls_margin = kwargs.pop(
        'learnability_filter_cls_margin', 0.10)
    learnability_filter_reg_min_score = kwargs.pop(
        'learnability_filter_reg_min_score', 0.10)

    icl_filter_model = kwargs.pop('icl_filter_model', None)
    icl_filter_cls_min_auc = kwargs.pop('icl_filter_cls_min_auc', 0.55)
    icl_filter_reg_min_r2 = kwargs.pop('icl_filter_reg_min_r2', 0.05)

    icl_scaling_filter = kwargs.pop('icl_scaling_filter', False)
    icl_scaling_min_improvement = kwargs.pop('icl_scaling_min_improvement', 0.03)

    use_icl = icl_filter_model is not None
    use_et = learnability_filter and not use_icl

    for attempt in range(max_retries + 1):
        data = generate_dataset(n_samples, n_features, task_type,
                                n_classes=n_classes, rng=rng, **kwargs)

        if quality_rules is not None and not _passes_quality_rules(
                data['X'], data['y'], task_type, quality_rules):
            data['filtered'] = True
            continue

        if not use_et and not use_icl:
            return data

        if use_icl:
            passed = _check_learnability_icl(
                data['X'], data['y'], task_type, icl_filter_model,
                cls_min_auc=icl_filter_cls_min_auc,
                reg_min_r2=icl_filter_reg_min_r2,
            )
        else:
            passed = _check_learnability(
                data['X'], data['y'], task_type,
                cls_min_score=learnability_filter_cls_min_score,
                cls_margin=learnability_filter_cls_margin,
                reg_min_score=learnability_filter_reg_min_score,
            )

        if passed and icl_scaling_filter:
            # Second gate: does more context actually help?
            passed = _check_icl_scaling(
                data['X'], data['y'], task_type,
                reg_min_score=learnability_filter_reg_min_score,
                min_improvement=icl_scaling_min_improvement,
            )

        if passed:
            return data

        data['filtered'] = True

    return data


def _apply_train_feature_augmentation(X_batch, rng, p_episode=0.6,
                                       p_col=0.40):
    """Train-time per-column feature distribution augmentation.

    Closes the synth/real distribution-shape gap. The literature consensus
    (TabICLv2, MITRA, CARTE) is to randomize per-column shape DURING training
    rather than fix it at inference. After this, the model sees skewed,
    log-shaped, sign-flipped, ranked, and quadratic columns — making heavy
    inference YJ unnecessary.

    Design (carefully chosen, see deep-think notes):
      - Nested gate: p_episode controls episode-level firing; within fired
        episodes, p_col controls per-column firing. Result is a bimodal
        distribution of "transformed-col fraction" per episode (matches
        real-data variability — some datasets have many heavy cols, some none).
      - 7 transform types (one of, weighted by realism in tabular data).
      - Cat protection: skip cols with unique_count < min(20, n_samples//50).
      - Re-standardize per-col after transform: preserves new shape but
        normalizes scale so model's internal normalizer behaves predictably.
      - Operates on each episode independently (no cross-episode leakage).
      - Returns a NEW float64 array (input not mutated).

    Args:
        X_batch: [B, N, F] feature matrix
        rng: np.random.Generator
        p_episode: per-episode firing probability
        p_col: per-column firing probability inside fired episodes

    Returns:
        X_aug: [B, N, F] augmented matrix
    """
    # Lazy import: scipy is heavy at module-load time and only needed here.
    try:
        from scipy.stats import yeojohnson
    except ImportError:
        yeojohnson = None  # YJ branch will fall through to identity

    B, N, F = X_batch.shape
    X_aug = X_batch.astype(np.float64, copy=True)

    cat_threshold_unique = min(20, max(5, N // 50))

    for b in range(B):
        if rng.random() >= p_episode:
            continue  # this episode skips augmentation entirely

        for j in range(F):
            col = X_aug[b, :, j]
            finite = np.isfinite(col)
            if finite.sum() < 4:
                continue

            # Cat protection: skip integer-valued / low-cardinality cols
            unique_count = len(np.unique(col[finite]))
            if unique_count < cat_threshold_unique:
                continue
            # Also skip if col looks integer-encoded
            if np.allclose(col[finite], np.round(col[finite]), atol=1e-6) \
                    and unique_count < N // 4:
                continue

            if rng.random() >= p_col:
                continue  # this col stays unchanged

            # Stack 1-2 transforms (90% one, 10% two — empirical compromise
            # between expressivity and producing pathological compositions).
            n_transforms = 2 if rng.random() < 0.10 else 1

            new_col = col.astype(np.float64).copy()

            for _ in range(n_transforms):
                t = rng.random()
                try:
                    if t < 0.30 and yeojohnson is not None:
                        # Yeo-Johnson with random lambda. Covers identity (1),
                        # log (0), 1/x (-1), square (2) at the edges and a
                        # smooth interpolation in between. Single most useful
                        # transform — matches the inference YJ pipeline.
                        lam = float(rng.uniform(-1.5, 1.5))
                        new_col = yeojohnson(new_col, lmbda=lam)
                    elif t < 0.45:
                        # Sign-preserving log: shrinks heavy tails while
                        # keeping zero-crossings. Common for finance/bio.
                        new_col = np.sign(new_col) * np.log(np.abs(new_col) + 1.0)
                    elif t < 0.60:
                        # Sign flip + offset: direction invariance without
                        # losing information.
                        offset = float(rng.standard_normal())
                        new_col = -new_col + offset
                    elif t < 0.70:
                        # Centered square: quadratic warp, loses sign
                        # information. Used cautiously (10% of transforms).
                        c = float(np.mean(new_col))
                        new_col = (new_col - c) ** 2
                    elif t < 0.80:
                        # Rank to [-1, 1]: strips scale and shape entirely,
                        # leaves only ordinal information. Robust for skewed.
                        ranks = np.argsort(np.argsort(new_col)).astype(np.float64)
                        new_col = 2.0 * ranks / max(N - 1, 1) - 1.0
                    elif t < 0.90:
                        # Affine: random scale + shift. Tests scale invariance
                        # of the model's internal normalizer.
                        a = float(np.exp(rng.uniform(np.log(0.1), np.log(10.0))))
                        b_off = float(rng.standard_normal() * 2.0)
                        new_col = a * new_col + b_off
                    # else (t >= 0.90): identity — column gets the gate but
                    # no actual transform. Keeps some "raw" cols even in
                    # fired episodes, prevents over-smoothing.
                except Exception:
                    # Numerical failure: skip this transform but keep the
                    # column (don't lose the data).
                    continue

                # Sanity: replace bad values
                new_col = np.nan_to_num(new_col, nan=0.0,
                                         posinf=0.0, neginf=0.0)

                # CRITICAL: re-standardize per-col after transform. YJ + log +
                # exp produce wildly-different scales; encoder assumes z-scored
                # input. This preserves NEW shape but normalizes scale.
                mu = float(np.mean(new_col))
                sd = float(np.std(new_col))
                if sd > 1e-8:
                    new_col = (new_col - mu) / sd
                else:
                    # Degenerate (all same value after transform). Skip the
                    # transform by reverting to original col so the cell isn't
                    # zeroed-out (which would break encoder normalization).
                    new_col = col.astype(np.float64).copy()
                    break

            X_aug[b, :, j] = new_col

    return X_aug


def generate_batch(batch_size, n_samples, n_features, task_type,
                   n_classes=None, rng=None, augment=False, augment_v3=False,
                   rich_reg_targets=True, scale_variation=True,
                   augment_v4=False, v4_filter=True, v4_no_edge_noise=True,
                   synth_v5=False, synth_v5_denoise=True,
                   synth_v5_declone=True, synth_v5_mixture=False,
                   reg_prior_prob=0.0, reg_denoise=False,
                   reg_deterministic_prob=0.20,
                   reg_dense=False,
                   scm_prior=False, scm_prior_prob=0.5,
                   probabilistic_labels=False, nominal_categoricals=False,
                   enhanced_missingness=False,
                   clean_lowdim_prob=0.0,
                   tree_prior_prob=0.0, lookup_prior_prob=0.0,
                   quadratic_surface_prob=0.0, sparse_nonlinear_prob=0.0,
                   gp_prior_prob=0.0,
                   cat_dominant_prob=0.0,
                   binary_fingerprint_prob=0.0,
                   temporal_prior_prob=0.0,
                   train_feature_augment_prob=0.0,
                   context_missingness_prob=0.0,
                   realistic_augmentation_prob=0.0,
                   y_transform_prob=0.0,
                   cap_injection_prob=0.0,
                   heavy_tail_prior_prob=0.0,
                   pareto_importance_prob=0.0,
                   latent_factor_prob=0.0,
                   high_cap_prob=0.0,
                   low_unique_y_prob=0.0,
                   learnability_filter=False,
                   learnability_filter_cls_min_score=0.60,
                   learnability_filter_cls_margin=0.10,
                   learnability_filter_reg_min_score=0.10,
                   icl_filter_model=None,
                   icl_filter_cls_min_auc=0.55,
                   icl_filter_reg_min_r2=0.05,
                   icl_scaling_filter=False,
                   icl_scaling_min_improvement=0.03,
                   quality_rules=None, filter_max_retries=3):
    """Generate a batch of synthetic datasets with the same dimensions.

    Args:
        batch_size: number of datasets in the batch
        n_samples: samples per dataset
        n_features: features per dataset
        task_type: 'cls' or 'reg'
        n_classes: number of classes (for cls)
        rng: numpy random generator
        augment: if True, apply synth_v2 augmentations
        augment_v3: if True, apply synth_v3 augmentations
        rich_reg_targets: if True AND augment_v3, rich regression targets
        scale_variation: if True AND augment_v3, random target scale
        augment_v4: if True, apply synth_v4 improvements
        v4_filter: if True AND augment_v4, ExtraTrees filtering
        learnability_filter: if True, ExtraTrees filtering (independent of synth_v4)
        learnability_filter_cls_min_score: floor on classification OOB score
        learnability_filter_cls_margin: minimum margin above chance for classification
        learnability_filter_reg_min_score: minimum regression OOB R2 to keep dataset
        v4_no_edge_noise: if True AND augment_v4, skip edge noise
        synth_v5: if True, apply synth_v5 SCM improvements
        synth_v5_denoise: if True AND synth_v5, reduce noise levels
        synth_v5_declone: if True AND synth_v5, de-clone multi-dim expansion
        synth_v5_mixture: if True, randomly select mode per dataset
        reg_prior_prob: probability of using regression-specific prior generator
                        for regression episodes (dense linear, GAM, etc.)
        reg_denoise: if True, reduce noise levels for regression episodes
        reg_deterministic_prob: probability that a regression-prior episode has
                   target_r2=1.0 (deterministic target), independent of reg_dense
        reg_dense: if True, dense-signal regression mode (flat importances,
                   fewer noise features, more SCM parents, higher R²)
        scm_prior: if True, enable TabICL prior generator
        scm_prior_prob: per-dataset probability of using TabICL prior
        quality_rules: optional mined quality rules dict. If provided, datasets
                       failing the rules are regenerated up to filter_max_retries.
        filter_max_retries: max retries for filtered dataset regeneration

    Returns:
        X_batch: np.ndarray [batch_size, n_samples, n_features]
        y_batch: np.ndarray [batch_size, n_samples]
        n_classes: int or None
    """
    if rng is None:
        rng = np.random.default_rng()

    X_list = []
    y_list = []
    actual_n_classes = n_classes

    for _ in range(batch_size):
        # --- Clean low-dim regime ---
        # Simple 5-30 feature datasets with high categorical fraction,
        # low noise. Covers Amazon-like tabular tasks.
        if clean_lowdim_prob > 0 and rng.random() < clean_lowdim_prob:
            data = _generate_clean_lowdim_episode(
                n_samples, n_features, task_type, n_classes, rng,
                probabilistic_labels=probabilistic_labels,
                nominal_categoricals=nominal_categoricals)
        # --- Tree-ensemble prior ---
        # Piecewise-constant targets from random decision tree ensembles.
        elif tree_prior_prob > 0 and rng.random() < tree_prior_prob:
            data = _generate_tree_prior_episode(
                n_samples, n_features, task_type, n_classes, rng,
                probabilistic_labels=probabilistic_labels)
        # --- Categorical lookup prior ---
        # Entity-lookup data: y = f(entity_id) + noise.
        elif lookup_prior_prob > 0 and rng.random() < lookup_prior_prob:
            data = _generate_lookup_prior_episode(
                n_samples, n_features, task_type, n_classes, rng,
                probabilistic_labels=probabilistic_labels)
        # --- Quadratic response surface prior ---
        elif quadratic_surface_prob > 0 and rng.random() < quadratic_surface_prob:
            data = _generate_quadratic_surface_episode(
                n_samples, n_features, task_type, n_classes, rng,
                reg_denoise=reg_denoise,
                probabilistic_labels=probabilistic_labels)
        # --- Sparse nonlinear high-dim prior ---
        elif sparse_nonlinear_prob > 0 and rng.random() < sparse_nonlinear_prob:
            data = _generate_sparse_nonlinear_episode(
                n_samples, n_features, task_type, n_classes, rng,
                reg_denoise=reg_denoise,
                probabilistic_labels=probabilistic_labels)
        # --- GP smooth function prior ---
        # y sampled from Gaussian Process with RBF/Matern kernels.
        # Produces smooth joint multivariate functions (sulfur, debutanizer,
        # space_ga, kin8nm, houses, physiochemical_protein patterns).
        elif gp_prior_prob > 0 and rng.random() < gp_prior_prob:
            data = _generate_gp_prior_episode(
                n_samples, n_features, task_type, n_classes, rng,
                reg_denoise=reg_denoise,
                probabilistic_labels=probabilistic_labels)
        # --- Categorical-dominant prior ---
        # 60-95% of cols are categorical (cardinalities 2-50, biased small).
        # Targets cat-heavy benchmarks (Ailerons, Buzzinsocialmedia,
        # Food_Delivery_Time, MIP-2016) where SCM under-generates.
        elif cat_dominant_prob > 0 and rng.random() < cat_dominant_prob:
            data = _generate_cat_dominant_episode(
                n_samples, n_features, task_type, n_classes, rng,
                probabilistic_labels=probabilistic_labels)
        # --- Binary fingerprint prior ---
        # All-binary high-dim data with sparse signal. Targets QSAR-TID-11
        # archetype (chemical fingerprints, drug-binding affinity).
        elif binary_fingerprint_prob > 0 and rng.random() < binary_fingerprint_prob:
            data = _generate_binary_fingerprint_episode(
                n_samples, n_features, task_type, n_classes, rng,
                probabilistic_labels=probabilistic_labels)
        # --- Temporal prior ---
        # Rows generated in temporal order with one of 5 patterns:
        # AR(1) autocorrelated features, lagged-y, concept drift, trend+seasonal,
        # pure trend on y. The trainer's eval_pos split becomes a time-ordered
        # train/test split automatically. Addresses the "no temporal prior" gap
        # (Food_Delivery_Time, NASA_PHM, Allstate, dataset_sales, etc.).
        elif temporal_prior_prob > 0 and rng.random() < temporal_prior_prob:
            data = _generate_temporal_prior_episode(
                n_samples, n_features, task_type, n_classes, rng,
                probabilistic_labels=probabilistic_labels)
        # TabICL prior: MLP/Tree SCM with rich activations and meta-distribution
        # HP sampling. For cls, applies Reg2Cls to convert continuous targets to
        # class labels. For reg, uses raw SCM output (standardized continuous y).
        elif (scm_prior and rng.random() < scm_prior_prob):
            from synthefy_tabular.training.scm_prior_generator import generate_scm_prior_dataset
            data = generate_scm_prior_dataset(
                n_samples, n_features, task_type,
                n_classes=n_classes, rng=rng)
        # For regression, use the regression-specific prior with probability
        # reg_prior_prob. These cover real-world regimes (dense linear, GAM,
        # interactions) that the SCM generator under-represents.
        elif (task_type == 'reg' and reg_prior_prob > 0
                and rng.random() < reg_prior_prob):
            # 50/50 split: rich SCM X vs pure Gaussian X.
            # SCM X teaches: messy features can have clean additive targets.
            # Gaussian X teaches: clean PCA'd data needs simple Ridge/Lasso/GAM.
            if rng.random() < 0.5:
                base_data = generate_dataset_filtered(
                    n_samples, n_features, task_type,
                    n_classes=n_classes, rng=rng, max_retries=filter_max_retries,
                    quality_rules=quality_rules,
                    learnability_filter=learnability_filter,
                    learnability_filter_cls_min_score=learnability_filter_cls_min_score,
                    learnability_filter_cls_margin=learnability_filter_cls_margin,
                    learnability_filter_reg_min_score=learnability_filter_reg_min_score,
                    icl_filter_model=icl_filter_model,
                    icl_filter_cls_min_auc=icl_filter_cls_min_auc,
                    icl_filter_reg_min_r2=icl_filter_reg_min_r2,
                    icl_scaling_filter=icl_scaling_filter,
                    icl_scaling_min_improvement=icl_scaling_min_improvement,
                    augment=augment,
                    augment_v3=augment_v3,
                    rich_reg_targets=False,
                    scale_variation=False,
                    augment_v4=augment_v4, v4_filter=v4_filter,
                    v4_no_edge_noise=v4_no_edge_noise,
                    synth_v5=synth_v5,
                    synth_v5_denoise=synth_v5_denoise,
                    synth_v5_declone=synth_v5_declone,
                    synth_v5_mixture=synth_v5_mixture,
                    reg_denoise=reg_denoise,
                    reg_dense=reg_dense,
                    nominal_categoricals=nominal_categoricals,
                    enhanced_missingness=enhanced_missingness)
                data = _generate_regression_prior(
                    n_samples, n_features, rng,
                    reg_denoise=reg_denoise,
                    reg_deterministic_prob=reg_deterministic_prob,
                    reg_dense=reg_dense,
                    pareto_importance_prob=pareto_importance_prob,
                    latent_factor_prob=latent_factor_prob,
                    X_scm=base_data['X'],
                    X_scm_target=base_data.get('X_target'))
            else:
                data = _generate_regression_prior(n_samples, n_features, rng,
                                                  reg_denoise=reg_denoise,
                                                  reg_deterministic_prob=reg_deterministic_prob,
                                                  reg_dense=reg_dense,
                                                  pareto_importance_prob=pareto_importance_prob,
                                                  latent_factor_prob=latent_factor_prob)
        else:
            data = generate_dataset_filtered(
                n_samples, n_features, task_type,
                n_classes=n_classes, rng=rng, max_retries=filter_max_retries,
                quality_rules=quality_rules,
                learnability_filter=learnability_filter,
                learnability_filter_cls_min_score=learnability_filter_cls_min_score,
                learnability_filter_cls_margin=learnability_filter_cls_margin,
                learnability_filter_reg_min_score=learnability_filter_reg_min_score,
                icl_filter_model=icl_filter_model,
                icl_filter_cls_min_auc=icl_filter_cls_min_auc,
                icl_filter_reg_min_r2=icl_filter_reg_min_r2,
                icl_scaling_filter=icl_scaling_filter,
                icl_scaling_min_improvement=icl_scaling_min_improvement,
                augment=augment,
                augment_v3=augment_v3,
                rich_reg_targets=rich_reg_targets,
                scale_variation=scale_variation,
                augment_v4=augment_v4, v4_filter=v4_filter,
                v4_no_edge_noise=v4_no_edge_noise,
                synth_v5=synth_v5,
                synth_v5_denoise=synth_v5_denoise,
                synth_v5_declone=synth_v5_declone,
                synth_v5_mixture=synth_v5_mixture,
                reg_denoise=reg_denoise,
                reg_dense=reg_dense,
                probabilistic_labels=probabilistic_labels,
                nominal_categoricals=nominal_categoricals,
                enhanced_missingness=enhanced_missingness)
        X_list.append(data['X'])
        y_list.append(data['y'])
        if actual_n_classes is None and task_type == 'cls':
            actual_n_classes = data['n_classes']

    # Debug: detect shape mismatch before np.stack fails
    shapes = [x.shape for x in X_list]
    if len(set(shapes)) > 1:
        y_shapes = [y.shape for y in y_list]
        raise ValueError(
            f"X shape mismatch in batch: {shapes}, y shapes: {y_shapes}, "
            f"n_samples={n_samples}, n_features={n_features}, "
            f"task_type={task_type}, meta={[d.get('meta',{}) for d in [data]]}")

    X_batch = np.stack(X_list, axis=0)
    y_batch = np.stack(y_list, axis=0)

    # Train-time feature distribution augmentation (V12 audit-driven addition).
    # Closes the train/inference distribution-shape gap that currently requires
    # heavy inference normalization (--yj-skew-threshold 10, poly 10, etc.).
    # Per-episode gate inside per-column gate: matches the bimodal real-world
    # pattern where some datasets have heavy-tailed cols and some have none.
    # Skips integer-valued / low-unique cols (cat protection) — same heuristic
    # used by inference cat detection.
    if train_feature_augment_prob > 0.0:
        X_batch = _apply_train_feature_augmentation(
            X_batch, rng, p_episode=train_feature_augment_prob)

    # Context missingness augmentation: inject 1-8% random NaN cells across
    # the full feature matrix. Real benchmark datasets have NaN in both
    # train and test rows, but CCMM masking only covers query rows.
    # Applied in prefetch workers (not trainer) so the compiled graph sees
    # NaN inputs from the first step, avoiding recompilation mid-training.
    if context_missingness_prob > 0 and rng.random() < context_missingness_prob:
        miss_ratio = rng.uniform(0.01, 0.08)
        B, N, F = X_batch.shape
        nan_mask = rng.random((B, N, F)) < miss_ratio
        X_batch = X_batch.astype(np.float64, copy=True)
        X_batch[nan_mask] = np.nan

    # Suppress numpy warnings from realistic augmentation (nanstd on
    # columns with ≤1 non-NaN value, exp overflow before clipping).
    import warnings as _warnings
    # Realistic augmentation: applies to all priors post-generation.
    # Addresses three gaps vs real data identified in benchmark analysis:
    #   1. Heavy-tailed targets (real: skew 1-7, kurtosis 5-80; synthetic: ~0)
    #   2. Correlated feature groups (real: mean_corr 0.1-0.5; new priors: ~0.03)
    #   3. Skewed feature distributions (real features are often log-normal)
    if realistic_augmentation_prob > 0:
        B, N, F = X_batch.shape
        _warnings.filterwarnings('ignore', category=RuntimeWarning)
        for b in range(B):
            if rng.random() >= realistic_augmentation_prob:
                continue

            # --- Heavy-tailed target transform (40% of augmented episodes) ---
            # Real datasets like sulfur (skew=6.8, kurt=77), debutanizer
            # (skew=1.7, kurt=4.5), space_ga (skew=-1.2, kurt=11.5) have
            # targets far from Gaussian. Our synthetic targets are near-Gaussian.
            if rng.random() < 0.4:
                y = y_batch[b].copy()
                transform = int(rng.integers(0, 4))
                if transform == 0:
                    # Exponential right-skew: y → exp(y * scale)
                    scale = rng.uniform(0.3, 1.0)
                    y = np.exp(np.clip(y * scale, -20, 20))
                elif transform == 1:
                    # Power transform: y → sign(y) * |y|^p
                    power = rng.uniform(1.5, 3.0)
                    y = np.sign(y) * np.abs(y) ** power
                elif transform == 2:
                    # Log-normal: y → exp(y) (always positive)
                    y = np.exp(np.clip(y * rng.uniform(0.5, 1.5), -20, 20))
                else:
                    # Asymmetric clip: heavy right tail
                    clip_lo = rng.uniform(-3, -1)
                    y = np.clip(y, clip_lo, None)
                    y = y ** 2 * np.sign(y - np.median(y))

                # Soft clip extreme values, re-standardize
                y = 50.0 * np.tanh(y / 50.0)
                mu, std = np.nanmean(y), np.nanstd(y)
                if std > 1e-8:
                    y = (y - mu) / std
                y_batch[b] = y.astype(np.float32)

            # --- Feature correlation injection (50% of augmented episodes) ---
            # Real datasets have correlated feature groups (mean_corr 0.1-0.5).
            # Our GP/quadratic/sparse priors generate near-independent features.
            if rng.random() < 0.5 and F >= 4:
                X = X_batch[b].copy()
                n_groups = int(rng.integers(2, min(5, F // 2 + 1)))
                group_size = max(2, F // n_groups)
                for g in range(n_groups):
                    start = g * group_size
                    end = min(start + group_size, F)
                    if end - start < 2:
                        continue
                    rho = rng.uniform(0.2, 0.7)
                    shared = rng.standard_normal(N)
                    for j in range(start, end):
                        col = X[:, j]
                        col_std = np.nanstd(col)
                        if col_std > 1e-8:
                            col_norm = (col - np.nanmean(col)) / col_std
                            X[:, j] = rho * shared + np.sqrt(1 - rho**2) * col_norm
                X_batch[b] = X.astype(X_batch.dtype)

            # --- Skewed feature distributions (30% of augmented episodes) ---
            # Real features are often log-normal, power-law, or bounded.
            if rng.random() < 0.3 and F >= 2:
                X = X_batch[b].copy()
                n_transform = int(rng.integers(1, max(2, F // 3 + 1)))
                cols = rng.choice(F, size=n_transform, replace=False)
                for col_idx in cols:
                    col = X[:, col_idx]
                    if np.nanstd(col) < 1e-8:
                        continue
                    ft = int(rng.integers(0, 3))
                    if ft == 0:
                        # Log-normal feature (clip input to prevent overflow)
                        X[:, col_idx] = np.exp(np.clip(col * rng.uniform(0.3, 0.8), -20, 20))
                    elif ft == 1:
                        # Squared feature (always positive, heavy tail)
                        X[:, col_idx] = col ** 2
                    else:
                        # Sigmoid-bounded feature
                        X[:, col_idx] = 1.0 / (1.0 + np.exp(-np.clip(col * rng.uniform(0.5, 2.0), -20, 20)))
                X_batch[b] = X.astype(X_batch.dtype)

    # --- Realistic y-target transforms ---
    # Real-world regression targets often have distributions our smooth SCM /
    # regression priors never produce. These transforms expose the model to
    # integer counts (stock_fardamento, colleges), censored values (boston MEDV
    # capped at 50, MIP-2016 timeout saturation), ordinal ratings (sensory
    # 11 unique values, half-integer spacing), bounded rating averages
    # (Goodreads 0-5 with quarter-step), and zero-inflated counts (socmob).
    # Each transform is followed by re-standardization (mean=0, std=1).
    if y_transform_prob > 0 and task_type == 'reg':
        _warnings.filterwarnings('ignore', category=RuntimeWarning)
        B = y_batch.shape[0]
        for b in range(B):
            if rng.random() >= y_transform_prob:
                continue
            y = y_batch[b].astype(np.float64, copy=True)
            # Work in a bounded pre-transform space to avoid exp-overflow
            y = np.clip(y, -6.0, 6.0)
            # Less aggressive mix: most transforms keep y continuous.
            # Rounding only happens in ~30% of transforms (15% integer counts +
            # 15% fine-ordinal), and rounding granularity is finer than before.
            t = rng.random()

            if t < 0.25:
                # Heavy-skew CONTINUOUS (no rounding) — most common for real
                # regression targets: log-normal scientific measurements,
                # finance, biology assays. No discretization.
                scale = rng.uniform(0.3, 0.8)
                y = np.exp(np.clip(y * scale, -20.0, 20.0))

            elif t < 0.45:
                # Censored / saturated upper tail (boston MEDV capped at 50,
                # MIP-2016 solve-time timeout at 72000). No rounding, just
                # percentile-based clipping.
                cap_pct = float(rng.uniform(80.0, 97.0))
                cap_val = float(np.percentile(y, cap_pct))
                y = np.minimum(y, cap_val)
                if rng.random() < 0.4:
                    floor_pct = float(rng.uniform(3.0, 15.0))
                    floor_val = float(np.percentile(y, floor_pct))
                    y = np.maximum(y, floor_val)

            elif t < 0.60:
                # Ordinal with many levels (15-40, finer granularity than
                # aggressive ratings). Simulates scoring systems with moderate
                # resolution like 0-100 scores or multi-panel averaged ratings.
                n_levels = int(rng.integers(15, 41))
                ranks = np.argsort(np.argsort(y)).astype(np.float64)
                y = np.floor(ranks / max(len(ranks), 1) * n_levels)
                step = float(rng.choice([0.1, 0.25, 0.5, 1.0]))
                offset = rng.uniform(-5.0, 15.0)
                y = y * step + offset

            elif t < 0.75:
                # Bounded rating average (Goodreads 0-5 floats that are means
                # of discrete ratings). Fine quantization: 0.05-0.5 step.
                lo = float(rng.uniform(0.0, 1.0))
                hi = float(rng.uniform(3.0, 5.0))
                y_std = max(float(np.std(y)), 1e-8)
                y = (y - float(np.mean(y))) / y_std
                y = np.clip(y * 0.3 + (lo + hi) / 2.0, lo, hi)
                step = float(rng.choice([0.05, 0.1, 0.25, 0.5]))
                y = np.round(y / step) * step

            elif t < 0.85:
                # Zero-inflated with CONTINUOUS positive values (socmob rates:
                # many zeros + long tail of continuous positive values). No
                # rounding of positives — preserves real-world structure.
                zero_prob = rng.uniform(0.10, 0.30)
                y_pos = np.exp(np.clip(y * rng.uniform(0.3, 0.7), -20.0, 20.0))
                zero_mask = rng.random(len(y)) < zero_prob
                y = np.where(zero_mask, 0.0, y_pos)

            else:
                # Integer counts (only 15% of transforms) — true count data
                # like stock_fardamento02 (values 1..833) or colleges
                # (enrollment counts). Rounds to integers.
                scale = rng.uniform(0.4, 0.9)
                base = rng.uniform(1.0, 20.0)
                y = np.round(np.exp(np.clip(y * scale, -20.0, 20.0)) * base)

            # Re-standardize: the trainer expects normalized y (handled
            # per-episode via context-only stats in _prepare_batch, but also
            # ensure finite values here).
            y = np.nan_to_num(y, nan=0.0, posinf=0.0, neginf=0.0)
            mu = float(np.mean(y))
            sd = float(np.std(y))
            if sd > 1e-8:
                y = (y - mu) / sd
            else:
                y = np.zeros_like(y)
            y_batch[b] = y.astype(np.float32)

    # --- Cap injection (minimal saturation/censoring) ---
    # Real regression datasets often have censored targets: boston MEDV capped
    # at 50 (10% of values hit the cap), MIP-2016 solver timeouts saturated
    # at 73296, ratings bounded at 5, etc. Our models fail to reach these caps
    # in prediction. Injecting saturation in training teaches the model that
    # "many samples can have identical y value" which fixes the inability to
    # predict near-cap values.
    #
    # Unlike y_transform, this does NOT change the shape/values/range of y —
    # just clips top/bottom quantile tails to a single value. Minimal
    # distributional disturbance.

    # --- Low-unique y (discrete-regression) ---
    # Round y to 5-15 unique levels to model datasets with bounded discrete
    # targets (Wine_Quality: 6 levels, sensory: 1-10 ratings, ordinal scales).
    # Runs BEFORE cap_injection so cap patterns can stack on top of a discrete
    # target distribution if both fire. Re-standardizes for episode sanity.
    if low_unique_y_prob > 0 and task_type == 'reg':
        B = y_batch.shape[0]
        for b in range(B):
            if rng.random() >= low_unique_y_prob:
                continue
            y = y_batch[b].astype(np.float64, copy=True)
            n_levels = int(rng.integers(5, 16))
            # Quantile-bucket so each level has roughly equal mass; preserves
            # the original target ordering and rough scale.
            edges = np.quantile(y, np.linspace(0, 1, n_levels + 1))
            edges[0] -= 1e-6  # ensure inclusive lower bound
            edges[-1] += 1e-6
            level_idx = np.clip(np.searchsorted(edges, y, side='right') - 1, 0, n_levels - 1)
            # Use bucket midpoints as the level value.
            mids = 0.5 * (edges[:-1] + edges[1:])
            y = mids[level_idx]
            mu = float(np.mean(y))
            sd = float(np.std(y))
            if sd > 1e-8:
                y = (y - mu) / sd
            y_batch[b] = y.astype(np.float32)

    if cap_injection_prob > 0 and task_type == 'reg':
        B = y_batch.shape[0]
        for b in range(B):
            if rng.random() >= cap_injection_prob:
                continue
            y = y_batch[b].astype(np.float64, copy=True)
            # High-fraction censoring branch (timeout pattern — MIP-2016 runtime
            # cap, SAT11 algo cutoff). When triggered, the cap percentile is
            # drawn from [50, 80] so 20-50% of values sit at the cap, vs
            # the default [80, 97] saturation pattern.
            use_high_cap = (high_cap_prob > 0 and rng.random() < high_cap_prob)
            # Upper cap (more common: boston, MIP-2016, Goodreads)
            if rng.random() < 0.75:
                if use_high_cap:
                    cap_pct = float(rng.uniform(50.0, 80.0))
                else:
                    cap_pct = float(rng.uniform(80.0, 97.0))
                cap_val = float(np.percentile(y, cap_pct))
                y = np.minimum(y, cap_val)
            # Lower floor (less common)
            if rng.random() < 0.30:
                if use_high_cap:
                    floor_pct = float(rng.uniform(20.0, 50.0))
                else:
                    floor_pct = float(rng.uniform(3.0, 20.0))
                floor_val = float(np.percentile(y, floor_pct))
                y = np.maximum(y, floor_val)
            # Re-standardize (context-only re-norm happens in trainer, but
            # ensure episode-level sanity)
            mu = float(np.mean(y))
            sd = float(np.std(y))
            if sd > 1e-8:
                y = (y - mu) / sd
            y_batch[b] = y.astype(np.float32)

    # --- Heavy-tail y priors (gated, continuous — no rounding) ---
    # Addresses real benchmarks our model can't predict: stock_fardamento02
    # (skew 17.7), CPS1988 (log-normal wages), Food_Delivery_Time (Poisson-ish),
    # sulfur (skew 6.6 bounded). Current synthetic y is near-Gaussian after
    # normalization — these transforms expose the model to heavy right tails
    # WITHOUT aggressive rounding (unlike failed y_transform experiment).
    #
    # Transforms are continuous-only:
    #   1. Log-normal: y = exp(y * scale) — natural heavy right tail
    #   2. Pareto-tailed: y = y + pareto(alpha) * sign(y) * scale
    #   3. Stronger outlier injection: 5-10% at 3-15x scale (vs default 1-5% at 3-8x)
    #
    # Re-standardized after transform to keep trainer's context-norm sane.
    if heavy_tail_prior_prob > 0 and task_type == 'reg':
        B = y_batch.shape[0]
        for b in range(B):
            if rng.random() >= heavy_tail_prior_prob:
                continue
            y = y_batch[b].astype(np.float64, copy=True)
            y = np.clip(y, -6.0, 6.0)
            t = rng.random()

            if t < 0.30:
                # Log-normal right-skew (continuous, no rounding)
                # Most common heavy-tail pattern in real data (income, counts,
                # solve times, prices). Scale chosen for skew ∈ ~[2, 8].
                scale = rng.uniform(0.4, 1.0)
                y = np.exp(y * scale)

            elif t < 0.55:
                # Pareto-tailed additive noise: inject rare extreme values
                # on top of existing signal. Targets sulfur/debutanizer style
                # heavy-right-tail distributions.
                alpha = rng.uniform(1.5, 4.0)  # lower alpha = heavier tail
                y_std = max(float(np.std(y)), 1e-8)
                tail_noise = rng.pareto(alpha, size=len(y)) * y_std
                # Apply to positive side only, randomly
                sign_mask = rng.choice([1.0, -1.0], size=len(y))
                y = y + tail_noise * sign_mask * rng.uniform(0.3, 0.8)

            elif t < 0.80:
                # Stronger outlier injection: larger fraction, larger magnitude
                # Represents rare extreme events (measurement errors, tail events)
                y_std = max(float(np.std(y)), 1e-8)
                n_out = max(1, int(len(y) * rng.uniform(0.05, 0.10)))
                out_idx = rng.choice(len(y), size=n_out, replace=False)
                scale = rng.uniform(3.0, 15.0)
                y[out_idx] = y[out_idx] + rng.standard_normal(n_out) * y_std * scale

            else:
                # Contaminated extreme mixture: 0.5-2% of rows × 30-100× scale.
                # Targets the actual Job_Profitability / catastrophic-outlier
                # pattern: 99% near-Gaussian bulk + tiny fraction of values
                # several orders of magnitude away from the bulk. Existing
                # branches above produce too-many / too-mild outliers vs the
                # real failure mode (Job_Profitability has ~0.2% of rows at
                # ~1500× the bulk std). One-sided to mimic real data where
                # the catastrophic tail is usually directional.
                y_std = max(float(np.std(y)), 1e-8)
                n_out = max(1, int(len(y) * rng.uniform(0.005, 0.02)))
                out_idx = rng.choice(len(y), size=n_out, replace=False)
                sign = rng.choice([-1.0, 1.0])
                scale = rng.uniform(30.0, 100.0)
                y[out_idx] = y[out_idx] + sign * np.abs(rng.standard_normal(n_out)) * y_std * scale

            # Re-standardize for normalized-y contract
            y = np.nan_to_num(y, nan=0.0, posinf=0.0, neginf=0.0)
            mu = float(np.mean(y))
            sd = float(np.std(y))
            if sd > 1e-8:
                y = (y - mu) / sd
            y_batch[b] = y.astype(np.float32)

    return X_batch, y_batch, actual_n_classes

