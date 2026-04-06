"""
 Decision Tree Interactive Visualizer
========================================
An interactive Streamlit app to explore how Decision Trees
partition data, choose splits, grow, and make predictions.
"""

import streamlit as st
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.colors import ListedColormap, to_rgba
from sklearn.datasets import make_moons, make_circles, make_classification, make_blobs
from sklearn.tree import DecisionTreeClassifier, export_text
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import graphviz
import warnings
warnings.filterwarnings("ignore")

#  Page Config 
st.set_page_config(
    page_title=" Decision Tree Visualizer",
    page_icon="",
    layout="wide",
    initial_sidebar_state="expanded",
)

#  Custom CSS for premium look 
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&display=swap');

html, body, [class*="st-"] {
    font-family: 'Inter', sans-serif;
}

.main-title {
    background: linear-gradient(135deg, #0f9b58 0%, #1a73e8 50%, #7c3aed 100%);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    font-size: 2.8rem;
    font-weight: 800;
    letter-spacing: -0.02em;
    margin-bottom: 0;
    line-height: 1.1;
}

.sub-title {
    color: #6b7280;
    font-size: 1.1rem;
    font-weight: 400;
    margin-top: 4px;
    margin-bottom: 24px;
}

.concept-box {
    background: linear-gradient(135deg, #f0fdf4 0%, #ecfdf5 100%);
    border-left: 4px solid #0f9b58;
    border-radius: 0 12px 12px 0;
    padding: 16px 20px;
    margin: 12px 0;
    font-size: 0.92rem;
    line-height: 1.6;
    color: #1e3a2f;
}

.observe-box {
    background: linear-gradient(135deg, #eff6ff 0%, #e8f0fe 100%);
    border-left: 4px solid #1a73e8;
    border-radius: 0 12px 12px 0;
    padding: 14px 18px;
    margin: 10px 0;
    font-size: 0.9rem;
    line-height: 1.5;
    color: #1e3a5f;
}

.metric-card {
    background: linear-gradient(135deg, #ffffff 0%, #f9fafb 100%);
    border: 1px solid #e5e7eb;
    border-radius: 16px;
    padding: 20px;
    text-align: center;
    box-shadow: 0 1px 3px rgba(0,0,0,0.06);
    transition: transform 0.2s, box-shadow 0.2s;
}

.metric-card:hover {
    transform: translateY(-2px);
    box-shadow: 0 4px 12px rgba(0,0,0,0.1);
}

.metric-value {
    font-size: 2rem;
    font-weight: 700;
    background: linear-gradient(135deg, #0f9b58, #1a73e8);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
}

.metric-label {
    font-size: 0.82rem;
    color: #6b7280;
    font-weight: 500;
    text-transform: uppercase;
    letter-spacing: 0.05em;
    margin-top: 4px;
}

.stTabs [data-baseweb="tab-list"] {
    gap: 8px;
}

.stTabs [data-baseweb="tab"] {
    border-radius: 10px 10px 0 0;
    padding: 10px 20px;
    font-weight: 600;
}

div[data-testid="stSidebar"] {
    background: linear-gradient(180deg, #0a1628 0%, #132040 100%);
}

div[data-testid="stSidebar"] label,
div[data-testid="stSidebar"] .stMarkdown p,
div[data-testid="stSidebar"] .stMarkdown h1,
div[data-testid="stSidebar"] .stMarkdown h2,
div[data-testid="stSidebar"] .stMarkdown h3,
div[data-testid="stSidebar"] span {
    color: #e2e8f0 !important;
}
</style>
""", unsafe_allow_html=True)

#  Helper Functions 

def generate_dataset(name, n_samples=300, noise_level=0.0, random_state=42):
    """Generate 2D classification datasets."""
    if name == "Moons":
        X, y = make_moons(n_samples=n_samples, noise=0.2 + noise_level * 0.3, random_state=random_state)
    elif name == "Circles":
        X, y = make_circles(n_samples=n_samples, noise=0.1 + noise_level * 0.25, factor=0.5, random_state=random_state)
    elif name == "Blobs":
        X, y = make_blobs(n_samples=n_samples, centers=3, cluster_std=1.0 + noise_level * 2.0, random_state=random_state)
        y = (y > 0).astype(int)
    elif name == "Linear":
        X, y = make_classification(
            n_samples=n_samples, n_features=2, n_redundant=0,
            n_informative=2, n_clusters_per_class=1,
            flip_y=noise_level * 0.15, random_state=random_state
        )
    else:  # XOR-like
        rng = np.random.RandomState(random_state)
        X = rng.randn(n_samples, 2)
        y = ((X[:, 0] * X[:, 1]) > 0).astype(int)
        X += rng.randn(n_samples, 2) * (0.3 + noise_level * 0.5)
    return X, y


def compute_gini(y):
    """Compute Gini impurity."""
    if len(y) == 0:
        return 0.0
    classes, counts = np.unique(y, return_counts=True)
    probs = counts / len(y)
    return 1.0 - np.sum(probs ** 2)


def compute_entropy(y):
    """Compute Shannon entropy."""
    if len(y) == 0:
        return 0.0
    classes, counts = np.unique(y, return_counts=True)
    probs = counts / len(y)
    probs = probs[probs > 0]
    return -np.sum(probs * np.log2(probs))


def compute_impurity_gain(X, y, feature, threshold, criterion='gini'):
    """Compute impurity gain for a given split."""
    impurity_fn = compute_gini if criterion == 'gini' else compute_entropy
    left_mask = X[:, feature] <= threshold
    right_mask = ~left_mask
    y_left, y_right = y[left_mask], y[right_mask]

    if len(y_left) == 0 or len(y_right) == 0:
        return 0.0

    parent_impurity = impurity_fn(y)
    n = len(y)
    child_impurity = (len(y_left) / n) * impurity_fn(y_left) + (len(y_right) / n) * impurity_fn(y_right)
    return parent_impurity - child_impurity


def find_best_split(X, y, criterion='gini'):
    """Find the best feature and threshold for splitting."""
    best_gain = -1
    best_feature = 0
    best_threshold = 0
    n_features = X.shape[1]

    all_splits = []

    for feature in range(n_features):
        values = np.unique(X[:, feature])
        thresholds = (values[:-1] + values[1:]) / 2.0
        # sample thresholds to keep it manageable
        if len(thresholds) > 30:
            idx = np.linspace(0, len(thresholds) - 1, 30, dtype=int)
            thresholds = thresholds[idx]

        for t in thresholds:
            gain = compute_impurity_gain(X, y, feature, t, criterion)
            all_splits.append((feature, t, gain))
            if gain > best_gain:
                best_gain = gain
                best_feature = feature
                best_threshold = t

    return best_feature, best_threshold, best_gain, all_splits


# Rich colour palette used across all boundary plots
_REGION_COLORS  = ['#1565c0', '#2e7d32', '#e65100']   # bold point colours
_REGION_LIGHTS  = ['#dbeafe', '#dcfce7', '#ffedd5']   # light fill colours
_REGION_MIDS    = ['#93c5fd', '#86efac', '#fdba74']   # mid confidence
_REGION_DARKS   = ['#3b82f6', '#22c55e', '#f97316']   # high confidence


def plot_decision_boundary(clf, X, y, ax, title="", alpha=0.35, show_regions=True,
                           show_confidence=True, show_centroids=True,
                           show_boundary_line=True, show_region_labels=True):
    """Enhanced decision boundary plot with confidence shading, sharp borders,
    centroid markers, and per-region class labels."""
    from matplotlib.colors import LinearSegmentedColormap
    import matplotlib.patheffects as pe

    classes = np.unique(y)
    n_classes = len(classes)
    bold_colors = _REGION_COLORS[:n_classes]

    #  mesh 
    pad = 0.45
    x_min, x_max = X[:, 0].min() - pad, X[:, 0].max() + pad
    y_min, y_max = X[:, 1].min() - pad, X[:, 1].max() + pad
    h = 0.03   # finer mesh → crisper boundary

    xx, yy = np.meshgrid(np.arange(x_min, x_max, h),
                          np.arange(y_min, y_max, h))
    grid_points = np.c_[xx.ravel(), yy.ravel()]

    Z_cls = clf.predict(grid_points).reshape(xx.shape)

    if show_regions:
        if show_confidence and hasattr(clf, 'predict_proba'):
            #  confidence shading 
            # For each class, build a custom white→class colormap and
            # paint max-confidence pixels with the class colour.
            proba = clf.predict_proba(grid_points)
            max_proba = proba.max(axis=1).reshape(xx.shape)   # 0.5 … 1.0

            # Render per-class confidence layer
            for ci, cls_idx in enumerate(clf.classes_):
                class_mask = Z_cls == cls_idx
                conf_layer = np.where(class_mask, max_proba, np.nan)

                # Build a fully-transparent-to-solid colormap for this class
                light = to_rgba(_REGION_LIGHTS[ci % len(_REGION_LIGHTS)])
                dark  = to_rgba(_REGION_DARKS[ci % len(_REGION_DARKS)])
                cmap_ci = LinearSegmentedColormap.from_list(
                    f'cls_{ci}',
                    [(0, (*light[:3], 0.0)),
                     (0.4, (*light[:3], 0.30)),
                     (1.0, (*dark[:3],  0.65))]
                )
                ax.pcolormesh(xx, yy, conf_layer, cmap=cmap_ci,
                              vmin=0.5, vmax=1.0, shading='auto', zorder=1)
        else:
            # Flat fill fallback
            flat_cmap = ListedColormap(_REGION_LIGHTS[:n_classes])
            ax.contourf(xx, yy, Z_cls, alpha=alpha, cmap=flat_cmap, zorder=1)

        if show_boundary_line:
            #  crisp boundary outline with subtle glow 
            ax.contour(xx, yy, Z_cls, levels=n_classes - 1,
                       colors='white', linewidths=3.5, alpha=0.6, zorder=3)
            ax.contour(xx, yy, Z_cls, levels=n_classes - 1,
                       colors=['#1e1e2e'], linewidths=1.4, alpha=0.85, zorder=4,
                       linestyles='solid')

        if show_region_labels:
            #  label each region near its centroid 
            from scipy import ndimage as ndi
            for ci, cls_idx in enumerate(clf.classes_):
                mask = (Z_cls == cls_idx)
                if not mask.any():
                    continue
                # centre of mass of the region in grid coords
                cy_idx, cx_idx = ndi.center_of_mass(mask)
                lx = x_min + cx_idx * h
                ly = y_min + cy_idx * h
                label_color = bold_colors[ci % len(bold_colors)]
                ax.text(lx, ly, f'Class {cls_idx}',
                        fontsize=10, fontweight=700, color='white',
                        ha='center', va='center', zorder=6,
                        path_effects=[
                            pe.withStroke(linewidth=3,
                                          foreground=label_color)
                        ])

    #  data points 
    cmap_bold = ListedColormap(bold_colors)
    scatter = ax.scatter(X[:, 0], X[:, 1],
                         c=y, cmap=cmap_bold,
                         edgecolors='white', s=55,
                         linewidths=0.9, alpha=0.92, zorder=7)

    if show_centroids:
        #  crosshair centroids 
        for ci, cls_idx in enumerate(classes):
            mask_pts = y == cls_idx
            cx, cy_val = X[mask_pts, 0].mean(), X[mask_pts, 1].mean()
            col = bold_colors[ci % len(bold_colors)]
            ax.scatter(cx, cy_val, marker='P', s=180, color='white',
                       edgecolors=col, linewidths=2.5, zorder=9)
            ax.scatter(cx, cy_val, marker='+', s=90,  color=col,
                       linewidths=2, zorder=10)

    #  axes styling 
    ax.set_xlim(x_min, x_max)
    ax.set_ylim(y_min, y_max)
    ax.set_xlabel("Feature 1", fontsize=10, fontweight=500)
    ax.set_ylabel("Feature 2", fontsize=10, fontweight=500)
    if title:
        ax.set_title(title, fontsize=12, fontweight=600, pad=10)
    ax.tick_params(labelsize=8)
    ax.grid(True, linewidth=0.4, alpha=0.25, color='#94a3b8')
    ax.set_facecolor('#f8fafc')
    return scatter


def build_tree_graphviz(clf, feature_names=None):
    """Build a Graphviz visualization of the tree."""
    from sklearn.tree import export_graphviz
    import io

    if feature_names is None:
        feature_names = ["Feature 1", "Feature 2"]

    dot_data = export_graphviz(
        clf,
        out_file=None,
        feature_names=feature_names,
        class_names=[f"Class {i}" for i in range(len(np.unique(clf.classes_)))],
        filled=True,
        rounded=True,
        special_characters=True,
        proportion=False,
        impurity=True
    )
    return dot_data


def get_prediction_path(clf, point):
    """Get the decision path for a single point."""
    node_indicator = clf.decision_path(point.reshape(1, -1))
    node_ids = node_indicator.indices.tolist()
    tree = clf.tree_

    rules = []
    for i, node_id in enumerate(node_ids):
        if tree.children_left[node_id] != tree.children_right[node_id]:
            feat = tree.feature[node_id]
            thresh = tree.threshold[node_id]
            feat_name = f"Feature {feat + 1}"
            val = point[feat]
            direction = "≤" if val <= thresh else ">"
            rules.append({
                'node': node_id,
                'feature': feat_name,
                'threshold': round(thresh, 3),
                'value': round(val, 3),
                'direction': direction,
                'depth': i
            })

    prediction = clf.predict(point.reshape(1, -1))[0]
    return rules, prediction, node_ids


#  Sidebar Controls 

with st.sidebar:
    st.markdown("##  Controls")
    st.markdown("---")

    st.markdown("###  Dataset")
    dataset_name = st.selectbox(
        "Choose Dataset",
        ["Moons", "Circles", "Blobs", "Linear", "XOR"],
        index=0,
        help="Select the type of 2D dataset to generate"
    )
    n_samples = st.slider("Number of Samples", 100, 800, 300, 50)

    st.markdown("---")
    st.markdown("###  Tree Parameters")

    max_depth = st.slider("Max Depth", 1, 15, 4, 1,
                          help="Maximum depth of the decision tree")
    min_samples_split = st.slider("Min Samples per Split", 2, 50, 5, 1,
                                  help="Minimum samples required to split a node")
    min_samples_leaf = st.slider("Min Samples per Leaf", 1, 30, 2, 1,
                                 help="Minimum samples required at a leaf node")

    criterion = st.radio("Impurity Criterion", ["gini", "entropy"],
                         format_func=lambda x: "Gini Impurity" if x == "gini" else "Entropy (Information Gain)")

    st.markdown("---")
    st.markdown("###  Options")

    noise_level = st.slider("Noise Level", 0.0, 1.0, 0.0, 0.05,
                            help="Add noise to the dataset")
    show_impurity = st.toggle("Show Impurity Values", value=True)
    random_seed = st.number_input("Random Seed", 0, 999, 42, 1)

#  Generate Data & Train Model 

X, y = generate_dataset(dataset_name, n_samples, noise_level, int(random_seed))
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=int(random_seed))

clf = DecisionTreeClassifier(
    max_depth=max_depth,
    min_samples_split=min_samples_split,
    min_samples_leaf=min_samples_leaf,
    criterion=criterion,
    random_state=int(random_seed)
)
clf.fit(X_train, y_train)

train_acc = accuracy_score(y_train, clf.predict(X_train))
test_acc = accuracy_score(y_test, clf.predict(X_test))

#  Header 

st.markdown('<h1 class="main-title"> Decision Tree Visualizer</h1>', unsafe_allow_html=True)
st.markdown('<p class="sub-title">Explore how Decision Trees partition data, choose splits, and make predictions interactively.</p>', unsafe_allow_html=True)

#  Metrics Row 
col1, col2, col3, col4 = st.columns(4)
with col1:
    st.markdown(f"""
    <div class="metric-card">
        <div class="metric-value">{train_acc:.1%}</div>
        <div class="metric-label">Train Accuracy</div>
    </div>
    """, unsafe_allow_html=True)
with col2:
    st.markdown(f"""
    <div class="metric-card">
        <div class="metric-value">{test_acc:.1%}</div>
        <div class="metric-label">Test Accuracy</div>
    </div>
    """, unsafe_allow_html=True)
with col3:
    st.markdown(f"""
    <div class="metric-card">
        <div class="metric-value">{clf.get_depth()}</div>
        <div class="metric-label">Tree Depth</div>
    </div>
    """, unsafe_allow_html=True)
with col4:
    st.markdown(f"""
    <div class="metric-card">
        <div class="metric-value">{clf.get_n_leaves()}</div>
        <div class="metric-label">Leaf Nodes</div>
    </div>
    """, unsafe_allow_html=True)

st.markdown("<br>", unsafe_allow_html=True)

#  Tabs 

tab1, tab2, tab3, tab4, tab5, tab6, tab7 = st.tabs([
    " Data & Splits",
    " Impurity Measures",
    " Split Selection",
    " Tree Growth",
    " Prediction Path",
    " Overfitting & Depth",
    " Noise & Pruning"
])

# 
# TAB 1: DATA & SPLITS
# 
with tab1:
    st.markdown('<div class="concept-box"> <strong>Concept:</strong> Decision Trees recursively split the dataset based on feature values. Each split divides data into subsets, aiming to increase purity in resulting nodes. The input space is partitioned into rectangular decision regions.</div>', unsafe_allow_html=True)

    col_left, col_right = st.columns([1, 1])

    with col_left:
        st.markdown("#### Raw Data Points")
        fig1, ax1 = plt.subplots(figsize=(7, 5))
        fig1.patch.set_facecolor('#fafafa')
        ax1.set_facecolor('#fafafa')
        cmap_bold = ListedColormap(['#1565c0', '#2e7d32', '#e65100'])
        scatter = ax1.scatter(X[:, 0], X[:, 1], c=y, cmap=cmap_bold,
                              edgecolors='white', s=45, linewidths=0.8, alpha=0.85)
        ax1.set_xlabel("Feature 1", fontsize=11, fontweight=500)
        ax1.set_ylabel("Feature 2", fontsize=11, fontweight=500)
        ax1.set_title(f"{dataset_name} Dataset ({n_samples} samples)", fontsize=13, fontweight=600)
        ax1.tick_params(labelsize=9)
        legend_handles = [mpatches.Patch(color=c, label=f'Class {i}')
                          for i, c in enumerate(['#1565c0', '#2e7d32', '#e65100']) if i in np.unique(y)]
        ax1.legend(handles=legend_handles, loc='upper right', fontsize=9, framealpha=0.9)
        plt.tight_layout()
        st.pyplot(fig1)

    with col_right:
        st.markdown("#### Decision Regions")

        # Extra controls inline
        dr_conf = st.toggle("Confidence shading", value=True, key="dr_conf")
        dr_cent = st.toggle("Class centroids",    value=True, key="dr_cent")

        fig2, ax2 = plt.subplots(figsize=(7, 5))
        fig2.patch.set_facecolor('#f8fafc')
        plot_decision_boundary(
            clf, X, y, ax2,
            title=f"Decision Boundary  (depth={clf.get_depth()},  leaves={clf.get_n_leaves()})",
            show_confidence=dr_conf,
            show_centroids=dr_cent,
            show_boundary_line=True,
            show_region_labels=True,
        )
        # Custom legend
        classes_present = np.unique(y)
        legend_handles = [
            mpatches.Patch(
                facecolor=_REGION_DARKS[i % len(_REGION_DARKS)],
                edgecolor='white', linewidth=0.8,
                label=f'Class {c} region'
            )
            for i, c in enumerate(classes_present)
        ] + [
            plt.scatter([], [], c=_REGION_COLORS[i % len(_REGION_COLORS)],
                        edgecolors='white', s=50, linewidths=0.9,
                        label=f'Class {c} point')
            for i, c in enumerate(classes_present)
        ]
        ax2.legend(handles=legend_handles, loc='upper right',
                   fontsize=8, framealpha=0.92,
                   facecolor='white', edgecolor='#e2e8f0')
        plt.tight_layout()
        st.pyplot(fig2)

    # Interactive split line
    st.markdown("---")
    st.markdown("####  Interactive Manual Split")
    st.markdown("Choose a feature and threshold to see how the data is split:")

    mcol1, mcol2 = st.columns(2)
    with mcol1:
        split_feature = st.selectbox("Split Feature", [0, 1],
                                     format_func=lambda x: f"Feature {x + 1}")
    with mcol2:
        feat_min = float(X[:, split_feature].min())
        feat_max = float(X[:, split_feature].max())
        split_thresh = st.slider("Split Threshold",
                                 feat_min, feat_max,
                                 float(np.median(X[:, split_feature])),
                                 step=(feat_max - feat_min) / 100)

    fig3, axes3 = plt.subplots(1, 2, figsize=(14, 5))
    fig3.patch.set_facecolor('#fafafa')

    for ax in axes3:
        ax.set_facecolor('#fafafa')

    # Before split
    axes3[0].scatter(X[:, 0], X[:, 1], c=y, cmap=cmap_bold,
                     edgecolors='white', s=40, linewidths=0.7, alpha=0.85)
    if split_feature == 0:
        axes3[0].axvline(x=split_thresh, color='#e53935', linewidth=2.5, linestyle='--', label=f'Split: F1 ≤ {split_thresh:.2f}')
    else:
        axes3[0].axhline(y=split_thresh, color='#e53935', linewidth=2.5, linestyle='--', label=f'Split: F2 ≤ {split_thresh:.2f}')
    axes3[0].legend(fontsize=9, loc='upper right')
    axes3[0].set_title("Data with Split Line", fontsize=12, fontweight=600)
    axes3[0].set_xlabel("Feature 1", fontsize=10)
    axes3[0].set_ylabel("Feature 2", fontsize=10)

    # After split - colored by region
    left_mask = X[:, split_feature] <= split_thresh
    right_mask = ~left_mask

    axes3[1].scatter(X[left_mask, 0], X[left_mask, 1], c='#1e88e5', marker='o',
                     edgecolors='white', s=40, linewidths=0.7, alpha=0.8, label=f'Left ({left_mask.sum()} pts)')
    axes3[1].scatter(X[right_mask, 0], X[right_mask, 1], c='#fb8c00', marker='^',
                     edgecolors='white', s=40, linewidths=0.7, alpha=0.8, label=f'Right ({right_mask.sum()} pts)')
    if split_feature == 0:
        axes3[1].axvline(x=split_thresh, color='#e53935', linewidth=2.5, linestyle='--')
    else:
        axes3[1].axhline(y=split_thresh, color='#e53935', linewidth=2.5, linestyle='--')
    axes3[1].legend(fontsize=9, loc='upper right')
    axes3[1].set_title("After Split (Colored by Region)", fontsize=12, fontweight=600)
    axes3[1].set_xlabel("Feature 1", fontsize=10)
    axes3[1].set_ylabel("Feature 2", fontsize=10)

    plt.tight_layout()
    st.pyplot(fig3)

    # Impurity of manual split
    gain = compute_impurity_gain(X, y, split_feature, split_thresh, criterion)
    parent_imp = compute_gini(y) if criterion == 'gini' else compute_entropy(y)

    mcol_a, mcol_b, mcol_c = st.columns(3)
    with mcol_a:
        st.metric("Parent Impurity", f"{parent_imp:.4f}")
    with mcol_b:
        left_imp = compute_gini(y[left_mask]) if criterion == 'gini' else compute_entropy(y[left_mask])
        right_imp = compute_gini(y[right_mask]) if criterion == 'gini' else compute_entropy(y[right_mask])
        st.metric("Left / Right Impurity", f"{left_imp:.4f} / {right_imp:.4f}")
    with mcol_c:
        st.metric("Information Gain", f"{gain:.4f}")

    st.markdown('<div class="observe-box"> <strong>What to observe:</strong> Notice how the Decision Tree creates rectangular regions by splitting parallel to the axes. The more splits, the more complex the boundary. Try different datasets to see how the tree adapts its partitioning strategy.</div>', unsafe_allow_html=True)


# 
# TAB 2: IMPURITY MEASURES
# 
with tab2:
    st.markdown('<div class="concept-box"> <strong>Concept:</strong> Decision Trees choose splits by measuring impurity. <strong>Gini Impurity</strong> = 1 − Σ(pᵢ²) measures the probability of misclassification. <strong>Entropy</strong> = −Σ(pᵢ log₂ pᵢ) measures the amount of disorder or uncertainty.</div>', unsafe_allow_html=True)

    # Gini vs Entropy comparison curve
    st.markdown("#### Gini vs Entropy for Binary Classification")
    st.markdown("How both metrics change as the class balance varies from 0% to 100%:")

    p_range = np.linspace(0.001, 0.999, 200)
    gini_values = 1 - p_range**2 - (1 - p_range)**2
    entropy_values = -p_range * np.log2(p_range) - (1 - p_range) * np.log2(1 - p_range)

    fig_imp, ax_imp = plt.subplots(figsize=(10, 5))
    fig_imp.patch.set_facecolor('#fafafa')
    ax_imp.set_facecolor('#fafafa')
    ax_imp.plot(p_range, gini_values, color='#e53935', linewidth=2.5, label='Gini Impurity', zorder=5)
    ax_imp.plot(p_range, entropy_values, color='#1e88e5', linewidth=2.5, label='Entropy', zorder=5)
    ax_imp.fill_between(p_range, gini_values, alpha=0.08, color='#e53935')
    ax_imp.fill_between(p_range, entropy_values, alpha=0.08, color='#1e88e5')
    ax_imp.axvline(x=0.5, color='#9e9e9e', linestyle=':', linewidth=1, alpha=0.7)
    ax_imp.annotate('Max Impurity\n(50/50 split)', xy=(0.5, 0.5), xytext=(0.65, 0.42),
                    fontsize=9, fontweight=500, color='#616161',
                    arrowprops=dict(arrowstyle='->', color='#9e9e9e', lw=1.2))
    ax_imp.set_xlabel("Proportion of Class 1 (p)", fontsize=11, fontweight=500)
    ax_imp.set_ylabel("Impurity Value", fontsize=11, fontweight=500)
    ax_imp.set_title("Gini Impurity vs Entropy (Binary Classification)", fontsize=13, fontweight=600)
    ax_imp.legend(fontsize=10, framealpha=0.9, loc='lower center')
    ax_imp.set_xlim(0, 1)
    ax_imp.set_ylim(0, 1.1)
    ax_imp.grid(True, alpha=0.2)
    plt.tight_layout()
    st.pyplot(fig_imp)

    # Node-level impurity
    st.markdown("---")
    st.markdown("#### Class Distribution & Impurity at Each Node")

    tree = clf.tree_
    n_nodes = tree.node_count

    node_data = []
    for i in range(min(n_nodes, 20)):  # limit to first 20 nodes
        node_samples = tree.n_node_samples[i]
        node_values = tree.value[i][0]
        is_leaf = tree.children_left[i] == tree.children_right[i]
        node_type = "Leaf" if is_leaf else "Internal"

        class_probs = node_values / node_values.sum()
        gini_val = 1.0 - np.sum(class_probs ** 2)
        entropy_val = -np.sum([p * np.log2(p) for p in class_probs if p > 0])

        node_data.append({
            'Node ID': i,
            'Type': node_type,
            'Samples': node_samples,
            'Gini': round(gini_val, 4),
            'Entropy': round(entropy_val, 4),
            **{f'Class {j}': int(v) for j, v in enumerate(node_values)}
        })

    df_nodes = pd.DataFrame(node_data)
    st.dataframe(df_nodes, use_container_width=True, hide_index=True)

    # Bar chart of impurity by node
    if show_impurity and len(node_data) > 0:
        fig_bars, ax_bars = plt.subplots(figsize=(12, 4))
        fig_bars.patch.set_facecolor('#fafafa')
        ax_bars.set_facecolor('#fafafa')
        x_pos = np.arange(len(node_data))
        width = 0.35
        bars1 = ax_bars.bar(x_pos - width/2, [d['Gini'] for d in node_data], width,
                            color='#e53935', alpha=0.8, label='Gini', edgecolor='white')
        bars2 = ax_bars.bar(x_pos + width/2, [d['Entropy'] for d in node_data], width,
                            color='#1e88e5', alpha=0.8, label='Entropy', edgecolor='white')
        ax_bars.set_xlabel("Node ID", fontsize=10, fontweight=500)
        ax_bars.set_ylabel("Impurity", fontsize=10, fontweight=500)
        ax_bars.set_title("Impurity per Node", fontsize=12, fontweight=600)
        ax_bars.set_xticks(x_pos)
        ax_bars.set_xticklabels([d['Node ID'] for d in node_data], fontsize=8)
        ax_bars.legend(fontsize=9)
        ax_bars.grid(axis='y', alpha=0.2)
        plt.tight_layout()
        st.pyplot(fig_bars)

    st.markdown('<div class="observe-box"> <strong>What to observe:</strong> Both Gini and Entropy are maximized at a 50/50 class split and minimized at pure nodes. Entropy has a slightly larger range (0 to 1.0 vs 0 to 0.5). Leaf nodes should have low impurity — that means the tree has found pure regions.</div>', unsafe_allow_html=True)


# 
# TAB 3: SPLIT SELECTION
# 
with tab3:
    st.markdown('<div class="concept-box"> <strong>Concept:</strong> At each node, the algorithm evaluates all possible (feature, threshold) pairs and selects the one that maximizes impurity reduction (information gain). This is the core decision-making mechanism of tree construction.</div>', unsafe_allow_html=True)

    best_feat, best_thresh, best_gain, all_splits = find_best_split(X_train, y_train, criterion)

    st.markdown(f"#### Best Split Found: **Feature {best_feat + 1} ≤ {best_thresh:.4f}** (Gain: {best_gain:.4f})")

    # All candidate splits
    splits_df = pd.DataFrame(all_splits, columns=['Feature', 'Threshold', 'Gain'])
    splits_df['Feature'] = splits_df['Feature'].map({0: 'Feature 1', 1: 'Feature 2'})
    splits_df = splits_df.sort_values('Gain', ascending=False).reset_index(drop=True)

    col_s1, col_s2 = st.columns([1, 1])

    with col_s1:
        st.markdown("#### Candidate Splits Ranked by Gain")
        st.dataframe(splits_df.head(15), use_container_width=True, hide_index=True)

    with col_s2:
        st.markdown("#### Information Gain by Threshold")
        fig_gain, ax_gain = plt.subplots(figsize=(7, 5))
        fig_gain.patch.set_facecolor('#fafafa')
        ax_gain.set_facecolor('#fafafa')

        for feat_val, color, label in [(0, '#1e88e5', 'Feature 1'), (1, '#e53935', 'Feature 2')]:
            mask = [s[0] == feat_val for s in all_splits]
            thresholds = [s[1] for s, m in zip(all_splits, mask) if m]
            gains = [s[2] for s, m in zip(all_splits, mask) if m]
            ax_gain.scatter(thresholds, gains, c=color, alpha=0.6, s=30, label=label, zorder=4)

        ax_gain.axhline(y=best_gain, color='#4caf50', linestyle='--', linewidth=1.5, alpha=0.7, label=f'Best Gain ({best_gain:.4f})')
        ax_gain.set_xlabel("Threshold Value", fontsize=10, fontweight=500)
        ax_gain.set_ylabel("Information Gain", fontsize=10, fontweight=500)
        ax_gain.set_title(f"Split Gain Analysis ({criterion.title()})", fontsize=12, fontweight=600)
        ax_gain.legend(fontsize=9)
        ax_gain.grid(True, alpha=0.2)
        plt.tight_layout()
        st.pyplot(fig_gain)

    # Interactive threshold slider
    st.markdown("---")
    st.markdown("####  Interactive Threshold Explorer")

    exp_feature = st.selectbox("Explore Feature", [0, 1],
                               format_func=lambda x: f"Feature {x + 1}",
                               key="explore_feat")

    feat_vals = np.sort(np.unique(X_train[:, exp_feature]))
    exp_thresh = st.slider(
        f"Threshold for Feature {exp_feature + 1}",
        float(feat_vals[0]), float(feat_vals[-1]),
        float(np.median(feat_vals)),
        step=float((feat_vals[-1] - feat_vals[0]) / 100),
        key="explore_thresh"
    )

    exp_gain = compute_impurity_gain(X_train, y_train, exp_feature, exp_thresh, criterion)
    left_m = X_train[:, exp_feature] <= exp_thresh
    right_m = ~left_m

    imp_fn = compute_gini if criterion == 'gini' else compute_entropy
    parent_imp_val = imp_fn(y_train)
    left_imp_val = imp_fn(y_train[left_m])
    right_imp_val = imp_fn(y_train[right_m])

    ecol1, ecol2, ecol3, ecol4 = st.columns(4)
    with ecol1:
        st.metric("Parent Impurity", f"{parent_imp_val:.4f}")
    with ecol2:
        st.metric("Left Impurity", f"{left_imp_val:.4f}", delta=f"{left_m.sum()} samples")
    with ecol3:
        st.metric("Right Impurity", f"{right_imp_val:.4f}", delta=f"{right_m.sum()} samples")
    with ecol4:
        delta_color = "normal" if exp_gain >= best_gain * 0.8 else "inverse"
        st.metric("Gain", f"{exp_gain:.4f}",
                  delta=f"{' Near Best' if exp_gain >= best_gain * 0.8 else ' Below Best'}")

    st.markdown('<div class="observe-box"> <strong>What to observe:</strong> The best split maximizes the information gain. Notice how different thresholds produce very different gains. The algorithm exhaustively checks all candidate splits at each node to find the optimal partition.</div>', unsafe_allow_html=True)


# 
# TAB 4: TREE GROWTH
# 
with tab4:
    st.markdown('<div class="concept-box"> <strong>Concept:</strong> The model grows recursively from the Root node, creating Internal Nodes with split rules and Leaf Nodes with predictions. The tree represents hierarchical decision rules — each path from root to leaf defines a classification rule.</div>', unsafe_allow_html=True)

    # Step-by-step tree growth
    st.markdown("#### Step-by-Step Tree Growth")
    growth_depth = st.slider("Grow tree to depth:", 1, min(max_depth, 10), min(3, max_depth), key="growth_depth")

    growth_cols = st.columns(min(growth_depth, 4))

    for d in range(min(growth_depth, 4)):
        depth_val = d + 1
        with growth_cols[d]:
            clf_step = DecisionTreeClassifier(
                max_depth=depth_val,
                min_samples_split=min_samples_split,
                min_samples_leaf=min_samples_leaf,
                criterion=criterion,
                random_state=int(random_seed)
            )
            clf_step.fit(X_train, y_train)
            fig_step, ax_step = plt.subplots(figsize=(5, 4))
            fig_step.patch.set_facecolor('#fafafa')
            ax_step.set_facecolor('#fafafa')
            plot_decision_boundary(clf_step, X_train, y_train, ax_step,
                                   title=f"Depth = {depth_val}", alpha=0.3)
            ax_step.tick_params(labelsize=7)
            plt.tight_layout()
            st.pyplot(fig_step)
            step_acc = accuracy_score(y_train, clf_step.predict(X_train))
            st.caption(f"Accuracy: {step_acc:.2%} | Leaves: {clf_step.get_n_leaves()}")

    # Full tree diagram
    st.markdown("---")
    st.markdown("####  Full Tree Structure")

    try:
        dot_data = build_tree_graphviz(clf)
        # Enhance graphviz styling
        dot_data = dot_data.replace('digraph Tree {', '''digraph Tree {
graph [bgcolor="#fafafa" pad="0.5" ranksep="0.8" nodesep="0.5"]
node [fontname="Inter" fontsize=10 penwidth=1.5]
edge [fontname="Inter" fontsize=9 color="#9e9e9e" penwidth=1.2]''')
        st.graphviz_chart(dot_data)
    except Exception as e:
        st.warning(f"Could not render tree diagram: {e}")
        st.code(export_text(clf, feature_names=["Feature 1", "Feature 2"]))

    # Text representation
    with st.expander(" Text Representation of Decision Rules"):
        st.code(export_text(clf, feature_names=["Feature 1", "Feature 2"],
                            max_depth=10, show_weights=True), language=None)

    st.markdown('<div class="observe-box"> <strong>What to observe:</strong> Watch how the boundary complexity increases with depth. At depth 1, only one split exists. Each additional depth level doubles the maximum number of regions. Deeper trees capture more detail but risk overfitting.</div>', unsafe_allow_html=True)


# 
# TAB 5: PREDICTION PATH
# 
with tab5:
    st.markdown('<div class="concept-box"> <strong>Concept:</strong> To predict, the tree starts at the root, evaluates the split condition at each node, and traverses left or right until reaching a leaf node. The leaf\'s majority class becomes the prediction. This creates an interpretable if-else decision chain.</div>', unsafe_allow_html=True)

    st.markdown("#### Select a Point to Trace its Decision Path")

    pcol1, pcol2 = st.columns(2)
    with pcol1:
        point_x = st.slider("Feature 1 value",
                             float(X[:, 0].min()), float(X[:, 0].max()),
                             float(np.mean(X[:, 0])), key="pred_x")
    with pcol2:
        point_y = st.slider("Feature 2 value",
                             float(X[:, 1].min()), float(X[:, 1].max()),
                             float(np.mean(X[:, 1])), key="pred_y")

    test_point = np.array([point_x, point_y])
    rules, prediction, node_ids = get_prediction_path(clf, test_point)

    pred_col1, pred_col2 = st.columns([1.2, 0.8])

    with pred_col1:
        st.markdown("#### Decision Boundary with Selected Point")
        fig_pred, ax_pred = plt.subplots(figsize=(8, 6))
        fig_pred.patch.set_facecolor('#fafafa')
        ax_pred.set_facecolor('#fafafa')
        plot_decision_boundary(clf, X, y, ax_pred, title="Prediction Visualization")
        ax_pred.scatter([point_x], [point_y], c='#e53935', s=250, marker='*',
                        edgecolors='#b71c1c', linewidths=1.5, zorder=10, label='Selected Point')
        ax_pred.annotate(f'Predicted: Class {prediction}',
                         xy=(point_x, point_y),
                         xytext=(point_x + 0.3, point_y + 0.3),
                         fontsize=10, fontweight=600, color='#e53935',
                         arrowprops=dict(arrowstyle='->', color='#e53935', lw=1.5),
                         bbox=dict(boxstyle='round,pad=0.3', facecolor='white', edgecolor='#e53935', alpha=0.9),
                         zorder=11)
        ax_pred.legend(fontsize=9, loc='upper right')
        plt.tight_layout()
        st.pyplot(fig_pred)

    with pred_col2:
        st.markdown("####  Decision Path")

        color_map = {True: '#4caf50', False: '#f44336'}

        if rules:
            for i, rule in enumerate(rules):
                went_left = rule['value'] <= rule['threshold']
                icon = "" if went_left else ""
                color = '#4caf50' if went_left else '#ff9800'
                st.markdown(f"""
                <div style="padding: 10px 14px; margin: 6px 0; border-radius: 10px;
                            background: linear-gradient(120deg, {'#e8f5e9' if went_left else '#fff3e0'}, white);
                            border-left: 4px solid {color}; font-size: 0.88rem;">
                    <strong>Node {rule['node']}</strong> (depth {rule['depth']})<br>
                    {icon} {rule['feature']} = <code>{rule['value']}</code>
                    {rule['direction']} <code>{rule['threshold']}</code>
                </div>
                """, unsafe_allow_html=True)

        st.markdown(f"""
        <div style="padding: 14px 18px; margin: 10px 0; border-radius: 12px;
                    background: linear-gradient(135deg, #e8f5e9, #c8e6c9);
                    border: 2px solid #4caf50; text-align: center;">
            <strong style="font-size: 1.1rem;"> Prediction: Class {prediction}</strong>
        </div>
        """, unsafe_allow_html=True)

    # Rules as code
    with st.expander(" Decision Rules as Code"):
        code_lines = []
        for i, rule in enumerate(rules):
            indent = " " * rule['depth']
            direction = "<=" if rule['value'] <= rule['threshold'] else ">"
            code_lines.append(f"{indent}if {rule['feature']} {direction} {rule['threshold']}:")
        code_lines.append(f"{' ' * (len(rules))}return Class {prediction}")
        st.code("\n".join(code_lines), language="python")

    st.markdown('<div class="observe-box"> <strong>What to observe:</strong> The prediction path shows exactly which rules are applied. Move the point to different regions and watch how the path changes. This interpretability is a key advantage of Decision Trees over black-box models.</div>', unsafe_allow_html=True)


# 
# TAB 6: OVERFITTING & DEPTH
# 
with tab6:
    st.markdown('<div class="concept-box"> <strong>Concept:</strong> Tree growth stops based on maximum depth, minimum samples per node, or no impurity improvement. Without these controls, trees can overfit — memorizing training noise instead of learning the true pattern. The gap between train and test accuracy indicates overfitting.</div>', unsafe_allow_html=True)

    st.markdown("#### Train vs Test Accuracy Across Depths")

    depths = list(range(1, 16))
    train_accs = []
    test_accs = []
    n_leaves = []

    for d in depths:
        clf_d = DecisionTreeClassifier(
            max_depth=d,
            min_samples_split=2,
            min_samples_leaf=1,
            criterion=criterion,
            random_state=int(random_seed)
        )
        clf_d.fit(X_train, y_train)
        train_accs.append(accuracy_score(y_train, clf_d.predict(X_train)))
        test_accs.append(accuracy_score(y_test, clf_d.predict(X_test)))
        n_leaves.append(clf_d.get_n_leaves())

    fig_over, ax_over = plt.subplots(figsize=(12, 5))
    fig_over.patch.set_facecolor('#fafafa')
    ax_over.set_facecolor('#fafafa')
    ax_over.plot(depths, train_accs, 'o-', color='#1e88e5', linewidth=2.5, markersize=7, label='Train Accuracy', zorder=5)
    ax_over.plot(depths, test_accs, 's-', color='#e53935', linewidth=2.5, markersize=7, label='Test Accuracy', zorder=5)
    ax_over.fill_between(depths, train_accs, test_accs, alpha=0.1, color='#ff9800', label='Overfitting Gap')

    # Mark the current depth
    ax_over.axvline(x=max_depth, color='#4caf50', linestyle='--', linewidth=1.5, alpha=0.7, label=f'Current Depth ({max_depth})')

    # Find best test depth
    best_test_depth = depths[np.argmax(test_accs)]
    ax_over.axvline(x=best_test_depth, color='#7c3aed', linestyle=':', linewidth=1.5, alpha=0.7, label=f'Best Test Depth ({best_test_depth})')

    ax_over.set_xlabel("Tree Depth", fontsize=11, fontweight=500)
    ax_over.set_ylabel("Accuracy", fontsize=11, fontweight=500)
    ax_over.set_title("Overfitting Analysis: Train vs Test Accuracy", fontsize=13, fontweight=600)
    ax_over.legend(fontsize=9, loc='lower right')
    ax_over.set_xticks(depths)
    ax_over.set_ylim(0.5, 1.05)
    ax_over.grid(True, alpha=0.2)
    plt.tight_layout()
    st.pyplot(fig_over)

    # Side-by-side: shallow vs deep
    st.markdown("---")
    st.markdown("#### Shallow vs Deep Tree Comparison")

    comp_col1, comp_col2 = st.columns(2)

    with comp_col1:
        clf_shallow = DecisionTreeClassifier(max_depth=2, criterion=criterion, random_state=int(random_seed))
        clf_shallow.fit(X_train, y_train)
        fig_sh, ax_sh = plt.subplots(figsize=(6, 5))
        fig_sh.patch.set_facecolor('#fafafa')
        ax_sh.set_facecolor('#fafafa')
        plot_decision_boundary(clf_shallow, X, y, ax_sh,
                               title=f"Shallow Tree (depth=2)\nTrain: {accuracy_score(y_train, clf_shallow.predict(X_train)):.1%} | Test: {accuracy_score(y_test, clf_shallow.predict(X_test)):.1%}")
        plt.tight_layout()
        st.pyplot(fig_sh)

    with comp_col2:
        clf_deep = DecisionTreeClassifier(max_depth=15, criterion=criterion, random_state=int(random_seed))
        clf_deep.fit(X_train, y_train)
        fig_dp, ax_dp = plt.subplots(figsize=(6, 5))
        fig_dp.patch.set_facecolor('#fafafa')
        ax_dp.set_facecolor('#fafafa')
        plot_decision_boundary(clf_deep, X, y, ax_dp,
                               title=f"Deep Tree (depth=15)\nTrain: {accuracy_score(y_train, clf_deep.predict(X_train)):.1%} | Test: {accuracy_score(y_test, clf_deep.predict(X_test)):.1%}")
        plt.tight_layout()
        st.pyplot(fig_dp)

    # Complexity chart
    fig_leaves, ax_leaves = plt.subplots(figsize=(12, 4))
    fig_leaves.patch.set_facecolor('#fafafa')
    ax_leaves.set_facecolor('#fafafa')
    ax_leaves.bar(depths, n_leaves, color=['#7c3aed' if d == max_depth else '#b0bec5' for d in depths],
                  edgecolor='white', linewidth=0.5, alpha=0.85)
    ax_leaves.set_xlabel("Tree Depth", fontsize=10, fontweight=500)
    ax_leaves.set_ylabel("Number of Leaves", fontsize=10, fontweight=500)
    ax_leaves.set_title("Model Complexity (Leaves) vs Depth", fontsize=12, fontweight=600)
    ax_leaves.set_xticks(depths)
    ax_leaves.grid(axis='y', alpha=0.2)
    plt.tight_layout()
    st.pyplot(fig_leaves)

    st.markdown('<div class="observe-box"> <strong>What to observe:</strong> Train accuracy always increases with depth, but test accuracy peaks and then declines — this is overfitting. The orange shaded region shows the gap. A shallow tree underfits (both accuracies low), while a very deep tree overfits (high train, low test). The optimal depth balances both.</div>', unsafe_allow_html=True)


# 
# TAB 7: NOISE & PRUNING
# 
with tab7:
    st.markdown('<div class="concept-box"> <strong>Concept:</strong> Real-world data contains noise and irrelevant features. Without pruning or regularization, trees create overly complex decision boundaries that match the noise. Pruning reduces tree complexity to improve generalization.</div>', unsafe_allow_html=True)

    st.markdown("#### Effect of Noise on Decision Boundaries")

    noise_levels = [0.0, 0.3, 0.6, 1.0]
    fig_noise, axes_noise = plt.subplots(1, 4, figsize=(20, 4.5))
    fig_noise.patch.set_facecolor('#fafafa')

    for idx, nl in enumerate(noise_levels):
        ax = axes_noise[idx]
        ax.set_facecolor('#fafafa')
        X_n, y_n = generate_dataset(dataset_name, n_samples, nl, int(random_seed))
        X_tn, X_te, y_tn, y_te = train_test_split(X_n, y_n, test_size=0.25, random_state=int(random_seed))

        clf_n = DecisionTreeClassifier(max_depth=max_depth, criterion=criterion, random_state=int(random_seed))
        clf_n.fit(X_tn, y_tn)
        plot_decision_boundary(clf_n, X_n, y_n, ax,
                               title=f"Noise: {nl:.0%}\nAcc: {accuracy_score(y_te, clf_n.predict(X_te)):.1%}",
                               alpha=0.3)
        ax.tick_params(labelsize=7)

    plt.tight_layout()
    st.pyplot(fig_noise)

    # Pruned vs Unpruned
    st.markdown("---")
    st.markdown("#### Pruned vs Unpruned Tree Comparison")

    st.markdown("**Post-pruning** via `ccp_alpha` (Cost Complexity Pruning):")

    ccp_alpha = st.slider("Pruning Strength (ccp_alpha)", 0.0, 0.1, 0.01, 0.002,
                          help="Higher values = more aggressive pruning")

    prune_col1, prune_col2 = st.columns(2)

    with prune_col1:
        clf_unpruned = DecisionTreeClassifier(max_depth=None, criterion=criterion, random_state=int(random_seed))
        clf_unpruned.fit(X_train, y_train)
        fig_up, ax_up = plt.subplots(figsize=(6, 5))
        fig_up.patch.set_facecolor('#fafafa')
        ax_up.set_facecolor('#fafafa')
        plot_decision_boundary(clf_unpruned, X, y, ax_up,
                               title=f"Unpruned (depth={clf_unpruned.get_depth()}, leaves={clf_unpruned.get_n_leaves()})\nTrain: {accuracy_score(y_train, clf_unpruned.predict(X_train)):.1%} | Test: {accuracy_score(y_test, clf_unpruned.predict(X_test)):.1%}")
        plt.tight_layout()
        st.pyplot(fig_up)

    with prune_col2:
        clf_pruned = DecisionTreeClassifier(max_depth=None, ccp_alpha=ccp_alpha, criterion=criterion, random_state=int(random_seed))
        clf_pruned.fit(X_train, y_train)
        fig_pr, ax_pr = plt.subplots(figsize=(6, 5))
        fig_pr.patch.set_facecolor('#fafafa')
        ax_pr.set_facecolor('#fafafa')
        plot_decision_boundary(clf_pruned, X, y, ax_pr,
                               title=f"Pruned (depth={clf_pruned.get_depth()}, leaves={clf_pruned.get_n_leaves()}, α={ccp_alpha})\nTrain: {accuracy_score(y_train, clf_pruned.predict(X_train)):.1%} | Test: {accuracy_score(y_test, clf_pruned.predict(X_test)):.1%}")
        plt.tight_layout()
        st.pyplot(fig_pr)

    # CCP Alpha Path
    st.markdown("---")
    st.markdown("#### Cost Complexity Pruning Path")

    path = clf_unpruned.cost_complexity_pruning_path(X_train, y_train)
    ccp_alphas = path.ccp_alphas
    impurities = path.impurities

    pruning_train_scores = []
    pruning_test_scores = []
    pruning_leaves = []

    for alpha in ccp_alphas:
        clf_a = DecisionTreeClassifier(ccp_alpha=alpha, criterion=criterion, random_state=int(random_seed))
        clf_a.fit(X_train, y_train)
        pruning_train_scores.append(accuracy_score(y_train, clf_a.predict(X_train)))
        pruning_test_scores.append(accuracy_score(y_test, clf_a.predict(X_test)))
        pruning_leaves.append(clf_a.get_n_leaves())

    fig_ccp, (ax_ccp1, ax_ccp2) = plt.subplots(1, 2, figsize=(14, 5))
    fig_ccp.patch.set_facecolor('#fafafa')

    ax_ccp1.set_facecolor('#fafafa')
    ax_ccp1.plot(ccp_alphas, pruning_train_scores, 'o-', color='#1e88e5', linewidth=2, markersize=4, label='Train', alpha=0.8)
    ax_ccp1.plot(ccp_alphas, pruning_test_scores, 's-', color='#e53935', linewidth=2, markersize=4, label='Test', alpha=0.8)
    ax_ccp1.set_xlabel("ccp_alpha", fontsize=10, fontweight=500)
    ax_ccp1.set_ylabel("Accuracy", fontsize=10, fontweight=500)
    ax_ccp1.set_title("Accuracy vs Pruning Strength", fontsize=12, fontweight=600)
    ax_ccp1.legend(fontsize=9)
    ax_ccp1.grid(True, alpha=0.2)

    ax_ccp2.set_facecolor('#fafafa')
    ax_ccp2.plot(ccp_alphas, pruning_leaves, 'D-', color='#7c3aed', linewidth=2, markersize=4, alpha=0.8)
    ax_ccp2.set_xlabel("ccp_alpha", fontsize=10, fontweight=500)
    ax_ccp2.set_ylabel("Number of Leaves", fontsize=10, fontweight=500)
    ax_ccp2.set_title("Tree Complexity vs Pruning Strength", fontsize=12, fontweight=600)
    ax_ccp2.grid(True, alpha=0.2)

    plt.tight_layout()
    st.pyplot(fig_ccp)

    st.markdown('<div class="observe-box"> <strong>What to observe:</strong> Increasing noise makes the decision boundary more jagged and erratic. Pruning simplifies the boundary — some training accuracy is sacrificed, but test accuracy often improves. The CCP path shows the trade-off: as alpha increases, the tree shrinks and eventually underfits.</div>', unsafe_allow_html=True)


#  Footer 
st.markdown("---")
st.markdown("""
<div style="text-align: center; padding: 20px; color: #9e9e9e; font-size: 0.85rem;">
     <strong>Decision Tree Interactive Visualizer</strong> · Built with Streamlit & Scikit-learn<br>
    Explore how trees partition data, choose splits, and make predictions.
</div>
""", unsafe_allow_html=True)