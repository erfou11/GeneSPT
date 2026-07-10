
import numpy as np
import pandas as pd
import scanpy as sc
import anndata as ad
from sklearn.cluster import MiniBatchKMeans
from sklearn.preprocessing import MaxAbsScaler

def load_scrna_from_txt(
    counts_path,
    normalize=True,
    store_raw_layer=True,
    target_genes=None,
    max_cells=5000,
    seed=0,
    target_sum=1e4,
    min_library_size=1.0,
    maxabs_scale=True,
):
    """
    Robust scRNA loader:
    - treat first column as index
    - auto-detect whether genes are rows or columns
    - keep only overlapping genes with ST
    - optional downsampling of cells
    """
    try:
        df = pd.read_csv(counts_path, sep="\t", index_col=0)
        if df.shape[1] == 0:
            raise ValueError("empty frame after tab read")
    except Exception:
        df = pd.read_csv(counts_path, sep=r"\s+", engine="python", index_col=0)

    df = df.apply(pd.to_numeric, errors="coerce")
    df = df.dropna(axis=0, how="all").dropna(axis=1, how="all").fillna(0.0)

    if target_genes is not None:
        target_genes = set(map(str, target_genes))

        col_overlap = len(set(map(str, df.columns)).intersection(target_genes))
        idx_overlap = len(set(map(str, df.index)).intersection(target_genes))
        print(f"[scRNA loader] overlap with ST genes -> columns: {col_overlap}, rows: {idx_overlap}")

        if idx_overlap > col_overlap:
            keep_genes = [g for g in df.index if str(g) in target_genes]
            if len(keep_genes) == 0:
                raise ValueError("No overlapping genes found on scRNA row index")
            df = df.loc[keep_genes].T
            print(f"[scRNA loader] detected gene x cell matrix, transposed to cell x gene: {df.shape}")
        else:
            keep_genes = [g for g in df.columns if str(g) in target_genes]
            if len(keep_genes) == 0:
                raise ValueError("No overlapping genes found on scRNA columns")
            df = df[keep_genes]
            print(f"[scRNA loader] detected cell x gene matrix: {df.shape}")
    else:
        print(f"[scRNA loader] loaded without target gene filtering: {df.shape}")

    if max_cells is not None and len(df) > max_cells:
        rng = np.random.default_rng(seed)
        keep_idx = rng.choice(len(df), size=max_cells, replace=False)
        df = df.iloc[np.sort(keep_idx)].copy()
        print(f"[scRNA loader] downsampled cells to {len(df)}")

    gene_names = list(map(str, df.columns))
    X = df.values.astype(np.float32)

    adata = ad.AnnData(X)
    adata.var_names = gene_names
    adata.obs_names = [f"cell_{i}" for i in range(X.shape[0])]

    if store_raw_layer:
        adata.layers["counts"] = X.copy()

    if normalize:
        size_factors = np.clip(X.sum(axis=1).astype(np.float32), float(min_library_size), None)
        X_norm = (X / size_factors[:, None]) * float(target_sum)
        adata.X = np.log1p(X_norm).astype(np.float32)
        adata.obs["size_factor"] = size_factors
        adata.uns["normalization_target_sum"] = float(target_sum)
        adata.uns["normalization_log1p"] = True
        if maxabs_scale:
            scaler = MaxAbsScaler()
            adata.X = scaler.fit_transform(adata.X.T).T.astype(np.float32)
            adata.uns["normalization_maxabs"] = True

    return adata

def align_scrna_to_st(adata_st, adata_scrna):
    common_genes = adata_st.var_names.intersection(adata_scrna.var_names)
    if len(common_genes) == 0:
        raise ValueError("No overlapping genes between ST and scRNA reference")
    adata_st_sub = adata_st[:, common_genes].copy()
    adata_scrna_sub = adata_scrna[:, common_genes].copy()
    return adata_st_sub, adata_scrna_sub

def _softmax(x, temperature=0.2):
    x = x / max(float(temperature), 1e-6)
    x = x - np.max(x)
    ex = np.exp(x)
    return ex / (np.sum(ex) + 1e-12)


def _default_tier_resolution_weights(n_scales, sigma=0.28):
    if n_scales <= 0:
        raise ValueError("n_scales must be positive")
    if n_scales == 1:
        return np.ones((3, 1), dtype=np.float32)

    positions = np.linspace(0.0, 1.0, n_scales, dtype=np.float32)
    anchors = np.asarray([0.12, 0.50, 0.88], dtype=np.float32)
    weights = []
    for anchor in anchors:
        raw = np.exp(-((positions - anchor) ** 2) / max(2.0 * sigma * sigma, 1e-6))
        raw = raw / np.clip(raw.sum(), 1e-12, None)
        weights.append(raw.astype(np.float32))
    return np.stack(weights, axis=0)


def _build_cell_prototypes(X_sc, n_prototypes, random_state, return_labels=False):
    n_prototypes = min(int(n_prototypes), X_sc.shape[0])
    km = MiniBatchKMeans(
        n_clusters=n_prototypes,
        random_state=random_state,
        batch_size=min(1024, X_sc.shape[0]),
        n_init=5,
    )
    labels = km.fit_predict(X_sc)

    prototypes = np.zeros((n_prototypes, X_sc.shape[1]), dtype=np.float32)
    for k in range(n_prototypes):
        members = X_sc[labels == k]
        if len(members) == 0:
            prototypes[k] = X_sc[np.random.randint(0, X_sc.shape[0])]
        else:
            prototypes[k] = members.mean(axis=0)
    prototypes = prototypes.astype(np.float32)
    if return_labels:
        return prototypes, labels.astype(np.int64)
    return prototypes


def _safe_corr_vector(X, y):
    X = np.asarray(X, dtype=np.float32)
    y = np.asarray(y, dtype=np.float32).reshape(-1)
    Xc = X - X.mean(axis=0, keepdims=True)
    yc = y - y.mean()
    denom = np.linalg.norm(Xc, axis=0) * max(np.linalg.norm(yc), 1e-12)
    corr = (Xc * yc[:, None]).sum(axis=0) / np.clip(denom, 1e-12, None)
    return np.nan_to_num(corr, nan=0.0, posinf=0.0, neginf=0.0).astype(np.float32)


def _solve_ridge_projection(X, Y, ridge_lambda=1.0):
    X = np.asarray(X, dtype=np.float32)
    Y = np.asarray(Y, dtype=np.float32)
    if X.ndim != 2 or Y.ndim != 2:
        raise ValueError("X and Y must be 2D arrays")
    if X.shape[0] != Y.shape[0]:
        raise ValueError("X and Y must have the same number of rows")
    if X.shape[1] == 0 or Y.shape[1] == 0:
        return np.zeros((X.shape[1], Y.shape[1]), dtype=np.float32)

    xtx = X.T @ X
    xty = X.T @ Y
    ridge = float(max(ridge_lambda, 1e-6))
    eye = np.eye(xtx.shape[0], dtype=np.float32)
    try:
        coef = np.linalg.solve(xtx + ridge * eye, xty)
    except np.linalg.LinAlgError:
        coef = np.linalg.pinv(xtx + ridge * eye) @ xty
    return coef.astype(np.float32)


def _truncated_svd_basis(X, n_components):
    X = np.asarray(X, dtype=np.float32)
    if X.ndim != 2:
        raise ValueError("X must be a 2D array")
    if X.shape[0] == 0 or X.shape[1] == 0:
        return np.zeros((X.shape[1], 0), dtype=np.float32), np.zeros((1, X.shape[1]), dtype=np.float32), np.zeros(0, dtype=np.float32)

    mean = X.mean(axis=0, keepdims=True).astype(np.float32)
    X_centered = X - mean
    if np.allclose(X_centered, 0.0):
        return np.zeros((X.shape[1], 0), dtype=np.float32), mean, np.zeros(0, dtype=np.float32)

    _, singular_vals, vt = np.linalg.svd(X_centered, full_matrices=False)
    k = min(int(n_components), vt.shape[0], X.shape[1])
    if k <= 0:
        return np.zeros((X.shape[1], 0), dtype=np.float32), mean, np.zeros(0, dtype=np.float32)

    basis = vt[:k].T.astype(np.float32)
    energy = np.square(singular_vals.astype(np.float32))
    explained = (energy[:k] / np.clip(energy.sum(), 1e-12, None)).astype(np.float32)
    return basis, mean.astype(np.float32), explained


def _derive_hidden_program_anchors(
    X_sc,
    hidden_idx,
    visible_idx,
    n_programs,
    program_anchor_topk,
    random_state,
):
    hidden_idx = np.asarray(hidden_idx, dtype=np.int64).reshape(-1)
    visible_idx = np.asarray(visible_idx, dtype=np.int64).reshape(-1)
    n_programs = max(1, min(int(n_programs), hidden_idx.size))

    gene_program_features = X_sc[:, hidden_idx].T.astype(np.float32)
    if gene_program_features.shape[0] == 1:
        hidden_program_labels = np.zeros(1, dtype=np.int64)
    else:
        gene_program_features = gene_program_features - gene_program_features.mean(axis=1, keepdims=True)
        gene_program_features = gene_program_features / np.clip(
            gene_program_features.std(axis=1, keepdims=True),
            1e-6,
            None,
        )
        gene_km = MiniBatchKMeans(
            n_clusters=n_programs,
            random_state=random_state,
            batch_size=min(256, gene_program_features.shape[0]),
            n_init=10,
        )
        hidden_program_labels = gene_km.fit_predict(gene_program_features).astype(np.int64)

    full_gene_program_labels = np.full(X_sc.shape[1], -1, dtype=np.int64)
    full_gene_program_labels[hidden_idx] = hidden_program_labels

    sc_visible = X_sc[:, visible_idx].astype(np.float32)
    program_anchor_indices = []
    for program_idx in range(n_programs):
        program_hidden_local = np.where(hidden_program_labels == program_idx)[0]
        program_hidden_global = hidden_idx[program_hidden_local]
        if program_hidden_global.size == 0:
            program_anchor_indices.append(np.asarray([], dtype=np.int64))
            continue

        program_signal = X_sc[:, program_hidden_global].mean(axis=1).astype(np.float32)
        corr = _safe_corr_vector(sc_visible, program_signal)
        positive_order = np.argsort(-corr)
        positive_mask = corr[positive_order] > 0
        selected_local = positive_order[positive_mask][: max(1, int(program_anchor_topk))]
        if selected_local.size == 0:
            fallback_order = np.argsort(-np.abs(corr))
            selected_local = fallback_order[: max(1, int(program_anchor_topk))]
        program_anchor_indices.append(visible_idx[selected_local].astype(np.int64))

    return hidden_program_labels.astype(np.int64), full_gene_program_labels.astype(np.int64), program_anchor_indices

def build_prototype_reference_from_scrna(
    adata_st,
    adata_scrna,
    observed_mask,
    n_prototypes=48,
    temperature=0.2,
    expr_layer=None,
    random_state=0,
    top_k=8,
    log_tag=None,
    return_details=False,
):
    """
    Build a sharpened prototype-mixture reference for each ST spot:
      1) cluster scRNA cells into a small set of prototypes
      2) each spot only mixes the top-k most similar prototypes on observed genes
    """
    X_st = np.asarray(
        adata_st.X if expr_layer is None else adata_st.layers[expr_layer],
        dtype=np.float32
    )
    X_sc = np.asarray(
        adata_scrna.X if expr_layer is None else adata_scrna.layers[expr_layer],
        dtype=np.float32
    )
    observed_mask = np.asarray(observed_mask, dtype=np.float32)

    if X_st.shape[1] != X_sc.shape[1]:
        raise ValueError(f"ST genes {X_st.shape[1]} != scRNA genes {X_sc.shape[1]}")
    if observed_mask.shape != X_st.shape:
        raise ValueError(f"observed_mask shape {observed_mask.shape} != ST shape {X_st.shape}")

    n_prototypes = min(int(n_prototypes), X_sc.shape[0])
    prefix = "[scRNA proto]" if log_tag is None else f"[scRNA proto:{log_tag}]"
    print(f"{prefix} clustering {X_sc.shape[0]} cells into {n_prototypes} prototypes")

    km = MiniBatchKMeans(
        n_clusters=n_prototypes,
        random_state=random_state,
        batch_size=min(1024, X_sc.shape[0]),
        n_init=5,
    )
    labels = km.fit_predict(X_sc)

    prototypes = np.zeros((n_prototypes, X_sc.shape[1]), dtype=np.float32)
    for k in range(n_prototypes):
        members = X_sc[labels == k]
        if len(members) == 0:
            prototypes[k] = X_sc[np.random.randint(0, X_sc.shape[0])]
        else:
            prototypes[k] = members.mean(axis=0)

    ref = np.zeros_like(X_st, dtype=np.float32)
    confidence = np.zeros(X_st.shape[0], dtype=np.float32)
    entropy_confidence = np.zeros(X_st.shape[0], dtype=np.float32)
    agreement_confidence = np.zeros(X_st.shape[0], dtype=np.float32)
    coverage_confidence = np.zeros(X_st.shape[0], dtype=np.float32)

    for i in range(X_st.shape[0]):
        keep = observed_mask[i] > 0.5
        observed_count = int(keep.sum())

        if observed_count == 0:
            weights = np.full(n_prototypes, 1.0 / max(n_prototypes, 1), dtype=np.float32)
            ref[i] = (prototypes * weights[:, None]).sum(axis=0)
            entropy_conf = 0.0
            agreement_conf = 0.0
            coverage_conf = 0.0
            confidence[i] = 0.0
            entropy_confidence[i] = 0.0
            agreement_confidence[i] = 0.0
            coverage_confidence[i] = 0.0
            if (i + 1) % 500 == 0:
                print(f"{prefix} built {i + 1}/{X_st.shape[0]} spots")
            continue

        st_vec = X_st[i, keep]
        proto_sub = prototypes[:, keep]

        st_norm = np.linalg.norm(st_vec)
        proto_norm = np.linalg.norm(proto_sub, axis=1)
        sims = (proto_sub @ st_vec) / np.clip(proto_norm * max(st_norm, 1e-12), 1e-12, None)

        k = min(max(1, int(top_k)), n_prototypes)
        top_idx = np.argpartition(sims, -k)[-k:]
        top_sims = sims[top_idx]
        top_weights = _softmax(top_sims, temperature=temperature)

        weights = np.zeros_like(sims, dtype=np.float32)
        weights[top_idx] = top_weights.astype(np.float32)
        ref[i] = (prototypes * weights[:, None]).sum(axis=0)

        if len(top_weights) <= 1:
            entropy_conf = 1.0
        else:
            entropy = -np.sum(top_weights * np.log(np.clip(top_weights, 1e-12, None)))
            entropy_conf = 1.0 - float(entropy / np.log(len(top_weights)))

        ref_obs = ref[i, keep]
        ref_norm = np.linalg.norm(ref_obs)
        agreement = float((ref_obs @ st_vec) / np.clip(ref_norm * max(st_norm, 1e-12), 1e-12, None))
        agreement_conf = float(np.clip(0.5 * (agreement + 1.0), 0.0, 1.0))
        coverage_conf = float(np.clip(observed_count / 16.0, 0.0, 1.0))
        spot_conf = float(np.clip(0.55 * agreement_conf + 0.35 * entropy_conf + 0.10 * coverage_conf, 0.0, 1.0))

        confidence[i] = spot_conf
        entropy_confidence[i] = float(np.clip(entropy_conf, 0.0, 1.0))
        agreement_confidence[i] = agreement_conf
        coverage_confidence[i] = coverage_conf

        if (i + 1) % 500 == 0:
            print(f"{prefix} built {i + 1}/{X_st.shape[0]} spots")

    ref = ref.astype(np.float32)
    if not return_details:
        return ref

    details = {
        "confidence": confidence.astype(np.float32),
        "entropy_confidence": entropy_confidence.astype(np.float32),
        "agreement_confidence": agreement_confidence.astype(np.float32),
        "coverage_confidence": coverage_confidence.astype(np.float32),
    }
    return ref, details


def build_relation_reference_from_scrna(
    adata_st,
    observed_mask,
    relation_projection,
    expr_layer=None,
):
    """
    Build a hidden-gene relation prior from observed ST genes only.

    Leakage safety:
    - `observed_mask` must mark hidden/target genes as 0
    - the ST contribution is always `X_st * observed_mask`
    - hidden-gene values come only from the scRNA-derived relation projection
    """
    X_st = np.asarray(
        adata_st.X if expr_layer is None else adata_st.layers[expr_layer],
        dtype=np.float32,
    )
    observed_mask = np.asarray(observed_mask, dtype=np.float32)
    relation_projection = np.asarray(relation_projection, dtype=np.float32)

    if observed_mask.shape != X_st.shape:
        raise ValueError(f"observed_mask shape {observed_mask.shape} != ST shape {X_st.shape}")
    if relation_projection.ndim != 2:
        raise ValueError(f"relation_projection must be 2D, got shape {relation_projection.shape}")
    if relation_projection.shape[0] != X_st.shape[1]:
        raise ValueError(
            f"relation_projection rows {relation_projection.shape[0]} != gene dimension {X_st.shape[1]}"
        )

    condition_expr = X_st * observed_mask
    return np.matmul(condition_expr, relation_projection).astype(np.float32)


def fuse_hidden_gene_priors(
    proto_hidden_ref,
    relation_hidden_ref,
    prior_mode="dual_prior",
    prior_fusion_mode="adaptive_confidence",
    prior_fixed_weight=0.5,
    proto_confidence=None,
):
    """
    Fuse prototype and relation priors into a hidden-gene prior matrix.

    Supported prior modes:
      - no_prior
      - proto_only
      - relation_only
      - dual_prior
      - shuffled_prior

    Supported fusion modes for dual_prior:
      - simple_mean
      - fixed_weight
      - adaptive_confidence
    """
    prior_mode = str(prior_mode)
    prior_fusion_mode = str(prior_fusion_mode)

    if prior_mode == "no_prior":
        return None, None

    if proto_hidden_ref is not None:
        proto_hidden_ref = np.asarray(proto_hidden_ref, dtype=np.float32)
    if relation_hidden_ref is not None:
        relation_hidden_ref = np.asarray(relation_hidden_ref, dtype=np.float32)

    if prior_mode == "proto_only":
        if proto_hidden_ref is None:
            raise ValueError("proto_only prior mode requires prototype hidden reference")
        fusion_weight = np.ones((proto_hidden_ref.shape[0], 1), dtype=np.float32)
        return proto_hidden_ref.astype(np.float32), fusion_weight

    if prior_mode == "relation_only":
        if relation_hidden_ref is None:
            raise ValueError("relation_only prior mode requires relation hidden reference")
        fusion_weight = np.zeros((relation_hidden_ref.shape[0], 1), dtype=np.float32)
        return relation_hidden_ref.astype(np.float32), fusion_weight

    if prior_mode == "shuffled_prior":
        prior_mode = "dual_prior"

    if prior_mode != "dual_prior":
        raise ValueError(f"Unsupported prior_mode={prior_mode!r}")
    if proto_hidden_ref is None or relation_hidden_ref is None:
        raise ValueError("dual_prior requires both prototype and relation hidden references")
    if proto_hidden_ref.shape != relation_hidden_ref.shape:
        raise ValueError(
            f"prototype hidden ref shape {proto_hidden_ref.shape} != relation hidden ref shape {relation_hidden_ref.shape}"
        )

    if prior_fusion_mode == "simple_mean":
        fusion_weight = np.full((proto_hidden_ref.shape[0], 1), 0.5, dtype=np.float32)
    elif prior_fusion_mode == "fixed_weight":
        fixed_weight = float(np.clip(prior_fixed_weight, 0.0, 1.0))
        fusion_weight = np.full((proto_hidden_ref.shape[0], 1), fixed_weight, dtype=np.float32)
    elif prior_fusion_mode == "adaptive_confidence":
        if proto_confidence is None:
            raise ValueError("adaptive_confidence fusion requires proto_confidence")
        fusion_weight = np.asarray(proto_confidence, dtype=np.float32).reshape(-1, 1)
        if fusion_weight.shape[0] != proto_hidden_ref.shape[0]:
            raise ValueError(
                f"proto_confidence rows {fusion_weight.shape[0]} != hidden prior rows {proto_hidden_ref.shape[0]}"
            )
        fusion_weight = np.clip(fusion_weight, 0.0, 1.0).astype(np.float32)
    else:
        raise ValueError(f"Unsupported prior_fusion_mode={prior_fusion_mode!r}")

    fused = (
        fusion_weight * proto_hidden_ref
        + (1.0 - fusion_weight) * relation_hidden_ref
    ).astype(np.float32)
    return fused, fusion_weight.astype(np.float32)


def build_hidden_gene_prior_guidance(
    adata_st,
    adata_scrna,
    observed_mask,
    hidden_gene_idx,
    relation_projection,
    prior_mode="dual_prior",
    prior_fusion_mode="adaptive_confidence",
    prior_fixed_weight=0.5,
    n_prototypes=48,
    temperature=0.2,
    expr_layer=None,
    random_state=0,
    top_k=8,
    log_tag=None,
):
    """
    Build a spot-adaptive hidden-gene prior for Stage B condition injection.

    This helper keeps the existing prototype/reference logic intact but makes the
    source and fusion modes explicit and ablatable.
    """
    hidden_gene_idx = np.asarray(hidden_gene_idx, dtype=np.int64).reshape(-1)
    if hidden_gene_idx.size == 0:
        return {
            "hidden_prior": None,
            "hidden_gene_idx": hidden_gene_idx.astype(np.int64),
            "prototype_reference": None,
            "relation_reference": None,
            "prototype_confidence": None,
            "fusion_weight": None,
            "summary": {
                "prior_mode": str(prior_mode),
                "prior_fusion_mode": str(prior_fusion_mode),
                "hidden_gene_count": 0,
            },
        }

    shuffle_spot_prior = str(prior_mode) == "shuffled_prior"
    fusion_prior_mode = "dual_prior" if shuffle_spot_prior else str(prior_mode)

    proto_ref, proto_details = build_prototype_reference_from_scrna(
        adata_st=adata_st,
        adata_scrna=adata_scrna,
        observed_mask=observed_mask,
        n_prototypes=n_prototypes,
        temperature=temperature,
        expr_layer=expr_layer,
        random_state=random_state,
        top_k=top_k,
        log_tag=log_tag,
        return_details=True,
    )
    relation_ref = build_relation_reference_from_scrna(
        adata_st=adata_st,
        observed_mask=observed_mask,
        relation_projection=relation_projection,
        expr_layer=expr_layer,
    )

    proto_hidden_ref = proto_ref[:, hidden_gene_idx].astype(np.float32)
    relation_hidden_ref = relation_ref.astype(np.float32)
    proto_confidence = np.asarray(proto_details["confidence"], dtype=np.float32).reshape(-1, 1)
    if relation_hidden_ref.shape != proto_hidden_ref.shape:
        raise ValueError(
            f"relation hidden ref shape {relation_hidden_ref.shape} != prototype hidden ref shape {proto_hidden_ref.shape}"
        )

    hidden_prior, fusion_weight = fuse_hidden_gene_priors(
        proto_hidden_ref=proto_hidden_ref,
        relation_hidden_ref=relation_hidden_ref,
        prior_mode=fusion_prior_mode,
        prior_fusion_mode=prior_fusion_mode,
        prior_fixed_weight=prior_fixed_weight,
        proto_confidence=proto_confidence,
    )

    if shuffle_spot_prior and hidden_prior is not None and hidden_prior.shape[0] > 1:
        roll_shift = 1
        hidden_prior = np.roll(hidden_prior, shift=roll_shift, axis=0)
        proto_ref = np.roll(proto_ref, shift=roll_shift, axis=0)
        relation_ref = np.roll(relation_ref, shift=roll_shift, axis=0)
        proto_confidence = np.roll(proto_confidence, shift=roll_shift, axis=0)
        if fusion_weight is not None:
            fusion_weight = np.roll(fusion_weight, shift=roll_shift, axis=0)
    else:
        roll_shift = 0

    return {
        "hidden_prior": hidden_prior,
        "hidden_gene_idx": hidden_gene_idx.astype(np.int64),
        "prototype_reference": proto_ref.astype(np.float32),
        "relation_reference": relation_ref.astype(np.float32),
        "prototype_confidence": proto_confidence.astype(np.float32),
        "fusion_weight": fusion_weight.astype(np.float32) if fusion_weight is not None else None,
        "summary": {
            "prior_mode": str(prior_mode),
            "prior_fusion_mode": str(prior_fusion_mode),
            "prior_fixed_weight": float(np.clip(prior_fixed_weight, 0.0, 1.0)),
            "hidden_gene_count": int(hidden_gene_idx.size),
            "shuffle_spot_prior": bool(shuffle_spot_prior),
            "shuffle_roll_shift": int(roll_shift),
            "prototype_confidence_mean": float(proto_confidence.mean()),
            "prototype_confidence_std": float(proto_confidence.std()),
            "agreement_mean": float(np.mean(proto_details["agreement_confidence"])),
            "entropy_mean": float(np.mean(proto_details["entropy_confidence"])),
            "coverage_mean": float(np.mean(proto_details["coverage_confidence"])),
            "fusion_weight_mean": float(np.mean(fusion_weight)) if fusion_weight is not None else None,
            "fusion_weight_std": float(np.std(fusion_weight)) if fusion_weight is not None else None,
        },
    }


def build_style_calibrated_reference_from_scrna(
    adata_st,
    adata_scrna,
    observed_mask,
    relation_projection,
    n_prototypes=48,
    temperature=0.2,
    expr_layer=None,
    random_state=0,
    top_k=8,
    style_rank=16,
    style_ridge=1.0,
    style_max_gate=0.35,
    style_delta_scale=1.0,
    return_details=False,
):
    """
    Build a style-calibrated hidden-gene reference.

    Content:
      - same prototype/relation blend as the stable confgate baseline
    Style:
      - infer a low-rank ST-specific residual code from observed genes
      - transfer that code to hidden genes through a scRNA-trained ridge map
    """
    X_st = np.asarray(
        adata_st.X if expr_layer is None else adata_st.layers[expr_layer],
        dtype=np.float32,
    )
    X_sc = np.asarray(
        adata_scrna.X if expr_layer is None else adata_scrna.layers[expr_layer],
        dtype=np.float32,
    )
    observed_mask = np.asarray(observed_mask, dtype=np.float32)
    relation_projection = np.asarray(relation_projection, dtype=np.float32)

    if observed_mask.shape != X_st.shape:
        raise ValueError(f"observed_mask shape {observed_mask.shape} != ST shape {X_st.shape}")
    if relation_projection.shape[0] != X_st.shape[1] or relation_projection.shape[1] != X_st.shape[1]:
        raise ValueError(
            f"relation_projection shape {relation_projection.shape} is not compatible with gene dimension {X_st.shape[1]}"
        )

    visible_idx = np.where(observed_mask.sum(axis=0) > 0)[0]
    hidden_idx = np.where(observed_mask.sum(axis=0) <= 0)[0]
    if visible_idx.size == 0 or hidden_idx.size == 0:
        raise ValueError("style-calibrated reference requires both visible and hidden genes")

    proto_ref, proto_details = build_prototype_reference_from_scrna(
        adata_st=adata_st,
        adata_scrna=adata_scrna,
        observed_mask=observed_mask,
        n_prototypes=n_prototypes,
        temperature=temperature,
        expr_layer=expr_layer,
        random_state=random_state,
        top_k=top_k,
        log_tag="style-st",
        return_details=True,
    )
    condition_expr = X_st * observed_mask
    relation_ref = np.matmul(condition_expr, relation_projection).astype(np.float32)
    proto_confidence = np.asarray(proto_details["confidence"], dtype=np.float32).reshape(-1, 1)
    content_ref = (
        proto_confidence * proto_ref
        + (1.0 - proto_confidence) * relation_ref
    ).astype(np.float32)

    sc_observed_mask = np.tile((observed_mask.sum(axis=0) > 0).astype(np.float32)[None, :], (X_sc.shape[0], 1))
    adata_sc_as_target = ad.AnnData(X_sc.astype(np.float32))
    adata_sc_as_target.var_names = adata_scrna.var_names.copy()
    sc_proto_ref, sc_proto_details = build_prototype_reference_from_scrna(
        adata_st=adata_sc_as_target,
        adata_scrna=adata_scrna,
        observed_mask=sc_observed_mask,
        n_prototypes=n_prototypes,
        temperature=temperature,
        expr_layer=None,
        random_state=random_state,
        top_k=top_k,
        log_tag="style-sc",
        return_details=True,
    )
    sc_condition_expr = X_sc * sc_observed_mask
    sc_relation_ref = np.matmul(sc_condition_expr, relation_projection).astype(np.float32)
    sc_proto_confidence = np.asarray(sc_proto_details["confidence"], dtype=np.float32).reshape(-1, 1)
    sc_content_ref = (
        sc_proto_confidence * sc_proto_ref
        + (1.0 - sc_proto_confidence) * sc_relation_ref
    ).astype(np.float32)

    st_observed_residual = (X_st[:, visible_idx] - content_ref[:, visible_idx]).astype(np.float32)
    style_basis, style_mean, style_explained = _truncated_svd_basis(
        st_observed_residual,
        n_components=max(1, int(style_rank)),
    )

    adapted_ref = content_ref.copy()
    style_gate = np.zeros(X_st.shape[0], dtype=np.float32)
    style_fit = np.zeros(X_st.shape[0], dtype=np.float32)
    style_energy = np.zeros(X_st.shape[0], dtype=np.float32)
    style_delta_hidden = np.zeros((X_st.shape[0], hidden_idx.size), dtype=np.float32)

    if style_basis.shape[1] > 0:
        sc_observed_residual = (X_sc[:, visible_idx] - sc_content_ref[:, visible_idx]).astype(np.float32)
        sc_hidden_residual = (X_sc[:, hidden_idx] - sc_content_ref[:, hidden_idx]).astype(np.float32)

        sc_style_coeff = np.matmul(sc_observed_residual - style_mean, style_basis).astype(np.float32)
        hidden_style_map = _solve_ridge_projection(
            sc_style_coeff,
            sc_hidden_residual,
            ridge_lambda=style_ridge,
        )

        st_style_coeff = np.matmul(st_observed_residual - style_mean, style_basis).astype(np.float32)
        style_delta_hidden = np.matmul(st_style_coeff, hidden_style_map).astype(np.float32)

        recon_observed_residual = np.matmul(st_style_coeff, style_basis.T).astype(np.float32)
        recon_error = np.mean(np.square(st_observed_residual - style_mean - recon_observed_residual), axis=1)
        residual_var = np.mean(np.square(st_observed_residual - style_mean), axis=1)
        style_fit = 1.0 - (recon_error / np.clip(residual_var, 1e-6, None))
        style_fit = np.clip(style_fit, 0.0, 1.0).astype(np.float32)

        style_energy = np.linalg.norm(st_style_coeff, axis=1).astype(np.float32)
        energy_norm = style_energy / np.clip(np.percentile(style_energy, 90), 1e-6, None)
        energy_norm = np.clip(energy_norm, 0.0, 1.0).astype(np.float32)

        agreement_conf = np.asarray(proto_details["agreement_confidence"], dtype=np.float32)
        coverage_conf = np.asarray(proto_details["coverage_confidence"], dtype=np.float32)
        raw_gate = (
            style_fit
            * energy_norm
            * (1.0 - agreement_conf)
            * (0.5 + 0.5 * coverage_conf)
        )
        style_gate = np.clip(raw_gate, 0.0, float(style_max_gate)).astype(np.float32)

        adapted_hidden = (
            content_ref[:, hidden_idx]
            + float(style_delta_scale) * style_gate[:, None] * style_delta_hidden
        ).astype(np.float32)
        adapted_hidden = np.clip(adapted_hidden, 0.0, None)
        adapted_ref[:, hidden_idx] = adapted_hidden

    adapted_ref = adapted_ref.astype(np.float32)
    if not return_details:
        return adapted_ref

    details = {
        "content_reference": content_ref.astype(np.float32),
        "prototype_reference": proto_ref.astype(np.float32),
        "relation_reference": relation_ref.astype(np.float32),
        "prototype_confidence": proto_confidence.reshape(-1).astype(np.float32),
        "agreement_confidence": np.asarray(proto_details["agreement_confidence"], dtype=np.float32),
        "coverage_confidence": np.asarray(proto_details["coverage_confidence"], dtype=np.float32),
        "style_basis_observed": style_basis.astype(np.float32),
        "style_basis_mean": style_mean.astype(np.float32),
        "style_basis_explained": style_explained.astype(np.float32),
        "style_gate": style_gate.astype(np.float32),
        "style_fit": style_fit.astype(np.float32),
        "style_energy": style_energy.astype(np.float32),
        "style_hidden_delta": style_delta_hidden.astype(np.float32),
        "visible_gene_idx": visible_idx.astype(np.int64),
        "hidden_gene_idx": hidden_idx.astype(np.int64),
    }
    return adapted_ref, details


def build_multi_resolution_prototype_reference_from_scrna(
    adata_st,
    adata_scrna,
    observed_mask,
    hidden_gene_tier,
    prototype_scales,
    base_temperature=0.2,
    expr_layer=None,
    random_state=0,
    top_k=8,
):
    """
    Build multiple prototype banks at different resolutions and blend them per gene
    according to hidden-gene difficulty tiers:
      - easy genes lean toward coarse prototypes
      - medium genes lean toward middle-scale prototypes
      - hard genes lean toward fine prototypes
    """
    prototype_scales = [int(s) for s in prototype_scales]
    if len(prototype_scales) == 0:
        raise ValueError("prototype_scales must not be empty")

    hidden_gene_tier = np.asarray(hidden_gene_tier, dtype=np.int64).reshape(-1)

    scale_refs = []
    scale_metadata = []
    mid_scale = float(prototype_scales[len(prototype_scales) // 2])

    for scale in prototype_scales:
        scale_ratio = max(scale / max(mid_scale, 1.0), 1e-6)
        scale_top_k = max(1, int(round(top_k * np.sqrt(scale_ratio))))
        scale_temperature = float(base_temperature / np.sqrt(scale_ratio))
        ref = build_prototype_reference_from_scrna(
            adata_st=adata_st,
            adata_scrna=adata_scrna,
            observed_mask=observed_mask,
            n_prototypes=scale,
            temperature=scale_temperature,
            expr_layer=expr_layer,
            random_state=random_state,
            top_k=scale_top_k,
            log_tag=f"scale{scale}",
        )
        scale_refs.append(ref.astype(np.float32))
        scale_metadata.append(
            {
                "scale": int(scale),
                "top_k": int(scale_top_k),
                "temperature": float(scale_temperature),
            }
        )

    tier_resolution_weights = _default_tier_resolution_weights(len(scale_refs))
    gene_tier = np.clip(hidden_gene_tier, 0, 2)
    gene_resolution_weights = tier_resolution_weights[gene_tier]

    combined_ref = np.zeros_like(scale_refs[0], dtype=np.float32)
    for scale_idx, ref in enumerate(scale_refs):
        combined_ref += ref * gene_resolution_weights[:, scale_idx][None, :]

    details = {
        "scale_refs": scale_refs,
        "scale_metadata": scale_metadata,
        "tier_resolution_weights": tier_resolution_weights.astype(np.float32),
    }
    return combined_ref.astype(np.float32), details


def build_program_conditioned_prototype_reference_from_scrna(
    adata_st,
    adata_scrna,
    observed_mask,
    hidden_gene_mask,
    n_prototypes=48,
    n_programs=3,
    program_anchor_topk=16,
    temperature=0.2,
    expr_layer=None,
    random_state=0,
    top_k=8,
    return_details=False,
):
    X_st = np.asarray(
        adata_st.X if expr_layer is None else adata_st.layers[expr_layer],
        dtype=np.float32,
    )
    X_sc = np.asarray(
        adata_scrna.X if expr_layer is None else adata_scrna.layers[expr_layer],
        dtype=np.float32,
    )
    observed_mask = np.asarray(observed_mask, dtype=np.float32)
    hidden_gene_mask = np.asarray(hidden_gene_mask, dtype=np.float32).reshape(-1) > 0.5

    if X_st.shape[1] != X_sc.shape[1]:
        raise ValueError(f"ST genes {X_st.shape[1]} != scRNA genes {X_sc.shape[1]}")
    if observed_mask.shape != X_st.shape:
        raise ValueError(f"observed_mask shape {observed_mask.shape} != ST shape {X_st.shape}")
    if hidden_gene_mask.shape[0] != X_st.shape[1]:
        raise ValueError("hidden_gene_mask length must match gene dimension")

    hidden_idx = np.where(hidden_gene_mask)[0]
    visible_idx = np.where(~hidden_gene_mask)[0]
    if hidden_idx.size == 0:
        raise ValueError("program-conditioned prototype requires at least one hidden gene")
    if visible_idx.size == 0:
        raise ValueError("program-conditioned prototype requires visible genes as anchors")

    n_programs = max(1, min(int(n_programs), hidden_idx.size))
    print(
        f"[scRNA programproto] clustering {X_sc.shape[0]} cells into {min(int(n_prototypes), X_sc.shape[0])} prototypes "
        f"and {hidden_idx.size} hidden genes into {n_programs} programs"
    )

    prototypes = _build_cell_prototypes(X_sc, n_prototypes=n_prototypes, random_state=random_state)
    hidden_program_labels, full_gene_program_labels, program_anchor_indices = _derive_hidden_program_anchors(
        X_sc=X_sc,
        hidden_idx=hidden_idx,
        visible_idx=visible_idx,
        n_programs=n_programs,
        program_anchor_topk=program_anchor_topk,
        random_state=random_state,
    )

    ref = np.zeros_like(X_st, dtype=np.float32)
    gene_confidence = np.ones_like(X_st, dtype=np.float32)
    program_confidence = np.zeros((X_st.shape[0], n_programs), dtype=np.float32)
    for program_idx in range(n_programs):
        program_hidden_local = np.where(hidden_program_labels == program_idx)[0]
        program_hidden_global = hidden_idx[program_hidden_local]
        if program_hidden_global.size == 0:
            continue
        anchor_global = np.asarray(program_anchor_indices[program_idx], dtype=np.int64)

        for spot_idx in range(X_st.shape[0]):
            keep = observed_mask[spot_idx] > 0.5
            anchor_keep = anchor_global[keep[anchor_global]]
            if anchor_keep.size < min(4, anchor_global.size):
                visible_keep = visible_idx[keep[visible_idx]]
                if visible_keep.size >= 4:
                    anchor_keep = visible_keep
                elif anchor_global.size > 0:
                    anchor_keep = anchor_global
                else:
                    anchor_keep = visible_idx

            st_vec = X_st[spot_idx, anchor_keep]
            proto_sub = prototypes[:, anchor_keep]
            st_norm = np.linalg.norm(st_vec)
            proto_norm = np.linalg.norm(proto_sub, axis=1)
            sims = (proto_sub @ st_vec) / np.clip(proto_norm * max(st_norm, 1e-12), 1e-12, None)

            current_top_k = min(max(1, int(top_k)), prototypes.shape[0])
            top_idx = np.argpartition(sims, -current_top_k)[-current_top_k:]
            top_sims = sims[top_idx]
            top_weights = _softmax(top_sims, temperature=temperature)

            weights = np.zeros_like(sims, dtype=np.float32)
            weights[top_idx] = top_weights.astype(np.float32)
            ref_hidden = (prototypes[:, program_hidden_global] * weights[:, None]).sum(axis=0)
            ref[spot_idx, program_hidden_global] = ref_hidden.astype(np.float32)

            if len(top_weights) <= 1:
                entropy_conf = 1.0
            else:
                entropy = -np.sum(top_weights * np.log(np.clip(top_weights, 1e-12, None)))
                entropy_conf = 1.0 - float(entropy / np.log(len(top_weights)))

            ref_anchor = (prototypes[:, anchor_keep] * weights[:, None]).sum(axis=0)
            ref_norm = np.linalg.norm(ref_anchor)
            agreement = float((ref_anchor @ st_vec) / np.clip(ref_norm * max(st_norm, 1e-12), 1e-12, None))
            agreement_conf = float(np.clip(0.5 * (agreement + 1.0), 0.0, 1.0))
            coverage_conf = float(np.clip(anchor_keep.size / max(float(anchor_global.size), 1.0), 0.0, 1.0))
            spot_program_conf = float(
                np.clip(0.55 * agreement_conf + 0.35 * entropy_conf + 0.10 * coverage_conf, 0.0, 1.0)
            )
            program_confidence[spot_idx, program_idx] = spot_program_conf
            gene_confidence[spot_idx, program_hidden_global] = spot_program_conf

        print(
            f"[scRNA programproto] program {program_idx + 1}/{n_programs} "
            f"hidden_genes={program_hidden_global.size} anchors={len(anchor_global)}"
        )

    if not return_details:
        return ref.astype(np.float32)

    details = {
        "gene_confidence": gene_confidence.astype(np.float32),
        "program_confidence": program_confidence.astype(np.float32),
        "gene_program_labels": full_gene_program_labels.astype(np.int64),
        "program_anchor_indices": [x.astype(np.int64) for x in program_anchor_indices],
    }
    return ref.astype(np.float32), details


def build_prototype_translator_reference_from_scrna(
    adata_st,
    adata_scrna,
    observed_mask,
    hidden_gene_mask,
    relation_projection,
    n_prototypes=48,
    temperature=0.2,
    expr_layer=None,
    random_state=0,
    top_k=8,
    ridge_lambda=1.0,
    residual_scale=0.35,
    min_cells=64,
    return_details=False,
):
    """
    Build a prototype-conditioned translator reference:
      1) use the existing global visible->hidden relation projection as the base
      2) cluster scRNA cells into prototypes
      3) fit a conservative prototype-specific residual translator on top of the global map

    This keeps the stable global prior but lets each spot borrow a small
    cell-state-specific correction from nearby prototypes.
    """
    X_st = np.asarray(
        adata_st.X if expr_layer is None else adata_st.layers[expr_layer],
        dtype=np.float32,
    )
    X_sc = np.asarray(
        adata_scrna.X if expr_layer is None else adata_scrna.layers[expr_layer],
        dtype=np.float32,
    )
    observed_mask = np.asarray(observed_mask, dtype=np.float32)
    hidden_gene_mask = np.asarray(hidden_gene_mask, dtype=np.float32).reshape(-1) > 0.5
    relation_projection = np.asarray(relation_projection, dtype=np.float32)

    if X_st.shape[1] != X_sc.shape[1]:
        raise ValueError(f"ST genes {X_st.shape[1]} != scRNA genes {X_sc.shape[1]}")
    if observed_mask.shape != X_st.shape:
        raise ValueError(f"observed_mask shape {observed_mask.shape} != ST shape {X_st.shape}")
    if hidden_gene_mask.shape[0] != X_st.shape[1]:
        raise ValueError("hidden_gene_mask length must match gene dimension")
    if relation_projection.shape != (X_st.shape[1], X_st.shape[1]):
        raise ValueError("relation_projection must be a square gene x gene matrix")

    hidden_idx = np.where(hidden_gene_mask)[0]
    visible_idx = np.where(~hidden_gene_mask)[0]
    if hidden_idx.size == 0:
        raise ValueError("prototype translator requires at least one hidden gene")
    if visible_idx.size == 0:
        raise ValueError("prototype translator requires visible genes")

    n_prototypes = min(int(n_prototypes), X_sc.shape[0])
    print(
        f"[scRNA prototrans] clustering {X_sc.shape[0]} cells into {n_prototypes} prototypes "
        f"for {visible_idx.size}->{hidden_idx.size} translation"
    )
    prototypes, labels = _build_cell_prototypes(
        X_sc,
        n_prototypes=n_prototypes,
        random_state=random_state,
        return_labels=True,
    )

    global_block = relation_projection[np.ix_(visible_idx, hidden_idx)].astype(np.float32)
    proto_visible = prototypes[:, visible_idx].astype(np.float32)
    proto_sizes = np.bincount(labels, minlength=n_prototypes).astype(np.int64)

    proto_residual_mean = np.zeros((n_prototypes, hidden_idx.size), dtype=np.float32)
    proto_residual_coef = np.zeros((n_prototypes, visible_idx.size, hidden_idx.size), dtype=np.float32)
    proto_support = np.zeros(n_prototypes, dtype=np.float32)

    for proto_idx in range(n_prototypes):
        members = X_sc[labels == proto_idx]
        if len(members) == 0:
            continue

        member_visible = members[:, visible_idx].astype(np.float32)
        member_hidden = members[:, hidden_idx].astype(np.float32)
        member_base = member_visible @ global_block
        member_residual = member_hidden - member_base

        visible_center = member_visible.mean(axis=0, keepdims=True).astype(np.float32)
        residual_center = member_residual.mean(axis=0, keepdims=True).astype(np.float32)
        centered_visible = member_visible - visible_center
        centered_residual = member_residual - residual_center

        effective_ridge = float(ridge_lambda * (1.0 + visible_idx.size / max(len(members), 1)))
        coef = _solve_ridge_projection(centered_visible, centered_residual, ridge_lambda=effective_ridge)

        proto_residual_mean[proto_idx] = residual_center.reshape(-1)
        proto_residual_coef[proto_idx] = coef
        proto_support[proto_idx] = float(len(members) / (len(members) + max(int(min_cells), 1)))

    condition_expr = X_st.astype(np.float32) * observed_mask.astype(np.float32)
    global_hidden_ref = (condition_expr[:, visible_idx] @ global_block).astype(np.float32)

    ref = np.zeros_like(X_st, dtype=np.float32)
    ref[:, visible_idx] = condition_expr[:, visible_idx]

    confidence = np.zeros(X_st.shape[0], dtype=np.float32)
    entropy_confidence = np.zeros(X_st.shape[0], dtype=np.float32)
    agreement_confidence = np.zeros(X_st.shape[0], dtype=np.float32)
    support_confidence = np.zeros(X_st.shape[0], dtype=np.float32)

    full_proto_visible = proto_visible.astype(np.float32)

    for spot_idx in range(X_st.shape[0]):
        visible_keep = visible_idx[observed_mask[spot_idx, visible_idx] > 0.5]
        if visible_keep.size < 8:
            visible_keep = visible_idx

        st_vec = X_st[spot_idx, visible_keep]
        proto_sub = prototypes[:, visible_keep]
        st_norm = np.linalg.norm(st_vec)
        proto_norm = np.linalg.norm(proto_sub, axis=1)
        sims = (proto_sub @ st_vec) / np.clip(proto_norm * max(st_norm, 1e-12), 1e-12, None)

        current_top_k = min(max(1, int(top_k)), n_prototypes)
        top_idx = np.argpartition(sims, -current_top_k)[-current_top_k:]
        top_sims = sims[top_idx]
        top_weights = _softmax(top_sims, temperature=temperature)

        if len(top_weights) <= 1:
            entropy_conf = 1.0
        else:
            entropy = -np.sum(top_weights * np.log(np.clip(top_weights, 1e-12, None)))
            entropy_conf = 1.0 - float(entropy / np.log(len(top_weights)))

        spot_visible = condition_expr[spot_idx, visible_idx]
        spot_visible_mask = observed_mask[spot_idx, visible_idx]
        weighted_visible_ref = (full_proto_visible[top_idx] * top_weights[:, None]).sum(axis=0)
        ref_obs = weighted_visible_ref[spot_visible_mask > 0.5]
        st_obs = spot_visible[spot_visible_mask > 0.5]
        ref_norm = np.linalg.norm(ref_obs)
        st_obs_norm = np.linalg.norm(st_obs)
        agreement = float((ref_obs @ st_obs) / np.clip(ref_norm * max(st_obs_norm, 1e-12), 1e-12, None))
        agreement_conf = float(np.clip(0.5 * (agreement + 1.0), 0.0, 1.0))

        support_conf = float(np.dot(top_weights, proto_support[top_idx]))
        spot_conf = float(np.clip(0.50 * agreement_conf + 0.25 * entropy_conf + 0.25 * support_conf, 0.0, 1.0))

        weighted_residual = np.zeros(hidden_idx.size, dtype=np.float32)
        for local_weight, proto_idx in zip(top_weights, top_idx):
            delta_visible = (spot_visible - proto_visible[proto_idx]) * spot_visible_mask
            local_residual = proto_residual_mean[proto_idx] + delta_visible @ proto_residual_coef[proto_idx]
            weighted_residual += float(local_weight) * local_residual.astype(np.float32)

        ref[spot_idx, hidden_idx] = (
            global_hidden_ref[spot_idx] + float(residual_scale) * spot_conf * weighted_residual
        ).astype(np.float32)
        confidence[spot_idx] = spot_conf
        entropy_confidence[spot_idx] = float(np.clip(entropy_conf, 0.0, 1.0))
        agreement_confidence[spot_idx] = agreement_conf
        support_confidence[spot_idx] = float(np.clip(support_conf, 0.0, 1.0))

        if (spot_idx + 1) % 500 == 0:
            print(f"[scRNA prototrans] built {spot_idx + 1}/{X_st.shape[0]} spots")

    if not return_details:
        return ref.astype(np.float32)

    details = {
        "confidence": confidence.astype(np.float32),
        "entropy_confidence": entropy_confidence.astype(np.float32),
        "agreement_confidence": agreement_confidence.astype(np.float32),
        "support_confidence": support_confidence.astype(np.float32),
        "prototype_sizes": proto_sizes.astype(np.int64),
    }
    return ref.astype(np.float32), details
