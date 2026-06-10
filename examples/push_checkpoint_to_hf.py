from synthefy_tabular.hf import push_checkpoint


push_checkpoint(
    "checkpoints/best_reg_r2.pt",
    repo_id="Synthefy/synthefy-tabular",
    private=True,
)
