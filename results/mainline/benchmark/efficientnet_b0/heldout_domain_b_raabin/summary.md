# Mainline Benchmark Summary

| Run | Mode | Backbone | Test Acc | Test Macro-F1 | Leakage Excluded | Eval TTA |
|---|---|---|---|---|---|---|
| real_only__efficientnet_b0__heldout_raabin__tf0p001__seed901__cf_eosi0p25_mono0p25 | real_only | efficientnet_b0 | 0.0642 | 0.0592 | 0 | none |
| real_only__efficientnet_b0__heldout_raabin__tf0p001__seed902__tta_hflip | real_only | efficientnet_b0 | 0.0919 | 0.0854 | 0 | hflip |
| real_only__efficientnet_b0__heldout_raabin__tf0p001__seed903__tta_hflip | real_only | efficientnet_b0 | 0.2372 | 0.1797 | 0 | hflip |
| real_only__efficientnet_b0__heldout_raabin__tf0p002__seed888__cf_eosi0p25_mono0p25 | real_only | efficientnet_b0 | 0.2241 | 0.1419 | 0 | none |
| real_only__efficientnet_b0__heldout_raabin__tf0p01__seed777 | real_only | efficientnet_b0 | 0.3648 | 0.2261 | 0 | none |
| real_only__efficientnet_b0__heldout_raabin__tf0p05__seed42 | real_only | efficientnet_b0 | 0.3335 | 0.2297 | 0 | none |
| real_only__efficientnet_b0__heldout_raabin__tf1p0__seed42 | real_only | efficientnet_b0 | 0.471 | 0.2863 | 0 | none |
| real_plus_synth__efficientnet_b0__heldout_raabin__tf0p02__seed42__synth_leakage_smoke_manifest | real_plus_synth | efficientnet_b0 | 0.146 | 0.1573 | 1 | none |
| real_plus_synth__efficientnet_b0__heldout_raabin__tf0p02__seed42__synth_synthetic_manifest | real_plus_synth | efficientnet_b0 | 0.1927 | 0.1748 | 0 | none |
