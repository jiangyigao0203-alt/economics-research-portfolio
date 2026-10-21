# Source provenance and renamed files

Python source was adapted from the original local projects. C++ source was selected from the supplied `CPPwork_place.zip`. Public files now have descriptive names, reader-facing comments, and Python docstrings; they are no longer byte-identical copies. The C++ combined program also uses relative input/output defaults. Historical result tables remain unchanged.

| Original source | Published file | Published SHA-256 |
| --- | --- | --- |
| `525data/CAMP.py` | [estimate_event_abnormal_returns.py](525/estimate_event_abnormal_returns.py) | `39c622d824779ab26618b7e1ae15dd6cd29c20162601b2328ea88574b9b1ee88` |
| `525data/Decomposition.py` | [decompose_monetary_policy_shocks.py](525/decompose_monetary_policy_shocks.py) | `fc947e82a2f977a4da579673bb3c82db3812b64210c6263996d86ba1d4f3136e` |
| `525data/diag.py` | [diagnose_equity_duration.py](525/diagnose_equity_duration.py) | `cfba07da748d2a76f76cca41d7e3b177b198710c9162fd60f2d81e1eb9e8b8d1` |
| `525data/get_daily_price.py` | [download_crsp_daily_prices.py](525/download_crsp_daily_prices.py) | `8b887f5e97d29575713d25aef602a75973d7ffd4b78b73c10171e9d1921ac061` |
| `525data/getdatawrdsmonthly.py` | [build_compustat_monthly_panel.py](525/build_compustat_monthly_panel.py) | `dddc64fcc0b7c771430c0ed2bdb59eb1f6b49e9fb08802daab031f6c4eac86ee` |
| `525data/impliedduration.py` | [estimate_equity_duration.py](525/estimate_equity_duration.py) | `043bd59ed20abe97b637f74a215e0f91cf642dce9f1727a5f6cce843e66af13d` |
| `525data/increase_test.py` | [test_duration_coefficient_trend.py](525/test_duration_coefficient_trend.py) | `92514b39512c699ae095272fa2d3b98b8bfb8f3de1e336985229a6babe7bc8aa` |
| `525data/market_data.py` | [download_market_and_risk_free_returns.py](525/download_market_and_risk_free_returns.py) | `2210d6dc8abcbae491279a1a6e2986a433a70da4ce225c04db6cce37139fed38` |
| `525data/panel.py` | [estimate_duration_shock_panel.py](525/estimate_duration_shock_panel.py) | `eef7c4b1509c55790fd43c9cef75b82e71d8b850b228fa6a523b40932f6cc3f1` |
| `525data/risk_factor.py` | [estimate_asset_market_betas.py](525/estimate_asset_market_betas.py) | `26d8fad161e4f3fe387b2deb1022999d268a7f32faa7c5d705f83a402e2c00da` |
| `575/BSADF.py` | [compute_bsadf.py](575/compute_bsadf.py) | `e1caf6b931d2700595a0028a7322b69e245021cde0de6b10a717e5e6eef83568` |
| `575/start.py` | [detect_bubbles_interval_union.py](575/detect_bubbles_interval_union.py) | `6399233095c33a12254c7f20dc8b1e188d06e9e212c4a99f51f53990263c80a3` |
| `575/arch2.py` | [plot_bubbles_and_garch_volatility.py](575/plot_bubbles_and_garch_volatility.py) | `a24bf2bec433b570b31bc3f17ce7895a7024720b53bee4a3e40c1ddd82fe4aa9` |
| `575/model.py` | [regress_bubble_episode_characteristics.py](575/regress_bubble_episode_characteristics.py) | `997af17e97080ac4ea12b524dc00679ef0d3b4124c9fff72859de2360db91732` |
| `575/MLE1.py` | [build_bubble_volatility_panel.py](575/build_bubble_volatility_panel.py) | `e6b02de918ba73b86f9e3a6dadca3114381a41ed51aa186072b44cbf8d675ae0` |
| `575/MLE2.py` | [evaluate_classifier_series_holdout.py](575/evaluate_classifier_series_holdout.py) | `046f29c72fbff7c9840d19cdc370ddb4f177946271ef6b37368d337d1d741c0c` |
| `575/MLE3.py` | [evaluate_classifier_noise_control.py](575/evaluate_classifier_noise_control.py) | `bd388e78f73f7b5648694a113c2f515a125a8ce8bae3c75e926d741035df8a25` |
| `575/get_etfs.py` | [download_etf_prices.py](575/download_etf_prices.py) | `ebefbdb3ae27074543cd2baff6dcc0f0899550d73e42086393b6e94651b49ce9` |
| `CPPwork_place.zip:CPPwork_place/ultimateADF.cpp` | [multi_market_bsadf_gsadf.cpp](575/cpp/multi_market_bsadf_gsadf.cpp) | `adbac234771c53bb4d190c2cfec8d59b53b424bca2b375fd5798f72740435574` |
| `CPPwork_place.zip:CPPwork_place/monte_carlo.cpp` | [bsadf_monte_carlo_lag0.cpp](575/cpp/bsadf_monte_carlo_lag0.cpp) | `d590cd7e193ad9160cb164706b20d6a7ab5281521f10aefa8950e9bca7f4fea3` |

All 18 Python syntax trees match the prior scripts after removing docstrings. C++ token comparisons allow only the documented path-default changes. These checks verify preservation of executable text, not numerical correctness or reproduction of historical results.
