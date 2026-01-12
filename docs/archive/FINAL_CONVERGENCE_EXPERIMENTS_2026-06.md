# 収束問題・最終実験ラウンド(2026-06)

> **ステータス**: ローカル記録のみ(未コミット)。実験はすべて使い捨てのgit worktreeで実施し、
> 完了後に worktree・ブランチとも破棄済み。メインリポジトリ(paper・src)は一切変更していない。
> 本ドキュメントは会話履歴から再構成した結果サマリ。

## 0. 背景と目的

`model_hill_mixture_hierarchical_reparam`(NumPyro/NUTS、K個のHill飽和曲線の混合)が
実データ K=3 で収束しない問題に対し、「プロジェクトを閉じる前の最後の一絞り」として
**5方針を並行検証 + Aの追加検証**を実施。過去の試行(NCP・ordered k・anchor・retry・
prior調整等)がいずれも幾何いじりの軸で全滅していたため、今回は**識別性そのものを変える/
推論パラダイムを変える**方向を中心に据えた。

- 主指標: **ラベル不変な `rhat_log_lik`**(混合の joint log-likelihood の R̂)。
  リラベル後の成分R̂は label switching 下で無意味になるため、判定はこちらを正本とした。
- 共通設定: 800 warmup / 800 samples、4 chains、target_accept=0.95、max_tree_depth=12、
  K=3、seed=0、init=median。**chainは逐次実行**(`numpyro.set_host_device_count` 未設定)
  だったため全実験が低速だった(A追検のみ並列化)。
- データ: Conjura 実データ org(集計 spend、target=`all_purchases`)。
  org選択の `min_channels` 閾値が実験ごとに微妙に異なり、**検証された org 集合は実験間で一部相違**。

## 1. 各方針の結果と判定

### A. Mixture-of-Experts(共変量gatingで時変重み) — 柔らかい層のみ解決
重みを定数 π_k → π_k(t)=softmax(b_k + W_k·z_t)、z_t=[t_std, sin(2π·idx/365), cos(2π·idx/365)]。

- **小規模 org(訂正後の正しい値)**: 78740ae9(T=452)`rhat_LL` 1.43→**1.00**、
  265fb78d(T=459)1.91→**1.00**。平均 1.67→1.003。divergence 15→2、tree-depth 200→0、
  elpd は MoE が改善(+4.8、+32.6)。pis_t の時間std ≈0.10。合成K=3で `rhat_LL`=1.02。
  - ※ 最初に報告された n=3/800/4ch の表(1.37→1.006、tree-depth悪化、時変ほぼ無し)は
    **別worktreeのモニタ出力が混入した誤データ**でエージェントが撤回。上記が正。
- **追加検証①(最難 org `aec6d5de`, T=1290, フル設定)**: baseline `rhat_LL`=3.157(3200 div)
  → **MoE 2.166(依然 not_converged、2209 div、min_bfmi 0.001、tree-depth 1175、
  elpd -5732→-6016 と悪化、bad Pareto-k 895)**。**硬い層は直せない**。
- **追加検証②(既知の時変gatingで合成パラメータ復元, T=600)**: 収束(`rhat_LL`=1.003、0 div)。
  pis_t 復元 MAE=0.238、成分相関 [0.95, 0.99, 1.00](**向きは当たる**)。だが
  真の pis_t 時間std [0.17, 0.43, 0.36] に対し復元 [0.10, 0.09, 0.04] で**振幅を約1/4に縮小**、
  重みは一様(1/3)へ収縮。Hill A 真値 [90,140,60]→[123,108,94]、k も大きく外す
  (**成分形状はほぼ復元できない**)。
- **判定**: gating は**ラベル対称性(柔らかい非識別)**を共変量で破って収束を回復するが、
  **真の分解不能(硬い非識別)は解決しない**。実データ pis_t std≈0.10 は「真の時変の縮小像」
  であり、レジーム構造の確証として読んではいけない。

### B. Sparse finite mixture(Dirichlet(0.5)・ordering/anchor撤去) — 部分的
Rousseau & Mengersen(2011)の e0 < d/2(d≈3 → e0=0.5)で余分成分が空になる、を検証。
過去の失敗(α=0.1)は過激すぎ+scaffoldingとの干渉が原因という仮説。

- 4a762f02(T=1030): `rhat_LL` 1.53→**1.00**(勝ち)
- 246b6e2f(T=766): 1.00→1.00(引き分け)
- aec6d5de(T=1290): 3.08→3.20(**改善せず・幾何悪化**、両者 3200 div、BFMI 0.011)
- **どの成分も綺麗に空にならず**(effective K ~2.3)。ただし label switching 下では
  対称Dirichletの平均重みは対称性で~均等になるため、平均重みからの effective K は
  「空化なし」の証拠にならない(測定上の留保)。
- **判定**: 仮説は部分的支持。撤去だけでは空化を起こせず、最難 org はむしろ悪化。
  prior/scaffolding の小細工では直らない根本問題の存在を示唆。

### C. Stacking for non-mixing computations(Yao, Vehtari & Gelman 2022) — 予測のみ救済
多峰を「直す」のでなく独立chainを LOO 効用で重み付き結合。

- pre-screen で最難 org を自動選択: `aec6d5de`(`rhat_LL`=2.61)。他2 org は 1.02, 1.01。
- M=6 独立 single-chain: 各 effK≈1.2(**各chainが実質1成分の別解**)、elpd -5723〜-5735、
  **pooled rhat=2.236**(別モード確定)。
- stacking 重み [0.272, 0, 0, 0.319, 0, 0.409]。結合 LOO elpd: stacked -5719.2 vs
  単一最良 -5723.1(**+3.9**)vs 等重み -5721.9(+2.7)。
- 単峰 org(rhat 1.001)での検証: stacking は1 chainに退化、等重みが僅かに勝つ
  → **多峰時のみ効く**ことを両側確認。
- **判定**: 多峰時に**予測**を安定的に救済(構造の復元はしない)。
  「各モード≒1成分解で予測的に等価」=**閾値以下では単一Hillが予測を失わない**ことの実証でもある。

### D. Mode-finding init(SVI+AutoDelta 多点MAP; Pathfinder は numpyro 0.19 に無し) — 境界例のみ
- 2b15eedf(T=1576): 1.002→1.003(既収束、不変)
- 246b6e2f(T=766): 1.002→1.000
- 72a86a20: baseline 1.072(min_ess 7)→ map_full 1.004(ess 1134)、
  map_reduced 1.002(ess 1111)。**境界 org を救済**。
- 平均 `rhat_LL` 1.025→1.001、min_ess 813→1256。
- **コスト3〜7倍**(8点MAP探索が支配的)。reduced-warmup は MAP 探索が律速のためほぼ無効。
  org選択が異なり**最難 `aec6d5de` は未検証**。
- **判定**: 境界/marginal org への戦術的補助(1-2スタートなら安価)。多峰には無効。

### E. Student-t 尤度(nu~Gamma(2, 0.1)) — 棄却
- 4a762f02: 1.53→1.30(not_conv、25 div、tree-depth 600、bad_k 33)
- 246b6e2f: 1.001→1.001(両者収束)
- aec6d5de: **2.96→3.12(悪化)**、relabeled 3.52→4.07
- 平均 `rhat_LL` 1.828→1.806、**effective_K 1.95→2.43(増=仮説と逆)**、
  **Pareto-k>0.7 件数 1.3→12.3(悪化)**、divergence 800→8(改善)だが tree-depth 0→1000(悪化)。
- **判定**: 棄却。divergence を tree-depth 飽和に付け替えるだけで、多峰=非識別は手つかず。
  非収束は外れ値ではなく真の混合多峰性が原因。

## 2. 横断的な結論

1. **同一 org `aec6d5de`(T=1290)が baseline・sparse・Student-t・MoE の全てで収束失敗**。
   失敗は推論機構ではなく**データの性質**。唯一 stacking が予測だけを救う。
2. **問題は2層**:
   - 第1層(ラベル対称性=柔らかい非識別): MoE の共変量gatingが直す。
   - 第2層(真の分解不能=硬い非識別): **どの手法でも直らない**。情報が無い。
3. **resolvability は単一の(非収束)事後から綺麗には出ない**(循環)。最も信頼できるのは
   合成データでの復元テスト/SBC。実データでは短fitの `rhat_LL` が代理になるが、
   非識別以外の原因でも上がりうるノイズの多い代理。

## 3. カスタマー・クラスタ仮説への含意

元の研究動機は「1データセット内に複数のカスタマー・クラスタがある」という仮説だった。
しかし:
- このモデルの**観測単位は「日」**(集計日次系列)であり、混合が分類するのは**日であって顧客ではない**。
- **1本の集計系列からは複数クラスタの応答曲線は原理的に識別不能**(異なるクラスタ構成が
  同一の集計を生む)。つまり「混合が収束しない」のではなく「**集計データはクラスタ問題に
  答えられない**」。これは resolvability 論そのもの。
- データに実在する顧客信号は `first_purchases`(新規)vs リピートの2区分のみ。

## 4. モデリング上の最終所見

- **混合は集計・データ希少なMMMには割に合わない**(単一Hillの飽和パラメータすら識別困難な中、
  応答パラメータをK倍+gateにするのは方向が逆 — 今回の収束失敗を製造している)。
- **季節性**は混合gatingではなく **baseline の Fourier 項**で扱うのが正攻法
  (現モデルの baseline は線形トレンドのみ=季節項なし)。
- **媒体効果の時変**を捉えたいなら **時変係数(TVC)/状態空間モデル**(滑らかさpriorで
  gatingより遥かに良く識別される)。
- **混合が正当化されるのは、識別の取っ手=非集計のセグメントデータがある時だけ**。
- **resolvability** は「データが支えない構造を当てるな」という診断・教訓として有効。
  使い分け案(閾値以下=単一Hill、以上=混合)は筋が通るが、閾値判定の循環に注意。

## 5. 方法論上の留意点(再利用時)

- ラベル不変 `rhat_log_lik` を正本に。リラベル後R̂・平均重みからの effective K は label
  switching 下でバイアス。
- `min_ess_bulk` は高次元 deterministics(`pis_t`, `hill_components`)の監視で過小に出る。
- chain は `numpyro.set_host_device_count(4)` で並列化すると約4倍速。今回の遅さの主因。
- org選択(`select_representative_timeseries` の `min_channels`)を固定しないと実験間で
  比較対象 org がずれる。

## 6. 後始末

- 全 worktree(`agent-*`)・ブランチ(`worktree-agent-*`)を破棄。
- 実験用に追加した権限(`Edit`/`Write`/`Bash(uv run python:*)` 等)を撤去。
- メインリポジトリは HEAD `1eca88e`、追跡ファイルの変更ゼロ(現状維持)。
- **方針: paper は現状維持。本ラウンドの成果物はすべて破棄。** 本ドキュメントのみローカルに残置。
