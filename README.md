# TP2 — Análise Empírica, Auditoria Algorítmica e Otimização para o Caixeiro Viajante Geométrico (`EUC_2D`)

**Disciplina:** DCC207 — Algoritmos 2 (Universidade Federal de Minas Gerais — UFMG)  
**Linguagem e Stack:** Python 3, NumPy, SciPy, NetworkX, Scikit-learn, Pandas, Matplotlib, LaTeX (padrão SBC)

Este repositório contém a implementação completa, instrumentada e auditada do **Trabalho Prático 2** sobre o Problema do Caixeiro Viajante Euclidiano Bidimensional (`EUC_2D` da biblioteca TSPLIB), integrando:

1. **Algoritmo Exato *Branch-and-Bound* (DFS In-Place)** com representação de estado via *bitmask*, pivotamento na raiz por vértice *outlier*, ordenação de ramos por custo reduzido e funil de poda em 3 níveis com potenciais lagrangianos de **Held-Karp**.
2. **Cinco Estratégias de Inicialização (*Warm-Start*) e Algoritmos Aproximados:** *Cold Start*, Vizinho Mais Próximo (*Nearest Neighbor*), Inserção por Fecho Convexo (*Convex Hull Insertion*), *Twice-around-the-tree* (2-aproximação) e *Christofides-Serdyukov* (3/2-aproximação).
3. **Instrumentação e *Profiling* em Nível de Subrotina** (tempo de parede via `time.perf_counter()` e pico de RAM via `tracemalloc`).
4. **Extração de Características Geométricas/Topológicas** e **Modelagem Estatística / Análise de Sensibilidade** (regressões assintóticas não-lineares e log-lineares, *Speedup* líquido, `RandomForestRegressor`, `DecisionTreeRegressor` e correlação de Spearman).
5. **Relatório Científico Completo em LaTeX (padrão SBC)** compilado em PDF com anexo de histórico auditado de prompts.

---

## 1. Estrutura do Repositório

```text
trabalho_22/
├── main.py                         # Orquestrador CLI com checkpointing incremental e timeout (1800s no modo full)
├── requirements.txt                # Dependências declaradas (numpy, scipy, networkx, scikit-learn, pandas, matplotlib)
├── enunciado_trabalho.pdf          # Enunciado oficial do TP2 (DCC207)
├── TECHNICAL-SPEC.md               # Especificação técnica de contratos, invariantes e complexidade
├── PLAN.md                         # Plano atômico de execução e testes
├── README.md                       # Documentação técnica detalhada do funcionamento e resultados
├── data/
│   └── tsplib/                     # 16 instâncias TSPLIB padronizadas na métrica EUC_2D (N entre 10 e 70)
├── src/
│   ├── __init__.py
│   ├── io_tsplib.py                # Parser TSPLIB, matriz EUC_2D vetorizada, MDS espectral e validação de tours
│   ├── profiler.py                 # Context manager `profile_block` (tempo e pico de memória por subrotina)
│   ├── features.py                 # Extração de métricas geométricas e topológicas (Hull, Entropia, Hopkins, MST, NN CV)
│   ├── heuristics.py               # Estratégias construtivas: Cold Start, Nearest Neighbor e Convex Hull Insertion
│   ├── approximations.py           # Algoritmos aproximados: Twice-around-the-tree e Christofides
│   ├── branch_and_bound.py         # Motor exato DFS In-Place + Held-Karp + funil de poda em 3 níveis + benchmark MST
│   └── analysis.py                 # Regressões assintóticas, gráficos (.png/.pdf), sensibilidade e tabelas LaTeX
├── results/
│   ├── metrics_log.csv             # Base completa do modo full (80 execuções: 16 instâncias x 5 estratégias, timeout 1800s)
│   ├── features_log.csv            # Features geométricas e topológicas das 16 instâncias
│   ├── mst_benchmark.csv           # Benchmark comparativo Prim denso em NumPy vs SciPy csgraph (submatriz 20x20)
│   ├── regression_models.csv       # Parâmetros ajustados, R² e RMSE das curvas de complexidade
│   ├── sensitivity_importance.csv  # Importância das features (Random Forest) e R² por alvo
│   ├── decision_tree_rules.txt     # Regras extraídas da DecisionTreeRegressor (profundidade 3) para Gap Christofides
│   ├── plots/                      # Gráficos em alta resolução (.png 300 dpi e .pdf)
│   └── tables/                     # Tabelas prontas para \input{} no LaTeX (.tex)
├── report/
│   ├── relatorio_sbc.tex           # Código-fonte do artigo científico em padrão SBC (com Anexo de Prompts)
│   ├── relatorio_sbc.pdf           # Relatório final compilado
│   ├── references.bib              # Referências bibliográficas BibTeX
│   ├── sbc-template.sty            # Estilo oficial SBC
│   └── sbc.bst                     # Estilo bibliográfico SBC
├── tests/
│   └── test_tsp_pipeline.py        # Suíte de 9 testes unitários e de integração end-to-end (isolados em TempDir)
└── tp2_tsp/                        # Diretório compatível com symlinks para data/, src/, results/, main.py e requirements.txt
```

---

## 2. Arquitetura e Detalhes Algorítmicos

### 2.1. Ingestão TSPLIB, Padronização `EUC_2D` e Matriz de Distâncias (`src/io_tsplib.py`)

O conjunto experimental é composto por **16 instâncias da TSPLIB** variando de $N = 10$ a $N = 70$:
- **Faixa Principal ($N \le 42$, 12 instâncias — 75% do conjunto):** `eil10`, `berlin12`, `burma14`, `ulysses16`, `st18`, `eil20`, `ulysses22`, `gr24`, `fri26`, `bayg29`, `bays29`, `swiss42`.
- **Faixa Limite / Estresse ($N \in [48, 70]$, 4 instâncias — 25% do conjunto):** `att48`, `eil51`, `berlin52`, `st70`.

Como várias instâncias clássicas da TSPLIB nessa faixa originalmente usam coordenadas geográficas em graus (`burma14`, `ulysses16`, `ulysses22`) ou matrizes explícitas de distância (`gr24`, `fri26`, `bays29`, `swiss42`), `ensure_tsplib_dataset` padroniza todo o conjunto para coordenadas bidimensionais `EUC_2D`:
1. Instâncias `GEO` recebem reescalonamento homotético ($\times 100$) para evitar que cidades próximas colapsem para distância inteira $0$ sob arredondamento.
2. Instâncias explícitas são projetadas em $\mathbb{R}^2$ via **Imersão Espectral Clássica de Torgerson (MDS)** em `_classical_mds_2d` usando `scipy.linalg.eigh` sobre a matriz de Gram duplamente centrada $B = -\frac{1}{2} H D^2 H$.
3. Subinstâncias menores (`eil10`, `berlin12`, `st18`, `eil20`) são extraídas de prefixos determinísticos das instâncias `EUC_2D` originais (`eil51`, `berlin52`, `st70`).

A matriz de distâncias `compute_euc2d_matrix` usa `scipy.spatial.distance.cdist(..., metric="euclidean")` e aplica a regra oficial da TSPLIB:
$$d_{ij} = \lfloor \lVert p_i - p_j \rVert_2 + 0.5 \rfloor \in \mathbb{Z}_{\ge 0}, \quad d_{ii} = 0.$$
O cálculo do custo fechado de qualquer ciclo hamiltoniano (`compute_tour_cost`) é vetorizado via `np.roll(idx, -1)` após validação de permutação em `validate_tour`.

---

### 2.2. Instrumentação e *Profiling* por Subrotina (`src/profiler.py`)

O *context manager* `profile_block(telemetry, prefix)` encapsula cada subrotina crítica (`mst`, `matching`, `eulerian`, `shortcutting`, `dfs_preorder`, `convex_hull`, `hull_insertion`, `heuristic_total`):
- Mede o tempo de parede com `time.perf_counter()`.
- Mede o pico de memória alocada com `tracemalloc`, utilizando `tracemalloc.reset_peak()` em blocos aninhados para isolar o pico da subrotina sem interromper o contexto externo.
- Registra `<prefix>_time` (segundos), `<prefix>_peak_kb` (KiB) e atualiza `peak_memory_kb`.

---

### 2.3. Extração de Features Geométricas e Topológicas (`src/features.py`)

`extract_instance_features(coords, dist_matrix)` extrai:
- `aspect_ratio`: razão $\frac{\min(\Delta x, \Delta y)}{\max(\Delta x, \Delta y)} \in (0, 1]$ do *bounding box* via `np.ptp(arr, axis=0)`.
- `convex_hull_ratio`: fração de vértices pertencentes ao fecho convexo (`len(ConvexHull(arr).vertices) / N`).
- `spatial_entropy`: entropia de Shannon normalizada (`scipy.stats.entropy`) sobre histograma 2D (`np.histogram2d`) das coordenadas em $[0, 1]^2$.
- `hopkins_statistic`: estatística determinística de Hopkins (semente `42`) via `scipy.spatial.KDTree`.
- `cluster_density`: fração de vértices agrupados pelo `sklearn.cluster.DBSCAN`.
- `mst_max_degree` e `mst_std_degree`: grau máximo (*hubs*) e desvio-padrão dos graus na MST via `scipy.sparse.csgraph.minimum_spanning_tree`.
- `nn_cv`: coeficiente de variação ($\sigma / \mu$) das distâncias ao vizinho mais próximo via `KDTree(..., k=2)`.

---

### 2.4. Heurísticas de *Warm-Start* e Algoritmos Aproximados (`src/heuristics.py`, `src/approximations.py`)

Todas as cinco estratégias seguem o contrato `(cost, tour, telemetry)` retornando uma permutação válida iniciada em `0`:

1. **Cold Start (`cold_start_tsp`):** rota trivial `[0, 1, ..., N-1]`. Complexidade $\mathcal{O}(N)$.
2. **Vizinho Mais Próximo (`nearest_neighbor_tsp`):** construção gulosa a partir do vértice `0` mascarando visitados e escolhendo `np.argmin`. Complexidade $\mathcal{O}(N^2)$.
3. **Inserção por Fecho Convexo (`convex_hull_insertion_tsp`):** inicializa a sub-rota com `scipy.spatial.ConvexHull(coords)` e avalia de forma vetorizada (`np.ix_`) o custo marginal $\Delta(u, v, k) = d_{uk} + d_{kv} - d_{uv}$ para inserir iterativamente os vértices internos.
4. **Twice-around-the-tree (`twice_around_the_tree_tsp`):** monta o grafo completo em `NetworkX` (restaurando eventuais arestas de peso `0` entre pontos coincidentes), extrai a MST via Prim (`"mst"`) e percorre em pré-ordem (`"dfs_preorder"`).
5. **Christofides-Serdyukov (`christofides_tsp`):** extrai a MST (`"mst"`), calcula o emparelhamento perfeito de peso mínimo sobre vértices ímpares com `min_weight_matching` — algoritmo Blossom de Edmonds (`"matching"`), constrói o multigrafo euleriano (`"eulerian"`) e realiza *shortcutting* linear (`"shortcutting"`).

---

### 2.5. Motor Exato *Branch-and-Bound* (`src/branch_and_bound.py`)

Para evitar esgotamento de RAM típico de *Best-First Search*, `BranchAndBoundTSP` implementa **DFS *In-Place*** com espaço de pilha $\mathcal{O}(N)$:

1. **Estado em *Bitmask*:** conjunto de visitados em um inteiro `visited_mask` manipulado em $\mathcal{O}(1)$.
2. **Pivotamento por *Outlier* e Quebra de Simetria:** enraíza a busca em `root_vertex = argmax_v (d_(1)(v) + d_(2)(v))` e impõe `path[1] < nxt` na folha para eliminar rotas espelhadas.
3. **Potenciais Lagrangianos de Held-Karp (`_compute_hk_potentials`):** 180 iterações de subgradiente em ponto fixo (`HK_SCALE = 256`), ordenando vizinhos pelo custo reduzido $\tilde{d}_{uv} - \lfloor \min_w \tilde{d}_{wv} / 2 \rfloor$.
4. **Funil de Poda em 3 Níveis Admissível:**
   - **Nível 1 ($\mathcal{O}(1)$):** limites de grau 2 original e penalizado em ponto fixo.
   - **Nível 2 ($\mathcal{O}(k)$):** filtro geométrico 2-Opt exigindo interseção própria **e** ganho inteiro estrito na métrica `EUC_2D`.
   - **Nível 3:** relaxação de 1-árvore de Held-Karp sobre não-visitados ($|U| \ge 3$) via Prim denso em listas Python (`_reduce_prim_edges`) com teto inteiro exato.
5. **Timeout Amortizado:** checa `time.perf_counter()` a cada `1024` nós (`(nodes_expanded & 1023) == 0`), gravando `"NA"` em `optimal_cost` quando abortado e preservando toda a telemetria.

---

## 3. Principais Resultados Obtidos (Execução Completa — Timeout de 30 min)

Todos os números abaixo correspondem à base consolidada em `results/` (modo `full`, `1800 s` por execução):

1. **Benchmark MST (`results/mst_benchmark.csv`):** em 400 repetições sobre submatrizes $20 \times 20$, o Prim denso em NumPy levou `0,0606 s` com pico de `2,61 KiB`, sendo **$2{,}73\times$ mais rápido** e consumindo **$27{,}5\times$ menos memória** que `scipy.sparse.csgraph.minimum_spanning_tree` (`0,1656 s` e `71,71 KiB`).
2. **Regressões Assintóticas (`results/regression_models.csv`):**
   - `MST Prim (ms)`: $\mathcal{O}(N^2)$ com $R^2 = 0{,}9397$ ($\text{RMSE} = 1{,}7781\,\text{ms}$).
   - `Twice-Around Total (ms)`: $\mathcal{O}(N^2)$ com $R^2 = 0{,}9317$ ($\text{RMSE} = 1{,}9498\,\text{ms}$).
   - `Matching Blossom (ms)`: $\mathcal{O}(N^3)$ com $R^2 = 0{,}6246$; `Christofides Total (ms)`: $R^2 = 0{,}8000$.
   - `Pico Memoria (KiB)`: $\mathcal{O}(N^2)$ com $R^2 = 0{,}9836$.
   - `B&B` ($N \le 29$, melhor estratégia resolvida na faixa $N \le 40$): crescimento exponencial $\approx 0{,}0596 \cdot (1{,}6285)^N$ nós ($R^2 = 0{,}6898$) e $\approx 2{,}52 \times 10^{-6} \cdot (1{,}6536)^N\,\text{s}$ ($R^2 = 0{,}6934$).
3. **Impacto do *Warm-Start* (`results/metrics_log.csv`):**
   - **13 das 16 instâncias** tiveram o ótimo certificado dentro de 1800 s (todas as 11 até $N = 29$, mais `swiss42 = 1273` e `att48 = 33522`); 29 das 80 execuções atingiram o limite de tempo (`eil51`, `berlin52` e `st70` não foram certificadas por nenhuma estratégia).
   - **Inserção por Fecho Convexo** atingiu gap médio de **`2,35%`** (tempo máximo `15,99 ms` em `st70`), certificou 13 instâncias e gerou *speedups* líquidos de até **`12.978,52x`** (`eil20`), **`3.801,06x`** (`berlin12`), **`1.258,49x`** (`ulysses22`) e **`894,38x`** (`bayg29`).
   - **Christofides** atingiu gap médio de **`9,13%`** (tempo máximo `59,80 ms` em `st70`), certificou 11 instâncias (incluindo `swiss42`) e alcançou *speedup* máximo de **`400,41x`** (`berlin12`).
   - **Gaps médios das demais estratégias:** Vizinho Mais Próximo `22,03%` e *Twice-around-the-tree* `25,63%`.
4. **Sensibilidade Topológica (`results/sensitivity_importance.csv` e `results/decision_tree_rules.txt`):**
   - `convex_hull_ratio` domina a explicação do gap da Inserção por Fecho Convexo (`65,8%`, $R^2 = 0{,}897$, $\rho = -0{,}704$), da Taxa de Poda do B&B (`61,5%`, $R^2 = 0{,}856$, $\rho = -0{,}829$), do Vizinho Mais Próximo (`37,5%`, $R^2 = 0{,}745$) e de Christofides (`30,7%`, $R^2 = 0{,}879$, $\rho = -0{,}628$).
   - `aspect_ratio` (`44,9%`, $\rho = +0{,}622$) e `nn_cv` (`22,2%`, $\rho = -0{,}612$) dominam o erro de `Twice-around-the-tree` ($R^2 = 0{,}881$).
   - A árvore de decisão aponta gaps elevados de Christofides (`11,01%` a `16,21%`) quando `spatial_entropy > 0,79`, `convex_hull_ratio <= 0,37` e `hopkins_statistic > 0,53` (ex.: `att48` com `17,87%` e `berlin52` com `14,55%`).

---

## 4. Como Executar e Reproduzir

### Pré-requisitos
```bash
source .venv/bin/activate
```

### Executar Testes Unitários e de Integração
Os testes são isolados em diretórios temporários e não alteram os resultados finais em `results/`:
```bash
.venv/bin/python -m unittest discover -s tests -p "test_*.py" -v
```

### Executar o Pipeline Experimental
- **Modo de validação rápida** (regera logs, gráficos e tabelas em ~1,5 minuto):
  ```bash
  .venv/bin/python main.py --mode fast-test --reset
  ```
- **Modo oficial completo** (timeout de `1800s` = 30 minutos por chamada ao B&B, com retomada automática via checkpoint):
  ```bash
  .venv/bin/python main.py --mode full
  ```

### Recompilar o Artigo Científico SBC em LaTeX
```bash
cd report
pdflatex -interaction=nonstopmode relatorio_sbc.tex
bibtex relatorio_sbc
pdflatex -interaction=nonstopmode relatorio_sbc.tex
pdflatex -interaction=nonstopmode relatorio_sbc.tex
```
