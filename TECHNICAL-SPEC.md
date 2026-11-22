# Especificação Técnica (`TECHNICAL-SPEC.md`) — TP2 Algoritmos 2 (TSP Geométrico)

## 1. Mapeamento Estrutural e Reutilização de Abstrações

O repositório implementa o TP2 de Algoritmos 2 conforme `enunciado_trabalho.pdf`. Seguindo a diretriz de zero reinvenção de algoritmos clássicos já otimizados em C/Fortran, o sistema delega operações geométricas, matriciais, espectrais, estatísticas e de grafos às seguintes bibliotecas:

- **`scipy.spatial.distance.cdist`**: cálculo vetorizado de distâncias euclidianas par-a-par em $\mathcal{O}(N^2)$.
- **`scipy.spatial.ConvexHull`**: construção do fecho convexo planar em $\mathcal{O}(N \log N)$.
- **`scipy.spatial.KDTree`**: consulta espacial de vizinhos mais próximos em $\mathcal{O}(N \log N)$.
- **`scipy.sparse.csgraph.minimum_spanning_tree`**: extração de Árvore Geradora Mínima sobre matrizes densas/esparsas.
- **`scipy.stats.entropy`**: entropia de Shannon sobre distribuição espacial 2D discretizada.
- **`scipy.optimize.curve_fit`**: ajuste não-linear de curvas assintóticas teóricas.
- **`networkx.minimum_spanning_tree` / `networkx.dfs_preorder_nodes`**: extração de MST (algoritmo de Prim) e percurso em pré-ordem para o algoritmo *Twice-around-the-tree*.
- **`networkx.algorithms.matching.min_weight_matching`**: algoritmo Blossom de Edmonds para emparelhamento perfeito de peso mínimo sobre o subgrafo induzido pelos vértices de grau ímpar no algoritmo de Christofides.
- **`networkx.MultiGraph` / `networkx.eulerian_circuit`**: construção do multigrafo euleriano e extração de circuito euleriano antes do *shortcutting*.
- **`sklearn.cluster.DBSCAN`**: identificação de densidade de agrupamento espacial.
- **`sklearn.linear_model.LinearRegression` / `sklearn.tree.DecisionTreeRegressor` / `sklearn.ensemble.RandomForestRegressor` / `sklearn.metrics`**: modelagem empírica de complexidade e análise de sensibilidade de features topológicas frente ao gap de aproximação.

Funções utilitárias internas (como cálculo vetorizado do custo de um ciclo hamiltoniano `compute_tour_cost` e validação de permutação `validate_tour`) são centralizadas em `src/io_tsplib.py` e reutilizadas por `heuristics.py`, `approximations.py`, `branch_and_bound.py` e `tests/`, evitando qualquer duplicação de código.

---

## 2. Contratos de Dados, Assinaturas, Pré-condições e Pós-condições

### 2.1. `src/io_tsplib.py`
- **`parse_tsplib(filepath: str | Path) -> np.ndarray`**
  - **Pré-condição**: `filepath` aponta para um arquivo `.tsp` existente contendo seção `NODE_COORD_SECTION` ou `DISPLAY_DATA_SECTION` com $N \ge 3$ vértices. Se o arquivo não existir, propaga `FileNotFoundError`; se não houver coordenadas válidas, propaga `ValueError`.
  - **Pós-condição**: Retorna `coords` do tipo `np.ndarray` (`dtype=np.float64`, shape `(N, 2)`).
- **`compute_euc2d_matrix(coords: np.ndarray) -> np.ndarray`**
  - **Pré-condição**: `coords` possui shape `(N, 2)` e `N >= 2`. Caso contrário, lança `ValueError`.
  - **Pós-condição**: Retorna `dist_matrix` simétrica (`dtype=np.int64`, shape `(N, N)`) com diagonal principal nula e entradas $d_{ij} = \lfloor \lVert p_i - p_j \rVert_2 + 0.5 \rfloor$.
- **`compute_tour_cost(dist_matrix: np.ndarray, tour: Sequence[int]) -> int`**
  - **Pré-condição**: `tour` é uma sequência não vazia de índices válidos em `[0, N-1]`.
  - **Pós-condição**: Retorna a soma inteira $\sum_{k=0}^{N-1} d(\text{tour}[k], \text{tour}[(k+1) \bmod N])$ calculada de forma vetorizada via indexação NumPy.
- **`ensure_tsplib_dataset(data_dir: str | Path) -> list[Path]`**
  - **Pré-condição**: `data_dir` é um caminho de diretório gravável.
  - **Pós-condição**: Garante a presença das 16 instâncias de teste (`N` entre 10 e 70, 80% na faixa principal $N \le 40$ e 20% na faixa limite $N \in [45, 75]$) no formato TSPLIB `EUC_2D` e retorna a lista de caminhos ordenada crescentemente por número de vértices $N$.

### 2.2. `src/profiler.py`
- **`SubroutineStats` (dataclass)**: armazena `elapsed_sec: float` e `peak_memory_kb: float`.
- **`profile_block(telemetry: dict[str, float], prefix: str) -> Iterator[None]` (Context Manager)**
  - **Pré-condição**: `telemetry` é um dicionário mutável; `prefix` identifica a subrotina (ex: `"mst"`, `"matching"`, `"eulerian"`, `"convex_hull"`, `"bb_total"`).
  - **Pós-condição**: Mede o tempo de parede com `time.perf_counter()` e o pico de memória alocada no escopo via `tracemalloc`, registrando `f"{prefix}_time"` (em segundos) e `f"{prefix}_peak_kb"` (em KiB) em `telemetry`, além de atualizar `telemetry["peak_memory_kb"]`.

### 2.3. `src/features.py`
- **`extract_instance_features(coords: np.ndarray, dist_matrix: np.ndarray) -> dict[str, float]`**
  - **Pré-condição**: `coords` shape `(N, 2)`, `dist_matrix` shape `(N, N)` com $N \ge 4$.
  - **Pós-condição**: Retorna dicionário contendo:
    1. `aspect_ratio`: $\frac{\min(\Delta x, \Delta y)}{\max(\Delta x, \Delta y)} \in (0, 1]$.
    2. `convex_hull_ratio`: proporção de pontos no fecho convexo (`len(hull.vertices) / N`).
    3. `spatial_entropy`: entropia de Shannon sobre histograma 2D normalizado das coordenadas.
    4. `hopkins_statistic` e `cluster_density`: estatística de tendência de agrupamento via `KDTree` e fração de pontos agrupados via `DBSCAN`.
    5. `mst_max_degree` e `mst_std_degree`: grau máximo (presença de *hubs*) e desvio-padrão dos graus na MST extraída via `scipy.sparse.csgraph.minimum_spanning_tree`.
    6. `nn_cv`: coeficiente de variação ($\sigma / \mu$) das distâncias ao vizinho mais próximo via `scipy.spatial.KDTree`.

### 2.4. `src/heuristics.py` e `src/approximations.py`
 Contrato unificado para todas as 5 estratégias de inicialização e aproximação:
- **Assinaturas**:
  - `cold_start_tsp(coords: np.ndarray, dist_matrix: np.ndarray) -> tuple[int, list[int], dict[str, float]]`
  - `nearest_neighbor_tsp(coords: np.ndarray, dist_matrix: np.ndarray) -> tuple[int, list[int], dict[str, float]]`
  - `convex_hull_insertion_tsp(coords: np.ndarray, dist_matrix: np.ndarray) -> tuple[int, list[int], dict[str, float]]`
  - `twice_around_the_tree_tsp(coords: np.ndarray, dist_matrix: np.ndarray) -> tuple[int, list[int], dict[str, float]]`
  - `christofides_tsp(coords: np.ndarray, dist_matrix: np.ndarray) -> tuple[int, list[int], dict[str, float]]`
- **Pré-condição**: `coords` `(N, 2)` e `dist_matrix` `(N, N)` válidos com $N \ge 3$.
- **Pós-condição**: Retornam `(cost, tour, telemetry)` onde `tour` é uma permutação exata de `0..N-1` com `tour[0] == 0`, `cost == compute_tour_cost(dist_matrix, tour)` e `telemetry` contém tempos e memória por subrotina.

### 2.5. `src/branch_and_bound.py`
- **`BranchAndBoundTSP(coords: np.ndarray, dist_matrix: np.ndarray, time_limit: float = 1800.0, use_geom_pruning: bool = True, mst_mode: str = "numpy_prim")`**
- **`solve(initial_ub: int | None = None, initial_tour: list[int] | None = None) -> dict[str, int | float | bool | list[int] | None]`**
  - **Invariantes Algorítmicas**:
    1. **Fixação de Raiz e Quebra de Simetria**: `path[0] = 0`, `visited_mask = 1`. Na folha (`count == N - 1`), exige-se `path[1] < nxt` para eliminar ciclos reversos equivalentes em grafos simétricos.
    2. **Admissibilidade do Limite Inferior de Grau 2 (Nível 1 - $\mathcal{O}(1)$)**:
       $$\text{LB}_{\text{fast}} = \text{curr\_cost} + \left\lfloor \frac{\text{rem\_two\_min} + \text{two\_min}[0] + \text{two\_min}[\text{curr}] + 1}{2} \right\rfloor$$
       Para garantir que $\text{LB}_{\text{fast}}$ seja sempre um limite inferior estrito/válido mesmo quando as arestas já percorridas no caminho parcial têm peso maior que as duas menores arestas incidentes a um vértice visitado, o termo acumulado para o extremo `0` usa a menor aresta de `0` para o conjunto não visitado (ou `min_edge[0]`) e para `curr` usa a menor aresta de `curr` para os não visitados (ou `min_edge[curr]`), garantindo $\text{LB}_{\text{fast}} \le \text{OPT}$ sem falsos positivos de poda.
    3. **Poda de 1-Árvore / MST sobre Não-Visitados (Nível 3)**:
       $$\text{LB}_{\text{mst}} = \text{curr\_cost} + w(\text{MST}(U)) + \min_{u \in U} d(0, u) + \min_{u \in U} d(\text{curr}, u)$$
       Como toda solução completando o caminho parcial precisa formar um caminho hamiltoniano sobre $U$ conectado a `curr` e a `0` (cuja união com qualquer aresta é uma 1-árvore sobre $U \cup \{0, \text{curr}\}$), $\text{LB}_{\text{mst}}$ é estritamente admissível ($\text{LB}_{\text{mst}} \le \text{OPT}$).
    4. **Poda Geométrica 2-Opt (Nível 2 - $\mathcal{O}(k)$)**:
       Descarta arestas `(curr, nxt)` que cruzam estritamente segmentos anteriores `(path[i], path[i+1])` quando o ganho triangular `d(a,b) + d(c,d) > d(a,c) + d(b,d)` na métrica `EUC_2D` garante que o descruzamento reduz estritamente o custo, preservando a otimalidade sob arredondamento `EUC_2D`.
    5. **Verificação de Timeout em $\mathcal{O}(1)$ Amortizado**:
       Checa `time.perf_counter()` apenas quando `(nodes_expanded & 1023) == 0`.

### 2.6. `src/analysis.py` e `main.py`
- **`run_full_analysis(metrics_csv: Path, features_csv: Path, output_dir: Path) -> dict`**
  - Ajusta curvas teóricas ($a N^2 + b N + c$ para MST/Twice-around, $a N^3 + b N^2 + c$ para Matching/Christofides, $a \cdot b^N$ via regressão log-linear para B&B), calcula $R^2$ e RMSE, avalia o *Speedup Líquido* dos 5 warm-starts, treina `DecisionTreeRegressor(max_depth=3)` e `RandomForestRegressor` para explicar o Gap de Aproximação a partir das features topológicas, e exporta gráficos (`.png` e `.pdf`) em `results/plots/` e tabelas LaTeX (`.tex`) em `results/tables/`.

---

## 3. Complexidade Assintótica e Propagação de Erros

| Módulo / Algoritmo | Complexidade de Tempo | Complexidade de Espaço |
|---|---|---|
| Matriz de Distâncias (`cdist`) | $\mathcal{O}(N^2)$ | $\mathcal{O}(N^2)$ |
| Cold Start | $\mathcal{O}(N)$ | $\mathcal{O}(N)$ |
| Vizinho Mais Próximo (NN) | $\mathcal{O}(N^2)$ | $\mathcal{O}(N)$ |
| Inserção por Fecho Convexo | $\mathcal{O}(N^2 \log N)$ ou $\mathcal{O}(N^3)$ | $\mathcal{O}(N^2)$ |
| Twice-Around-The-Tree | $\mathcal{O}(N^2)$ | $\mathcal{O}(N^2)$ |
| Christofides (Edmonds Blossom) | $\mathcal{O}(N^3)$ | $\mathcal{O}(N^2)$ |
| Branch-and-Bound (DFS In-Place) | $\mathcal{O}(N! \cdot N^2)$ pior caso | $\mathcal{O}(N^2)$ (pilha $\mathcal{O}(N)$) |

- **Política de Erros**: Nenhuma exceção genérica (`except Exception:`) é silenciada. Erros de I/O (`FileNotFoundError`, `OSError`), entradas malformadas (`ValueError`) ou falhas de convergência numérica (`RuntimeError`) são tratados explicitamente com tipos específicos ou propagados integralmente.
