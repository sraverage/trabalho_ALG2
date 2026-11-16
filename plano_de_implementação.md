# Especificação Técnica e Plano de Implementação — TP2 Algoritmos 2 (TSP)

## 1. Contexto e Diretrizes Globais para o Agente de IA

Você é um Engenheiro de Algoritmos Sênior encarregado de implementar o Trabalho Prático 2 da disciplina DCC207 (Algoritmos 2 - UFMG) em **Python 3**. O objetivo é resolver o Problema do Caixeiro Viajante (TSP) Geométrico (`EUC_2D` da TSPLIB) combinando um algoritmo exato (**Branch-and-Bound**) com cinco estratégias de inicialização (**Warm-Start**), além de realizar *profiling* de subrotinas, modelagem de regressão assintótica e análise de sensibilidade topológica.

### REGRA DE OURO: USO MÁXIMO DE BIBLIOTECAS EXTERNAS (ZERO REIVENÇÃO DA RODA)

O enunciado permite o uso da biblioteca padrão do Python, `NumPy`, `SciPy`, `Scikit-learn`, `Pandas` e bibliotecas de grafos (`NetworkX`). **Você DEVE delegar o máximo possível de cálculos geométricos, matriciais, estatísticos e de grafos para essas bibliotecas otimizadas em C/Fortran**, escrevendo do zero **apenas** a lógica de orquestração e o motor do Branch-and-Bound.

* **NÃO implemente** cálculo de distância par-a-par com `for` duplo: use `scipy.spatial.distance.cdist`.

* **NÃO implemente** Fecho Convexo (Graham Scan / Jarvis March) do zero: use `scipy.spatial.ConvexHull`.

* **NÃO implemente** o algoritmo de Blossom (Edmonds) para emparelhamento perfeito de peso mínimo do zero: use `networkx.algorithms.matching.min_weight_matching`.

* **NÃO implemente** Árvore Geradora Mínima (MST) do zero nas heurísticas, aproximações e extração de features: use `scipy.sparse.csgraph.minimum_spanning_tree` ou `networkx.minimum_spanning_tree`.

* **NÃO implemente** Circuito Euleriano do zero: use `networkx.MultiGraph` e `networkx.eulerian_circuit`.

* **NÃO implemente** regressão ou árvores de decisão do zero: use `scipy.optimize.curve_fit` e `sklearn`.

* **Obrigatoriedade de Profiling:** Embora as chamadas usem bibliotecas prontas (`networkx`/`scipy`), **toda chamada a uma subrotina crítica (como MST e Matching)** deve ser encapsulada em um bloco de medição de tempo (`time.perf_counter()`) e memória (`tracemalloc`) para cumprir a exigência de profiling em nível de subrotina do enunciado.

## 2. Arquitetura de Diretórios do Projeto

Construa o projeto exatamente com a seguinte estrutura modular:

```
tp2_tsp/
├── data/
│   └── tsplib/                 # Arquivos .tsp baixados ou gerados (EUC_2D, N entre 10 e 75)
├── results/
│   ├── metrics_log.csv         # Log consolidado (checkpoint por instância/algoritmo)
│   ├── features_log.csv        # Features geométricas e topológicas extraídas das instâncias
│   └── plots/                  # Gráficos gerados para o artigo SBC
├── src/
│   ├── __init__.py
│   ├── io_tsplib.py            # Parser TSPLIB + Matriz de Distâncias EUC_2D via SciPy
│   ├── profiler.py             # Decoradores/Context Managers de tempo e memória por subrotina
│   ├── features.py             # Extração das 5 features topológicas/geométricas via SciPy/Sklearn
│   ├── approximations.py       # Twice-around-the-tree e Christofides (usando NetworkX/SciPy)
│   ├── heuristics.py           # Cold-start, Nearest Neighbor e Convex Hull Insertion (via SciPy)
│   ├── branch_and_bound.py     # Algoritmo Exato B&B (DFS In-Place + Bitmask + Funil de Poda)
│   └── analysis.py             # Regressão de complexidade (SciPy) e Análise de Sensibilidade (Sklearn)
├── main.py                     # Pipeline principal com checkpointing e controle de timeout (30 min)
└── requirements.txt            # numpy, scipy, networkx, scikit-learn, pandas, matplotlib, seaborn

```

## 3. Ordem Estrita de Implementação (Passo a Passo)

### Fase 1: Ingestão de Dados, Discretização `EUC_2D` e Profiling (`io_tsplib.py` e `profiler.py`)

1. **Parser TSPLIB (`io_tsplib.py`):**

   * Crie um leitor capaz de ler arquivos `.tsp` (seção `NODE_COORD_SECTION`) e extrair um array `numpy` de shape `(N, 2)` com as coordenadas `(x, y)`.

   * Crie também um utilitário que baixe automaticamente (via `urllib` ou `requests`) ou carregue localmente um conjunto de 14 a 16 instâncias da TSPLIB respeitando a distribuição do edital:

     * **80% na Faixa Principal (**$N \le 40$**):** ex: `burma14` (convertido/sintético EUC_2D se necessário), `ulysses16`, `ulysses22`, `gr24`, `fri26`, `bayg29`, `bays29`, `danzig42` / `swiss42` (ou subconjuntos truncados de instâncias `EUC_2D` puras como `eil51`, `berlin52`, `st70` nos tamanhos $N \in \{12, 16, 20, 24, 28, 32, 36, 40\}$ caso alguma instância clássica não seja `EUC_2D` nativa). *Nota: garanta que todas sejam avaliadas com a métrica `EUC_2D`.*

     * **20% na Faixa Limite (**$N \in [45, 75]$**):** `att48`, `eil51`, `berlin52`, `st70`, `eil76`.

2. **Matriz de Distâncias Vetorizada:**

   * Calcule a matriz de distâncias usando `scipy.spatial.distance.cdist(coords, coords, metric='euclidean')`.

   * Aplique a regra oficial da TSPLIB `EUC_2D`: `dist_matrix = np.floor(raw_dist + 0.5).astype(np.int64)`. Zere a diagonal principal (`np.fill_diagonal(dist_matrix, 0)`).

3. **Módulo de Instrumentação (`profiler.py`):**

   * Implemente um `ContextManager` ou decorador que meça:

     * Tempo de parede (`time.perf_counter()`).

     * Pico de alocação de memória RAM (`tracemalloc.get_traced_memory()`).

   * Deve registrar métricas separadas por subrotina (`mst_time`, `matching_time`, `eulerian_time`, `convex_hull_time`, `bb_total_time`, `bb_mst_bound_time`, `peak_memory_kb`).

### Fase 2: Extração de Features das Instâncias (`features.py`)

Implemente uma função `extract_instance_features(coords: np.ndarray, dist_matrix: np.ndarray) -> dict` usando exclusivamente funções prontas do `scipy`, `numpy` e `sklearn`:

1. **Razão de Aspecto (`aspect_ratio`):** Usando `np.ptp(coords, axis=0)` (amplitude máxima em X e Y), calcule $\frac{\min(\Delta x, \Delta y)}{\max(\Delta x, \Delta y)}$.

2. **Proporção de Pontos no Fecho Convexo (`convex_hull_ratio`):** Execute `hull = scipy.spatial.ConvexHull(coords)` e retorne `len(hull.vertices) / len(coords)`.

3. **Entropia Espacial / Densidade de Clustering (`spatial_entropy` e `hopkins_statistic`):**

   * Use `scipy.stats.entropy` sobre um histograma 2D (`np.histogram2d`) das coordenadas normalizadas para medir a entropia da distribuição espacial.

   * Complementarmente, use `sklearn.cluster.DBSCAN` ou a distância média aos $k$-vizinhos mais próximos via `scipy.spatial.KDTree` para medir a tendência de agrupamento (clustering).

4. **Estatísticas da MST (`mst_max_degree`, `mst_std_degree`):**

   * Gere a MST com `mst = scipy.sparse.csgraph.minimum_spanning_tree(dist_matrix)`.

   * Converta para binário (`(mst > 0).astype(int)`), some linhas e colunas para obter o grau de cada vértice na árvore e extraia o grau máximo (`np.max(degrees)`) — que identifica a presença de *hubs*.

5. **Coeficiente de Variação do Vizinho Mais Próximo (`nn_cv`):**

   * Com `scipy.spatial.KDTree(coords).query(coords, k=2)`, pegue o vetor de distâncias para o vizinho mais próximo ($k=2$, índice 1) e retorne `np.std(nn_dists) / np.mean(nn_dists)`.

### Fase 3: Algoritmos Aproximados e Heurísticas de Warm-Start (`approximations.py` e `heuristics.py`)

Todos os 5 métodos devem retornar uma tupla Padronizada: `(cost: int, tour: list[int], telemetry: dict)`. O `tour` deve ser uma permutação de `0` a `N-1` começando no vértice `0`.

1. **Estratégia 1 — Cold Start / Trivial (`heuristics.py`):**

   * Rota fixa `[0, 1, 2, ..., N-1]`. Custo calculado de forma vetorizada com indexação do NumPy.

2. **Estratégia 2 — Busca Gulosa / Vizinho Mais Próximo (`heuristics.py`):**

   * Começa em `0`. A cada passo, usa `np.argmin` mascarando os visitados na linha da `dist_matrix` para ir ao vértice livre mais próximo.

3. **Estratégia 3 — Inserção por Fecho Convexo (`heuristics.py`):**

   * Obtenha o ciclo inicial diretamente de `list(scipy.spatial.ConvexHull(coords).vertices)` (que já vem ordenado no sentido anti-horário).

   * Enquanto houver vértices fora do subtour: use operações matriciais do `numpy` para calcular o custo de inserção $c(i, k) + c(k, j) - c(i, j)$ de todos os nós livres $k$ em todas as arestas $(i, j)$ do subtour. Insira de forma gulosa o nó $k^*$ que minimiza a razão ou o aumento de custo. Meça o tempo do `ConvexHull` no `telemetry`.

4. **Estratégia 4 — Algoritmo Aproximado Twice-Around-The-Tree (`approximations.py`):**

   * **Subrotina 1 (com profiling):** Construa um grafo `G = nx.from_numpy_array(dist_matrix)` e extraia a MST via `T = nx.minimum_spanning_tree(G, algorithm='prim')`.

   * **Subrotina 2 (com profiling):** Execute ` tour = list(nx.dfs_preorder_nodes(T, source=0))`. Como o `dfs_preorder_nodes` já retorna os vértices na ordem exata de pré-ordem (eliminando duplicatas automaticamente, equivalente ao *shortcutting* do circuito euleriano na árvore duplicada), basta calcular o custo desse ciclo fechado.

5. **Estratégia 5 — Algoritmo Aproximado de Christofides (`approximations.py`):**

   * **Subrotina 1 - MST (com profiling):** `T = nx.minimum_spanning_tree(G, algorithm='prim')`.

   * **Subrotina 2 - Vértices de Grau Ímpar:** Filtre `odd_nodes = [v for v, d in T.degree() if d % 2 == 1]`.

   * **Subrotina 3 - Matching Perfeito de Peso Mínimo (com profiling de tempo e memória):** Crie o subgrafo induzido `odd_subgraph = G.subgraph(odd_nodes)` e chame **`matching = nx.algorithms.matching.min_weight_matching(odd_subgraph, weight='weight')`**.

   * **Subrotina 4 - Multigrafo e Circuito Euleriano (com profiling):** Crie `M = nx.MultiGraph(T)`, adicione as arestas do `matching` com `M.add_edges_from(matching)` e gere o circuito via `euler_circuit = nx.eulerian_circuit(M, source=0)`.

   * **Subrotina 5 - Shortcutting (Ciclo Hamiltoniano):** Percorra o `euler_circuit` mantendo um `set` de visitados para construir a lista `tour` sem repetições e calcule o custo final.

### Fase 4: O Algoritmo Exato Branch-and-Bound (`branch_and_bound.py`)

Para atingir máxima performance em Python sem estourar a RAM, **NÃO use `heapq` (Best-First Search)** e **NÃO copie matrizes**. Implemente **Busca em Profundidade (DFS In-Place)** com as seguintes otimizações obrigatórias:

1. **Pré-computação Estática no `__init__`:**

   * `self.sorted_neighbors = np.argsort(dist_matrix, axis=1)`: Para cada nó, ordena os vizinhos do mais próximo ao mais distante (Branch Ordering).

   * `sorted_dists = np.sort(dist_matrix, axis=1)`

   * `self.two_min_sum = sorted_dists[:, 1] + sorted_dists[:, 2]`: Vetor com a soma das duas menores arestas de cada vértice.

2. **Quebra de Simetria:**

   * Fixe o vértice inicial em `0` (`path[0] = 0`, `visited_mask = 1`).

   * Na folha (quando `count == n - 1`, escolhendo o último vértice `nxt` antes de voltar para `0`), imponha `if path[1] > nxt: continue`. Isso elimina rotas invertidas espelhadas e corta 50% da árvore.

3. **Representação de Estado em Inteiro (Bitmask):**

   * Passe `visited_mask: int` na recursão. Para checar se `u` foi visitado: `visited_mask & (1 << u)`. Para marcar: `visited_mask | (1 << u)`. Isso funciona perfeitamente em Python para $N \le 75$ pois inteiros em Python têm precisão arbitrária e zero alocação de listas.

4. **Funil de Poda em 3 Níveis (Hierarquia de Lower Bounds):**

   * **Nível 1 — Poda Rápida de Grau 2 em** $\mathcal{O}(1)$**:** Mantenha na recursão `rem_two_min` (soma de `two_min_sum` dos vértices não visitados). Calcule:
     `fast_lb = curr_cost + (rem_two_min + two_min_sum[0] + two_min_sum[curr] + 1) // 2`
     Se `fast_lb >= self.best_cost`: incremente `nodes_pruned_fast` e retorne (`return`).

   * **Nível 2 — Poda Geométrica de Cruzamento 2-Opt em** $\mathcal{O}(k)$**:** Antes de adicionar a aresta `(curr, nxt)`, verifique se o segmento `coords[curr] -> coords[nxt]` se intersecta geometricamente com alguma aresta anterior `coords[path[i]] -> coords[path[i+1]]` do caminho parcial. Se houver cruzamento estrito, descarte o ramo (`nodes_pruned_geom += 1`).

   * **Nível 3 — Poda Forte de 1-Árvore (MST sobre não-visitados):** Se `n - count > 3` e o nó passou pelo Nível 1:

     * Extraia os índices não visitados `unvisited`.

     * Como $N - \text{count}$ é pequeno (tipicamente entre 4 e 70), extraia a submatriz com indexação do NumPy: `sub_mat = self.dist[np.ix_(unvisited, unvisited)]`.

     * Calcule o peso da MST e a menor conexão de `0` para `unvisited` (`self.dist[0, unvisited].min()`) e de `curr` para `unvisited` (`self.dist[curr, unvisited].min()`).

     * *Detalhe de Engenharia:* Para calcular o peso da MST de `sub_mat` milhões de vezes sem overhead de conversão de objetos, implemente um Prim denso vetorizado em NumPy (atualizando um array `min_e = np.minimum(min_e, sub_mat[u])` em um loop de tamanho `k`) **OU** use `scipy.sparse.csgraph.minimum_spanning_tree(sub_mat).sum()` (meça no profiler qual dos dois tem menor overhead para matrizes $20 \times 20$ e documente!).

     * Se `curr_cost + mst_weight + min_0 + min_curr >= self.best_cost`: incremente `nodes_pruned_mst` e retorne.

5. **Controle de Timeout de Baixo Overhead (30 Minutos = 1800s):**

   * A cada chamada de `_dfs`, incremente `self.nodes_expanded += 1`.

   * Apenas quando `(self.nodes_expanded & 1023) == 0`, cheque `time.perf_counter() - self.start_time > self.time_limit`. Se estourar, marque `self.aborted = True` e desempilhe imediatamente. Quando abortado, o custo ótimo deve ser reportado como `NA` no CSV final, mas todas as contagens (`nodes_expanded`, `nodes_pruned_fast`, `nodes_pruned_mst`, `best_ub_found_so_far`, `peak_memory_kb`) devem ser salvas!

### Fase 5: Modelagem Estatística, Regressão e Análise de Sensibilidade (`analysis.py`)

Use `pandas`, `scipy.optimize`, `sklearn` e `matplotlib`/`seaborn` para gerar todas as análises pedidas no artigo científico:

1. **Modelagem de Complexidade Empírica (Regressão Assintótica):**

   * Com `scipy.optimize.curve_fit`, ajuste os tempos coletados das subrotinas e algoritmos em função de $N$:

     * Para MST e Twice-Around-The-Tree: ajuste a curva polinomial $f(N) = a \cdot N^2 + b \cdot N + c$.

     * Para Matching de Christofides: ajuste $f(N) = a \cdot N^3 + b \cdot N^2 + c$.

     * Para o Branch-and-Bound (tempo e nós expandidos na faixa $N \le 40$): ajuste uma curva exponencial $f(N) = a \cdot b^N$ (aplicando regressão linear `sklearn.linear_model.LinearRegression` sobre $\log(\text{tempo})$ vs $N$).

   * Calcule e salve em tabela os valores de $R^2$ (`sklearn.metrics.r2_score`) e **RMSE** (`sklearn.metrics.root_mean_squared_error`) para tempo e memória.

2. **Avaliação do Impacto do Warm-Start:**

   * Gere gráficos de barras agrupadas (escala logarítmica no eixo Y) comparando `nodes_expanded` e `nodes_pruned` para as 5 estratégias de inicialização em cada instância.

   * Calcule a métrica de *Speedup Líquido*: $\frac{T_{\text{B\&B}}(\text{Cold Start})}{T_{\text{Heurística}} + T_{\text{B\&B}}(\text{Warm Start})}$ para responder se o tempo gasto na heurística compensa a redução da árvore.

3. **Análise de Sensibilidade (Correlação Features vs. Gap de Aproximação):**

   * Para cada instância com ótimo conhecido (do B&B ou do gabarito TSPLIB), calcule o **Gap de Aproximação (%)**: $\text{Gap} = 100 \times \frac{\text{Custo}_{\text{Heurística}} - \text{Ótimo}}{\text{Ótimo}}$.

   * Treine um modelo `sklearn.ensemble.RandomForestRegressor` e uma `sklearn.tree.DecisionTreeRegressor` (profundidade máxima 3) onde `X` são as 5 features topológicas extraídas na Fase 2 e `y` é o Gap do Christofides / Twice-Around-The-Tree / taxa de poda do B&B.

   * Extraia a importância das features (`feature_importances_`) e gere uma matriz de correlação de Spearman (`df.corr(method='spearman')`) e gráficos de dispersão mostrando quais características geométricas atuam como instâncias adversariais.

### Fase 6: Pipeline de Execução com Checkpointing (`main.py`)

1. O script `main.py` deve aceitar flags de linha de comando (`--mode fast-test` com time limit de 10s por instância para validar todo o pipeline em 2 minutos, e `--mode full` com time limit de 1800s para o experimento oficial).

2. **Checkpointing Incremental:** A cada execução de `(instância, algoritmo)` concluída, grave a linha imediatamente no arquivo `results/metrics_log.csv` (usando `mode='a'`) e dê `flush()`. Se o computador desligar ou o processo for interrompido após 3 horas, ao rodar `main.py` novamente ele deve ler `metrics_log.csv`, pular os pares `(instância, algoritmo)` já executados e continuar de onde parou.

3. Ao final da coleta, `main.py` deve invocar automaticamente `src/analysis.py` para gerar todas as tabelas em formato LaTeX (`.tex` prontas para `\input{}` no template SBC) e salvar todos os gráficos em alta resolução (`.png` e `.pdf`) na pasta `results/plots/`.