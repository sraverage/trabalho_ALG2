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
├── main.py                         # Orquestrador CLI com checkpointing incremental e timeout
├── requirements.txt                # Dependências declaradas (numpy, scipy, networkx, scikit-learn, pandas, matplotlib)
├── enunciado_trabalho.pdf          # Enunciado oficial do TP2 (DCC207)
├── TECHNICAL-SPEC.md               # Especificação técnica de contratos, invariantes e complexidade
├── PLAN.md                         # Plano atômico de execução e testes
├── README.md                       # Documentação técnica detalhada do funcionamento e entregas
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
│   ├── metrics_log.csv             # Base completa de telemetria (80 execuções: 16 instâncias x 5 estratégias)
│   ├── features_log.csv            # Features geométricas e topológicas das 16 instâncias
│   ├── mst_benchmark.csv           # Benchmark comparativo Prim in-place vs SciPy csgraph (submatriz 20x20)
│   ├── regression_models.csv       # Parâmetros ajustados, R² e RMSE das curvas de complexidade
│   ├── sensitivity_importance.csv  # Importância das features (Random Forest) e R² por alvo
│   ├── decision_tree_rules.txt     # Regras extraídas da DecisionTreeRegressor (profundidade 3) para Gap Christofides
│   ├── plots/                      # Gráficos em alta resolução (.png 300 dpi e .pdf)
│   └── tables/                     # Tabelas prontas para \input{} no LaTeX (.tex)
├── report/
│   ├── relatorio_sbc.tex           # Código-fonte do artigo científico em padrão SBC (com Anexo de Prompts)
│   ├── relatorio_sbc.pdf           # Relatório final compilado (10 páginas)
│   ├── references.bib              # Referências bibliográficas BibTeX
│   ├── sbc-template.sty            # Estilo oficial SBC
│   └── sbc.bst                     # Estilo bibliográfico SBC
├── tests/
│   └── test_tsp_pipeline.py        # Suíte de 9 testes unitários e de integração end-to-end
└── tp2_tsp/                        # Diretório compatível com symlinks para data/, src/, results/, main.py e requirements.txt
```

---

## 2. Como Tudo Funciona (Arquitetura e Detalhes Algorítmicos)

### 2.1. Ingestão TSPLIB, Padronização `EUC_2D` e Matriz de Distâncias (`src/io_tsplib.py`)

O conjunto experimental é composto por **16 instâncias da TSPLIB** variando de $N = 10$ a $N = 70$:
- **Faixa Principal ($N \le 42$, 12 instâncias — 75% do conjunto):** `eil10`, `berlin12`, `burma14`, `ulysses16`, `st18`, `eil20`, `ulysses22`, `gr24`, `fri26`, `bayg29`, `bays29`, `swiss42`.
- **Faixa Limite / Estresse ($N \in [48, 70]$, 4 instâncias — 25% do conjunto):** `att48`, `eil51`, `berlin52`, `st70`.

Como várias instâncias clássicas da TSPLIB nessa faixa de tamanho originalmente usam coordenadas geográficas em graus (`burma14`, `ulysses16`, `ulysses22`) ou matrizes explícitas de distância (`gr24`, `fri26`, `bays29`, `swiss42`), `ensure_tsplib_dataset` padroniza todo o conjunto para coordenadas bidimensionais `EUC_2D`:
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
- Mede o pico de memória alocada com `tracemalloc`. Quando um rastreamento pai já está ativo, utiliza `tracemalloc.reset_peak()` para isolar o pico específico da subrotina sem interromper o contexto externo.
- Registra `<prefix>_time` (segundos), `<prefix>_peak_kb` (KiB) e atualiza `peak_memory_kb`.

---

### 2.3. Extração de Features Geométricas e Topológicas (`src/features.py`)

A função `extract_instance_features(coords, dist_matrix)` extrai métricas estruturais para explicar a dificuldade de cada instância:
- `aspect_ratio`: razão $\frac{\min(\Delta x, \Delta y)}{\max(\Delta x, \Delta y)} \in (0, 1]$ do *bounding box* via `np.ptp(arr, axis=0)`.
- `convex_hull_ratio`: fração de vértices pertencentes ao fecho convexo (`len(ConvexHull(arr).vertices) / N`).
- `spatial_entropy`: entropia de Shannon normalizada (`scipy.stats.entropy`) sobre histograma 2D (`np.histogram2d`) das coordenadas normalizadas em $[0, 1]^2$.
- `hopkins_statistic`: estatística determinística de Hopkins (semente fixa `42`) via `scipy.spatial.KDTree`, comparando distâncias quadráticas ao vizinho mais próximo de pontos de sondagem uniformes contra os pontos reais (valores maiores indicam maior tendência a *clusters*).
- `cluster_density`: fração de vértices agrupados pelo `sklearn.cluster.DBSCAN` (não classificados como ruído `-1`).
- `mst_max_degree` e `mst_std_degree`: grau máximo (presença de *hubs*) e desvio-padrão dos graus na MST obtida por `scipy.sparse.csgraph.minimum_spanning_tree`.
- `nn_cv`: coeficiente de variação ($\sigma / \mu$) das distâncias ao vizinho mais próximo via `KDTree(..., k=2)`.

---

### 2.4. Heurísticas de *Warm-Start* e Algoritmos Aproximados (`src/heuristics.py`, `src/approximations.py`)

Todas as cinco estratégias seguem o contrato único `(cost, tour, telemetry)` retornando uma permutação válida iniciada em `0`:

1. **Cold Start (`cold_start_tsp`):** rota trivial `[0, 1, ..., N-1]`. Complexidade $\mathcal{O}(N)$.
2. **Vizinho Mais Próximo (`nearest_neighbor_tsp`):** construção gulosa a partir do vértice `0` mascarando visitados e escolhendo `np.argmin` em cada passo. Complexidade $\mathcal{O}(N^2)$.
3. **Inserção por Fecho Convexo (`convex_hull_insertion_tsp`):** inicializa a sub-rota com os vértices de `scipy.spatial.ConvexHull(coords)` e, a cada iteração, avalia de forma **vetorizada** via `np.ix_` a matriz de custo marginal $\Delta(u, v, k) = d_{uk} + d_{kv} - d_{uv}$ para todos os nós livres $k$ e todas as arestas $(u, v)$ do subciclo, inserindo o par ótimo até fechar o tour e rotacionando para iniciar em `0`.
4. **Twice-around-the-tree (`twice_around_the_tree_tsp`):** constrói o grafo completo em `NetworkX` (incluindo explicitamente eventuais arestas de peso `0` entre pontos coincidentes após arredondamento), extrai a MST com `nx.minimum_spanning_tree(..., algorithm="prim")` sob `profile_block(..., "mst")` e percorre em pré-ordem com `nx.dfs_preorder_nodes(..., source=0)` sob `profile_block(..., "dfs_preorder")`. Garantia de 2-aproximação no caso métrico.
5. **Christofides-Serdyukov (`christofides_tsp`):** extrai a MST via Prim (`"mst"`), filtra vértices de grau ímpar, calcula o emparelhamento perfeito de peso mínimo com `nx.algorithms.matching.min_weight_matching` — algoritmo Blossom de Edmonds (`"matching"`), une as arestas em um `nx.MultiGraph` e extrai o circuito euleriano via `nx.eulerian_circuit` (`"eulerian"`), realizando *shortcutting* linear (`"shortcutting"`). Garantia teórica de $3/2$-aproximação.

---

### 2.5. Motor Exato *Branch-and-Bound* (`src/branch_and_bound.py`)

Para evitar esgotamento de RAM e pressão de *garbage collection* típicos de *Best-First Search* (`heapq` com cópias de matrizes $\mathcal{O}(N^2)$ na fronteira), o `BranchAndBoundTSP` implementa **Busca em Profundidade (DFS *In-Place*)** com espaço de pilha estritamente $\mathcal{O}(N)$:

1. **Representação de Estado em *Bitmask*:** o conjunto de vértices visitados é um inteiro Python `visited_mask` manipulado com operações bit-a-bit em $\mathcal{O}(1)$ (`visited_mask & (1 << u)` e `visited_mask | (1 << u)`).
2. **Pivotamento na Raiz por *Outlier* e Quebra de Simetria:** a busca é enraizada em `root_vertex = argmax_v (d_(1)(v) + d_(2)(v))`, decidindo na raiz e na folha as arestas do vértice mais isolado (crítico em instâncias como `ulysses16`). Na penúltima camada (`count == n - 1`), impõe-se `path[1] < nxt` para eliminar $50\%$ das rotas espelhadas equivalentes. Ao fim, o melhor ciclo é rotacionado de volta ao vértice `0`.
3. **Potenciais Lagrangianos de Held-Karp em Ponto Fixo (`_compute_hk_potentials`):** antes da DFS, o construtor executa até `180` iterações de ascensão por subgradiente sobre 1-árvores enraizadas em `root_vertex` com passo tipo Polyak amortecido e centrado ($\sum_v \pi_v = 0$). Os multiplicadores são convertidos para inteiros em escala fixa `HK_SCALE = 256` (`self.pi_int`), gerando a matriz de custos penalizados $\tilde{d}_{uv} = 256 \cdot d_{uv} + \Pi_u + \Pi_v$.
4. **Ordenação de Vizinhos por Custo Reduzido:** para cada vértice, os vizinhos candidatos são pré-ordenados por $\tilde{d}_{uv} - \lfloor \min_w \tilde{d}_{wv} / 2 \rfloor$, guiando a DFS diretamente pelos arcos promissores da relaxação de Held-Karp.
5. **Funil de Poda em 3 Níveis Estritamente Admissível:**
   - **Nível 1 — Limites Rápidos de Grau 2 em $\mathcal{O}(1)$:** verifica dois limites incrementais: o limite de grau 2 original $\text{LB}_{\text{fast}}$ (usando apenas 1 menor aresta nos extremos `root` e `curr`, e 2 menores arestas nos vértices livres $U$) e o limite penalizado $\text{LB}_{\text{fast}}^{(\Pi)}$ em escala `HK_TWO_SCALE = 512`. Também poda transições cujo custo parcial `curr_cost + dist_curr[nxt] >= best_cost`.
   - **Nível 2 — Filtro Geométrico 2-Opt sob Discretização `EUC_2D` (`_violates_2opt_geometry`):** descarta a aresta `(curr, nxt)` apenas quando há interseção própria no plano via teste de orientação CCW (`_segments_strictly_intersect`) **e** ganho inteiro estrito $d(u,v) + d(\text{curr},\text{nxt}) > d(u,\text{curr}) + d(v,\text{nxt})$, evitando descartar empates de arredondamento inteiro.
   - **Nível 3 — Relaxação Lagrangiana de 1-Árvore de Held-Karp (`_compute_mst_lower_bound`):** para $|U| \ge 3$, calcula o peso da MST sobre $U$ na matriz penalizada $\tilde{d}$ via Prim denso *in-place* em listas Python (`_reduce_prim_edges`), soma as duas menores conexões de `root` e `curr` para $U$, desconta $\Pi_{\text{root}} + \Pi_{\text{curr}} + 2\sum_{u \in U}\Pi_u$ e aplica teto inteiro exato sobre `256`.
6. **Controle de Timeout de Baixo *Overhead*:** verifica `time.perf_counter()` apenas a cada `1024` nós (`(nodes_expanded & 1023) == 0`). Quando o tempo expira, registra `aborted = True`, grava `"NA"` em `optimal_cost` e preserva toda a telemetria (`nodes_expanded`, contadores por tipo de poda, `best_ub_found_so_far`, `max_depth_reached`, `peak_memory_kb`).
7. **Benchmark Isolado de Subrotina MST (`benchmark_mst_subroutines`):** compara diretamente a implementação densa em NumPy (`_mst_weight_numpy_prim`) contra `scipy.sparse.csgraph.minimum_spanning_tree` em submatrizes $20 \times 20$.

---

### 2.6. Pipeline com Checkpointing e Análise Estatística (`main.py`, `src/analysis.py`)

- **Checkpointing Incremental:** a cada par `(instance, strategy)` concluído, `main.py` escreve a linha imediatamente em `results/metrics_log.csv` com `flush()`. Caso interrompido, uma nova execução sem `--reset` carrega os pares já concluídos e retoma de onde parou.
- **Modelagem Assintótica (`_build_complexity_regressions`):** ajusta curvas via `scipy.optimize.curve_fit` ($aN^2 + bN + c$ para MST, Twice-around e Pico de Memória; $aN^3 + bN^2 + c$ para Matching Blossom e Christofides) e regressão log-linear (`sklearn.linear_model.LinearRegression`) para tempo e nós expandidos do B&B ($a \cdot b^N$) sobre instâncias resolvidas na faixa $N \le 40$, calculando $R^2$ e RMSE.
- **Avaliação de *Speedup* Líquido:** calcula $\text{Speedup} = \frac{T_{\text{B\&B}}(\text{Cold Start})}{T_{\text{Heurística}} + T_{\text{B\&B}}(\text{Warm-Start})}$ para responder se o investimento de tempo na heurística inicial compensa.
- **Análise de Sensibilidade (`_run_sensitivity_analysis`):** treina `RandomForestRegressor(n_estimators=100, random_state=42)` e `DecisionTreeRegressor(max_depth=3, random_state=42)` sobre as 6 features topológicas para prever os gaps percentuais e a taxa de poda do B&B, além de gerar a matriz de correlação de Spearman e exportar gráficos (`.png` / `.pdf`) e tabelas LaTeX (`.tex`).

---

## 3. O Que Foi Feito e Principais Resultados Obtidos

### Resumo das Implementações e Auditorias Aplicadas
- Implementação completa dos 7 módulos de produção em `src/`, orquestrador `main.py`, suíte de testes em `tests/test_tsp_pipeline.py` e artigo científico SBC em `report/relatorio_sbc.tex` (`10` páginas compiladas em `report/relatorio_sbc.pdf`).
- **Correções algorítmicas auditadas:**
  - Padronização `EUC_2D` via reescalonamento `GEO` ($\times 100$) e projeção espectral MDS para matrizes explícitas (incluindo atualização do ótimo de referência exato de `bayg29` em `EUC_2D` para `9073`).
  - Correção da admissibilidade do limite rápido de grau 2 nos extremos do caminho parcial e verificação de desigualdade estrita inteira no filtro geométrico 2-Opt.
  - Pivotamento na raiz pelo vértice *outlier* e incorporação de multiplicadores lagrangianos de Held-Karp em ponto fixo (`HK_SCALE = 256`), resolvendo **100% das instâncias até $N = 29$** (11 das 12 instâncias da faixa principal — $91{,}7\%$) dentro do orçamento de `4,5 s` do modo `fast-test`.

### Principais Descobertas Empíricas (Base `results/`)
1. **Benchmark MST (`results/mst_benchmark.csv`):** para 400 repetições em submatrizes $20 \times 20$, o Prim denso em NumPy levou `0,0349 s` com pico de `2,50 KiB`, sendo **$2{,}69\times$ mais rápido** e consumindo **$7{,}56\times$ menos memória** que `scipy.sparse.csgraph.minimum_spanning_tree` (`0,0940 s` e `18,87 KiB`).
2. **Regressões Assintóticas (`results/regression_models.csv`):**
   - `MST Prim (ms)`: $\mathcal{O}(N^2)$ com $R^2 = 0{,}9735$ ($\text{RMSE} = 0{,}5150\,\text{ms}$).
   - `Twice-Around Total (ms)`: $\mathcal{O}(N^2)$ com $R^2 = 0{,}9656$ ($\text{RMSE} = 0{,}5901\,\text{ms}$).
   - `Matching Blossom (ms)`: $\mathcal{O}(N^3)$ com $R^2 = 0{,}7916$; `Christofides Total (ms)`: $R^2 = 0{,}9185$.
   - `Pico Memoria (KiB)`: $\mathcal{O}(N^2)$ com $R^2 = 0{,}9837$.
   - `B&B` ($N \le 29$, melhor estratégia resolvida): crescimento exponencial $\approx 0{,}0596 \cdot (1{,}628)^N$ nós ($R^2 = 0{,}6898$) e $\approx 6{,}27\times 10^{-7} \cdot (1{,}682)^N\,\text{s}$ ($R^2 = 0{,}7016$).
3. **Impacto do *Warm-Start* (`results/metrics_log.csv`):**
   - **Inserção por Fecho Convexo** atingiu gap médio de apenas **`2,20%`** (tempo máximo `3,19 ms` em `st70`) e resolveu exatamente todas as 11 instâncias até $N=29$, gerando *speedups* líquidos de até **`2.117,81x`** (`berlin12`), **`719,41x`** (`eil20`), **`168,03x`** (`fri26`) e **`23,68x`** (`ulysses16`).
   - **Christofides** atingiu gap médio de **`8,97%`** (tempo máximo `31,18 ms` em `st70`), resolvendo 7 instâncias exatamente com *speedups* de até **`353,35x`** (`berlin12`), **`39,92x`** (`st18`) e **`28,18x`** (`ulysses16`).
   - Na faixa de estresse ($N \in [42, 70]$), o *Branch-and-Bound* esgota o orçamento de tempo registrando `"NA"` em `optimal_cost` com telemetria preservada, enquanto as heurísticas finalizam em poucos milissegundos.
4. **Sensibilidade Topológica (`results/sensitivity_importance.csv` e `results/decision_tree_rules.txt`):**
   - `convex_hull_ratio` é a feature dominante na explicação do gap da Inserção por Fecho Convexo (`60,9%`, $R^2 = 0{,}871$), Vizinho Mais Próximo (`40,4%`, $R^2 = 0{,}728$), Taxa de Poda do B&B (`58,4%`, $R^2 = 0{,}853$, $\rho = -0{,}815$) e Christofides (`26,3%`, $R^2 = 0{,}879$, $\rho = -0{,}593$).
   - `aspect_ratio` (`49,2%`, $\rho = +0{,}636$) e `nn_cv` (`19,1%`) dominam o erro de `Twice-around-the-tree` ($R^2 = 0{,}883$).
   - Instâncias com alta entropia espacial (`spatial_entropy > 0,79`), baixa fração no fecho convexo (`convex_hull_ratio <= 0,37`) e forte agrupamento (`hopkins_statistic > 0,46`) elevam o gap médio de Christofides para **`14,60%`**.

---

## 4. Como Executar e Reproduzir

### Pré-requisitos
Crie/use o ambiente virtual `.venv` com os pacotes de `requirements.txt` (`numpy`, `scipy`, `networkx`, `scikit-learn`, `pandas`, `matplotlib`):

```bash
source .venv/bin/activate
```

### Executar Testes Unitários e de Integração
```bash
.venv/bin/python -m unittest discover -s tests -p "test_*.py" -v
```

### Executar o Pipeline Experimental
- **Modo de validação rápida** (regera todos os logs em `results/`, gráficos e tabelas LaTeX em ~1,5 minuto):
  ```bash
  .venv/bin/python main.py --mode fast-test --reset
  ```
- **Modo oficial completo** (timeout de `1800s` = 30 minutos por chamada ao B&B, com retomada automática via checkpoint se interrompido):
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
