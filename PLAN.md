# Plano de Execução Atômico (`PLAN.md`) — TP2 Algoritmos 2 (TSP Geométrico)

## Tarefa 1: Infraestrutura de Dados TSPLIB, Matriz `EUC_2D` e Profiling (`src/io_tsplib.py`, `src/profiler.py`)
- **Arquivos tocados**: `src/__init__.py`, `src/io_tsplib.py`, `src/profiler.py`, `tests/test_tsp_pipeline.py`, `requirements.txt`.
- **Teste (TDD)**: Validar parsing de coordenadas `.tsp`, simetria e arredondamento `EUC_2D` ($\lfloor d + 0.5 \rfloor$), cálculo vetorizado de custo de rota e captura de tempo/memória no `profile_block`.
- **Comando de verificação**: `python3 -m unittest discover -s tests -p "test_*.py"`

## Tarefa 2: Extração de Features Geométricas e Topológicas (`src/features.py`)
- **Arquivos tocados**: `src/features.py`, `tests/test_tsp_pipeline.py`.
- **Teste (TDD)**: Verificar invariantes matemáticas das features (`aspect_ratio` $\in (0, 1]$, `convex_hull_ratio` $\in (0, 1]$, `mst_max_degree` $\ge 2$, `nn_cv` $\ge 0$, `spatial_entropy` $\ge 0$) sobre configurações geométricas sintéticas (quadrado, círculo, clusters).
- **Comando de verificação**: `python3 -m unittest discover -s tests -p "test_*.py"`

## Tarefa 3: Heurísticas Construtivas e Algoritmos Aproximados (`src/heuristics.py`, `src/approximations.py`)
- **Arquivos tocados**: `src/heuristics.py`, `src/approximations.py`, `tests/test_tsp_pipeline.py`.
- **Teste (TDD)**: Verificar que os 5 métodos (`cold_start`, `nearest_neighbor`, `convex_hull_insertion`, `twice_around_the_tree`, `christofides`) retornam permutações válidas iniciando em `0`, custos consistentes com `compute_tour_cost`, garantia teórica de aproximação ($\le 2 \cdot \text{OPT}$ e $\le 1.5 \cdot \text{OPT}$ na métrica euclidiana) e telemetria de subrotinas preenchida.
- **Comando de verificação**: `python3 -m unittest discover -s tests -p "test_*.py"`

## Tarefa 4: Algoritmo Exato Branch-and-Bound DFS In-Place (`src/branch_and_bound.py`)
- **Arquivos tocados**: `src/branch_and_bound.py`, `tests/test_tsp_pipeline.py`.
- **Teste (TDD)**: Confirmar que o Branch-and-Bound encontra exatamente o mesmo custo ótimo para todas as 5 estratégias de *warm-start* (invariante de otimalidade exata), reduz o número de nós expandidos conforme o *upper bound* inicial melhora, compara o tempo do Prim vetorizado vs SciPy MST e respeita o limite de tempo (`aborted=True` quando `time_limit` expira).
- **Comando de verificação**: `python3 -m unittest discover -s tests -p "test_*.py"`

## Tarefa 5: Modelagem Estatística, Regressão, Sensibilidade e Pipeline (`src/analysis.py`, `main.py`)
- **Arquivos tocados**: `src/analysis.py`, `main.py`, `tests/test_tsp_pipeline.py`.
- **Teste (TDD)**: Testar o checkpointing incremental em `metrics_log.csv`, a geração de `features_log.csv`, o ajuste de curvas assintóticas ($R^2$, RMSE), a árvore de decisão/Random Forest de sensibilidade e a exportação de gráficos (`.png`, `.pdf`) e tabelas LaTeX (`.tex`).
- **Comando de verificação**: `python3 main.py --mode fast-test && python3 -m unittest discover -s tests -p "test_*.py"`

## Tarefa 6: Execução Experimental Completa e Relatório Científico SBC em LaTeX (`report/relatorio_sbc.tex`)
- **Arquivos tocados**: `results/metrics_log.csv`, `results/features_log.csv`, `results/plots/*`, `results/tables/*`, `report/sbc-template.sty`, `report/sbc.bst`, `report/references.bib`, `report/relatorio_sbc.tex`, `report/relatorio_sbc.pdf`.
- **Verificação**: Compilar `relatorio_sbc.pdf` com `pdflatex` + `bibtex` sem erros e executar auditoria de mutação (`adversarial-verify`).
