---
description: >-
  How the graph Laplacian finds communities of chunks, why its eigenvectors make good
  coordinates, and how the same matrix spreads relevance.
---

# The spectral part

Dense embeddings tell you what a chunk *says*. SpectraFuse also wants to know where a chunk
*sits*: which chunks belong to the same neighbourhood of the corpus, even when they share few
words. The spectral part answers that from the chunk graph alone, with linear algebra instead of
hand-written rules.

<div class="spectrafuse-embed" markdown>
<iframe src="../../interactive/spectral.html" title="The spectral part, explained visually" loading="lazy"></iframe>

**Interactive explainer:** watch each eigenvector "wobble" the graph, drag the strength of the
bridge edge, and watch relevance diffuse. It includes the code behind every figure. [Open it in a full window](../interactive/spectral.html).
</div>

## A seven-chunk example

Chunks c1–c3 are about Project Orion, c4 is the Orion team chunk (*"the research is carried out
by Helix Labs"*), and c5–c7 are about Helix Labs. Links inside each group are strong; one weak
link, c4–c6, is the bridge a multi-hop question needs.

| Edges | Weight $W_{ij}$ | Meaning |
|---|---|---|
| c1–c2, c2–c3, c1–c3 | 1.0 | Orion chunks, tightly linked |
| c4–c2, c4–c3 | 0.8 | the team chunk belongs with Orion |
| c4–c6 | 0.3 | the bridge: shared entity *Helix Labs* |
| c5–c6, c6–c7, c5–c7 | 1.0 | Helix chunks, tightly linked |

## Springs and the Laplacian

Picture every chunk as a ball and every edge as a spring whose stiffness is its weight. A
chunk's *degree* $d_i = \sum_j W_{ij}$ is the total stiffness attached to it. SpectraFuse uses the
symmetric normalised Laplacian (paper Eq. 11), which divides each spring by the degrees at both
ends so that heavily connected chunks don't dominate:

$$
L_{sym} = I - D^{-1/2} \, W \, D^{-1/2}
$$

Give every chunk a number $x_i$. The Laplacian measures how much those numbers *disagree
across springs*:

$$
x^{\top} L_{sym}\, x = \sum_{(i,j) \in E} W_{ij} \left( \frac{x_i}{\sqrt{d_i}} - \frac{x_j}{\sqrt{d_j}} \right)^2
$$

Each term is **stiffness × stretch²**. Giving very different numbers to chunks joined by a stiff
spring is expensive; giving different numbers across a weak spring is cheap. So the cheapest
assignments keep strongly linked chunks equal and put all the disagreement on weak links, and
weak links are exactly the boundaries between communities.

## Eigenvectors are the cheapest patterns

The cheapest assignments are the eigenvectors of $L_{sym}$, in order of their eigenvalues
(paper Eq. 12):

$$
L_{sym}\, u_k = \lambda_k u_k, \qquad 0 = \lambda_1 \le \lambda_2 \le \dots \le \lambda_n \le 2
$$

Each eigenvalue is the "spring energy" of its pattern. The first pattern is trivial: it moves
every chunk in proportion to $\sqrt{d_i}$, nothing stretches, and $\lambda_1 = 0$. The second,
$u_2$ (the *Fiedler vector*), is the cheapest way to make chunks disagree at all: the relaxed
solution of splitting the graph in two while cutting as little stiffness as possible.

In the example, $\lambda_2 = 0.066$, and about four fifths of that energy sits on the single
c4–c6 spring: $u_2$ keeps each group internally calm and does its stretching on the bridge.
The next eigenvalue is $\lambda_3 \approx 0.99$, a big jump.

!!! info

    **The eigengap rule of thumb:** the number of eigenvalues close to zero before a big jump is
    roughly the number of well-separated communities. Here $\lambda_1, \lambda_2 \approx 0$, then
    $\lambda_3 \approx 1$: two communities.

## Spectral coordinates

Stack the first $m$ non-trivial eigenvectors as columns, and read the table row by row. Each
chunk's row is its *spectral coordinate* (paper Eq. 13):

$$
z_i = \left[\, u_2(i),\ u_3(i),\ \dots,\ u_{m+1}(i) \,\right] \in \mathbb{R}^m
$$

SpectraFuse keeps $m$ = `spectral_dims` = 32 coordinates per chunk. Strongly linked chunks get
similar numbers in every cheap pattern, so they land close together: communities become
clusters of points. SpectraFuse scales each row to length 1 and runs k-means to label the
communities (`n_clusters`, by default about $\sqrt{n/2}$).

The coordinates come from the graph, not the text. *"The research is done by Helix Labs"* and
*"Helix Labs is headquartered in Pune"* have little text in common, but they can sit close
together because the graph ties them.

## The same matrix spreads relevance

Put relevance on the seed chunks and let it diffuse like heat for time $t$ (paper Eq. 17):

$$
s(t) = e^{-t L_{sym}}\, s(0) = \sum_k e^{-t \lambda_k} \left(u_k^{\top} s(0)\right) u_k
$$

Each pattern fades at its own rate $e^{-t\lambda_k}$. Expensive patterns, which tell chunks
inside a group apart, vanish almost at once, so relevance evens out within a group. Cheap
patterns such as $u_2$ fade slowly, so the difference *between* groups persists.

Exact diffusion needs every eigenvector, which is too expensive for a large corpus. The paper
approximates it with Personalized PageRank over the candidate subgraph (Eq. 18):

$$
s^{(r+1)} = \beta\, s(0) + (1 - \beta)\, \tilde{A}\, s^{(r)}, \qquad \tilde{A} = D^{-1/2} W D^{-1/2} = I - L_{sym}
$$

## How SpectraFuse uses it

**Closeness during re-scoring.** With the default `proximity_mode = "hybrid"`, a candidate's
proximity is the mean of its PageRank score (scaled to 0…1) and its spectral closeness to the
seeds. Each coordinate is first damped by $e^{-t\lambda}$, so smooth, community-level
directions count most:

$$
\tilde{z}_c = z_c \odot e^{-t\lambda}, \qquad
\text{spectral}(c) = \tfrac{1}{2}\left(1 + \cos\!\left(\tilde{z}_c,\ \textstyle\sum_{s \in S_0} \rho_s \tilde{z}_s\right)\right)
$$

where $\rho_s$ is each seed's share of the relevance and $t$ = `diffusion_t` (1.0). Then
$\text{score} = (1-\alpha)\,\text{relevance} + \alpha\,\text{proximity}$, with $\alpha$ = 0.3.

**Community coverage during selection.** A chunk from a spectral cluster that is not yet in the
context earns a bonus:

$$
\text{ClusterGain}(c \mid C) = \frac{1}{1 + \left|\{\, j \in C : \text{cluster}(j) = \text{cluster}(c) \,\}\right|}
$$

This is what lets a chunk from a new community beat a near-duplicate of a chunk already chosen.

**Other proximity modes.** `"ppr"` and `"spectral"` use one of the two signals; `"heat"`
evaluates Eq. 17 from the stored eigenpairs; `"none"` switches the graph signal off. See
[Configuration](../guides/configuration.md#retrieval).

## In the code

| What | Where |
|---|---|
| $L_{sym}$ and the dense eigensolver (`numpy.linalg.eigh`) | `spectrafuse/utils/spectral.py`: `normalized_laplacian()`, `spectral_coords()` |
| Sparse eigensolver above 2,000 chunks (`scipy`, the `scale` extra) | `spectral.py`: `sparse_spectral_coords()` |
| Clusters | `spectrafuse/utils/clusters.py`: `cluster_coords()` |
| PageRank, spectral closeness, heat kernel | `spectrafuse/utils/diffusion.py` |

`tools/validate_release.py` checks these on a real 10,335-chunk index: the stored eigenpairs
satisfy $L u = \lambda u$ to $5.3 \times 10^{-15}$, and the PageRank iteration matches its
closed form to $1.7 \times 10^{-17}$
([production-readiness report](https://github.com/aion7891/SpectraFuse/blob/main/docs/PRODUCTION_READINESS.md)).
