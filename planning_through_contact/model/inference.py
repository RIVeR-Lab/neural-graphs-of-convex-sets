from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import numpy as np
import torch
from torch import Tensor


@dataclass(frozen=True)
class RoundingResult:
    edge_indices: list[int]
    node_indices: list[int]
    success: bool


def project_flows_qp(
    *,
    edge_index: Tensor,
    phi_hat: Tensor,
    num_nodes: int,
    source_idx: int,
    target_idx: int,
) -> Tensor:
    """
    Solve the projection QP (Eq. 39–41):

        (39)  ϕ_proj = argmin_{ϕ∈R^{|E|}}  1/2 ||ϕ - ϕ̂||_2^2
              s.t.  Bϕ = b,   0 ≤ ϕ ≤ 1

        (40)  B_{v,e} = +1 if edge e leaves v
                      = -1 if edge e enters v
                      =  0 otherwise

        (41)  b_v = +1 (v = s),  -1 (v = t),  0 otherwise

    using Drake.
    """
    try:
        from pydrake.solvers import MathematicalProgram, Solve  # type: ignore
    except ModuleNotFoundError as e:  # pragma: no cover
        raise ModuleNotFoundError(
            "Drake (pydrake) is required for QP projection but is not available."
        ) from e

    edge_index = edge_index.detach().cpu()
    phi_hat = phi_hat.detach().cpu().float().view(-1)
    E = int(edge_index.size(1))

    src = edge_index[0].numpy().astype(int)
    dst = edge_index[1].numpy().astype(int)

    # Build node-edge incidence matrix B (Eq. 40).
    B = np.zeros((num_nodes, E), dtype=np.float64)
    for e in range(E):
        B[src[e], e] += 1.0
        B[dst[e], e] -= 1.0

    # Build RHS b enforcing unit flow from source s to target t (Eq. 41).
    b = np.zeros((num_nodes,), dtype=np.float64)
    b[source_idx] = 1.0
    b[target_idx] = -1.0

    prog = MathematicalProgram()
    phi = prog.NewContinuousVariables(E, "phi")
    # Constraints in (39): 0 <= phi <= 1 and B phi = b.
    prog.AddBoundingBoxConstraint(0.0, 1.0, phi)
    prog.AddLinearEqualityConstraint(B, b, phi)
    # Objective in (39): (1/2) ||phi - phi_hat||^2.
    prog.AddQuadraticErrorCost(np.eye(E), phi_hat.numpy(), phi)

    res = Solve(prog)
    if not res.is_success():  # pragma: no cover
        raise RuntimeError("QP projection failed to solve successfully.")

    phi_proj = torch.tensor(res.GetSolution(phi), dtype=torch.float32)
    return phi_proj


def randomized_rounding(
    *,
    edge_index: Tensor,
    phi: Tensor,
    source_idx: int,
    target_idx: int,
    max_steps: int = 512,
    generator: Optional[torch.Generator] = None,
) -> RoundingResult:
    """
    Randomized rounding described in Sec. I-G.3:
    sample an outgoing edge at each node proportional to its flow.
    """
    # Sec. I-G.3: interpret φ as a distribution over outgoing edges at each node and sample until target.
    edge_index = edge_index.detach().cpu()
    phi = phi.detach().cpu().float().view(-1)
    E = int(edge_index.size(1))
    N = int(max(edge_index.max().item() + 1, source_idx + 1, target_idx + 1))

    src = edge_index[0]
    dst = edge_index[1]

    out_edges: list[list[int]] = [[] for _ in range(N)]
    for e in range(E):
        out_edges[int(src[e].item())].append(e)

    node_path = [int(source_idx)]
    edge_path: list[int] = []

    cur = int(source_idx)
    for _ in range(int(max_steps)):
        if cur == int(target_idx):
            return RoundingResult(edge_indices=edge_path, node_indices=node_path, success=True)
        candidates = out_edges[cur]
        if len(candidates) == 0:
            break
        probs = phi[candidates].clamp(min=0.0)
        s = float(probs.sum().item())
        if s <= 0.0:
            break
        probs = probs / s
        choice_local = int(torch.multinomial(probs, num_samples=1, generator=generator).item())
        e = int(candidates[choice_local])
        nxt = int(dst[e].item())
        edge_path.append(e)
        node_path.append(nxt)
        cur = nxt

    return RoundingResult(edge_indices=edge_path, node_indices=node_path, success=False)


def best_of_n_rounding(
    *,
    edge_index: Tensor,
    phi: Tensor,
    source_idx: int,
    target_idx: int,
    n: int = 64,
    max_steps: int = 512,
    edge_costs: Optional[Tensor] = None,
    generator: Optional[torch.Generator] = None,
) -> RoundingResult:
    """
    Sample N candidate paths and pick the minimum-cost one (Sec. I-G.4).
    If `edge_costs` is None, uses path length as a proxy cost.
    """
    best: Optional[RoundingResult] = None
    best_cost: float = float("inf")

    edge_costs_t = edge_costs.detach().cpu().float().view(-1) if edge_costs is not None else None

    for _ in range(int(n)):
        res = randomized_rounding(
            edge_index=edge_index,
            phi=phi,
            source_idx=source_idx,
            target_idx=target_idx,
            max_steps=max_steps,
            generator=generator,
        )
        if not res.success:
            continue

        if edge_costs_t is None:
            cost = float(len(res.edge_indices))
        else:
            idx = torch.tensor(res.edge_indices, dtype=torch.long)
            cost = float(edge_costs_t[idx].sum().item())

        if cost < best_cost:
            best_cost = cost
            best = res

    if best is None:
        return RoundingResult(edge_indices=[], node_indices=[int(source_idx)], success=False)
    return best

