from __future__ import annotations

import logging
import math
import os
import time
from dataclasses import dataclass

logger = logging.getLogger(__name__)

import torch
import torch.nn.functional as F

from perturbnet.image_io import decode_image_b64, encode_image_b64
from perturbnet.model import logits_for_images, predict_index

DEFAULT_EPSILON_LADDER: tuple[float, ...] = (
    0.0035,
    0.0045,
    0.006,
    0.008,
    0.012,
    0.018,
    0.025,
    0.03,
)

DEFAULT_MIN_SSIM = 0.98
DEFAULT_MIN_PSNR_DB = 38.0
DEFAULT_MAX_LINF_DELTA = 0.03
DEFAULT_LINF_WEIGHT = 0.7
DEFAULT_RMSE_WEIGHT = 0.3


@dataclass
class AttackCandidate:
    adv: torch.Tensor
    pred_index: int
    linf: float
    rmse: float
    ssim: float
    psnr_db: float
    perturbation_score: float


def linf_norm(clean: torch.Tensor, adv: torch.Tensor) -> float:
    return float((adv - clean).abs().max().item())


def rmse(clean: torch.Tensor, adv: torch.Tensor) -> float:
    return float(torch.sqrt(torch.mean((adv - clean) ** 2)).item())


def project_linf(clean: torch.Tensor, adv: torch.Tensor, epsilon: float) -> torch.Tensor:
    return torch.max(torch.min(adv, clean + epsilon), clean - epsilon).clamp(0.0, 1.0)


def compute_ssim(x_clean: torch.Tensor, x_adv: torch.Tensor, kernel_size: int = 11) -> float:
    if x_clean.ndim != 3 or x_adv.ndim != 3 or x_clean.shape != x_adv.shape:
        return 0.0
    padding = kernel_size // 2
    x = x_clean.unsqueeze(0)
    y = x_adv.unsqueeze(0)
    c1 = 0.01**2
    c2 = 0.03**2
    mu_x = F.avg_pool2d(x, kernel_size=kernel_size, stride=1, padding=padding)
    mu_y = F.avg_pool2d(y, kernel_size=kernel_size, stride=1, padding=padding)
    sigma_x = F.avg_pool2d(x * x, kernel_size=kernel_size, stride=1, padding=padding) - mu_x * mu_x
    sigma_y = F.avg_pool2d(y * y, kernel_size=kernel_size, stride=1, padding=padding) - mu_y * mu_y
    sigma_xy = F.avg_pool2d(x * y, kernel_size=kernel_size, stride=1, padding=padding) - mu_x * mu_y
    numerator = (2.0 * mu_x * mu_y + c1) * (2.0 * sigma_xy + c2)
    denominator = (mu_x * mu_x + mu_y * mu_y + c1) * (sigma_x + sigma_y + c2)
    ssim_map = numerator / (denominator + 1e-12)
    return float(ssim_map.mean().item())


def compute_psnr_db(x_clean: torch.Tensor, x_adv: torch.Tensor) -> float:
    mse = float(torch.mean((x_adv - x_clean) ** 2).item())
    if mse <= 1e-12:
        return 99.0
    return 10.0 * math.log10(1.0 / mse)


def _early_exit_score_threshold() -> float:
    return float(os.getenv("PERTURB_MINER_EARLY_EXIT_SCORE", "0.905"))


def early_exit_score_threshold() -> float:
    return _early_exit_score_threshold()


def _hard_retry_budget_ms() -> float:
    return float(os.getenv("PERTURB_MINER_HARD_RETRY_BUDGET_MS", "10000"))


def _png_roundtrip(adv: torch.Tensor, device: torch.device) -> torch.Tensor:
    return decode_image_b64(encode_image_b64(adv)).to(device)


def _apgd_steps(mode: str) -> int:
    if mode == "refine":
        return int(os.getenv("PERTURB_MINER_APGD_REFINE_STEPS", "40"))
    return int(os.getenv("PERTURB_MINER_APGD_STEPS", "24"))


def _apgd_restarts() -> int:
    return int(os.getenv("PERTURB_MINER_APGD_RESTARTS", "1"))


def targeted_apgd_linf(
    model: torch.nn.Module,
    clean: torch.Tensor,
    source_index: int,
    target_index: int | None,
    epsilon: float,
    steps: int,
    restarts: int,
    targeted: bool,
) -> torch.Tensor:
    device = clean.device
    best_adv = clean.clone()
    best_success: torch.Tensor | None = None
    best_norm = float("inf")

    for restart in range(restarts):
        if restart == 0:
            adv = clean.clone()
        else:
            noise = torch.empty_like(clean).uniform_(-epsilon, epsilon)
            adv = project_linf(clean, clean + noise, epsilon)

        step_size = max(epsilon, 2.0 * epsilon)
        momentum = torch.zeros_like(clean)
        loss_prev = float("inf")
        checkpoint = max(steps // 4, 1)

        for step in range(steps):
            adv = adv.detach()
            adv.requires_grad_(True)
            logits = logits_for_images(model=model, image_bchw=adv.unsqueeze(0))

            if targeted and target_index is not None:
                target = torch.tensor([target_index], device=device)
                loss = F.cross_entropy(logits, target)
                grad = torch.autograd.grad(loss, adv)[0]
            else:
                source = torch.tensor([source_index], device=device)
                loss = F.cross_entropy(logits, source)
                grad = torch.autograd.grad(loss, adv)[0]

            grad_norm = grad.abs().mean().clamp(min=1e-12)
            momentum = 0.75 * momentum + grad / grad_norm
            if targeted:
                adv = project_linf(clean, adv.detach() - step_size * momentum.sign(), epsilon)
            else:
                adv = project_linf(clean, adv.detach() + step_size * momentum.sign(), epsilon)

            if step > 0 and step % checkpoint == 0:
                current_loss = float(loss.detach().item())
                if current_loss > 0.75 * loss_prev:
                    step_size *= 0.5
                loss_prev = current_loss

        adv = adv.detach()
        pred = predict_index(model=model, image_chw=adv)
        norm = linf_norm(clean, adv)
        if pred != source_index and norm >= 1e-12:
            if norm < best_norm:
                best_success = adv.clone()
                best_norm = norm
        elif best_success is None:
            best_adv = adv.clone()

    return best_success if best_success is not None else best_adv


def binary_search_min_linf(
    model: torch.nn.Module,
    clean: torch.Tensor,
    adv: torch.Tensor,
    source_index: int,
    min_delta: float,
    epsilon: float,
    iterations: int = 14,
) -> torch.Tensor:
    delta = adv - clean
    max_norm = linf_norm(clean, adv)
    if max_norm <= min_delta:
        return adv

    lo = min_delta / max(max_norm, 1e-12)
    hi = min(1.0, epsilon / max(max_norm, 1e-12))
    best = adv

    for _ in range(iterations):
        mid = (lo + hi) / 2.0
        candidate = (clean + mid * delta).clamp(0.0, 1.0)
        norm = linf_norm(clean, candidate)
        if norm > epsilon:
            hi = mid
            continue
        pred = predict_index(model=model, image_chw=candidate)
        if pred != source_index and norm >= min_delta:
            best = candidate
            hi = mid
        else:
            lo = mid

    return best


def _probe_target_flip_rmse(
    model: torch.nn.Module,
    clean: torch.Tensor,
    source_index: int,
    target_index: int,
    epsilon: float,
    min_delta: float,
) -> float:
    """Quick APGD probe: return RMSE if flip succeeds, else infinity."""
    adv = targeted_apgd_linf(
        model=model,
        clean=clean,
        source_index=source_index,
        target_index=target_index,
        epsilon=epsilon,
        steps=max(4, int(os.getenv("PERTURB_MINER_PROBE_STEPS", "4"))),
        restarts=1,
        targeted=True,
    )
    if predict_index(model=model, image_chw=adv) == source_index:
        return float("inf")
    adv = binary_search_min_linf(
        model=model,
        clean=clean,
        adv=adv,
        source_index=source_index,
        min_delta=min_delta,
        epsilon=epsilon,
        iterations=6,
    )
    if predict_index(model=model, image_chw=adv) == source_index:
        return float("inf")
    return rmse(clean, adv)


def _rank_by_logit_margin(
    model: torch.nn.Module,
    clean: torch.Tensor,
    source_index: int,
    k: int,
) -> list[int]:
    with torch.no_grad():
        logits = logits_for_images(model=model, image_bchw=clean.unsqueeze(0))[0]
    source_logit = float(logits[source_index].item())
    margins: list[tuple[int, float]] = []
    for idx in range(int(logits.shape[0])):
        if idx == source_index:
            continue
        margins.append((idx, float(logits[idx].item()) - source_logit))
    margins.sort(key=lambda item: item[1], reverse=True)
    limit = max(1, k)
    return [idx for idx, _ in margins[:limit]]


def ranked_attack_targets(
    model: torch.nn.Module,
    clean: torch.Tensor,
    source_index: int,
    k: int = 5,
    epsilon: float = DEFAULT_MAX_LINF_DELTA,
    min_delta: float = 0.003,
    use_rmse_probe: bool = False,
) -> list[int]:
    """Rank targets by logit margin; optional RMSE probe when fast pass fails."""
    if not use_rmse_probe:
        return _rank_by_logit_margin(model=model, clean=clean, source_index=source_index, k=k)

    margins = _rank_by_logit_margin(model=model, clean=clean, source_index=source_index, k=max(k, 16))
    margin_map = {idx: rank for rank, idx in enumerate(margins)}

    probe_k = max(k, int(os.getenv("PERTURB_MINER_PROBE_K", "5")))
    probe_candidates = margins[:probe_k]
    scored: list[tuple[int, float, int]] = []
    for idx in probe_candidates:
        probe_rmse = _probe_target_flip_rmse(
            model=model,
            clean=clean,
            source_index=source_index,
            target_index=idx,
            epsilon=epsilon,
            min_delta=min_delta,
        )
        scored.append((idx, probe_rmse, margin_map.get(idx, 999)))

    scored.sort(key=lambda item: (item[1], item[2]))
    limit = max(1, k)
    return [idx for idx, _, _ in scored[:limit]]


def refine_high_gradient_mask(
    model: torch.nn.Module,
    clean: torch.Tensor,
    adv: torch.Tensor,
    source_index: int,
    target_index: int | None,
    epsilon: float,
    min_delta: float,
    effective_max_delta: float,
    bs_iterations: int = 8,
) -> torch.Tensor:
    device = clean.device
    delta = adv - clean
    best = adv
    best_score = -1.0

    adv_tmp = adv.detach()
    adv_tmp.requires_grad_(True)
    logits = logits_for_images(model=model, image_bchw=adv_tmp.unsqueeze(0))
    if target_index is not None:
        loss = F.cross_entropy(logits, torch.tensor([target_index], device=device))
    else:
        loss = F.cross_entropy(logits, torch.tensor([source_index], device=device))
    grad = torch.autograd.grad(loss, adv_tmp)[0]
    pixel_importance = grad.abs().amax(dim=0)

    keep_ratios = (1.0, 0.85, 0.7, 0.55, 0.45, 0.35, 0.25)
    for keep_ratio in keep_ratios:
        if keep_ratio >= 1.0:
            masked = delta
        else:
            flat = pixel_importance.reshape(-1)
            k = max(1, int(flat.numel() * keep_ratio))
            threshold = torch.topk(flat, k=k, largest=True).values.min()
            pixel_mask = (pixel_importance >= threshold).float()
            masked = delta * pixel_mask

        candidate_adv = project_linf(clean, clean + masked, epsilon)
        candidate_adv = binary_search_min_linf(
            model=model,
            clean=clean,
            adv=candidate_adv,
            source_index=source_index,
            min_delta=min_delta,
            epsilon=epsilon,
            iterations=bs_iterations,
        )
        scored = evaluate_candidate(
            clean=clean,
            adv=candidate_adv,
            source_index=source_index,
            min_delta=min_delta,
            effective_max_delta=effective_max_delta,
            model=model,
            use_png=True,
        )
        if scored is not None and scored.perturbation_score > best_score:
            best = scored.adv
            best_score = scored.perturbation_score

    return best


def evaluate_candidate(
    clean: torch.Tensor,
    adv: torch.Tensor,
    source_index: int,
    min_delta: float,
    effective_max_delta: float,
    min_ssim: float = DEFAULT_MIN_SSIM,
    min_psnr_db: float = DEFAULT_MIN_PSNR_DB,
    model: torch.nn.Module | None = None,
    use_png: bool = True,
) -> AttackCandidate | None:
    scored_adv = adv
    if use_png and model is not None:
        scored_adv = _png_roundtrip(adv, clean.device)

    norm = linf_norm(clean, scored_adv)
    if norm < min_delta or norm > effective_max_delta:
        return None

    if model is not None:
        pred_index = predict_index(model=model, image_chw=scored_adv)
    else:
        pred_index = -1

    if pred_index == source_index:
        return None

    ssim = compute_ssim(clean, scored_adv)
    if ssim < min_ssim:
        return None

    psnr_db = compute_psnr_db(clean, scored_adv)
    if psnr_db < min_psnr_db:
        return None

    adv_rmse = rmse(clean, scored_adv)
    denom = max(1e-12, effective_max_delta - min_delta)
    linf_ratio = min(max((norm - min_delta) / denom, 0.0), 1.0)
    linf_score = (1.0 - linf_ratio) ** 2
    rmse_ratio = min(max(adv_rmse / max(1e-12, effective_max_delta), 0.0), 1.0)
    rmse_score = (1.0 - rmse_ratio) ** 2
    perturbation_score = (
        DEFAULT_LINF_WEIGHT * linf_score + DEFAULT_RMSE_WEIGHT * rmse_score
    ) / (DEFAULT_LINF_WEIGHT + DEFAULT_RMSE_WEIGHT)

    return AttackCandidate(
        adv=scored_adv,
        pred_index=pred_index,
        linf=norm,
        rmse=adv_rmse,
        ssim=ssim,
        psnr_db=psnr_db,
        perturbation_score=float(perturbation_score),
    )


def _epsilon_ladder(challenge_epsilon: float, min_delta: float, max_linf_delta: float) -> list[float]:
    cap = min(challenge_epsilon, max_linf_delta)
    ladder = [e for e in DEFAULT_EPSILON_LADDER if min_delta <= e <= cap + 1e-12]
    if cap not in ladder and cap >= min_delta:
        ladder.append(cap)
    ladder = sorted(set(ladder))
    return ladder or [cap]


def _hard_retry_thresholds() -> tuple[float, float]:
    score_floor = float(os.getenv("PERTURB_MINER_HARD_RETRY_SCORE", "0.895"))
    rmse_ceiling = float(os.getenv("PERTURB_MINER_HARD_RETRY_RMSE", "0.003"))
    return score_floor, rmse_ceiling


def _collect_attack_candidates(
    model: torch.nn.Module,
    clean: torch.Tensor,
    source_index: int,
    target_indices: list[int],
    ladder: list[float],
    min_delta: float,
    effective_max: float,
    steps: int,
    restarts: int,
    bs_iterations: int,
    early_exit_score: float,
) -> tuple[list[AttackCandidate], AttackCandidate | None]:
    candidates: list[AttackCandidate] = []
    search_plans: list[tuple[int | None, bool]] = [(idx, True) for idx in target_indices]
    search_plans.append((None, False))

    for target_index, targeted in search_plans:
        logger.info(f"Searching for target={target_index} targeted={targeted}")
        for epsilon in ladder:
            logger.info(f"Searching for epsilon={epsilon}")
            adv = targeted_apgd_linf(
                model=model,
                clean=clean,
                source_index=source_index,
                target_index=target_index,
                epsilon=epsilon,
                steps=steps,
                restarts=restarts,
                targeted=targeted,
            )
            pred = predict_index(model=model, image_chw=adv)
            if pred == source_index:
                logger.info(f"Skipping epsilon={epsilon} because pred={pred} == source_index={source_index}")
                continue

            adv = binary_search_min_linf(
                model=model,
                clean=clean,
                adv=adv,
                source_index=source_index,
                min_delta=min_delta,
                epsilon=epsilon,
                iterations=bs_iterations,
            )
            logger.info(f"Binary search min Linf for epsilon={epsilon} found adv with norm={linf_norm(clean, adv)}")
            adv = refine_high_gradient_mask(
                model=model,
                clean=clean,
                adv=adv,
                source_index=source_index,
                target_index=target_index,
                epsilon=epsilon,
                min_delta=min_delta,
                effective_max_delta=effective_max,
                bs_iterations=max(8, bs_iterations // 2),
            )

            candidate = evaluate_candidate(
                clean=clean,
                adv=adv,
                source_index=source_index,
                min_delta=min_delta,
                effective_max_delta=effective_max,
                model=model,
                use_png=True,
            )
            if candidate is not None:
                logger.info(f"Evaluated candidate with perturbation_score={candidate.perturbation_score}")
                candidates.append(candidate)
                if candidate.perturbation_score >= early_exit_score:
                    logger.info(f"Early exit with perturbation_score={candidate.perturbation_score}")
                    return candidates, candidate
                break
    return candidates, None


def run_quality_linf_attack(
    model: torch.nn.Module,
    clean: torch.Tensor,
    source_index: int,
    challenge_epsilon: float,
    min_delta: float,
    max_linf_delta: float = DEFAULT_MAX_LINF_DELTA,
) -> tuple[torch.Tensor, int, float, float]:
    attack_started = time.perf_counter()
    effective_max = min(challenge_epsilon, max_linf_delta)
    ladder = _epsilon_ladder(challenge_epsilon, min_delta, max_linf_delta)
    target_k = int(os.getenv("PERTURB_MINER_TOP_K", "5"))
    target_count = int(os.getenv("PERTURB_MINER_TARGET_COUNT", "3"))
    early_exit_score = _early_exit_score_threshold()

    steps = _apgd_steps("search")
    restarts = _apgd_restarts()
    bs_iterations = int(os.getenv("PERTURB_MINER_BS_ITERATIONS", "14"))

    # Fast pass: logit-margin targets only (no RMSE probes).
    margin_targets = ranked_attack_targets(
        model=model,
        clean=clean,
        source_index=source_index,
        k=target_k,
        epsilon=effective_max,
        min_delta=min_delta,
        use_rmse_probe=False,
    )[: max(1, target_count)]

    candidates, early = _collect_attack_candidates(
        model=model,
        clean=clean,
        source_index=source_index,
        target_indices=margin_targets,
        ladder=ladder,
        min_delta=min_delta,
        effective_max=effective_max,
        steps=steps,
        restarts=restarts,
        bs_iterations=bs_iterations,
        early_exit_score=early_exit_score,
    )
    if early is not None:
        return early.adv, early.pred_index, early.linf, early.perturbation_score

    # Fallback: RMSE-probed targets when fast pass found nothing.
    if not candidates:
        logger.info("Fast pass found no candidates; ranking targets with RMSE probe")
        probe_targets = ranked_attack_targets(
            model=model,
            clean=clean,
            source_index=source_index,
            k=target_k,
            epsilon=effective_max,
            min_delta=min_delta,
            use_rmse_probe=True,
        )[: max(1, target_count)]
        probe_candidates, probe_early = _collect_attack_candidates(
            model=model,
            clean=clean,
            source_index=source_index,
            target_indices=probe_targets,
            ladder=ladder,
            min_delta=min_delta,
            effective_max=effective_max,
            steps=steps,
            restarts=restarts,
            bs_iterations=bs_iterations,
            early_exit_score=early_exit_score,
        )
        if probe_early is not None:
            return probe_early.adv, probe_early.pred_index, probe_early.linf, probe_early.perturbation_score
        candidates.extend(probe_candidates)

    logger.info(f"Found {len(candidates)} candidates")
    if not candidates:
        logger.info(f"No candidates found, using APGD-Linf with epsilon={effective_max}")
        adv = targeted_apgd_linf(
            model=model,
            clean=clean,
            source_index=source_index,
            target_index=None,
            epsilon=effective_max,
            steps=_apgd_steps("refine"),
            restarts=max(restarts, 2),
            targeted=False,
        )
        adv = project_linf(clean, adv, effective_max)
        pred = predict_index(model=model, image_chw=adv)
        logger.info(f"No candidates found, using APGD-Linf with epsilon={effective_max}")
        return adv, pred, linf_norm(clean, adv), 0.0

    best = max(
        candidates,
        key=lambda c: (c.perturbation_score, -c.linf, -c.rmse),
    )

    score_floor, rmse_ceiling = _hard_retry_thresholds()
    elapsed_ms = (time.perf_counter() - attack_started) * 1000.0
    retry_budget_ms = _hard_retry_budget_ms()
    needs_hard_retry = best.perturbation_score < score_floor or best.rmse > rmse_ceiling

    if needs_hard_retry and elapsed_ms >= retry_budget_ms:
        logger.info(
            f"Skipping hard-image retry: elapsed_ms={elapsed_ms:.0f} >= budget_ms={retry_budget_ms:.0f}"
        )
    elif needs_hard_retry:
        logger.info(
            f"Hard-image retry: score={best.perturbation_score:.6f} rmse={best.rmse:.6f} "
            f"elapsed_ms={elapsed_ms:.0f} thresholds score<{score_floor} rmse>{rmse_ceiling}"
        )
        retry_targets = ranked_attack_targets(
            model=model,
            clean=clean,
            source_index=source_index,
            k=max(target_k, 8),
            epsilon=effective_max,
            min_delta=min_delta,
            use_rmse_probe=True,
        )[: max(1, target_count + 2)]
        retry_steps = int(os.getenv("PERTURB_MINER_HARD_RETRY_APGD_STEPS", "60"))
        retry_restarts = max(restarts, int(os.getenv("PERTURB_MINER_HARD_RETRY_RESTARTS", "2")))
        retry_bs = int(os.getenv("PERTURB_MINER_HARD_RETRY_BS_ITERATIONS", "20"))
        retry_candidates, retry_early = _collect_attack_candidates(
            model=model,
            clean=clean,
            source_index=source_index,
            target_indices=retry_targets,
            ladder=ladder,
            min_delta=min_delta,
            effective_max=effective_max,
            steps=retry_steps,
            restarts=retry_restarts,
            bs_iterations=retry_bs,
            early_exit_score=early_exit_score,
        )
        if retry_early is not None:
            return retry_early.adv, retry_early.pred_index, retry_early.linf, retry_early.perturbation_score
        if retry_candidates:
            retry_best = max(
                retry_candidates,
                key=lambda c: (c.perturbation_score, -c.linf, -c.rmse),
            )
            if retry_best.perturbation_score > best.perturbation_score:
                best = retry_best

    return best.adv, best.pred_index, best.linf, best.perturbation_score


def apply_png_safe_shrink(
    model: torch.nn.Module,
    clean: torch.Tensor,
    adv: torch.Tensor,
    source_index: int,
    epsilon: float,
    min_delta: float,
    effective_max_delta: float | None = None,
    target_index: int | None = None,
    skip_refine: bool = False,
) -> tuple[torch.Tensor, int, float]:
    """Final safety pass on PNG-decoded tensor with optional RMSE refinement."""
    effective_max = effective_max_delta if effective_max_delta is not None else epsilon
    adv = _png_roundtrip(adv, clean.device)
    safe_epsilon = max(min_delta, epsilon - (2.0 / 255.0))
    current_norm = linf_norm(clean, adv)
    if current_norm > safe_epsilon:
        scale = safe_epsilon / max(current_norm, 1e-12)
        shrunk = (clean + scale * (adv - clean)).clamp(0.0, 1.0)
        shrunk = _png_roundtrip(shrunk, clean.device)
        if predict_index(model=model, image_chw=shrunk) != source_index:
            adv = shrunk

    if not skip_refine:
        flip_target = target_index
        if flip_target is None:
            pred = predict_index(model=model, image_chw=adv)
            if pred != source_index:
                flip_target = pred

        if flip_target is not None and flip_target != source_index:
            adv = refine_high_gradient_mask(
                model=model,
                clean=clean,
                adv=adv,
                source_index=source_index,
                target_index=flip_target,
                epsilon=safe_epsilon,
                min_delta=min_delta,
                effective_max_delta=min(effective_max, safe_epsilon),
                bs_iterations=6,
            )
    pred = predict_index(model=model, image_chw=adv)
    return adv, pred, linf_norm(clean, adv)
