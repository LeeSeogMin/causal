#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
제12장: 복잡계 이론과 정책 시뮬레이션
실습 예제(합성 데이터): 서울시 올빼미버스와 창발적 수요

목적
----
- 실제 서울시 데이터(통신/택시)를 재현하는 것이 아니라,
  "개별 이동(미시) → OD 집계(거시) → 노선 후보 도출"의 흐름을
  작게 실행해보는 교육용 예제이다.

핵심 아이디어
-------------
1) 심야 시간대(00:00~04:00) 이동을 'OD(출발-도착) 수요'로 합성 생성한다.
2) 시간대별 OD를 그래프로 보고, '갑작스런 수요 버스트(burst)'를 탐지한다.
3) 수요가 큰 거점들을 간단한 휴리스틱으로 연결해 노선 후보를 제안한다.

주의
----
- 결과 해석은 "원리 이해"에 초점이 있다.
  (실무 노선 설계에는 도로망/정류장/승하차 제약/운행시간/운영비 등 추가 고려 필요)
"""

from __future__ import annotations

import argparse
import os
from dataclasses import dataclass
from datetime import datetime, timedelta
from typing import Dict, Iterable, List, Tuple

import numpy as np
import pandas as pd
import networkx as nx
import matplotlib.pyplot as plt
from matplotlib import font_manager


def configure_korean_font() -> None:
    """
    환경에 따라 한글 폰트를 최대한 안전하게 설정한다.

    - Windows: Malgun Gothic
    - macOS: AppleGothic
    - Linux: NanumGothic(있으면) → DejaVu Sans
    """

    preferred = ["Malgun Gothic", "AppleGothic", "NanumGothic", "DejaVu Sans"]
    available = {f.name for f in font_manager.fontManager.ttflist}

    for name in preferred:
        if name in available:
            plt.rcParams["font.family"] = name
            break

    plt.rcParams["axes.unicode_minus"] = False
    plt.rcParams["font.size"] = 10


@dataclass(frozen=True)
class Zone:
    name: str
    xy: Tuple[float, float]
    zone_type: str  # "hub" | "residential" | "mixed"


def build_seoul_zones() -> List[Zone]:
    """
    서울의 대표 거점을 단순화한 2D 좌표(가상)로 정의한다.
    좌표는 지도 정확도가 아니라 시각화/거리계산을 위한 상대값이다.
    """

    return [
        Zone("홍대", (0.18, 0.55), "hub"),
        Zone("강남", (0.78, 0.35), "hub"),
        Zone("종로", (0.48, 0.62), "hub"),
        Zone("동대문", (0.62, 0.62), "hub"),
        Zone("건대", (0.72, 0.55), "hub"),
        Zone("서울역", (0.38, 0.52), "mixed"),
        Zone("여의도", (0.28, 0.48), "mixed"),
        Zone("신림", (0.45, 0.22), "residential"),
        Zone("잠실", (0.88, 0.45), "residential"),
        Zone("강서", (0.06, 0.52), "residential"),
        Zone("노원", (0.72, 0.85), "residential"),
        Zone("강동", (0.93, 0.62), "residential"),
    ]


def time_bins(start: str = "00:00", end: str = "04:00", freq_minutes: int = 15) -> List[str]:
    start_dt = datetime.strptime(start, "%H:%M")
    end_dt = datetime.strptime(end, "%H:%M")

    labels: List[str] = []
    current = start_dt
    while current < end_dt:
        labels.append(current.strftime("%H:%M"))
        current += timedelta(minutes=freq_minutes)
    return labels


def _euclidean(a: Tuple[float, float], b: Tuple[float, float]) -> float:
    return float(np.hypot(a[0] - b[0], a[1] - b[1]))


def _zone_weights_by_type(zones: List[Zone]) -> Dict[str, float]:
    weights = {}
    for z in zones:
        if z.zone_type == "hub":
            weights[z.name] = 2.8
        elif z.zone_type == "mixed":
            weights[z.name] = 1.6
        else:
            weights[z.name] = 1.0
    return weights


def generate_synthetic_od(
    *,
    zones: List[Zone],
    seed: int,
    base_trips_per_bin: int,
    shock_trips_per_bin: int,
    shock_start: str,
    shock_end: str,
) -> pd.DataFrame:
    """
    심야 OD를 합성 생성한다.

    Returns
    -------
    df : columns = [time_bin, time, origin, destination, base_count, shock_count, count]
    """

    rng = np.random.default_rng(seed)

    labels = time_bins()
    n_bins = len(labels)
    name_to_zone = {z.name: z for z in zones}
    zone_names = [z.name for z in zones]

    # 기본 수요: (허브 → 주거) 중심, 거리감쇠 포함
    base_origin_weight = _zone_weights_by_type(zones)
    dest_weight = {z.name: (2.2 if z.zone_type == "residential" else 1.2) for z in zones}

    # 창발(버스트) 수요: 특정 시간대에 '홍대'에서 귀가 수요 급증(가정)
    shock_bins = set()
    for idx, t in enumerate(labels):
        if shock_start <= t < shock_end:
            shock_bins.add(idx)

    shock_origin = "홍대"
    shock_destinations = ["강서", "노원", "신림", "잠실"]
    shock_p = np.array([0.25, 0.30, 0.25, 0.20], dtype=float)

    def trips_curve(bin_index: int) -> int:
        # 00:00~02:00 구간에 수요가 조금 더 많은 단순 곡선(가상)
        center = 6.0  # 01:30 (15분 bin 기준)
        sigma = 4.0
        bump = 0.55 * np.exp(-((bin_index - center) ** 2) / (2 * sigma**2))
        return int(round(base_trips_per_bin * (0.85 + bump)))

    base_counter: Dict[Tuple[int, str, str], int] = {}
    shock_counter: Dict[Tuple[int, str, str], int] = {}

    coords = {z.name: z.xy for z in zones}

    for b in range(n_bins):
        n_trips = trips_curve(b)

        # origin 분포: 허브(상업지) 비중이 큰 설정
        origin_scores = np.array([base_origin_weight[n] for n in zone_names], dtype=float)
        origin_p = origin_scores / origin_scores.sum()
        origins = rng.choice(zone_names, size=n_trips, replace=True, p=origin_p)

        for origin in origins:
            # destination 분포: 주거지 선호 + 거리감쇠(가상)
            scores = []
            for dest in zone_names:
                if dest == origin:
                    scores.append(0.0)
                    continue
                dist = _euclidean(coords[origin], coords[dest])
                score = dest_weight[dest] * np.exp(-2.2 * dist)
                scores.append(score)
            scores_arr = np.array(scores, dtype=float)
            scores_arr = scores_arr / scores_arr.sum()
            destination = rng.choice(zone_names, p=scores_arr)
            key = (b, origin, destination)
            base_counter[key] = base_counter.get(key, 0) + 1

        if b in shock_bins and shock_trips_per_bin > 0:
            destinations = rng.choice(shock_destinations, size=shock_trips_per_bin, p=shock_p)
            for destination in destinations:
                key = (b, shock_origin, destination)
                shock_counter[key] = shock_counter.get(key, 0) + 1

    base_df = pd.DataFrame(
        [(b, labels[b], o, d, c) for (b, o, d), c in base_counter.items()],
        columns=["time_bin", "time", "origin", "destination", "base_count"],
    )
    shock_df = pd.DataFrame(
        [(b, labels[b], o, d, c) for (b, o, d), c in shock_counter.items()],
        columns=["time_bin", "time", "origin", "destination", "shock_count"],
    )

    if shock_df.empty:
        df = base_df.copy()
        df["shock_count"] = 0
    else:
        df = base_df.merge(
            shock_df[["time_bin", "origin", "destination", "shock_count"]],
            on=["time_bin", "origin", "destination"],
            how="outer",
        ).fillna(0)

    df["base_count"] = df["base_count"].astype(int)
    df["shock_count"] = df["shock_count"].astype(int)
    df["count"] = df["base_count"] + df["shock_count"]

    # time 컬럼 보완(merge로 인해 누락될 수 있음)
    if "time" not in df.columns:
        df["time"] = df["time_bin"].map(lambda b: labels[int(b)])
    else:
        df["time"] = df["time"].fillna(df["time_bin"].map(lambda b: labels[int(b)]))

    return df.sort_values(["time_bin", "count"], ascending=[True, False]).reset_index(drop=True)


def build_flow_graph(df: pd.DataFrame) -> nx.DiGraph:
    """집계 OD를 방향 그래프로 구성한다."""
    G = nx.DiGraph()
    od = df.groupby(["origin", "destination"], as_index=False)["count"].sum()
    for _, row in od.iterrows():
        G.add_edge(row["origin"], row["destination"], weight=int(row["count"]))
    return G


def compute_edge_burst(df: pd.DataFrame) -> pd.DataFrame:
    """
    시간대별로 변동이 큰(버스트가 있는) OD를 간단 점수로 탐지한다.

    burst_score = (max - mean) / (std + 1)
    """

    pivot = df.pivot_table(
        index="time_bin",
        columns=["origin", "destination"],
        values="count",
        aggfunc="sum",
        fill_value=0,
    )

    edges = []
    for (origin, destination) in pivot.columns:
        series = pivot[(origin, destination)].values.astype(float)
        mean = float(series.mean())
        std = float(series.std(ddof=0))
        max_count = float(series.max())
        max_bin = int(series.argmax())
        burst_score = (max_count - mean) / (std + 1.0)
        edges.append(
            {
                "origin": origin,
                "destination": destination,
                "mean": mean,
                "std": std,
                "max": max_count,
                "max_time_bin": max_bin,
                "burst_score": burst_score,
            }
        )

    return pd.DataFrame(edges).sort_values("burst_score", ascending=False).reset_index(drop=True)


def propose_route(
    *,
    zones: List[Zone],
    od: pd.DataFrame,
    n_stops: int,
) -> List[str]:
    """
    수요가 큰 거점을 고르고(총 유입+유출),
    좌표 기반 최근접 탐욕으로 정류장 순서를 만든다(교육용 휴리스틱).
    """

    coords = {z.name: z.xy for z in zones}

    outflow = od.groupby("origin")["count"].sum()
    inflow = od.groupby("destination")["count"].sum()
    total = (outflow.add(inflow, fill_value=0)).sort_values(ascending=False)
    stops = total.head(n_stops).index.tolist()

    # 시작점: 가장 서쪽(작은 x) 정류장으로 고정하면 경로가 덜 뒤엉킨다(가상)
    start = min(stops, key=lambda n: coords[n][0])
    route = [start]
    remaining = set(stops) - {start}

    while remaining:
        last = route[-1]
        next_stop = min(remaining, key=lambda n: _euclidean(coords[last], coords[n]))
        route.append(next_stop)
        remaining.remove(next_stop)

    return route


def plot_network(
    *,
    zones: List[Zone],
    od: pd.DataFrame,
    burst_edges: pd.DataFrame,
    route: List[str],
    top_edges: int,
    save_path: str,
    show: bool,
) -> None:
    coords = {z.name: z.xy for z in zones}
    zone_type = {z.name: z.zone_type for z in zones}

    fig = plt.figure(figsize=(12, 8))
    ax = plt.gca()

    # 노드
    colors = {"hub": "#ff6b6b", "mixed": "#4dabf7", "residential": "#51cf66"}
    for z in zones:
        x, y = z.xy
        ax.scatter(x, y, s=260, c=colors[z.zone_type], edgecolors="black", linewidths=1.2, zorder=3)
        ax.text(x, y + 0.02, z.name, ha="center", va="bottom", fontsize=9, zorder=4)

    # 엣지(상위 top_edges개만)
    od_top = od.sort_values("count", ascending=False).head(top_edges).copy()
    max_w = float(od_top["count"].max()) if len(od_top) else 1.0

    burst_set = {(r.origin, r.destination) for r in burst_edges.head(5).itertuples(index=False)}

    for row in od_top.itertuples(index=False):
        o, d, w = row.origin, row.destination, int(row.count)
        x1, y1 = coords[o]
        x2, y2 = coords[d]
        width = 0.5 + 3.0 * (w / max_w)

        is_burst = (o, d) in burst_set
        edge_color = "#e03131" if is_burst else "#868e96"
        alpha = 0.75 if is_burst else 0.35
        ax.plot([x1, x2], [y1, y2], color=edge_color, linewidth=width, alpha=alpha, zorder=1)

    # 노선 후보(점선)
    for a, b in zip(route, route[1:]):
        x1, y1 = coords[a]
        x2, y2 = coords[b]
        ax.plot([x1, x2], [y1, y2], color="black", linewidth=2.2, linestyle="--", alpha=0.9, zorder=2)

    # 범례(간단)
    legend_labels = [
        ("상업/환승 거점", colors["hub"]),
        ("혼합 거점", colors["mixed"]),
        ("주거 거점", colors["residential"]),
        ("버스트 OD(상위)", "#e03131"),
        ("노선 후보(휴리스틱)", "black"),
    ]
    handles = []
    for label, color in legend_labels[:3]:
        handles.append(plt.Line2D([], [], marker="o", linestyle="", color=color, markeredgecolor="black", markersize=10, label=label))
    handles.append(plt.Line2D([], [], color="#e03131", linewidth=3, label=legend_labels[3][0]))
    handles.append(plt.Line2D([], [], color="black", linewidth=2, linestyle="--", label=legend_labels[4][0]))
    ax.legend(handles=handles, loc="upper left", framealpha=0.95)

    ax.set_title("합성 심야 이동 네트워크: 수요(엣지) + 버스트 + 노선 후보", fontsize=13, fontweight="bold")
    ax.set_xlim(-0.02, 1.02)
    ax.set_ylim(0.05, 0.95)
    ax.axis("off")
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches="tight", facecolor="white")
    if show:
        plt.show()
    plt.close(fig)


def plot_burst_timeseries(
    df: pd.DataFrame,
    burst_edges: pd.DataFrame,
    shock_start: str,
    shock_end: str,
    *,
    save_path: str,
    show: bool,
) -> None:
    labels = time_bins()

    fig = plt.figure(figsize=(12, 6))
    ax = plt.gca()

    top_edges = burst_edges.head(3)
    if top_edges.empty:
        ax.text(0.5, 0.5, "표시할 버스트 엣지가 없습니다.", ha="center", va="center")
        plt.show()
        return

    pivot = df.pivot_table(
        index="time_bin",
        columns=["origin", "destination"],
        values="count",
        aggfunc="sum",
        fill_value=0,
    )

    for r in top_edges.itertuples(index=False):
        series = pivot[(r.origin, r.destination)].values
        label = f"{r.origin} → {r.destination}"
        ax.plot(range(len(series)), series, linewidth=2.2, label=label)

    # 창발 구간 강조(가상)
    shock_bins = [i for i, t in enumerate(labels) if shock_start <= t < shock_end]
    if shock_bins:
        ax.axvspan(min(shock_bins), max(shock_bins), alpha=0.12, color="red", label="버스트 구간(가정)")

    ax.set_xticks(range(0, len(labels), 2))
    ax.set_xticklabels([labels[i] for i in range(0, len(labels), 2)], rotation=0)
    ax.set_xlabel("시간(15분 단위)", fontsize=11)
    ax.set_ylabel("OD 수요(건)", fontsize=11)
    ax.set_title("버스트(창발) 후보 OD의 시간대별 수요 변화", fontsize=13, fontweight="bold")
    ax.grid(True, alpha=0.25)
    ax.legend()
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches="tight", facecolor="white")
    if show:
        plt.show()
    plt.close(fig)


def main(argv: Iterable[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="올빼미버스 창발 수요 실습(합성 데이터)")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--base-trips", type=int, default=120, help="15분 bin당 기본 이동 수(대략)")
    parser.add_argument("--shock-trips", type=int, default=40, help="버스트 구간에서 추가되는 이동 수(15분 bin당)")
    parser.add_argument("--shock-start", type=str, default="01:00", help="버스트 시작(HH:MM)")
    parser.add_argument("--shock-end", type=str, default="01:45", help="버스트 종료(HH:MM)")
    parser.add_argument("--top-edges", type=int, default=15, help="시각화에 표시할 상위 엣지 수")
    parser.add_argument("--route-stops", type=int, default=6, help="노선 후보 정류장 수(상위 거점)")
    parser.add_argument("--save-dir", type=str, default="diagrams", help="그림 저장 폴더")
    parser.add_argument("--no-show", action="store_true", help="화면 표시(plt.show) 생략")
    args = parser.parse_args(list(argv) if argv is not None else None)

    configure_korean_font()

    zones = build_seoul_zones()
    df = generate_synthetic_od(
        zones=zones,
        seed=args.seed,
        base_trips_per_bin=args.base_trips,
        shock_trips_per_bin=args.shock_trips,
        shock_start=args.shock_start,
        shock_end=args.shock_end,
    )

    od = df.groupby(["origin", "destination"], as_index=False)[["count", "shock_count"]].sum()
    total_trips = int(od["count"].sum())

    print("=" * 80)
    print("실습: 올빼미버스와 창발적 수요(합성 데이터)")
    print("=" * 80)
    print(f"- 시간 bin: {len(time_bins())}개(15분 단위), 구간: 00:00~04:00")
    print(f"- 총 이동(합성): {total_trips:,}건")
    print(f"- 버스트(가정) 구간: {args.shock_start}~{args.shock_end} (추가 {args.shock_trips}건/bin)")

    print("\n[1] OD 상위 10개(총량 기준)")
    top10 = od.sort_values("count", ascending=False).head(10).copy()
    top10["share"] = (top10["count"] / total_trips) * 100
    for i, r in enumerate(top10.itertuples(index=False), 1):
        print(f"{i:2d}. {r.origin:6s} → {r.destination:6s} | {int(r.count):4d}건 | {r.share:5.1f}%")

    print("\n[2] '창발(버스트)' 후보 OD(변동성 기반) 상위 5개")
    burst = compute_edge_burst(df).head(5).copy()
    labels = time_bins()
    for i, r in enumerate(burst.itertuples(index=False), 1):
        max_time = labels[int(r.max_time_bin)]
        print(
            f"{i:2d}. {r.origin:6s} → {r.destination:6s} | burst={r.burst_score:5.2f} | "
            f"max={int(r.max):4d}건 @ {max_time} | mean={r.mean:5.1f}"
        )

    print("\n[3] 노선 후보(휴리스틱): 상위 거점 기반 정류장 연결")
    route = propose_route(zones=zones, od=od, n_stops=args.route_stops)
    route_str = " → ".join(route)
    print(f"- 정류장({len(route)}개): {route_str}")

    covered = od[od["origin"].isin(route) & od["destination"].isin(route)]["count"].sum()
    shock_covered = od[od["origin"].isin(route) & od["destination"].isin(route)]["shock_count"].sum()
    print(f"- 커버리지(노선 정류장 내부 OD 합): {int(covered):,}건 ({covered/total_trips*100:,.1f}%)")
    if int(od["shock_count"].sum()) > 0:
        print(f"- 버스트 수요 커버리지: {int(shock_covered):,}건 ({shock_covered/od['shock_count'].sum()*100:,.1f}%)")

    print("\n[4] 시각화")
    print("  - (a) 수요 네트워크 + 버스트 OD + 노선 후보")
    print("  - (b) 버스트 후보 OD의 시간대별 수요")

    os.makedirs(args.save_dir, exist_ok=True)
    network_png = os.path.join(args.save_dir, "12-owlbus-emergent-network.png")
    burst_png = os.path.join(args.save_dir, "12-owlbus-emergent-burst-timeseries.png")

    plot_network(
        zones=zones,
        od=od,
        burst_edges=compute_edge_burst(df),
        route=route,
        top_edges=args.top_edges,
        save_path=network_png,
        show=(not args.no_show),
    )
    plot_burst_timeseries(
        df,
        compute_edge_burst(df),
        args.shock_start,
        args.shock_end,
        save_path=burst_png,
        show=(not args.no_show),
    )
    print(f"\n  - 저장: {network_png}")
    print(f"  - 저장: {burst_png}")

    print("\n완료: 이 예제는 '합성 데이터' 기반 원리 실습입니다.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
