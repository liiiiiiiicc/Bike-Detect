from __future__ import annotations

import itertools
import math
from pathlib import Path

import gurobipy as gp
from gurobipy import GRB
import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch


Q = 15

DEPOT = "O"
STATIONS = ["A", "B", "C", "D"]
NODES = [DEPOT] + STATIONS

NODE_NAMES = {
    "O": "配送中心",
    "A": "至臻楼",
    "B": "友园15号楼",
    "C": "食堂",
    "D": "安楼",
}

COORDS = {
    "O": (300.0, 377.0),
    "A": (0.0, 0.0),
    "B": (573.0, 712.0),
    "C": (806.0, 325.0),
    "D": (461.0, 0.0),
}

DEMANDS = {
    "morning": {"A": 0, "B": 40, "C": 0, "D": 0},
    "noon": {"A": 30, "B": 0, "C": 0, "D": 30},
    "evening": {"A": 30, "B": 30, "C": 30, "D": 30},
}

CURRENT_COUNTS = {
    "A": 10,
    "B": 45,
    "C": 35,
    "D": 30,
}

TIME_PERIOD = "evening"

ALPHA_STATION = 1.0
BETA_STATION = 0.2

ALPHA_CENTER = 2.0
BETA_CENTER = 1.0

MAX_LEGS = 6

OUTPUT_DIR = Path("bike_route_output")


plt.rcParams["font.sans-serif"] = ["Microsoft YaHei", "SimHei", "Arial Unicode MS"]
plt.rcParams["axes.unicode_minus"] = False


def euclidean_distance(i: str, j: str) -> float:
    xi, yi = COORDS[i]
    xj, yj = COORDS[j]
    return math.hypot(xi - xj, yi - yj)


DIST = {
    (i, j): euclidean_distance(i, j)
    for i in NODES
    for j in NODES
    if i != j
}


def alpha(node: str) -> float:
    return ALPHA_CENTER if node == DEPOT else ALPHA_STATION


def beta(node: str) -> float:
    return BETA_CENTER if node == DEPOT else BETA_STATION


def generate_routes(max_legs: int) -> list[list[str]]:
    routes = []

    for legs in range(2, max_legs + 1):
        middle_len = legs - 1

        for middle in itertools.product(STATIONS, repeat=middle_len):
            route = [DEPOT] + list(middle) + [DEPOT]

            valid = True
            for a, b in zip(route[:-1], route[1:]):
                if a == b:
                    valid = False
                    break

            if valid:
                routes.append(route)

    return routes


def solve_for_fixed_route(
    route: list[str],
    current_counts: dict[str, int],
    demands: dict[str, int],
    center_supply_allowed: bool,
) -> dict | None:
    legs = list(zip(route[:-1], route[1:]))

    model = gp.Model("fixed_route_rebalance")
    model.Params.OutputFlag = 0

    q = model.addVars(len(legs), vtype=GRB.INTEGER, lb=0, ub=Q, name="bikes")

    inventory = {}
    for t in range(len(legs) + 1):
        for station in STATIONS:
            inventory[t, station] = model.addVar(vtype=GRB.INTEGER, lb=0, name=f"inv_{t}_{station}")

    for station in STATIONS:
        model.addConstr(inventory[0, station] == current_counts[station], name=f"init_{station}")

    for t, (i, j) in enumerate(legs):
        for station in STATIONS:
            expr = inventory[t, station]

            if i == station:
                expr = expr - q[t]

            if j == station:
                expr = expr + q[t]

            model.addConstr(inventory[t + 1, station] == expr, name=f"flow_{t}_{station}")

        if i == DEPOT and not center_supply_allowed:
            model.addConstr(q[t] == 0, name=f"no_center_supply_{t}")

        if j == DEPOT:
            model.addConstr(q[t] == 0, name=f"empty_return_{t}")

    for station in STATIONS:
        model.addConstr(inventory[len(legs), station] >= demands[station], name=f"demand_{station}")

    total_cost = gp.quicksum(
        alpha(i) * DIST[i, j] + beta(i) * q[t]
        for t, (i, j) in enumerate(legs)
    )

    model.setObjective(total_cost, GRB.MINIMIZE)
    model.optimize()

    if model.Status != GRB.OPTIMAL:
        return None

    steps = []
    for t, (i, j) in enumerate(legs):
        bikes = int(round(q[t].X))
        distance = DIST[i, j]
        vehicle_cost = alpha(i) * distance
        bike_cost = beta(i) * bikes

        steps.append(
            {
                "step": t + 1,
                "from": i,
                "to": j,
                "from_name": NODE_NAMES[i],
                "to_name": NODE_NAMES[j],
                "bikes": bikes,
                "distance": distance,
                "vehicle_cost": vehicle_cost,
                "bike_cost": bike_cost,
                "total_cost": vehicle_cost + bike_cost,
            }
        )

    final_counts = {
        station: int(round(inventory[len(legs), station].X))
        for station in STATIONS
    }

    return {
        "route": route,
        "steps": steps,
        "final_counts": final_counts,
        "objective": model.ObjVal,
        "vehicle_cost": sum(step["vehicle_cost"] for step in steps),
        "bike_cost": sum(step["bike_cost"] for step in steps),
    }


def solve_complete_plan(
    current_counts: dict[str, int],
    time_period: str,
    max_legs: int = MAX_LEGS,
) -> dict:
    if time_period not in DEMANDS:
        raise ValueError(f"Unknown time period: {time_period}. Use one of {list(DEMANDS)}")

    demands = DEMANDS[time_period]

    total_current = sum(current_counts[i] for i in STATIONS)
    total_required = sum(demands[i] for i in STATIONS)

    center_supply_allowed = total_current < total_required

    routes = generate_routes(max_legs)

    best = None
    feasible_count = 0

    for route in routes:
        result = solve_for_fixed_route(
            route=route,
            current_counts=current_counts,
            demands=demands,
            center_supply_allowed=center_supply_allowed,
        )

        if result is None:
            continue

        feasible_count += 1

        if best is None or result["objective"] < best["objective"]:
            best = result

    if best is None:
        raise RuntimeError(
            f"No feasible route found with MAX_LEGS={max_legs}. Try increasing MAX_LEGS."
        )

    best["time_period"] = time_period
    best["current_counts"] = current_counts
    best["demands"] = demands
    best["total_current"] = total_current
    best["total_required"] = total_required
    best["center_supply_allowed"] = center_supply_allowed
    best["feasible_route_count"] = feasible_count
    best["candidate_route_count"] = len(routes)

    return best


def print_result(result: dict) -> None:
    print("\n" + "=" * 80)
    print("完整调度计划")
    print("=" * 80)

    print(f"时间段: {result['time_period']}")
    print(f"现场总车数: {result['total_current']}")
    print(f"最低需求总数: {result['total_required']}")
    print(f"是否允许配送中心补车: {result['center_supply_allowed']}")
    print(f"候选路线数量: {result['candidate_route_count']}")
    print(f"可行路线数量: {result['feasible_route_count']}")

    print("\n站点库存:")
    print(f"{'站点':<12}{'当前位置':>10}{'最低需求':>10}{'调度后':>10}")
    for station in STATIONS:
        print(
            f"{NODE_NAMES[station]:<12}"
            f"{result['current_counts'][station]:>10}"
            f"{result['demands'][station]:>10}"
            f"{result['final_counts'][station]:>10}"
        )

    print("\n最优路线:")
    print(" -> ".join(f"{node}({NODE_NAMES[node]})" for node in result["route"]))

    print("\n执行步骤:")
    print(
        f"{'步':>4}{'起点':<12}{'终点':<12}{'运车数':>8}"
        f"{'距离':>10}{'车辆成本':>12}{'单车成本':>12}{'合计':>12}"
    )

    for step in result["steps"]:
        move_type = "空驶" if step["bikes"] == 0 else f"{step['bikes']}辆"
        print(
            f"{step['step']:>4}"
            f"{step['from_name']:<12}"
            f"{step['to_name']:<12}"
            f"{move_type:>8}"
            f"{step['distance']:>10.1f}"
            f"{step['vehicle_cost']:>12.1f}"
            f"{step['bike_cost']:>12.1f}"
            f"{step['total_cost']:>12.1f}"
        )

    print("\n成本汇总:")
    print(f"车辆行驶成本: {result['vehicle_cost']:.2f}")
    print(f"单车搬运成本: {result['bike_cost']:.2f}")
    print(f"总成本: {result['objective']:.2f}")


def bezier_point(p0, p1, rad: float, t: float = 0.5):
    x0, y0 = p0
    x1, y1 = p1

    mx = (x0 + x1) / 2
    my = (y0 + y1) / 2

    dx = x1 - x0
    dy = y1 - y0

    # Match matplotlib Arc3:
    # control point = midpoint + rad * (dy, -dx)
    cx = mx + rad * dy
    cy = my - rad * dx

    x = (1 - t) ** 2 * x0 + 2 * (1 - t) * t * cx + t**2 * x1
    y = (1 - t) ** 2 * y0 + 2 * (1 - t) * t * cy + t**2 * y1

    return x, y


def curved_rad(step_index: int, from_node: str, to_node: str) -> float:
    # Smaller curvature than before so labels stay visually tied to the line.
    if from_node == DEPOT or to_node == DEPOT:
        base = 0.13
    else:
        base = 0.10

    # Opposite directions get opposite curvature.
    if from_node < to_node:
        sign = 1
    else:
        sign = -1

    # Alternate slightly to avoid exact overlap between repeated paths.
    if step_index % 2 == 0:
        sign *= -1

    return sign * base


def plot_result(result: dict, output_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)

    fig = plt.figure(figsize=(14, 8))
    ax = fig.add_axes([0.06, 0.08, 0.66, 0.84])
    panel = fig.add_axes([0.75, 0.08, 0.22, 0.84])
    panel.axis("off")

    for node, (x, y) in COORDS.items():
        if node == DEPOT:
            ax.scatter(x, y, s=300, marker="s", color="#3A3A3A", zorder=4, label="配送中心")
        else:
            ax.scatter(x, y, s=260, color="#2E86DE", zorder=4, label="站点" if node == "A" else None)

        label = f"{node} {NODE_NAMES[node]}"
        if node in STATIONS:
            label += (
                f"\n{result['current_counts'][node]}"
                f" -> {result['final_counts'][node]}"
                f" / min {result['demands'][node]}"
            )

        ax.text(x + 12, y + 12, label, fontsize=10, zorder=5)

    colors = [
        "#7F8C8D",
        "#E74C3C",
        "#8E44AD",
        "#16A085",
        "#F39C12",
        "#2980B9",
        "#D35400",
        "#2C3E50",
    ]

    for step in result["steps"]:
        idx = step["step"]
        i, j = step["from"], step["to"]
        p0 = COORDS[i]
        p1 = COORDS[j]

        is_empty = step["bikes"] == 0
        color = "#8A8A8A" if is_empty else colors[idx % len(colors)]
        linestyle = "--" if is_empty else "-"
        linewidth = 2.0 if is_empty else 3.0
        rad = curved_rad(idx, i, j)

        arrow = FancyArrowPatch(
            p0,
            p1,
            connectionstyle=f"arc3,rad={rad}",
            arrowstyle="-|>",
            mutation_scale=20,
            linewidth=linewidth,
            linestyle=linestyle,
            color=color,
            alpha=0.82,
            shrinkA=23,
            shrinkB=23,
            zorder=3,
        )
        ax.add_patch(arrow)

        # Label is placed directly on the corresponding Bezier curve.
        # Put the label exactly at the midpoint of the same arc used by FancyArrowPatch.
        lx, ly = bezier_point(p0, p1, rad, t=0.5)

        label = f"{idx} 空驶" if is_empty else f"{idx} 运{step['bikes']}辆"

        ax.text(
            lx,
            ly,
            label,
            fontsize=10,
            fontweight="bold",
            color=color,
            ha="center",
            va="center",
            bbox={
                "facecolor": "white",
                "edgecolor": color,
                "boxstyle": "round,pad=0.22",
                "alpha": 0.94,
            },
            zorder=7,
        )

    ax.set_title(f"共享单车完整调度路线 - {result['time_period']}", fontsize=15)
    ax.set_xlabel("x / m")
    ax.set_ylabel("y / m")
    ax.grid(True, linestyle="--", alpha=0.35)
    ax.axis("equal")
    ax.legend(loc="upper right")

    panel.text(0.0, 0.98, "执行顺序", fontsize=15, fontweight="bold", va="top")

    y = 0.92
    for step in result["steps"]:
        if step["bikes"] == 0:
            line1 = f"{step['step']}. 空驶"
        else:
            line1 = f"{step['step']}. 运送 {step['bikes']} 辆"

        line2 = f"{step['from']} {step['from_name']} -> {step['to']} {step['to_name']}"
        line3 = f"距离 {step['distance']:.1f} m，成本 {step['total_cost']:.1f}"

        panel.text(0.0, y, line1, fontsize=11, fontweight="bold", va="top")
        panel.text(0.0, y - 0.035, line2, fontsize=10, va="top")
        panel.text(0.0, y - 0.07, line3, fontsize=9, color="#555555", va="top")

        y -= 0.125

    panel.text(0.0, max(y - 0.02, 0.05), "成本汇总", fontsize=13, fontweight="bold", va="top")
    panel.text(
        0.0,
        max(y - 0.07, 0.0),
        f"车辆行驶成本: {result['vehicle_cost']:.1f}\n"
        f"单车搬运成本: {result['bike_cost']:.1f}\n"
        f"总成本: {result['objective']:.1f}",
        fontsize=10,
        va="top",
    )

    fig.savefig(output_path, dpi=220)
    print(f"\n可视化结果已保存到: {output_path}")


def main() -> None:
    result = solve_complete_plan(
        current_counts=CURRENT_COUNTS,
        time_period=TIME_PERIOD,
        max_legs=MAX_LEGS,
    )

    print_result(result)
    plot_result(result, OUTPUT_DIR / f"complete_route_{TIME_PERIOD}.png")


if __name__ == "__main__":
    main()