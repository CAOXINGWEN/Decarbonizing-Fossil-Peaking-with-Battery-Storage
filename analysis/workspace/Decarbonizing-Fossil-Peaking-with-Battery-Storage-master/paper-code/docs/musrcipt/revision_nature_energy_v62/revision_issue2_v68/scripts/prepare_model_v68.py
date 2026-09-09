"""Freeze the v67 capacity model and derive the v68 nonnegative-capacity LP."""
from pathlib import Path
import ast
import difflib
import hashlib
import json
import shutil

HERE = Path(__file__).resolve().parent
WORK = HERE.parent
REV = WORK.parent
OLD = REV / 'capacity_expansion_validation/run_v62_multiyear_capacity_expansion.py'


def main():
    before = OLD.read_text()
    expected = '6a7c5e3d642b03b04c0ad4b683b8a722b0abf745f1e1378723ca24a5fd80ea7c'
    assert hashlib.sha256(OLD.read_bytes()).hexdigest() == expected
    (WORK / 'before').mkdir(exist_ok=True)
    for name in ('musript_new_v67.tex', 'musript_new_v67.pdf',
                 'musript_new_v67_supplementary.tex', 'musript_new_v67_supplementary.pdf'):
        destination = WORK / 'before' / name
        if destination.exists():
            assert destination.read_bytes() == (REV / name).read_bytes()
        else:
            shutil.copy2(REV / name, destination)
    shutil.copy2(OLD, WORK / 'before' / OLD.name)
    tree = ast.parse(before)
    # Preserve only the registered LP and its dependencies. Legacy solver and CLI
    # are not exposed by this model; the dedicated runner owns output locations.
    omit = {'solve_country_design', 'parse_list', 'parse_float_list', 'parse_args', 'main'}
    nodes = [n for n in tree.body if not (isinstance(n, ast.FunctionDef) and n.name in omit)
             and not (isinstance(n, ast.If) and ast.get_source_segment(before, n).startswith('if __name__'))]
    source = '\n\n'.join(ast.get_source_segment(before, n) for n in nodes) + '\n'
    patches = [
        ('HERE = Path(__file__).resolve().parent',
         'HERE = Path(__file__).resolve().parents[2] / "capacity_expansion_validation"'),
        ('    for index in capacity_index:\n        bounds[int(index)] = (0.0, maximum_task)\n', ''),
        ('"natural_year_count": int(scenario_count // 2),',
         '"natural_year_count": len({int(item["year"]) for item in scenarios}),'),
        ('"presolve": True,', '"presolve": True,\n            "threads": 1,'),
        ('    solution = np.asarray(result.x, dtype=float)\n',
         '''    solution = np.asarray(result.x, dtype=float)
    maximum_inequality_violation = float(np.maximum(inequality_matrix @ solution - inequality_rhs_array, 0).max())
    maximum_equality_residual = float(np.abs(equality_matrix @ solution - equality_rhs_array).max())
    stationarity = objective - equality_matrix.T @ result.eqlin.marginals - inequality_matrix.T @ result.ineqlin.marginals - result.lower.marginals
    dual_objective = float(equality_rhs_array @ result.eqlin.marginals + inequality_rhs_array @ result.ineqlin.marginals)
'''),
        ('    risk_adjusted_value = float(\n',
         '''    lp_cvar_epigraph_value = cvar_operating_value
    op = np.asarray(operating_values, dtype=float)
    cvar_operating_value = float(min(t + np.maximum(op - t, 0).sum() / ((1.0 - CVAR_ALPHA) * scenario_count) for t in op))
    risk_adjusted_value = float(
'''),
        ('"maximum_capacity_upper_bound_slack": float(maximum_task - np.max(capacity_values)),',
         '''"task_peak_minus_maximum_capacity": float(maximum_task - np.max(capacity_values)),
        "capacity_bound": "nonnegative_without_task_peak_upper_bound",
        "maximum_inequality_violation": maximum_inequality_violation,
        "maximum_equality_residual": maximum_equality_residual,
        "maximum_nonnegativity_violation": float(np.maximum(-solution, 0).max()),
        "maximum_stationarity_residual": float(np.abs(stationarity).max()),
        "primal_dual_objective_gap_eur": float(result.fun - dual_objective),
        "lp_cvar_epigraph_value": lp_cvar_epigraph_value,
        "cvar_reporting_method": "empirical_equal_probability_scenario_costs",
        "solver_iterations": int(result.nit),'''),
    ]
    for old, new in patches:
        assert source.count(old) == 1, old
        source = source.replace(old, new)
    assert 'bounds[int(index)] = (0.0, maximum_task)' not in source
    ast.parse(source)
    target = HERE / 'capacity_model_v68.py'
    target.write_text(source)
    (WORK / 'model_changes.diff').write_text(''.join(difflib.unified_diff(
        before.splitlines(True), source.splitlines(True), fromfile=OLD.name, tofile=target.name)))
    hashes = {str(p.relative_to(REV)): hashlib.sha256(p.read_bytes()).hexdigest()
              for p in [OLD, target, *sorted((WORK / 'before').glob('musript*'))]}
    (WORK / 'model_provenance.json').write_text(json.dumps(hashes, indent=2) + '\n')
    print(target)


if __name__ == '__main__':
    main()
