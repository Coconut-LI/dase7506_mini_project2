"""Check the selected candidate's reasoning for arithmetic errors.

Reads an errors JSON (as written by show.py) and, for each entry's
`selected_text`, extracts every explicit ``X = Y`` / ``X ≈ Y`` equation,
evaluates the left-hand side with a safe arithmetic evaluator, and flags the
equation when the stated right-hand side disagrees with the computed value.

Only explicit equations are checked: a step like ``60 x 3 = 180`` is verified,
but a bare number without ``=`` is not. Tolerances are relative (with an
absolute floor of 1) so rounded decimals such as ``1/12 = 0.0833`` are not
flagged, while genuine slips (``60 x 3 = 190``) are.

Prints the error rate (fraction of selected answers containing at least one
flagged equation) and writes a per-equation report for inspection.
"""
import argparse
import ast
import json
import operator
import re
from pathlib import Path

# --- safe arithmetic evaluator (numbers and + - * / **, parentheses) ---
_BINOPS = {
    ast.Add: operator.add,
    ast.Sub: operator.sub,
    ast.Mult: operator.mul,
    ast.Div: operator.truediv,
    ast.Pow: operator.pow,
}


def _eval_node(node):
    if isinstance(node, ast.Expression):
        return _eval_node(node.body)
    if isinstance(node, ast.Constant):
        if isinstance(node.value, (int, float)):
            return float(node.value)
        raise ValueError('non-numeric constant')
    if isinstance(node, ast.BinOp) and type(node.op) in _BINOPS:
        return _BINOPS[type(node.op)](_eval_node(node.left), _eval_node(node.right))
    if isinstance(node, ast.UnaryOp):
        value = _eval_node(node.operand)
        if isinstance(node.op, ast.USub):
            return -value
        if isinstance(node.op, ast.UAdd):
            return value
    raise ValueError('unsupported node')


def safe_eval(expr):
    return _eval_node(ast.parse(expr, mode='eval'))


def normalize(text):
    """Flatten markdown/LaTeX/unicode into plain arithmetic text."""
    t = str(text)
    t = re.sub(r'\\frac\s*\{([^{}]*)\}\s*\{([^{}]*)\}', r'(\1)/(\2)', t)
    t = re.sub(r'\\(?:text|mathrm|mathbf|mathit|mbox)\s*\{[^{}]*\}', '', t)
    t = t.replace('\\left', '').replace('\\right', '')
    t = t.replace('\\times', '*').replace('\\cdot', '*').replace('\\div', '/')
    t = t.replace('\\approx', '\u2248').replace('\\boxed', '').replace('\\box', '')
    t = re.sub(r'\\[a-zA-Z]+', '', t)
    t = t.replace('\u00d7', '*').replace('\u00f7', '/').replace('\u00b7', '*').replace('\u2212', '-')
    t = t.replace('$', '')                 # currency and LaTeX math delimiters
    t = re.sub(r'\*\*', '', t)             # markdown bold
    t = re.sub(r'(?<=\d),(?=\d{3}(?!\d))', '', t)  # thousands separators
    return t


_EQ_OP = re.compile(r'(=|≈)')


def extract_equations(text, eq_tol=1e-3, approx_tol=1e-2):
    t = normalize(text)
    found = []
    for m in _EQ_OP.finditer(t):
        op = m.group(1)
        pos = m.start()
        # Left-hand side: walk left over arithmetic characters.
        j = pos - 1
        while j >= 0 and t[j] in '0123456789.+-*/() ':
            j -= 1
        lhs = t[j + 1:pos].strip()
        # Right-hand side: leading (signed) number.
        rm = re.match(r'\s*([-+]?\d[\d.,]*)', t[pos + 1:])
        rhs = rm.group(1) if rm else ''
        # A "calculation" needs a real expression, not a bare number.
        if not lhs or not rhs or not any(ch in lhs for ch in '+-*/('):
            continue
        # LHS must start with a digit or '('; a leading '+/-' means the walk
        # hit a variable/word and truncated the expression (e.g. "a + 24 = ...").
        if lhs[0] not in '0123456789(':
            continue
        # RHS must be a complete number, not a link in a longer chain
        # (e.g. "8 - 6 = 2 + 0" or "= 8 - 6 = 2").
        after = pos + 1 + rm.end()
        while after < len(t) and t[after] == ' ':
            after += 1
        if after < len(t) and t[after] in '+-*/=≈(':
            continue
        try:
            computed = safe_eval(lhs)
            stated = float(rhs.replace(',', ''))
        except Exception:
            continue
        tol = approx_tol if op == '≈' else eq_tol
        matched = abs(computed - stated) <= tol * max(1.0, abs(computed), abs(stated))
        found.append({
            'lhs': lhs,
            'op': op,
            'rhs': rhs,
            'computed': round(computed, 6),
            'stated': stated,
            'matched': bool(matched),
        })
    return found


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--input', type=Path,
                        default=Path('runs/show_set_attention/validation_errors.json'))
    parser.add_argument('--output', type=Path, default=Path('runs/calc_check.json'))
    parser.add_argument('--eq-tol', type=float, default=1e-3,
                        help='Relative tolerance for "=" equations')
    parser.add_argument('--approx-tol', type=float, default=1e-2,
                        help='Relative tolerance for "≈" equations')
    args = parser.parse_args()

    payload = json.loads(args.input.read_text())
    if isinstance(payload, dict):
        payload = payload.get('errors', [])
    if not isinstance(payload, list):
        raise ValueError('Expected a JSON list of error records')

    results = []
    for e in payload:
        equations = extract_equations(e['selected_text'], args.eq_tol, args.approx_tol)
        flagged = [q for q in equations if not q['matched']]
        results.append({
            'id': e.get('id'),
            'selected_index': e.get('selected_index'),
            'selected_parsed_answer': e.get('selected_parsed_answer'),
            'gold_answer_parsed': e.get('gold_answer_parsed'),
            'n_equations': len(equations),
            'has_calc_error': len(flagged) > 0,
            'flagged': flagged,
            'all_equations': equations,
        })

    total = len(results)
    with_error = sum(1 for r in results if r['has_calc_error'])
    rate = 100.0 * with_error / total if total else 0.0

    print(json.dumps({
        'input': str(args.input),
        'total_selected': total,
        'with_calc_error': with_error,
        'error_rate_percent': round(rate, 2),
        'eq_tol': args.eq_tol,
        'approx_tol': args.approx_tol,
    }, indent=2))

    examples = [q for r in results for q in r['flagged']][:5]
    if examples:
        print('\nSample flagged equations:')
        for q in examples:
            print(f"  {q['lhs']} {q['op']} {q['rhs']}  (computed {q['computed']}, stated {q['stated']})")

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(results, indent=2, ensure_ascii=False) + '\n')
    print(f'\nWrote detailed report to {args.output}')


if __name__ == '__main__':
    main()
