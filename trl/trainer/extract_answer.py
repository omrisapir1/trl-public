import re
import regex  # grader.py uses 'regex' which is different from 're'
from math import isclose
from typing import Union, Optional
import sympy
from sympy import simplify, N
from sympy.parsing.sympy_parser import parse_expr
from sympy.parsing.latex import parse_latex
# Note: latex2sympy2 is a specific dependency mentioned in math_utils.py
# and used in grader.py's symbolic_equal._parse
try:
    from latex2sympy2 import latex2sympy
except ImportError:
    print("Warning: latex2sympy2 not installed. Symbolic comparison might be less robust.")
    # Define a dummy function if not available
    def latex2sympy(x):
        raise NotImplementedError("latex2sympy2 is required")

# ---------------------------------------------------------------------------
# Simplified Answer Extraction (Normally from parser.py -> extract_answer)
# ---------------------------------------------------------------------------

def extract_final_answer(llm_output: str) -> Optional[str]:
    """
    Extracts the final answer, typically enclosed in \\boxed{}.
    Searches from the end of the string backwards.
    Handles variations like \boxed {answer} or \boxed{ answer }.
    """
    # Search for the last occurrence of \boxed{...}
    # Use regex to handle potential spaces around the curly braces
    match = re.search(r"\\boxed\s*\{(.*?)\}", llm_output, re.DOTALL | re.IGNORECASE)

    if match:
        # Return the content inside the \boxed{}
        answer = match.group(1).strip()
        # Optional: Remove potential leftover LaTeX environments like \begin{...} \end{...}
        # that might sometimes accidentally wrap the answer inside the box.
        answer = re.sub(r"\\begin\{.*?\}(.*?)\\end\{.*?\}", r"\1", answer, flags=re.DOTALL).strip()
        return answer
    else:
        # Fallback: Maybe the answer is just the last line if no box is found?
        # This is less reliable and depends on the model's output format.
        lines = llm_output.strip().split('\n')
        if lines:
            last_line = lines[-1].strip()
            # A heuristic: if the last line looks like a final answer (e.g., is a number or simple expression)
            # You might need more sophisticated checks here depending on expected formats.
            # For now, just return the last line as a fallback.
            # A better fallback might try to find the *last* number in the output.
            return last_line
        return None # No answer found


# ---------------------------------------------------------------------------
# Core Comparison Logic (from grader.py)
# ---------------------------------------------------------------------------

def choice_answer_clean(pred: str) -> str:
    """Cleans prediction specifically for multiple-choice A,B,C,D,E answers."""
    if pred is None: return ""
    pred = str(pred).strip("\n").rstrip(".").rstrip("/").strip(" ").lstrip(":")
    # Clean the answer based on the dataset
    tmp = re.findall(r"\b(A|B|C|D|E)\b", pred.upper()) # Use re, not regex here as in original
    if tmp:
        # Take the last found letter if multiple are present
        pred_last = tmp[-1]
    else:
        # If no explicit letter found, return the stripped prediction
        # (might be a number or formula even if GT is a choice)
         pred_last = pred.strip().strip(".")

    # Remove the period at the end, again!
    pred_last = pred_last.rstrip(".").rstrip("/")
    return pred_last

def parse_digits(num: Union[str, int, float]) -> Optional[float]:
    """Converts a string to a float, handling commas and percentages."""
    if isinstance(num, (int, float)):
        return float(num)
    if not isinstance(num, str):
        return None

    num_str = regex.sub(",", "", num.strip()) # Use regex here as in original
    try:
        return float(num_str)
    except ValueError:
        if num_str.endswith("%"):
            num_str = num_str[:-1]
            if num_str.endswith("\\"): # Handle latex \%
                num_str = num_str[:-1]
            try:
                return float(num_str) / 100
            except ValueError:
                pass
    return None

def is_digit(num: Union[str, int, float]) -> bool:
    """Checks if the input can be parsed as a number by parse_digits."""
    return parse_digits(num) is not None

def numeric_equal(prediction: float, reference: float) -> bool:
    """Checks if two numbers are close enough."""
    # Note that relative tolerance has significant impact
    # on the result of the synthesized GSM-Hard dataset
    # Original code didn't specify rounding based on reference integer status,
    # using isclose directly matches the simplified logic seen in symbolic_equal fallback
    return isclose(reference, prediction, rel_tol=1e-4)


def symbolic_equal(a: str, b: str) -> bool:
    """Compares two strings symbolically using SymPy."""
    if a is None or b is None: return False

    # Basic check
    if a == b: return True

    def _parse(s):
        s = str(s) # Ensure it's a string
        # Pre-processing similar to math_utils.clean_expr_str might be needed here
        # for robustness, e.g., replacing \times, handling spaces.
        s = s.replace("\\$", "").replace("$", "") # Basic cleaning
        s = s.replace("\\%", "").replace("%", "")
        s = s.replace("\\\\", "\\") # Reduce double backslashes

        parsers = [parse_latex, latex2sympy, parse_expr]
        for f in parsers:
            try:
                parsed = f(s)
                # Ensure it's a SymPy object, not just returned string on failure
                if isinstance(parsed, sympy.Expr) or isinstance(parsed, sympy.logic.boolalg.Boolean):
                     # Substitute pi and i like in math_utils
                    if isinstance(parsed, sympy.Expr):
                        parsed = parsed.subs({sympy.Symbol("pi"): sympy.pi})
                        parsed = parsed.subs({sympy.Symbol("i"): sympy.I})
                    return parsed
            except Exception: # Catch specific parse errors if possible
                pass
        # print(f"Warning: Could not parse '{s}' with any method.")
        return s # Return original string if all parsing fails

    parsed_a = _parse(a)
    parsed_b = _parse(b)

    # If parsing failed and returned strings, compare directly
    if isinstance(parsed_a, str) or isinstance(parsed_b, str):
         # print(f"Comparing as strings after parse failure: '{parsed_a}' vs '{parsed_b}'")
         return str(parsed_a) == str(parsed_b)

    # --- SymPy Comparisons ---
    # Direct equality check with SymPy objects
    try:
        if parsed_a == parsed_b:
            return True
    except Exception as e:
        # print(f"Sympy direct == failed: {e}")
        pass # Avoid crashes on complex types

    # Simplify and check equality (most robust method)
    try:
        # Use sympy.simplify on the difference
        # equals(True) checks if simplification results in zero
        if sympy.simplify(parsed_a - parsed_b).equals(0):
            return True
    except Exception as e:
        # print(f"Sympy simplify failed: {e}")
        pass # Handles non-subtractable types, etc.

    # Check using .equals() method directly
    try:
        if parsed_a.equals(parsed_b):
            return True
    except AttributeError:
        pass # Handle cases where .equals is not available
    except Exception as e:
        # print(f"Sympy .equals() failed: {e}")
        pass

    # For equations, check if LHS-RHS difference is symbolically equivalent
    try:
        if isinstance(parsed_a, sympy.Eq) and isinstance(parsed_b, sympy.Eq):
             diff_a = parsed_a.lhs - parsed_a.rhs
             diff_b = parsed_b.lhs - parsed_b.rhs
             if sympy.simplify(diff_a - diff_b).equals(0) or \
                sympy.simplify(diff_a + diff_b).equals(0): # Check if one is negative of other
                 return True
    except Exception as e:
        # print(f"Sympy equation check failed: {e}")
        pass

    # Fallback: Numerical evaluation
    try:
        # N() evaluates numerically. Use evalf() for precision control if needed.
        num_a = N(parsed_a)
        num_b = N(parsed_b)
        # Check if both are numbers after evaluation
        if isinstance(num_a, (sympy.Number, float, int)) and \
           isinstance(num_b, (sympy.Number, float, int)):
            if numeric_equal(float(num_a), float(num_b)):
                return True
    except (TypeError, ValueError, AttributeError, NotImplementedError) as e:
        # print(f"Sympy numerical N() check failed: {e}")
        pass # Handle cases where numerical evaluation isn't possible/meaningful

    # Matrix comparison (simplified version without rounding)
    try:
        if isinstance(parsed_a, sympy.Matrix) and isinstance(parsed_b, sympy.Matrix):
            if parsed_a.shape == parsed_b.shape:
                # Check element-wise equality after simplification
                if sympy.simplify(parsed_a - parsed_b).equals(sympy.zeros(*parsed_a.shape)):
                     return True
    except Exception as e:
        # print(f"Sympy matrix check failed: {e}")
        pass

    # print(f"Symbolic check failed for: {a} vs {b}")
    return False


def str_to_pmatrix(input_str: str) -> str:
    """Converts simple brace/bracketed lists like '{1,2},{3,4}' to pmatrix latex."""
    input_str = input_str.strip()
    # Handle outer brackets/parens potentially containing the list
    input_str = input_str.strip("()[]")
    # Find segments like {a,b} or [a b] (approximating matrix rows)
    # This is a simplified regex, the original logic isn't shown but was likely more robust
    rows_str = regex.findall(r"\{.*?\}|\[.*?\]", input_str)
    if not rows_str: return input_str # Return original if no rows found

    pmatrix_rows = []
    for r in rows_str:
        r = r.strip("{}[]")
        # Replace commas or spaces with latex column separator '&'
        elements = regex.split(r",\s*|\s+", r.strip())
        pmatrix_rows.append(" & ".join(elements))

    # Join rows with latex row separator '\\'
    pmatrix_content = " \\\\ ".join(pmatrix_rows)
    return r"\begin{pmatrix}" + pmatrix_content + r"\end{pmatrix}"


def math_equal(
    prediction: Union[bool, float, int, str, None],
    reference: Union[bool, float, int, str, None],
    include_percentage: bool = True,
    is_close_numerical: bool = True, # Use isclose for numbers?
    # timeout functionality removed for simplicity
) -> bool:
    """
    Comprehensive comparison of math answers (numeric or symbolic).
    Adapted from grader.py.
    """
    if prediction is None or reference is None:
        # print("Comparison failed: None value provided.")
        return False

    # Convert bools to strings for consistency
    pred_str = str(prediction).strip()
    ref_str = str(reference).strip()

    # 1. Direct String Equality (case-insensitive)
    if pred_str.lower() == ref_str.lower():
        # print("Comparison success: Direct string match (case-insensitive).")
        return True

    # 2. Multiple Choice Check (handles 'A', 'B', etc.)
    # Check if reference looks like a multiple choice answer key
    if ref_str in ["A", "B", "C", "D", "E"]:
        cleaned_pred = choice_answer_clean(pred_str)
        if cleaned_pred == ref_str:
            # print(f"Comparison success: Multiple choice match ('{cleaned_pred}' == '{ref_str}').")
            return True
        # else:
            # print(f"Multiple choice mismatch: Cleaned prediction '{cleaned_pred}' != Reference '{ref_str}'.")

    # 3. Numerical Equality Check
    # Check if both can be parsed as digits
    is_pred_digit = is_digit(pred_str)
    is_ref_digit = is_digit(ref_str)

    if is_pred_digit and is_ref_digit:
        pred_num = parse_digits(pred_str)
        ref_num = parse_digits(ref_str)
        if pred_num is None or ref_num is None: # Should not happen if is_digit is True, but safety check
            # print("Comparison failed: Could not parse numbers after is_digit check.")
            return False

        # Handle potential percentage difference (e.g., 50 vs 0.5)
        possible_ref_values = {ref_num}
        if include_percentage:
            possible_ref_values.add(ref_num / 100.0)
            possible_ref_values.add(ref_num * 100.0)

        for ref_val in possible_ref_values:
            try:
                if is_close_numerical:
                    if numeric_equal(pred_num, ref_val):
                        # print(f"Comparison success: Numerical match (isclose) between {pred_num} and {ref_val}.")
                        return True
                else: # Exact numerical equality
                    if ref_val == pred_num:
                        # print(f"Comparison success: Exact numerical match between {pred_num} and {ref_val}.")
                        return True
            except Exception as e:
                # print(f"Numeric comparison exception: {e}")
                continue # Avoid crashes, try next reference value
        # print(f"Numerical mismatch: {pred_num} not close to any of {possible_ref_values}.")
        # If number check fails here, continue to symbolic check below

    # If one is a number and the other isn't, they usually don't match (unless symbolically)
    # No explicit 'return False' here; let symbolic check handle cases like "1/2" vs "0.5"

    # 4. Symbolic Equality Check

    # Pre-processing: Handle specific formats before sympy parsing
    # a) Convert pmatrix if needed (e.g., if pred has \pmatrix but ref has {1,2})
    # Note: This logic is a bit simplistic. A robust version would check format consistency.
    # Simplified check: if one looks like latex matrix and other doesn't, try conversion.
    is_pred_latex_matrix = r'\begin{pmatrix}' in pred_str or r'\begin{bmatrix}' in pred_str
    is_ref_latex_matrix = r'\begin{pmatrix}' in ref_str or r'\begin{bmatrix}' in ref_str
    if is_pred_latex_matrix and not is_ref_latex_matrix and '{' in ref_str:
        try:
            ref_str = str_to_pmatrix(ref_str)
            # print(f"Converted reference to pmatrix: {ref_str}")
        except Exception as e:
            # print(f"Failed to convert reference to pmatrix: {e}")
            pass # Conversion failed, proceed with original strings
    elif is_ref_latex_matrix and not is_pred_latex_matrix and '{' in pred_str:
        try:
            pred_str = str_to_pmatrix(pred_str)
            # print(f"Converted prediction to pmatrix: {pred_str}")
        except Exception as e:
            # print(f"Failed to convert prediction to pmatrix: {e}")
            pass # Conversion failed

    # b) Handle different bracket types for sets/intervals/tuples
    temp_pred_str, temp_ref_str = pred_str, ref_str
    # Check if one uses () and other uses [] for potential interval/tuple match
    pred_is_paren = temp_pred_str.startswith("(") and temp_pred_str.endswith(")")
    ref_is_brack = temp_ref_str.startswith("[") and temp_ref_str.endswith("]")
    pred_is_brack = temp_pred_str.startswith("[") and temp_pred_str.endswith("]")
    ref_is_paren = temp_ref_str.startswith("(") and temp_ref_str.endswith(")")

    # If brackets mismatch, try stripping them for a direct comparison of contents
    # Be careful not to strip if they are needed for sympy parsing (e.g., function calls)
    # Simple approach: strip if they seem to be just containers
    if (pred_is_paren and ref_is_brack) or (pred_is_brack and ref_is_paren):
         # Strip outer brackets/parens
         stripped_pred = temp_pred_str[1:-1].strip()
         stripped_ref = temp_ref_str[1:-1].strip()
         # Try comparing stripped content directly AND symbolically
         if stripped_pred == stripped_ref:
              # print("Comparison success: Match after stripping mismatched brackets.")
              return True
         # Also try symbolic comparison on stripped versions
         if symbolic_equal(stripped_pred, stripped_ref):
              # print("Comparison success: Symbolic match after stripping mismatched brackets.")
              return True

    # c) Handle simple equations (e.g., x=1 vs 1=x or x=1 vs x-1=0)
    # This needs careful implementation to avoid false positives.
    # The original grader code had logic for this; simplified here.
    if pred_str.count('=') == 1 and ref_str.count('=') == 1:
        try:
            pred_lhs, pred_rhs = [p.strip() for p in pred_str.split('=', 1)]
            ref_lhs, ref_rhs = [r.strip() for r in ref_str.split('=', 1)]
            # Try symbolic equality of the equations represented as LHS - RHS = 0
            if symbolic_equal(f"({pred_lhs}) - ({pred_rhs})", f"({ref_lhs}) - ({ref_rhs})"):
                # print("Comparison success: Equation match (LHS-RHS symbolic check).")
                return True
            # Also check if one is negative of the other, e.g., x=1 vs 1-x=0
            if symbolic_equal(f"({pred_lhs}) - ({pred_rhs})", f"-(({ref_lhs}) - ({ref_rhs}))"):
                # print("Comparison success: Equation match (LHS-RHS negative symbolic check).")
                return True
        except Exception as e:
            # print(f"Equation comparison failed: {e}")
            pass # Avoid crashes if splitting/parsing fails

    # d) Handle case like 'x=5' vs just '5' (if reference is just the value)
    if pred_str.count('=') == 1 and ref_str.count('=') == 0:
        try:
            pred_var, pred_val = [p.strip() for p in pred_str.split('=', 1)]
            # Heuristic: if the variable part is short (like 'x')
            if len(pred_var) <= 2:
                 # Compare the value part with the reference
                 if math_equal(pred_val, ref_str, include_percentage, is_close_numerical):
                     # print("Comparison success: Equation value matches reference value.")
                     return True
        except Exception: pass
    # And the reverse case
    if ref_str.count('=') == 1 and pred_str.count('=') == 0:
        try:
            ref_var, ref_val = [r.strip() for r in ref_str.split('=', 1)]
            if len(ref_var) <= 2:
                if math_equal(pred_str, ref_val, include_percentage, is_close_numerical):
                    # print("Comparison success: Prediction value matches equation value.")
                    return True
        except Exception: pass

    # 5. Final Symbolic Equality Check (using the prepared strings)
    # This is the main workhorse for complex expressions, fractions, etc.
    if symbolic_equal(pred_str, ref_str):
        # print("Comparison success: Symbolic match.")
        return True

    # If all checks fail
    # print(f"Comparison failed: No match found between '{pred_str}' and '{ref_str}'.")
    return False

