"""Build eval/questions.jsonl from the hand-written items below.

Each answerable item has an `evidence` snippet copied from the lecture PDF.
This script verifies the snippet occurs (whitespace-insensitive) in
the extracted text and records every matching character span plus PDF pages.
Unanswerable items carry no evidence; the script verifies that their key
terms never occur in the corpus.

Usage: python eval/build_questions.py
"""
import json
import re
import sys
from pathlib import Path

import pymupdf

ROOT = Path(__file__).resolve().parent.parent
PDF = ROOT / "data" / "Calculus-Lecture.pdf"
OUT = ROOT / "eval" / "questions.jsonl"

# (question, gold_answer, evidence)  -- for unanswerable items the third
# element is a list of lowercase terms that must NOT occur in the corpus.
ITEMS = [
    ("What condition on x1 and x2 must hold for a function to be called increasing on an interval?",
     "f(x1) < f(x2) whenever x1 < x2, for every pair of points in the interval.",
     "If f(x1) < f(x2) whenever x1 < x2, then f is said to be increasing on I."),
    ("How is the domain of a composition of two functions determined?",
     "It consists of the x in the domain of g for which g(x) lies in the domain of f.",
     "The domain of f ◦g consists of the numbers x in the domain of g for which g(x) lies in the domain of f"),
    ("What does it mean for a trigonometric function to be periodic, and what is the period of sine?",
     "There is p > 0 with f(x+p) = f(x) for all x; the smallest such p is the period. Sine has period 2π.",
     "A function f(x) is periodic if there is a number p > 0 such that f(x + p) = f(x) for every value of x."),
    ("How is the area of a circular sector computed from its radius and angle?",
     "A = (1/2) r^2 θ, i.e. half the radius times the arc length (arc length = rθ).",
     "Arc length ℓ= rθ Area A = 1 2rℓ= 1 2r2θ"),
    ("What test tells you whether a curve in the plane can be the graph of a function?",
     "The vertical line test: no vertical line may intersect the graph more than once.",
     "no vertical line can intersect the graph of a function more than once."),
    ("What property must a function have to be called one-to-one?",
     "It never takes the same value twice: f(x1) != f(x2) whenever x1 != x2.",
     "A function f is called a one-to-one function if it never takes on the same value twice"),
    ("How can you tell graphically whether a function is one-to-one?",
     "Horizontal line test: no horizontal line intersects its graph more than once.",
     "A function is one-to-one if and only if no horizontal line intersects its graph more than once."),
    ("What is the special name and notation for the logarithm with base e?",
     "The natural logarithm, written ln x.",
     "The logarithm with base e is called the natural logarithm and has a special notation"),
    ("Which numbers and which range does the arcsine function return?",
     "arcsin x is the number in [-pi/2, pi/2] whose sine is x.",
     "y = arcsin x is the number in [−π/2, π/2] for which sin y = x"),
    ("How do you compute the average change of a function across an interval of inputs?",
     "dy/dx = (f(x2) - f(x1)) / (x2 - x1) = (f(x1+h) - f(x1))/h.",
     "The average rate of change of y = f(x) with respect to x over [x1, x2], x2 = x1 + h, is"),
    ("In informal terms, what does it mean to say a function has a limit L at a point a?",
     "f(x) can be made arbitrarily close to L by restricting x sufficiently close to a (but not equal to a).",
     "if we can make the values of f(x) arbitrarily close to L (as close to L as we like) by restricting x to be sufﬁciently close to a"),
    ("If a function is squeezed between two functions that share the same limit, what can be concluded?",
     "The squeezed function has that same limit (Sandwich Theorem).",
     "Suppose also that lim x→a g(x) = lim x→a h(x) = L. Then limx→a f(x) = L."),
    ("What is the value of the limit of sin θ over θ as θ goes to zero?",
     "It equals 1 (θ measured in radians).",
     "Limits involving sin θ/θ lim θ→0 sin θ θ = 1."),
    ("When does a two-sided limit exist in terms of one-sided limits?",
     "Exactly when the left-hand and right-hand limits both exist and are equal.",
     "if and only if it has left-hand and right-hand limits there and these one-sided limits are equal"),
    ("What three conditions must be met for a function to be continuous at a point?",
     "f(a) is defined, the limit as x->a exists, and the limit equals f(a).",
     "A function f(x) is continuous at a point x = a if and only if it meets the following three conditions."),
    ("What does the Intermediate Value Theorem guarantee for a continuous function on a closed interval?",
     "For any N between f(a) and f(b) there is a c in (a, b) with f(c) = N.",
     "Then there exists a number c ∈(a, b) such that f(c) = N."),
    ("When is a line called a horizontal asymptote of a curve?",
     "When the limit of f(x) as x->inf or x->-inf equals L; the line is y = L.",
     "The line y = L is called a horizontal asymptote of the curve y = f(x) if either"),
    ("How is a vertical asymptote defined using limits?",
     "x = a is a vertical asymptote if a one-sided limit of f at a is +-inf.",
     "A line x = a is a vertical asymptote of the graph of a function y = f(x) if either"),
    ("What is the formal limit definition of the derivative of f at a point x0?",
     "f'(x0) = lim_{h->0} (f(x0+h) - f(x0))/h, provided the limit exists.",
     "The derivative of a function f(x) at a point x0, denoted f ′(x0), is"),
    ("If a function has a derivative at a point, what follows about its continuity there?",
     "It must be continuous at that point (differentiability implies continuity).",
     "Differentiability Implies Continuity: If f has a derivative at x = c, then f is continuous at x = c."),
    ("What is the formula for differentiating a product of two functions?",
     "(fg)' = f'g + fg'.",
     "If both f and g are differentiable, then so is their product fg, and"),
    ("How do you differentiate a power of x?",
     "d/dx x^n = n x^(n-1).",
     "Power Rule: If n is any real number, then"),
    ("What does the derivative of a position function represent, and how is speed related to velocity?",
     "ds/dt is the instantaneous velocity; speed is the absolute value of velocity.",
     "Speed is the absolute value of velocity"),
    ("What is the term for the derivative of the derivative of a function?",
     "The second derivative, written f''.",
     "This new function is called the second derivative of f, because it is the derivative of the ﬁrst derivative."),
    ("How is the derivative of a composition of two functions computed?",
     "Chain Rule: (f o g)'(x) = f'(g(x)) * g'(x).",
     "then the composite function (f ◦g)(x) = f(g(x)) is differentiable at x"),
    ("What are the steps for differentiating an equation that defines y implicitly?",
     "Differentiate both sides with respect to x treating y as a function of x, then collect dy/dx terms and solve for it.",
     "Differentiate the equation (both sides) with respect to x, treating y as a differentiable function of x."),
    ("What is the procedure for logarithmic differentiation?",
     "Take natural logs of both sides, differentiate implicitly, solve for y', replace y with f(x).",
     "Take natural logarithms of both sides of an equation y = f(x) and use the Laws of Logarithms"),
    ("How can the number e be expressed as a limit?",
     "e = lim_{x->0} (1 + x)^(1/x).",
     "The number e can be calculated as the limit"),
    ("What are the steps of the recommended strategy for problems where several quantities change with time and are tied by an equation?",
     "Name variables as functions of t, write an equation relating them, differentiate with respect to t, plug in known values to find the unknown rate.",
     "Differentiate with respect to t to obtain a related rates equation."),
    ("What is the formula for the linear approximation of a function near a point?",
     "L(x) = f(a) + f'(a)(x - a), the linearization of f at a.",
     "then the approximating function L(x) := f(a) + f ′(a)(x −a) (3.45) is the linearization of f at a."),
    ("How is the differential dy defined for a differentiable function?",
     "dy = f'(x) dx, where dx is an independent variable.",
     "The differential dy is a dependent variable, deﬁned as dy = f ′(x)dx"),
    ("What conditions must a quadratic approximation to f at x = a satisfy?",
     "Q(a) = f(a), Q'(a) = f'(a) and Q''(a) = f''(a).",
     "A quadratic approximation Q(x) = b0 + b1(x −a) + b2(x −a)2 to f(x) at x = a is deﬁned to satisfy the properties"),
    ("Under what hypothesis does a function attain both a maximum and a minimum on an interval?",
     "If f is continuous on a closed interval [a, b], it attains an absolute max and min there (Extreme Value Theorem).",
     "If f is continuous on a closed interval [a, b], then f attains both an absolute maximum value M and an absolute minimum value m in [a, b]"),
    ("What is a critical point of a function?",
     "An interior point of the domain where f' is zero or undefined.",
     "An interior point in the domain of a function f where f ′ is zero or undeﬁned is a critical point of f."),
    ("What does Rolle's Theorem say?",
     "If f is continuous on [a, b], differentiable on (a, b) and f(a) = f(b), then f'(c) = 0 for some c in (a, b).",
     "If f(a) = f(b), then there is at least one number c ∈(a, b) at which f ′(c) = 0."),
    ("What does the Mean Value Theorem state about the slope of a function on an interval?",
     "There is a point c in (a, b) where f'(c) = (f(b) - f(a))/(b - a).",
     "Then there is at least one point c ∈(a, b) at which f ′(c) = f(b) −f(a) b −a ."),
    ("When is a graph concave up on an interval?",
     "When f' is increasing on the interval (equivalently f'' > 0).",
     "(a) concave up (convex) on an open interval I if f ′ is increasing on I;"),
    ("What is a point of inflection?",
     "A point on the graph with a tangent line where the concavity changes.",
     "A point (c, f(c)) where the graph of a function has a tangent line and where the concavity changes is a point of inﬂection."),
    ("How does the second derivative classify a point where the first derivative is zero?",
     "f'(c)=0 and f''(c)<0 gives a local max; f''(c)>0 a local min; f''(c)=0 is inconclusive.",
     "If f ′(c) = 0 and f ′′(c) < 0, then f has a local maximum at x = c."),
    ("Which kinds of limit quotients can the derivative-based rule for indeterminate forms be applied to?",
     "Indeterminate forms 0/0 and inf/inf (L'Hopital's rule).",
     "this limit may or may not exist and is called an indeterminate form of type 0/0."),
    ("What are the steps for solving an applied optimization problem?",
     "Read the problem, introduce variables, write the quantity as a function of one variable, test critical points and endpoints.",
     "Express the quantity to be optimized as a function of a single variable."),
    ("What iteration formula is used to approximate a root of f(x) = 0 by successive guesses?",
     "x_{n+1} = x_n - f(x_n)/f'(x_n), with f'(x_n) != 0 (Newton's method).",
     "Use the ﬁrst approximation to get a second, the second to get a third, and so on"),
    ("What is an antiderivative of a function?",
     "A function F with F'(x) = f(x) for all x in the interval.",
     "A function F is an antiderivative of f on an interval I if F ′(x) = f(x) for all x in I."),
    ("What is the name for the collection of all antiderivatives of a function?",
     "The indefinite integral of f with respect to x.",
     "The collection of all antiderivatives of f is called the indeﬁnite integral of f with respect to x"),
    # ---- not answerable from this corpus (key terms verified absent by the build script)
    ("How is a definite integral defined as the limit of Riemann sums?", None, ["riemann"]),
    ("What is the Taylor series of e^x around zero?", None, ["taylor", "maclaurin"]),
    ("How do you compute partial derivatives of a function of two variables?", None, ["partial deriv"]),
    ("State the Fundamental Theorem of Calculus.", None, ["fundamental theorem"]),
    ("How do you determine whether an infinite series converges using the ratio test?", None, ["ratio test", "series"]),
    ("What is the Laplace transform of a derivative?", None, ["laplace"]),
]


def load_pages():
    doc = pymupdf.open(str(PDF))
    texts = [p.get_text().replace("\x00", "") for p in doc]
    spans, pos = [], 0
    for t in texts:
        spans.append((pos, pos + len(t)))
        pos += len(t)
    return "".join(texts), spans


def find_span(corpus, snippet):
    pat = r"\s+".join(re.escape(tok) for tok in snippet.split())
    return [m.span() for m in re.finditer(pat, corpus)]


def main():
    corpus, page_spans = load_pages()
    rows, problems = [], []
    for i, (q, gold, ev) in enumerate(ITEMS, start=1):
        row = {"id": f"q{i:02d}", "question": q}
        if gold is None:
            low = corpus.lower()
            present = [t for t in ev if t in low]
            if present:
                problems.append((row["id"], "unanswerable term present", present))
            row.update(answerable=False, gold_answer=None, evidence=None,
                       spans=[], pdf_pages=[])
        else:
            hits = find_span(corpus, ev)
            if not hits:
                problems.append((row["id"], "no match", ev[:60]))
                continue
            if len(hits) > 1:
                print(f"note: {row['id']} evidence occurs {len(hits)} times; all spans accepted")
            pages = sorted({n + 1 for s, e in hits
                            for n, (a, b) in enumerate(page_spans) if a < e and s < b})
            row.update(answerable=True, gold_answer=gold, evidence=ev,
                       spans=[list(h) for h in hits], pdf_pages=pages)
        rows.append(row)
    if problems:
        for p in problems:
            print("PROBLEM", p)
        sys.exit(1)
    with open(OUT, "w", encoding="utf-8", newline="\n") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    n_ans = sum(r["answerable"] for r in rows)
    print(f"wrote {len(rows)} questions ({n_ans} answerable) to {OUT}")


if __name__ == "__main__":
    main()
