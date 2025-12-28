Let’s do do-calculus the way it *wants* to be learned: as a small set of graph moves that let you turn scary-looking intervention expressions like
[
p(y\mid do(x))
]
into things you can actually estimate from observational data.

---

## 1) The basic objects (notation + what they mean)

### Variables

* (X): “treatment” / intervention variable (could be gene knockdown, drug exposure, CRISPR perturbation…)
* (Y): outcome (phenotype, expression program, survival, etc.)
* (Z, W): other measured covariates (cell state, batch, donor, severity, etc.)
* (U): unobserved confounder (latent factors, unmeasured biology)

### Two kinds of probability statements

**Observational (seeing):**
[
p(y\mid x) \quad \text{or} \quad p(y\mid x,z)
]
This is what you learn by conditioning in regular statistics: “among samples with (X=x), what’s (Y)?”

**Interventional (doing):**
[
p(y\mid do(x))
]
This means: “what would (Y) look like if we *forced* (X) to (x) (like a perfect perturbation), regardless of what would naturally cause (X) to be that value?”

Key point: generally
[
p(y\mid do(x)) \neq p(y\mid x)
]
because conditioning doesn’t break confounding, but interventions do.

### Graph language (DAGs)

We represent causal assumptions with a directed acyclic graph (DAG).
Edges mean direct causal influence. Confounding is represented by hidden common causes or bidirected edges (X \leftrightarrow Y) (a shorthand for an unobserved (U) causing both).

### d-separation (quick intuition)

A set of variables (S) “blocks” paths between (A) and (B) in a DAG if conditioning on (S) cuts off all “information flow” paths. That’s the engine behind “which independences hold” and thus which do-calculus rule you’re allowed to use.

---

## 2) The point of do-calculus (why it exists)

There are two big tasks:

1. **Identification:** can we rewrite (p(y\mid do(x))) using only observational quantities like (p(y\mid x,z)), (p(z\mid x)), etc.?
2. **Estimation:** once identified, estimate it from data.

Do-calculus solves (1) using graph rules.

If you can’t rewrite it, you don’t have enough assumptions + measurements to identify the causal effect from observational data alone.

---

## 3) The “surgery” idea (do-operator as graph surgery)

Doing (do(x)) means: **cut** all arrows *into* (X), then set (X=x).

Call the intervened graph (G_{\bar X}) (“bar X” = incoming edges to (X) removed).

Sometimes we also cut outgoing edges from a variable (rare in basic use; appears in the formal rules).

---

## 4) The 3 rules of do-calculus (what they *let* you swap)

Do-calculus gives conditions (based on d-separation in modified graphs) under which you can:

### Rule 1: Insert/delete observations

Swap conditioning on (Z):
[
p(y\mid do(x), z, w) = p(y\mid do(x), w)
]
**if** (Y \perp Z \mid X,W) in the intervened graph (G_{\bar X}).

Intuition: after forcing (X), learning (Z) gives no extra info about (Y) (given (W)).

---

### Rule 2: Action/observation exchange

Swap observing (Z) vs doing (Z):
[
p(y\mid do(x), do(z), w) = p(y\mid do(x), z, w)
]
**if** (Y \perp Z \mid X,W) in a graph where incoming edges to (X) are cut and (roughly) outgoing edges from (Z) are cut in the right way (this is the most technical rule; we’ll use it in examples where it’s clean).

Intuition: under certain graphical conditions, “setting (Z)” is equivalent to “conditioning on (Z)” for predicting (Y) once (X,W) are handled.

---

### Rule 3: Insert/delete actions

Remove an intervention:
[
p(y\mid do(x), do(z), w) = p(y\mid do(x), w)
]
**if** (Y \perp Z \mid X,W) in a graph where you cut incoming edges to (X) and also cut incoming edges to (Z) that are *not ancestors* of (W).

Intuition: intervening on (Z) doesn’t matter for (Y) once you intervene on (X) and condition on (W), because there’s no active causal route left from (Z) to (Y).

---

If those sounded abstract: good. They’re supposed to. The *real* learning happens in worked examples.

---

## 5) Worked Example A: The back-door adjustment (classic, but we’ll derive it)

### Graph

[
Z \to X \to Y,\quad Z \to Y
]
(Z) is a confounder (common cause of (X) and (Y)).

Goal: identify (p(y\mid do(x))).

### Step 1: Start with law of total probability

[
p(y\mid do(x)) = \sum_z p(y, z \mid do(x))
]

### Step 2: Expand joint into conditional × marginal

[
p(y, z \mid do(x)) = p(y\mid z, do(x)), p(z\mid do(x))
]
So
[
p(y\mid do(x))=\sum_z p(y\mid z, do(x)), p(z\mid do(x))
]

### Step 3: Simplify (p(z\mid do(x)))

Intervening on (X) cuts arrows **into (X)**, but does not change how (Z) is generated (since (Z\to X), not (X\to Z)).
So in (G_{\bar X}), (Z) is unaffected:
[
p(z\mid do(x)) = p(z)
]

### Step 4: Simplify (p(y\mid z, do(x)))

In the graph with incoming edges to (X) removed, conditioning on (Z) blocks the back-door path (X \leftarrow Z \to Y). So after fixing (X), the distribution of (Y) given (Z) matches observational conditioning:
[
p(y\mid z, do(x)) = p(y\mid x, z)
]

### Final back-door formula

[
\boxed{p(y\mid do(x))=\sum_z p(y\mid x,z),p(z)}
]

That’s do-calculus in action: turn an interventional query into observational terms.

---

## 6) Worked Example B: Front-door adjustment (the “confounding but still identifiable” magic trick)

### Graph

* Unobserved confounder between (X) and (Y): (X \leftrightarrow Y)
* A mediator (M) on the causal path: (X \to M \to Y)
* No other path from (X) to (Y) except through (M)
* (X) blocks confounding into (M) (typically no (X\leftrightarrow M))

Goal: identify (p(y\mid do(x))) even though (X) and (Y) are confounded.

### Step 1: Total probability over mediator

[
p(y\mid do(x)) = \sum_m p(y\mid m, do(x)), p(m\mid do(x))
]

### Step 2: Simplify (p(m\mid do(x)))

Because (M) is caused by (X), and under (do(x)) we set (X=x):
[
p(m\mid do(x)) = p(m\mid x)
]
(observationally estimable)

So now:
[
p(y\mid do(x)) = \sum_m p(y\mid m, do(x)), p(m\mid x)
]

### Step 3: Replace (p(y\mid m, do(x))) with (p(y\mid do(m)))

Key graphical idea: once you fix (M), (X) has no direct effect on (Y) (all causal influence goes through (M)). With the front-door conditions, you can show:
[
p(y\mid m, do(x)) = p(y\mid do(m))
]

So:
[
p(y\mid do(x)) = \sum_m p(y\mid do(m)), p(m\mid x)
]

### Step 4: Identify (p(y\mid do(m))) via back-door through (X)

Now consider effect of (M) on (Y). The confounder (U) affects (X) and (Y), and (X) affects (M). So (X) becomes a back-door adjustment set for (M\to Y) (under the front-door assumptions).

Thus:
[
p(y\mid do(m)) = \sum_{x'} p(y\mid m, x'), p(x')
]

### Combine

[
\boxed{
p(y\mid do(x))=\sum_m p(m\mid x)\sum_{x'} p(y\mid m,x')p(x')
}
]

This is one of the most satisfying “graph wizardry” results in causal inference: even with unmeasured confounding between (X) and (Y), you can sometimes identify the helping causal effect using a mediator that meets the right conditions.

---

## 7) Worked Example C: The “why adjustment sets are not just feature selection”

### Graph (collider trap)

[
X \to C \leftarrow U \to Y
]
Here (C) is a **collider**: two arrows collide into it.

* Without conditioning on (C), the path (X \to C \leftarrow U \to Y) is blocked at collider (C).
* If you condition on (C), you **open** the path and create spurious association between (X) and (U), which then leaks into (Y).

So (p(y\mid x, c)) can be *more biased* than (p(y\mid x)). Do-calculus makes this precise: the required conditional independences fail after conditioning on a collider.

This is why “just control for more variables” is not a safe rule in causal land.

---

## 8) How this plugs into causal-bio-lab thinking (bio-flavored intuition)

In computational biology, the typical “villains” are:

* **Cell state** as a confounder (e.g., cycling cells both get transfected better *and* express genes differently)
* **Batch / donor** as confounders
* **Selection bias** (you only observe cells that survive perturbation; survival becomes a collider)
* **Mediators** (perturbation changes pathway activity (M), which changes phenotype (Y))

Do-calculus is the formal way to decide:

* which covariates are safe to adjust for,
* when a mediator can rescue identification (front-door),
* and when your dataset simply cannot identify the effect without interventions or extra measurements.

---

## 9) A mini “muscle memory” checklist

When you see (p(\cdot\mid do(\cdot))):

1. Write the DAG you believe.
2. Ask: can I use **back-door**? (find a set (Z) that blocks all back-door paths from (X) to (Y) and doesn’t include descendants of (X))
3. If not, ask: can I use **front-door**?
4. If still not, it might be non-identifiable from observational data alone (needs experiments, instruments, proxies, or stronger assumptions).

---

If you want the next step that gets *really* useful for coding in causal-bio-lab: I can walk through **a full identification-by-hand** for a slightly messier biology-ish DAG (e.g., (X)=perturbation, (Z)=cell cycle, (B)=batch, (M)=pathway activation, (S)=survival/selection collider), and show exactly where each do-calculus rule fires—and where it refuses to.
