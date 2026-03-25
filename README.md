# Learning Geometric Algebra Through Transformer Geometry

A self-guided introduction to **Geometric Algebra** (Clifford Algebra) using the internal geometry of transformer language models as your laboratory.

> *Rotors, bivectors, and holonomy in the hidden-state field of transformers*

**Authors:** Agus Sudjianto, Sandi Setiawan, Aijun Zhang

## What This Is

This repository contains:

1. **A textbook** (`monograph_ga_learning.pdf`) that teaches Geometric Algebra one concept at a time, immediately grounding each concept in real data from transformer language models.
2. **14 companion Jupyter notebooks** (`tutorials_ga_learning/`) with working code, visualizations, and exercises.
3. **A Python library** (`layer_time_ga`) that wraps standard PyTorch/NumPy linear algebra with GA semantics: bivectors, rotors, holonomy, and commutator structure.

## Why Learn GA Through Transformers?

Geometric Algebra was born in physics and thrives in computer graphics and robotics. But transformers offer something these fields do not: a high-dimensional space (k = 256 after whitening) where *every* GA concept --- bivectors, rotors, holonomy, commutators --- appears naturally in the data.

In 3D physics, a bivector is just a dressed-up cross product. In R^256, bivectors have 32,640 independent components, multiple principal planes, and rich internal structure. This is where GA earns its keep.

By learning GA through this lens, you gain two things at once: a deep understanding of a beautiful algebra, and a new way to see inside the models that are reshaping technology.

## Structure

### Part I: Vectors and Products (Chapters 1--4)
| Chapter | GA Concept | Transformer Insight |
|---------|-----------|-------------------|
| 1. Vectors Live Somewhere | Inner product, grade-0 | Hidden states encode alignment |
| 2. The Product That Does Everything | Geometric product | Both alignment and plane in one operation |
| 3. Planes, Not Axes | Bivectors, principal planes | Layer rotations have multiple planes |
| 4. When Your Coordinates Lie | Orthonormal frames, Cl(k,0) | Whitening establishes valid GA |

### Part II: Rotations and Rotors (Chapters 5--8)
| Chapter | GA Concept | Transformer Insight |
|---------|-----------|-------------------|
| 5. Rotations Without Matrices | Rotors, sandwich product | Rotations with explicit planes |
| 6. What a Layer Actually Does | Versor decomposition | Grade-0 (stretch) vs grade-2 (rotation) |
| 7. Reading the Planes | Plane evolution, similarity | Rotation planes evolve across layers |
| 8. The Eigenvalue Story | Grade-0 dominance | Stretching controls gradients, not rotation |

### Part III: Curvature and Commutators (Chapters 9--11)
| Chapter | GA Concept | Transformer Insight |
|---------|-----------|-------------------|
| 9. When Order Matters | Commutators, Lie algebra so(k) | Non-commutativity lives in specific planes |
| 10. Walking in Circles | Holonomy, curvature bivector | Curvature has direction, not just magnitude |
| 11. How Much Computation? | Capacity, Jacobi identity | Decompose complexity by plane |

### Part IV: The Full Picture (Chapters 12--14)
| Chapter | GA Concept | Transformer Insight |
|---------|-----------|-------------------|
| 12. Which Planes Matter? | Dependency + bivectors | Identify functionally relevant planes |
| 13. Diagnosing and Steering | Applications | Plane-specific interventions |
| 14. What GA Gave Us | Retrospective | Directions invisible to scalar summaries |

## Installation

```bash
pip install ga-transformer-geometry
```

Or install from source:

```bash
git clone https://github.com/asudjianto-xml/GA-Transformer-Geometry.git
cd GA-Transformer-Geometry
pip install -e .
```

With tutorial dependencies:

```bash
pip install -e ".[tutorials]"
```

## Quick Start

```python
import ltg_ga

# Load a model
model = ltg_ga.load_model("Qwen/Qwen2.5-7B")

# Run GA analysis
result = ltg_ga.analyse("The capital of France is", model=model)

# See the summary
result.summary()

# Generate the 4-panel GA summary plot
result.plot_ga_summary(save_path="ga_summary.png")
```

## The GA Toolkit

### Core Algebra (`layer_time_ga.algebra`)

```python
from layer_time_ga.algebra import (
    Bivector, Rotor,
    bivector_from_skew, rotor_from_orthogonal,
    rotor_compose, rotor_inverse,
    commutator_bivector, grade_decomposition,
    geometric_product_vectors,
)

# Geometric product of two vectors
gp = geometric_product_vectors(a, b)
print(f"Scalar: {gp['scalar']}")        # grade-0 (alignment)
print(f"Bivector: {gp['bivector'].norm}")  # grade-2 (plane)

# Commutator of two bivectors
comm = commutator_bivector(B1, B2)
planes = comm.principal_planes(n_planes=3)  # which planes fail to commute
```

### Decomposition (`layer_time_ga.decomposition`)

```python
from layer_time_ga.decomposition import extract_rotor_field

rf = extract_rotor_field(H_whitened)
for vd in rf.decompositions:
    print(f"Layer {vd.layer_index}: angle={vd.rotor.angle:.4f}, "
          f"kappa={vd.condition_number:.1f}")
    planes = vd.bivector.principal_planes(n_planes=3)
```

### Curvature (`layer_time_ga.curvature`)

```python
from layer_time_ga.curvature import holonomy_rotor, commutator_field

# Holonomy: curvature with direction
hr = holonomy_rotor(H_whitened, l=20, t=2)
print(f"Scalar curvature: {hr.scalar_curvature}")
print(f"Curvature plane: {hr.principal_plane}")

# Commutator field: pairwise non-commutativity
comm_norms = commutator_field(rf.bivectors)
```

## The Key Mapping

| Matrix Operation | GA Equivalent | What GA Adds |
|-----------------|--------------|-------------|
| Dot product a^T b | Inner product a . b | Same |
| Skew-symmetric A | Bivector B | Principal plane decomposition |
| Orthogonal U in SO(k) | Rotor R = exp(-B/2) | Explicit plane + angle |
| Polar decomp T = UP | Versor decomp T = RP | Grade-2 x grade-0 separation |
| \|\|U - I\|\|_F | Rotor angle theta | Plus: which plane |
| \|\|[A_i, A_j]\|\|_F | \|\|[B_i, B_j]\|\| | Plus: decompose into planes |
| \|\|Omega - I\|\|_F | Holonomy rotor | Plus: curvature direction |

## Prerequisites

- **Linear algebra**: matrix multiplication, eigenvalues, SVD
- **Python**: NumPy-level comfort
- **No prior GA knowledge** --- the book starts from vectors
- **No prior ML knowledge** --- Appendix A covers transformers

## Running the Tutorials

```bash
pip install -e ".[tutorials]"
cd tutorials_ga_learning
jupyter lab
```

Start with `ch01_vectors_live_somewhere.ipynb` and work through sequentially.

## Running Tests

```bash
pip install -e ".[dev]"
pytest tests/ -v
```

## Citation

If you use this work, please cite:

```
Sudjianto, A., Setiawan, S., and Zhang, A. (2026).
Learning Geometric Algebra Through Transformer Geometry.
https://github.com/asudjianto-xml/GA-Transformer-Geometry
```

## License

Apache 2.0. See [LICENSE](LICENSE).
