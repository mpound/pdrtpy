Chambolle's Dual-Projection Algorithm
=======================================

Chambolle's dual-projection algorithm is a method for removing noise from an image—or
more generally, making a signal less jagged—while trying not to blur its meaningful
edges. It does this by repeatedly finding and limiting the "pressure" that would cause
overly sharp, noise-like pixel-to-pixel changes, then using that pressure to produce a
cleaner image. The original method was introduced for total-variation minimization and
applied to denoising, image zooming, and interface-motion problems. (`math.ucla
<https://www.math.ucla.edu/~lvese/285j.1.09f/Chambolle.pdf>`_)

The problem it solves
------------------------

Suppose you have a grainy photograph. You want a cleaned version that:

- Remains close to the observed photograph, so it does not invent a different scene.
- Is smooth inside areas that should be smooth, such as a clear sky or a painted wall.
- Keeps real boundaries sharp, such as the outline between a person and the background.

Chambolle's method usually targets the **total variation (TV) denoising** objective:

.. math::

   \min_u
   \underbrace{\frac{1}{2}\|u-f\|_2^2}_{\text{do not stray far from the observation}}
   +
   \underbrace{\lambda\,\mathrm{TV}(u)}_{\text{discourage needless roughness}}

Here:

- :math:`f` is the noisy input image.
- :math:`u` is the cleaned image to recover.
- :math:`\mathrm{TV}(u)` roughly adds up the strength of all neighboring-pixel changes.
- :math:`\lambda` controls the compromise between fidelity and smoothness.

Unlike a simple blur filter, TV regularization penalizes lots of little fluctuations but
can preserve a single strong transition. That is why it is commonly used for
edge-preserving denoising. (`ipol <https://www.ipol.im/pub/art/2013/61/>`_)

What "dual projection" means
-------------------------------

The main trick is that, instead of directly manipulating every pixel of the cleaned
image, the algorithm works with an auxiliary field, often called :math:`p`.

Think of :math:`p` as a tiny two-dimensional arrow at each pixel:

- Its horizontal part describes a pressure related to left–right variation.
- Its vertical part describes a pressure related to up–down variation.
- Collectively, these arrows indicate where the method should suppress excessive local
  changes.

This is the **dual** viewpoint: solve an equivalent problem in terms of the corrective
pressure field rather than trying to minimize the image roughness head-on. Chambolle's
approach is specifically based on such a dual formulation. (`ipol
<https://www.ipol.im/pub/art/2013/61/article_lr.pdf>`_)

The **projection** part is a safety rule. At every pixel, the arrow is not allowed to
become too large:

.. math::

   \lVert p(i,j)\rVert \leq 1

If an updated arrow is too long, the algorithm scales it back to the boundary of that
allowed unit circle. In plain language:

    "You may apply smoothing pressure here, but only up to a fixed maximum strength."

That bound is the mathematical mechanism that represents the TV penalty.

How the loop works
---------------------

Beginning with no corrective pressure—:math:`p=0`—the algorithm cycles through the
following operations:

1. **Measure local changes**
   Look at where the current image estimate has strong horizontal or vertical
   differences. This is the image gradient.

2. **Update the pressure field**
   Adjust :math:`p` in response to those differences. Regions with many small,
   irregular changes tend to build a corrective pressure that smooths them.

3. **Project it back into bounds**
   If any arrow becomes stronger than allowed, shrink its length so its magnitude is at
   most 1. This prevents the method from oversmoothing or using an invalid dual
   solution.

4. **Recover the image**
   Convert the field of arrows into a pixel correction using its **divergence**—roughly,
   whether smoothing pressure is flowing into or out of each pixel. A typical
   reconstruction has the form

   .. math::

      u = f-\lambda\,\operatorname{div}(p)

5. **Repeat until stable**
   Stop when updates no longer noticeably change the result.

The algorithm can be regarded as a projected, gradient-style iteration on the dual
problem; it has a convergence guarantee when its step size obeys an appropriate
restriction. (`math.ucla <https://www.math.ucla.edu/~lvese/285j.1.09f/Chambolle.pdf>`_)

A physical analogy
---------------------

Imagine the image is a landscape:

- Random noise looks like lots of tiny, unwanted bumps.
- Important image edges look like cliffs.

The algorithm sends out many limited-strength "smoothing forces." They flatten the tiny
bumps within otherwise uniform regions. But because each force is capped and the
objective still requires similarity to the original image, the method does not simply
flatten every cliff into a gradual slope.

That behavior is the key advantage over ordinary smoothing: the method can reduce noise
in flat regions while keeping major boundaries relatively crisp.

The regularization control
------------------------------

The parameter :math:`\lambda` is the user-facing tradeoff:

.. list-table::
   :header-rows: 1

   * - Choice of :math:`\lambda`
     - Typical outcome
   * - Small
     - Output stays very close to the noisy image; less denoising.
   * - Moderate
     - Much random texture is reduced while principal edges remain.
   * - Large
     - Strong denoising and flatter regions, but weak details and fine textures may
       disappear.

A characteristic artifact of TV denoising is **staircasing**: a smoothly shaded area can
become a set of nearly flat patches separated by small jumps. That is not necessarily a
coding error; it follows from TV's preference for images that are piecewise smooth or
nearly piecewise constant.

Relation to ISTA
--------------------

Both ISTA (see :doc:`ista`) and Chambolle's algorithm are iterative regularization
methods: they balance "fit the measurements" against "prefer a simpler result."

.. list-table::
   :header-rows: 1

   * - Feature
     - ISTA
     - Chambolle's projection method
   * - Typical simplicity preference
     - Few nonzero coefficients—sparsity
     - Few or limited-strength spatial changes—low total variation
   * - Core cleanup operation
     - Soft-threshold coefficients
     - Project the dual gradient field onto a bounded set
   * - Typical applications
     - Sparse regression, compressed sensing, sparse signal recovery
     - Edge-preserving image denoising, deblurring, reconstruction
   * - Intuition
     - Remove weak components
     - Remove small local fluctuations while preserving major boundaries

In one sentence: **ISTA repeatedly throws away small coefficients; Chambolle repeatedly
limits the smoothing-pressure field that controls local image variation.**
