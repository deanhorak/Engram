# Layer-adaptive K=512 budget selection

Status: **rejected after sequence-disjoint confirmation**

Individual-layer exact-magnitude interventions were measured at K=256, 384, 512, 640, and 768
on four configuration-selection sequences. Dynamic programming minimized the sum of individual
teacher-student KL values while fixing the total active records to 15,360, exactly the same as
uniform K=512 across 30 layers.

The selected schedule assigns K=768 to layers 0 and 29; K=640 to layers 1, 10, and 22; K=384 to
layers 3, 4, 8, 14, 18, 25, and 26; and K=512 elsewhere. Its additive selection objective is 12.0%
lower than uniform K=512 on the selection split.

The schedule was then frozen and evaluated once on 16 distinct held-out sequences. It fails every
causal threshold: KL 0.1341, top-1 agreement 0.7862, NLL delta +0.1104, and final-hidden relative
L2 0.1852. It is slightly worse overall than the prior uniform exact K=512 reference, so this
fixed layer-adaptive magnitude schedule is not promoted.
