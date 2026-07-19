import numpy as np

from engram.corrections import CorrectionCapsule, CorrectionManager


def test_capsule_selection_and_uncertainty_escalation():
    capsule = CorrectionCapsule(np.ones(4), 0.2, np.ones((4, 1)) * 0.1, np.ones((1, 4)) * 0.2)
    manager = CorrectionManager([capsule], uncertainty_threshold=0.4)
    decision = manager.decide(np.ones(4), 0.8)
    assert decision.capsule_index == 0
    assert decision.extra_cycles == 1 and decision.expand_vocabulary
    assert not np.array_equal(manager.apply(np.ones(4), decision), np.ones(4))
