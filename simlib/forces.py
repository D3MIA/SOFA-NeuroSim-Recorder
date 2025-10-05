import numpy as np
import Sofa, Sofa.Core

# Forces are recorded/exported in Newtons upstream
FORCE_TO_NEWTON = 1e-3 


class ExternalForceAggregator(Sofa.Core.Controller):
    """
    Combine per-frame external forces from multiple deformers and write them to
    the MechanicalObject.externalForce so simulation and recorders see the final
    applied value. Keeps the last non-zero to persist at end-of-animation.
    """
    def __init__(self, mo, deformers=None, name="externalForceAggregator", **kw):
        super().__init__(name=name, **kw)
        self.mo = mo
        self.deformers = list(deformers) if deformers is not None else []
        self._combined = None
        self._last_nonzero = None

    def onAnimateBeginEvent(self, *_):
        N = len(self.mo.position.value)
        combined = np.zeros((N, 3), dtype=np.float32)
        any_nonzero = False
        for d in self.deformers:
            F = getattr(d, '_frame_force', None)
            if F is None:
                continue
            if isinstance(F, list):
                F = np.array(F, dtype=np.float32)
            if F.shape != combined.shape:
                continue
            combined += F
            any_nonzero = any_nonzero or (np.abs(F).sum() > 0)
        if any_nonzero:
            self._last_nonzero = combined.copy()
        self._combined = combined
        try:
            if hasattr(self.mo, 'externalForce') and hasattr(self.mo.externalForce, 'value'):
                self.mo.externalForce.value = combined.tolist()
        except Exception:
            pass

    def onAnimateEndEvent(self, *_):
        """
        Re-assert the combined external force at end-of-frame so any controller
        (e.g., AnimationRecorder) reading in onAnimateEndEvent sees the final value.
        Some solvers zero out externalForce during the step; this ensures persistence
        for recording/inspection.
        """
        try:
            payload = None
            if self._combined is not None:
                payload = self._combined
            elif self._last_nonzero is not None:
                payload = self._last_nonzero
            if payload is not None and hasattr(self.mo, 'externalForce') and hasattr(self.mo.externalForce, 'value'):
                self.mo.externalForce.value = payload.tolist()
        except Exception:
            pass

    def onEndAnimation(self, *_):
        try:
            if self._last_nonzero is not None:
                if hasattr(self.mo, 'externalForce') and hasattr(self.mo.externalForce, 'value'):
                    self.mo.externalForce.value = self._last_nonzero.tolist()
        except Exception:
            pass
        self._combined = None

    @property
    def combined_frame_force(self):
        return self._combined
