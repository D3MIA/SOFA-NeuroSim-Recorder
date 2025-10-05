import os
import Sofa, Sofa.Core

class VisualWatchdog(Sofa.Core.Controller):
    def __init__(self, loader, ogl_model, check_every=10, **kw):
        super().__init__(**kw)
        self.loader = loader
        self.ogl = ogl_model
        self.k = max(1, int(check_every))
        self._step = 0
        try:
            self._orig_filename = str(self.loader.filename.value)
        except Exception:
            self._orig_filename = None

    def _reload(self):
        try:
            if self._orig_filename and os.path.exists(self._orig_filename):
                self.loader.filename.value = self._orig_filename
                if hasattr(self.ogl, 'src'):
                    self.ogl.src.value = '@surf'
                print('[VisualWatchdog] Surface reloaded')
        except Exception:
            pass

    def onAnimateBeginEvent(self, *_):
        self._step += 1
        if (self._step % self.k) != 0:
            return
        try:
            count = len(self.ogl.position.value) if hasattr(self.ogl, 'position') else 0
            if count == 0:
                self._reload()
        except Exception:
            pass
