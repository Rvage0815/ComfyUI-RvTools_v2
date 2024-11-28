from ..core import CATEGORY

SCHEDULERS_RESTART = ('normal', 'karras', 'exponential', 'sgm_uniform', 'simple', 'ddim_uniform', 'beta', 'simple_test')

class RvPasser_Scheduler_Restart:
    def __init__(self):
        pass

    @classmethod
    def INPUT_TYPES(cls):
        return {"required": {"scheduler": (SCHEDULERS_RESTART, {"forceInput": True}),}}

    CATEGORY = CATEGORY.MAIN.value + CATEGORY.PASSER.value
    RETURN_TYPES = (SCHEDULERS_RESTART,)
    RETURN_NAMES = ("scheduler",)

    FUNCTION = "passthrough"

    def passthrough(self, scheduler):
        return (scheduler,)    

NODE_NAME = 'Pass Scheduler (Restart) [RvTools]'
NODE_DESC = 'Pass Scheduler (Restart)'

NODE_CLASS_MAPPINGS = {
   NODE_NAME: RvPasser_Scheduler_Restart
}

NODE_DISPLAY_NAME_MAPPINGS = {
    NODE_NAME: NODE_DESC
}
