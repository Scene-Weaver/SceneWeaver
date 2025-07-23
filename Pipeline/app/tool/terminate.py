import os

from app.tool.base import BaseTool
from app.tool.update_infinigen import update_infinigen

_TERMINATE_DESCRIPTION = """Terminate the scene synthesis when the request is met OR if the assistant cannot proceed further with the task.
When you have finished all the tasks, call this tool to end the work."""


class Terminate(BaseTool):
    name: str = "terminate"
    description: str = _TERMINATE_DESCRIPTION
    parameters: dict = {
        "type": "object",
        "properties": {
            "status": {
                "type": "string",
                "description": "The finish status of the interaction.",
                "enum": ["success", "failure"],
            }
        },
        "required": ["status"],
    }

    def execute(self, status: str) -> str:
        """Finish the current execution"""
        iter = int(os.getenv("iter"))
        try:
            success = update_infinigen("finalize_scene", iter, "")
            assert success
            return "Successfully terminate."
        except Exception:
            return "Error terminate"
