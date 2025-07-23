import json
import os
from typing import Any, List, Optional

import dill

from app.config import config
from app.evaluation import eval_scene
from app.exceptions import TokenLimitExceeded
from app.llm import LLM
from app.logger import logger
from app.prompt.scenedesigner_multistep import NEXT_STEP_PROMPT, SYSTEM_PROMPT
from app.prompt.sceneinfo import sceneinfo_prompt
from app.schema import (
    ROLE_TYPE,
    TOOL_CHOICE_TYPE,
    AgentState,
    Memory,
    Message,
    ToolCall,
    ToolChoice,
)
from app.tool.add_gpt import AddGPTExecute
from app.tool.add_relation import AddRelationExecute
from app.tool.init_gpt import InitGPTExecute
from app.tool.remove_obj import RemoveExecute
from app.tool.terminate import Terminate
from app.tool.tool_collection import ToolCollection
from app.tool.update_layout import UpdateLayoutExecute
from app.tool.update_rotation import UpdateRotationExecute
from app.tool.update_size import UpdateSizeExecute
from app.utils import dict2str, encode_image, lst2str


class SceneDesigner:
    """
    A versatile general-purpose agent that uses planning to solve various tasks.

    This agent extends BrowserAgent with a comprehensive set of tools and capabilities,
    including Python execution, web browsing, file operations, and information retrieval
    to handle a wide range of user requests.
    """

    name: str = "SceneDesigner"
    # description: str = (
    #     "A versatile agent that can solve various tasks using multiple tools"
    # )

    system_prompt: str = SYSTEM_PROMPT.format(directory=config.workspace_root)
    next_step_prompt: str = NEXT_STEP_PROMPT
    tool_choices: TOOL_CHOICE_TYPE = ToolChoice.AUTO  # type: ignore
    # llm: Optional[LLM] = Field(default_factory=LLM)
    llm = LLM()
    # memory: Memory = Field(default_factory=Memory, description="Agent's memory store")
    special_tool_names = Terminate().name

    max_observe: int = 10000
    max_steps: int = 20
    duplicate_threshold: int = 2

    available_tools = ToolCollection(
        InitGPTExecute(),  # InitMetaSceneExecute(), InitPhySceneExecute(),
        # AddAcdcExecute(),
        AddGPTExecute(),
        AddRelationExecute(),
        UpdateLayoutExecute(),
        UpdateRotationExecute(),
        UpdateSizeExecute(),
        Terminate(),
        RemoveExecute(),
    )

    current_step: int = 0
    memory = Memory()
    state = AgentState.IDLE
    # @model_validator(mode="after")
    # def initialize_agent(self) :
    #     """Initialize agent with default settings if not provided."""
    #     if self.llm is None or not isinstance(self.llm, LLM):
    #         self.llm = LLM(config_name=self.name.lower())
    #     if not isinstance(self.memory, Memory):
    #         self.memory = Memory()
    #     return self

    @property
    def messages(self) -> List[Message]:
        """Retrieve a list of messages from the agent's memory."""
        return self.memory.messages

    def step(self) -> str:
        if self.current_step != 0:
            eval_results = self.eval(iter=self.current_step - 1)
            isvalid = self.check_valid(self.current_step - 1)
            if not isvalid:
                save_dir = os.getenv("save_dir")
                iter = self.current_step - 1
                os.system(
                    f"cp {save_dir}/record_scene/render_{iter}_marked.jpg {save_dir}/record_scene/render_{iter}_marked_failed.jpg"
                )
                os.system(
                    f"cp {save_dir}/record_scene/render_{iter}.jpg {save_dir}/record_scene/render_{iter}_failed.jpg"
                )
                os.system(
                    f"cp {save_dir}/record_files/metric_{iter}.json {save_dir}/record_files/metric_{iter}_failed.json"
                )
                os.system(
                    f"cp {save_dir}/record_files/scene_{iter}.blend {save_dir}/record_files/scene_{iter}_failed.blend"
                )
                os.system(
                    f"cp {save_dir}/record_files/env_{iter}.pkl {save_dir}/record_files/env_{iter}_failed.pkl"
                )
                os.system(
                    f"cp {save_dir}/record_files/house_bbox_{iter}.pkl {save_dir}/record_files/house_bbox_{iter}_failed.pkl"
                )
                os.system(
                    f"cp {save_dir}/record_files/p_{iter}.pkl {save_dir}/record_files/p_{iter}_failed.pkl"
                )
                os.system(
                    f"cp {save_dir}/record_files/solved_bbox_{iter}.pkl {save_dir}/record_files/solved_bbox_{iter}_failed.pkl"
                )
                os.system(
                    f"cp {save_dir}/record_files/solver_{iter}.pkl {save_dir}/record_files/solver_{iter}_failed.pkl"
                )
                os.system(
                    f"cp {save_dir}/record_files/state_{iter}.pkl {save_dir}/record_files/state_{iter}_failed.pkl"
                )
                os.system(
                    f"cp {save_dir}/record_files/terrain_{iter}.pkl {save_dir}/record_files/terrain_{iter}_failed.pkl"
                )
                os.system(
                    f"cp {save_dir}/pipeline/metric_{iter}.json {save_dir}/pipeline/metric_{iter}_failed.json"
                )
                return "Failed"

        """Execute a single step: think and act."""
        retry = 0
        while True and retry < 5:
            should_act = self.think()
            if self.tool_calls != []:
                break
            retry += 1

        if not should_act:
            return "Thinking complete - no action needed"

        act_results = self.act()

        if (
            self.current_step == self.max_steps - 1
            or self.tool_calls[0].function.name == "terminate"
        ):  # evaluate final step
            eval_results = self.eval(iter=self.current_step)

        # if self.memory.messages[-1].name!="terminate":
        #     eval_results = self.eval(self.current_step)
        return act_results

    def check_valid(self, iter):
        save_dir = os.getenv("save_dir")
        json_name = f"{save_dir}/pipeline/metric_{iter}.json"
        with open(json_name, "r") as f:
            grades_new = json.load(f)
            score_new = [
                v["grade"]
                for k, v in grades_new["GPT score (0-10, higher is better)"].items()
            ]
            score_new = sum(score_new)

        if iter == 0:
            if score_new >= 20:
                return True
            else:
                return False
        else:
            json_name = f"{save_dir}/pipeline/metric_{iter-1}.json"
            with open(json_name, "r") as f:
                grades_old = json.load(f)
                score_old = [
                    v["grade"]
                    for k, v in grades_old["GPT score (0-10, higher is better)"].items()
                ]
                score_old = sum(score_old)

            if score_old - score_new >= 5:
                return False
            else:
                return True

    def eval(self, iter):
        user_demand = os.getenv("UserDemand")
        # iter = int(os.getenv("iter"))
        grades = eval_scene(iter, user_demand)
        save_dir = os.getenv("save_dir")
        json_name = f"{save_dir}/pipeline/metric_{iter}.json"
        with open(json_name, "r") as f:
            grades = json.load(f)

        result = dict2str(grades)
        result = "The evaluated reults of the current scene is : \n" + result
        if self.max_observe:
            result = result[: self.max_observe]

        logger.info(f"🎯 Evaluation Results: '{result}'")

        # Add tool response to memory
        user_msg = Message.user_message(result)
        self.memory.add_message(user_msg)

        return result

    def load_sceneinfo(self):
        save_dir = os.getenv("save_dir")
        image_path = f"{save_dir}/record_scene/render_{self.current_step-1}_marked.jpg"
        with open(
            f"{save_dir}/record_scene/layout_{self.current_step-1}.json", "r"
        ) as f:
            layout = json.load(f)
        roomsize = layout["roomsize"]
        roomsize = lst2str(roomsize)
        structure = dict2str(layout["structure"])
        layout = dict2str(layout["objects"])

        prompt = sceneinfo_prompt.format(
            roomtype=os.getenv("roomtype"),
            roomsize=roomsize,
            layout=layout,
            structure=structure,
        )

        base64_image = encode_image(image_path)
        return prompt, base64_image

    def think(self) -> bool:
        """Process current state and decide next actions using tools"""
        if self.current_step > 0:
            sceneinfo_prompt, base64_image = self.load_sceneinfo()
            user_msg = Message.user_message(sceneinfo_prompt, base64_image=base64_image)
            self.messages.append(user_msg)

        # if self.next_step_prompt:
        #     user_msg = Message.user_message(self.next_step_prompt)
        #     self.messages.append(user_msg)

        # ideas_lst = {
        #     0:"Initialize foundational kitchen layout.",
        #     1: "Add small appliances and countertop accessories.",
        #     2: "Add dining table and chairs if space allows.",
        #     3:"Add wall-mounted shelves/cabinets.",
        #     4:"Add small objects inside/on storage.",
        #     5:"Refine object relations for tidiness and logic.",
        #     6:"End the scene synthesis."
        # }

        # ideas_lst = {
        #             1:"Initialize the foundational kitchen layout using a data-driven method to ensure structural plausibility and typical kitchen organization.",
        #             2:"Add essential kitchen appliances and fixtures if not already present (e.g., refrigerator, stove/oven, sink, dishwasher).",
        #             3:"Add supporting furniture such as a kitchen island or dining table with chairs, if space allows.",
        #             4:"Add storage elements like wall-mounted cabinets or shelves if not already present.",
        #             5:"Place small kitchen appliances (e.g., microwave, toaster, coffee maker) on available countertop space.",
        #             6:"Add a few small objects on countertops and inside open shelves/cabinets (e.g., dishes, pots, utensils, spice jars).",
        #             7:"Add functional wall-mounted objects (e.g., clock, small artwork, or a wall-mounted spice rack).",
        #             8:"Adjust the placement of large objects to ensure there is enough circulation space and that work triangles (sink, stove, refrigerator) are logically arranged.",
        #             9:"Refine object rotations so that chairs face the table, appliances are oriented correctly, and items on countertops are aligned naturally.",
        #             10:"Terminate the scene synthesis if the kitchen is neither crowded nor empty and all functional and spatial requirements are met."
        # }
        ideas_lst = {
            1: "Initialize the Scene (init_gpt)",
            2: "Add Relations (add_relation)",
            3: "Add Large Supporting Objects (add_gpt)",
            4: "Add any missing large supportingobjects such as a kitchen island, pantry cabinet, or bar stools, if space allows, to enhance functionality without crowding.",
            4: "Add Small Functional Objects (add_gpt)",
            5: "Add Objects Inside Cabinets/Shelves (add_gpt)",
            6: "Add Wall-Mounted Functional/Decorative Objects (add_gpt)",
            7: "Adjust Layout if Needed (update_layout)",
            8: "Adjust Rotations if Needed (update_rotation)",
            9: "Adjust Sizes if Needed (update_size)",
            10: "Remove Redundant or Crowded Objects (remove_object)",
            11: "Terminate (terminate)",
        }

        idea_this_step = ideas_lst[self.current_step + 1]
        user_msg = Message.user_message(idea_this_step)
        self.messages.append(user_msg)
        # scene0149_00
        retry = 0
        while True and retry < 3:
            try:
                if len(self.messages) > 7:
                    messages = [self.messages[0]] + self.messages[-3:]
                else:
                    messages = self.messages
                # Get response with tool options
                response = self.llm.ask_tool(
                    messages=messages,
                    system_msgs=(
                        [Message.system_message(self.system_prompt)]
                        if self.system_prompt
                        else None
                    ),
                    tools=self.available_tools.to_params(),
                    tool_choice=self.tool_choices,
                )
                self.tool_calls = tool_calls = (
                    response.tool_calls if response and response.tool_calls else []
                )
                if self.tool_calls == []:
                    retry += 1
                else:
                    if self.current_step > 0:
                        del self.messages[-2]
                    break

            except ValueError:
                raise
            except Exception as e:
                # Check if this is a RetryError containing TokenLimitExceeded
                if hasattr(e, "__cause__") and isinstance(
                    e.__cause__, TokenLimitExceeded
                ):
                    token_limit_error = e.__cause__
                    logger.error(
                        f"🚨 Token limit error (from RetryError): {token_limit_error}"
                    )
                    self.memory.add_message(
                        Message.assistant_message(
                            f"Maximum token limit reached, cannot continue execution: {str(token_limit_error)}"
                        )
                    )
                    self.state = AgentState.FINISHED
                    return False
                raise

        content = response.content if response and response.content else ""

        # Log response info
        logger.info(f"✨ {self.name}'s thoughts: {content}")
        logger.info(
            f"🛠️ {self.name} selected {len(tool_calls) if tool_calls else 0} tools to use"
        )
        if tool_calls:
            logger.info(
                f"🧰 Tools being prepared: {[call.function.name for call in tool_calls]}"
            )
            logger.info(f"🔧 Tool arguments: {tool_calls[0].function.arguments}")

        try:
            if response is None:
                raise RuntimeError("No response received from the LLM")

            # Handle different tool_choices modes
            if self.tool_choices == ToolChoice.NONE:
                if tool_calls:
                    logger.warning(
                        f"🤔 Hmm, {self.name} tried to use tools when they weren't available!"
                    )
                if content:
                    self.memory.add_message(Message.assistant_message(content))
                    return True
                return False

            # Create and add assistant message
            assistant_msg = (
                Message.from_tool_calls(content=content, tool_calls=self.tool_calls)
                if self.tool_calls
                else Message.assistant_message(content)
            )
            self.memory.add_message(assistant_msg)

            if self.tool_choices == ToolChoice.REQUIRED and not self.tool_calls:
                return True  # Will be handled in act()

            # For 'auto' mode, continue with content if no commands but content exists
            if self.tool_choices == ToolChoice.AUTO and not self.tool_calls:
                return bool(content)

            return bool(self.tool_calls)
        except Exception as e:
            logger.error(f"🚨 Oops! The {self.name}'s thinking process hit a snag: {e}")
            self.memory.add_message(
                Message.assistant_message(
                    f"Error encountered while processing: {str(e)}"
                )
            )
            return False

    def act(self) -> str:
        """Execute tool calls and handle their results"""
        if not self.tool_calls:
            if self.tool_choices == ToolChoice.REQUIRED:
                TOOL_CALL_REQUIRED = "Tool calls required but none provided"
                raise ValueError(TOOL_CALL_REQUIRED)

            # Return last message content if no tool calls
            return self.messages[-1].content or "No content or commands to execute"

        results = []
        for command in self.tool_calls:
            # Reset base64_image for each tool call

            self._current_base64_image = None

            result = self.execute_tool(command)

            if self.max_observe:
                result = result[: self.max_observe]

            logger.info(
                f"🎯 Tool '{command.function.name}' completed its mission! Result: {result}"
            )

            # Add tool response to memory
            tool_msg = Message.tool_message(
                content=result,
                tool_call_id=command.id,
                name=command.function.name,
                base64_image=self._current_base64_image,
            )
            self.memory.add_message(tool_msg)
            results.append(result)

        return "\n\n".join(results)

    def execute_tool(self, command: ToolCall) -> str:
        """Execute a single tool call with robust error handling"""
        if not command or not command.function or not command.function.name:
            return "Error: Invalid command format"

        name = command.function.name
        if name not in self.available_tools.tool_map:
            return f"Error: Unknown tool '{name}'"

        try:
            # Parse arguments

            args = json.loads(command.function.arguments or "{}")

            # Execute the tool
            logger.info(f"🔧 Activating tool: '{name}'...")
            result = self.available_tools.execute(name=name, tool_input=args)

            # Handle special tools
            self._handle_special_tool(name=name, result=result)

            # # Check if result is a ToolResult with base64_image
            # if hasattr(result, "base64_image") and result.base64_image:
            #     # Store the base64_image for later use in tool_message
            #     basedir = "~/workspace/SceneWeaver/record_scene"
            #     self._current_base64_image = f"{basedir}/render_{self.current_step}_marked.jpg"

            # # Format result for display
            # observation = (
            #     f"Observed output of cmd `{name}` executed:\n{str(result)}"
            #     if result
            #     else f"Cmd `{name}` completed with no output"
            # )
            # return observation

            # Format result for display (standard case)
            observation = (
                f"Observed output of cmd `{name}` executed:\n{str(result)}"
                if result
                else f"Cmd `{name}` completed with no output"
            )

            return observation
        except json.JSONDecodeError:
            error_msg = f"Error parsing arguments for {name}: Invalid JSON format"
            logger.error(
                f"📝 Oops! The arguments for '{name}' don't make sense - invalid JSON, arguments:{command.function.arguments}"
            )
            return f"Error: {error_msg}"
        except Exception as e:
            error_msg = f"⚠️ Tool '{name}' encountered a problem: {str(e)}"
            logger.exception(error_msg)
            return f"Error: {error_msg}"

    def _handle_special_tool(self, name: str, result: Any, **kwargs):
        """Handle special tool execution and state changes"""
        if not self._is_special_tool(name):
            return

        if self._should_finish_execution(name=name, result=result, **kwargs):
            # Set agent state to finished
            logger.info(f"🏁 Special tool '{name}' has completed the task!")
            self.state = AgentState.FINISHED

    @staticmethod
    def _should_finish_execution(**kwargs) -> bool:
        """Determine if tool execution should finish the agent"""
        return True

    def _is_special_tool(self, name: str) -> bool:
        """Check if tool name is in special tools list"""

        return name.lower() in [n.lower() for n in self.special_tool_names]

    def run(self, request: Optional[str] = None) -> str:
        """Execute the agent's main loop asynchronously.

        Args:
            request: Optional initial user request to process.

        Returns:
            A string summarizing the execution results.

        Raises:
            RuntimeError: If the agent is not in IDLE state at start.
        """
        # if self.state != AgentState.IDLE:
        #     raise RuntimeError(f"Cannot run agent from state: {self.state}")

        if request:
            self.update_memory("user", request)

        results: List[str] = []

        self.current_step = 6
        save_dir = os.getenv("save_dir")


        while self.current_step < self.max_steps and self.state != AgentState.FINISHED:
            if self.current_step > 0:
                with open(
                    f"{save_dir}/pipeline/memory_{self.current_step-1}.pkl", "rb"
                ) as file:
                    self.memory = dill.load(file)

                with open(f"{save_dir}/pipeline/roomtype.txt", "r") as f:
                    roomtype = f.readline().strip()
                    os.environ["roomtype"] = roomtype

            os.environ["iter"] = str(self.current_step)

            logger.info(
                f"Executing step {self.current_step}/{self.max_steps} for {save_dir}"
            )
            step_result = self.step()
            if step_result == "Failed":
                self.current_step -= 1
                continue

            # Check for stuck state
            if self.is_stuck():
                self.handle_stuck_state()
            results.append(f"Step {self.current_step}: {step_result}")

            with open(
                f"{save_dir}/pipeline/memory_{self.current_step}.pkl", "wb"
            ) as file:
                dill.dump(self.memory, file)
            roomtype = os.getenv("roomtype")
            with open(f"{save_dir}/pipeline/roomtype.txt", "w") as f:
                f.write(roomtype)

            self.current_step += 1
            if self.tool_calls[0].function.name == "terminate":
                self.state = AgentState.FINISHED
                results.append("Terminated: successfullly stop.")

        if self.current_step >= self.max_steps:
            self.current_step = 0
            self.state = AgentState.IDLE
            results.append(f"Terminated: Reached max steps ({self.max_steps})")

        return "\n".join(results) if results else "No steps executed"

    def handle_stuck_state(self):
        """Handle stuck state by adding a prompt to change strategy"""
        stuck_prompt = "\
        Observed duplicate responses. Consider new strategies and avoid repeating ineffective paths already attempted."
        self.next_step_prompt = f"{stuck_prompt}\n{self.next_step_prompt}"
        logger.warning(f"Agent detected stuck state. Added prompt: {stuck_prompt}")

    def is_stuck(self) -> bool:
        """Check if the agent is stuck in a loop by detecting duplicate content"""
        if len(self.memory.messages) < 2:
            return False

        last_message = self.memory.messages[-1]
        if not last_message.content:
            return False

        # Count identical content occurrences
        duplicate_count = sum(
            1
            for msg in reversed(self.memory.messages[:-1])
            if msg.role == "assistant" and msg.content == last_message.content
        )

        return duplicate_count >= self.duplicate_threshold

    def update_memory(
        self,
        role: ROLE_TYPE,  # type: ignore
        content: str,
        base64_image: Optional[str] = None,
        **kwargs,
    ) -> None:
        """Add a message to the agent's memory.

        Args:
            role: The role of the message sender (user, system, assistant, tool).
            content: The message content.
            base64_image: Optional base64 encoded image.
            **kwargs: Additional arguments (e.g., tool_call_id for tool messages).

        Raises:
            ValueError: If the role is unsupported.
        """
        message_map = {
            "user": Message.user_message,
            "system": Message.system_message,
            "assistant": Message.assistant_message,
            "tool": lambda content, **kw: Message.tool_message(content, **kw),
        }

        if role not in message_map:
            raise ValueError(f"Unsupported message role: {role}")

        # Create message with appropriate parameters based on role
        kwargs = {"base64_image": base64_image, **(kwargs if role == "tool" else {})}
        self.memory.add_message(message_map[role](content, **kwargs))
