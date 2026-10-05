"""In-memory non-SSA control-flow IR.

The object graph is plain and traversable:

Module -> Function -> BasicBlock -> Instruction / Terminator
                   \\- Slot (named local storage, including parameters)
Temp (expression results) are owned by their function and referenced from
instructions. Every identifier is an integer assigned in a deterministic
order, never an object address.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional, Union


@dataclass(frozen=True)
class Temp:
    """A temporary value produced by an instruction."""

    id: int
    type: str  # "int" or "bool"

    @property
    def name(self) -> str:
        return f"%t{self.id}"


@dataclass(frozen=True)
class Slot:
    """A named local storage slot (parameters are slots too).

    Inner declarations shadowing an outer name get distinct slots, so the
    slot id is the identity; ``name`` is informational.
    """

    id: int
    name: str
    type: str  # "int" or "bool"

    @property
    def label(self) -> str:
        return f"%s{self.id}"


@dataclass
class Instruction:
    """A non-terminating instruction.

    op is one of:
      const  : dest = value (int/bool literal)
      load   : dest = slot
      store  : slot = operands[0]
      add sub mul div eq ne lt le gt ge : dest = operands[0] op operands[1]
      copy   : dest = operands[0]
      call   : dest = callee(operands)
    """

    op: str
    dest: Optional[Temp]
    operands: list[Temp] = field(default_factory=list)
    slot: Optional[Slot] = None
    callee: Optional[str] = None
    value: Optional[Union[int, bool]] = None


@dataclass
class Terminator:
    """An explicit block terminator.

    op is "return" (value optional), "br" (targets[0]) or
    "cbr" (condition + targets[0] if true, targets[1] if false).
    """

    op: str
    value: Optional[Temp] = None
    condition: Optional[Temp] = None
    targets: list["BasicBlock"] = field(default_factory=list)

    @property
    def is_return(self) -> bool:
        return self.op == "return"


@dataclass
class BasicBlock:
    id: int
    instructions: list[Instruction] = field(default_factory=list)
    terminator: Optional[Terminator] = None

    @property
    def label(self) -> str:
        return f"b{self.id}"

    @property
    def is_terminated(self) -> bool:
        return self.terminator is not None


@dataclass
class Parameter:
    name: str
    type: str


@dataclass
class Function:
    name: str
    params: list[Parameter]
    return_type: str  # "int", "bool" or "void"
    slots: list[Slot] = field(default_factory=list)
    temps: list[Temp] = field(default_factory=list)
    blocks: list[BasicBlock] = field(default_factory=list)

    @property
    def entry(self) -> BasicBlock:
        return self.blocks[0]


@dataclass
class Module:
    functions: list[Function] = field(default_factory=list)
