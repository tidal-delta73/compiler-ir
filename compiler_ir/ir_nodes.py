"""The non-SSA control-flow IR object model.

The IR is intentionally tiny and directly traversable::

    Module
      Function            (signature, local slots, basic blocks)
        Block             (ordered instructions plus one terminator)
          Instruction     (Const / Copy / BinOp / Call, all write a dest)
          Terminator      (Return / Jump / Branch)

Two kinds of value references appear as operands:

* :class:`Temp` -- a numbered temporary produced by an instruction;
* :class:`Slot` -- a numbered, mutable local storage slot (parameters and
  ``let`` variables).

Temporaries are numbered by DFS visitation order and are mostly single
assignment; the sole exception is the result slot of a short-circuit
boolean operation, which is written once on each participating path, as
is natural in non-SSA form.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional, Union


@dataclass(frozen=True)
class Temp:
    id: int
    type: str

    def __str__(self) -> str:
        return f"%t{self.id}"


@dataclass(frozen=True)
class Slot:
    """A mutable local slot (parameter or ``let`` variable)."""

    id: int
    type: str

    def __str__(self) -> str:
        return f"%v{self.id}"


ValueRef = Union[Temp, Slot]


@dataclass
class Const:
    dest: Temp
    value: Union[int, bool]

    @property
    def op(self) -> str:
        return "const"


@dataclass
class Copy:
    dest: Union[Temp, Slot]
    src: ValueRef

    @property
    def op(self) -> str:
        return "copy"


@dataclass
class BinOp:
    dest: Temp
    operator: str
    left: ValueRef
    right: ValueRef
    kind: str  # "arith" or "compare"
    type: str  # result type

    @property
    def op(self) -> str:
        return self.kind


@dataclass
class Call:
    dest: Temp
    name: str
    args: list[ValueRef]
    type: str

    @property
    def op(self) -> str:
        return "call"


Instruction = Union[Const, Copy, BinOp, Call]


@dataclass
class Return:
    value: Optional[ValueRef]

    @property
    def op(self) -> str:
        return "return"


@dataclass
class Jump:
    target: "Block"

    @property
    def op(self) -> str:
        return "jump"


@dataclass
class Branch:
    cond: ValueRef
    true_target: "Block"
    false_target: "Block"

    @property
    def op(self) -> str:
        return "br"


Terminator = Union[Return, Jump, Branch]


@dataclass
class Block:
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
    slot: Slot


@dataclass
class Function:
    name: str
    params: list[Parameter]
    ret_type: str
    locals: list[Slot]
    blocks: list[Block]
    entry: Block

    def is_void(self) -> bool:
        return self.ret_type == "void"


@dataclass
class Module:
    functions: list[Function]
