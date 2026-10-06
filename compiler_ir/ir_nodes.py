"""The control-flow IR object model.

The IR is intentionally tiny and directly traversable::

    Module
      Function            (signature, local slots, basic blocks)
        Block             (phi nodes, ordered instructions, one terminator)
          Phi             (SSA only; one incoming value per predecessor)
          Instruction     (Const / Copy / BinOp / Call, all write a dest)
          Terminator      (Return / Jump / Branch)

Two flavors of module exist:

* non-SSA (produced by :func:`lower_module`): value references are
  :class:`Temp` -- numbered temporaries produced by instructions -- and
  :class:`Slot`, numbered mutable local storage (parameters and ``let``
  variables).  Temporaries are mostly single assignment; the sole exception
  is the result temporary of a short-circuit boolean operation, which is
  written once on each participating path, as is natural in non-SSA form.
* SSA (produced by :func:`to_ssa`): every parameter and every instruction
  result is a unique definition, slots never appear, and control-flow joins
  carry :class:`Phi` nodes listed before the block's ordinary instructions.

Both flavors reuse the same node classes; a module's ``ssa`` flag (and each
function's) says which one it is.
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


@dataclass(eq=False)
class Phi:
    """An SSA phi function: merges definitions from predecessor blocks.

    ``entries`` maps a predecessor :class:`Block` to the value that is live
    on that edge.  Entries are kept sorted by predecessor label whenever a
    module is emitted by :func:`to_ssa`; only reachable predecessors occur.
    """

    dest: Temp
    entries: dict["Block", ValueRef] = field(default_factory=dict)

    @property
    def op(self) -> str:
        return "phi"

    @property
    def type(self) -> str:
        return self.dest.type


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


@dataclass(eq=False)
class Block:
    id: int
    instructions: list[Instruction] = field(default_factory=list)
    terminator: Optional[Terminator] = None
    # SSA only; empty in non-SSA blocks.  Phi nodes precede the ordinary
    # instructions.
    phis: list[Phi] = field(default_factory=list)

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
    # SSA only: the unique parameter definition.  None in non-SSA functions,
    # where the parameter is read through ``slot``.
    temp: Optional[Temp] = None


@dataclass
class Function:
    name: str
    params: list[Parameter]
    ret_type: str
    locals: list[Slot]
    blocks: list[Block]
    entry: Block
    ssa: bool = False

    def is_void(self) -> bool:
        return self.ret_type == "void"


@dataclass
class Module:
    functions: list[Function]
    ssa: bool = False
