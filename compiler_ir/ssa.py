"""SSA conversion for the non-SSA control-flow IR.

:func:`to_ssa` takes the ``Module`` produced by ``lower_module`` and returns
a *new* ``Module`` in static single-assignment form; the input module is
left unchanged.  In the result:

* every parameter and every instruction result is a unique SSA definition
  (a fresh ``Temp``), so all ``Slot`` reads and writes disappear -- a slot
  write becomes pure renaming, a slot read becomes the value currently
  reaching it;
* the multi-path writes to the shared result temporary of a short-circuit
  ``and``/``or`` are split into one definition per path plus a ``Phi`` at
  the merge block;
* a ``Phi`` is materialised at a control-flow join only when a value with
  several reachable definitions is actually observed afterwards; phis
  whose incoming values all agree, or whose result is never used, are
  removed again;
* phi incoming edges are recorded in ascending predecessor-label order and
  unreachable predecessors never contribute an edge.

SSA value numbers are assigned per function from zero, determined solely
by parameter order, block order and the order of phis and instructions
within each block -- never by set iteration or object identity -- so
converting and rendering the same input twice is byte-identical, and
converting an already-SSA module is a fixed point.
"""
from .ir_nodes import (
    BinOp,
    Block,
    Branch,
    Call,
    Const,
    Copy,
    Function,
    Jump,
    Module,
    Parameter,
    Phi,
    Return,
    Slot,
    Temp,
)

__all__ = ["to_ssa"]


def to_ssa(module: Module) -> Module:
    """Convert a lowered non-SSA ``Module`` to a new SSA ``Module``.

    The input module is not modified.  Only modules produced by
    ``lower_module`` (or by an earlier ``to_ssa`` call) are supported;
    anything that is not a ``Module`` raises ``TypeError``.
    """
    if not isinstance(module, Module):
        raise TypeError(
            f"to_ssa() expects a Module, got {type(module).__name__}"
        )
    return Module([_FunctionConverter(func).convert() for func in module.functions])


class _FunctionConverter:
    """Convert one non-SSA function to SSA form.

    The construction follows Braun et al., "Simple and Efficient
    Construction of Static Single Assignment Form": blocks are visited in
    their existing order -- which the lowerer guarantees is a DFS order in
    which every predecessor except a loop back edge comes first -- and the
    value reaching a use is looked up recursively through predecessors,
    materialising phi nodes at joins on demand.  Loop headers are handled
    by sealing: a block is sealed once all its predecessors have been
    processed, at which point its incomplete phis are filled in.
    """

    def __init__(self, func: Function):
        self.func = func
        self.old_blocks = list(func.blocks)
        self.new_blocks = [Block(b.id) for b in self.old_blocks]
        self.new_by_id = {b.id: b for b in self.new_blocks}
        self.preds = {b.id: [] for b in self.old_blocks}
        self.succs = {b.id: [] for b in self.old_blocks}
        # var (Slot or multiply-defined Temp) -> {block id: reaching Temp}.
        self.current_def: dict = {}
        # Block id -> [(var, phi)] for phis awaiting their back-edge values.
        self.incomplete_phis: dict = {}
        self.processed: set = set()
        self.sealed: set = set()
        # Singly-defined Temp of the input -> its SSA renaming.
        self.value_map: dict = {}
        # Temps written on more than one path (short-circuit results).
        self.multi_def: set = set()
        self._next_temp = 0

    # -- small helpers ------------------------------------------------------

    def _fresh(self, typ: str) -> Temp:
        # Placeholder numbering; a final pass renumbers everything in
        # parameter/block/instruction order.
        temp = Temp(self._next_temp, typ)
        self._next_temp += 1
        return temp

    def write_variable(self, var, bid: int, value) -> None:
        self.current_def.setdefault(var, {})[bid] = value

    # -- structural analysis --------------------------------------------------

    @staticmethod
    def _targets(term) -> list:
        if isinstance(term, Jump):
            return [term.target]
        if isinstance(term, Branch):
            return [term.true_target, term.false_target]
        return []

    def _build_cfg(self) -> None:
        for block in self.old_blocks:
            for target in self._targets(block.terminator):
                self.succs[block.id].append(target.id)
                self.preds[target.id].append(block.id)
        for bid in self.preds:
            self.preds[bid].sort()

    def _reachable(self) -> set:
        seen = {self.func.entry.id}
        stack = [self.func.entry.id]
        while stack:
            bid = stack.pop()
            for succ in self.succs[bid]:
                if succ not in seen:
                    seen.add(succ)
                    stack.append(succ)
        return seen

    def _analyse_defs(self) -> None:
        counts: dict = {}
        for block in self.old_blocks:
            for phi in block.phis:
                counts[phi.dest] = counts.get(phi.dest, 0) + 1
            for ins in block.instructions:
                if isinstance(ins.dest, Temp):
                    counts[ins.dest] = counts.get(ins.dest, 0) + 1
        self.multi_def = {t for t, n in counts.items() if n > 1}
        # Pre-assign renamings for all singly-defined temporaries so that
        # uses are resolved independently of processing order (an existing
        # phi may refer to a value defined in a later block).
        for block in self.old_blocks:
            for phi in block.phis:
                self.value_map[phi.dest] = self._fresh(phi.dest.type)
            for ins in block.instructions:
                dest = ins.dest
                if isinstance(dest, Temp) and dest not in self.multi_def:
                    self.value_map[dest] = self._fresh(dest.type)

    def _convert_params(self) -> list:
        params = []
        entry = self.func.entry.id
        for param in self.func.params:
            temp = self._fresh(param.slot.type)
            if isinstance(param.slot, Slot):
                # A parameter is the initial definition of its slot.
                self.write_variable(param.slot, entry, temp)
            else:
                # Already an SSA parameter (idempotent re-conversion).
                self.value_map[param.slot] = temp
            params.append(Parameter(param.name, temp))
        return params

    # -- variable resolution ----------------------------------------------------

    def read_value(self, ref, bid: int):
        """Resolve an operand of the old module to its SSA value."""
        if isinstance(ref, Slot):
            return self.read_variable(ref, bid)
        if isinstance(ref, Temp) and ref in self.multi_def:
            return self.read_variable(ref, bid)
        return self.value_map[ref]

    def read_variable(self, var, bid: int):
        defs = self.current_def.get(var)
        if defs is not None and bid in defs:
            return defs[bid]
        value = self._read_variable_recursive(var, bid)
        self.current_def.setdefault(var, {})[bid] = value
        return value

    def _read_variable_recursive(self, var, bid: int):
        if bid not in self.sealed:
            # A loop header whose back edge has not been processed yet:
            # record an incomplete phi, to be filled when the block seals.
            phi = Phi(self._fresh(var.type), [], var.type)
            self.new_by_id[bid].phis.append(phi)
            self.incomplete_phis.setdefault(bid, []).append((var, phi))
            value = phi.dest
        else:
            preds = self.preds[bid]
            if not preds:
                raise ValueError(
                    f"use of uninitialised variable {var} in "
                    f"function {self.func.name!r}"
                )
            if len(preds) == 1:
                value = self.read_variable(var, preds[0])
            else:
                phi = Phi(self._fresh(var.type), [], var.type)
                self.new_by_id[bid].phis.append(phi)
                # Publish the phi before filling operands to break cycles.
                self.current_def.setdefault(var, {})[bid] = phi.dest
                for pred in preds:
                    phi.incoming.append(
                        (self.new_by_id[pred], self.read_variable(var, pred))
                    )
                value = self._try_remove_trivial_phi(phi, bid)
        self.current_def.setdefault(var, {})[bid] = value
        return value

    def _seal(self, bid: int) -> None:
        self.sealed.add(bid)
        for var, phi in self.incomplete_phis.pop(bid, []):
            if phi not in self.new_by_id[bid].phis:
                continue  # already removed as trivial
            for pred in self.preds[bid]:
                phi.incoming.append(
                    (self.new_by_id[pred], self.read_variable(var, pred))
                )
            self._try_remove_trivial_phi(phi, bid)

    def _try_remove_trivial_phi(self, phi, bid: int):
        """Replace a phi whose reachable in-edges all carry the same value."""
        same = None
        for _, value in phi.incoming:
            if value == phi.dest:
                continue  # self-reference through a back edge
            if same is None:
                same = value
            elif value != same:
                return phi.dest  # genuinely different definitions: keep
        if same is None:
            return phi.dest
        self.new_by_id[bid].phis.remove(phi)
        users = self._replace_uses(phi.dest, same)
        seen = set()
        for user_phi, user_bid in users:
            if id(user_phi) in seen:
                continue
            seen.add(id(user_phi))
            # Only complete phis (all operands known) may be re-examined;
            # incomplete ones are checked when their block is sealed.
            if (
                user_bid in self.sealed
                and user_phi in self.new_by_id[user_bid].phis
            ):
                self._try_remove_trivial_phi(user_phi, user_bid)
        return same

    def _replace_uses(self, old, new) -> list:
        """Rewrite every use of ``old`` to ``new``; return the phis touched."""
        users = []
        for block in self.new_blocks:
            for phi in block.phis:
                if any(value == old for _, value in phi.incoming):
                    phi.incoming = [
                        (pred, new if value == old else value)
                        for pred, value in phi.incoming
                    ]
                    users.append((phi, block.id))
            for ins in block.instructions:
                if isinstance(ins, Copy):
                    if ins.src == old:
                        ins.src = new
                elif isinstance(ins, BinOp):
                    if ins.left == old:
                        ins.left = new
                    if ins.right == old:
                        ins.right = new
                elif isinstance(ins, Call):
                    ins.args = [new if a == old else a for a in ins.args]
            term = block.terminator
            if isinstance(term, Branch):
                if term.cond == old:
                    term.cond = new
            elif isinstance(term, Return):
                if term.value is not None and term.value == old:
                    term.value = new
        for defs in self.current_def.values():
            for bid, value in defs.items():
                if value == old:
                    defs[bid] = new
        return users

    # -- instruction and terminator conversion ---------------------------------

    def _temp_dest(self, dest, bid: int):
        if dest in self.multi_def:
            # One SSA definition per path for a multiply-written temporary.
            temp = self._fresh(dest.type)
            self.write_variable(dest, bid, temp)
            return temp
        return self.value_map[dest]

    def _convert_instruction(self, ins, bid: int, new_block) -> None:
        if isinstance(ins, Const):
            dest = self._temp_dest(ins.dest, bid)
            new_block.instructions.append(Const(dest, ins.value))
        elif isinstance(ins, Copy):
            src = self.read_value(ins.src, bid)
            if isinstance(ins.dest, Slot):
                # A slot write becomes pure renaming; no IR is emitted.
                self.write_variable(ins.dest, bid, src)
                return
            dest = self._temp_dest(ins.dest, bid)
            new_block.instructions.append(Copy(dest, src))
        elif isinstance(ins, BinOp):
            left = self.read_value(ins.left, bid)
            right = self.read_value(ins.right, bid)
            dest = self._temp_dest(ins.dest, bid)
            new_block.instructions.append(
                BinOp(dest, ins.operator, left, right, ins.kind, ins.type)
            )
        elif isinstance(ins, Call):
            args = [self.read_value(a, bid) for a in ins.args]
            dest = self._temp_dest(ins.dest, bid)
            new_block.instructions.append(Call(dest, ins.name, args, ins.type))
        else:
            raise AssertionError(f"unknown instruction: {ins!r}")

    def _convert_terminator(self, term, bid: int):
        if term is None:
            return None
        if isinstance(term, Return):
            value = None
            if term.value is not None:
                value = self.read_value(term.value, bid)
            return Return(value)
        if isinstance(term, Jump):
            return Jump(self.new_by_id[term.target.id])
        if isinstance(term, Branch):
            cond = self.read_value(term.cond, bid)
            return Branch(
                cond,
                self.new_by_id[term.true_target.id],
                self.new_by_id[term.false_target.id],
            )
        raise AssertionError(f"unknown terminator: {term!r}")

    def _process_block(self, block) -> None:
        bid = block.id
        new_block = self.new_by_id[bid]
        for phi in block.phis:
            # Already-SSA input: keep existing phis, rewriting operands.
            incoming = [
                (self.new_by_id[pred.id], self.read_value(value, bid))
                for pred, value in phi.incoming
            ]
            incoming.sort(key=lambda pair: pair[0].id)
            new_block.phis.append(Phi(self.value_map[phi.dest], incoming, phi.type))
        for ins in block.instructions:
            self._convert_instruction(ins, bid, new_block)
        new_block.terminator = self._convert_terminator(block.terminator, bid)

    # -- dead phi sweep ---------------------------------------------------------

    @staticmethod
    def _instruction_uses(ins) -> list:
        if isinstance(ins, Copy):
            return [ins.src]
        if isinstance(ins, BinOp):
            return [ins.left, ins.right]
        if isinstance(ins, Call):
            return list(ins.args)
        return []

    def _sweep_dead_phis(self) -> None:
        # A phi whose result nobody uses is removed; removals can cascade,
        # so iterate to a fixed point.  Only phis are removable -- every
        # ordinary instruction is kept to preserve side-effect order.
        changed = True
        while changed:
            changed = False
            used = set()
            for block in self.new_blocks:
                for phi in block.phis:
                    for _, value in phi.incoming:
                        used.add(value)
                for ins in block.instructions:
                    used.update(self._instruction_uses(ins))
                term = block.terminator
                if isinstance(term, Branch):
                    used.add(term.cond)
                elif isinstance(term, Return) and term.value is not None:
                    used.add(term.value)
            for block in self.new_blocks:
                kept = [phi for phi in block.phis if phi.dest in used]
                if len(kept) != len(block.phis):
                    block.phis = kept
                    changed = True

    # -- final deterministic numbering -------------------------------------------

    def _renumber(self, params) -> None:
        # SSA value numbers are a pure function of parameter order, block
        # order and intra-block phi/instruction order.
        mapping = {}
        counter = 0

        def assign(temp):
            nonlocal counter
            new = Temp(counter, temp.type)
            mapping[temp] = new
            counter += 1
            return new

        for param in params:
            param.slot = assign(param.slot)
        for block in self.new_blocks:
            for phi in block.phis:
                phi.dest = assign(phi.dest)
            for ins in block.instructions:
                ins.dest = assign(ins.dest)

        def remap(ref):
            return mapping.get(ref, ref)

        for block in self.new_blocks:
            for phi in block.phis:
                phi.incoming = [
                    (pred, remap(value)) for pred, value in phi.incoming
                ]
            for ins in block.instructions:
                if isinstance(ins, Copy):
                    ins.src = remap(ins.src)
                elif isinstance(ins, BinOp):
                    ins.left = remap(ins.left)
                    ins.right = remap(ins.right)
                elif isinstance(ins, Call):
                    ins.args = [remap(a) for a in ins.args]
            term = block.terminator
            if isinstance(term, Branch):
                term.cond = remap(term.cond)
            elif isinstance(term, Return) and term.value is not None:
                term.value = remap(term.value)

    # -- driver -----------------------------------------------------------------

    def convert(self) -> Function:
        self._build_cfg()
        reachable = self._reachable()
        # Unreachable predecessors never contribute a phi in-edge.  (The
        # lowerer does not produce unreachable blocks; this is defensive.)
        for bid in self.preds:
            self.preds[bid] = [p for p in self.preds[bid] if p in reachable]
        self._analyse_defs()
        params = self._convert_params()

        self.sealed.add(self.func.entry.id)
        for block in self.old_blocks:
            if block.id not in reachable:
                continue
            self._process_block(block)
            self.processed.add(block.id)
            for succ in self.succs[block.id]:
                if succ not in self.sealed and all(
                    p in self.processed for p in self.preds[succ]
                ):
                    self._seal(succ)

        self.new_blocks = [b for b in self.new_blocks if b.id in reachable]
        self._sweep_dead_phis()
        self._renumber(params)
        return Function(
            self.func.name,
            params,
            self.func.ret_type,
            [],  # slots are eliminated by the conversion
            self.new_blocks,
            self.new_by_id[self.func.entry.id],
        )
