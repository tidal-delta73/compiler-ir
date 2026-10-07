"""Conversion of the non-SSA control-flow IR into pruned SSA form.

The entry point :func:`to_ssa` takes a :class:`Module` produced by
:func:`lower_module` and returns a brand new, traversable and renderable
SSA :class:`Module`; the input module is never mutated.  Feeding an SSA
module back in yields a structurally and textually equivalent copy (no new
phi nodes, unchanged numbering), and any non-:class:`Module` object raises
:class:`TypeError`.

Algorithm (Cytron et al., with pruning)
--------------------------------------
1. The internal control-flow analysis layer
   (:mod:`compiler_ir.cfg`) supplies reachability, predecessors,
   reverse postorder, the dominator tree and dominance frontiers.
2. Every value definition site (a parameter slot, a slot copy, or a
   possibly multi-written temporary such as a short-circuit result) gets a
   phi at its iterated dominance frontier.
3. A dominator-tree walk renames every definition to a fresh SSA value and
   fills phi entries from the stacks live on each outgoing edge.  Only
   reachable edges are visited, so an unreachable predecessor never appears
   in a phi.
4. Dead (result-unused) and trivial (one distinct incoming value, ignoring
   the phi itself and missing edges) phis are removed to a fixpoint; chains
   and loop self-loops collapse in turn.
5. Values are numbered deterministically per function -- parameters in
   order, then blocks in id order, phis before ordinary instructions -- so
   repeated conversion of the same module is byte-identical and independent
   of set or dict iteration order and object identity.
"""
from .cfg import analyze_function
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
    Temp,
)


def to_ssa(module: Module) -> Module:
    """Return an SSA copy of ``module``.

    The input module is left untouched.  A module already in SSA form is
    copied to an equivalent SSA module (same phi nodes, same numbering).

    :raises TypeError: if ``module`` is not a :class:`Module` instance.
    """
    if not isinstance(module, Module):
        raise TypeError(
            f"to_ssa expects a Module, got {type(module).__name__}"
        )
    if getattr(module, "ssa", False):
        return _clone_ssa_module(module)
    return Module(
        [_Converter(func).convert() for func in module.functions], ssa=True
    )


# --------------------------------------------------------------------------
# Operand helpers (work uniformly on phi entries, instructions, terminators)
# --------------------------------------------------------------------------


def _operands(ins):
    if isinstance(ins, Copy):
        return [ins.src]
    if isinstance(ins, BinOp):
        return [ins.left, ins.right]
    if isinstance(ins, Call):
        return list(ins.args)
    return []


def _map_operands(ins, fn) -> None:
    if isinstance(ins, Copy):
        ins.src = fn(ins.src)
    elif isinstance(ins, BinOp):
        ins.left = fn(ins.left)
        ins.right = fn(ins.right)
    elif isinstance(ins, Call):
        ins.args = [fn(arg) for arg in ins.args]


def _term_operands(term):
    if isinstance(term, Return) and term.value is not None:
        return [term.value]
    if isinstance(term, Branch):
        return [term.cond]
    return []


def _map_term_operands(term, fn) -> None:
    if isinstance(term, Return) and term.value is not None:
        term.value = fn(term.value)
    elif isinstance(term, Branch):
        term.cond = fn(term.cond)


# --------------------------------------------------------------------------
# Per-function conversion
# --------------------------------------------------------------------------


class _Converter:
    def __init__(self, func: Function):
        self.func = func
        self.cfg = analyze_function(func)
        self._fresh_counter = 0

    def fresh(self, typ: str) -> Temp:
        temp = Temp(self._fresh_counter, typ)
        self._fresh_counter += 1
        return temp

    def convert(self) -> Function:
        cfg = self.cfg
        entry = self.func.entry
        frontiers = cfg.frontiers

        new_blocks = [Block(block.id) for block in cfg.blocks]
        block_map = dict(zip(cfg.blocks, new_blocks))

        # -- definition sites, in original DFS definition order ----------
        param_temp: dict = {}
        defs: dict[object, list[Block]] = {}
        ref_order: dict[object, int] = {}

        def register(ref, block: Block) -> None:
            if ref not in ref_order:
                ref_order[ref] = len(ref_order)
            defs.setdefault(ref, []).append(block)

        for param in self.func.params:
            temp = self.fresh(param.slot.type)
            param_temp[param.slot] = temp
            register(param.slot, entry)

        for block in cfg.blocks:
            if block not in cfg.reachable:
                continue
            for ins in block.instructions:
                register(ins.dest, block)

        # -- phi placeholders at iterated dominance frontiers ------------
        phi_origin: dict[Phi, object] = {}
        block_phis: dict[Block, list[Phi]] = {}
        already: dict[Block, set[object]] = {
            block: set() for block in cfg.reachable
        }
        for ref in sorted(ref_order, key=lambda r: ref_order[r]):
            worklist = list(defs[ref])
            while worklist:
                site = worklist.pop()
                for frontier in frontiers.get(site, ()):
                    if ref in already[frontier]:
                        continue
                    already[frontier].add(ref)
                    phi = Phi(self.fresh(ref.type), {})
                    phi_origin[phi] = ref
                    block_phis.setdefault(frontier, []).append(phi)
                    if not any(s is frontier for s in worklist):
                        worklist.append(frontier)

        for block in block_phis:
            block_phis[block].sort(key=lambda phi: ref_order[phi_origin[phi]])

        # -- renaming over the dominator tree ----------------------------
        stacks: dict[object, list[Temp]] = {
            slot: [temp] for slot, temp in param_temp.items()
        }

        # Dominator-tree children come pre-ordered by block index from the
        # shared control-flow analysis.
        children = cfg.children

        def top(ref):
            stack = stacks.get(ref)
            return stack[-1] if stack else None

        loose: dict[object, Temp] = {}

        def use(ref):
            mapped = top(ref)
            if mapped is None:
                # Only reached on edges where the name is genuinely
                # undefined; in lowered IR such edges are unreachable, and
                # the placeholder disappears together with its dead phi.
                mapped = loose.get(ref)
                if mapped is None:
                    mapped = self.fresh(ref.type)
                    loose[ref] = mapped
            return mapped

        def rename(old: Block, new: Block) -> None:
            saved = {ref: len(stack) for ref, stack in stacks.items()}

            for phi in block_phis.get(old, ()):
                stacks.setdefault(phi_origin[phi], []).append(phi.dest)

            for ins in old.instructions:
                if isinstance(ins, Copy):
                    # A copy is pure aliasing in SSA: the defined name now
                    # denotes the renamed source value, no instruction (and
                    # no fresh definition) is emitted.  This is also what
                    # lets branches assigning the same value share one SSA
                    # value and keep their merge phi trivial.
                    stacks.setdefault(ins.dest, []).append(use(ins.src))
                    continue
                cloned = self._clone_ins(ins, use)
                new.instructions.append(cloned)
                stacks.setdefault(ins.dest, []).append(cloned.dest)

            new.terminator = self._clone_terminator(old.terminator,
                                                    block_map, use)

            # Fill the phi of every reachable successor: current block is
            # the predecessor supplying top(ref) on this edge.
            for successor in cfg.succs[old]:
                if successor not in cfg.reachable:
                    continue
                for phi in block_phis.get(successor, ()):
                    phi.entries[new] = use(phi_origin[phi])

            for child in children[old]:
                rename(child, block_map[child])

            for ref, length in saved.items():
                del stacks[ref][length:]

        rename(entry, block_map[entry])

        for old, new in block_map.items():
            new.phis = list(block_phis.get(old, ()))
            if old not in cfg.reachable:
                self._clone_unreachable(old, new, block_map)

        new_params = [
            Parameter(param.name, param.slot, param_temp[param.slot])
            for param in self.func.params
        ]
        self._prune(new_blocks)
        self._renumber(new_params, new_blocks)

        return Function(
            self.func.name,
            new_params,
            self.func.ret_type,
            [],
            new_blocks,
            block_map[entry],
            ssa=True,
        )

    # -- cloning -----------------------------------------------------------

    def _clone_ins(self, ins, use):
        if isinstance(ins, Const):
            return Const(self.fresh(ins.dest.type), ins.value)
        if isinstance(ins, Copy):
            return Copy(self.fresh(ins.dest.type), use(ins.src))
        if isinstance(ins, BinOp):
            return BinOp(
                self.fresh(ins.dest.type), ins.operator,
                use(ins.left), use(ins.right), ins.kind, ins.type,
            )
        return Call(
            self.fresh(ins.dest.type), ins.name,
            [use(arg) for arg in ins.args], ins.type,
        )

    def _clone_terminator(self, term, block_map, use):
        if isinstance(term, Return):
            return Return(None if term.value is None else use(term.value))
        if isinstance(term, Jump):
            return Jump(block_map[term.target])
        if isinstance(term, Branch):
            return Branch(
                use(term.cond),
                block_map[term.true_target],
                block_map[term.false_target],
            )
        return None  # pragma: no cover - lowerer always terminates blocks

    def _clone_unreachable(self, old: Block, new: Block, block_map) -> None:
        """Structural clone for blocks unreachable from the entry.

        ``lower_module`` never emits such blocks; this keeps the transform
        total without inserting phi nodes that no reachable edge feeds.
        """
        local: dict[object, Temp] = {}

        def local_use(ref):
            mapped = local.get(ref)
            if mapped is None:
                mapped = self.fresh(ref.type)
                local[ref] = mapped
            return mapped

        new.phis = []
        new.instructions = []
        for ins in old.instructions:
            if isinstance(ins, Copy):
                local[ins.dest] = local_use(ins.src)
                continue
            cloned = self._clone_ins(ins, local_use)
            local[ins.dest] = cloned.dest
            new.instructions.append(cloned)
        new.terminator = self._clone_terminator(
            old.terminator, block_map, local_use
        )

    # -- phi pruning -------------------------------------------------------

    def _prune(self, blocks: list[Block]) -> None:
        """Remove dead and trivial phi nodes to a fixpoint.

        A phi is *dead* when nothing outside the phi-only reference graph
        reads its result: phi-only cycles (e.g. a loop-carried temporary
        nobody reads) evaporate together.  A phi is *trivial* when its
        incoming values collapse to one distinct value once previously
        removed phis are chased through; a loop self-reference on its own
        result counts as such.  Dead phis are dropped as a batch first,
        then trivial phis one at a time; the two passes alternate until
        neither fires.
        """
        while self._remove_dead_phis(blocks) | self._remove_trivial_phis(
            blocks
        ):
            pass

    def _live_phi_dests(self, blocks: list[Block]) -> set[int]:
        all_phis = [phi for block in blocks for phi in block.phis]

        # Worklist starts from temps consumed by ordinary instructions and
        # terminators; reaching a phi's result marks that phi live and in
        # turn activates the phis feeding it.  A closed phi-only cycle is
        # never seeded and so dies wholesale.
        worklist: list[int] = []
        for block in blocks:
            for ins in block.instructions:
                worklist.extend(id(v) for v in _operands(ins))
            worklist.extend(id(v) for v in _term_operands(block.terminator))

        live: set[int] = set()
        phi_by_dest = {id(phi.dest): phi for phi in all_phis}
        while worklist:
            temp_id = worklist.pop()
            if temp_id in live:
                continue
            live.add(temp_id)
            phi = phi_by_dest.get(temp_id)
            if phi is not None:
                for value in phi.entries.values():
                    if value is not None and value is not phi.dest:
                        worklist.append(id(value))
        return live

    def _remove_dead_phis(self, blocks: list[Block]) -> bool:
        live = self._live_phi_dests(blocks)
        changed = False
        for block in blocks:
            for phi in list(block.phis):
                if id(phi.dest) not in live:
                    block.phis.remove(phi)
                    changed = True
        return changed

    def _remove_trivial_phis(self, blocks: list[Block]) -> bool:
        representative: dict[Temp, Temp] = {}

        def canonical(value: Temp) -> Temp:
            seen = set()
            while value in representative and id(value) not in seen:
                seen.add(id(value))
                value = representative[value]
            return value

        for block in blocks:
            for phi in list(block.phis):
                incoming = {
                    canonical(value)
                    for value in phi.entries.values()
                    if value is not None and value is not phi.dest
                }
                if len(incoming) == 1:
                    replacement = next(iter(incoming))
                    representative[phi.dest] = replacement
                    block.phis.remove(phi)
                    self._replace(blocks, phi.dest, replacement)
                    return True
        return False

    def _replace(self, blocks: list[Block], old: Temp, new: Temp) -> None:
        def subst(value):
            return new if value is old else value

        for block in blocks:
            for phi in block.phis:
                phi.entries = {
                    pred: subst(value) for pred, value in phi.entries.items()
                }
            for ins in block.instructions:
                _map_operands(ins, subst)
            _map_term_operands(block.terminator, subst)

    # -- canonical numbering ----------------------------------------------

    def _renumber(
        self, params: list[Parameter], blocks: list[Block]
    ) -> None:
        """Assign deterministic per-function SSA ids in place.

        Parameters take ids first in parameter order; within each block
        (visited in id order) phi results precede ordinary instruction
        results.  Everything after renumbering refers to the new temps.
        """
        number: dict[Temp, Temp] = {}
        next_id = 0

        def assign(temp: Temp) -> Temp:
            nonlocal next_id
            mapped = number.get(temp)
            if mapped is None:
                mapped = Temp(next_id, temp.type)
                next_id += 1
                number[temp] = mapped
            return mapped

        for param in params:
            param.temp = assign(param.temp)

        for block in blocks:
            for phi in block.phis:
                assign(phi.dest)
            for ins in block.instructions:
                assign(ins.dest)

        def numbered(value):
            if value is None:
                return None
            mapped = number.get(value)
            if mapped is None:
                # Defensive: an operand with no reachable definition.
                # Never occurs for lowered, pruned IR.
                mapped = assign(value)
            return mapped

        for block in blocks:
            for phi in block.phis:
                dest = numbered(phi.dest)
                phi.entries = {
                    pred: numbered(value)
                    for pred, value in sorted(
                        phi.entries.items(), key=lambda item: item[0].id
                    )
                }
                phi.dest = dest
            for ins in block.instructions:
                ins.dest = numbered(ins.dest)
                _map_operands(ins, numbered)
            _map_term_operands(block.terminator, numbered)


# --------------------------------------------------------------------------
# Idempotent cloning of an already-SSA module
# --------------------------------------------------------------------------


def _clone_ssa_module(module: Module) -> Module:
    return Module(
        [_clone_ssa_function(func) for func in module.functions], ssa=True
    )


def _clone_ssa_function(func: Function) -> Function:
    new_blocks = [Block(block.id) for block in func.blocks]
    block_map = dict(zip(func.blocks, new_blocks))

    number: dict[Temp, Temp] = {}
    next_id = 0

    def assign(temp: Temp) -> Temp:
        nonlocal next_id
        mapped = number.get(temp)
        if mapped is None:
            mapped = Temp(next_id, temp.type)
            next_id += 1
            number[temp] = mapped
        return mapped

    new_params: list[Parameter] = []
    for param in func.params:
        new_params.append(
            Parameter(param.name, param.slot, assign(param.temp))
        )
    for block, new_block in zip(func.blocks, new_blocks):
        for phi in block.phis:
            assign(phi.dest)
        for ins in block.instructions:
            assign(ins.dest)

    def ref(value):
        return number[value] if value is not None else None

    for block, new_block in zip(func.blocks, new_blocks):
        for phi in block.phis:
            entries = {
                block_map[pred]: ref(value)
                for pred, value in sorted(
                    phi.entries.items(), key=lambda item: item[0].id
                )
            }
            new_block.phis.append(Phi(ref(phi.dest), entries))
        for ins in block.instructions:
            if isinstance(ins, Const):
                clone = Const(ref(ins.dest), ins.value)
            elif isinstance(ins, Copy):
                clone = Copy(ref(ins.dest), ref(ins.src))
            elif isinstance(ins, BinOp):
                clone = BinOp(
                    ref(ins.dest), ins.operator, ref(ins.left),
                    ref(ins.right), ins.kind, ins.type,
                )
            else:
                clone = Call(
                    ref(ins.dest), ins.name,
                    [ref(arg) for arg in ins.args], ins.type,
                )
            new_block.instructions.append(clone)
        term = block.terminator
        if isinstance(term, Return):
            new_block.terminator = Return(ref(term.value))
        elif isinstance(term, Jump):
            new_block.terminator = Jump(block_map[term.target])
        elif isinstance(term, Branch):
            new_block.terminator = Branch(
                ref(term.cond),
                block_map[term.true_target],
                block_map[term.false_target],
            )

    return Function(
        func.name,
        new_params,
        func.ret_type,
        list(func.locals),
        new_blocks,
        block_map[func.entry],
        ssa=True,
    )
