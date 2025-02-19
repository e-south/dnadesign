"""
--------------------------------------------------------------------------------
<dnadesign project>
dnadesign/densegen/optimizer_wrapper.py

Module Author(s): Eric J. South
Dunlop Lab
--------------------------------------------------------------------------------
"""

import time
import random
import dense_arrays as da

def random_fill(length: int, gc_min: float = 0.40, gc_max: float = 0.60) -> str:
    nucleotides = "atgc"  # lower-case nucleotides for gap fill
    for _ in range(1000):
        seq = "".join(random.choices(nucleotides, k=length))
        gc_content = (seq.count("g") + seq.count("c")) / length
        if gc_min <= gc_content <= gc_max:
            return seq
    return seq


class DenseArrayOptimizer:
    def __init__(self, library: list, sequence_length: int, solver: str = "CBC",
                 solver_options: list = None, fixed_elements: dict = None, fill_gap: bool = False,
                 fill_gap_end: str = "3prime", fill_gc_min: float = 0.40, fill_gc_max: float = 0.60):
        # Filter out motifs that are not strings, are empty, or contain invalid characters.
        valid_nucleotides = {"A", "T", "G", "C"}
        filtered_library = []
        for motif in library:
            if not isinstance(motif, str):
                continue
            motif_str = motif.strip()
            if not motif_str or motif_str.lower() == "none":
                continue
            motif_upper = motif_str.upper()
            if set(motif_upper).issubset(valid_nucleotides):
                filtered_library.append(motif_str)
        if not filtered_library:
            raise ValueError("After filtering, the library is empty or contains no valid motifs.")
        self.library = filtered_library

        self.sequence_length = sequence_length
        self.solver = solver
        self.solver_options = solver_options if solver_options is not None else []
        # Allow fixed_elements to be None.
        self.fixed_elements = fixed_elements.copy() if fixed_elements is not None else {}
        self.fill_gap = fill_gap
        self.fill_gap_end = fill_gap_end
        self.fill_gc_min = fill_gc_min
        self.fill_gc_max = fill_gc_max

    def _convert_none(self, value):
        """Convert a string 'none' to Python None, otherwise return the value unchanged."""
        if isinstance(value, str) and value.strip().lower() == "none":
            return None
        return value

    def get_optimizer_instance(self) -> da.Optimizer:
        lib_for_opt = self.library.copy()
        converted_constraints = []
        constraints = self.fixed_elements.get("promoter_constraints")
        if constraints:
            for constraint in constraints:
                # Convert any string "None" values to actual None.
                processed_constraint = {k: self._convert_none(v) for k, v in constraint.items()}
                # Skip constraint if it is None or if all its values are None.
                if not any(value is not None for value in processed_constraint.values()):
                    continue
                new_constraint = dict(processed_constraint)
                # Convert list values to tuples for specific keys.
                for key in ["upstream_pos", "downstream_pos", "spacer_length"]:
                    if key in new_constraint and isinstance(new_constraint[key], list):
                        new_constraint[key] = tuple(new_constraint[key])
                # For upstream and downstream, only add if the value is not None.
                upstream = new_constraint.get("upstream")
                downstream = new_constraint.get("downstream")
                if upstream is not None and upstream not in lib_for_opt:
                    lib_for_opt.append(upstream)
                if downstream is not None and downstream not in lib_for_opt:
                    lib_for_opt.append(downstream)
                # Remove keys with None values.
                new_constraint = {k: v for k, v in new_constraint.items() if v is not None}
                converted_constraints.append(new_constraint)

        opt_inst = da.Optimizer(library=lib_for_opt, sequence_length=self.sequence_length)
        for constraint in converted_constraints:
            try:
                opt_inst.add_promoter_constraints(**constraint)
            except KeyError as err:
                print(f"Error adding promoter constraint {constraint}: missing key {err}")
                raise
        if "side_biases" in self.fixed_elements and self.fixed_elements["side_biases"]:
            side_biases = self.fixed_elements["side_biases"]
            left = side_biases.get("left")
            right = side_biases.get("right")
            if left and any(item is not None for item in left):
                opt_inst.add_side_biases(left=left, right=right)
        return opt_inst


    def optimize(self, timeout_seconds: int = 30) -> da.DenseArray:
        start_time = time.time()
        opt_inst = self.get_optimizer_instance()
        local_solver_options = self.solver_options.copy()
        time_limit_sec = None
        if self.solver.upper() == "CBC":
            remaining_options = []
            for opt in local_solver_options:
                if opt.startswith("TimeLimit="):
                    try:
                        time_limit_sec = float(opt.split("=")[1])
                    except ValueError:
                        print("Warning: Could not parse TimeLimit option; ignoring it.")
                else:
                    remaining_options.append(opt)
            local_solver_options = remaining_options

        try:
            if self.solver.upper() == "CBC" and time_limit_sec is not None:
                opt_inst.build_model(self.solver, solver_options=[])
                if hasattr(opt_inst.model, "SupportsTimeLimit") and opt_inst.model.SupportsTimeLimit():
                    opt_inst.model.SetTimeLimit(int(time_limit_sec * 1000))
                    print(f"TimeLimit set to {int(time_limit_sec * 1000)} ms for CBC.")
                else:
                    print("Warning: CBC solver does not support time limits on this system.")
                solution = opt_inst.solve()
            else:
                solution = opt_inst.optimal(solver=self.solver, solver_options=local_solver_options)
        except Exception as e:
            print(f"Optimization attempt failed: {e}.")
            raise

        if solution and solution.nb_motifs > 0:
            if self.fill_gap and len(solution.sequence) < self.sequence_length:
                gap = self.sequence_length - len(solution.sequence)
                fill_seq = random_fill(gap, self.fill_gc_min, self.fill_gc_max)
                if self.fill_gap_end.lower() == "5prime":
                    solution.sequence = fill_seq + solution.sequence
                else:
                    solution.sequence = solution.sequence + fill_seq
                setattr(solution, "meta_gap_fill", True)
                setattr(solution, "meta_gap_fill_details", {
                    "fill_gap": gap,
                    "fill_end": self.fill_gap_end,
                    "fill_gc_range": (self.fill_gc_min, self.fill_gc_max)
                })
            return solution
        else:
            raise ValueError("Optimization returned an invalid solution.")
