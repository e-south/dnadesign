"""
--------------------------------------------------------------------------------
<dnadesign project>
/densehairpins/solver.py

Module Author(s): Eric J. South
Dunlop Lab
--------------------------------------------------------------------------------
"""

import time
import random
import dense_arrays as da
import itertools as it

class DenseArrayOptimizerWrapper:
    def __init__(self, library: list, sequence_length: int, solver: str = "CBC",
                 solver_options: list = None, fixed_elements: dict = None, fill_gap: bool = False,
                 fill_gap_end: str = "3prime", fill_gc_min: float = 0.40, fill_gc_max: float = 0.60):
        valid_nucleotides = {"A", "T", "G", "C"}
        filtered_binding_sites = []
        self.roster = []  # Store original "TF:binding_site" strings for metadata.
        for motif in library:
            if not isinstance(motif, str):
                continue
            motif_str = motif.strip()
            if not motif_str or motif_str.lower() == "none":
                continue
            if ":" in motif_str:
                parts = motif_str.split(":", 1)
                binding_site = parts[1].strip().upper()
            else:
                binding_site = motif_str.upper()
            if set(binding_site).issubset(valid_nucleotides):
                filtered_binding_sites.append(binding_site)
                self.roster.append(motif_str)
        if not filtered_binding_sites:
            raise ValueError("After filtering, the library is empty or contains no valid motifs.")
        self.library = filtered_binding_sites
        self.sequence_length = sequence_length
        self.solver = solver
        self.solver_options = solver_options if solver_options is not None else []
        self.fixed_elements = fixed_elements.copy() if fixed_elements is not None else {}
        self.fill_gap = fill_gap
        self.fill_gap_end = fill_gap_end
        self.fill_gc_min = fill_gc_min
        self.fill_gc_max = fill_gc_max

    def _convert_none(self, value):
        if isinstance(value, str) and value.strip().lower() == "none":
            return None
        return value

    def get_optimizer_instance(self) -> da.Optimizer:
        lib_for_opt = self.library.copy()
        converted_constraints = []
        constraints = self.fixed_elements.get("promoter_constraints")
        if constraints:
            for constraint in constraints:
                processed_constraint = {k: self._convert_none(v) for k, v in constraint.items()}
                if not any(value is not None for value in processed_constraint.values()):
                    continue
                new_constraint = dict(processed_constraint)
                for key in ["upstream_pos", "downstream_pos", "spacer_length"]:
                    if key in new_constraint and isinstance(new_constraint[key], list):
                        new_constraint[key] = tuple(new_constraint[key])
                upstream = new_constraint.get("upstream")
                downstream = new_constraint.get("downstream")
                if upstream is not None and upstream not in lib_for_opt:
                    lib_for_opt.append(upstream)
                if downstream is not None and downstream not in lib_for_opt:
                    lib_for_opt.append(downstream)
                new_constraint = {k: v for k, v in new_constraint.items() if v is not None and k != "name"}
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

    def optimize_iteratively(self, quota: int, timeout_seconds: int = 30) -> list:
        """
        Build the model once and then iteratively solve and forbid previous solutions.
        Returns a list of solution dictionaries.
        """
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

        # Build the model once.
        try:
            if self.solver.upper() == "CBC" and time_limit_sec is not None:
                opt_inst.build_model(self.solver, solver_options=[])
                if hasattr(opt_inst.model, "SupportsTimeLimit") and opt_inst.model.SupportsTimeLimit():
                    opt_inst.model.SetTimeLimit(int(time_limit_sec * 1000))
                    print(f"TimeLimit set to {int(time_limit_sec * 1000)} ms for CBC.")
                else:
                    print("Warning: CBC solver does not support time limits on this system.")
            else:
                opt_inst.build_model(self.solver, solver_options=local_solver_options)
        except Exception as e:
            print(f"Model build failed: {e}")
            raise

        solutions = []
        for i in range(quota):
            try:
                if self.solver.upper() == "CBC" and time_limit_sec is not None:
                    solution = opt_inst.solve()
                else:
                    solution = opt_inst.optimal(solver=self.solver, solver_options=local_solver_options)
            except Exception as e:
                print(f"Failed to obtain solution at iteration {i}: {e}")
                break

            if not solution or solution.nb_motifs <= 0:
                print(f"Invalid solution at iteration {i}.")
                break

            used_offsets = solution.offset_indices_in_order()  # List of (offset, index).
            indices = [index for offset, index in used_offsets]
            roster_used = [self.roster[index % len(self.roster)] for index in indices]
            sol_dict = {
                "entry_id": random.randint(1000, 9999),
                "tf_roster": roster_used,
                "sequence": solution.sequence,
                "meta_visual": str(solution),
                "offsets": used_offsets,
                "compression_ratio": getattr(solution, "compression_ratio", None)
            }
            solutions.append(sol_dict)

            if (i + 1) % 5 == 0:
                print(f"Progress: {i + 1} solutions generated so far.")

            try:
                opt_inst.forbid(solution)
            except Exception as e:
                print(f"Failed to forbid solution at iteration {i}: {e}")
                break

        return solutions


def run_solver(library, sequence_length, solver, solver_options):
    optimizer = DenseArrayOptimizerWrapper(library, sequence_length, solver, solver_options)
    solution_dict = optimizer.optimize()  # Assuming an optimize() method exists if needed.
    return solution_dict


def run_solver_iteratively(library, sequence_length, solver, solver_options, quota):
    optimizer = DenseArrayOptimizerWrapper(library, sequence_length, solver, solver_options)
    return optimizer.optimize_iteratively(quota)


def save_solver_output(solutions, output_file):
    import pickle
    with open(output_file, 'wb') as f:
        pickle.dump(solutions, f)
