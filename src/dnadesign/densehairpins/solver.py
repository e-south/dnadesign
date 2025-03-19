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

def extract_solver_params(options):
    """
    From a list of option strings (e.g., "TimeLimit=5", "Threads=16"), extract known parameters.
    Returns a tuple of (params_dict, remaining_options).
    """
    params = {}
    remaining_options = []
    for opt in options:
        if "=" in opt:
            key, val = opt.split("=", 1)
            key = key.strip()
            val = val.strip()
            if key.lower() == "timelimit":
                try:
                    params["TimeLimit"] = float(val)
                except ValueError:
                    print(f"Warning: Could not parse TimeLimit value '{val}'; ignoring it.")
                continue
            elif key.lower() == "threads":
                try:
                    params["Threads"] = int(val)
                except ValueError:
                    print(f"Warning: Could not parse Threads value '{val}'; ignoring it.")
                continue
        remaining_options.append(opt)
    return params, remaining_options

class DenseArrayOptimizerWrapper:
    def __init__(self, library: list, sequence_length: int, solver: str = "CBC",
                 solver_options: list = None, fixed_elements: dict = None, fill_gap: bool = False,
                 fill_gap_end: str = "3prime", fill_gc_min: float = 0.40, fill_gc_max: float = 0.60):
        """
        Accepts library as a list of dictionaries with keys "TF" and "sequence" (or as strings).
        Builds:
          - self.lib_sequences: list of nucleotide sequences (for Dense Arrays),
          - self.tf_list: corresponding list of TF names,
          - self.full_library: the original library (for random subsampling).
        """
        self.full_library = library[:]  # Save the original input library.
        self.lib_sequences = []  # For Dense Arrays: only nucleotide sequences.
        self.tf_list = []        # Parallel list of TF names.
        for item in library:
            if isinstance(item, dict):
                tf = item.get("TF", "").strip()
                seq = item.get("sequence", "").strip().upper()
                if not tf or not seq:
                    continue
                self.lib_sequences.append(seq)
                self.tf_list.append(tf)
            elif isinstance(item, str):
                motif_str = item.strip()
                if not motif_str or motif_str.lower() == "none":
                    continue
                if ":" in motif_str:
                    tf_part, seq = motif_str.split(":", 1)
                    tf = tf_part.strip()
                    seq = seq.strip().upper()
                    self.lib_sequences.append(seq)
                    self.tf_list.append(tf)
                else:
                    continue
        if not self.lib_sequences:
            raise ValueError("After filtering, the library is empty or contains no valid motifs.")
        self.sequence_length = sequence_length
        self.solver = solver
        self.solver_options = solver_options if solver_options is not None else []
        self.fixed_elements = fixed_elements.copy() if fixed_elements is not None else {}
        self.fill_gap = fill_gap
        self.fill_gap_end = fill_gap_end
        self.fill_gc_min = fill_gc_min
        self.fill_gc_max = fill_gc_max
        # For random_subsample_per_solve mode, store forbidden DenseArray solutions.
        self.forbidden_solutions = []

    def _convert_none(self, value):
        if isinstance(value, str) and value.strip().lower() == "none":
            return None
        return value

    def get_optimizer_instance(self, library_subset=None) -> da.Optimizer:
        """
        Build an optimizer instance.
        If library_subset is provided, it should be a list of dictionaries with keys "TF" and "sequence".
        Otherwise, use the full library (self.lib_sequences and self.tf_list).
        """
        if library_subset is not None:
            lib_seq = []
            tf_list = []
            for item in library_subset:
                if isinstance(item, dict):
                    tf = item.get("TF", "").strip()
                    seq = item.get("sequence", "").strip().upper()
                    if not tf or not seq:
                        continue
                    lib_seq.append(seq)
                    tf_list.append(tf)
                elif isinstance(item, str):
                    if ":" in item:
                        tf_part, seq = item.split(":", 1)
                        tf = tf_part.strip()
                        seq = seq.strip().upper()
                        lib_seq.append(seq)
                        tf_list.append(tf)
            opt_inst = da.Optimizer(library=lib_seq, sequence_length=self.sequence_length)
            opt_inst.current_tf_list = tf_list
        else:
            opt_inst = da.Optimizer(library=self.lib_sequences.copy(), sequence_length=self.sequence_length)
            opt_inst.current_tf_list = self.tf_list.copy()
        # Add promoter constraints.
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
                if upstream is not None and upstream not in opt_inst.library:
                    opt_inst.library.append(upstream)
                if downstream is not None and downstream not in opt_inst.library:
                    opt_inst.library.append(downstream)
                new_constraint = {k: v for k, v in new_constraint.items() if v is not None and k != "name"}
                converted_constraints.append(new_constraint)
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
        # Do not add forbidden solutions here; they will be added after model build.
        return opt_inst

    def optimize_iteratively(self, quota: int, timeout_seconds: int = 30, save_callback=None,
                               random_subsample_per_solve: bool = False, subsample_size: int = None) -> list:
        """
        If random_subsample_per_solve is false, build the model once and iteratively solve.
        If true, for each iteration, build a new model from a fresh random subsample (of size subsample_size)
        drawn from the full library (using self.full_library), re-add previously forbidden solutions one at a time
        (after model build), then solve.
        Returns a list of solution dictionaries.
        """
        solutions = []
        if not random_subsample_per_solve:
            # Fixed library mode:
            opt_inst = self.get_optimizer_instance()
            params, local_solver_options = extract_solver_params(self.solver_options.copy())
            time_limit_sec = params.get("TimeLimit", None)
            threads = params.get("Threads", None)
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
                if threads is not None:
                    if self.solver.upper() == "CBC":
                        if hasattr(opt_inst.model, "SetNumThreads"):
                            opt_inst.model.SetNumThreads(threads)
                            print(f"Threads set to {threads} for CBC.")
                        else:
                            print("Warning: CBC model does not support setting thread count.")
                    elif self.solver.upper() == "GUROBI":
                        try:
                            opt_inst.model.setParam("Threads", threads)
                            print(f"Threads set to {threads} for GUROBI.")
                        except Exception as e:
                            print(f"Warning: Failed to set Threads for GUROBI: {e}")
                    else:
                        if hasattr(opt_inst.model, "SetNumThreads"):
                            opt_inst.model.SetNumThreads(threads)
                            print(f"Threads set to {threads}.")
                        else:
                            print("Warning: Model does not support setting thread count.")
            except Exception as e:
                print(f"Model build failed: {e}")
                raise

            for i in range(quota):
                print(f"Solver iteration {i+1}/{quota} starting (fixed library)...")
                try:
                    solution = opt_inst.solve()
                except Exception as e:
                    print(f"Failed to obtain solution at iteration {i}: {e}")
                    break
                if not solution or solution.nb_motifs <= 0:
                    print(f"Invalid solution at iteration {i}.")
                    break
                used_offsets = solution.offset_indices_in_order()
                indices = [index for offset, index in used_offsets]
                roster_used = [self.tf_list[index % len(self.tf_list)] for index in indices]
                sol_dict = {
                    "entry_id": random.randint(1000, 9999),
                    "tf_roster": roster_used,
                    "sequence": solution.sequence,
                    "meta_visual": str(solution),
                    "offsets": used_offsets,
                    "compression_ratio": getattr(solution, "compression_ratio", None)
                }
                solutions.append(sol_dict)
                print(f"Iteration {i+1} completed. {len(solutions)} solution(s) generated so far.")
                try:
                    opt_inst.forbid(solution)
                    print("Solution forbidden successfully.")
                    self.forbidden_solutions.append(solution)
                except Exception as e:
                    print(f"Failed to forbid solution at iteration {i}: {e}")
                    break
                if save_callback is not None:
                    save_callback(solutions)
        else:
            # Random subsample mode:
            if subsample_size is None:
                raise ValueError("subsample_size must be provided when random_subsample_per_solve is true.")
            # Use self.full_library (the original library as provided) for subsampling.
            for i in range(quota):
                current_subsample = random.sample(self.full_library, subsample_size)
                current_tf_list = []
                current_dict_list = []
                for item in current_subsample:
                    if isinstance(item, dict):
                        tf = item.get("TF", "").strip()
                        seq = item.get("sequence", "").strip().upper()
                        current_tf_list.append(tf)
                        current_dict_list.append({"TF": tf, "sequence": seq})
                    elif isinstance(item, str):
                        if ":" in item:
                            tf_part, seq = item.split(":", 1)
                            tf = tf_part.strip()
                            seq = seq.strip().upper()
                            current_tf_list.append(tf)
                            current_dict_list.append({"TF": tf, "sequence": seq})
                opt_inst = self.get_optimizer_instance(library_subset=current_dict_list)
                params, local_solver_options = extract_solver_params(self.solver_options.copy())
                time_limit_sec = params.get("TimeLimit", None)
                threads = params.get("Threads", None)
                try:
                    if self.solver.upper() == "CBC" and time_limit_sec is not None:
                        opt_inst.build_model(self.solver, solver_options=[])
                        if hasattr(opt_inst.model, "SupportsTimeLimit") and opt_inst.model.SupportsTimeLimit():
                            opt_inst.model.SetTimeLimit(int(time_limit_sec * 1000))
                            print(f"TimeLimit set to {int(time_limit_sec * 1000)} ms for CBC (iteration {i+1}).")
                        else:
                            print("Warning: CBC solver does not support time limits on this system.")
                    else:
                        opt_inst.build_model(self.solver, solver_options=local_solver_options)
                    if threads is not None:
                        if self.solver.upper() == "CBC":
                            if hasattr(opt_inst.model, "SetNumThreads"):
                                opt_inst.model.SetNumThreads(threads)
                                print(f"Threads set to {threads} for CBC (iteration {i+1}).")
                            else:
                                print("Warning: CBC model does not support setting thread count.")
                        elif self.solver.upper() == "GUROBI":
                            try:
                                opt_inst.model.setParam("Threads", threads)
                                print(f"Threads set to {threads} for GUROBI (iteration {i+1}).")
                            except Exception as e:
                                print(f"Warning: Failed to set Threads for GUROBI: {e}")
                        else:
                            if hasattr(opt_inst.model, "SetNumThreads"):
                                opt_inst.model.SetNumThreads(threads)
                                print(f"Threads set to {threads} (iteration {i+1}).")
                            else:
                                print("Warning: Model does not support setting thread count.")
                except Exception as e:
                    print(f"Model build failed at iteration {i}: {e}")
                    break

                # Re-add forbidden solutions one at a time.
                for sol in self.forbidden_solutions:
                    try:
                        opt_inst.forbid(sol)
                        print(f"Re-added a forbidden solution (iteration {i+1}).")
                    except Exception as e:
                        print(f"Warning: Failed to re-add a forbidden solution: {e}")

                print(f"Solver iteration {i+1}/{quota} starting (random subsample)...")
                try:
                    solution = opt_inst.solve()
                except Exception as e:
                    print(f"Failed to obtain solution at iteration {i}: {e}")
                    break
                if not solution or solution.nb_motifs <= 0:
                    print(f"Invalid solution at iteration {i}.")
                    break
                used_offsets = solution.offset_indices_in_order()
                indices = [index for offset, index in used_offsets]
                roster_used = [current_tf_list[index % len(current_tf_list)] for index in indices]
                sol_dict = {
                    "entry_id": random.randint(1000, 9999),
                    "tf_roster": roster_used,
                    "sequence": solution.sequence,
                    "meta_visual": str(solution),
                    "offsets": used_offsets,
                    "compression_ratio": getattr(solution, "compression_ratio", None)
                }
                solutions.append(sol_dict)
                print(f"Iteration {i+1} completed. {len(solutions)} solution(s) generated so far.")
                try:
                    opt_inst.forbid(solution)
                    print(f"Solution forbidden successfully (iteration {i+1}).")
                    self.forbidden_solutions.append(solution)
                except Exception as e:
                    print(f"Failed to forbid solution at iteration {i}: {e}")
                    break
                if save_callback is not None:
                    save_callback(solutions)
        return solutions

def run_solver(library, sequence_length, solver, solver_options):
    optimizer = DenseArrayOptimizerWrapper(library, sequence_length, solver, solver_options)
    solution_dict = optimizer.optimize()  # Assuming an optimize() method exists if needed.
    return solution_dict

def run_solver_iteratively(library, sequence_length, solver, solver_options, quota, save_callback=None,
                           random_subsample_per_solve: bool = False, subsample_size: int = None):
    optimizer = DenseArrayOptimizerWrapper(library, sequence_length, solver, solver_options)
    return optimizer.optimize_iteratively(quota, save_callback=save_callback,
                                          random_subsample_per_solve=random_subsample_per_solve,
                                          subsample_size=subsample_size)

def save_solver_output(solutions, output_file):
    """
    Saves the given solutions to output_file.
    If output_file exists, appends the new solutions without duplicating previous ones.
    """
    import pickle
    try:
        with open(output_file, 'wb') as f:
            pickle.dump(solutions, f)
        print(f"Solver output saved to {output_file}")
    except Exception as e:
        print(f"Error saving solver output: {e}")
