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
    """
    Generates a random nucleotide string of the given length with GC content between gc_min and gc_max.
    """
    nucleotides = "ATGC"
    for _ in range(1000):
        seq = "".join(random.choices(nucleotides, k=length))
        gc_content = (seq.count("G") + seq.count("C")) / length
        if gc_min <= gc_content <= gc_max:
            return seq
    return seq

class DenseArrayOptimizer:
    def __init__(self, library: list, sequence_length: int, solver: str = "CBC",
                 solver_options: list = None, fixed_elements: dict = None, fill_gap: bool = False,
                 fill_gap_end: str = "3prime", fill_gc_min: float = 0.40, fill_gc_max: float = 0.60):
        """
        :param library: List of binding site sequences (strings) to optimize.
        :param sequence_length: Target length for the dense array.
        :param solver: Solver name (e.g., "CBC", "GUROBI").
        :param solver_options: List of solver-specific options.
        :param fixed_elements: Optional dictionary with fixed elements.
        :param fill_gap: If True, fill the gap if the optimized sequence is shorter than sequence_length.
        :param fill_gap_end: "5prime" or "3prime" indicating where to append the fill.
        :param fill_gc_min, fill_gc_max: Desired GC content range for the fill.
        """
        assert isinstance(library, list) and library, "Library must be a non-empty list of motifs."
        self.library = library
        self.sequence_length = sequence_length
        self.solver = solver
        self.solver_options = solver_options if solver_options is not None else []
        self.fixed_elements = fixed_elements.copy() if fixed_elements is not None else {}
        self.fill_gap = fill_gap
        self.fill_gap_end = fill_gap_end
        self.fill_gc_min = fill_gc_min
        self.fill_gc_max = fill_gc_max

    def get_optimizer_instance(self) -> da.Optimizer:
        """
        Create and return a new da.Optimizer instance with fixed elements applied.
        """
        lib_for_opt = self.library.copy()
        converted_constraints = []
        if "promoter_constraints" in self.fixed_elements and self.fixed_elements["promoter_constraints"]:
            for constraint in self.fixed_elements["promoter_constraints"]:
                new_constraint = dict(constraint)
                for key in ["upstream_pos", "downstream_pos", "spacer_length"]:
                    if key in new_constraint and isinstance(new_constraint[key], list):
                        new_constraint[key] = tuple(new_constraint[key])
                upstream = new_constraint.get("upstream")
                downstream = new_constraint.get("downstream")
                if upstream and upstream not in lib_for_opt:
                    lib_for_opt.append(upstream)
                if downstream and downstream not in lib_for_opt:
                    lib_for_opt.append(downstream)
                converted_constraints.append(new_constraint)
        opt_inst = da.Optimizer(library=lib_for_opt, sequence_length=self.sequence_length)
        for constraint in converted_constraints:
            opt_inst.add_promoter_constraints(**constraint)
        if "side_biases" in self.fixed_elements and self.fixed_elements["side_biases"]:
            side_biases = self.fixed_elements["side_biases"]
            left = side_biases.get("left")
            right = side_biases.get("right")
            if left and any(item is not None for item in left):
                opt_inst.add_side_biases(left=left, right=right)
        return opt_inst

    def optimize(self, timeout_seconds: int = 30) -> da.DenseArray:
        """
        Generate a single solution, applying gap fill if enabled and
        setting a solver time limit when using CBC.
        """
        start_time = time.time()
        # Create a fresh optimizer instance.
        opt_inst = self.get_optimizer_instance()
        
        # Prepare a local copy of solver options.
        local_solver_options = self.solver_options.copy()
        
        # If using CBC, remove unsupported options and extract a time limit.
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
                    # You might ignore other options for CBC as needed.
                    remaining_options.append(opt)
            local_solver_options = remaining_options

        try:
            if self.solver.upper() == "CBC" and time_limit_sec is not None:
                # Build model with no options and then set the time limit directly.
                opt_inst.build_model(self.solver, solver_options=[])
                # Set time limit in milliseconds.
                opt_inst.model.SetTimeLimit(int(time_limit_sec * 1000))
                solution = opt_inst.optimal(solver=self.solver, solver_options=[])
            else:
                solution = opt_inst.optimal(solver=self.solver, solver_options=local_solver_options)
        except Exception as e:
            print(f"Optimization attempt failed: {e}.")
            raise

        if solution and solution.nb_motifs > 0:
            sol_seq = str(solution)
            if self.fill_gap and len(sol_seq) < self.sequence_length:
                gap = self.sequence_length - len(sol_seq)
                fill_seq = random_fill(gap, self.fill_gc_min, self.fill_gc_max)
                if self.fill_gap_end.lower() == "5prime":
                    sol_seq = fill_seq + sol_seq
                else:
                    sol_seq = sol_seq + fill_seq
                setattr(solution, "meta_gap_fill", True)
                setattr(solution, "meta_gap_fill_details", {
                    "fill_gap": gap,
                    "fill_end": self.fill_gap_end,
                    "fill_gc_range": (self.fill_gc_min, self.fill_gc_max)
                })
            return solution
        else:
            raise ValueError("Optimization returned an invalid solution.")
