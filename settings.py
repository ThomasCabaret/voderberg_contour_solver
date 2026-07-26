"""Central configuration for the Voderberg contour solver."""
from __future__ import annotations

# Orientation and topology conventions.
FORWARD = 1
REVERSE = -1
DIRECT = 1
REFLECTED = -1
SAME = 1
OPPOSITE = -1
MARKERS = ("A_start", "A_end", "B_start", "B_end")
LOCI = ("P0", "A", "P1", "B")
PROTOTYPE_ORIENTATION = "counterclockwise"
REFERENCE_INTERIOR_SIDE = 1
OPPOSITE_INTERIOR_SIDE = -1
DEFAULT_ALLOW_REFLECTIONS = True

# Bounded formal enumeration.
# Increase these values to reduce truncation.  The audit CLI accepts the same
# values through --max-depth and --max-states.  Zero means "unbounded", which
# may fail to terminate on cyclic systems.
FORMAL_SOLVER_UNLIMITED_VALUE = 0
DEFAULT_AUDIT_MAX_DEPTH = 5
DEFAULT_AUDIT_MAX_STATES = 100
DEFAULT_FORMAL_MAX_CYCLE_UNROLLS = 3
FORMAL_CYCLE_CAP_DISABLED_VALUE = 0
DEFAULT_ENABLE_POSITIVE_LENGTH_FILTER = True
DEFAULT_ENABLE_SOLUTION_CANONICALIZATION = True
DEFAULT_ENABLE_CANONICAL_PROFILE_REDUCTION = True
DEFAULT_ENABLE_PROFILE_SUBSUMPTION_REDUCTION = True

# Global decorated-contour feasibility layers.  The exact rational angular and
# length blocks can be toggled independently.  The polynomial layers are
# hierarchical: signed area requires the chord/length layer.
DEFAULT_ENABLE_GLOBAL_LINEAR_ANGLE_FILTER = True
DEFAULT_ENABLE_GLOBAL_LINEAR_LENGTH_FILTER = True
DEFAULT_ENABLE_CHORD_LENGTH_LAYER = True
DEFAULT_ENABLE_SIGNED_AREA_LAYER = True

# Exact partial formal solver.  The residual graph must be complete before any
# result is labelled exact.  Only finite and fixed-context power cycles are
# compiled for downstream use; more general morphic SCCs remain explicit.
DEFAULT_FORMAL_SOLVER_MODE = "exact-partial"
FORMAL_SOLVER_MODE_CHOICES = ("exact-partial", "legacy-bounded")
DEFAULT_EXACT_GRAPH_MAX_NODES = 500
DEFAULT_EXACT_GRAPH_MAX_EDGES = 3000
DEFAULT_EXACT_MAX_FAMILIES_PER_CASE = 10000
# Parametric families remain symbolic by default.  The legacy geometric
# pipeline receives only genuinely finite formal families unless an explicit
# expansion policy is selected on the audit command line.
DEFAULT_FAMILY_EXPANSION_POLICY = "none"
FAMILY_EXPANSION_POLICY_CHOICES = ("none", "minimum", "fixed", "range")
DEFAULT_FAMILY_REPRESENTATIVE_EXPONENT = 1
DEFAULT_FAMILY_EXPANSION_MAX_EXPONENT = 3
DEFAULT_FAMILY_EXPANSION_MAX_SPECIALIZATIONS = 10000
# Backward-compatible alias.  New code should use DEFAULT_FAMILY_EXPANSION_POLICY.
DEFAULT_EXPAND_PARAMETRIC_REPRESENTATIVES = False
DEFAULT_ENABLE_CURVE_TERM_SPECIALIZATION = True

DEFAULT_VODERBERG_TYPE_SELECTION = "all"
VODERBERG_TYPE_SELECTION_CHOICES = ("all", "type1", "type2", "type1+type2")
DEEP_AUDIT_MAX_DEPTH = 10
DEEP_AUDIT_MAX_STATES = 1000
AGGRESSIVE_AUDIT_MAX_DEPTH = 20
AGGRESSIVE_AUDIT_MAX_STATES = 10000
DEFAULT_AUDIT_EXAMPLE_LIMIT = 10
DEFAULT_SHOW_AUDIT_PROGRESS = True
DEFAULT_AUDIT_PROGRESS_INTERVAL = 100
DEFAULT_RUN_PARITY_DIAGNOSTICS = True

# Stable project filenames.
SETTINGS_FILENAME = "settings.py"
PROJECT_CLI_FILENAME = "project_cli.py"
AUDIT_SCRIPT_FILENAME = "audit_geometric_filters.py"
WEB_SCRIPT_FILENAME = "results_web.py"
GEOMETRY_SCRIPT_FILENAME = "geometry_search_viewer.py"
TEST_MODULE_PATTERN = "test*.py"
AUDIT_SUMMARY_FILENAME = "geometric_filter_audit.json"
AUDIT_PROFILES_FILENAME = "geometric_filter_profiles.json"
AUDIT_SURVIVORS_FILENAME = "geometric_filter_survivors.json"
FORMAL_EQUATION_AUDIT_FILENAME = "formal_equation_audit.json"
GEOMETRY_CANDIDATES_FILENAME = "geometric_candidates.json"
GEOMETRY_CHECKPOINT_FILENAME = "geometry_search_checkpoint.sqlite3"
TEST_RESULTS_FILENAME = "TEST_RESULTS.txt"
PROJECT_MANIFEST_FILENAME = "PROJECT_MANIFEST.txt"
README_FILENAME = "README.md"
INTEGRATION_REPORT_FILENAME = "INTEGRATION_REPORT.json"
Z3_SCRIPT_FILENAME = "joint_translation_z3.py"
Z3_DEFAULT_SMT2_FILENAME = "joint_translation_z3_problem.smt2"
Z3_DEFAULT_METADATA_FILENAME = "joint_translation_z3_metadata.json"
Z3_AUDIT_FILENAME = "joint_translation_z3_audit.json"

# Local web viewer.
DEFAULT_WEB_HOST = "127.0.0.1"
DEFAULT_WEB_PORT = 8765
DEFAULT_WEB_RESULTS_FILE = AUDIT_SURVIVORS_FILENAME
WEB_SCHEMA_VERSION = "formal-profile-v4"
WEB_DEFAULT_PAGE_SIZE = 200
WEB_PAGE_SIZE_OPTIONS = (50, 100, 200, 500)
WEB_STREAM_CHUNK_SIZE = 1024 * 1024

# Optional exact polynomial relaxation.
Z3_REQUIRE_ALL_CHORDS_NONZERO = True
Z3_DEFAULT_TIMEOUT_MS = 30000
Z3_INCLUDE_MODEL_IN_REPORT = False
DEFAULT_PREPARE_JOINT_TRANSLATION = True
DEFAULT_RUN_Z3 = True
DEFAULT_Z3_MAX_PROFILES = 0

# Heuristic polygonal drawing search.
GEOMETRY_DEFAULT_INTERMEDIATE_POINTS = 1
GEOMETRY_DEFAULT_MAX_PROFILES = 0
GEOMETRY_DEFAULT_ATTEMPTS_PER_PROFILE = 2
GEOMETRY_DEFAULT_MAX_ITERATIONS = 180
GEOMETRY_DEFAULT_POPULATION_SIZE = 8
GEOMETRY_DEFAULT_SEED = 1729
GEOMETRY_DEFAULT_RESUME = True
GEOMETRY_ENFORCE_CONTACT_TEMPLATE_CONSTRAINTS = True
GEOMETRY_TEMPLATE_CONSTRAINT_TOLERANCE = 1.0e-10
GEOMETRY_LENGTH_MIN = 0.25
GEOMETRY_LENGTH_MAX = 2.0
GEOMETRY_ANGLE_MARGIN = 0.03
GEOMETRY_CLOSURE_TOLERANCE = 2.0e-3
GEOMETRY_TURN_TOLERANCE = 2.0e-3
GEOMETRY_MIN_ABS_AREA = 2.0e-3
GEOMETRY_INTERSECTION_PENALTY = 5000.0
GEOMETRY_CLOSURE_WEIGHT = 2000.0
GEOMETRY_TURN_WEIGHT = 500.0
GEOMETRY_AREA_WEIGHT = 500.0
GEOMETRY_VIEW_WIDTH = 1100
GEOMETRY_VIEW_HEIGHT = 760

# Method-status labels.
STATUS_EXACT = "exact_within_model"
STATUS_SOUND_INCOMPLETE = "sound_rejection_incomplete_coverage"
STATUS_BOUNDED = "bounded_incomplete_enumeration"
STATUS_EXPERIMENTAL = "experimental_not_in_core_rejection_pipeline"
STATUS_HEURISTIC = "heuristic_candidate_search"
