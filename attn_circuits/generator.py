"""
Synthetic language data generator.

Generates token sequences from a known mixture of rule types:
  - Bigrams:       a → x
  - Trigrams:      # a → y
  - Skip-bigrams:  a ... @ → z
  - Induction:     a b ... a → b
  - Bracket:       ( ... content → )
  - Chained:       rule_A output feeds rule_B

Each sequence is generated autoregressively. The generator knows which rule
fired at each position (rule labels), used for diagnostics and ablations.

Usage:
    gen = LanguageGenerator(rules=BIGRAM_RULES, mode='isolated')
    tokens, labels = gen.sample_sequence(length=64)

    gen = LanguageGenerator(rules=ALL_RULES, mixing_weights=WEIGHTS, mode='mixed')
    tokens, labels = gen.sample_sequence(length=128)
"""

import random
import os
import numpy as np
from dataclasses import dataclass, field
from typing import Optional
from collections import defaultdict


# ---------------------------------------------------------------------------
# Vocabulary
# ---------------------------------------------------------------------------

ENTITY   = ['a', 'b', 'c', 'd', 'e', 'f', 'g', 'h']
MARKER   = ['#', '@', '!', '$']
# Outputs: 3 shared across all rule classes + 3 unique per class = 12 total
#   X0-X2: shared (each produced by one bigram + one trigram + one skip-bigram)
#   B0-B2: unique to bigrams
#   T0-T2: unique to trigrams
#   S0-S2: unique to skip-bigrams
SHARED_OUT  = ['X0', 'X1', 'X2']
BIGRAM_OUT  = ['B0', 'B1', 'B2']
TRIGRAM_OUT = ['T0', 'T1', 'T2']
SKIP_OUT    = ['S0', 'S1', 'S2']
BRACKET  = ['(', ')']

VOCAB = ENTITY + MARKER + SHARED_OUT + BIGRAM_OUT + TRIGRAM_OUT + SKIP_OUT + BRACKET
TOKEN2ID = {t: i for i, t in enumerate(VOCAB)}
ID2TOKEN = {i: t for t, i in TOKEN2ID.items()}
VOCAB_SIZE = len(VOCAB)

# Keep old names for backward compat
OUTPUT = SHARED_OUT + BIGRAM_OUT + TRIGRAM_OUT + SKIP_OUT
NOISE = []  # no dedicated noise tokens; noise emits uniform over full vocab


# ---------------------------------------------------------------------------
# Rule definitions
# ---------------------------------------------------------------------------

@dataclass
class BigramRule:
    """a → output. Fires when last token == trigger."""
    name: str
    trigger: str        # e.g. 'a'
    output: str         # e.g. 'x'
    rule_class: str = 'bigram'


@dataclass
class TrigramRule:
    """prev_prev prev → output. Fires on two-token context."""
    name: str
    trigger1: str       # e.g. '#'
    trigger2: str       # e.g. 'a'
    output: str         # e.g. 'y'
    rule_class: str = 'trigram'


@dataclass
class SkipBigramRule:
    """anchor ... current_marker → output. Fires when anchor appeared anywhere in
    window and current token is current_marker."""
    name: str
    anchor: str         # e.g. 'a'
    current_marker: str # e.g. '@'
    output: str         # e.g. 'z'
    max_skip: int = 8
    rule_class: str = 'skip_bigram'


@dataclass
class InductionRule:
    """a b ... a → b. Copy b whenever a recurs."""
    name: str
    trigger: str        # the recurring token, e.g. 'c'
    rule_class: str = 'induction'


@dataclass
class BracketRule:
    """( ... content → ). Opening bracket sets context; content raises ) prob."""
    name: str
    open_token: str = '('
    close_token: str = ')'
    content_tokens: list = field(default_factory=lambda: ENTITY[:4])
    max_content_len: int = 5
    rule_class: str = 'bracket'


@dataclass
class ChainedRule:
    """rule_A fires, placing output_A into stream; rule_B fires when output_A appears."""
    name: str
    rule_A: object      # any rule above that produces an output token
    rule_B: object      # rule triggered by rule_A's output
    rule_class: str = 'chained'


# ---------------------------------------------------------------------------
# Default rule sets
# ---------------------------------------------------------------------------

# ---------------------------------------------------------------------------
# Rule instances
#
# Design principles:
#   - 3 shared outputs (X0-X2): each produced by one bigram, one trigram,
#     AND one skip-bigram. The model must use context to disambiguate.
#   - 3 unique outputs per class (B0-B2, T0-T2, S0-S2): unambiguous
#     fingerprints identifying which mechanism fired.
#   - Trigrams reuse bigram entities but produce different outputs,
#     forcing the model to attend to the marker at position -2.
#   - Induction for ALL entities (a-h): copies whatever followed the
#     entity last time. Competes with other rules via mixing weights.
#   - No dedicated noise tokens; noise emits uniform over full vocab.
# ---------------------------------------------------------------------------

BIGRAM_RULES = [
    # Shared outputs (also produced by a trigram and a skip-bigram)
    BigramRule('bigram_a_X0', trigger='a', output='X0'),
    BigramRule('bigram_b_X1', trigger='b', output='X1'),
    BigramRule('bigram_c_X2', trigger='c', output='X2'),
    # Unique bigram outputs
    BigramRule('bigram_d_B0', trigger='d', output='B0'),
    BigramRule('bigram_e_B1', trigger='e', output='B1'),
    BigramRule('bigram_f_B2', trigger='f', output='B2'),
]

# Trigrams: prev_token + entity → output.
# Two types of trigger1:
#   - Entity trigger1 (g, h): reliable, since g/h have no bigram rules.
#     Sequence: ... g d → (bigram d→B0 vs trigram g,d→X0)
#   - Marker trigger1 ($, !, #, @): less frequent, since skip-bigrams may
#     intercept. Using $ and ! first (fewest skip-bigram anchors).
TRIGRAM_RULES = [
    # Shared outputs — entity trigger1 (reliable)
    TrigramRule('trigram_gd_X0',      trigger1='g', trigger2='d', output='X0'),
    TrigramRule('trigram_he_X1',      trigger1='h', trigger2='e', output='X1'),
    # Shared output — marker trigger1 ($ has only 1 skip anchor: 'a')
    TrigramRule('trigram_dollar_f_X2', trigger1='$', trigger2='f', output='X2'),
    # Unique trigram outputs — marker trigger1
    TrigramRule('trigram_bang_a_T0',   trigger1='!', trigger2='a', output='T0'),
    TrigramRule('trigram_hash_b_T1',   trigger1='#', trigger2='b', output='T1'),
    TrigramRule('trigram_at_c_T2',     trigger1='@', trigger2='c', output='T2'),
]

# Skip-bigrams: anchor ... marker → output.
SKIP_BIGRAM_RULES = [
    # Shared outputs (different trigger than bigram/trigram for same output)
    SkipBigramRule('skip_g_hash_X0',    anchor='g', current_marker='#', output='X0', max_skip=4),
    SkipBigramRule('skip_h_at_X1',      anchor='h', current_marker='@', output='X1', max_skip=4),
    SkipBigramRule('skip_a_dollar_X2',  anchor='a', current_marker='$', output='X2', max_skip=4),
    # Unique skip-bigram outputs
    SkipBigramRule('skip_b_bang_S0',    anchor='b', current_marker='!', output='S0', max_skip=4),
    SkipBigramRule('skip_c_hash_S1',    anchor='c', current_marker='#', output='S1', max_skip=4),
    SkipBigramRule('skip_d_at_S2',      anchor='d', current_marker='@', output='S2', max_skip=4),
]

# Induction: ALL entities. "a X ... a" → predict X (copy what followed a last time).
# Competes with bigrams/trigrams/skip-bigrams via mixing weights (~50% override).
INDUCTION_RULES = [
    InductionRule('induction_a', trigger='a'),
    InductionRule('induction_b', trigger='b'),
    InductionRule('induction_c', trigger='c'),
    InductionRule('induction_d', trigger='d'),
    InductionRule('induction_e', trigger='e'),
    InductionRule('induction_f', trigger='f'),
    InductionRule('induction_g', trigger='g'),
    InductionRule('induction_h', trigger='h'),
]

BRACKET_RULES = [
    BracketRule('bracket_paren'),
]

DEPTH1_RULES = BIGRAM_RULES + TRIGRAM_RULES + SKIP_BIGRAM_RULES
ALL_RULES = DEPTH1_RULES + INDUCTION_RULES + BRACKET_RULES


# ---------------------------------------------------------------------------
# Scalable vocab + rule factory
# ---------------------------------------------------------------------------

def make_scaled_language(n_entity=40, n_marker=16, n_output=40, n_noise=28,
                         n_bigram=30, n_trigram=30, n_skip=30):
    """Build a larger vocabulary and rule set programmatically.

    Returns a dict with keys: entities, markers, outputs, noise_tokens, bracket,
    vocab, token2id, id2token, vocab_size,
    bigram_rules, trigram_rules, skip_bigram_rules, depth1_rules, depth1_mixing.
    """
    entities = [f'e{i}' for i in range(n_entity)]
    markers = [f'm{i}' for i in range(n_marker)]
    outputs = [f'o{i}' for i in range(n_output)]
    noise_tokens = [f'n{i}' for i in range(n_noise)]
    bracket = ['(', ')']

    vocab = entities + markers + outputs + noise_tokens + bracket
    token2id = {t: i for i, t in enumerate(vocab)}
    id2token = {i: t for t, i in token2id.items()}

    # Bigrams: entity_i → output_(i % n_output)
    bigram_rules = []
    for i in range(min(n_bigram, n_entity)):
        e, o = entities[i], outputs[i % n_output]
        bigram_rules.append(BigramRule(f'bigram_{e}_{o}', trigger=e, output=o))

    # Trigrams: marker_j + entity_k → output_m  (different output than bigram for same entity)
    trigram_rules = []
    for i in range(n_trigram):
        m = markers[i % n_marker]
        e = entities[i % n_entity]
        o = outputs[(i + n_bigram) % n_output]  # offset so outputs differ from bigrams
        trigram_rules.append(TrigramRule(f'trigram_{m}_{e}_{o}', trigger1=m, trigger2=e, output=o))

    # Skip-bigrams: entity_k ... marker_j → output_m
    skip_rules = []
    for i in range(n_skip):
        e = entities[(i + n_bigram) % n_entity]  # offset from bigram entities
        m = markers[i % n_marker]
        o = outputs[(i + n_bigram + n_trigram) % n_output]
        skip_rules.append(SkipBigramRule(f'skip_{e}_{m}_{o}', anchor=e, current_marker=m, output=o))

    depth1_rules = bigram_rules + trigram_rules + skip_rules

    depth1_mixing = {
        'bigram':      0.38,
        'trigram':     0.31,
        'skip_bigram': 0.30,
        'noise':       0.01,
    }

    return {
        'entities': entities, 'markers': markers, 'outputs': outputs,
        'noise_tokens': noise_tokens, 'bracket': bracket,
        'vocab': vocab, 'token2id': token2id, 'id2token': id2token,
        'vocab_size': len(vocab),
        'bigram_rules': bigram_rules, 'trigram_rules': trigram_rules,
        'skip_bigram_rules': skip_rules, 'depth1_rules': depth1_rules,
        'depth1_mixing': depth1_mixing,
    }


# Pre-built 128-token language
LANG128 = make_scaled_language(
    n_entity=40, n_marker=16, n_output=40, n_noise=30, n_bigram=30, n_trigram=30, n_skip=30,
)

DEPTH1_MIXING = {
    'bigram':     0.38,
    'trigram':    0.31,
    'skip_bigram':0.30,
    'noise':      0.01,  # uniform over all vocab
}

DEFAULT_MIXING = {
    'bigram':     0.25,
    'trigram':    0.22,
    'skip_bigram':0.22,
    'induction':  0.20,  # ~50% override when competing with any single rule
    'bracket':    0.01,
    'noise':      0.10,  # uniform over all vocab (no dedicated noise tokens)
}


# ---------------------------------------------------------------------------
# Generator
# ---------------------------------------------------------------------------

class LanguageGenerator:
    def __init__(
        self,
        rules: list,
        mixing_weights: Optional[dict] = None,
        mode: str = 'mixed',   # 'isolated' or 'mixed'
        window_size: int = 16,
        seed: Optional[int] = None,
        vocab: Optional[list] = None,
        token2id: Optional[dict] = None,
        id2token: Optional[dict] = None,
    ):
        self.rules = rules
        self.mode = mode
        self.window_size = window_size
        self.rng = random.Random(seed)

        # Vocab — use custom or fall back to module-level default
        self.vocab = vocab or VOCAB
        self.token2id = token2id or TOKEN2ID
        self.id2token = id2token or ID2TOKEN

        # Group rules by class
        self.rules_by_class = defaultdict(list)
        for r in rules:
            self.rules_by_class[r.rule_class].append(r)

        self.mixing_weights = mixing_weights or DEFAULT_MIXING

        # Induction state: maps trigger token → list of tokens that followed it
        self._induction_memory = defaultdict(list)

    def _noise_token(self, context=None):
        # Noise emits uniformly from full vocab — no bias toward noise tokens.
        # This prevents noise→noise chains and makes noise positions unpredictable.
        return self.rng.choice(self.vocab)

    def _try_bigram(self, context):
        if not context:
            return None, None
        last = context[-1]
        for rule in self.rules_by_class['bigram']:
            if last == rule.trigger:
                return rule.output, rule.name
        return None, None

    def _try_trigram(self, context):
        if len(context) < 2:
            return None, None
        t1, t2 = context[-2], context[-1]
        for rule in self.rules_by_class['trigram']:
            if t1 == rule.trigger1 and t2 == rule.trigger2:
                return rule.output, rule.name
        return None, None

    def _try_skip_bigram(self, context):
        if not context:
            return None, None
        current = context[-1]
        for rule in self.rules_by_class['skip_bigram']:
            if current != rule.current_marker:
                continue
            window = context[-rule.max_skip-1:-1]
            if rule.anchor in window:
                return rule.output, rule.name
        return None, None

    def _try_induction(self, context):
        """If current token t appeared before and was followed by s, output s."""
        if not context:
            return None, None
        current = context[-1]
        for rule in self.rules_by_class['induction']:
            if current != rule.trigger:
                continue
            # Look back in context for a previous occurrence of trigger
            for i in range(len(context) - 2, -1, -1):
                if context[i] == rule.trigger and i + 1 < len(context) - 1:
                    # token that followed the previous occurrence
                    followed_by = context[i + 1]
                    return followed_by, rule.name
        return None, None

    def _emit_bracket_sequence(self):
        """Emit a full ( content ) sequence with rule labels."""
        if not self.rules_by_class['bracket']:
            return [], []
        rule = self.rng.choice(self.rules_by_class['bracket'])
        n = self.rng.randint(1, rule.max_content_len)
        content = [self.rng.choice(rule.content_tokens) for _ in range(n)]
        tokens = [rule.open_token] + content + [rule.close_token]
        labels = ['bracket_open'] + [f'bracket_content_{rule.name}'] * n + [f'bracket_close_{rule.name}']
        return tokens, labels

    def _sample_next_token(self, context):
        """
        Given context window, return (next_token, rule_label).

        Two phases:
        1. If any rule fires on the current context, dispatch via mixing weights.
        2. Otherwise, PROACTIVELY emit a token that sets up a future rule fire.
           This prevents long noise runs and keeps rule frequency close to mixing weights.
        """
        # --- Phase 1: check what rules fire right now ---
        candidates = {}

        t, label = self._try_trigram(context)
        if t: candidates['trigram'] = (t, label)

        t, label = self._try_bigram(context)
        if t: candidates['bigram'] = (t, label)

        t, label = self._try_skip_bigram(context)
        if t: candidates['skip_bigram'] = (t, label)

        t, label = self._try_induction(context)
        if t: candidates['induction'] = (t, label)

        if candidates:
            if self.mode == 'isolated':
                for cls in ['trigram', 'bigram', 'skip_bigram', 'induction']:
                    if cls in candidates:
                        return candidates[cls]

            # Mixed mode: weighted dispatch among fired rules
            active_classes = [c for c in candidates if c in self.mixing_weights]
            if active_classes:
                weights = [self.mixing_weights.get(c, 0.0) for c in active_classes]
                total = sum(weights)
                if total > 0:
                    chosen = self.rng.choices(active_classes, weights=weights, k=1)[0]
                    return candidates[chosen]

        # --- Phase 2: no rule fires — proactively set up a future rule ---

        # If last token is a trigram trigger1 (marker), complete the trigram setup.
        # Without this, trigrams are extremely rare because they need 2 consecutive
        # setup tokens and the random dispatch rarely chains them.
        if context and self.rules_by_class.get('trigram'):
            matching = [r for r in self.rules_by_class['trigram']
                        if r.trigger1 == context[-1]]
            if matching:
                rule = self.rng.choice(matching)
                return rule.trigger2, 'setup'

        setup_classes = [c for c in self.mixing_weights
                         if c != 'noise' and self.rules_by_class.get(c)]
        noise_w = self.mixing_weights.get('noise', 0.01)
        weights = [self.mixing_weights[c] for c in setup_classes] + [noise_w]
        options = setup_classes + ['noise']
        chosen = self.rng.choices(options, weights=weights, k=1)[0]

        if chosen == 'noise':
            return self._noise_token(context), 'noise'

        if chosen == 'bigram':
            rule = self.rng.choice(self.rules_by_class['bigram'])
            return rule.trigger, 'setup'

        if chosen == 'trigram':
            # If last token is already a trigram trigger1, emit matching trigger2
            if context:
                matching = [r for r in self.rules_by_class['trigram']
                            if r.trigger1 == context[-1]]
                if matching:
                    rule = self.rng.choice(matching)
                    return rule.trigger2, 'setup'
            # Otherwise emit a trigger1 (marker)
            rule = self.rng.choice(self.rules_by_class['trigram'])
            return rule.trigger1, 'setup'

        if chosen == 'skip_bigram':
            rule = self.rng.choice(self.rules_by_class['skip_bigram'])
            window = context[-rule.max_skip:] if context else []
            if rule.anchor in window:
                return rule.current_marker, 'setup'
            else:
                return rule.anchor, 'setup'

        if chosen == 'induction':
            rule = self.rng.choice(self.rules_by_class['induction'])
            return rule.trigger, 'setup'

        return self._noise_token(context), 'noise'

    def sample_sequence(self, length: int = 64):
        """
        Generate a token sequence of given length.
        Returns (token_ids, rule_labels).
        rule_labels[i] is the name of the rule that determined token i.

        Special case: bracket rules emit multi-token spans. If a bracket rule is
        triggered mid-sequence, the full span is inserted.
        """
        tokens = []
        labels = []

        # Seed context with a noise token
        context = [self._noise_token()]
        tokens.append(self.token2id[context[-1]])
        labels.append('noise_seed')

        while len(tokens) < length:
            # Check if we should start a bracket sequence
            if (self.rules_by_class['bracket'] and
                self.rng.random() < self.mixing_weights.get('bracket', 0.0) and
                len(tokens) + 3 <= length):
                span_tokens, span_labels = self._emit_bracket_sequence()
                for t, l in zip(span_tokens, span_labels):
                    if len(tokens) >= length:
                        break
                    tokens.append(self.token2id[t])
                    labels.append(l)
                    context = (context + [t])[-self.window_size:]
                continue

            next_tok, label = self._sample_next_token(context)
            tokens.append(self.token2id[next_tok])
            labels.append(label)
            context = (context + [next_tok])[-self.window_size:]

        return tokens[:length], labels[:length]

    def sample_batch(self, n_sequences: int, length: int = 64):
        """Generate a batch of sequences. Returns (token_ids, rule_labels) lists."""
        all_tokens, all_labels = [], []
        for _ in range(n_sequences):
            toks, labs = self.sample_sequence(length)
            all_tokens.append(toks)
            all_labels.append(labs)
        return all_tokens, all_labels


# ---------------------------------------------------------------------------
# Isolated test sets (for behavioral evaluation)
# ---------------------------------------------------------------------------

def make_isolated_test_set(rule, n=500, length=32, seed=42, lang=None):
    """
    Generate sequences using only one rule, for behavioral testing.
    Returns (sequences, labels, rule_fire_positions) where rule_fire_positions[i]
    is the list of token indices where the rule fires in sequence i.
    lang: optional dict from make_scaled_language() for custom vocab.
    """
    kwargs = {}
    if lang is not None:
        kwargs = dict(vocab=lang['vocab'], token2id=lang['token2id'], id2token=lang['id2token'])
    gen = LanguageGenerator(rules=[rule], mode='isolated', seed=seed, **kwargs)
    sequences, labels = gen.sample_batch(n, length)
    fire_positions = []
    for lab_seq in labels:
        positions = [i for i, l in enumerate(lab_seq) if l == rule.name or l.startswith(rule.name)]
        fire_positions.append(positions)
    return sequences, labels, fire_positions


# ---------------------------------------------------------------------------
# Fast vectorized generator
# ---------------------------------------------------------------------------

class FastLanguageGenerator:
    """
    Optimized batch generator using numpy for bulk random number pre-generation
    and pure-Python integer lookup tables for O(1) rule matching.

    Same API as LanguageGenerator but ~3-5x faster for sample_batch().
    Works entirely with integer token IDs internally, avoiding string lookups
    during the inner generation loop.

    Usage:
        gen = FastLanguageGenerator(rules=DEPTH1_RULES, mixing_weights=DEPTH1_MIXING,
                                     mode='mixed', seed=42)
        tokens, labels = gen.sample_batch(128, length=257)
    """

    def __init__(
        self,
        rules: list,
        mixing_weights: Optional[dict] = None,
        mode: str = 'mixed',
        window_size: int = 16,
        seed: Optional[int] = None,
        vocab: Optional[list] = None,
        token2id: Optional[dict] = None,
        id2token: Optional[dict] = None,
    ):
        self.rules = rules
        self.mode = mode
        self.window_size = window_size
        self.np_rng = np.random.RandomState(seed if seed is not None else 42)

        # Vocab — use custom or fall back to module-level default
        self.vocab = vocab or VOCAB
        self.token2id = token2id or TOKEN2ID
        self.id2token = id2token or ID2TOKEN
        self.vocab_size = len(self.vocab)

        # Group rules by class
        self.rules_by_class = defaultdict(list)
        for r in rules:
            self.rules_by_class[r.rule_class].append(r)

        self.mixing_weights = mixing_weights or DEFAULT_MIXING

        # Build pure-Python lookup tables (no numpy in hot path)
        self._build_lookup_tables()
        self._build_mixing_tables()

    def _build_lookup_tables(self):
        """Build pure-Python O(1) lookup tables for rule matching."""
        t2i = self.token2id

        # --- Bigram: dict last_token_id -> (output_id, label_str) ---
        self._bigram_lut = {}  # int -> (int, str)
        for rule in self.rules_by_class['bigram']:
            self._bigram_lut[t2i[rule.trigger]] = (t2i[rule.output], rule.name)

        # Bigram triggers as a list (for setup phase random choice)
        self._bigram_trigger_list = [t2i[r.trigger] for r in self.rules_by_class['bigram']]

        # --- Trigram: dict (t1_id, t2_id) -> (output_id, label_str) ---
        self._trigram_lut = {}
        for rule in self.rules_by_class['trigram']:
            self._trigram_lut[(t2i[rule.trigger1], t2i[rule.trigger2])] = (t2i[rule.output], rule.name)

        # Trigram trigger1 -> list of trigger2 ids (for setup phase)
        self._trigram_by_t1 = {}  # int -> list[int]
        for rule in self.rules_by_class['trigram']:
            t1 = t2i[rule.trigger1]
            if t1 not in self._trigram_by_t1:
                self._trigram_by_t1[t1] = []
            self._trigram_by_t1[t1].append(t2i[rule.trigger2])
        self._has_trigrams = bool(self.rules_by_class['trigram'])

        # All trigram trigger1 tokens as list (for setup: emit a trigger1)
        self._trigram_t1_list = [t2i[r.trigger1] for r in self.rules_by_class['trigram']]

        # --- Skip-bigram: for each marker_id, list of (anchor_id, output_id, max_skip, label) ---
        self._skip_by_marker = {}  # int -> list of tuples
        for rule in self.rules_by_class['skip_bigram']:
            mid = t2i[rule.current_marker]
            if mid not in self._skip_by_marker:
                self._skip_by_marker[mid] = []
            self._skip_by_marker[mid].append(
                (t2i[rule.anchor], t2i[rule.output], rule.max_skip, rule.name)
            )

        # Skip-bigram rule arrays for setup phase
        n_skip = len(self.rules_by_class['skip_bigram'])
        self._skip_anchors = [t2i[r.anchor] for r in self.rules_by_class['skip_bigram']]
        self._skip_markers = [t2i[r.current_marker] for r in self.rules_by_class['skip_bigram']]
        self._skip_max_skips = [r.max_skip for r in self.rules_by_class['skip_bigram']]
        self._n_skip = n_skip

        # --- Induction: set of trigger token IDs ---
        self._induction_triggers = {}  # int -> str (trigger_id -> label)
        for rule in self.rules_by_class['induction']:
            self._induction_triggers[t2i[rule.trigger]] = rule.name
        self._induction_trigger_list = [t2i[r.trigger] for r in self.rules_by_class['induction']]
        self._has_induction = bool(self.rules_by_class['induction'])

        # --- Bracket ---
        self._has_brackets = bool(self.rules_by_class['bracket'])
        if self._has_brackets:
            self._bracket_data = []
            for rule in self.rules_by_class['bracket']:
                self._bracket_data.append((
                    t2i[rule.open_token],
                    t2i[rule.close_token],
                    [t2i[t] for t in rule.content_tokens],
                    rule.max_content_len,
                    rule.name,
                ))

    def _build_mixing_tables(self):
        """Pre-compute cumulative weight tables for fast weighted random choice."""
        mw = self.mixing_weights

        # Phase 1 class order (must match LanguageGenerator priority)
        self._p1_classes = []  # list of (class_name, weight)
        for cls in ['trigram', 'bigram', 'skip_bigram', 'induction']:
            w = mw.get(cls, 0.0)
            if w > 0:
                self._p1_classes.append((cls, w))

        # Phase 2 setup: cumulative probabilities
        setup_classes = [c for c in mw if c != 'noise' and self.rules_by_class.get(c)]
        noise_w = mw.get('noise', 0.01)
        options = setup_classes + ['noise']
        weights = [mw[c] for c in setup_classes] + [noise_w]
        total = sum(weights)
        cum = 0.0
        self._setup_cum = []  # list of (cumulative_prob, class_name)
        for opt, w in zip(options, weights):
            cum += w / total
            self._setup_cum.append((cum, opt))

        # Number of rules per class
        self._n_bigram = len(self.rules_by_class['bigram'])
        self._n_trigram = len(self.rules_by_class['trigram'])
        self._n_induction = len(self.rules_by_class['induction'])

        # Bracket probability
        self._bracket_prob = mw.get('bracket', 0.0)

        # Isolated mode priority order
        self._isolated_priority = ['trigram', 'bigram', 'skip_bigram', 'induction']

    def sample_batch(self, n_sequences: int, length: int = 64):
        """
        Generate a batch of sequences.
        Returns (token_ids, rule_labels) — same format as LanguageGenerator.sample_batch().

        Uses bulk numpy random generation + pure Python inner loop with O(1) lookups.
        """
        B = n_sequences
        L = length
        V = self.vocab_size
        ws = self.window_size

        # Pre-generate ALL random numbers in bulk (the main optimization)
        max_steps = L + 20
        total_rands = B * max_steps

        # Flatten for sequential consumption — avoids 2D indexing overhead
        phase1_flat = self.np_rng.random(total_rands)
        setup_flat = self.np_rng.random(total_rands)
        rule_flat = self.np_rng.random(total_rands)
        noise_flat = self.np_rng.randint(0, V, size=total_rands)
        bracket_flat = self.np_rng.random(total_rands)
        bracket_len_flat = self.np_rng.randint(1, 6, size=total_rands)
        bracket_content_flat = self.np_rng.randint(0, V, size=total_rands)

        # Pre-fetch lookup tables as local variables (avoids self. overhead in tight loop)
        bigram_lut = self._bigram_lut
        trigram_lut = self._trigram_lut
        skip_by_marker = self._skip_by_marker
        induction_triggers = self._induction_triggers
        has_trigrams = self._has_trigrams
        has_induction = self._has_induction
        has_brackets = self._has_brackets
        bracket_prob = self._bracket_prob
        mode_isolated = (self.mode == 'isolated')
        p1_classes = self._p1_classes
        setup_cum = self._setup_cum
        trigram_by_t1 = self._trigram_by_t1
        bigram_trigger_list = self._bigram_trigger_list
        trigram_t1_list = self._trigram_t1_list
        skip_anchors = self._skip_anchors
        skip_markers = self._skip_markers
        skip_max_skips = self._skip_max_skips
        n_skip = self._n_skip
        n_bigram = self._n_bigram
        n_trigram = self._n_trigram
        n_induction = self._n_induction
        induction_trigger_list = self._induction_trigger_list
        isolated_priority = self._isolated_priority

        all_tokens = []
        all_labels = []

        ri = 0  # running random index

        for b in range(B):
            tokens = [0] * L  # pre-allocate
            labels = [None] * L
            # Context as a plain Python list (faster than numpy for small sizes)
            ctx = []
            tok_count = 0

            # Seed token
            seed_id = int(noise_flat[ri])
            tokens[0] = seed_id
            labels[0] = 'noise_seed'
            ctx.append(seed_id)
            tok_count = 1
            ri += 1

            while tok_count < L:
                r_phase1 = phase1_flat[ri]
                r_setup = setup_flat[ri]
                r_rule = rule_flat[ri]
                r_noise = int(noise_flat[ri])
                r_bracket = bracket_flat[ri]
                r_bracket_len = int(bracket_len_flat[ri])
                r_bracket_content_base = ri  # index into bracket_content_flat

                # --- Bracket check ---
                if has_brackets and r_bracket < bracket_prob and tok_count + 3 <= L:
                    bd = self._bracket_data[0]
                    open_id, close_id, content_pool, max_clen, bname = bd
                    n_content = min(r_bracket_len, max_clen)
                    n_pool = len(content_pool)
                    # open
                    tokens[tok_count] = open_id
                    labels[tok_count] = 'bracket_open'
                    ctx.append(open_id)
                    if len(ctx) > ws:
                        ctx = ctx[-ws:]
                    tok_count += 1
                    # content
                    content_label = f'bracket_content_{bname}'
                    for ci in range(n_content):
                        if tok_count >= L:
                            break
                        cid = content_pool[int(bracket_content_flat[r_bracket_content_base + ci]) % n_pool]
                        tokens[tok_count] = cid
                        labels[tok_count] = content_label
                        ctx.append(cid)
                        if len(ctx) > ws:
                            ctx = ctx[-ws:]
                        tok_count += 1
                    # close
                    if tok_count < L:
                        tokens[tok_count] = close_id
                        labels[tok_count] = f'bracket_close_{bname}'
                        ctx.append(close_id)
                        if len(ctx) > ws:
                            ctx = ctx[-ws:]
                        tok_count += 1
                    ri += 1
                    continue

                # --- Normal token: Phase 1 (rule fire check) ---
                ctx_len = len(ctx)
                last_id = ctx[-1]
                prev_id = ctx[-2] if ctx_len >= 2 else -1

                # Check which rules fire
                tri_result = trigram_lut.get((prev_id, last_id))
                bi_result = bigram_lut.get(last_id)

                # Skip-bigram
                skip_result = None
                skip_rules = skip_by_marker.get(last_id)
                if skip_rules:
                    for anchor_id, output_id, max_skip, slabel in skip_rules:
                        start = max(0, ctx_len - max_skip - 1)
                        end = ctx_len - 1
                        if end > start:
                            for si in range(start, end):
                                if ctx[si] == anchor_id:
                                    skip_result = (output_id, slabel)
                                    break
                        if skip_result:
                            break

                # Induction
                ind_result = None
                if has_induction and last_id in induction_triggers:
                    for si in range(ctx_len - 2, -1, -1):
                        if ctx[si] == last_id and si + 1 < ctx_len - 1:
                            ind_result = (ctx[si + 1], induction_triggers[last_id])
                            break

                # Collect candidates
                has_any = tri_result or bi_result or skip_result or ind_result
                if has_any:
                    if mode_isolated:
                        if tri_result:
                            tok_id, tok_label = tri_result
                        elif bi_result:
                            tok_id, tok_label = bi_result
                        elif skip_result:
                            tok_id, tok_label = skip_result
                        else:
                            tok_id, tok_label = ind_result
                    else:
                        # Mixed mode: weighted dispatch among fired rules
                        # Build candidate list and weights inline
                        cands = []
                        total_w = 0.0
                        if tri_result:
                            w = self.mixing_weights.get('trigram', 0.0)
                            cands.append((w, tri_result))
                            total_w += w
                        if bi_result:
                            w = self.mixing_weights.get('bigram', 0.0)
                            cands.append((w, bi_result))
                            total_w += w
                        if skip_result:
                            w = self.mixing_weights.get('skip_bigram', 0.0)
                            cands.append((w, skip_result))
                            total_w += w
                        if ind_result:
                            w = self.mixing_weights.get('induction', 0.0)
                            cands.append((w, ind_result))
                            total_w += w

                        if total_w > 0:
                            target = r_phase1 * total_w
                            cum = 0.0
                            chosen_result = cands[-1][1]  # default to last
                            for w, result in cands:
                                cum += w
                                if cum > target:
                                    chosen_result = result
                                    break
                            tok_id, tok_label = chosen_result
                        else:
                            tok_id, tok_label = r_noise, 'noise'

                    tokens[tok_count] = tok_id
                    labels[tok_count] = tok_label
                    ctx.append(tok_id)
                    if len(ctx) > ws:
                        ctx = ctx[-ws:]
                    tok_count += 1
                    ri += 1
                    continue

                # --- Phase 2: proactive setup ---

                # Special trigram completion: if last token is a trigger1
                if has_trigrams and last_id in trigram_by_t1:
                    t2_list = trigram_by_t1[last_id]
                    tok_id = t2_list[int(r_rule * len(t2_list)) % len(t2_list)]
                    tokens[tok_count] = tok_id
                    labels[tok_count] = 'setup'
                    ctx.append(tok_id)
                    if len(ctx) > ws:
                        ctx = ctx[-ws:]
                    tok_count += 1
                    ri += 1
                    continue

                # Weighted choice of setup class
                chosen_cls = 'noise'
                for cum_p, cls_name in setup_cum:
                    if r_setup < cum_p:
                        chosen_cls = cls_name
                        break

                if chosen_cls == 'noise':
                    tok_id = r_noise
                    tok_label = 'noise'
                elif chosen_cls == 'bigram':
                    tok_id = bigram_trigger_list[int(r_rule * n_bigram) % n_bigram]
                    tok_label = 'setup'
                elif chosen_cls == 'trigram':
                    if last_id in trigram_by_t1:
                        t2_list = trigram_by_t1[last_id]
                        tok_id = t2_list[int(r_rule * len(t2_list)) % len(t2_list)]
                    else:
                        tok_id = trigram_t1_list[int(r_rule * n_trigram) % n_trigram]
                    tok_label = 'setup'
                elif chosen_cls == 'skip_bigram':
                    ridx = int(r_rule * n_skip) % n_skip
                    anchor_id = skip_anchors[ridx]
                    marker_id = skip_markers[ridx]
                    max_skip = skip_max_skips[ridx]
                    start = max(0, ctx_len - max_skip)
                    found_anchor = False
                    for si in range(start, ctx_len):
                        if ctx[si] == anchor_id:
                            found_anchor = True
                            break
                    tok_id = marker_id if found_anchor else anchor_id
                    tok_label = 'setup'
                elif chosen_cls == 'induction':
                    tok_id = induction_trigger_list[int(r_rule * n_induction) % n_induction]
                    tok_label = 'setup'
                else:
                    tok_id = r_noise
                    tok_label = 'noise'

                tokens[tok_count] = tok_id
                labels[tok_count] = tok_label
                ctx.append(tok_id)
                if len(ctx) > ws:
                    ctx = ctx[-ws:]
                tok_count += 1
                ri += 1

            all_tokens.append(tokens)
            all_labels.append(labels)

        return all_tokens, all_labels

    def sample_sequence(self, length: int = 64):
        """Generate a single sequence. Returns (token_ids, rule_labels)."""
        tokens, labels = self.sample_batch(1, length)
        return tokens[0], labels[0]


# ---------------------------------------------------------------------------
# Cached data loader
# ---------------------------------------------------------------------------

class CachedDataLoader:
    """
    Pre-generates and caches synthetic data to disk for fast training.

    Usage:
        # Generate cache
        loader = CachedDataLoader.generate(
            generator, n_sequences=50000, seq_len=257,
            save_path='attn_circuits/data/cache_depth1.npz'
        )

        # Load cache
        loader = CachedDataLoader('attn_circuits/data/cache_depth1.npz')

        # Sample batch (random indices from cache)
        tokens, labels = loader.sample_batch(batch_size=128)
    """

    def __init__(self, cache_path: str, seed: int = 42):
        """Load a pre-generated cache from disk."""
        data = np.load(cache_path, allow_pickle=True)
        self.token_ids = data['token_ids']  # (N, L) int32
        self.rule_labels = data['rule_labels']  # (N, L) object array of strings
        self.n_sequences = self.token_ids.shape[0]
        self.seq_len = self.token_ids.shape[1]
        self.rng = np.random.RandomState(seed)
        print(f"CachedDataLoader: loaded {self.n_sequences} sequences of length {self.seq_len} from {cache_path}")

    def sample_batch(self, batch_size: int, length: int = None):
        """
        Sample a random batch from the cache.
        Returns (token_ids, rule_labels) in the same format as LanguageGenerator.sample_batch().
        """
        indices = self.rng.randint(0, self.n_sequences, size=batch_size)
        L = length if length is not None else self.seq_len
        L = min(L, self.seq_len)
        tokens = self.token_ids[indices, :L]
        labels = self.rule_labels[indices, :L]

        # Convert to list-of-lists format matching LanguageGenerator API
        all_tokens = [row.tolist() for row in tokens]
        all_labels = [row.tolist() for row in labels]
        return all_tokens, all_labels

    @staticmethod
    def generate(generator, n_sequences: int, seq_len: int, save_path: str,
                 batch_size: int = 1000, verbose: bool = True):
        """
        Generate data using a LanguageGenerator or FastLanguageGenerator and save to disk.
        Returns a CachedDataLoader for the generated cache.

        Args:
            generator: LanguageGenerator or FastLanguageGenerator instance
            n_sequences: total number of sequences to generate
            seq_len: length of each sequence (use seq_len+1 for input/target split)
            save_path: path to save the .npz file
            batch_size: how many sequences to generate per batch (for progress)
            verbose: print progress
        """
        os.makedirs(os.path.dirname(save_path) if os.path.dirname(save_path) else '.', exist_ok=True)

        all_tokens = np.zeros((n_sequences, seq_len), dtype=np.int32)
        all_labels = np.empty((n_sequences, seq_len), dtype=object)

        generated = 0
        while generated < n_sequences:
            bs = min(batch_size, n_sequences - generated)
            tokens, labels = generator.sample_batch(bs, length=seq_len)
            for i in range(bs):
                all_tokens[generated + i] = tokens[i]
                all_labels[generated + i] = labels[i]
            generated += bs
            if verbose and generated % 5000 == 0:
                print(f"  Generated {generated}/{n_sequences} sequences...")

        np.savez_compressed(save_path, token_ids=all_tokens, rule_labels=all_labels)
        if verbose:
            size_mb = os.path.getsize(save_path) / (1024 * 1024)
            print(f"Cache saved: {save_path} ({size_mb:.1f} MB, {n_sequences} x {seq_len})")

        return CachedDataLoader(save_path)


# ---------------------------------------------------------------------------
# Quick demo
# ---------------------------------------------------------------------------

if __name__ == '__main__':
    print("=== Bigram only ===")
    gen = LanguageGenerator(rules=BIGRAM_RULES, mode='isolated', seed=0)
    toks, labs = gen.sample_sequence(24)
    print(' '.join(ID2TOKEN[t] for t in toks))
    print(' '.join(labs))

    print("\n=== Induction only ===")
    gen = LanguageGenerator(rules=INDUCTION_RULES, mode='isolated', seed=0)
    toks, labs = gen.sample_sequence(32)
    print(' '.join(ID2TOKEN[t] for t in toks))
    print(' '.join(labs))

    print("\n=== Mixed ===")
    gen = LanguageGenerator(rules=ALL_RULES, mode='mixed', seed=0)
    toks, labs = gen.sample_sequence(48)
    print(' '.join(ID2TOKEN[t] for t in toks))
    print(' '.join(labs))
