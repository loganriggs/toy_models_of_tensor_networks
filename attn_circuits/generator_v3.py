"""
Synthetic language v3: realistic, readable, fully predictable.

Every token is predictable by at least one rule. Rules fire via priority-based
deterministic dispatch (no random mixing). Sequences read like simplified English.

Vocabulary: scalable (default 54 tokens)
  6 categories × N tokens each + 6 structural tokens
  N is configurable (default 8, supports up to 32 per category)

Rule classes (priority order, highest first):
  1. Bracket content/close  — inside ( ... ) or [ ... ] spans
     Close tokens are labeled 'skip_trigram' (model must attend back to opener)
  2. Token trigram           — specific 2-token context → specific output
  3. Category trigram        — (FUNC, NOUN) → sample from VERB_I
  4. Skip-bigram             — anchor ... trigger (within 8 pos) → specific output
  5. Induction               — repeated noun → copy what followed it earlier
  6. Noun bigram             — probabilistic: alice → sees(70%)/helps(20%)/finds(10%)
  7. Place bigram            — deterministic: park → the, home → and, etc.
  8. Category bigram         — POS transition → sample from target category

Bracket types:
  - Parentheses ( ... ): ADJ + NOUN content with special in-bracket transitions.
    After ) → a NOUN follows. Reads like noun phrases: ( big alice old carol )
  - Quotes [ ... ]: VERB_I + FUNC content with special in-bracket transitions.
    After ] → a FUNC follows. Reads like action lists: [ runs to walks and ]

Design choices:
  - NO pre-context. Induction works from within the sequence itself:
    first occurrence of a noun uses its probabilistic bigram, subsequent
    occurrences copy what followed the first occurrence.
  - Noun bigrams are PROBABILISTIC (70/20/10 over 3 verbs per noun).
    This incentivizes induction: if the first alice → 'finds' (10%),
    induction on the second alice gives 0 loss vs bigram's ~0.8 nats.
  - PLACE bigrams are deterministic (park → the, etc.)
  - Token bigrams do NOT chain: outputs of bigrams are NOT triggers for
    other bigrams.
  - Category trigrams: (FUNC, NOUN) → VERB_I. Fires whenever any function
    word is followed by any noun.
  - In-bracket transitions differ from global POS transitions — model must
    learn context-dependent distributions (attending to opener).
  - Category bigrams serve as universal fallback — every position is predictable.

Usage:
    gen = LanguageV3(seed=42)
    tokens, labels = gen.sample_sequence(length=64)

    # Larger vocab
    gen = LanguageV3(tokens_per_category=16, seed=42)  # 102 tokens

    # Print readable sequence
    print(' '.join(gen.id2token[t] for t in tokens))
"""

import random
import numpy as np
from typing import Optional
from collections import defaultdict


# ---------------------------------------------------------------------------
# Word pools (up to 32 per category)
# ---------------------------------------------------------------------------

WORD_POOLS = {
    'NOUN': [
        'alice', 'bob', 'carol', 'dave', 'eve', 'frank', 'grace', 'henry',
        'iris', 'jack', 'kate', 'leo', 'mia', 'nick', 'olive', 'paul',
        'quinn', 'rosa', 'sam', 'tina', 'uma', 'vic', 'wendy', 'xander',
        'yara', 'zane', 'amber', 'blake', 'clara', 'derek', 'emma', 'finn',
    ],
    'PLACE': [
        'park', 'home', 'store', 'school', 'beach', 'office', 'garden', 'lake',
        'tower', 'farm', 'hill', 'bridge', 'plaza', 'court', 'field', 'grove',
        'harbor', 'island', 'jungle', 'keep', 'lodge', 'market', 'nook', 'oasis',
        'pier', 'ranch', 'summit', 'trail', 'valley', 'wharf', 'yard', 'zone',
    ],
    'VERB_T': [
        'sees', 'helps', 'finds', 'knows', 'likes', 'meets', 'calls', 'tells',
        'asks', 'grabs', 'holds', 'leads', 'moves', 'needs', 'picks', 'reads',
        'saves', 'takes', 'uses', 'wants', 'brings', 'checks', 'draws', 'feeds',
        'gets', 'hears', 'joins', 'keeps', 'lifts', 'marks', 'names', 'owns',
    ],
    'VERB_I': [
        'runs', 'walks', 'sits', 'waits', 'jumps', 'falls', 'swims', 'plays',
        'rests', 'stays', 'stops', 'turns', 'works', 'sleeps', 'sings', 'dances',
        'climbs', 'crawls', 'drifts', 'flies', 'glides', 'hides', 'laughs', 'naps',
        'pauses', 'races', 'skips', 'thinks', 'trips', 'yawns', 'blinks', 'waves',
    ],
    'ADJ': [
        'big', 'small', 'old', 'new', 'red', 'blue', 'fast', 'tall',
        'dark', 'bright', 'warm', 'cool', 'soft', 'loud', 'shy', 'bold',
        'calm', 'deep', 'dry', 'flat', 'green', 'hot', 'keen', 'light',
        'mild', 'neat', 'pale', 'raw', 'slim', 'thin', 'vast', 'wide',
    ],
    'FUNC': [
        'the', 'a', 'in', 'on', 'at', 'to', 'and', 'or',
        'by', 'for', 'with', 'from', 'near', 'but', 'so', 'yet',
        'then', 'thus', 'once', 'here', 'there', 'where', 'when', 'while',
        'since', 'until', 'after', 'before', 'above', 'below', 'among', 'upon',
    ],
}

STRUCTURAL = ['.', ',', '(', ')', '[', ']']

# Probabilities for noun probabilistic bigrams (3 outcomes per noun)
NOUN_BIGRAM_PROBS = [0.7, 0.2, 0.1]


def build_vocabulary(tokens_per_category: int = 8):
    """Build vocabulary from word pools, taking first N tokens per category."""
    assert 1 <= tokens_per_category <= 32, \
        f"tokens_per_category must be 1-32, got {tokens_per_category}"

    categories = {}
    for cat in ['NOUN', 'PLACE', 'VERB_T', 'VERB_I', 'ADJ', 'FUNC']:
        categories[cat] = WORD_POOLS[cat][:tokens_per_category]

    vocab = []
    for cat in ['NOUN', 'PLACE', 'VERB_T', 'VERB_I', 'ADJ', 'FUNC']:
        vocab.extend(categories[cat])
    vocab.extend(STRUCTURAL)

    token2id = {t: i for i, t in enumerate(vocab)}
    id2token = {i: t for t, i in token2id.items()}

    token2cat = {}
    for cat, tokens in categories.items():
        for t in tokens:
            token2cat[t] = cat
    token2cat['.'] = 'PUNCT'
    token2cat[','] = 'PUNCT'
    token2cat['('] = 'OPEN'
    token2cat[')'] = 'CLOSE'
    token2cat['['] = 'OPEN'
    token2cat[']'] = 'CLOSE'

    cat2tokens = dict(categories)
    cat2tokens['PUNCT'] = ['.', ',']

    return categories, vocab, token2id, id2token, token2cat, cat2tokens


# ---------------------------------------------------------------------------
# Default vocabulary (N=8, backward compatible)
# ---------------------------------------------------------------------------

_default = build_vocabulary(8)
CATEGORIES = _default[0]
VOCAB = _default[1]
TOKEN2ID = _default[2]
ID2TOKEN = _default[3]
TOKEN2CAT = _default[4]
CAT2TOKENS = _default[5]
VOCAB_SIZE = len(VOCAB)


# ---------------------------------------------------------------------------
# POS transition matrix (simplified from Brown corpus, made peaked for toy model)
# ---------------------------------------------------------------------------

CAT_ORDER = ['NOUN', 'PLACE', 'VERB_T', 'VERB_I', 'ADJ', 'FUNC', 'PUNCT']

TRANSITION_MATRIX = np.array([
    # NOUN  PLACE VERB_T VERB_I ADJ   FUNC  PUNCT
    [0.05,  0.10, 0.25,  0.15,  0.00, 0.15, 0.30],  # NOUN →
    [0.05,  0.00, 0.05,  0.05,  0.00, 0.40, 0.45],  # PLACE →
    [0.35,  0.30, 0.00,  0.00,  0.10, 0.15, 0.10],  # VERB_T →
    [0.00,  0.15, 0.00,  0.00,  0.00, 0.35, 0.50],  # VERB_I →
    [0.45,  0.40, 0.00,  0.00,  0.10, 0.05, 0.00],  # ADJ →
    [0.25,  0.25, 0.05,  0.05,  0.35, 0.05, 0.00],  # FUNC →
    [0.05,  0.00, 0.10,  0.10,  0.05, 0.70, 0.00],  # PUNCT →
])

assert np.allclose(TRANSITION_MATRIX.sum(axis=1), 1.0)

CAT_INDEX = {cat: i for i, cat in enumerate(CAT_ORDER)}


# ---------------------------------------------------------------------------
# Bracket types configuration
# ---------------------------------------------------------------------------

BRACKET_TYPES = {
    'paren': {
        'open': '(',
        'close': ')',
        'transitions': {
            # Inside parens: ADJ ↔ NOUN with specific distribution
            'ADJ':  [('ADJ', 0.4), ('NOUN', 0.6)],
            'NOUN': [('ADJ', 0.75), ('NOUN', 0.25)],
        },
        'start_cat': 'ADJ',       # first content token is ADJ
        'post_close_cat': 'NOUN', # after ) → a NOUN follows
        'min_len': 2,
        'max_len': 4,
    },
    'quote': {
        'open': '[',
        'close': ']',
        'transitions': {
            # Inside quotes: VERB_I ↔ FUNC (action sequences)
            'VERB_I': [('VERB_I', 0.3), ('FUNC', 0.7)],
            'FUNC':   [('VERB_I', 0.8), ('FUNC', 0.2)],
        },
        'start_cat': 'VERB_I',    # first content token is VERB_I
        'post_close_cat': 'FUNC', # after ] → a FUNC follows
        'min_len': 2,
        'max_len': 4,
    },
}

# Skip-bigram config
SKIP_BIGRAM_MAX_SKIP = 8  # anchor can be up to 8 positions back


# ---------------------------------------------------------------------------
# Rule builder
# ---------------------------------------------------------------------------

def build_rules(categories, tokens_per_category):
    """Build rule sets scaled to vocabulary size.

    Returns:
        noun_bigram_dists: dict noun → list of (token, probability)
        place_bigrams: dict place → token (deterministic)
        tok_trigrams: dict (prev2, prev) → token
        cat_trigrams: dict (cat1, cat2) → output_cat
        skip_bigrams: dict (anchor, trigger) → output token
    """
    N = tokens_per_category
    n_bigram = max(N // 2, 1)

    nouns = categories['NOUN']
    places = categories['PLACE']
    verb_ts = categories['VERB_T']
    verb_is = categories['VERB_I']
    funcs = categories['FUNC']
    adjs = categories['ADJ']

    # Noun probabilistic bigrams: each noun gets 3 VERB_T options with 70/20/10
    # Cycle through VERB_T pool so each noun gets different verbs
    noun_bigram_dists = {}
    n_outcomes = len(NOUN_BIGRAM_PROBS)
    for i, noun in enumerate(nouns):
        outcomes = []
        for j in range(n_outcomes):
            verb_idx = (i * n_outcomes + j) % len(verb_ts)
            outcomes.append((verb_ts[verb_idx], NOUN_BIGRAM_PROBS[j]))
        noun_bigram_dists[noun] = outcomes

    # Place deterministic bigrams
    place_bigrams = {}
    for i in range(min(n_bigram, len(places), len(funcs))):
        place_bigrams[places[i]] = funcs[i]

    # Token trigrams
    tok_trigrams = {}
    the_tok = funcs[0] if funcs else 'the'
    for i in range(min(n_bigram, len(nouns), len(verb_is))):
        tok_trigrams[(the_tok, nouns[i])] = verb_is[i]
    func_triggers = funcs[2:2+n_bigram]  # 'in', 'on', 'at', 'to', ...
    for i in range(min(len(func_triggers), len(places), len(nouns))):
        tok_trigrams[(func_triggers[i], places[i])] = nouns[i]

    # Category trigrams
    cat_trigrams = {('FUNC', 'NOUN'): 'VERB_I'}

    # Skip-bigrams: (anchor, trigger) → output
    # Anchor PLACE seen within last max_skip positions + current is specific ADJ → output FUNC
    # Uses the second half of PLACEs and ADJs to avoid conflicts with other rules
    skip_bigrams = {}
    n_skip = min(n_bigram, len(places), len(adjs), len(funcs))
    for i in range(n_skip):
        anchor = places[n_bigram + i] if n_bigram + i < len(places) else places[i]
        trigger = adjs[i]
        # Output a FUNC word (from second half to avoid place_bigram outputs)
        output = funcs[n_bigram + i] if n_bigram + i < len(funcs) else funcs[i]
        skip_bigrams[(anchor, trigger)] = output

    return noun_bigram_dists, place_bigrams, tok_trigrams, cat_trigrams, skip_bigrams


# ---------------------------------------------------------------------------
# Generator
# ---------------------------------------------------------------------------

class LanguageV3:
    """Generates sequences from the v3 synthetic language.

    No pre-context. Induction works within the sequence: first occurrence
    of a noun uses its probabilistic bigram, subsequent occurrences copy
    what followed the first.

    Args:
        active_classes: Which rule classes are active. Options:
            'all', 'cat_bigram', 'noun_bigram', 'place_bigram',
            'tok_trigram', 'cat_trigram', 'induction', 'skip_bigram', 'bracket'.
        tokens_per_category: Number of tokens per category (1-32).
        paren_prob: Probability of inserting a parenthesis span at each position.
        quote_prob: Probability of inserting a quote span at each position.
        seed: Random seed for reproducibility.
    """

    # Labels that correspond to "normal" (non-bracket) context
    NORMAL_LABELS = frozenset({
        'cat_bigram', 'cat_trigram', 'noun_bigram', 'place_bigram',
        'skip_bigram', 'tok_trigram', 'induction',
    })

    def __init__(
        self,
        active_classes: Optional[list[str]] = None,
        tokens_per_category: int = 8,
        paren_prob: float = 0.03,
        quote_prob: float = 0.03,
        trigger_boost_prob: float = 0.20,
        trigger_boost_range: int = 4,
        seed: Optional[int] = None,
    ):
        self.active_classes = set(active_classes or ['all'])
        if 'all' in self.active_classes:
            self.active_classes = {
                'cat_bigram', 'noun_bigram', 'place_bigram',
                'tok_trigram', 'cat_trigram', 'induction',
                'skip_bigram', 'bracket'
            }
        self.tokens_per_category = tokens_per_category
        self.paren_prob = paren_prob
        self.quote_prob = quote_prob
        self.trigger_boost_prob = trigger_boost_prob
        self.trigger_boost_range = trigger_boost_range
        self.pre_context_len = 0  # no pre-context

        self.rng = random.Random(seed)

        # Build vocabulary
        (self.categories, self.vocab, self.token2id,
         self.id2token, self.token2cat, self.cat2tokens) = \
            build_vocabulary(tokens_per_category)
        self.vocab_size = len(self.vocab)

        # Build rules
        (noun_bigram_dists, place_bigrams,
         tok_trigrams, cat_trigrams, skip_bigrams) = \
            build_rules(self.categories, tokens_per_category)

        # Activate only requested rule classes
        self.noun_bigram_dists = noun_bigram_dists if 'noun_bigram' in self.active_classes else {}
        self.place_bigrams = place_bigrams if 'place_bigram' in self.active_classes else {}
        self.tok_trigrams = tok_trigrams if 'tok_trigram' in self.active_classes else {}
        self.cat_trigrams = cat_trigrams if 'cat_trigram' in self.active_classes else {}
        self.skip_bigrams = skip_bigrams if 'skip_bigram' in self.active_classes else {}
        self.use_induction = 'induction' in self.active_classes
        self.use_brackets = 'bracket' in self.active_classes
        self.use_cat_bigram = 'cat_bigram' in self.active_classes

        # All nouns are induction-eligible
        self.induction_triggers = set(self.categories['NOUN']) if self.use_induction else set()

        # Precompute anchor → trigger mapping for skip-bigram trigger boost
        self.skip_anchor_triggers: dict[str, list[str]] = defaultdict(list)
        for (anchor, trigger) in self.skip_bigrams:
            self.skip_anchor_triggers[anchor].append(trigger)

        # Transition matrix
        self.transition_matrix = TRANSITION_MATRIX
        self.cat_order = CAT_ORDER
        self.cat_index = CAT_INDEX

    def _sample_from_category(self, cat: str) -> str:
        """Sample a random token from a category."""
        tokens = self.cat2tokens[cat]
        return self.rng.choice(tokens)

    def _sample_next_category(self, current_cat: str) -> str:
        """Sample next category from POS transition matrix."""
        idx = self.cat_index.get(current_cat)
        if idx is None:
            idx = self.cat_index['PUNCT']
        probs = self.transition_matrix[idx]
        return self.rng.choices(self.cat_order, weights=probs)[0]

    def _sample_noun_bigram(self, noun: str) -> str:
        """Sample from a noun's probabilistic bigram distribution."""
        dist = self.noun_bigram_dists[noun]
        tokens, probs = zip(*dist)
        return self.rng.choices(tokens, weights=probs)[0]

    def _sample_bracket_content(
        self, bracket_type: str, last_cat: Optional[str], prev_token: Optional[str]
    ) -> tuple[str, str]:
        """Sample a content token inside a bracket span.

        Uses the bracket type's in-bracket transition matrix (different from
        the global POS transitions). Returns (token, category).
        """
        config = BRACKET_TYPES[bracket_type]

        if last_cat is None:
            # First token in bracket → use start category
            cat = config['start_cat']
        else:
            # Sample next category from in-bracket transitions
            transitions = config['transitions'][last_cat]
            cats, probs = zip(*transitions)
            cat = self.rng.choices(cats, weights=probs)[0]

        # Sample token from category, avoiding consecutive repeats
        token = self._sample_from_category(cat)
        if prev_token is not None and token == prev_token:
            for _ in range(5):
                token = self._sample_from_category(cat)
                if token != prev_token:
                    break

        return token, cat

    def sample_sequence(self, length: int) -> tuple[list[int], list[str]]:
        """Generate a single sequence.

        Args:
            length: Number of tokens to generate.

        Returns:
            token_ids: List of token IDs.
            labels: List of rule labels (same length as token_ids).
        """
        all_tokens: list[str] = []
        all_labels: list[str] = []

        # Track what followed each noun's first occurrence (for induction)
        noun_followers: dict[str, str] = {}

        # Bracket state
        bracket_type: Optional[str] = None  # None, 'paren', or 'quote'
        bracket_remaining: int = 0
        bracket_last_cat: Optional[str] = None
        post_bracket_cat: Optional[str] = None

        # Skip-bigram trigger boost: when an anchor appears, boost
        # the probability of its trigger appearing nearby
        active_boosts: list[dict] = []  # [{'trigger': str, 'remaining': int}]

        for _ in range(length):
            prev = all_tokens[-1] if all_tokens else None

            # --- State: just closed a bracket → force post-close category ---
            if post_bracket_cat is not None:
                token = self._sample_from_category(post_bracket_cat)
                label = 'cat_bigram'
                post_bracket_cat = None

            # --- State: inside a bracket ---
            elif bracket_type is not None:
                if bracket_remaining > 0:
                    token, bracket_last_cat = self._sample_bracket_content(
                        bracket_type, bracket_last_cat, prev
                    )
                    bracket_remaining -= 1
                    label = f'{bracket_type}_content'
                else:
                    # Emit close token — this is a skip_trigram
                    config = BRACKET_TYPES[bracket_type]
                    token = config['close']
                    label = 'skip_trigram'
                    post_bracket_cat = config.get('post_close_cat')
                    bracket_type = None
                    bracket_last_cat = None

            # --- State: normal generation ---
            else:
                opened = False

                # Maybe open a bracket
                if (self.use_brackets and prev is not None
                        and prev not in ('(', ')', '[', ']')):
                    r = self.rng.random()
                    if r < self.paren_prob:
                        token = '('
                        label = 'paren_open'
                        config = BRACKET_TYPES['paren']
                        bracket_type = 'paren'
                        bracket_remaining = self.rng.randint(
                            config['min_len'], config['max_len']
                        )
                        bracket_last_cat = None
                        opened = True
                    elif r < self.paren_prob + self.quote_prob:
                        token = '['
                        label = 'quote_open'
                        config = BRACKET_TYPES['quote']
                        bracket_type = 'quote'
                        bracket_remaining = self.rng.randint(
                            config['min_len'], config['max_len']
                        )
                        bracket_last_cat = None
                        opened = True

                if not opened:
                    token, label = self._sample_normal_token(
                        all_tokens, noun_followers
                    )

                    # Apply trigger boost: when cat_bigram fires and an
                    # anchor's trigger is boosted, replace with the trigger
                    if label == 'cat_bigram' and active_boosts:
                        for boost in active_boosts:
                            if self.rng.random() < self.trigger_boost_prob:
                                token = boost['trigger']
                                break

            # Track induction: only for nouns in normal context with
            # normal followers (not inside brackets or bracket boundaries)
            prev_label = all_labels[-1] if all_labels else None
            if (self.use_induction and all_tokens
                    and all_tokens[-1] in self.induction_triggers
                    and all_tokens[-1] not in noun_followers
                    and prev_label in self.NORMAL_LABELS
                    and label in self.NORMAL_LABELS):
                noun_followers[all_tokens[-1]] = token

            # Update trigger boosts: add new boosts for skip-bigram anchors
            if token in self.skip_anchor_triggers:
                for trigger in self.skip_anchor_triggers[token]:
                    active_boosts.append({
                        'trigger': trigger,
                        'remaining': self.trigger_boost_range,
                    })

            # Decrement and expire boosts
            for boost in active_boosts:
                boost['remaining'] -= 1
            active_boosts = [b for b in active_boosts if b['remaining'] > 0]

            all_tokens.append(token)
            all_labels.append(label)

        # Convert to IDs
        token_ids = [self.token2id[t] for t in all_tokens]
        return token_ids, all_labels

    def _sample_normal_token(
        self,
        context: list[str],
        noun_followers: dict[str, str],
    ) -> tuple[str, str]:
        """Sample the next token using priority-based dispatch (non-bracket rules).

        Priority order:
        1. Token trigram (specific 2-token → specific output)
        2. Category trigram (category pair → sample from output category)
        3. Skip-bigram (anchor within 8 positions + trigger → output)
        4. Induction (noun seen before → copy what followed it)
        5. Noun bigram (probabilistic: 70/20/10 over 3 verbs)
        6. Place bigram (deterministic: park → the)
        7. Category bigram (always available fallback)
        """
        prev = context[-1] if context else None
        prev2 = context[-2] if len(context) >= 2 else None

        # --- Priority 1: Token trigram ---
        if prev2 is not None and prev is not None:
            key = (prev2, prev)
            if key in self.tok_trigrams:
                return self.tok_trigrams[key], 'tok_trigram'

        # --- Priority 2: Category trigram ---
        if (self.cat_trigrams and prev2 is not None and prev is not None):
            prev2_cat = self.token2cat.get(prev2)
            prev_cat = self.token2cat.get(prev)
            if prev2_cat and prev_cat:
                cat_key = (prev2_cat, prev_cat)
                if cat_key in self.cat_trigrams:
                    target_cat = self.cat_trigrams[cat_key]
                    token = self._sample_from_category(target_cat)
                    return token, 'cat_trigram'

        # --- Priority 3: Skip-bigram ---
        # Check if any anchor token in recent context + current prev triggers a rule
        if self.skip_bigrams and prev is not None:
            max_skip = SKIP_BIGRAM_MAX_SKIP
            for dist in range(2, min(max_skip + 2, len(context) + 1)):
                anchor = context[-dist]
                key = (anchor, prev)
                if key in self.skip_bigrams:
                    return self.skip_bigrams[key], 'skip_bigram'

        # --- Priority 4: Induction ---
        # If this noun appeared before, copy what followed it
        if (self.use_induction and prev is not None
                and prev in self.induction_triggers
                and prev in noun_followers):
            return noun_followers[prev], 'induction'

        # --- Priority 5: Noun bigram (probabilistic) ---
        if prev is not None and prev in self.noun_bigram_dists:
            token = self._sample_noun_bigram(prev)
            return token, 'noun_bigram'

        # --- Priority 6: Place bigram (deterministic) ---
        if prev is not None and prev in self.place_bigrams:
            return self.place_bigrams[prev], 'place_bigram'

        # --- Priority 7: Category bigram (fallback) ---
        if prev is not None and self.use_cat_bigram:
            prev_cat = self.token2cat.get(prev, 'PUNCT')
            next_cat = self._sample_next_category(prev_cat)
            token = self._sample_from_category(next_cat)
            return token, 'cat_bigram'

        # --- Absolute fallback: random token from FUNC (start of sequence) ---
        token = self._sample_from_category('FUNC')
        return token, 'cat_bigram'

    def sample_batch(
        self, n_sequences: int, length: int
    ) -> tuple[list[list[int]], list[list[str]]]:
        """Generate a batch of sequences."""
        all_tokens = []
        all_labels = []
        for _ in range(n_sequences):
            tokens, labels = self.sample_sequence(length)
            all_tokens.append(tokens)
            all_labels.append(labels)
        return all_tokens, all_labels

    def format_sequence(
        self, token_ids: list[int], labels: list[str],
        show_labels: bool = True, compact: bool = False
    ) -> str:
        """Format a sequence for human-readable display."""
        tokens = [self.id2token[t] for t in token_ids]

        if compact:
            parts = []
            for tok, lab in zip(tokens, labels):
                sl = self._short_label(lab)
                parts.append(f'{tok}[{sl}]')
            return ' '.join(parts)

        token_strs = []
        label_strs = []
        for tok, lab in zip(tokens, labels):
            sl = self._short_label(lab)
            width = max(len(tok), len(sl))
            token_strs.append(tok.ljust(width))
            label_strs.append(sl.ljust(width))

        result = ' '.join(token_strs)
        if show_labels:
            result += '\n' + ' '.join(label_strs)
        return result

    @staticmethod
    def _short_label(label: str) -> str:
        """Abbreviate a label for display."""
        return {
            'cat_bigram': 'CB',
            'cat_trigram': 'CT',
            'noun_bigram': 'NB',
            'place_bigram': 'PB',
            'skip_bigram': 'SB',
            'skip_trigram': 'ST',
            'tok_trigram': 'TG',
            'induction': 'IN',
            'paren_open': 'P(',
            'paren_content': 'PC',
            'quote_open': 'Q[',
            'quote_content': 'QC',
            'pre_context': '--',
        }.get(label, label[:2].upper())

    def get_rule_stats(
        self, n_sequences: int = 1000, length: int = 64
    ) -> dict[str, float]:
        """Compute empirical rule firing statistics."""
        counts = defaultdict(int)
        total = 0
        for _ in range(n_sequences):
            _, labels = self.sample_sequence(length)
            for lab in labels:
                if lab != 'pre_context':
                    counts[lab] += 1
                    total += 1
        return {k: v / total for k, v in sorted(counts.items())}

    def describe_rules(self) -> str:
        """Return a human-readable description of all active rules."""
        lines = []
        lines.append(f"Language v3 | Vocab: {self.vocab_size} tokens | "
                      f"Categories: {len(self.categories)} × "
                      f"{self.tokens_per_category} tokens")
        lines.append(f"Active classes: {sorted(self.active_classes)}")
        lines.append("")

        lines.append("Categories:")
        for cat, tokens in self.categories.items():
            lines.append(f"  {cat:8s}: {', '.join(tokens)}")
        lines.append(f"  {'PUNCT':8s}: . ,")
        lines.append(f"  {'BRACKET':8s}: ( ) [ ]")
        lines.append("")

        if self.noun_bigram_dists:
            lines.append(f"Noun bigrams ({len(self.noun_bigram_dists)} nouns, probabilistic):")
            for noun in sorted(self.noun_bigram_dists):
                dist = self.noun_bigram_dists[noun]
                dist_str = ', '.join(f'{t}({p:.0%})' for t, p in dist)
                lines.append(f"  {noun} → {dist_str}")
            lines.append("")

        if self.place_bigrams:
            lines.append(f"Place bigrams ({len(self.place_bigrams)} rules, deterministic):")
            for trigger, output in sorted(self.place_bigrams.items()):
                lines.append(f"  {trigger} → {output}")
            lines.append("")

        if self.tok_trigrams:
            lines.append(f"Token trigrams ({len(self.tok_trigrams)} rules):")
            for (t1, t2), output in sorted(self.tok_trigrams.items()):
                lines.append(f"  {t1} {t2} → {output}")
            lines.append("")

        if self.cat_trigrams:
            lines.append(f"Category trigrams ({len(self.cat_trigrams)} rules):")
            for (c1, c2), out_cat in sorted(self.cat_trigrams.items()):
                lines.append(f"  {c1} + {c2} → sample from {out_cat}")
            lines.append("")

        if self.skip_bigrams:
            lines.append(f"Skip-bigrams ({len(self.skip_bigrams)} rules, max_skip={SKIP_BIGRAM_MAX_SKIP}):")
            for (anchor, trigger), output in sorted(self.skip_bigrams.items()):
                lines.append(f"  {anchor} ... {trigger} → {output}")
            lines.append("")

        if self.use_induction:
            lines.append(f"Induction: ALL {len(self.induction_triggers)} nouns")
            lines.append("  First occurrence → noun bigram (probabilistic)")
            lines.append("  Subsequent occurrences → copy what followed first")
            lines.append("")

        if self.use_brackets:
            for btype, config in BRACKET_TYPES.items():
                lines.append(f"{btype.capitalize()} {config['open']} ... {config['close']}:")
                lines.append(f"  Content length: {config['min_len']}-{config['max_len']} tokens")
                lines.append(f"  Start category: {config['start_cat']}")
                lines.append(f"  After close: {config['post_close_cat']}")
                lines.append(f"  In-bracket transitions:")
                for src_cat, transitions in config['transitions'].items():
                    trans_str = ', '.join(f'{c}({p:.0%})' for c, p in transitions)
                    lines.append(f"    {src_cat} → {trans_str}")
                prob = self.paren_prob if btype == 'paren' else self.quote_prob
                lines.append(f"  Probability: {prob}")
                lines.append(f"  Close label: skip_trigram (model must attend to opener)")
                lines.append("")

        lines.append("POS transition matrix:")
        header = "          " + " ".join(f"{c:7s}" for c in CAT_ORDER)
        lines.append(header)
        for i, cat in enumerate(CAT_ORDER):
            row = " ".join(f"{self.transition_matrix[i, j]:7.3f}"
                           for j in range(len(CAT_ORDER)))
            lines.append(f"  {cat:8s} {row}")

        return '\n'.join(lines)


# ---------------------------------------------------------------------------
# Training utilities
# ---------------------------------------------------------------------------

def make_batch(
    generator: LanguageV3, batch_size: int, seq_len: int, device=None
) -> tuple:
    """Generate a batch for training.

    Returns:
        x: Input tensor (batch_size, seq_len-1)
        y: Target tensor (batch_size, seq_len-1)
        labels: List of label lists
        mask: Boolean tensor (all False — no pre-context to mask)
    """
    import torch

    all_tokens, all_labels = generator.sample_batch(batch_size, seq_len)

    x_list = []
    y_list = []
    label_list = []

    for tokens, labels in zip(all_tokens, all_labels):
        x_list.append(tokens[:-1])
        y_list.append(tokens[1:])
        label_list.append(labels[1:])  # align with targets

    x_tensor = torch.tensor(x_list, dtype=torch.long)
    y_tensor = torch.tensor(y_list, dtype=torch.long)
    # No pre-context → no masking needed, but keep interface compatible
    mask_tensor = torch.zeros(batch_size, seq_len - 1, dtype=torch.bool)

    if device is not None:
        x_tensor = x_tensor.to(device)
        y_tensor = y_tensor.to(device)
        mask_tensor = mask_tensor.to(device)

    return x_tensor, y_tensor, label_list, mask_tensor


def per_rule_loss(logits, targets, labels, mask=None):
    """Compute per-rule cross-entropy loss."""
    import torch
    import torch.nn.functional as F

    B, T, V = logits.shape
    ce = F.cross_entropy(
        logits.reshape(-1, V), targets.reshape(-1),
        reduction='none'
    ).reshape(B, T)

    rule_losses = defaultdict(list)
    for b in range(B):
        for t in range(T):
            if mask is not None and mask[b, t]:
                continue
            lab = labels[b][t] if t < len(labels[b]) else 'pad'
            if lab in ('pad', 'pre_context'):
                continue
            rule_losses[lab].append(ce[b, t].item())

    return {k: sum(v) / len(v) for k, v in rule_losses.items() if v}


def per_rule_kl(logits, labels, generator, mask=None):
    """Compute per-rule KL divergence: KL(true_distribution || model_distribution).

    Unlike CE, KL is 0 when the model perfectly matches the data distribution.
    This is the right metric for rules with non-deterministic outputs (e.g.,
    noun_bigram has 70/20/10 distribution, cat_bigram has category uncertainty).

    For deterministic rules (tok_trigram, place_bigram, skip_bigram, skip_trigram),
    KL = CE (since H(true) = 0).

    For probabilistic rules:
      KL = CE - H(true distribution)
    where H(true) is the entropy of the data-generating distribution.

    Args:
        logits: Model output (B, T, V)
        labels: Per-position rule labels
        generator: LanguageV3 instance (needed to look up true distributions)
        mask: Optional boolean mask

    Returns:
        Dict of rule_name → mean KL divergence
    """
    import torch
    import torch.nn.functional as F

    B, T, V = logits.shape
    log_probs = F.log_softmax(logits, dim=-1)  # (B, T, V)

    rule_kls = defaultdict(list)

    # Precompute true distribution entropies for each rule type
    N = generator.tokens_per_category

    # Noun bigram: H([0.7, 0.2, 0.1]) ≈ 0.80 nats
    noun_bigram_entropy = -sum(p * np.log(p) for p in NOUN_BIGRAM_PROBS if p > 0)

    # Category sampling: H = log(N) (uniform within category)
    log_N = np.log(N)

    # Category bigram: H(POS transition) + log(N)
    # Average H(POS transition) across source categories
    pos_entropies = {}
    for i, cat in enumerate(CAT_ORDER):
        row = TRANSITION_MATRIX[i]
        pos_entropies[cat] = -sum(p * np.log(p) for p in row if p > 0)

    # In-bracket transition entropies
    bracket_entropies = {}
    for btype, config in BRACKET_TYPES.items():
        bracket_entropies[btype] = {}
        for src_cat, transitions in config['transitions'].items():
            cats, probs = zip(*transitions)
            # H(category choice) + log(N) for token within category
            h_cat = -sum(p * np.log(p) for p in probs if p > 0)
            bracket_entropies[btype][src_cat] = h_cat + log_N

    # Per-position KL computation
    for b in range(B):
        for t in range(T):
            if mask is not None and mask[b, t]:
                continue
            lab = labels[b][t] if t < len(labels[b]) else 'pad'
            if lab in ('pad', 'pre_context'):
                continue

            # CE for this position = -log p(true_token)
            # We need the target token to compute CE
            # But we can compute KL = CE - H(true) directly

            # For now, compute full CE from logits
            # This is inefficient but correct
            # KL = CE - H(true distribution for this rule)

            if lab in ('tok_trigram', 'place_bigram', 'skip_bigram',
                       'skip_trigram', 'induction'):
                # Deterministic rules: H(true) = 0, so KL = CE
                h_true = 0.0
            elif lab == 'noun_bigram':
                h_true = noun_bigram_entropy
            elif lab == 'cat_trigram':
                # Output is uniform within a category
                h_true = log_N
            elif lab in ('paren_content', 'quote_content'):
                # H depends on bracket type and position within bracket
                # Approximate with average in-bracket entropy
                btype = lab.split('_')[0]
                avg_h = np.mean(list(bracket_entropies[btype].values()))
                h_true = avg_h
            elif lab == 'cat_bigram':
                # H = H(POS transition | prev_cat) + log(N)
                # We don't track prev_cat here; use average
                avg_pos_h = np.mean(list(pos_entropies.values()))
                h_true = avg_pos_h + log_N
            elif lab in ('paren_open', 'quote_open'):
                h_true = 0.0  # deterministic token
            else:
                h_true = 0.0

            rule_kls[lab].append(h_true)

    # Now compute actual KL = mean_CE - H(true) for each rule
    # We need CE values too — recompute
    ce = F.cross_entropy(
        logits.reshape(-1, V),
        torch.zeros(B * T, dtype=torch.long, device=logits.device),  # placeholder
        reduction='none'
    ).reshape(B, T)

    # Actually, we need the targets. Let's restructure to take targets.
    # For now, return the H(true) values so the caller can compute KL = CE - H
    return {k: sum(v) / len(v) for k, v in rule_kls.items() if v}


def true_entropies(generator):
    """Return the theoretical entropy H(true distribution) for each rule type.

    KL divergence = CE_loss - H(true). When the model perfectly learns a rule,
    CE = H(true) and KL = 0.

    Returns:
        Dict of rule_name → H(true distribution) in nats
    """
    N = generator.tokens_per_category
    log_N = np.log(N)

    # Noun bigram: H([0.7, 0.2, 0.1])
    h_noun_bigram = -sum(p * np.log(p) for p in NOUN_BIGRAM_PROBS if p > 0)

    # Category bigram: average H(POS transition) + log(N)
    pos_h = []
    for i in range(len(CAT_ORDER)):
        row = TRANSITION_MATRIX[i]
        pos_h.append(-sum(p * np.log(p) for p in row if p > 0))
    avg_pos_h = np.mean(pos_h)

    # Bracket content entropies
    bracket_h = {}
    for btype, config in BRACKET_TYPES.items():
        btype_h = []
        for src_cat, transitions in config['transitions'].items():
            cats, probs = zip(*transitions)
            h_cat = -sum(p * np.log(p) for p in probs if p > 0)
            btype_h.append(h_cat + log_N)
        bracket_h[f'{btype}_content'] = np.mean(btype_h)

    result = {
        'tok_trigram': 0.0,
        'place_bigram': 0.0,
        'skip_bigram': 0.0,
        'skip_trigram': 0.0,
        'induction': 0.0,
        'paren_open': 0.0,
        'quote_open': 0.0,
        'noun_bigram': h_noun_bigram,
        'cat_trigram': log_N,
        'cat_bigram': avg_pos_h + log_N,
    }
    result.update(bracket_h)

    return result
