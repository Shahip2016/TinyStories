"""
Curated vocabulary for TinyStories dataset generation.

The paper restricts story content to ~1500 words that a typical 3-4 year old
child would understand. Words are organized by part of speech (nouns, verbs,
adjectives) for the random sampling strategy used in story generation prompts.

Each story prompt randomly selects one noun, one verb, and one adjective that
must appear in the generated story, along with optional narrative features
(dialogue, moral value, plot twist, etc.) to ensure dataset diversity.

Reference: Section 2 of Eldan & Li (2023), arXiv:2305.07759
"""

import random
from typing import List, Tuple

# ============================================================================
# Nouns (~500 common nouns a 3-4 year old would know)
# ============================================================================
NOUNS = [
    # People & Family
    "mom", "dad", "baby", "boy", "girl", "friend", "brother", "sister",
    "grandma", "grandpa", "teacher", "doctor", "king", "queen", "prince",
    "princess", "child", "man", "woman", "people", "family", "kid",
    # Animals
    "dog", "cat", "bird", "fish", "bunny", "bear", "duck", "frog", "horse",
    "cow", "pig", "chicken", "mouse", "elephant", "lion", "tiger", "monkey",
    "butterfly", "bee", "ant", "turtle", "snake", "sheep", "goat", "deer",
    "owl", "penguin", "whale", "dolphin", "fox", "wolf", "rabbit", "puppy",
    "kitten", "bug", "spider", "worm", "dinosaur", "dragon",
    # Body
    "hand", "foot", "head", "eye", "ear", "nose", "mouth", "arm", "leg",
    "finger", "toe", "hair", "face", "teeth", "tummy", "heart", "knee",
    # Food & Drink
    "apple", "banana", "cookie", "cake", "candy", "ice cream", "milk",
    "juice", "water", "bread", "cheese", "pizza", "soup", "egg",
    "carrot", "orange", "strawberry", "grape", "pie", "sandwich",
    # Home & Objects
    "house", "door", "window", "bed", "chair", "table", "lamp", "book",
    "toy", "ball", "doll", "block", "puzzle", "car", "truck", "train",
    "bike", "box", "bag", "key", "cup", "plate", "spoon", "fork",
    "blanket", "pillow", "mirror", "clock", "phone", "picture", "map",
    "bell", "candle", "basket", "ladder", "rope", "wheel", "button",
    "gift", "present", "letter", "card", "coin", "crown", "sword",
    # Clothing
    "hat", "shoe", "dress", "shirt", "coat", "sock", "boot", "glove",
    "scarf", "pants",
    # Nature
    "sun", "moon", "star", "cloud", "rain", "snow", "wind", "tree",
    "flower", "grass", "leaf", "rock", "river", "lake", "ocean", "sea",
    "mountain", "hill", "forest", "garden", "park", "field", "island",
    "sand", "mud", "dirt", "seed", "rainbow", "storm", "ice",
    # Places
    "school", "store", "farm", "zoo", "library", "castle", "village",
    "town", "city", "church", "bridge", "road", "path", "street",
    "room", "kitchen", "bathroom", "bedroom", "playground", "cave",
    "barn", "tower",
    # Time & Concepts
    "day", "night", "morning", "time", "year", "name", "story", "song",
    "game", "dream", "wish", "idea", "secret", "surprise", "adventure",
    "trip", "journey", "lesson", "rule", "truth", "lie", "magic",
    "color", "shape", "number", "word", "voice", "sound", "noise",
    "music", "dance", "party", "birthday", "holiday", "christmas",
    "summer", "winter", "spring", "fall",
    # Misc Objects
    "fire", "light", "shadow", "hole", "nest", "web", "fence", "gate",
    "sign", "flag", "kite", "balloon", "bubble", "crayon", "paint",
    "paper", "pencil", "brush", "string", "stick", "stone", "shell",
    "feather", "paw", "tail", "wing", "fur",
]

# ============================================================================
# Verbs (~300 common verbs)
# ============================================================================
VERBS = [
    # Movement
    "run", "walk", "jump", "fly", "swim", "climb", "crawl", "skip",
    "hop", "dance", "slide", "roll", "spin", "fall", "trip", "march",
    "chase", "race", "rush", "hurry", "sneak", "tiptoe", "wander",
    "travel", "move", "step", "bounce", "swing", "hang", "reach",
    # Actions with objects
    "play", "eat", "drink", "cook", "bake", "cut", "draw", "paint",
    "build", "fix", "break", "open", "close", "push", "pull", "throw",
    "catch", "kick", "hit", "drop", "pick", "carry", "hold", "lift",
    "put", "set", "give", "take", "bring", "send", "hide", "find",
    "dig", "plant", "water", "clean", "wash", "dry", "pour", "mix",
    "stir", "shake", "wrap", "tie", "fill", "pack", "fold", "tear",
    "squeeze", "stretch", "twist", "turn", "count", "sort", "match",
    # Communication
    "say", "tell", "ask", "talk", "speak", "call", "shout", "yell",
    "whisper", "sing", "read", "write", "spell", "name", "answer",
    "explain", "promise", "warn", "thank", "invite", "greet",
    # Emotions & Mental
    "love", "like", "want", "need", "wish", "hope", "believe", "think",
    "know", "learn", "remember", "forget", "understand", "wonder",
    "worry", "care", "feel", "cry", "laugh", "smile", "frown", "sigh",
    "fear", "trust", "miss", "enjoy", "hate", "mind", "dream",
    # Senses
    "see", "look", "watch", "hear", "listen", "smell", "taste", "touch",
    "feel", "notice", "spot",
    # Social
    "help", "share", "hug", "kiss", "wave", "nod", "bow", "clap",
    "cheer", "fight", "argue", "forgive", "agree", "follow", "lead",
    "join", "meet", "visit", "welcome", "save", "protect", "teach",
    # State / Being
    "be", "have", "do", "go", "come", "stay", "sit", "stand", "lie",
    "sleep", "wake", "wait", "stop", "start", "begin", "end", "finish",
    "grow", "change", "become", "seem", "appear", "happen", "belong",
    "live", "die", "try", "work", "rest", "fit", "last",
    # Misc
    "make", "let", "keep", "use", "show", "point", "choose", "decide",
    "win", "lose", "succeed", "fail", "pretend", "imagine", "create",
    "discover", "explore", "celebrate", "practice", "return", "leave",
    "arrive", "cross", "enter", "exit", "land", "splash", "crash",
    "bark", "meow", "roar", "chirp", "buzz",
]

# ============================================================================
# Adjectives (~300 common adjectives)
# ============================================================================
ADJECTIVES = [
    # Size
    "big", "small", "little", "tiny", "huge", "tall", "short", "long",
    "wide", "thin", "fat", "deep", "thick",
    # Color
    "red", "blue", "green", "yellow", "orange", "purple", "pink",
    "black", "white", "brown", "gray", "golden",
    # Temperature & Weather
    "hot", "cold", "warm", "cool", "wet", "dry", "sunny", "rainy",
    "cloudy", "snowy", "windy", "icy", "foggy",
    # Texture & Physical
    "soft", "hard", "smooth", "rough", "sharp", "flat", "round",
    "sticky", "fluffy", "furry", "shiny", "sparkly", "bright", "dark",
    "light", "heavy", "loud", "quiet",
    # Taste & Smell
    "sweet", "sour", "yummy", "delicious", "fresh", "rotten", "stinky",
    # Emotions
    "happy", "sad", "angry", "scared", "brave", "proud", "shy",
    "excited", "nervous", "worried", "lonely", "tired", "sleepy",
    "silly", "funny", "serious", "calm", "gentle", "kind", "mean",
    "nice", "friendly", "grumpy", "cheerful", "grateful", "sorry",
    "jealous", "curious", "surprised", "confused", "stubborn",
    "patient", "lazy", "clever", "wise", "foolish",
    # Quality
    "good", "bad", "great", "wonderful", "beautiful", "pretty", "ugly",
    "clean", "dirty", "messy", "neat", "perfect", "broken", "new",
    "old", "young", "fast", "slow", "quick", "early", "late",
    "easy", "difficult", "simple", "special", "different", "same",
    "important", "strange", "weird", "crazy", "amazing", "awesome",
    "terrible", "horrible", "fantastic",
    # State
    "hungry", "thirsty", "full", "empty", "open", "closed", "lost",
    "found", "free", "busy", "ready", "safe", "dangerous", "strong",
    "weak", "sick", "healthy", "alive", "dead", "real", "fake",
    "true", "right", "wrong", "fair", "unfair", "lucky", "poor",
    "rich",
    # Other
    "favorite", "own", "other", "next", "last", "first", "best",
    "worst", "only", "every", "many", "few", "more", "most",
    "enough", "whole", "certain", "possible", "magic", "powerful",
    "cozy", "comfy", "fragile", "polite", "rude", "helpful",
    "useless", "careful", "careless", "colorful", "humble",
    "mysterious", "ancient", "modern", "wild", "tame",
]

# ============================================================================
# Narrative Features (from Section 2 of the paper)
# ============================================================================
# These are randomly selected to add structural diversity to generated stories.

STORY_FEATURES = [
    "Dialogue (the story should contain at least one dialogue)",
    "BadEnding (the story has a bad ending)",
    "Conflict (the story should contain a conflict or problem that needs resolution)",
    "MoralValue (the story should teach a moral lesson or convey a positive value)",
    "Foreshadowing (the story should use foreshadowing or setup and payoff)",
    "Twist (the story should contain an unexpected plot twist)",
]


def sample_story_seed(rng: random.Random = None) -> Tuple[str, str, str, List[str]]:
    """Sample a random (noun, verb, adjective, features) tuple for story generation.

    This implements the diversity strategy from the paper: each story prompt
    randomly selects one word from each category and 0-3 narrative features.

    Args:
        rng: Optional Random instance for reproducibility.

    Returns:
        Tuple of (noun, verb, adjective, list_of_features)
    """
    if rng is None:
        rng = random.Random()

    noun = rng.choice(NOUNS)
    verb = rng.choice(VERBS)
    adjective = rng.choice(ADJECTIVES)

    # Randomly select 0 to 3 features
    n_features = rng.randint(0, 3)
    features = rng.sample(STORY_FEATURES, k=min(n_features, len(STORY_FEATURES)))

    return noun, verb, adjective, features


def build_generation_prompt(
    noun: str,
    verb: str,
    adjective: str,
    features: List[str],
) -> str:
    """Build a story generation prompt following the paper's template.

    The prompt instructs a large LM (GPT-3.5/GPT-4) to write a short story
    constrained to vocabulary a 3-4 year old would understand, incorporating
    the specified word and narrative feature requirements.

    Args:
        noun: Required noun to include in the story.
        verb: Required verb to include in the story.
        adjective: Required adjective to include in the story.
        features: List of narrative feature descriptions.

    Returns:
        Formatted prompt string.
    """
    prompt = (
        "Write a short story (3-5 paragraphs) which only uses very simple words "
        "that a 3 year old child would likely understand. "
        f"The story should use the verb '{verb}', "
        f"the noun '{noun}' and the adjective '{adjective}'. "
    )

    if features:
        feature_text = ". ".join(features)
        prompt += f"The story has the following features: {feature_text}. "

    prompt += "Remember to only use simple words!"

    return prompt


def build_instruct_prompt(
    noun: str,
    verb: str,
    adjective: str,
    features: List[str],
    summary: str = None,
    sentence: str = None,
) -> str:
    """Build a TinyStories-Instruct generation prompt.

    The Instruct variant adds constraints like a required summary or sentence
    that must be incorporated, training models to follow specific instructions.

    Args:
        noun: Required noun to include in the story.
        verb: Required verb to include in the story.
        adjective: Required adjective to include in the story.
        features: List of narrative feature descriptions.
        summary: Optional one-sentence summary the story must follow.
        sentence: Optional specific sentence that must appear in the story.

    Returns:
        Formatted instruct prompt string.
    """
    prompt = build_generation_prompt(noun, verb, adjective, features)

    if summary:
        prompt += f"\n\nThe story should roughly follow this summary: {summary}"

    if sentence:
        prompt += f'\n\nThe story must contain this sentence: "{sentence}"'

    return prompt
