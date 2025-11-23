from src.core.crossproduct import SignLanguageCP
from src.core.ground import SignLanguageGroundEnvironment
from src.core.label import SignLanguageLF
from src.core.machine import SignLanguageRM


def cross_product_factory(render_mode: str = "human") -> SignLanguageCP:
    """Create a cross product for the sign language environment."""
    env = SignLanguageGroundEnvironment(render_mode=render_mode)
    labelling_function = SignLanguageLF()
    machine = SignLanguageRM()
    return SignLanguageCP(
        env,
        machine,
        labelling_function,
        max_steps=500,
    )
