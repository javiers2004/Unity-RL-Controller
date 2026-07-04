from urc.core.registry import algorithms

# sb3_ppo no se importa aquí a propósito: depende del extra opcional "sb3"
# (pesado: torch, gymnasium...). Se registra como disponible bajo demanda:
# `algorithms.get("sb3-ppo")` lo importa la primera vez que se pide de verdad.
algorithms.register_lazy(
    "sb3-ppo", "urc.algorithms.sb3_ppo", install_hint='pip install "urc[sb3]"'
)
